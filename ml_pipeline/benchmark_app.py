import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
import inspect
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from pipelines_torch.benchmark import BenchmarkRunner
from pipelines_torch.models import CLASSIFICATION_MODEL_REGISTRY, REGRESSION_MODEL_REGISTRY
from data_augmentation import augmentations as aug
from utils.data import load_csv, prepare_embeddings_data
from utils.utils import RESULTS_DIR_OUT


def _parse_json(text: str, default):
    text = (text or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        st.error(f"Invalid JSON: {exc}")
        return default


def _filter_kwargs(callable_obj, params):
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return params
    has_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    if has_kwargs:
        return params
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in params.items() if k in allowed}


def _safe_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _best_index(df: pd.DataFrame, metric: str) -> int:
    if metric not in df.columns:
        return max(len(df) - 1, 0)
    values = _safe_float_series(df[metric])
    if values.isna().all():
        return max(len(df) - 1, 0)
    if "loss" in metric:
        return int(values.idxmin())
    return int(values.idxmax())


def _list_result_dirs(base_dir: str) -> list:
    if not os.path.isdir(base_dir):
        return []
    return sorted([p for p in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, p))])


def _load_index(path_start: str) -> Dict[str, dict]:
    base_dir = os.path.join(RESULTS_DIR_OUT, path_start)
    index_path = os.path.join(base_dir, "index.jsonl")
    if not os.path.exists(index_path):
        return {}
    entries = {}
    with open(index_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            entries[entry["checkpoint_id"]] = entry
    return entries


def _collect_metrics(path_start: str) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    base_dir = os.path.join(RESULTS_DIR_OUT, path_start)
    if not os.path.isdir(base_dir):
        return pd.DataFrame(), {}
    files = sorted([f for f in os.listdir(base_dir) if f.endswith("_metrics.csv")])
    index = _load_index(path_start)
    histories = {}
    rows = []
    for fname in files:
        df = pd.read_csv(os.path.join(base_dir, fname))
        stem = Path(fname).stem
        if not stem.endswith("_metrics"):
            continue
        checkpoint_id = stem[:-8]
        meta = index.get(checkpoint_id, {})
        model_name = meta.get("model_name", "unknown")
        aug_name = meta.get("augmentation_name", "unknown")
        label = f"{model_name} | {aug_name} | {checkpoint_id}"
        histories[label] = df
        rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "model": model_name,
                "augmentation": aug_name,
                "history": df,
            }
        )
    if not rows:
        return pd.DataFrame(), histories

    all_cols = set().union(*(r["history"].columns for r in rows))
    metric_cols = [c for c in all_cols if c not in {"fold", "epoch"}]
    summary = []
    for r in rows:
        df = r["history"]
        metric = next((m for m in ("roc_auc", "r2_score", "f1_score", "pr_auc", "val_loss", "loss") if m in df.columns), None)
        idx = _best_index(df, metric) if metric else max(len(df) - 1, 0)
        best = df.iloc[idx].to_dict()
        best_row = {
            "checkpoint_id": r["checkpoint_id"],
            "model": r["model"],
            "augmentation": r["augmentation"],
            "best_epoch": idx,
        }
        for col in metric_cols:
            if col in best:
                best_row[col] = best[col]
        summary.append(best_row)
    return pd.DataFrame(summary), histories


def _prepare_data(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list,
    task_type: str,
    encode_categoricals: bool,
    embeddings_mode: bool,
):
    label_map = None
    if embeddings_mode:
        X, y = prepare_embeddings_data(df, target_col, embedding_column=feature_cols[0])
        return X, y, label_map

    X_df = df[feature_cols].copy()
    if encode_categoricals:
        X_df = pd.get_dummies(X_df, drop_first=False)

    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    numeric_cols = X_df.select_dtypes(include=["number"]).columns
    if len(numeric_cols) > 0:
        X_df[numeric_cols] = X_df[numeric_cols].fillna(X_df[numeric_cols].median())
    X_df = X_df.fillna(0)

    X = X_df.to_numpy()
    y = df[target_col].to_numpy()

    if task_type == "classification":
        if y.dtype == object or str(y.dtype).startswith("category"):
            from sklearn.preprocessing import LabelEncoder
            enc = LabelEncoder()
            y = enc.fit_transform(y)
            label_map = {int(i): str(c) for i, c in enumerate(enc.classes_)}
    return X, y, label_map


def _augmentation_registry():
    def mixup_wrapper(X, y, **kwargs):
        return aug.mixup_augmentation(
            X, y, alpha=kwargs.get("alpha", 0.2), random_state=kwargs.get("random_state", 42)
        )

    def mixup_smote_wrapper(X, y, **kwargs):
        return aug.mixup_smote_augmentation(
            X,
            y,
            n_samples=len(X),
            alpha=kwargs.get("alpha", 0.2),
            random_state=kwargs.get("random_state", 42),
            max_factor=kwargs.get("max_factor", 2.0),
        )

    return {
        "none": aug.none_augmentation,
        "smote": aug.smote_augmentation,
        "borderline_smote": aug.borderline_smote_augmentation,
        "svm_smote": aug.svm_smote_augmentation,
        "kmeans_smote": aug.kmeans_smote_augmentation,
        "adasyn": aug.adasyn_augmentation,
        "smoteenn": aug.smoteenn_augmentation,
        "smotetomek": aug.smotetomek_augmentation,
        "mixup": mixup_wrapper,
        "mixup_smote": mixup_smote_wrapper,
    }


st.set_page_config(page_title="ML Benchmark UI", layout="wide")
st.title("ML Benchmark UI")

with st.sidebar:
    st.header("Dataset")
    source = st.radio("Source", ["Upload CSV", "Path"], horizontal=True)
    df = None
    if source == "Upload CSV":
        upload = st.file_uploader("CSV file", type=["csv"])
        if upload:
            df = pd.read_csv(upload)
    else:
        default_path = "ml_pipeline/recipes_df.csv"
        csv_path = st.text_input("CSV path", value=default_path)
        if csv_path and os.path.exists(csv_path):
            df = load_csv(csv_path)
        elif csv_path:
            st.warning("CSV path not found.")

    st.header("Task")
    task_type = st.selectbox("Task type", ["classification", "regression"])

    if df is not None:
        columns = list(df.columns)
        target_col = st.selectbox("Target column", columns)
        feature_cols = st.multiselect(
            "Feature columns",
            [c for c in columns if c != target_col],
            default=[c for c in columns if c != target_col],
        )
        embeddings_mode = False
        if len(feature_cols) == 1:
            embeddings_mode = st.checkbox("Single column contains embeddings", value=False)
        encode_categoricals = st.checkbox("One-hot encode categorical features", value=True, disabled=embeddings_mode)
    else:
        target_col = None
        feature_cols = []
        embeddings_mode = False
        encode_categoricals = True

    st.header("Models")
    registry = CLASSIFICATION_MODEL_REGISTRY if task_type == "classification" else REGRESSION_MODEL_REGISTRY
    model_names = sorted(registry.keys())
    selected_models = st.multiselect("Select models", model_names, default=model_names[:3])
    global_params_text = st.text_area("Global model params (JSON)", value="")
    per_model_params_text = st.text_area("Per-model params (JSON)", value="")

    st.header("Augmentations")
    aug_registry = _augmentation_registry()
    aug_names = list(aug_registry.keys())
    selected_augs = st.multiselect("Select augmentations", aug_names, default=["none"])
    max_factor = st.number_input("Max factor", min_value=1.0, max_value=10.0, value=2.0, step=0.5)

    st.header("Training")
    epochs = st.number_input("Epochs", min_value=1, max_value=500, value=10, step=1)
    batch_size = st.number_input("Batch size", min_value=1, max_value=2048, value=32, step=1)
    learning_rate = st.number_input("Learning rate", min_value=1e-6, max_value=1.0, value=1e-4, format="%.6f")
    weight_decay = st.number_input("Weight decay", min_value=0.0, max_value=1.0, value=0.0, format="%.6f")
    dropout = st.number_input("Dropout", min_value=0.0, max_value=0.9, value=0.0, format="%.2f")
    early_stopping = st.number_input("Early stopping (epochs)", min_value=0, max_value=100, value=0, step=1)
    use_kfold = st.checkbox("Use k-fold CV", value=True)
    k_folds = st.number_input("K folds", min_value=2, max_value=10, value=5, step=1)
    device = st.selectbox("Device", ["cpu", "cuda"])

    st.header("Saving")
    default_run = f"streamlit_runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path_start = st.text_input("Run folder (under results/)", value=default_run)
    save_to_hf = st.checkbox("Save to HuggingFace Hub", value=False)
    hf_repo_name = st.text_input("HF repo name", value="") if save_to_hf else None
    hf_token = st.text_input("HF token", value="", type="password") if save_to_hf else None

    run_clicked = st.button("Run benchmark", use_container_width=True)


if df is None:
    st.info("Load a CSV to configure the benchmark.")
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(), use_container_width=True)

if run_clicked:
    if not target_col or not feature_cols:
        st.error("Select target and feature columns before running.")
        st.stop()
    if not selected_models:
        st.error("Select at least one model.")
        st.stop()
    if not path_start:
        st.error("Provide a run folder name.")
        st.stop()

    X, y, label_map = _prepare_data(
        df,
        target_col=target_col,
        feature_cols=feature_cols,
        task_type=task_type,
        encode_categoricals=encode_categoricals,
        embeddings_mode=embeddings_mode,
    )

    if label_map:
        st.caption(f"Label encoding: {label_map}")

    input_dim = X.shape[1]
    num_classes = int(len(np.unique(y))) if task_type == "classification" else 1
    output_dim = 1

    global_params = _parse_json(global_params_text, {})
    per_model_params = _parse_json(per_model_params_text, {})

    model_configs = []
    for name in selected_models:
        model_cls = registry[name]
        params = {"input_dim": input_dim, "num_classes": num_classes, "output_dim": output_dim}
        params.update(global_params)
        if isinstance(per_model_params, dict):
            params.update(per_model_params.get(name, {}))
        params = _filter_kwargs(model_cls, params)
        model_configs.append({"name": name, "class": model_cls, "params": params})

    augmentations = [(a, aug_registry[a]) for a in selected_augs] if selected_augs else [("none", aug.none_augmentation)]

    runner = BenchmarkRunner(
        model_configs=model_configs,
        augmentations=augmentations,
        metrics=None,
        task_type=task_type,
        device=device,
        epochs=int(epochs),
        batch_size=int(batch_size),
        early_stopping=int(early_stopping) if early_stopping > 0 else None,
        save_to_hf=bool(save_to_hf),
        hf_repo_name=hf_repo_name or None,
        hf_token=hf_token or None,
        dropout=dropout if dropout > 0 else None,
        weight_decay=weight_decay if weight_decay > 0 else None,
        learning_rate=learning_rate,
        use_kfold=bool(use_kfold),
        k_folds=int(k_folds),
        path_start=path_start,
        max_factor=float(max_factor),
    )

    with st.spinner("Running benchmark..."):
        runner.run(X, y)
    st.session_state["last_run"] = path_start
    st.success(f"Benchmark finished. Results saved under results/{path_start}")


st.subheader("Results")
results_dir = RESULTS_DIR_OUT
existing_runs = _list_result_dirs(results_dir)
last_run = st.session_state.get("last_run")
default_run = last_run if last_run in existing_runs else (existing_runs[-1] if existing_runs else "")
selected_run = st.selectbox("Select a run folder", options=[""] + existing_runs, index=(existing_runs.index(default_run) + 1) if default_run else 0)

if selected_run:
    summary_df, histories = _collect_metrics(selected_run)
    if summary_df.empty:
        st.info("No metrics found for this run.")
    else:
        metric_options = [c for c in summary_df.columns if c not in {"model", "augmentation", "best_epoch"}]
        primary_metric = st.selectbox("Primary metric for ranking", metric_options, index=0) if metric_options else None
        if primary_metric:
            ranked = summary_df.sort_values(by=primary_metric, ascending=("loss" in primary_metric))
        else:
            ranked = summary_df

        tab_summary, tab_details, tab_best = st.tabs(["Summary", "Details", "Best model"])

        with tab_summary:
            st.dataframe(ranked, use_container_width=True)
            if primary_metric:
                chart_df = ranked.copy()
                chart_df["label"] = chart_df["model"] + " | " + chart_df["augmentation"]
                chart_df = chart_df.set_index("label")[[primary_metric]]
                st.bar_chart(chart_df, use_container_width=True)

        with tab_details:
            if histories:
                selection = st.selectbox("Select model/augmentation", list(histories.keys()))
                st.dataframe(histories[selection], use_container_width=True)

        with tab_best:
            if primary_metric and not ranked.empty:
                best = ranked.iloc[0]
                model_name = best["model"]
                aug_name = best["augmentation"]
                checkpoint_id = best["checkpoint_id"]
                base_dir = os.path.join(RESULTS_DIR_OUT, selected_run)
                index = _load_index(selected_run)
                entry = index.get(checkpoint_id, {})
                artifact_path = entry.get("artifact_path")
                model_path = os.path.join(base_dir, artifact_path) if artifact_path else None
                st.markdown(f"**Best run**: `{model_name}` + `{aug_name}`")
                st.caption(f"Checkpoint: `{checkpoint_id}`")
                if model_path and os.path.isdir(model_path):
                    st.markdown(f"Saved model dir: `{model_path}`")
                elif model_path and os.path.isfile(model_path):
                    st.markdown(f"Saved model file: `{model_path}`")
                else:
                    st.warning("Saved model not found on disk.")

                if st.button("Copy best model to results/best_model", use_container_width=True):
                    best_path = os.path.join(base_dir, "best_model")
                    if model_path and os.path.isdir(model_path):
                        if os.path.exists(best_path):
                            shutil.rmtree(best_path)
                        shutil.copytree(model_path, best_path)
                        st.success(f"Copied to {best_path}")
                    elif model_path and os.path.isfile(model_path):
                        suffix = Path(model_path).suffix
                        target = best_path + suffix
                        shutil.copyfile(model_path, target)
                        st.success(f"Copied to {target}")
                    else:
                        st.error("Nothing to copy.")

                if save_to_hf and hf_repo_name:
                    if st.button("Upload best model to HF Hub", use_container_width=True):
                        try:
                            from huggingface_hub import HfApi
                            api = HfApi(token=hf_token or None)
                            if model_path and os.path.isdir(model_path):
                                api.upload_folder(folder_path=model_path, repo_id=hf_repo_name, repo_type="model")
                            elif model_path and os.path.isfile(model_path):
                                api.upload_file(path_or_fileobj=model_path, path_in_repo=os.path.basename(model_path), repo_id=hf_repo_name, repo_type="model")
                            else:
                                st.error("Best model path missing.")
                                st.stop()
                            st.success(f"Uploaded to {hf_repo_name}")
                        except Exception as exc:
                            st.error(f"HF upload failed: {exc}")
            else:
                st.info("Select a primary metric to pick the best model.")
