"""Streamlit entry point for library-managed tabular comparisons."""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold, TimeSeriesSplit

from ml_pipeline.skore_benchmark import evaluate_models, load_report, registry_estimator
from ml_pipeline.utils.utils import RESULTS_DIR_OUT


def comparison_splitter(task, protocol, folds, seed):
    if protocol == "Disjoint groups":
        return GroupKFold(folds)
    if protocol == "Chronological row order":
        return TimeSeriesSplit(folds)
    cls = StratifiedKFold if task == "classification" else KFold
    return cls(folds, shuffle=True, random_state=seed)


def comparison_models(names, task, parameters, seed):
    models = {}
    suffix = "classifier" if task == "classification" else "regressor"
    for name in names:
        params = {"random_state": seed, **parameters.get(name, {})}
        if name == "hist_gradient_boosting":
            cls = HistGradientBoostingClassifier if task == "classification" else HistGradientBoostingRegressor
            models[name] = cls(**params)
        else:
            models[name] = registry_estimator(f"{name}_{suffix}", **params)
    return models


def render_standard_comparison():
    st.subheader("Standard model comparison")
    st.caption("Compare models with skrub preprocessing inside each training fold and skore reports. "
               "Use development data here; keep a final test set separate.")
    root = Path(RESULTS_DIR_OUT) / "standard_comparisons"
    with st.sidebar:
        source = st.radio("Dataset source", ["Upload CSV", "Path"], key="standard_source")
        frame = None
        if source == "Upload CSV":
            upload = st.file_uploader("CSV file", type=["csv"], key="standard_upload")
            if upload is not None:
                frame = pd.read_csv(upload)
        else:
            path = st.text_input("CSV path", key="standard_path")
            if path:
                try:
                    frame = pd.read_csv(path)
                except (OSError, ValueError) as error:
                    st.error(str(error))
        task = st.selectbox("Task", ["classification", "regression"], key="standard_task")
        protocol = st.selectbox("Validation", ["Independent rows", "Disjoint groups", "Chronological row order"],
                                key="standard_protocol")
        folds = st.number_input("Folds", min_value=2, max_value=10, value=5, key="standard_folds")
        seed = st.number_input("Seed", min_value=0, value=42, key="standard_seed")
        jobs = st.number_input("Parallel jobs", min_value=1, max_value=8, value=1, key="standard_jobs")
        names = st.multiselect("Models", ["linear", "random_forest", "hist_gradient_boosting",
                                          "constant", "xgboost", "lightgbm", "catboost", "tabicl", "kumo"],
                               default=["linear", "random_forest", "hist_gradient_boosting"], key="standard_models")
        native = st.checkbox("Use model-native preprocessing", value=False, key="standard_native")
        if "kumo" in names:
            st.caption("For Kumo's mixed-table recipe, use model-native preprocessing. "
                       "Set size and device in its parameters. Kumo needs the optional kumo environment.")
        params_text = st.text_area("Per-model parameters (JSON)", value="{}", key="standard_params")
        run_name = st.text_input("New run name", value=datetime.now().strftime("%Y%m%d_%H%M%S"),
                                 key="standard_name")

    if frame is not None:
        st.dataframe(frame.head(), use_container_width=True)
        target = st.selectbox("Target", frame.columns.tolist(), key="standard_target")
        group = None
        if protocol == "Disjoint groups":
            group = st.selectbox("Group column", [c for c in frame.columns if c != target], key="standard_group")
        columns = [c for c in frame.columns if c not in {target, group}]
        features = st.multiselect("Features", columns, default=columns, key=f"standard_features_{target}_{group}")
        if protocol == "Chronological row order":
            st.caption("Rows must already be in chronological order. This split does not purge forward labels; "
                       "use the research runtime for those protocols.")
        if st.button("Run comparison", key="standard_run"):
            try:
                if not names or not features or frame[target].isna().any():
                    raise ValueError("Select models and features, and provide nonmissing targets.")
                if not run_name or Path(run_name).name != run_name or run_name in {".", ".."}:
                    raise ValueError("Use a single new folder name for this comparison.")
                parameters = json.loads(params_text)
                if (not isinstance(parameters, dict) or set(parameters) - set(names)
                        or any(not isinstance(p, dict) for p in parameters.values())):
                    raise ValueError("Parameters must map selected model names to parameter objects.")
                models = comparison_models(names, task, parameters, int(seed))
                with st.spinner("Evaluating development folds..."):
                    evaluate_models(
                        models, frame[features], frame[target],
                        splitter=comparison_splitter(task, protocol, int(folds), int(seed)),
                        groups=frame[group] if group else None,
                        preprocessing="none" if native else "skrub", n_jobs=int(jobs),
                        output_dir=root / run_name,
                    )
                st.session_state["standard_last_run"] = run_name
                st.success(f"Saved comparison to {root / run_name}")
            except (ImportError, ValueError, TypeError, OSError) as error:
                st.error(str(error))
    else:
        st.info("Load a CSV to configure a comparison, or reopen a saved report below.")

    runs = sorted(p.name for p in root.glob("*") if (p / "report.joblib").is_file())
    if runs:
        last = st.session_state.get("standard_last_run", runs[-1])
        selected = st.selectbox("Saved comparison", runs, index=runs.index(last) if last in runs else 0,
                                key="standard_saved")
        try:
            report = load_report(root / selected / "report.joblib")
            st.dataframe(report.metrics.summarize().frame(), use_container_width=True)
            st.download_button("Download metrics", (root / selected / "metrics.csv").read_bytes(),
                               file_name=f"{selected}_metrics.csv", mime="text/csv")
        except ImportError as error:
            st.error(f"Install the evaluation extra to reopen this report: {error}")
