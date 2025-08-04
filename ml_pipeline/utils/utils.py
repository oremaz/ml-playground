import os
import joblib
import torch
import pandas as pd
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def save_model(model, model_name, path_start=None):
    if path_start is not None:
        base_dir = os.path.join(RESULTS_DIR, path_start)
    else:
        base_dir = RESULTS_DIR
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, f"{model_name}.pt")
    if hasattr(model, "state_dict"):
        torch.save(model.state_dict(), path)
    elif hasattr(model, "model"):
        joblib.dump(model.model, path)
    elif hasattr(model, "save_pretrained"):
        model.save_pretrained(os.path.join(base_dir, model_name))
    # Add more logic for LLMs if needed

def load_model(model_class, model_name, params, path_start=None):
    if path_start is not None:
        path = os.path.join(RESULTS_DIR, path_start, f"{model_name}.pt")
    else:
        path = os.path.join(RESULTS_DIR, f"{model_name}.pt")
    model = model_class(**params)
    if hasattr(model, "load_state_dict"):
        model.load_state_dict(torch.load(path))
    elif hasattr(model, "model"):
        model.model = joblib.load(path)
    elif hasattr(model, "from_pretrained"):
        model = model_class.from_pretrained(os.path.join(RESULTS_DIR, model_name))
    return model

def save_metrics(metrics, model_name, phase, path_start=None):
    if path_start is not None:
        base_dir = os.path.join(RESULTS_DIR, path_start)
    else:
        base_dir = RESULTS_DIR
    os.makedirs(base_dir, exist_ok=True)
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(base_dir, f"{model_name}_{phase}_metrics.csv"), index=False)
