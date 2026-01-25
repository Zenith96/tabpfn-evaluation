import pandas as pd
import os

COLUMNS = [
    "dataset",
    "model",
    "evaluation_type",
    "condition",
    "seed",
    "accuracy",
    "balanced_accuracy",
    "f1_score",
    "fit_time",
    "predict_time"
]

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def save_result(row: dict, relative_path: str):
    full_path = os.path.join(PROJECT_ROOT, relative_path)

    df = pd.DataFrame([row], columns=COLUMNS)

    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    if os.path.exists(full_path):
        df.to_csv(full_path, mode="a", header=False, index=False)
    else:
        df.to_csv(full_path, index=False)
