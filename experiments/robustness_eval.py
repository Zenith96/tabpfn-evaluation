import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    brier_score_loss
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tabpfn import TabPFNClassifier

from utils.result_logger import save_result

# =======================
# CONFIG
# =======================
DATASET = "Breast Cancer"
EVAL_TYPE = "robustness"
DATASET_PATH = "../datasets/breast_cancer/breast_cancer.csv"

NOISE_LEVEL = 0.1
DUPLICATE_RATIO = 0.3
REDUCED_RATIO = 0.5

# =======================
# LOAD DATA
# =======================
df = pd.read_csv(DATASET_PATH)
X = pd.get_dummies(df.drop(columns=["target"]))
y = df["target"]

# =======================
# ROBUSTNESS FUNCTIONS
# =======================
def add_noise(X, noise_level):
    return X + np.random.normal(0, noise_level, X.shape)

def add_duplicates(X, y, ratio):
    n_dup = int(len(X) * ratio)
    idx = np.random.choice(len(X), n_dup, replace=True)
    return (
        pd.concat([X, X.iloc[idx]], ignore_index=True),
        pd.concat([y, y.iloc[idx]], ignore_index=True)
    )

def reduce_dataset(X, y, ratio):
    n = int(len(X) * ratio)
    return X.iloc[:n], y.iloc[:n]

# =======================
# MODELS
# =======================
models = {
    "TabPFN": TabPFNClassifier(device="cpu"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

experiments = {
    "original": (X, y),
    "noise": (add_noise(X, NOISE_LEVEL), y),
    "duplicates": add_duplicates(X, y, DUPLICATE_RATIO),
    "reduced": reduce_dataset(X, y, REDUCED_RATIO)
}

# =======================
# RUN EXPERIMENTS
# =======================
for condition, (X_exp, y_exp) in experiments.items():
    X_train, X_test, y_train, y_test = train_test_split(
        X_exp, y_exp, test_size=0.2, random_state=42
    )

    # CPU safety for TabPFN
    if len(X_train) > 1000:
        X_train = X_train.sample(1000, random_state=42)
        y_train = y_train.loc[X_train.index]

    for model_name, model in models.items():
        start_fit = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_fit

        start_pred = time.time()
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        pred_time = time.time() - start_pred

        row = {
            "dataset": DATASET,
            "model": model_name,
            "evaluation_type": EVAL_TYPE,
            "condition": condition,
            "seed": 42,
            "accuracy": accuracy_score(y_test, preds),
            "balanced_accuracy": balanced_accuracy_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "fit_time": fit_time,
            "predict_time": pred_time
        }

        save_result(row, "results/robustness_eval.csv")
