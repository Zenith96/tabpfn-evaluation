import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
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

DATASET = "Breast Cancer"
EVAL_TYPE = "seed_sensitivity"
DATASET_PATH = "../datasets/breast_cancer/breast_cancer.csv"
SEEDS = [0, 42, 99]

df = pd.read_csv(DATASET_PATH)
X = pd.get_dummies(df.drop(columns=["target"]))
y = df["target"]

models = {
    "TabPFN": TabPFNClassifier(device="cpu"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for seed in SEEDS:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # CPU safety
    if len(X_train) > 1000:
        X_train = X_train.sample(1000, random_state=seed)
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
            "condition": f"seed_{seed}",
            "seed": seed,
            "accuracy": accuracy_score(y_test, preds),
            "balanced_accuracy": balanced_accuracy_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "fit_time": fit_time,
            "predict_time": pred_time
        }

        save_result(row, "results/seed_sensitivity.csv")
