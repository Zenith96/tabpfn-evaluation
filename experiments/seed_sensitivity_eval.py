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


# =======================
# CONFIG
# =======================
DATASET_PATH = "../datasets/breast_cancer/breast_cancer.csv"
TARGET_COLUMN = "target"
SEEDS = [0, 42, 99]


# =======================
# LOAD DATA
# =======================
df = pd.read_csv(DATASET_PATH)
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X = pd.get_dummies(X)


# =======================
# MODELS
# =======================
models = {
    "TabPFN": TabPFNClassifier(device="cpu", ignore_pretraining_limits=True),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}


# =======================
# SEED SENSITIVITY EVAL
# =======================
results = {
    name: {
        "acc": [],
        "bal_acc": [],
        "f1": [],
        "brier": [],
        "fit_time": [],
        "pred_time": []
    }
    for name in models
}

for seed in SEEDS:
    print(f"\n--- Seed = {seed} ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    for name, model in models.items():
        # Fit time
        start_fit = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_fit

        # Predict time
        start_pred = time.time()
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        pred_time = time.time() - start_pred

        acc = accuracy_score(y_test, preds)
        bal_acc = balanced_accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        brier = brier_score_loss(y_test, probs)

        results[name]["acc"].append(acc)
        results[name]["bal_acc"].append(bal_acc)
        results[name]["f1"].append(f1)
        results[name]["brier"].append(brier)
        results[name]["fit_time"].append(fit_time)
        results[name]["pred_time"].append(pred_time)

        print(
            f"{name:20s} | "
            f"Acc: {acc:.4f} | "
            f"BalAcc: {bal_acc:.4f} | "
            f"F1: {f1:.4f} | "
            f"Brier: {brier:.4f}"
        )


# =======================
# SUMMARY
# =======================
print("\nSEED SENSITIVITY SUMMARY (mean ± std)\n")

for name, m in results.items():
    print(name)
    print(f"  Accuracy:          {np.mean(m['acc']):.4f} ± {np.std(m['acc']):.4f}")
    print(f"  Balanced Accuracy: {np.mean(m['bal_acc']):.4f} ± {np.std(m['bal_acc']):.4f}")
    print(f"  F1-score:          {np.mean(m['f1']):.4f} ± {np.std(m['f1']):.4f}")
    print(f"  Brier Score:       {np.mean(m['brier']):.4f} ± {np.std(m['brier']):.4f}")
    print(f"  Fit Time (s):      {np.mean(m['fit_time']):.4f} ± {np.std(m['fit_time']):.4f}")
    print(f"  Pred Time (s):     {np.mean(m['pred_time']):.4f} ± {np.std(m['pred_time']):.4f}\n")
