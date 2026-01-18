import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tabpfn import TabPFNClassifier


# =======================
# CONFIG
# =======================
DATASET_PATH = "../datasets/breast_cancer/breast_cancer.csv"
TARGET_COLUMN = "target"
N_RUNS = 5


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
# MULTI-RUN EVALUATION
# =======================
results = {
    name: {"acc": [], "bal_acc": [], "f1": []}
    for name in models
}

for run in range(N_RUNS):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=run
    )

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results[name]["acc"].append(accuracy_score(y_test, preds))
        results[name]["bal_acc"].append(balanced_accuracy_score(y_test, preds))
        results[name]["f1"].append(f1_score(y_test, preds))


# =======================
# OUTPUT RESULTS
# =======================
print("\nSTABILITY EVALUATION RESULTS\n")

for name, metrics in results.items():
    acc_mean = np.mean(metrics["acc"])
    acc_std = np.std(metrics["acc"])

    bal_mean = np.mean(metrics["bal_acc"])
    bal_std = np.std(metrics["bal_acc"])

    f1_mean = np.mean(metrics["f1"])
    f1_std = np.std(metrics["f1"])

    print(f"{name}")
    print(f"  Accuracy:          {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  Balanced Accuracy: {bal_mean:.4f} ± {bal_std:.4f}")
    print(f"  F1-score:          {f1_mean:.4f} ± {f1_std:.4f}\n")
