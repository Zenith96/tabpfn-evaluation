import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from tabpfn import TabPFNClassifier


# =======================
# CONFIG
# =======================
DATASET_PATH = "../datasets/breast_cancer/breast_cancer.csv"
TARGET_COLUMN = "target"

TRAIN_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]
RANDOM_STATE = 42


# =======================
# LOAD DATA
# =======================
df = pd.read_csv(DATASET_PATH)
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X = pd.get_dummies(X)


# =======================
# FIX TEST SET
# =======================
X_full_train, X_test, y_full_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)


# =======================
# MODELS
# =======================
models = {
    "TabPFN": TabPFNClassifier(
        device="cpu",
        ignore_pretraining_limits=True
    ),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE
    )
}


# =======================
# SIZE SENSITIVITY EXPERIMENT
# =======================
print("\nDATASET SIZE SENSITIVITY RESULTS\n")

for ratio in TRAIN_RATIOS:
    n_samples = int(len(X_full_train) * ratio)

    X_train = X_full_train.iloc[:n_samples]
    y_train = y_full_train.iloc[:n_samples]

    print(f"\n--- Training Data: {int(ratio * 100)}% ({n_samples} samples) ---")

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        print(f"{name:20s} | Acc: {acc:.4f} | F1: {f1:.4f}")
