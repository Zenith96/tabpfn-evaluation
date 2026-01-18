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
TEST_SIZE = 0.2
RANDOM_STATE = 42


# =======================
# LOAD DATA
# =======================
df = pd.read_csv(DATASET_PATH)
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
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
        random_state=42
    )
}


# =======================
# EVALUATION
# =======================
rows = []

for name, model in models.items():
    # ---- Fit time ----
    start_fit = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    # ---- Predict time ----
    start_pred = time.time()
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    pred_time = time.time() - start_pred

    # ---- Metrics ----
    acc = accuracy_score(y_test, preds)
    bal_acc = balanced_accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    brier = brier_score_loss(y_test, probs)

    rows.append({
        "Model": name,
        "Accuracy": acc,
        "Balanced Accuracy": bal_acc,
        "F1-score": f1,
        "Brier Score": brier,
        "Fit Time (s)": fit_time,
        "Predict Time (s)": pred_time
    })


# =======================
# OUTPUT
# =======================
results_df = pd.DataFrame(rows)

print("\nFINAL MODEL COMPARISON SUMMARY\n")
print(results_df.round(4))

# Optional: save for paper / report
results_df.to_csv(
    "../results/tabpfn_vs_baselines_summary.csv",
    index=False
)
