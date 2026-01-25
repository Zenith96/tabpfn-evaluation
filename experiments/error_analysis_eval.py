# ============================================================
# ERROR ANALYSIS SCRIPT (IMPROVED — SAVES TRUE ERROR METRICS)
# ============================================================

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tabpfn import TabPFNClassifier

from utils.result_logger import save_result

# CONFIG
DATASET = "Breast Cancer"
EVAL_TYPE = "error_analysis"
CONDITION = "clean"
DATASET_PATH = "../datasets/breast_cancer/breast_cancer.csv"
RANDOM_STATE = 42

# LOAD DATA
df = pd.read_csv(DATASET_PATH)
X = pd.get_dummies(df.drop(columns=["target"]))
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# MODELS
models = {
    "TabPFN": TabPFNClassifier(device="cpu"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
}

error_indices = {}
accuracies = {}

# COLLECT ERROR INDICES
for name, model in models.items():

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    errors = set(X_test.index[preds != y_test])

    error_indices[name] = errors
    accuracies[name] = accuracy_score(y_test, preds)


# COMPUTE ERROR METRICS
all_errors = set().union(*error_indices.values())

for name in models.keys():

    model_errors = error_indices[name]

    others = set().union(*[
        error_indices[m] for m in models if m != name
    ])

    unique_errors = len(model_errors - others)
    total_errors = len(model_errors)
    shared_errors = len(model_errors & others)

    row = {
        "dataset": DATASET,
        "model": name,
        "evaluation_type": EVAL_TYPE,
        "condition": CONDITION,
        "seed": RANDOM_STATE,
        "accuracy": accuracies[name],
        "total_errors": total_errors,
        "unique_errors": unique_errors,
        "shared_errors": shared_errors,
        "test_size": len(X_test)
    }

  # ============================================================
# SAVE USING PANDAS (BYPASS save_result LIMITATION)
# ============================================================

rows = []

for name in models.keys():

    model_errors = error_indices[name]

    others = set().union(*[
        error_indices[m] for m in models if m != name
    ])

    unique_errors = len(model_errors - others)
    total_errors = len(model_errors)
    shared_errors = len(model_errors & others)

    row = {
        "dataset": DATASET,
        "model": name,
        "accuracy": accuracies[name],
        "total_errors": total_errors,
        "unique_errors": unique_errors,
        "shared_errors": shared_errors,
        "test_size": len(X_test)
    }

    rows.append(row)


# SAVE FULL DATAFRAME
output_df = pd.DataFrame(rows)

output_df.to_csv(
    "../results/error_analysis_detailed.csv",
    index=False
)

print(output_df)
print("Saved correct error_analysis_detailed.csv")