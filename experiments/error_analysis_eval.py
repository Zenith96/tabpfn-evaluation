import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from tabpfn import TabPFNClassifier


# =======================
# CONFIG
# =======================
DATASET_PATH = "../datasets/breast_cancer/breast_cancer.csv"
TARGET_COLUMN = "target"
RANDOM_STATE = 42


# =======================
# LOAD DATA
# =======================
df = pd.read_csv(DATASET_PATH)
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X = pd.get_dummies(X)


# =======================
# TRAIN-TEST SPLIT
# =======================
X_train, X_test, y_train, y_test = train_test_split(
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
# FIT & COLLECT ERRORS
# =======================
error_indices = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # indices where prediction is wrong
    errors = X_test.index[preds != y_test]
    error_indices[name] = set(errors)

    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")


# =======================
# ERROR OVERLAP ANALYSIS
# =======================
print("\nERROR OVERLAP ANALYSIS\n")

model_names = list(error_indices.keys())

for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        m1, m2 = model_names[i], model_names[j]
        overlap = error_indices[m1].intersection(error_indices[m2])

        print(
            f"{m1} ∩ {m2}: {len(overlap)} shared misclassified samples"
        )


# =======================
# UNIQUE ERRORS
# =======================
print("\nUNIQUE ERRORS\n")

for name in model_names:
    others = set().union(
        *[error_indices[m] for m in model_names if m != name]
    )
    unique = error_indices[name] - others

    print(f"{name}: {len(unique)} unique misclassifications")
