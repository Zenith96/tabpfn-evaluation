import pandas as pd
import numpy as np
import time

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
NOISE_LEVEL = 0.1          # 10% Gaussian noise
DUPLICATE_RATIO = 0.3      # 30% duplicated rows
REDUCED_RATIO = 0.5        # use only 50% of data


# =======================
# DATA LOADING
# =======================
df = pd.read_csv(DATASET_PATH)
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X = pd.get_dummies(X)


# =======================
# ROBUSTNESS FUNCTIONS
# =======================
def add_noise(X, noise_level):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise


def add_duplicates(X, y, ratio):
    n_dup = int(len(X) * ratio)
    dup_indices = np.random.choice(len(X), n_dup, replace=True)

    X_dup = X.iloc[dup_indices]
    y_dup = y.iloc[dup_indices]

    X_new = pd.concat([X, X_dup], ignore_index=True)
    y_new = pd.concat([y, y_dup], ignore_index=True)

    return X_new, y_new


def reduce_dataset(X, y, ratio):
    n = int(len(X) * ratio)
    return X.iloc[:n], y.iloc[:n]


# =======================
# MODEL EVALUATION
# =======================
def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    start = time.time()
    preds = model.predict(X_test)
    end = time.time()

    return (
        accuracy_score(y_test, preds),
        f1_score(y_test, preds),
        end - start
    )


# =======================
# EXPERIMENTS
# =======================
experiments = {
    "Original": (X, y),
    "With Noise": (add_noise(X, NOISE_LEVEL), y),
    "With Duplicates": add_duplicates(X, y, DUPLICATE_RATIO),
    "Reduced Data": reduce_dataset(X, y, REDUCED_RATIO)
}

models = {
    "TabPFN": TabPFNClassifier(device="cpu", ignore_pretraining_limits=True),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

print("\nROBUSTNESS EVALUATION RESULTS\n")

for exp_name, (X_exp, y_exp) in experiments.items():
    print(f"\n--- {exp_name} ---")

    X_train, X_test, y_train, y_test = train_test_split(
        X_exp, y_exp, test_size=0.2, random_state=42
    )

    for model_name, model in models.items():
        acc, f1, t = evaluate(model, X_train, X_test, y_train, y_test)
        print(f"{model_name:20s} | Acc: {acc:.4f} | F1: {f1:.4f} | Time: {t:.4f}s")
