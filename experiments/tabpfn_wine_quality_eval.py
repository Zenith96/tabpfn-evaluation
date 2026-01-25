import sys
import os

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from tabpfn import TabPFNClassifier
from utils.result_logger import save_result

# =========================
# Metadata
# =========================
DATASET = "Wine Quality"
MODEL_NAME = "TabPFN"
EVAL_TYPE = "standard"
CONDITION = "clean"

# =========================
# Load dataset
# =========================
df = pd.read_csv("../datasets/wine_quality/wine_quality.csv")

X = df.drop(columns=["target"])
y = df["target"]

# =========================
# Evaluation
# =========================
SEEDS = [0, 1, 2, 3, 4]

for seed in SEEDS:
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=seed,
        stratify=y
    )

    # 🔑 HARD CAP for CPU (MANDATORY for Wine Quality)
    MAX_TRAIN_SAMPLES = 1000
    if len(X_train) > MAX_TRAIN_SAMPLES:
        X_train = X_train.sample(MAX_TRAIN_SAMPLES, random_state=seed)
        y_train = y_train.loc[X_train.index]

    # Model
    model = TabPFNClassifier(device="cpu")

    # Fit time
    start_fit = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    # Prediction time
    start_pred = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_pred

    # Metrics
    row = {
        "dataset": DATASET,
        "model": MODEL_NAME,
        "evaluation_type": EVAL_TYPE,
        "condition": CONDITION,
        "seed": seed,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="weighted"),
        "fit_time": fit_time,
        "predict_time": predict_time
    }

    # Save results
    save_result(row, "results/standard_eval.csv")
