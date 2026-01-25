import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier
from utils.result_logger import save_result

# Metadata
DATASET = "Adult Income"
MODEL_NAME = "TabPFN"
EVAL_TYPE = "standard"
CONDITION = "clean"

# Load dataset
df = pd.read_csv("../datasets/adult_income/adult_income.csv")

X = df.drop(columns=["target"])
y = df["target"]

# Encode categorical features
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = LabelEncoder().fit_transform(X[col])

SEEDS = [0, 1, 2, 3, 4]

for seed in SEEDS:
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Hard cap
    MAX_TRAIN_SAMPLES = 1000
    if len(X_train) > MAX_TRAIN_SAMPLES:
        X_train = X_train.sample(MAX_TRAIN_SAMPLES, random_state=seed)
        y_train = y_train.loc[X_train.index]

    model = TabPFNClassifier(device="cpu")

    start_fit = time.time()
    model.fit(X_train, y_train)
    fit_time = time.time() - start_fit

    start_pred = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_pred

    row = {
        "dataset": DATASET,
        "model": MODEL_NAME,
        "evaluation_type": EVAL_TYPE,
        "condition": CONDITION,
        "seed": seed,
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "fit_time": fit_time,
        "predict_time": predict_time
    }

    save_result(row, "results/standard_eval.csv")
