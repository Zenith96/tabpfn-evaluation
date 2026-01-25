import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from tabpfn import TabPFNClassifier
from utils.result_logger import save_result

DATASET = "Breast Cancer"
MODEL_NAME = "TabPFN"
EVAL_TYPE = "standard"
CONDITION = "clean"

df = pd.read_csv("../datasets/breast_cancer/breast_cancer.csv")

X = df.drop(columns=["target"])
y = df["target"]

SEEDS = [0, 1, 2, 3, 4]

for seed in SEEDS:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

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
