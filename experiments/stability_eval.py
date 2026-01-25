import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tabpfn import TabPFNClassifier

from utils.result_logger import save_result

DATASET = "Breast Cancer"
EVAL_TYPE = "stability"
DATASET_PATH = "../datasets/breast_cancer/breast_cancer.csv"
N_RUNS = 5

df = pd.read_csv(DATASET_PATH)
X = pd.get_dummies(df.drop(columns=["target"]))
y = df["target"]

models = {
    "TabPFN": TabPFNClassifier(device="cpu"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

for run in range(N_RUNS):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=run
    )

    # CPU safety
    if len(X_train) > 1000:
        X_train = X_train.sample(1000, random_state=run)
        y_train = y_train.loc[X_train.index]

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        row = {
            "dataset": DATASET,
            "model": model_name,
            "evaluation_type": EVAL_TYPE,
            "condition": f"run_{run}",
            "seed": run,
            "accuracy": accuracy_score(y_test, preds),
            "balanced_accuracy": balanced_accuracy_score(y_test, preds),
            "f1_score": f1_score(y_test, preds),
            "fit_time": None,
            "predict_time": None
        }

        save_result(row, "results/stability_eval.csv")
