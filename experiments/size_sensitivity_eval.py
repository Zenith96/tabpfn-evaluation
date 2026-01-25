import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tabpfn import TabPFNClassifier

from utils.result_logger import save_result

DATASET = "Breast Cancer"
EVAL_TYPE = "size_sensitivity"
DATASET_PATH = "../datasets/breast_cancer/breast_cancer.csv"
TRAIN_RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]
RANDOM_STATE = 42

df = pd.read_csv(DATASET_PATH)
X = pd.get_dummies(df.drop(columns=["target"]))
y = df["target"]

X_full_train, X_test, y_full_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

models = {
    "TabPFN": TabPFNClassifier(device="cpu"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
}

for ratio in TRAIN_RATIOS:
    n_samples = int(len(X_full_train) * ratio)

    X_train = X_full_train.iloc[:n_samples]
    y_train = y_full_train.iloc[:n_samples]

    # CPU safety
    if len(X_train) > 1000:
        X_train = X_train.sample(1000, random_state=RANDOM_STATE)
        y_train = y_train.loc[X_train.index]

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        row = {
            "dataset": DATASET,
            "model": model_name,
            "evaluation_type": EVAL_TYPE,
            "condition": f"train_ratio_{ratio}",
            "seed": RANDOM_STATE,
            "accuracy": accuracy_score(y_test, preds),
            "balanced_accuracy": None,
            "f1_score": f1_score(y_test, preds),
            "fit_time": None,
            "predict_time": None
        }

        save_result(row, "results/size_sensitivity.csv")
