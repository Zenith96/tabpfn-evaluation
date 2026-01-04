import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    start = time.time()
    y_pred = model.predict(X_test)
    end = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    inference_time = end - start

    return accuracy, f1, inference_time


def get_baseline_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
