import pandas as pd
from sklearn.model_selection import train_test_split
from baseline_utils import evaluate_model, get_baseline_models


# CHANGE THIS PATH FOR DIFFERENT DATASETS
DATASET_PATH = "../datasets/breast_cancer/breast_cancer.csv"
TARGET_COLUMN = "target"


# Load dataset
df = pd.read_csv(DATASET_PATH)

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# One-hot encode if needed
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = get_baseline_models()

print("Baseline Model Evaluation Results")

for name, model in models.items():
    acc, f1, time_taken = evaluate_model(
        model, X_train, X_test, y_train, y_test
    )

    print(f"\n{name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Inference Time (seconds): {time_taken:.4f}")
