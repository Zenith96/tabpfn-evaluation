import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tabpfn import TabPFNClassifier

# Load dataset
df = pd.read_csv("../datasets/breast_cancer/breast_cancer.csv")

# Split features and target
X = df.drop(columns=["target"])
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize TabPFN
model = TabPFNClassifier(device="cpu")

# Fit (in-context learning)
model.fit(X_train, y_train)

# Measure inference time
start_time = time.time()
y_pred = model.predict(X_test)
end_time = time.time()

inference_time = end_time - start_time

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Output results
print("TabPFN Evaluation Results (Breast Cancer Dataset)")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Inference Time (seconds): {inference_time:.4f}")
