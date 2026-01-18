import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
from tabpfn import TabPFNClassifier

# Load dataset
df = pd.read_csv("../datasets/adult_income/adult_income.csv")

X = df.drop(columns=["target"])
y = df["target"]

# Encode categorical features
for col in X.columns:
    if X[col].dtype == "object":
        X[col] = LabelEncoder().fit_transform(X[col])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔑 HARD CAP (this is the key)
MAX_TRAIN_SAMPLES = 1000
if len(X_train) > MAX_TRAIN_SAMPLES:
    X_train = X_train.sample(MAX_TRAIN_SAMPLES, random_state=42)
    y_train = y_train.loc[X_train.index]

# Model
model = TabPFNClassifier(device="cpu")
model.fit(X_train, y_train)

# Inference
start = time.time()
y_pred = model.predict(X_test)
end = time.time()

# Metrics
print("TabPFN Evaluation - Adult Income Dataset")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred, average="binary"))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("Inference Time (s):", end - start)
