import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tabpfn import TabPFNClassifier

# Load dataset
df = pd.read_csv("../datasets/adult_income/adult_income.csv")

X = df.drop(columns=["target"])
y = df["target"]

# One-hot encode categorical features
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = TabPFNClassifier(device="cpu",ignore_pretraining_limits=True)
model.fit(X_train, y_train)

start = time.time()
y_pred = model.predict(X_test)
end = time.time()

print("TabPFN Evaluation - Adult Income Dataset")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1-score:", f1_score(y_test, y_pred))
print("Inference Time (s):", end - start)
