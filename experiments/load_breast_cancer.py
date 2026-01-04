from sklearn.datasets import load_breast_cancer
import pandas as pd

# Load dataset
data = load_breast_cancer()

# Create DataFrame
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Combine features and target
df = pd.concat([X, y], axis=1) # Axis = 1 means we need to add by columns 

# Save to CSV
df.to_csv("../datasets/breast_cancer/breast_cancer.csv", index=False)

print("Breast Cancer dataset saved successfully.")
print("Shape:", df.shape)
