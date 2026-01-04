import pandas as pd
from sklearn.datasets import fetch_openml

# Load Adult Income dataset
data = fetch_openml(name="adult", version=2, as_frame=True)
df = data.frame

# Rename target column
df.rename(columns={"class": "target"}, inplace=True)

# Encode target
df["target"] = df["target"].apply(lambda x: 1 if x == ">50K" else 0)

# Save dataset
df.to_csv("../datasets/adult_income/adult_income.csv", index=False)

print("Adult Income dataset saved successfully.")
print("Shape:", df.shape)
