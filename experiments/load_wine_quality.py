import pandas as pd

# Load Wine Quality dataset (red wine)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

df = pd.read_csv(url, sep=";")

# Convert to classification (binary)
df["target"] = df["quality"].apply(lambda x: 1 if x >= 6 else 0)
df.drop(columns=["quality"], inplace=True)

# Save dataset
df.to_csv("../datasets/wine_quality/wine_quality.csv", index=False)

print("Wine Quality dataset saved successfully.")
print("Shape:", df.shape)
