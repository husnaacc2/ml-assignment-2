import pandas as pd

# Raw column names from UCI
columns = [
    "id", "target",
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
    "smoothness_mean", "compactness_mean", "concavity_mean",
    "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se",
    "smoothness_se", "compactness_se", "concavity_se",
    "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst",
    "smoothness_worst", "compactness_worst", "concavity_worst",
    "concave_points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Read raw data
df = pd.read_csv("data/wdbc.data", header=None, names=columns)

# Convert labels: M -> 1, B -> 0
df["target"] = df["target"].map({"M": 1, "B": 0})

# Drop ID column
df.drop("id", axis=1, inplace=True)

# Move target column to the END
cols = [c for c in df.columns if c != "target"] + ["target"]
df = df[cols]

# Save final dataset
df.to_csv("data/breast_cancer.csv", index=False)

print("知道 breast_cancer.csv created successfully")
print("Shape:", df.shape)
print("Last column:", df.columns[-1])
