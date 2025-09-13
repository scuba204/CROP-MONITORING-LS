import pandas as pd

# === Load datasets ===
env = pd.read_csv("gee_env_features_final.csv")  # Environmental features
faostat_yield = pd.read_csv("faostat_lesotho_yield_only.csv")  # Yield data
faostat_area = pd.read_csv("faostat_lesotho_area_shares.csv")  # Area share data
faostat_rotation_proxy = pd.read_csv("faostat_rotation_proxy_features.csv")  # Crop rotation proxy data

# === Merge FAOSTAT files ===
faostat = pd.merge(
    faostat_yield,
    faostat_area,
    on=["Year", "Item"],
    how="outer"
)

# === Merge with environmental features ===
merged = pd.merge(
    faostat,
    env,
    on=["Year", "Item"],
    how="inner"  # keep only overlapping Year + Item
)

# === Save merged dataset ===
merged.to_csv("crop_rotation_training_data.csv", index=False)
print("✅ Saved merged dataset -> crop_rotation_training_data.csv")

# === Summary Checks ===
print("\n📊 Dataset Summary")
print("-" * 40)
print("Rows:", len(merged))
print("Columns:", merged.shape[1])
print("\nCrops included:", merged["Item"].unique())
print("\nRecords per crop:")
print(merged["Item"].value_counts())

print("\n🔍 Missing values check:")
print(merged.isnull().sum())
