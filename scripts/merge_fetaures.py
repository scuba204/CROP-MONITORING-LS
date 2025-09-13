# clean_rotation_dataset.py
import pandas as pd

# Load merged dataset
df = pd.read_csv("rotation_training.csv")

# === Drop duplicate/irrelevant columns ===
df = df.drop(columns=[
    "Area_y", "Yield_Unit_y", "Yield_y", "Flag_y", "Flag Description_y",
    "prev_area_share_y"
], errors="ignore")

# === Rename columns to standard names ===
df = df.rename(columns={
    "Area_x": "Area",
    "Yield_x": "Yield",
    "Yield_Unit_x": "Yield_Unit",
    "Flag_x": "Flag",
    "Flag Description_x": "Flag_Description",
    "prev_area_share_x": "prev_area_share"
})

# === Reorder columns for readability ===
col_order = [
    "Year", "Item", "Area", "Yield", "Yield_Unit", "Flag", "Flag_Description",
    "Area_ha", "total_area", "area_share", "prev_area_share",
    # proxy features (FAOSTAT rotation proxy file probably added columns here)
] + [c for c in df.columns if c not in [
    "Year", "Item", "Area", "Yield", "Yield_Unit", "Flag", "Flag_Description",
    "Area_ha", "total_area", "area_share", "prev_area_share"
]]

df = df[col_order]

# === Save cleaned dataset ===
df.to_csv("rotation_training_clean.csv", index=False)

print("✅ Cleaned dataset saved to rotation_training_clean.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
