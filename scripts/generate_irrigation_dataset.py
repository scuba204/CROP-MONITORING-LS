# generate_irrigation_dataset.py

import pandas as pd
import numpy as np

# -------------------------------
# CONFIG
# -------------------------------
CROPS = ["Barley", "Oats", "Maize", "Tomato", "Potato",
         "Cabbage", "Sorghum", "Wheat", "Beans", "Green Beans", "Sweet Potato"]

GROWTH_STAGES = ["Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity"]
SOIL_TEXTURES = ["Clay", "Loam", "Sandy Loam"]
IRRIGATION_METHODS = ["Drip", "Sprinkler", "Furrow"]

NUM_SAMPLES_PER_CROP = 50
PLOT_AREA_M2 = [50, 100, 150, 200]
OUTPUT_CSV = "data/irrigation_training.csv"

# -------------------------------
# Crop-specific Kc values (FAO-56 style, simplified per stage)
# -------------------------------
CROP_KC = {
    "Maize":        {"Seedling": 0.35, "Vegetative": 0.7, "Flowering": 1.2, "Fruiting": 1.15, "Maturity": 0.6},
    "Wheat":        {"Seedling": 0.35, "Vegetative": 0.75, "Flowering": 1.15, "Fruiting": 1.05, "Maturity": 0.4},
    "Barley":       {"Seedling": 0.35, "Vegetative": 0.75, "Flowering": 1.15, "Fruiting": 1.05, "Maturity": 0.4},
    "Oats":         {"Seedling": 0.35, "Vegetative": 0.75, "Flowering": 1.15, "Fruiting": 1.05, "Maturity": 0.4},
    "Sorghum":      {"Seedling": 0.35, "Vegetative": 0.7, "Flowering": 1.1, "Fruiting": 1.05, "Maturity": 0.35},
    "Potato":       {"Seedling": 0.5,  "Vegetative": 0.8, "Flowering": 1.15, "Fruiting": 1.1, "Maturity": 0.75},
    "Sweet Potato": {"Seedling": 0.5,  "Vegetative": 0.75, "Flowering": 1.15, "Fruiting": 1.05, "Maturity": 0.7},
    "Tomato":       {"Seedling": 0.5,  "Vegetative": 0.85, "Flowering": 1.15, "Fruiting": 1.1, "Maturity": 0.85},
    "Cabbage":      {"Seedling": 0.45, "Vegetative": 0.9, "Flowering": 1.1, "Fruiting": 1.05, "Maturity": 0.95},
    "Beans":        {"Seedling": 0.4,  "Vegetative": 0.8, "Flowering": 1.1, "Fruiting": 1.0, "Maturity": 0.9},
    "Green Beans":  {"Seedling": 0.4,  "Vegetative": 0.8, "Flowering": 1.1, "Fruiting": 1.0, "Maturity": 0.9}
}

# -------------------------------
# Helper: compute crop water requirement in liters
# -------------------------------
def compute_optimal_water_liters(et0_mm, kc, rainfall_mm, area_m2):
    cwr_mm = max(et0_mm * kc - rainfall_mm, 0)  # ensure >= 0
    return cwr_mm * area_m2  # mm × m² = liters

# -------------------------------
# Generate dataset
# -------------------------------
data = []
np.random.seed(42)

for crop in CROPS:
    for _ in range(NUM_SAMPLES_PER_CROP):
        growth_stage = np.random.choice(GROWTH_STAGES)
        soil_texture = np.random.choice(SOIL_TEXTURES)
        soil_depth_cm = np.random.randint(30, 80)
        water_holding_capacity_mm = np.random.randint(50, 120)
        organic_matter_pct = np.random.uniform(2, 8)
        pH = np.random.uniform(5.5, 7.5)

        temp_max_C = np.random.randint(25, 35)
        temp_min_C = np.random.randint(15, 25)
        rainfall_mm = np.random.randint(0, 10)
        solar_radiation_MJ_m2 = np.random.randint(15, 25)
        relative_humidity_pct = np.random.randint(50, 80)
        wind_speed_m_s = np.random.randint(1, 5)
        et0_mm = np.random.uniform(3.5, 5.5)

        plant_density_per_m2 = np.random.randint(2, 5)
        row_spacing_m = np.random.choice([0.5, 0.75, 1.0])
        irrigation_method = np.random.choice(IRRIGATION_METHODS)
        area_m2 = np.random.choice(PLOT_AREA_M2)

        # crop-specific Kc
        kc = CROP_KC[crop][growth_stage]

        optimal_water_liters = compute_optimal_water_liters(et0_mm, kc, rainfall_mm, area_m2)

        row = {
            "crop_type": crop,
            "growth_stage": growth_stage,
            "soil_texture": soil_texture,
            "soil_depth_cm": soil_depth_cm,
            "water_holding_capacity_mm": water_holding_capacity_mm,
            "organic_matter_%": round(organic_matter_pct, 1),
            "pH": round(pH, 1),
            "temp_max_C": temp_max_C,
            "temp_min_C": temp_min_C,
            "rainfall_mm": rainfall_mm,
            "solar_radiation_MJ_m2": solar_radiation_MJ_m2,
            "relative_humidity_%": relative_humidity_pct,
            "wind_speed_m_s": wind_speed_m_s,
            "et0_mm": round(et0_mm, 2),
            "plant_density_per_m2": plant_density_per_m2,
            "row_spacing_m": row_spacing_m,
            "irrigation_method": irrigation_method,
            "area_m2": area_m2,
            "kc_value": kc,  # <-- store kc used
            "optimal_water_liters": round(optimal_water_liters, 2)
        }
        data.append(row)

# -------------------------------
# Save dataset
# -------------------------------
df = pd.DataFrame(data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Dataset saved to {OUTPUT_CSV}")
print(df.head())
