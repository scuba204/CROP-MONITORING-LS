import numpy as np
import pandas as pd
import os
import json

np.random.seed(42)
n_samples = 2000  # Increased to allow more variation

# Crop type distribution
crop_types = ['maize', 'potato', 'sweet_potato']
crop_probs = [0.5, 0.3, 0.2]

# Coordinate bounds for Lesotho (approximate)
lat_min, lat_max = -30.6, -28.5
lon_min, lon_max = 27.0, 29.5

# Function to generate synthetic sample
def generate_sample(label, crop):
    if label == 0:  # Healthy
        ndvi = np.random.normal(0.7, 0.05)
        return {
            'b5': np.random.normal(1400, 100),
            'b6': np.random.normal(1600, 100),
            'b7': np.random.normal(1700, 120),
            'b11': np.random.normal(1800, 100),
            'b12': np.random.normal(1900, 100),
            'ndvi': ndvi,
            'ndwi': np.random.normal(0.3, 0.05),
            'soil_moisture': np.random.normal(0.25, 0.05),
            'temperature': np.random.normal(298, 3),  # Kelvin
            'humidity': np.random.normal(65, 10),
            'precipitation': np.random.normal(2.5, 1.0),
            'crop_type': crop,
            'label': label
        }
    else:  # Diseased
        ndvi = np.random.normal(0.4, 0.1)
        return {
            'b5': np.random.normal(1100, 100),
            'b6': np.random.normal(1300, 100),
            'b7': np.random.normal(1400, 120),
            'b11': np.random.normal(2100, 100),
            'b12': np.random.normal(2200, 100),
            'ndvi': ndvi,
            'ndwi': np.random.normal(0.1, 0.05),
            'soil_moisture': np.random.normal(0.15, 0.05),
            'temperature': np.random.normal(303, 3),
            'humidity': np.random.normal(75, 10),
            'precipitation': np.random.normal(1.5, 0.8),
            'crop_type': crop,
            'label': label
        }

# Generate dataset
samples = []
for _ in range(n_samples):
    label = np.random.choice([0, 1], p=[0.6, 0.4])
    crop = np.random.choice(crop_types, p=crop_probs)
    sample = generate_sample(label, crop)
    # Generate random coordinates within bounds
    sample['x'] = np.random.uniform(lon_min, lon_max)
    sample['y'] = np.random.uniform(lat_min, lat_max)
    samples.append(sample)

df = pd.DataFrame(samples)
df['ndvi'] = df['ndvi'].clip(0, 1)
df['ndwi'] = df['ndwi'].clip(0, 1)

# Save to file
os.makedirs("data", exist_ok=True)
df.to_csv("data/disease_training.csv", index=False)

# Save metadata
metadata = {
    "description": "Synthetic crop disease training data with spatial coordinates",
    "n_samples": n_samples,
    "label_distribution": {"healthy": 0.6, "diseased": 0.4},
    "crop_types": crop_types,
    "coordinate_bounds": {
        "lat_min": lat_min, "lat_max": lat_max,
        "lon_min": lon_min, "lon_max": lon_max
    },
    "features": list(df.columns)
}

with open("data/disease_training_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Synthetic dataset with coordinates saved to data/disease_training.csv")
print("📄 Metadata saved to data/disease_training_metadata.json")
