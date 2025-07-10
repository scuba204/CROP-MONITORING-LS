import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os, json

np.random.seed(42)
n_samples = 2000

crop_types = ['maize', 'potato', 'beans', 'wheat']
crop_probs = [0.4, 0.2, 0.2, 0.2]

# District centroids and buffers (deg)
zones = {
    'maize':     {'lat': -29.5166, 'lon': 27.8311},
    'potato':    {'lat': -29.8041, 'lon': 27.5026},
    'beans':     {'lat': -28.8638, 'lon': 28.0479},
    'wheat':     {'lat': -28.7654, 'lon': 28.2468},
    'weed':      {'lat': -29.0,    'lon': 28.0}
}
buffer = 0.1  # ~11 km buffer

def sample_coords(crop):
    z = zones[crop]
    lon = np.random.uniform(z['lon'] - buffer, z['lon'] + buffer)
    lat = np.random.uniform(z['lat'] - buffer, z['lat'] + buffer)
    return lon, lat
def generate_sample(label, crop_type):
    x, y = sample_coords(crop_type if label == 0 else 'weed')

    if label == 0:
        feats = {
            'type': 'crop', 'crop_type': crop_type, 'label': label,
            'B2': np.random.normal(1200, 90),
            'B3': np.random.normal(1300, 85),
            'B4': np.random.normal(1400, 80),
            'B5': np.random.normal(1350, 80),
            'B6': np.random.normal(1450, 70),
            'B7': np.random.normal(1550, 60),
            'B8A': np.random.normal(1600, 60),
            'B11': np.random.normal(1650, 70),
            'B12': np.random.normal(1700, 75),
        }
    else:
        feats = {
            'type': 'weed', 'crop_type': 'weed', 'label': label,
            'B2': np.random.normal(1000, 100),
            'B3': np.random.normal(1100, 95),
            'B4': np.random.normal(1200, 90),
            'B5': np.random.normal(1150, 100),
            'B6': np.random.normal(1250, 90),
            'B7': np.random.normal(1350, 80),
            'B8A': np.random.normal(1400, 85),
            'B11': np.random.normal(1450, 90),
            'B12': np.random.normal(1500, 95),
        }

    # Add coordinates and additional features
    feats.update({
        'x': x, 'y': y,
        'ndvi': np.clip(np.random.normal(0.78 if label == 0 else 0.45, 0.04 if label == 0 else 0.08), 0, 1),
        'ndwi': np.clip(np.random.normal(0.3 if label == 0 else 0.12, 0.04 if label == 0 else 0.05), 0, 1),
        'lst': np.random.normal(293 if label == 0 else 297, 1.5 if label == 0 else 2.5),
        'irradiance': np.random.normal(210 if label == 0 else 180, 15 if label == 0 else 20)
    })

    return feats


samples = []
for _ in range(n_samples):
    label = np.random.choice([0, 1], p=[0.7, 0.3])
    crop = np.random.choice(crop_types, p=crop_probs) if label == 0 else 'weed'
    samples.append(generate_sample(label, crop))

df = pd.DataFrame(samples)
gdf = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df.x, df.y)], crs="EPSG:4326")

os.makedirs("data", exist_ok=True)
gdf.to_file("data/crop_weed_training.shp")
df.to_csv("data/crop_weed_training.csv", index=False)  # <-- Add this line to save x and y

with open("data/crop_weed_training_metadata.json", "w") as f:
    json.dump({
        "n_samples": n_samples,
        "zones": zones, "buffer_deg": buffer, "crs": "EPSG:4326",
        "features": list(df.columns)}, f, indent=2)

print("✅ Shapefile and CSV saved")
