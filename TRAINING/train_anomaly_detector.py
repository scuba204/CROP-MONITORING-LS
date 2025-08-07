import sys
import os
import ee
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import IsolationForest
import joblib
import json

# ==============================================================================
# Step 0: Initial Configuration and Path Setup
# ==============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.extract_features import extract_spectral_features

# --- Configuration ---
csv_file_name = "crop_weed_training.csv"
model_dir = "models"
model_name = "crop_anomaly_detector.pkl"

spectral_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
training_start_date = "2025-07-26"
training_end_date = "2025-08-02"

# ==============================================================================
# Helper Function for Feature Engineering
# ==============================================================================

def calculate_vegetation_indices(df):
    """
    Calculates various vegetation indices from Sentinel-2 band data.
    """
    df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'])
    df['EVI'] = 2.5 * (df['B8'] - df['B4']) / (df['B8'] + 6 * df['B4'] - 7.5 * df['B3'] + 1)
    df['NDWI'] = (df['B8'] - df['B11']) / (df['B8'] + df['B11'])
    df['NDRE'] = (df['B8'] - df['B5']) / (df['B8'] + df['B5'])
    df['GNDVI'] = (df['B8'] - df['B3']) / (df['B8'] + df['B3'])
    df['MSI'] = df['B11'] / df['B8']
    df.replace([float('inf'), float('-inf')], 0, inplace=True)
    return df

# ==============================================================================
# Main Anomaly Detection Training Pipeline
# ==============================================================================

# === STEP 1: Load Labeled Field Data and Filter for 'crop' ===
csv_path = os.path.join("data", csv_file_name)
try:
    df_raw = pd.read_csv(csv_path)
    print(f"Successfully loaded {len(df_raw)} records from {csv_path}")
    
    # Filter to get only the 'crop' data points
    df_crop = df_raw[df_raw['label'] == 'crop'].copy()
    print(f"Using {len(df_crop)} points for training the anomaly detector.")

    geometry = [Point(xy) for xy in zip(df_crop.longitude, df_crop.latitude)]
    gdf = gpd.GeoDataFrame(df_crop, geometry=geometry, crs="EPSG:4326")
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_path}. Please ensure it's in the 'data' folder.")
    exit()

# === STEP 2: Extract Spectral Features from Earth Engine ===
print(f"\nExtracting spectral features for {len(gdf)} crop points from Earth Engine...")
df_features = extract_spectral_features(gdf, training_start_date, training_end_date, spectral_bands)

if df_features.empty:
    print("No features were extracted for crop points. Exiting.")
    exit()

# === STEP 3: Feature Engineering - Calculate Vegetation Indices ===
print("Calculating vegetation indices...")
df_features = calculate_vegetation_indices(df_features)

final_features = spectral_bands + ['NDVI', 'EVI', 'NDWI', 'NDRE', 'GNDVI', 'MSI']
missing_features = [f for f in final_features if f not in df_features.columns]
if missing_features:
    print(f"Warning: The following features are missing and will be excluded: {missing_features}")
    final_features = [f for f in final_features if f in df_features.columns]
    if not final_features:
        print("Error: No valid features to train the model. Exiting.")
        exit()

# === STEP 4: Train the Anomaly Detection Model ===
print("\nTraining the Isolation Forest anomaly detection model...")
# The contamination parameter is the expected proportion of outliers in the data.
# A small value (e.g., 0.01) is a good starting point assuming weeds are rare.
# You can tune this parameter later based on your field observations.
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(df_features[final_features])

print("Model training complete.")

# === STEP 5: Save the Model and Features ===
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, model_name)
joblib.dump(model, model_path)
print(f"\nAnomaly detection model saved to {model_path}")

features_path = os.path.join(model_dir, "anomaly_features.json")
with open(features_path, "w") as f:
    json.dump(final_features, f)
print(f"Feature names saved to {features_path}")