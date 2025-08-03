import sys
import os
import ee
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import joblib
import json
import numpy as np

# ==============================================================================
# Step 0: Initial Configuration and Path Setup
# ==============================================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.extract_features import extract_spectral_features

# --- Configuration ---
field_data_csv = "unlabelled_field_data.csv"
model_dir = "models"
model_name = "crop_anomaly_detector.pkl"  # Change to the anomaly detector model
features_file = "anomaly_features.json"
predictions_output_csv = "predictions_output.csv"

# Define the date range for your prediction data
prediction_start_date = "2025-07-26"
prediction_end_date = "2025-08-02"

# Define the core spectral bands you need for your model
spectral_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

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
# Main Prediction Pipeline
# ==============================================================================

# === STEP 1: Load Pre-trained Anomaly Detection Model and Features ===
try:
    model_path = os.path.join(model_dir, model_name)
    model = joblib.load(model_path)
    print(f"Successfully loaded anomaly detection model from {model_path}")

    features_path = os.path.join(model_dir, features_file)
    with open(features_path, 'r') as f:
        # Load the full list of features the model expects
        features = json.load(f)
    print(f"Loaded {len(features)} feature names.")

except FileNotFoundError as e:
    print(f"Error: Model or features file not found. Please ensure you have run 'train_anomaly_detector.py' first.")
    print(e)
    sys.exit(1)

# === STEP 2: Load Unlabeled Field Data ===
csv_path = os.path.join("data", field_data_csv)
try:
    df_raw = pd.read_csv(csv_path)
    print(f"Successfully loaded {len(df_raw)} records from {csv_path}")

    if 'longitude' not in df_raw.columns or 'latitude' not in df_raw.columns:
        print("Error: The CSV file must contain 'longitude' and 'latitude' columns.")
        sys.exit(1)

    geometry = [Point(xy) for xy in zip(df_raw.longitude, df_raw.latitude)]
    gdf = gpd.GeoDataFrame(df_raw, geometry=geometry, crs="EPSG:4326")
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_path}. Please ensure it's in the 'data' folder.")
    sys.exit(1)

# === STEP 3: Extract ONLY RAW Spectral Features from Earth Engine ===
print("\nExtracting RAW spectral features for unlabeled data from Earth Engine...")
df_features = extract_spectral_features(gdf, prediction_start_date, prediction_end_date, spectral_bands)

if df_features.empty:
    print("No features were extracted from Earth Engine. Exiting.")
    sys.exit(1)

# === STEP 4: Feature Engineering - Calculate Vegetation Indices ===
print("Calculating vegetation indices (NDVI, EVI, NDWI, NDRE, GNDVI, MSI)...")
df_features = calculate_vegetation_indices(df_features)

# Ensure the features are in the same order as the trained model
# The `features` variable loaded from the JSON file contains the correct list
# The `df_features` DataFrame now has all the required columns
X_to_predict = df_features[features]

# === STEP 5: Make Predictions with the Anomaly Detector ===
print("\nMaking predictions using the anomaly detection model...")

# The Isolation Forest model returns 1 for an inlier (normal) and -1 for an outlier (anomaly)
predictions = model.predict(X_to_predict)

# Map the numerical predictions to human-readable labels
# 1 -> inlier -> 'crop'
# -1 -> anomaly -> 'weed'
df_features['predicted_label'] = np.where(predictions == 1, 'crop', 'weed')

# === STEP 6: Save Predictions to CSV ===
# Create the output directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Select and save the relevant columns
output_df = df_features[['longitude', 'latitude', 'predicted_label']]
output_path = os.path.join("data", predictions_output_csv)
output_df.to_csv(output_path, index=False)

print(f"\nPredictions saved to {output_path}")
print("Prediction Summary:")
print(output_df['predicted_label'].value_counts())