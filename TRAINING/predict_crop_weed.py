import sys
import os
import ee
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
import joblib
import json

# ==============================================================================
# Step 0: Initial Configuration and Path Setup
# ==============================================================================

# Get the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the feature extraction function from your scripts directory
from scripts.extract_features import extract_spectral_features

# --- Configuration ---
# The path to your unlabelled data CSV file
new_data_csv = "data/unlabelled_field_data.csv"
model_dir = "models"

# Define the date range for the satellite imagery for your new data
# Make sure this date range is relevant to the new data's collection period.
prediction_start_date = "2025-07-27" # Example date
prediction_end_date = "2025-08-01"   # Example date

# ==============================================================================
# Helper Function for Feature Engineering (same as in training script)
# ==============================================================================

def calculate_vegetation_indices(df):
    """
    Calculates various vegetation indices from Sentinel-2 band data.
    The indices are added as new columns to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing Sentinel-2 band data (B3, B4, B8, B5, B11).

    Returns:
        pd.DataFrame: The original DataFrame with new vegetation index columns.
    """
    # --- Existing Indices ---
    # Normalized Difference Vegetation Index (NDVI)
    df['NDVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'])
    
    # Enhanced Vegetation Index (EVI)
    # C1=6, C2=7.5, L=1 based on Sentinel-2 recommended constants
    df['EVI'] = 2.5 * (df['B8'] - df['B4']) / (df['B8'] + 6 * df['B4'] - 7.5 * df['B3'] + 1)
    
    # Normalized Difference Water Index (NDWI)
    df['NDWI'] = (df['B8'] - df['B11']) / (df['B8'] + df['B11'])

    # --- New Indices ---
    # Normalized Difference Red Edge Index (NDRE)
    # Uses NIR (B8) and a Red Edge band (B5)
    # Formula: (NIR - RedEdge) / (NIR + RedEdge)
    df['NDRE'] = (df['B8'] - df['B5']) / (df['B8'] + df['B5'])

    # Green Normalized Difference Vegetation Index (GNDVI)
    # Uses NIR (B8) and the Green band (B3)
    # Formula: (NIR - Green) / (NIR + Green)
    df['GNDVI'] = (df['B8'] - df['B3']) / (df['B8'] + df['B3'])
    
    # Moisture Stress Index (MSI)
    # Uses SWIR (B11) and NIR (B8)
    # Formula: SWIR / NIR
    df['MSI'] = df['B11'] / df['B8']

    # Handle potential division by zero
    df.replace([float('inf'), float('-inf')], 0, inplace=True)
    
    return df
# ==============================================================================
# Main Prediction Pipeline
# ==============================================================================

# === STEP 1: Load the Trained Model and Feature Names ===
model_path = os.path.join(model_dir, "crop_classifier_model.pkl")
features_path = os.path.join(model_dir, "crop_classifier_features.json")

try:
    print("Loading trained model and features...")
    clf = joblib.load(model_path)
    with open(features_path, "r") as f:
        feature_names = json.load(f)
    print(f"Model loaded successfully. The model was trained on {len(feature_names)} features.")
except FileNotFoundError:
    print(f"Error: Model or feature file not found in '{model_dir}'. Please run the training script first.")
    exit()

# === STEP 2: Load New Unlabelled Data ===
try:
    df_raw = pd.read_csv(new_data_csv)
    print(f"\nSuccessfully loaded {len(df_raw)} unlabelled records from {new_data_csv}")
    print("Unlabelled DataFrame head:")
    print(df_raw.head())
except FileNotFoundError:
    print(f"Error: New data CSV file not found at '{new_data_csv}'.")
    exit()

# Convert to GeoDataFrame
if 'longitude' in df_raw.columns and 'latitude' in df_raw.columns:
    geometry = [Point(xy) for xy in zip(df_raw.longitude, df_raw.latitude)]
    gdf_unlabelled = gpd.GeoDataFrame(df_raw, geometry=geometry, crs="EPSG:4326")
else:
    print("Error: Expected 'longitude' and 'latitude' columns not found in the new data CSV.")
    exit()

# === STEP 3: Extract Spectral Features for New Data ===
# The bands list must be the same as the one used for training
spectral_bands_for_extraction = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
print(f"\nExtracting {len(spectral_bands_for_extraction)} spectral features for {len(gdf_unlabelled)} unlabelled points from Earth Engine...")
df_new_features = extract_spectral_features(gdf_unlabelled, prediction_start_date, prediction_end_date, spectral_bands_for_extraction)

if df_new_features.empty:
    print("No features were extracted for the new data. Exiting.")
    exit()

# === STEP 4: Feature Engineering - Calculate Vegetation Indices for New Data ===
print("Calculating vegetation indices for new data...")
df_new_features = calculate_vegetation_indices(df_new_features)

# === STEP 5: Make Predictions ===
# Ensure the new data has the exact same features as the training data, in the same order
X_predict = df_new_features[feature_names]

print("\nMaking predictions on the new data...")
predictions = clf.predict(X_predict)

# === STEP 6: Display Results ===
# Add the predictions back to the original DataFrame
df_raw['predicted_label'] = predictions

# Map numerical predictions back to original labels for readability
inverse_label_mapping = {0: 'crop', 1: 'weed'}
df_raw['predicted_label'] = df_raw['predicted_label'].map(inverse_label_mapping)

print("\n=== Predictions Complete ===")
print("New data with predicted labels:")
print(df_raw.head())

# Optional: Save the results to a new CSV file
output_path = os.path.join("data", "predictions_output.csv")
df_raw.to_csv(output_path, index=False)
print(f"\nPredictions saved to {output_path}")