#!/usr/bin/env python3
"""
predict.py:
A script to perform crop vs. weed classification on unlabelled field data
using a pre-trained RandomForestClassifier model. It extracts spectral features
from Google Earth Engine, engineers vegetation indices and delta features,
and outputs predictions to a CSV file.
"""

import sys
import os
import argparse
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import joblib
import ee

# ==============================================================================
# Step 0: Initial Configuration and Path Setup
# ==============================================================================

# Ensure the project root is in the Python path for module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.extract_features import extract_spectral_features

# --- Initialize Earth Engine ---
try:
    ee.Authenticate()
    ee.Initialize(project="winged-tenure-464005-p9")
    print("Earth Engine initialized successfully.")
except Exception as e:
    print(f"Error initializing Earth Engine: {e}")
    print("Please ensure you have authenticated with 'earthengine authenticate'")
    sys.exit(1)


def parse_args():
    """Parses command-line arguments for the prediction script."""
    parser = argparse.ArgumentParser(
        description="Make predictions on unlabelled field data using a trained crop classifier."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default="unlabelled_field_data.csv",
        help="Name of the input CSV file in the 'data' directory."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="crop_vs_weed_model.pkl",
        help="Filename of the crop classifier model in the 'models' directory."
    )
    parser.add_argument(
        "--features-file",
        type=str,
        default="feature_list.json",
        help="Filename of the JSON file containing the list of features the model expects."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="predictions_output.csv",
        help="Filename for the output CSV file containing the predictions."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-07-01",
        help="Start date for Sentinel-2 imagery (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-08-05", # Updated end date to match the error log
        help="End date for Sentinel-2 imagery (YYYY-MM-DD)."
    )
    return parser.parse_args()


# ==============================================================================
# Helper Functions for Feature Engineering
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
    # ADDED MISSING INDICES
    df['OSAVI'] = (df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 0.16) * (1 + 0.16)
    df['SAVI'] = ((df['B8'] - df['B4']) / (df['B8'] + df['B4'] + 0.5)) * (1.5)
    df['RVI'] = df['B8'] / df['B4']
    
    # Handle division by zero
    df.replace([float('inf'), float('-inf')], np.nan, inplace=True)
    df.fillna(0, inplace=True)
    
    return df

def calculate_delta_features(df, features_from_model):
    """
    Calculates delta features (change over time) for each vegetation index
    if more than one date of imagery is available.
    
    Args:
        df (pd.DataFrame): DataFrame containing raw features and indices.
        features_from_model (list): The list of feature names the model expects.

    Returns:
        pd.DataFrame: A pivoted DataFrame with a complete set of features,
                      including placeholders for missing dates/deltas.
    """
    unique_dates = sorted(df['date'].unique())
    
    if len(unique_dates) < 2:
        print("⚠️ Only one date of imagery found. Creating single-date feature set.")
        
        # Get the latest date available
        latest_date_str = unique_dates[0].strftime("%Y%m%d")
        
        # Create a dictionary to hold the features for the single date
        data_dict = {}
        for feature_name in features_from_model:
            # Check if the feature name corresponds to a single-date feature
            if latest_date_str in feature_name and '_Δ_' not in feature_name:
                base_metric = feature_name.split('_')[0]
                # Populate the dictionary with the calculated features
                data_dict[feature_name] = df.loc[df['date'] == unique_dates[0], base_metric].values
            else:
                # For all other features (earlier dates or deltas), create a placeholder of zeros
                data_dict[feature_name] = np.zeros(len(df[df['date'] == unique_dates[0]]))
        
        # Create a new DataFrame from the dictionary to ensure all columns exist
        df_pivoted = pd.DataFrame(data_dict)
        # Add the 'id' column back for merging later
        df_pivoted['id'] = df[df['date'] == unique_dates[0]]['id'].values
        df_pivoted.set_index('id', inplace=True)
        
    else:
        print(f"✅ {len(unique_dates)} dates of imagery found. Calculating delta features.")
        # Pivot the data to create columns for each date
        df_pivoted = df.pivot_table(index='id', columns='date', values=['NDVI', 'EVI', 'NDWI', 'NDRE', 'GNDVI', 'MSI', 'OSAVI', 'SAVI', 'RVI'])
        df_pivoted.columns = [f'{col[0]}_{col[1].strftime("%Y%m%d")}' for col in df_pivoted.columns]

        # Calculate delta features between the first and last dates
        first_date_str = unique_dates[0].strftime("%Y%m%d")
        last_date_str = unique_dates[-1].strftime("%Y%m%d")
        
        delta_metrics = ['NDVI', 'EVI', 'NDWI', 'NDRE', 'GNDVI', 'MSI', 'OSAVI', 'SAVI', 'RVI']
        for metric in delta_metrics:
            df_pivoted[f'{metric}_Δ_{last_date_str}'] = df_pivoted[f'{metric}_{last_date_str}'] - df_pivoted[f'{metric}_{first_date_str}']
    
    return df_pivoted
def main():
    """Main function to run the prediction pipeline."""
    args = parse_args()
    
    model_dir = "models"
    data_dir = "data"

    # Define the core spectral bands needed for the model
    spectral_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

    # === STEP 1: Load Pre-trained Classifier Model and Features ===
    try:
        model_path = os.path.join(model_dir, args.model_name)
        model = joblib.load(model_path)
        print(f"✅ Successfully loaded crop classifier model from {model_path}")

        features_path = os.path.join(model_dir, args.features_file)
        with open(features_path, 'r') as f:
            features = json.load(f)
        print(f"✅ Loaded {len(features)} feature names.")

    except FileNotFoundError as e:
        print(f"❌ Error: Model or features file not found. Please ensure you have run a training script first.")
        print(e)
        sys.exit(1)

    # === STEP 2: Load Unlabeled Field Data ===
    csv_path = os.path.join(data_dir, args.input_csv)
    try:
        df_raw = pd.read_csv(csv_path)
        print(f"✅ Successfully loaded {len(df_raw)} records from {csv_path}")

        if 'longitude' not in df_raw.columns or 'latitude' not in df_raw.columns:
            print("❌ Error: The input CSV must contain 'longitude' and 'latitude' columns.")
            sys.exit(1)

        df_raw['id'] = range(len(df_raw))
        geometry = [Point(xy) for xy in zip(df_raw.longitude, df_raw.latitude)]
        gdf = gpd.GeoDataFrame(df_raw, geometry=geometry, crs="EPSG:4326")
        gdf['id'] = range
    except FileNotFoundError:
        print(f"❌ Error: CSV file not found at {csv_path}. Please ensure it's in the 'data' folder.")
        sys.exit(1)

    # === STEP 3: Extract RAW Spectral Features from Earth Engine ===
    print(f"\n🌍 Extracting raw spectral features from Earth Engine for dates {args.start_date} to {args.end_date}...")
    df_features = extract_spectral_features(
        gdf,
        args.start_date,
        args.end_date,
        spectral_bands
    )
    df_features['id'] = df_features['id'].astype(int)

    if df_features.empty:
        print("❌ No features were extracted from Earth Engine. Exiting.")
        sys.exit(1)
    
    # NEW LOGIC: Check if 'date' column exists and add it if it doesn't
    if 'date' not in df_features.columns:
        print("⚠️ 'date' column not found, adding placeholder date.")
        # If there's only one date, the column won't be created. We add it manually.
        df_features['date'] = pd.to_datetime(args.end_date)
    
    # === STEP 4: Feature Engineering ===
    print("📈 Calculating vegetation indices...")
    df_features = calculate_vegetation_indices(df_features)

    print("📈 Engineering delta features...")
    df_features = calculate_delta_features(df_features, features)

    # Add this code block
    print("\n--- Data Inspection ---")
    print(f"Number of rows in feature DataFrame: {len(df_features)}")
    print(f"Number of columns in feature DataFrame: {len(df_features.columns)}")
    print("First 5 rows of the feature DataFrame:")
    print(df_features.head())
    print("\nDescriptive statistics for the feature DataFrame:")
    print(df_features.describe())
    print("-----------------------")

    # Ensure the features are in the same order as the trained model
    X_to_predict = df_features[features]

    # === STEP 5: Make Predictions with the Classifier Model ===
    print("\n🔮 Making predictions using the crop classifier model...")
    
    class_mapping = {label: name for label, name in enumerate(model.classes_)}
    
    predictions = model.predict(X_to_predict)
    df_features['predicted_label'] = pd.Series(predictions).map(class_mapping)

    # === STEP 6: Save Predictions to CSV ===
    os.makedirs(data_dir, exist_ok=True)
    
    # The output df has changed due to pivoting. Use the original df for location info.
    output_df = df_raw[['longitude', 'latitude', 'id']].merge(df_features[['predicted_label']], on='id')
    output_df = output_df[['longitude', 'latitude', 'predicted_label']]
    
    output_path = os.path.join(data_dir, args.output_csv)
    output_df.to_csv(output_path, index=False)

    print(f"\n✅ Predictions saved to {output_path}")
    print("Prediction Summary:")
    print(output_df['predicted_label'].value_counts())
    print("\n✅ Script finished successfully.")


if __name__ == "__main__":
    main()