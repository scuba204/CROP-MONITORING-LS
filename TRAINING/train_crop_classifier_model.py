import sys
import os
import ee
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, balanced_accuracy_score
import joblib
import json

# ==============================================================================
# Step 0: Initial Configuration and Path Setup
# ==============================================================================

# Get the path of the directory containing the currently executed script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the script's directory (the project root)
project_root = os.path.dirname(script_dir)

# Add the project root to sys.path if it's not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the feature extraction function from your scripts directory
from scripts.extract_features import extract_spectral_features

# --- Configuration ---
csv_file_name = "crop_weed_training.csv"
model_dir = "models"
label_mapping = {'crop': 0, 'weed': 1}

# Define the core spectral bands you need for your model
spectral_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Define the date range for your training data
training_start_date = "2025-07-25"
training_end_date = "2025-08-01"

# ==============================================================================
# Helper Function for Feature Engineering
# ==============================================================================

def calculate_vegetation_indices(df):
    """
    Calculates various vegetation indices from Sentinel-2 band data.
    The indices are added as new columns to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing Sentinel-2 band data.

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
    df['NDRE'] = (df['B8'] - df['B5']) / (df['B8'] + df['B5'])

    # Green Normalized Difference Vegetation Index (GNDVI)
    df['GNDVI'] = (df['B8'] - df['B3']) / (df['B8'] + df['B3'])
    
    # Moisture Stress Index (MSI)
    df['MSI'] = df['B11'] / df['B8']

    # Handle potential division by zero
    df.replace([float('inf'), float('-inf')], 0, inplace=True)
    
    return df

# ==============================================================================
# Main Training Pipeline
# ==============================================================================

# === STEP 1: Load Labeled Field Data and Structure for GEE ===
csv_path = os.path.join("data", csv_file_name)

try:
    df_raw = pd.read_csv(csv_path)
    print(f"Successfully loaded {len(df_raw)} records from {csv_path}")
    print("Initial DataFrame head:")
    print(df_raw.head())
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_path}. Please ensure it's in the 'data' folder.")
    exit()

if 'longitude' in df_raw.columns and 'latitude' in df_raw.columns:
    geometry = [Point(xy) for xy in zip(df_raw.longitude, df_raw.latitude)]
    gdf = gpd.GeoDataFrame(df_raw, geometry=geometry, crs="EPSG:4326")
else:
    print("Error: Expected 'longitude' and 'latitude' columns not found in the CSV.")
    exit()

if 'label' in gdf.columns:
    gdf['label'] = gdf['label'].map(label_mapping)
else:
    print("Error: 'label' column not found in the GeoDataFrame after initial load.")
    exit()

# === STEP 2: Extract Spectral Features from Earth Engine ===
print(f"\nExtracting {len(spectral_bands)} spectral features for {len(gdf)} points from Earth Engine...")
df_features = extract_spectral_features(gdf, training_start_date, training_end_date, spectral_bands)

if df_features.empty:
    print("No features were extracted from Earth Engine. Exiting.")
    exit()

# === STEP 3: Feature Engineering - Calculate Vegetation Indices ===
print("Calculating vegetation indices (NDVI, EVI, NDWI, NDRE, GNDVI, MSI)...")
df_features = calculate_vegetation_indices(df_features)

# Define the final list of features for the model
final_features = spectral_bands + ['NDVI', 'EVI', 'NDWI', 'NDRE', 'GNDVI', 'MSI']

# Ensure all final features are present after calculations
missing_features = [f for f in final_features if f not in df_features.columns]
if missing_features:
    print(f"Warning: The following features are missing and will be excluded: {missing_features}")
    final_features = [f for f in final_features if f in df_features.columns]
    if not final_features:
        print("Error: No valid features to train the model. Exiting.")
        exit()

print(f"Using a total of {len(final_features)} features for training.")
print("\nFinal features DataFrame head:")
print(df_features[final_features + ['label']].head())

# === STEP 4: Prepare Features (X) and Labels (y) for ML ===
X = df_features[final_features]
y = df_features["label"]

# === STEP 5: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === STEP 6: Hyperparameter Tuning with GridSearchCV (UPDATED SECTION) ===
print("\nStarting hyperparameter tuning for RandomForestClassifier...")

# Expanded the parameter grid to search a wider range of values
param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a GridSearchCV object with balanced accuracy as the scoring metric
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5, # 5-fold cross-validation
    scoring='balanced_accuracy', # Use balanced accuracy to handle class imbalance
    n_jobs=-1, # Use all available CPU cores
    verbose=2 # Verbosity level
)

# Run the grid search to find the best model
grid_search.fit(X_train, y_train)

# Get the best model from the search
best_clf = grid_search.best_estimator_
print(f"\nBest hyperparameters found: {grid_search.best_params_}")
print(f"Best balanced accuracy score from cross-validation: {grid_search.best_score_:.4f}")

# === STEP 7: Evaluate Best Model ===
y_pred = best_clf.predict(X_test)
print("\n=== Model Evaluation Report ===")
print(f"Balanced Accuracy Score on Test Set: {balanced_accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["Crop", "Weed"]))

# === STEP 8: Save Best Model and Features ===
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "crop_classifier_model.pkl")
joblib.dump(best_clf, model_path)
print(f"\nBest model saved to {model_path}")

features_path = os.path.join(model_dir, "crop_classifier_features.json")
with open(features_path, "w") as f:
    json.dump(list(X.columns), f)
print(f"Feature names saved to {features_path}")