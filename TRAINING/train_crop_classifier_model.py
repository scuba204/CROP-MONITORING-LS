import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import json
import os # Import os for path handling

# Import the feature extraction function from your new module
from scripts.extract_features import extract_spectral_features # [cite: 1]

# --- Configuration ---
csv_file_name = "crop_weed_training.csv"
model_dir = "models"
# Define a mapping for your labels to numerical values
label_mapping = {'crop': 0, 'weed': 1}

# Define the bands you need for your model (Sentinel-2)
spectral_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"] # [cite: 1]

# Adjust these dates to match your ground truth data collection period,
# or a period when crops/weeds are spectrally distinguishable.
training_start_date = "2025-07-17" # Example start date
training_end_date = "2025-07-27"   # Example end date

# === STEP 1: Load Labeled Field Data and Structure for GEE ===

# Use os.path.join for better cross-OS compatibility
csv_path = os.path.join("data", csv_file_name)

try:
    df_raw = pd.read_csv(csv_path)
    print(f"Successfully loaded {len(df_raw)} records from {csv_path}")
    print("Initial DataFrame head:")
    print(df_raw.head())
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_path}. Please ensure it's in the 'data' folder.")
    exit() # Exit if the file isn't found

# Convert to GeoDataFrame
# Use 'longitude' and 'latitude' as they are standard and were in your provided CSV data.
if 'longitude' in df_raw.columns and 'latitude' in df_raw.columns:
    geometry = [Point(xy) for xy in zip(df_raw.longitude, df_raw.latitude)]
    gdf = gpd.GeoDataFrame(df_raw, geometry=geometry, crs="EPSG:4326")
    print("\nGeoDataFrame head (with geometry column):")
    print(gdf.head())
else:
    print("Error: Expected 'longitude' and 'latitude' columns not found in the CSV.")
    exit()

# Encode Text Labels to Numerical Labels (e.g., 'crop' -> 0, 'weed' -> 1)
if 'label' in gdf.columns:
    gdf['label'] = gdf['label'].map(label_mapping)
    if gdf['label'].isnull().any():
        unmapped_labels = df_raw[gdf['label'].isnull()]['label'].unique()
        print(f"\nWarning: Some labels were not mapped to numerical values: {unmapped_labels}")
        print("Please check your label_mapping or CSV data for consistency.")
    print("\nLabels converted to numerical format:")
    print(gdf['label'].value_counts())
else:
    print("Error: 'label' column not found in the GeoDataFrame after initial load.")
    exit()

# === STEP 2: Extract Features from Earth Engine ===
print(f"\nExtracting {len(spectral_bands)} spectral features for {len(gdf)} points from Earth Engine...")
print(f"Using date range: {training_start_date} to {training_end_date}")

# Call the extract_spectral_features function
df_features = extract_spectral_features(gdf, training_start_date, training_end_date, spectral_bands) # [cite: 1]

# --- Post-extraction checks ---
if df_features.empty:
    print("No features were extracted from Earth Engine. Exiting.")
    exit()

# Ensure the 'label' column is present in the extracted DataFrame (should be carried over)
if 'label' not in df_features.columns:
    print("Error: 'label' column missing after feature extraction. This is critical for training.")
    exit()

# Check if all expected bands were actually extracted
missing_extracted_bands = [b for b in spectral_bands if b not in df_features.columns]
if missing_extracted_bands:
    print(f"Warning: The following bands were not found in the extracted data: {missing_extracted_bands}")
    # Update spectral_bands list to only include available bands, to avoid KeyError
    spectral_bands = [b for b in spectral_bands if b in df_features.columns]
    if not spectral_bands: # If no bands were extracted at all
        print("Error: No spectral bands could be extracted. Cannot proceed with training.")
        exit()
    print(f"Proceeding with available bands: {spectral_bands}")


print("\nExtracted features DataFrame head:")
print(df_features.head())

# === STEP 3: Prepare Features (X) and Labels (y) for ML ===
# Now, X and y are derived from the DataFrame that includes GEE-extracted features.
X = df_features[spectral_bands] # Use the bands that were actually extracted from GEE
y = df_features["label"]

# === STEP 4: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === STEP 5: Train Classifier ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === STEP 6: Evaluate Model ===
y_pred = clf.predict(X_test)
print("\n=== Model Evaluation Report ===")
print(classification_report(y_test, y_pred, target_names=["Crop", "Weed"]))

# === STEP 7: Save Model ===
os.makedirs(model_dir, exist_ok=True) # Ensure 'models' directory exists

model_path = os.path.join(model_dir, "crop_classifier_model.pkl")
joblib.dump(clf, model_path)
print(f"\nModel saved to {model_path}")

# Save feature names used for training (crucial for prediction consistency)
features_path = os.path.join(model_dir, "crop_classifier_features.json")
with open(features_path, "w") as f:
    json.dump(list(X.columns), f)
print(f"Feature names saved to {features_path}")