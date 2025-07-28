# In your main app script or a new feature_extractor.py
import ee
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
# Assuming _mask_s2_clouds is available, perhaps from gee_functions.py
from .gee_functions import _mask_s2_clouds # Adjust import path as needed

def extract_spectral_features(gdf: gpd.GeoDataFrame,
                              start_date: str,
                              end_date: str,
                              bands: list) -> pd.DataFrame:
    """
    Extracts spectral band values from Sentinel-2 for a given GeoDataFrame of points
    within a specified date range.
    """
    ee.Initialize(project="winged-tenure-464005-p9") # Ensure GEE is initialized

    # Convert GeoDataFrame to Earth Engine FeatureCollection
    # Ensure your GDF has a unique ID column, e.g., 'id' if not already present
    # For now, let's assume the original index can serve as an ID if no other exists.
    # It's better to explicitly add one if not guaranteed unique and present.
    features_ee = ee.FeatureCollection([
        ee.Feature(p.geometry.__geo_interface__, p.drop('geometry').to_dict())
        for idx, p in gdf.iterrows()
    ])

    # Define Sentinel-2 Image Collection
    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filterBounds(features_ee.geometry()) # Filter by the extent of your points

    # Apply cloud masking
    s2_masked_collection = s2_collection.map(_mask_s2_clouds)

    # Check if there are images after filtering and masking
    if s2_masked_collection.size().getInfo() == 0:
        print(f"Warning: No Sentinel-2 images found for date range {start_date} to {end_date} after cloud masking.")
        # Return an empty DataFrame or handle as appropriate
        return pd.DataFrame(columns=['id', 'label'] + bands) # Example empty df

    # Reduce images to a single image (e.g., median) for feature extraction
    # Using .median() is often better than .mean() for outlier resistance
    # You might consider .mean() if you prefer.
    composite_image = s2_masked_collection.median().select(bands)

    # Extract values at each point
    # Properties from input features will be retained.
    # `scale` is crucial - use Sentinel-2's native resolution (e.g., 10m for visible/NIR, 20m for SWIR)
    extracted_features_ee = composite_image.reduceRegions(
        collection=features_ee,
        reducer=ee.Reducer.mean(), # Mean value if a point covers multiple pixels
        scale=10 # Sentinel-2's resolution for most bands (B2,B3,B4,B8). Adjust for SWIR (B11,B12) if needed or use composite scale.
                 # For consistent features, 10m is often chosen, resulting in resampling for 20m/60m bands.
    )

    # Convert to a client-side list of dictionaries and then to a Pandas DataFrame
    features_list = extracted_features_ee.getInfo()['features']
    extracted_df = pd.DataFrame([f['properties'] for f in features_list])

    # Ensure original 'label' column is carried over, and any point identifiers
    # It's good practice to merge back with the original GDF if you need all its columns.
    # For simplicity here, we assume 'label' is in the properties extracted.
    return extracted_df

# --- Integration into your train_crop_classifier.py ---

# ... (Previous imports) ...

# === STEP 1: Load Labeled Field Data ===
csv_path = "data/crop_weed_training.csv"
df_raw = pd.read_csv(csv_path)

geometry = [Point(xy) for xy in zip(df_raw.x, df_raw.y)] # Or df_raw.longitude, df_raw.latitude
gdf = gpd.GeoDataFrame(df_raw, geometry=geometry, crs="EPSG:4326")

# === New Step: Extract Features from Earth Engine ===
training_start_date = "2024-05-01" # Adjust this to your ground truth collection period
training_end_date = "2024-07-31"   # Adjust this to your ground truth collection period

# Define the bands you need for your model
spectral_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"]

print(f"Extracting {len(spectral_bands)} spectral features for {len(gdf)} points from Earth Engine...")
# The returned `df_features` will contain your original columns (e.g., 'label')
# plus the new spectral band columns (e.g., 'B2', 'B3', etc.)
df_features = extract_spectral_features(gdf, training_start_date, training_end_date, spectral_bands)

# Ensure the 'label' column is present in the extracted DataFrame
if 'label' not in df_features.columns:
    print("Error: 'label' column missing after feature extraction. Check your GEE feature properties.")
    exit()

# === STEP 2: Prepare Features and Labels for ML ===
# Now, X and y are derived from the DataFrame that includes GEE-extracted features.
X = df_features[spectral_bands] # Use the bands extracted from GEE
y = df_features["label"]

# ... (Continue with Step 3: Train-Test Split, Step 4: Train Classifier, etc.) ...