# In your main app script or a new feature_extractor.py
import ee
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
# Assuming _mask_s2_clouds is available, perhaps from gee_functions.py
from scripts.gee_functions import _mask_s2_clouds # Adjust import path as needed

def extract_spectral_features(gdf: gpd.GeoDataFrame,
                              start_date: str,
                              end_date: str,
                              bands: list) -> pd.DataFrame:
    """
    Extracts spectral band values from Sentinel-2 for a given GeoDataFrame of points
    within a specified date range.
    """
    # GEE Initialization:
    # It's good practice to pass your project ID for billing/quota management.
    ee.Initialize(project="winged-tenure-464005-p9")

    # Convert GeoDataFrame to Earth Engine FeatureCollection
    # This loop correctly converts each row of your GeoDataFrame (properties + geometry)
    # into an ee.Feature, which is then collected into an ee.FeatureCollection.
    # The `p.drop('geometry').to_dict()` ensures all non-geometry columns (like 'label', 'latitude', 'longitude')
    # are passed as properties to the ee.Feature.
    features_ee = ee.FeatureCollection([
        ee.Feature(p.geometry.__geo_interface__, p.drop('geometry').to_dict())
        for idx, p in gdf.iterrows()
    ])

    # Define Sentinel-2 Image Collection
    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filterBounds(features_ee.geometry()) # Filters the collection to only images that overlap your ROI.

    # Apply cloud masking
    # Relies on the correctness of _mask_s2_clouds from gee_functions.py.
    s2_masked_collection = s2_collection.map(_mask_s2_clouds)

    # Check if there are images after filtering and masking
    # Excellent error handling! Prevents crashes if no valid imagery is found.
    if s2_masked_collection.size().getInfo() == 0:
        print(f"Warning: No Sentinel-2 images found for date range {start_date} to {end_date} after cloud masking.")
        # Returns an empty DataFrame with expected columns, which is a robust way to handle this.
        # Ensure 'id' is a column if you intend to return it, otherwise adjust.
        # Your previous `gdf` has 'latitude', 'longitude', 'label'. So, perhaps:
        # return pd.DataFrame(columns=['latitude', 'longitude', 'label'] + bands)
        # Or even better: create a new dataframe from gdf, add nan columns for bands.
        empty_df = gdf.copy()
        for band in bands:
            empty_df[band] = float('nan')
        return empty_df.drop(columns='geometry') # Drop geometry as we return a pandas df
        # The current return is fine if you're sure about 'id' presence or just using it as a template.

    # Reduce images to a single image (e.g., median) for feature extraction
    # Using .median() is generally robust for time-series composites as it's less sensitive to outliers
    # than .mean(). `.select(bands)` ensures only the desired bands are in the composite.
    composite_image = s2_masked_collection.median().select(bands)

    # Extract values at each point
    # `reduceRegions` is the correct and efficient way to extract values for multiple points.
    # `reducer=ee.Reducer.mean()`: This will calculate the mean pixel value within the specified scale
    # at each point. You could also use `ee.Reducer.median()` for consistency with the composite.
    # `scale=10`: This is crucial. Sentinel-2 has native resolutions of 10m (B2, B3, B4, B8)
    # and 20m (B5, B6, B7, B8A, B11, B12). By setting scale=10, all 20m bands will be
    # resampled to 10m before extraction. This is a common and acceptable practice for consistency,
    # but be aware that it involves resampling.
    extracted_features_ee = composite_image.reduceRegions(
        collection=features_ee,
        reducer=ee.Reducer.mean(),
        scale=10
    )

    # Convert to a client-side list of dictionaries and then to a Pandas DataFrame
    # .getInfo() fetches the data from GEE to your local machine.
    # This loop correctly extracts the 'properties' dictionary for each feature,
    # which should contain your original 'label', 'latitude', 'longitude', and the new band values.
    features_list = extracted_features_ee.getInfo()['features']
    extracted_df = pd.DataFrame([f['properties'] for f in features_list])

    # Ensure original 'label' column is carried over, and any point identifiers
    # The `extracted_df` will automatically include 'latitude', 'longitude', 'label'
    # if they were present in your original `gdf` and passed as properties during `ee.Feature` creation.
    # This function is well-designed to handle that.
    return extracted_df

# --- Integration into your train_crop_classifier.py (This part is outside the function) ---

# ... (Previous imports) ...

# === STEP 1: Load Labeled Field Data ===
# This block correctly loads your ground truth and converts it to a GeoDataFrame.
csv_path = "data/crop_weed_training.csv"
df_raw = pd.read_csv(csv_path)

# Correctly uses longitude, latitude for Point creation
geometry = [Point(xy) for xy in zip(df_raw.longitude, df_raw.latitude)]
gdf = gpd.GeoDataFrame(df_raw, geometry=geometry, crs="EPSG:4326")

# Correctly converts labels to numerical format
label_mapping = {'crop': 0, 'weed': 1}
gdf['label'] = gdf['label'].map(label_mapping)
if gdf['label'].isnull().any():
    unmapped_labels = df_raw[gdf['label'].isnull()]['label'].unique()
    print(f"\nWarning: Some labels were not mapped to numerical values: {unmapped_labels}")
    print("Please check your label_mapping or CSV data.")


# === New Step: Extract Features from Earth Engine ===
# This is where you call the function.
training_start_date = "2024-05-01" # Adjust this to your ground truth collection period
training_end_date = "2024-07-31"   # Adjust this to your ground truth collection period

# Define the bands you need for your model
spectral_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"]

print(f"Extracting {len(spectral_bands)} spectral features for {len(gdf)} points from Earth Engine...")
df_features = extract_spectral_features(gdf, training_start_date, training_end_date, spectral_bands)

# Ensure the 'label' column is present in the extracted DataFrame
if 'label' not in df_features.columns:
    print("Error: 'label' column missing after feature extraction. Check your GEE feature properties.")
    # Consider raising an error here instead of just printing and exiting for more robust error handling.
    raise ValueError("Missing 'label' column after GEE feature extraction.")
# Also, check if all expected bands were extracted (e.g., if a band wasn't available)
missing_extracted_bands = [b for b in spectral_bands if b not in df_features.columns]
if missing_extracted_bands:
    print(f"Warning: The following bands were not found in the extracted data: {missing_extracted_bands}")
    # You might want to remove these from spectral_bands or handle them (e.g., fill with NaN)
    # For now, the script will proceed and X = df_features[spectral_bands] will error if a band is truly missing.
    spectral_bands = [b for b in spectral_bands if b in df_features.columns] # Adapt bands list


# === STEP 2: Prepare Features and Labels for ML ===
X = df_features[spectral_bands] # Use the bands extracted from GEE
y = df_features["label"]

# ... (Continue with Step 3: Train-Test Split, Step 4: Train Classifier, etc.) ...