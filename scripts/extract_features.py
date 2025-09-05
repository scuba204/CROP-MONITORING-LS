import os
import sys
import ee
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
# Assuming _mask_s2_clouds is available, perhaps from gee_functions.py

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from  scripts.gee_functions import _mask_s2_clouds # Adjust import path as needed

def extract_spectral_features(gdf, start_date, end_date, bands, scale=10, cloud_filter=10):
    """
    Extracts spectral features from Sentinel-2 imagery for a given set of points
    and date range.

    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame with point geometries and unique IDs.
        start_date (str): The start date for the image collection filter (YYYY-MM-DD).
        end_date (str): The end date for the image collection filter (YYYY-MM-DD).
        bands (list): List of Sentinel-2 band names to extract.
        scale (int): The scale in meters for the image data to be sampled at.
        cloud_filter (int): The maximum percentage of cloud cover allowed.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted band values for each point
                      and each available date.
    """
    print("Connecting to Google Earth Engine...")
    try:
        ee.Initialize(project="winged-tenure-464005-p9")
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        return pd.DataFrame()

    points_fc = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point([p.x, p.y]), {'id': i}) for i, p in zip(gdf['id'], gdf.geometry)]
    )

    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))
        .select(bands)
        .map(lambda image: image.addBands(image.metadata('system:time_start')))
        .sort('system:time_start')
    )

    # --- CORRECTION STARTS HERE ---
    # Retrieve a list of all images in the collection
    image_list = s2_collection.toList(s2_collection.size())
    num_images = image_list.size().getInfo()
    print(f"Found {num_images} images in the collection.")

    if num_images == 0:
        return pd.DataFrame()

    results = []
    
    # Process each image in the list
    for i in range(num_images):
        image = ee.Image(image_list.get(i))
        date_millis = image.get('system:time_start').getInfo()
        date_str = pd.to_datetime(date_millis, unit='ms').strftime('%Y-%m-%d')
        
        # Define a reducer to get a dictionary of values for each point
        def reduce_region_function(feature):
            point = feature.geometry()
            # Use a try-except to handle cases where a point might not be within an image.
            try:
                values = image.reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=point,
                    scale=scale
                ).getInfo()
                # Check for None values and replace with a placeholder (e.g., -9999)
                if values:
                    values = {key: values[key] if values[key] is not None else -9999 for key in values}
                else:
                    values = {band: -9999 for band in bands}
            except ee.EEException:
                values = {band: -9999 for band in bands}
            
            return feature.set(values)

        # Map the reduction function over the points
        extracted_features = points_fc.map(reduce_region_function)
        
        # Convert the results to a DataFrame
        data = extracted_features.getInfo()['features']
        df = pd.json_normalize(data)
        
        # Clean up the DataFrame
        df = df.rename(columns={f'properties.{b}': b for b in bands})
        df = df.rename(columns={'properties.id': 'id'})
        df = df.drop(columns=['type', 'geometry.type', 'geometry.coordinates'])
        
        # Add the date column and append to results
        df['date'] = pd.to_datetime(date_str)
        results.append(df)
    
    # Concatenate all results into a single DataFrame
    final_df = pd.concat(results, ignore_index=True)
    # --- CORRECTION ENDS HERE ---

    print(f"Extracted features for {len(final_df['id'].unique())} points on {len(final_df['date'].unique())} unique dates.")
    return final_df

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
training_start_date = "2025-07-26" # Adjust this to your ground truth collection period
training_end_date = "2025-08-10"   # Adjust this to your ground truth collection period

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