import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

# --- Configuration ---
csv_file_name = r"data/crop_weed_training.csv" # Your uploaded file
# Define a mapping for your labels to numerical values
label_mapping = {'crop': 0, 'weed': 1}

# --- Step 1: Load Labeled Field Data into a Pandas DataFrame ---
try:
    df = pd.read_csv(csv_file_name)
    print(f"Successfully loaded {len(df)} records from {csv_file_name}")
    print("Initial DataFrame head:")
    print(df.head())
except FileNotFoundError:
    print(f"Error: The file '{csv_file_name}' was not found.")
    # Exit or handle the error appropriately
    exit()

# --- Step 2: Convert to GeoDataFrame ---

# Ensure expected coordinate columns exist
if 'latitude' in df.columns and 'longitude' in df.columns:
    # Create Shapely Point objects from latitude and longitude
    # Note: Point(longitude, latitude) is the correct order for Shapely
    geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
else:
    print("Error: 'latitude' and/or 'longitude' columns not found in the CSV.")
    exit()

# Create the GeoDataFrame
# Assign CRS 'EPSG:4326' (WGS84) which is standard for geographic coordinates
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
print("\nGeoDataFrame head (with geometry column):")
print(gdf.head())

# --- Step 3: Encode Text Labels to Numerical Labels ---
if 'label' in gdf.columns:
    # Apply the mapping to the 'label' column
    gdf['label'] = gdf['label'].map(label_mapping)
    # Check for any unmapped labels (e.g., if there's a typo in the CSV)
    if gdf['label'].isnull().any():
        unmapped_labels = df[gdf['label'].isnull()]['label'].unique()
        print(f"\nWarning: Some labels were not mapped to numerical values: {unmapped_labels}")
        print("Please check your label_mapping or CSV data.")
    print("\nLabels converted to numerical format:")
    print(gdf['label'].value_counts())
else:
    print("Error: 'label' column not found in the GeoDataFrame.")
    exit()


# Now, 'gdf' is your structured data, ready for feature extraction from Earth Engine.
# The 'label' column is also numerical.