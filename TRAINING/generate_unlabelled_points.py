import sys
import os
import ee
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ==============================================================================
# Step 0: Initial Configuration and Path Setup
# ==============================================================================

# Get the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Initialize the Earth Engine API
try:
    ee.Initialize(project="winged-tenure-464005-p9")
    print("Google Earth Engine initialized successfully.")
except Exception as e:
    print(f"Error initializing Google Earth Engine: {e}")
    print("Please ensure you have authenticated and initialized the Earth Engine API.")
    print("Run `earthengine authenticate` in your terminal and then try again.")
    sys.exit(1)

# --- Configuration ---
# Output file name for the generated points
output_csv_path = "data/unlabelled_field_data.csv"

# ==============================================================================
# Main Point Generation Script
# ==============================================================================

# === STEP 1: Define Your Area of Interest (AOI) ===
# The coordinates are a list of [longitude, latitude] pairs for the vertices.
# This uses a region in Lesotho(ROMA).
aoi_polygon = ee.Geometry.Polygon([
    [27.732592, -29.454724],
    [27.733199, -29.454641],
    [27.733412, -29.455111],
    [27.732618, -29.455123]
])

# For a real-world application, you would replace this with the coordinates of your field.
print(f"\nArea of interest defined: {aoi_polygon.getInfo()}")

# === STEP 2: Generate Points within the AOI ===
# You have a choice here:
#   a) Generate a large number of random points.
#   b) Generate a fixed-size grid of points.
#
# We'll use random points as it's a common and simple approach.
# You can adjust the `points` parameter to increase or decrease the number of points.
num_points = 5000  # Number of random points to generate
seed = 42          # A seed for reproducibility

print(f"Generating {num_points} random points within the AOI...")
points_feature_collection = ee.FeatureCollection.randomPoints(
    region=aoi_polygon,
    points=num_points,
    seed=seed
)

# === STEP 3: Convert the GEE FeatureCollection to a Pandas DataFrame ===
# GEE API calls are server-side, so we need to get the data to the client.
points_list = points_feature_collection.getInfo()['features']

# Extract the longitude and latitude for each point
coordinates = [feature['geometry']['coordinates'] for feature in points_list]
longitudes = [coord[0] for coord in coordinates]
latitudes = [coord[1] for coord in coordinates]

# Create a Pandas DataFrame
df_unlabelled = pd.DataFrame({
    'longitude': longitudes,
    'latitude': latitudes
})

print(f"\nSuccessfully generated a DataFrame with {len(df_unlabelled)} points.")
print("Unlabelled points DataFrame head:")
print(df_unlabelled.head())

# === STEP 4: Save the DataFrame to a CSV file ===
os.makedirs("data", exist_ok=True)
df_unlabelled.to_csv(output_csv_path, index=False)
print(f"\nUnlabelled points saved to {output_csv_path}")