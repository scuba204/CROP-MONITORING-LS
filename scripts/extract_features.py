import os
import sys
import ee
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.gee_functions import _mask_s2_clouds



def extract_spectral_features(gdf, start_date, end_date, bands, scale=10, cloud_filter=10):
    """
    Extracts spectral features from Sentinel-2 imagery for a given set of points
    and date range.
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
        .sort('system:time_start')
    )
    
    # --- CORRECTED CODE ---
    def map_over_images(image):
        date_millis = image.get('system:time_start')
        
        # A function to reduce a single point's data for the current image.
        def reduce_region_function(feature):
            point = feature.geometry()
            values = image.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=scale
            ).rename(bands, [b for b in bands])
            return feature.set(values).set('date', date_millis)

        return points_fc.map(reduce_region_function)

    # Map the function over the image collection and flatten the result.
    feature_collection = s2_collection.map(map_over_images).flatten()
    
    # Get the results from Earth Engine in a single request.
    try:
        data = feature_collection.getInfo()['features']
    except Exception as e:
        print(f"Earth Engine API Error: {e}")
        return pd.DataFrame()

    if not data:
        print("No features extracted.")
        return pd.DataFrame()

    df = pd.json_normalize(data)
    
    # Clean up the DataFrame
    df = df.rename(columns={f'properties.{b}': b for b in bands})
    df = df.rename(columns={'properties.id': 'id'})
    df['date'] = pd.to_datetime(df['properties.date'], unit='ms')
    
    properties_cols_to_drop = [col for col in df.columns if col.startswith('properties.') and col not in ['properties.id', 'properties.date']]
    df = df.drop(columns=properties_cols_to_drop)

    print(f"Extracted features for {len(df['id'].unique())} points on {len(df['date'].unique())} unique dates.")
    return df