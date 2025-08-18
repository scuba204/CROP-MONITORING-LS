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

    points_fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([p.x, p.y]), {'id': int(i)}) for i, p in zip(gdf['id'], gdf.geometry)
    ])

    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))
        .select(bands)
        .sort('system:time_start')
    )
    
    def map_over_points(feature):
        point = feature.geometry()
        
        def reduce_image(image):
            date_millis = image.get('system:time_start')
            values = image.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=scale
            )
            return ee.Feature(None, values.set('date', date_millis).set('id', feature.get('id')))

        return s2_collection.map(reduce_image)

    feature_collection = points_fc.map(map_over_points).flatten()
    
    try:
        data = feature_collection.getInfo()['features']
    except Exception as e:
        print(f"Earth Engine API Error: {e}")
        return pd.DataFrame()

    if not data:
        print("No features extracted.")
        return pd.DataFrame()

    df = pd.json_normalize(data)
    
    # --- CORRECTED CODE ---
    # Build a list of all expected columns.
    expected_cols = ['properties.id', 'properties.date'] + [f'properties.{b}' for b in bands]
    
    # Select only the columns you expect, which prevents issues if a column is missing.
    df = df[expected_cols]
    
    # Create the renaming dictionary for all columns.
    rename_dict = {'properties.id': 'id', 'properties.date': 'date'}
    rename_dict.update({f'properties.{b}': b for b in bands})
    
    # Rename the columns in a single, safe operation.
    df = df.rename(columns=rename_dict)
    
    # Ensure id is an integer.
    df['id'] = df['id'].astype(int)
    df['date'] = pd.to_datetime(df['date'], unit='ms')

    print(f"Extracted features for {len(df['id'].unique())} points on {len(df['date'].unique())} unique dates.")
    return df