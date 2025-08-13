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

    # The GeoDataFrame needs an 'id' column for the feature collection.
    # We will assume it is passed in the GeoDataFrame itself.
    points_fc = ee.FeatureCollection(
        [ee.Feature(ee.Geometry.Point([p.x, p.y]), {'id': i}) for i, p in zip(gdf['id'], gdf.geometry)]
    )

    s2_collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))
        .map(_mask_s2_clouds) # Correctly apply the cloud masking function
        .select(bands)
        .sort('system:time_start')
    )
    
    image_list = s2_collection.toList(s2_collection.size())
    num_images = image_list.size().getInfo()
    print(f"Found {num_images} images in the collection.")

    if num_images == 0:
        return pd.DataFrame()

    results = []
    
    for i in range(num_images):
        image = ee.Image(image_list.get(i))
        date_millis = image.get('system:time_start').getInfo()
        date_str = pd.to_datetime(date_millis, unit='ms').strftime('%Y-%m-%d')
        
        def reduce_region_function(feature):
            point = feature.geometry()
            try:
                values = image.reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=point,
                    scale=scale
                ).getInfo()
                if values:
                    values = {key: values[key] if values[key] is not None else -9999 for key in values}
                else:
                    values = {band: -9999 for band in bands}
            except ee.EEException:
                values = {band: -9999 for band in bands}
            
            return feature.set(values)

        extracted_features = points_fc.map(reduce_region_function)
        
        data = extracted_features.getInfo()['features']
        df = pd.json_normalize(data)
        
        df = df.rename(columns={f'properties.{b}': b for b in bands})
        df = df.rename(columns={'properties.id': 'id'})
        
        # Check if the properties.label column exists before renaming
        if 'properties.label' in df.columns:
            df = df.rename(columns={'properties.label': 'label'})
        
        df = df.drop(columns=['type', 'geometry.type', 'geometry.coordinates'])
        
        df['date'] = pd.to_datetime(date_str)
        results.append(df)
    
    final_df = pd.concat(results, ignore_index=True)

    print(f"Extracted features for {len(final_df['id'].unique())} points on {len(final_df['date'].unique())} unique dates.")
    return final_df