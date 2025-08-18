#!/usr/bin/env python3
"""
extract_features.py
Refactored script to enforce fixed Earth Engine project, robust feature extraction, and safe handling of large datasets.
"""

import os
import sys
from typing import List, Optional, Union

import ee
import pandas as pd
import geopandas as gpd

# Ensure project root is on path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.gee_functions import _mask_s2_clouds  # pixel-level cloud/shadow mask

EE_PROJECT = 'winged-tenure-464005-p9'

class GEEInitError(RuntimeError):
    pass


def initialize_ee_project():
    try:
        ee.Initialize(project=EE_PROJECT)
        print(f"✅ Earth Engine initialized with project: {EE_PROJECT}")
    except Exception as e:
        raise GEEInitError(f"Failed to initialize Earth Engine with project {EE_PROJECT}: {e}")


def _ensure_ids(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if 'id' not in gdf.columns or gdf['id'].isna().any():
        gdf = gdf.reset_index(drop=True)
        gdf['id'] = gdf.index.astype(int)
    return gdf


def _gdf_to_feature_collection(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    gdf_wgs84 = gdf.to_crs(4326)
    features = []
    for idx, row in gdf_wgs84.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty or geom.geom_type != 'Point':
            continue
        features.append(ee.Feature(ee.Geometry.Point([geom.x, geom.y]), {'id': int(row['id'])}))
    return ee.FeatureCollection(features)


def _choose_reducer(name: str) -> ee.Reducer:
    mapping = {
        'first': ee.Reducer.first(), 'mode': ee.Reducer.first(),
        'mean': ee.Reducer.mean(), 'avg': ee.Reducer.mean(), 'average': ee.Reducer.mean(),
        'median': ee.Reducer.median(),
        'min': ee.Reducer.min(), 'minimum': ee.Reducer.min(),
        'max': ee.Reducer.max(), 'maximum': ee.Reducer.max(),
        'percentile_10': ee.Reducer.percentile([10]), 'p10': ee.Reducer.percentile([10]),
        'percentile_90': ee.Reducer.percentile([90]), 'p90': ee.Reducer.percentile([90])
    }
    return mapping.get(name.lower(), ee.Reducer.median())


def extract_spectral_features(
    gdf: gpd.GeoDataFrame,
    start_date: str,
    end_date: str,
    bands: List[str],
    scale: int = 10,
    cloud_filter: int = 10,
    reducer: str = 'median',
    include_geometry: bool = False,
    scale_reflectance: bool = False,
    export_to_drive: bool = False,
    drive_folder: Optional[str] = None,
    file_name: str = 's2_point_features',
    max_records_client: int = 100000
) -> Union[pd.DataFrame, ee.batch.Task, None]:

    print("Connecting to Google Earth Engine…")
    initialize_ee_project()

    gdf = _ensure_ids(gdf)
    points_fc = _gdf_to_feature_collection(gdf)
    reducer_fn = _choose_reducer(reducer)

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_filter))
        .map(_mask_s2_clouds)
        .select(bands)
        .sort('system:time_start')
    )

    if scale_reflectance:
        s2 = s2.map(lambda img: img.toFloat().divide(10000).copyProperties(img, img.propertyNames()))

    def _sample_image(image: ee.Image) -> ee.FeatureCollection:
        return image.sampleRegions(
            collection=points_fc,
            scale=scale,
            geometries=include_geometry,
            tileScale=4
        ).map(lambda f: f.set({
            'date': image.date().millis(),
            'image_id': image.id(),
        }))

    fc = s2.map(_sample_image).flatten()

    # Estimate dataset size and switch to Drive export if too large
    num_points = len(gdf)
    num_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days + 1
    estimated_samples = num_points * num_days

    if estimated_samples > 5000:
        print(f"⚠️ Large dataset (~{estimated_samples} samples). Using Drive export.")
        export_to_drive = True

    if export_to_drive:
        description = f"{file_name}_S2_{start_date}_to_{end_date}"
        task = ee.batch.Export.table.toDrive(
            collection=fc,
            description=description,
            folder=drive_folder or 'EE_Exports',
            fileNamePrefix=file_name,
            fileFormat='CSV'
            # removed project parameter here
        )
        task.start()
        print(f"Started export task: {description}")
        return task

    print("Fetching results to client (small dataset)…")
    try:
        data = ee.FeatureCollection(fc.limit(max_records_client)).getInfo()['features']
    except Exception as e:
        print(f"Earth Engine API Error during getInfo(): {e}")
        return pd.DataFrame()

    if not data:
        print("No features extracted.")
        return pd.DataFrame()

    df = pd.json_normalize(data)
    expected_cols = ['properties.id', 'properties.date', 'properties.image_id'] + [f'properties.{b}' for b in bands]
    available_cols = [c for c in expected_cols if c in df.columns]
    df = df[available_cols]

    rename_map = {f'properties.{b}': b for b in bands}
    rename_map.update({'properties.id': 'id', 'properties.date': 'date', 'properties.image_id': 'image_id'})
    df = df.rename(columns=rename_map)

    df['id'] = df['id'].astype(int)
    df['date'] = pd.to_datetime(df['date'], unit='ms')
    df = df.sort_values([c for c in ['id','date','image_id'] if c in df.columns]).reset_index(drop=True)

    print(f"Extracted {len(df)} samples across {df['id'].nunique()} points and {df['date'].nunique()} dates.")
    return df
