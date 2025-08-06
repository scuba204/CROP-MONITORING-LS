#!/usr/bin/env python3
# scripts/feature_engineering.py

import os
import sys
import yaml
import pandas as pd
import ee
import geopandas as gpd
from shapely.geometry import mapping

# -----------------------------------------------------------------------------
# Ensure project root is on sys.path so we can import our modules
# -----------------------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.data_loader import load_field_data


def load_config(path: str = None) -> dict:
    """
    Load YAML configuration from config.yaml at the project root.
    """
    project_root = os.path.dirname(os.path.dirname(__file__))
    config_path = path or os.path.join(project_root, "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if not cfg:
        raise ValueError(f"Config file {config_path} is empty or invalid YAML")

    return cfg


def init_earth_engine(cfg: dict) -> None:
    """
    Initialize the Earth Engine API.
    Optionally, you can pass a project ID via cfg['gee']['project_id'].
    """
    # If you have a service-account / project to specify, use:
    # ee.Initialize(project=cfg["gee"].get("project_id"))
    ee.Initialize()
    print("Earth Engine initialized")


def get_collection(cfg: dict, roi_geojson: dict) -> ee.ImageCollection:
    """
    Build and filter an ImageCollection from Earth Engine based on the cfg.
    """
    col_id = cfg["gee"]["collection"]
    start = cfg["data"]["date_range"]["start"]
    end = cfg["data"]["date_range"]["end"]
    cloud_thresh = cfg["gee"]["cloud_threshold"]

    # Create EE ROI geometry
    ee_roi = ee.Geometry.Polygon(roi_geojson["coordinates"])

    collection = (
        ee.ImageCollection(col_id)
          .filterDate(start, end)
          .filterBounds(ee_roi)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_thresh))
    )
    print(f"Loaded EE collection {col_id} from {start} to {end}")
    return collection


def add_indices(img: ee.Image, index_defs: list[dict]) -> ee.Image:
    """
    Add spectral index bands to an input Image.
    index_defs is a list of dicts with 'name' and 'expression' keys.
    """
    def compute_band(defn):
        # defn["expression"] is a string like "(B8 - B4) / (B8 + B4)"
        return img.expression(
            defn["expression"],
            img.rename(img.bandNames())  # ensure correct band names mapping
        ).rename(defn["name"])

    bands = [compute_band(d) for d in index_defs]
    return img.addBands(ee.Image.cat(bands))


def sample_time_series(
    gdf: gpd.GeoDataFrame,
    ee_col: ee.ImageCollection,
    cfg: dict
) -> pd.DataFrame:
    """
    For each point in gdf, buffer by cfg["gee"]["buffer_radius"],
    sample each image in ee_col, compute mean of each index band,
    and return a long-form pandas DataFrame with columns:
      [id, date, <index1>, <index2>, ...]
    """
    buffer_radius = cfg["gee"]["buffer_radius"]
    index_names = [d["name"] for d in cfg["indices"]]

    all_features = []

    for idx, row in gdf.iterrows():
        # Build buffered EE geometry
        pt = ee.Geometry.Point([row.geometry.x, row.geometry.y])
        geom_buf = pt.buffer(buffer_radius)

        # Map over collection: compute indices, reduceRegion, tag with date
        def map_fn(img):
            img_with_idx = add_indices(img, cfg["indices"])
            stats = img_with_idx.select(index_names).reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geom_buf,
                scale=10,
                bestEffort=True
            )
            return stats.set("system:time_start", img.get("system:time_start"))

        ts = ee_col.map(map_fn)

        # Pull to client
        features = ts.getInfo().get("features", [])
        if not features:
            continue

        df = pd.json_normalize([f["properties"] for f in features])
        # Convert time and add ID
        df["date"] = pd.to_datetime(df["system:time_start"], unit="ms")
        df["id"] = idx

        # Keep only id, date, and our indices
        keep_cols = ["id", "date"] + index_names
        all_features.append(df[keep_cols])

    if not all_features:
        raise RuntimeError("No time-series features extracted for any point.")

    return pd.concat(all_features, ignore_index=True)


if __name__ == "__main__":
    # 1. Load configuration
    cfg = load_config()

    # 2. Initialize Earth Engine
    init_earth_engine(cfg)

    # 3. Load field geometries with labels
    fields = load_field_data(cfg)
    print(f"Loaded {len(fields)} training points")

    # 4. Build ROI and EE collection
    unioned = fields.geometry.union_all
    roi_geojson = mapping(unioned)
    ee_collection = get_collection(cfg, roi_geojson)

    # 5. Sample time-series indices for each point
    feature_df = sample_time_series(fields, ee_collection, cfg)
    print(f"Extracted features: {feature_df.shape[0]} rows, {feature_df.shape[1]} columns")

    # 6. Save to CSV
    out_path = os.path.join(project_root, "data", "features.csv")
    feature_df.to_csv(out_path, index=False)
    print(f"Feature table saved to {out_path}")
