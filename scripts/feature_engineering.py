#!/usr/bin/env python3
import os
import sys
import yaml
import pandas as pd
import ee
import geopandas as gpd
from shapely.geometry import mapping
from shapely.set_operations import union_all # Shapely 2.x

# Project root import hack
project_root = os.path.dirname(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from scripts.data_loader import load_field_data

# ──────────────────────────────────────────────────────────────────────────────
def load_config(path: str = None) -> dict:
    cfg_path = path or os.path.join(project_root, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(cfg_path)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError("Empty or invalid YAML")
    return cfg

def init_earth_engine(cfg: dict) -> None:
    pid = cfg["gee"].get("project_id")
    ee.Initialize(project=pid) if pid else ee.Initialize(project="winged-tenure-464005-p9")
    print("Earth Engine initialized")

def get_collection(cfg: dict, roi_geojson: dict) -> ee.ImageCollection:
    col = cfg["gee"]["collection"]
    start = cfg["data"]["date_range"]["start"]
    end = cfg["data"]["date_range"]["end"]
    ct = cfg["gee"]["cloud_threshold"]
    roi = ee.Geometry(roi_geojson)
    coll = (
        ee.ImageCollection(col)
        .filterDate(start, end)
        .filterBounds(roi)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", ct))
    )
    print(f"Loaded EE collection '{col}' {start}→{end}, cloud≤{ct}%")
    return coll

def add_indices(img: ee.Image, index_defs: list[dict]) -> ee.Image:
    """
    Add spectral indices by expression, providing a dict
    that maps each symbol in the expression to an ee.Image
    or constant.
    """
    new_bands = []

    for defn in index_defs:
        expr = defn["expression"]
        var_map = {}

        # Build the mapping: symbol -> ee.Image or number
        for symbol, source in defn["vars"].items():
            if isinstance(source, (int, float)):
                var_map[symbol] = source
            else:
                # assume `source` is a band name like "B8"
                var_map[symbol] = img.select(source)

        # Compute and rename
        idx_band = img.expression(expr, var_map).rename(defn["name"])
        new_bands.append(idx_band)

    # Concatenate all new bands onto the original image
    return img.addBands(ee.Image.cat(new_bands))

def sample_time_series(
    gdf: gpd.GeoDataFrame,
    ee_col: ee.ImageCollection,
    cfg: dict
) -> pd.DataFrame:
    buf = cfg["gee"]["buffer_radius"]
    idx_names = [d["name"] for d in cfg["indices"]]
    rows = []

    for idx, row in gdf.iterrows():
        point = ee.Geometry.Point([row.geometry.x, row.geometry.y])
        buf_geom = point.buffer(buf)

        def mapper(img):
            with_idx = add_indices(img, cfg["indices"])
            stats = with_idx.select(idx_names).reduceRegion(
                ee.Reducer.mean(),
                buf_geom,
                scale=10,
                bestEffort=True
            )
            # FIX: Wrap the dictionary in an ee.Feature to satisfy the .map() requirement
            return ee.Feature(None, stats.set("system:time_start", img.get("system:time_start")))

        ts = ee_col.map(mapper).getInfo().get("features", [])
        if not ts:
            continue

        df = pd.json_normalize([f["properties"] for f in ts])
        df["date"] = pd.to_datetime(df["system:time_start"], unit="ms")
        df["id"] = idx
        rows.append(df[["id", "date"] + idx_names])

    if not rows:
        raise RuntimeError("No time-series features extracted.")
    return pd.concat(rows, ignore_index=True)

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config()
    init_earth_engine(cfg)

    fields = load_field_data(cfg)
    print(f"Loaded {len(fields)} training points")

    # Create unified ROI
    unioned = union_all(fields.geometry)
    geojson = mapping(unioned)
    ee_col = get_collection(cfg, geojson)

    feature_df = sample_time_series(fields, ee_col, cfg)
    print(f"Extracted {feature_df.shape[0]}×{feature_df.shape[1]}")

    out = os.path.join(project_root, "data", "features.csv")
    feature_df.to_csv(out, index=False)
    print(f"Saved features to {out}")