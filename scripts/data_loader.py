import yaml
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os

def load_config(path=None):
    # If no path passed, locate config.yaml at project root
    if path:
        config_path = path
    else:
        # __file__ is scripts/data_loader.py → go up one level
        project_root = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(project_root, "config.yaml")

    print(f"Loading config from: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    if not cfg:
        raise ValueError(f"Config file {config_path} is empty or invalid YAML")

    return cfg
if __name__ == "__main__":
    cfg = load_config()
    print("Loaded config:", cfg)
    gdf = load_field_data(cfg)
    print(f"Loaded {len(gdf)} records.")
    print(gdf.head())
