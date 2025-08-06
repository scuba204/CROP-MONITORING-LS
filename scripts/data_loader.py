import os
import yaml
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def load_config(path=None):
    # Locate config.yaml next to your project root
    if path:
        config_path = path
    else:
        project_root = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(project_root, "config.yaml")
    print("Loading config from:", config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError("Empty or invalid config.yaml")
    return cfg

def load_field_data(cfg):
    # Must use the same cfg variable
    csv_path = cfg["data"]["csv_path"]
    df = pd.read_csv(csv_path)
    if "longitude" in df and "latitude" in df:
        geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
        return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    else:
        raise KeyError("Missing 'longitude'/'latitude' in CSV")

if __name__ == "__main__":
    # This block only runs when you call the script directly
    cfg = load_config()
    print("Config keys:", list(cfg.keys()))
    gdf = load_field_data(cfg)
    print(f"Loaded {len(gdf)} geometries")
    print(gdf.head())
