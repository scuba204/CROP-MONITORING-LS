import os
import yaml
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

def load_config(path=None):
    # Default to disease config
    if path:
        config_path = path
    else:
        project_root = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(project_root, "configs", "config_disease.yaml")

    print("Loading config from:", config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(config_path)

    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError("Empty or invalid disease_config.yaml")
    return cfg
def load_field_data(cfg):
    csv_path = cfg["data"]["csv_path"]

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Require disease column for labels
    if "disease" not in df.columns:
        raise KeyError("CSV must contain a 'disease' column")

    if "longitude" in df and "latitude" in df:
        geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
        return gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    else:
        print("⚠️ No longitude/latitude found, returning plain DataFrame")
        return df
