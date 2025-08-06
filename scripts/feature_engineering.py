import os
import yaml
import pandas as pd
import ee
from shapely.geometry import mapping
import geopandas as gpd

def load_config(path=None):
    project_root = os.path.dirname(os.path.dirname(__file__))
    config_path = path or os.path.join(project_root, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg

def init_earth_engine():
    ee.Initialize(project="winged-tenure-464005-p9")
    print("Earth Engine initialized")

def get_collection(cfg, geometry):
    ee_col = (
        ee.ImageCollection(cfg["gee"]["collection"])
          .filterDate(cfg["data"]["date_range"]["start"],
                      cfg["data"]["date_range"]["end"])
          .filterBounds(ee.Geometry.Polygon(geometry))
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE",
                               cfg["gee"]["cloud_threshold"]))
    )
    return ee_col

def add_indices(img, index_defs):
    def _compute(expr, name):
        return img.expression(expr, img.toDictionary()).rename(name)
    index_images = [
        _compute(defn["expression"], defn["name"])
        for defn in index_defs
    ]
    return img.addBands(ee.Image.cat(index_images))

def sample_time_series(gdf, ee_col, cfg):
    # Buffer each point, then sample mean index value per image
    features = []
    for _, row in gdf.iterrows():
        pt_geom = ee.Geometry.Point(row.geometry.x, row.geometry.y)\
                       .buffer(cfg["gee"]["buffer_radius"])
        ts = ee_col.map(lambda img: add_indices(img, cfg["indices"]))\
                   .select([d["name"] for d in cfg["indices"]])\
                   .map(lambda img: img.reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=pt_geom,
                        scale=10
                   ).set("date", img.date().format("YYYY-MM-dd")))
        df = pd.DataFrame(ts.getInfo()["features"])  # may need pagination for large ROI
        df["id"] = row.name
        features.append(df)
    return pd.concat(features, ignore_index=True)

if __name__ == "__main__":
    cfg = load_config()
    init_earth_engine()

    # Load your fields
    from scripts.data_loader import load_field_data
    fields = load_field_data(cfg)

    # Build EE collection and sample
    ee_col = get_collection(cfg, mapping(fields.unary_union))
    feature_df = sample_time_series(fields, ee_col, cfg)

    # Save to CSV
    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
    feature_df.to_csv(out_path, index=False)
    print(f"Feature table saved to {out_path}")
