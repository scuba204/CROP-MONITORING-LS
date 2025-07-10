
import ee
import os
import csv
import geopandas as gpd
from shapely.geometry import mapping
from datetime import datetime, timedelta
from geemap import ee_export_image
from scripts.gee_functions import get_ndvi


ee.Initialize(project='winged-tenure-464005-p9')

# Set date for automation (yesterday to today)
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')


farm_gdf = gpd.read_file("data/farms.geojson")
farm_features = [ee.Geometry(mapping(geom)) for geom in farm_gdf.geometry]


os.makedirs('outputs/daily_maps', exist_ok=True)
os.makedirs('outputs/farm_logs', exist_ok=True)


log_path = 'outputs/farm_logs/farm_ndvi_log.csv'
write_header = not os.path.exists(log_path)

with open(log_path, 'a', newline='') as log_file:
    writer = csv.writer(log_file)
    if write_header:
        writer.writerow(['date', 'farm_id', 'mean_ndvi'])

    for i, farm in enumerate(farm_features):
        try:

            ndvi_img = get_ndvi(start_date, end_date, farm)
            stats = ndvi_img.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=farm,
                scale=10,
                maxPixels=1e9
            ).getInfo()

            mean_ndvi = stats.get('NDVI', None)

            if mean_ndvi is not None:
                print(f"Farm {i+1} - NDVI: {mean_ndvi:.3f}")
                writer.writerow([end_date, i+1, round(mean_ndvi, 4)])

                # Export optional GeoTIFF
                export_path = f"outputs/daily_maps/farm_{i+1}_ndvi_{end_date}.tif"
                ee_export_image(
                    image=ndvi_img,
                    filename=export_path,
                    region=farm,
                    scale=10,
                    file_per_band=False
                )
            else:
                print(f"Farm {i+1} - No NDVI data available.")
        except Exception as e:
            print(f"Farm {i+1} - ERROR: {e}")
