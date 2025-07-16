import ee
import os
import csv
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import mapping
from datetime import datetime, timedelta
from matplotlib.backends.backend_pdf import PdfPages
from scripts.gee_functions import (
    get_ndvi, get_soil_moisture, get_precipitation,
    get_land_surface_temperature, get_humidity, get_irradiance, get_simulated_hyperspectral,
    get_soil_organic_matter, get_soil_ph, get_soil_texture, get_evapotranspiration,
    get_soil_nitrogen, get_soil_phosphorus, get_soil_potassium
)

# Initialize Earth Engine
ee.Initialize(project='winged-tenure-464005-p9')

# Set date range (yesterday to today)
end_date = datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')

# Load district boundaries
shp_path = r'C:\Users\MY PC\Documents\GIS DATA\BOUNDARIES\LSO_adm\LSO_adm1.shp'
district_gdf = gpd.read_file(shp_path)
district_features = [(i + 1, ee.Geometry(mapping(geom))) for i, geom in enumerate(district_gdf.geometry)]

# Output directories
os.makedirs('outputs/daily_maps', exist_ok=True)
os.makedirs('outputs/district_logs', exist_ok=True)
os.makedirs('outputs/visualizations', exist_ok=True)
os.makedirs('outputs/pdfs', exist_ok=True)

# Log path
log_path = 'outputs/district_logs/district_full_log.csv'
write_header = not os.path.exists(log_path)

# Open CSV log file
with open(log_path, 'a', newline='') as log_file:
    writer = csv.writer(log_file)
    if write_header:
        writer.writerow([
            'date', 'district_id', 'NDVI', 'SoilMoisture', 'Precipitation', 'LST', 'Humidity',
            'Irradiance', 'SOM', 'Soil_pH', 'Soil_Texture', 'ET', 'Nitrogen', 'Phosphorus', 'Potassium',
            'B5', 'B6', 'B7', 'B11', 'B12'
        ])

    # Loop through each district
    for district_id, district_geom in district_features:
        try:
            # Retrieve all parameters
            ndvi = get_ndvi(start_date, end_date, district_geom)
            soil = get_soil_moisture(start_date, end_date, district_geom)
            precip = get_precipitation(start_date, end_date, district_geom)
            lst = get_land_surface_temperature(start_date, end_date, district_geom)
            humidity = get_humidity(start_date, end_date, district_geom)
            irradiance = get_irradiance(start_date, end_date, district_geom)
            som = get_soil_organic_matter(district_geom)
            soil_ph = get_soil_ph(district_geom)
            texture = get_soil_texture(district_geom)
            et = get_evapotranspiration(start_date, end_date, district_geom)
            nitrogen = get_soil_nitrogen(district_geom)
            phosphorus = get_soil_phosphorus(district_geom)
            potassium = get_soil_potassium(district_geom)
            hyper = get_simulated_hyperspectral(start_date, end_date, district_geom)

            # Select hyperspectral bands
            b5 = hyper.select("B5")
            b6 = hyper.select("B6")
            b7 = hyper.select("B7")
            b11 = hyper.select("B11")
            b12 = hyper.select("B12")

            # Stack all parameters for region reduction
            image = ee.Image.cat([
                ndvi.rename("NDVI"), soil.rename("SoilMoisture"), precip.rename("Precipitation"),
                lst.rename("LST"), humidity.rename("Humidity"), irradiance.rename("Irradiance"),
                som.rename("SOM"), soil_ph.rename("Soil_pH"), texture.rename("Soil_Texture"),
                et.rename("ET"), nitrogen.rename("Nitrogen"), phosphorus.rename("Phosphorus"),
                potassium.rename("Potassium"), b5.rename("B5"), b6.rename("B6"),
                b7.rename("B7"), b11.rename("B11"), b12.rename("B12")
            ])

            # Compute mean statistics for the district
            stats = image.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=district_geom,
                scale=30,
                maxPixels=1e9
            ).getInfo()

            # Write to CSV
            record = [
                end_date, district_id,
                round(stats.get("NDVI", 0), 3), round(stats.get("SoilMoisture", 0), 3),
                round(stats.get("Precipitation", 0), 3), round(stats.get("LST", 0), 3),
                round(stats.get("Humidity", 0), 3), round(stats.get("Irradiance", 0), 3),
                round(stats.get("SOM", 0), 3), round(stats.get("Soil_pH", 0), 3),
                round(stats.get("Soil_Texture", 0), 3), round(stats.get("ET", 0), 3),
                round(stats.get("Nitrogen", 0), 3), round(stats.get("Phosphorus", 0), 3),
                round(stats.get("Potassium", 0), 3), round(stats.get("B5", 0), 2),
                round(stats.get("B6", 0), 2), round(stats.get("B7", 0), 2),
                round(stats.get("B11", 0), 2), round(stats.get("B12", 0), 2)
            ]
            writer.writerow(record)

            print(f"✅ District {district_id} recorded successfully.")

        except Exception as e:
            print(f"❌ District {district_id} - ERROR: {e}")

# Generate trend plots and export to PDF
try:
    df = pd.read_csv(log_path)
    params = ['NDVI', 'SoilMoisture', 'Precipitation', 'LST', 'ET']
    with PdfPages('outputs/pdfs/parameter_trends.pdf') as pdf:
        for param in params:
            plt.figure(figsize=(10, 5))
            for did in df['district_id'].unique():
                subset = df[df['district_id'] == did]
                plt.plot(pd.to_datetime(subset['date']), subset[param], label=f'District {did}')
            plt.title(f"Trend of {param} Over Time")
            plt.xlabel("Date")
            plt.ylabel(param)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"outputs/visualizations/{param}_trend.png")
            pdf.savefig()
            plt.close()
    print("📄 PDF report saved to outputs/pdfs/parameter_trends.pdf")
except Exception as e:
    print(f"⚠️ Could not generate PDF report: {e}")
