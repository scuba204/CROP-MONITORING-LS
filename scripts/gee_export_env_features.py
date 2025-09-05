# gee_export_env_features.py
import ee
import geemap
import pandas as pd

# Initialize Earth Engine
ee.Initialize(project='winged-tenure-464005-p9')  # replace with your GEE project

# === CONFIG ===
# Years to process
START_YEAR = 2000
END_YEAR = 2023

# Crops you want to include (must match FAOSTAT Item names, e.g. "Potatoes")
CROPS = ["Potatoes","Wheat","Sorghum","Maize","Peas","Oats","Barley","Beans"]

# ROI: Lesotho boundary
roi = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0") \
        .filter(ee.Filter.eq("ADM0_NAME", "Lesotho"))

# === DATASETS ===
chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")        # Rainfall
era5 = ee.ImageCollection("ECMWF/ERA5/DAILY")               # Temperature
modis = ee.ImageCollection("MODIS/061/MOD13Q1")             # NDVI
soil_ph_img = ee.Image("projects/soilgrids-isric/phh2o_mean") # Soil pH
soc_img = ee.Image("projects/soilgrids-isric/ocd_mean") # Soil Organic Carbon
solar_irradiation = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")  # Solar Irradiation

# === FUNCTIONS ===
def annual_mean(imgcol, band, year):
    """Aggregate an image collection by year and take mean over ROI"""
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    img = imgcol.filterDate(start, end).select(band).mean()
    return img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi.geometry(),
        scale=5000,
        maxPixels=1e13
    ).get(band)

# Collect results
records = []

for year in range(START_YEAR, END_YEAR+1):
    print(f"Processing {year}...")
    
    # Rainfall (mm)
    rain = annual_mean(chirps, "precipitation", year).getInfo()

    #Irradiation (J/m²)
    solar = annual_mean(solar_irradiation, "surface_solar_radiation_downwards", year).getInfo()
    
    
    # Temperature (°C)
    temp = annual_mean(modis, "LST_Day_1km", year).getInfo()
    if temp is not None:
        temp = temp *(0.02) - 273.15  # convert from Kelvin
    
    # NDVI (scaled)
    ndvi = annual_mean(modis, "NDVI", year).getInfo()
    if ndvi is not None:
        ndvi = ndvi * 0.0001  # scale factor
    
    # Soil (static → same for all years)
    soil_ph = soil_ph_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi.geometry(),
        scale=250,
        maxPixels=1e13
    ).get("phh2o_0-5cm_mean").getInfo()
    
    soil_soc = soc_img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi.geometry(),
        scale=250,
        maxPixels=1e13
    ).get("ocd_0-5cm_mean").getInfo()
    
    # Save a row per crop (since soil/climate are not crop-specific at national scale)
    for crop in CROPS:
        records.append({
            "Year": year,
            "Item": crop,
            "rainfall_mean": rain,
            "temp_mean": temp,
            "ndvi_mean": ndvi,
            "soil_ph": soil_ph,
            "soil_soc": soil_soc,
            "solar_mean": solar
        })

# Convert to DataFrame
df = pd.DataFrame(records)

# Save
df.to_csv("gee_soil_features.csv", index=False)
print("✅ Saved features to gee_soil_features.csv")
print("All done! You now have environmental features for modeling.")
print("Make sure to merge this with your FAOSTAT data for complete features.")
