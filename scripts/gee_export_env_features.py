# gee_export_env_features.py
import ee
import pandas as pd

# Initialize Earth Engine
ee.Initialize(project='winged-tenure-464005-p9')

# === CONFIG ===
START_YEAR = 2000
END_YEAR = 2023
CROPS = ["Potatoes", "Wheat", "Sorghum", "Maize", "Peas", "Oats", "Barley", "Beans"]

roi = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0") \
        .filter(ee.Filter.eq("ADM0_NAME", "Lesotho"))

# === DATASETS ===
datasets = {
    "rain": ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY"),            # Rainfall (mm)
    "modis_lst": ee.ImageCollection("MODIS/061/MOD11A2"),           # LST (Kelvin*0.02)
    "modis_ndvi": ee.ImageCollection("MODIS/061/MOD13Q1"),          # NDVI (×0.0001)
    "modis_et": ee.ImageCollection("MODIS/061/MOD16A2"),            # ET (×0.1 mm/8-day)
    "era5": ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY"),           # ERA5 hourly vars
    "soil_ph": ee.Image("projects/soilgrids-isric/phh2o_mean"),
    "soil_soc": ee.Image("projects/soilgrids-isric/ocd_mean"),
    "soil_nitrogen": ee.Image("projects/soilgrids-isric/nitrogen_mean"),
    "soil_cec": ee.Image("projects/soilgrids-isric/cec_mean"),
}

# === FUNCTIONS ===
def annual_mean(imgcol, band, year):
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    img = imgcol.filterDate(start, end).select(band).mean()
    stats = img.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=roi.geometry(),
        scale=5000,
        maxPixels=1e13
    )
    return stats.get(band)  # may be None if no data


def compute_relative_humidity(temp_c, dewpoint_c):
    """Compute relative humidity (%) from T (°C) and Td (°C)."""
    if temp_c is None or dewpoint_c is None:
        return None
    try:
        rh = 100 * (
            (6.112 * ee.Number(1).exp().pow((17.67 * dewpoint_c) / (dewpoint_c + 243.5)))
            / (6.112 * ee.Number(1).exp().pow((17.67 * temp_c) / (temp_c + 243.5)))
        )
        return float(rh.getInfo())
    except Exception:
        return None

# === STATIC SOIL FEATURES ===
soil_features = {}
soil_features["soil_ph"] = datasets["soil_ph"].reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=roi.geometry(),
    scale=250,
    maxPixels=1e13
).get("phh2o_0-5cm_mean").getInfo()

soil_features["soil_soc"] = datasets["soil_soc"].reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=roi.geometry(),
    scale=250,
    maxPixels=1e13
).get("ocd_0-5cm_mean").getInfo()

soil_features["soil_nitrogen"] = datasets["soil_nitrogen"].reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=roi.geometry(),
    scale=250,
    maxPixels=1e13
).get("nitrogen_0-5cm_mean").getInfo()

soil_features["soil_cec"] = datasets["soil_cec"].reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=roi.geometry(),
    scale=250,
    maxPixels=1e13
).get("cec_0-5cm_mean").getInfo()

# === MAIN LOOP ===
records = []

for year in range(START_YEAR, END_YEAR + 1):
    print(f"Processing {year}...")

    # Rainfall (mm)
    rain = annual_mean(datasets["rain"], "precipitation", year)

    # MODIS LST (Kelvin*0.02 → °C)
    lst = annual_mean(datasets["modis_lst"], "LST_Day_1km", year)
    temp_c = None
    if lst is not None:
        temp_c = lst.getInfo() * 0.02 - 273.15

    # NDVI
    ndvi = annual_mean(datasets["modis_ndvi"], "NDVI", year)
    if ndvi is not None:
        ndvi = ndvi.getInfo() * 0.0001

    # Evapotranspiration (mm)
    et = annual_mean(datasets["modis_et"], "ET", year)
    if et is not None:
        et = et.getInfo() * 0.1 # scale factor

    # Solar Irradiation (J/m²)
    solar = annual_mean(datasets["era5"], "surface_solar_radiation_downwards", year)

    # Relative Humidity (%): use ERA5 2m temperature + dewpoint
    t2m = annual_mean(datasets["era5"], "temperature_2m", year, scale=11000)  # Kelvin
    td = annual_mean(datasets["era5"], "dewpoint_temperature_2m", year, scale=11000)  # Kelvin
    rh = None
    if t2m and td:
        temp_c2 = t2m.getInfo() - 273.15
        dewpoint_c = td.getInfo() - 273.15
        # Clausius–Clapeyron formula
        rh = 100 * (
            (6.112 * (2.718281828 ** ((17.67 * dewpoint_c) / (dewpoint_c + 243.5))))
            / (6.112 * (2.718281828 ** ((17.67 * temp_c2) / (temp_c2 + 243.5))))
        )

    # Save per crop
    for crop in CROPS:
        records.append({
            "Year": year,
            "Item": crop,
            "rainfall_mean": rain.getInfo() if rain else None,
            "temp_mean": temp_c,
            "ndvi_mean": ndvi,
            "et_mean": et,
            "solar_mean": solar.getInfo() if solar else None,
            "rel_humidity": rh,
            **soil_features
        })

# Save to CSV
df = pd.DataFrame(records)
df.to_csv("gee_env_features.csv", index=False)
print("✅ Saved features to gee_env_features.csv")
