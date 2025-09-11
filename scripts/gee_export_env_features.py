import ee
import pandas as pd

# Initialize Earth Engine
try:
    ee.Initialize(project='winged-tenure-464005-p9')
except ee.EEException as e:
    print(f"Failed to initialize Earth Engine: {e}")
    exit()

# === CONFIG ===
START_YEAR = 2000
END_YEAR = 2023
CROPS = ["Potatoes", "Wheat", "Sorghum", "Maize", "Peas", "Oats", "Barley", "Beans"]
MISSING_VALUE = -999  # placeholder for missing data

roi = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0") \
        .filter(ee.Filter.eq("ADM0_NAME", "Lesotho"))

# === DATASETS ===
DATASETS = {
    "rain": ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY"),
    "modis_lst": ee.ImageCollection("MODIS/061/MOD11A2"),
    "modis_ndvi": ee.ImageCollection("MODIS/061/MOD13Q1"),
    "modis_et": ee.ImageCollection("MODIS/061/MOD16A2"),
    "era5": ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY"),
    "soil_ph": ee.Image("projects/soilgrids-isric/phh2o_mean"),
    "soil_soc": ee.Image("projects/soilgrids-isric/ocd_mean"),
    "soil_nitrogen": ee.Image("projects/soilgrids-isric/nitrogen_mean"),
    "soil_cec": ee.Image("projects/soilgrids-isric/cec_mean"),
}

# === STATIC SOIL FEATURES ===
soil_features = {}
try:
    soil_features["soil_ph"] = DATASETS["soil_ph"].reduceRegion(
        reducer=ee.Reducer.mean(), geometry=roi.geometry(),
        scale=250, maxPixels=1e13
    ).get("phh2o_0-5cm_mean").getInfo() or MISSING_VALUE

    soil_features["soil_soc"] = DATASETS["soil_soc"].reduceRegion(
        reducer=ee.Reducer.mean(), geometry=roi.geometry(),
        scale=250, maxPixels=1e13
    ).get("ocd_0-5cm_mean").getInfo() or MISSING_VALUE

    soil_features["soil_nitrogen"] = DATASETS["soil_nitrogen"].reduceRegion(
        reducer=ee.Reducer.mean(), geometry=roi.geometry(),
        scale=250, maxPixels=1e13
    ).get("nitrogen_0-5cm_mean").getInfo() or MISSING_VALUE

    soil_features["soil_cec"] = DATASETS["soil_cec"].reduceRegion(
        reducer=ee.Reducer.mean(), geometry=roi.geometry(),
        scale=250, maxPixels=1e13
    ).get("cec_0-5cm_mean").getInfo() or MISSING_VALUE

except ee.EEException as e:
    print(f"Failed to retrieve static soil features: {e}")
    soil_features = {
        "soil_ph": MISSING_VALUE,
        "soil_soc": MISSING_VALUE,
        "soil_nitrogen": MISSING_VALUE,
        "soil_cec": MISSING_VALUE
    }

# === FUNCTIONS ===
def get_annual_mean(img_col, band_name, factor=1, offset=0):
    """Computes the annual mean for a given band."""
    start_date = ee.Date.fromYMD(year, 1, 1)
    end_date = start_date.advance(1, "year")

    img_filtered = img_col.filterDate(start_date, end_date)

    mean_img = ee.Image(ee.Algorithms.If(
        img_filtered.size().gt(0),
        img_filtered.select(band_name).mean().multiply(factor).add(offset),
        ee.Image.constant(MISSING_VALUE)
    )).rename(band_name)

    return mean_img


def compute_annual_features(year):
    """Computes a single image with all annual mean features for a given year."""
    rain_img = get_annual_mean(DATASETS["rain"], "precipitation")
    lst_img = get_annual_mean(DATASETS["modis_lst"], "LST_Day_1km", factor=0.02, offset=-273.15)
    ndvi_img = get_annual_mean(DATASETS["modis_ndvi"], "NDVI", factor=0.0001)
    et_img = get_annual_mean(DATASETS["modis_et"], "ET", factor=0.1)
    solar_img = get_annual_mean(DATASETS["era5"], "surface_solar_radiation_downwards")

    t2m_img = get_annual_mean(DATASETS["era5"], "temperature_2m")
    td_img = get_annual_mean(DATASETS["era5"], "dewpoint_temperature_2m")

    rh_img = ee.Image(ee.Algorithms.If(
        t2m_img.bandNames().size().eq(1).And(td_img.bandNames().size().eq(1)),
        ee.Image(t2m_img).subtract(273.15).set("dewpoint", ee.Image(td_img).subtract(273.15)),
        ee.Image.constant(MISSING_VALUE).rename("rel_humidity")
    ))

    rh_img = ee.Image(ee.Algorithms.If(
        rh_img.bandNames().size().eq(1),
        rh_img.expression(
            "100 * (6.112 * exp((17.67 * dewpoint) / (dewpoint + 243.5))) / "
            "(6.112 * exp((17.67 * temp) / (temp + 243.5)))",
            {'temp': rh_img.select(0), 'dewpoint': rh_img.get('dewpoint')}
        ).rename("rel_humidity"),
        rh_img
    ))

    return rain_img.addBands(lst_img).addBands(ndvi_img).addBands(et_img).addBands(solar_img).addBands(rh_img)


def safe_reduceRegion(img, bands, scale=5000):
    """Safely reduce an image and return dict with MISSING_VALUE for missing values."""
    try:
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=roi.geometry(),
            scale=scale,
            maxPixels=1e13
        ).getInfo()
    except Exception:
        return {band: MISSING_VALUE for band in bands}

    clean = {}
    for band in bands:
        val = stats.get(band) if stats else None
        clean[band] = val if val is not None else MISSING_VALUE
    return clean

# === MAIN LOOP ===
records = []
try:
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Processing {year}...")

        annual_image = compute_annual_features(year)
        bands = annual_image.bandNames().getInfo()
        annual_stats = safe_reduceRegion(annual_image, bands)

        for crop in CROPS:
            records.append({
                "Year": year,
                "Item": crop,
                **annual_stats,
                **soil_features
            })

    if records:
        df = pd.DataFrame(records)
        df.to_csv("gee_env_features_final.csv", index=False)
        print("✅ Saved features to gee_env_features_final.csv")
    else:
        print("⚠️ No data was processed. No CSV file was saved.")

except Exception as e:
    print(f"An error occurred: {e}")
