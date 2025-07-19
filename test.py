import ee
import logging
from datetime import datetime
from TESTS.TEST_GEE_FUNCTIONS import get_ndvi
from scripts import gee_functions
(
    get_ndvi, gee_functions.get_precipitation, gee_functions.get_land_surface_temperature,
    gee_functions.get_humidity, gee_functions.get_irradiance, gee_functions.get_evapotranspiration,
    gee_functions.get_soil_moisture, gee_functions.get_simulated_hyperspectral,
    gee_functions.get_soil_texture, gee_functions.get_soil_property
)

# Initialize Earth Engine
ee.Initialize(project='winged-tenure-464005-p9')
logging.basicConfig(level=logging.INFO)

# Date range & ROI for testing
START_DATE = '2024-07-01'
END_DATE   = '2024-07-10'
ROI = ee.Geometry.BBox(26.999, -30.5, 29.5, -28.5)

def summarize_image(img: ee.Image, label: str):
    """
    Compute summary stats for the image in the ROI.
    """
    try:
        band_names = img.bandNames().getInfo()
        stats = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=ROI,
            scale=500,
            maxPixels=1e9
        ).getInfo()
        logging.info(f"[{label}] Bands: {band_names}")
        for b in band_names:
            logging.info(f"  {b}: mean = {round(stats.get(b, 'n/a'), 3)}")
    except Exception as e:
        logging.error(f"[{label}] Failed to summarize: {e}")

def test_all():
    logging.info("📦 Testing NDVI")
    summarize_image(get_ndvi(START_DATE, END_DATE, ROI), "NDVI")

    logging.info("📦 Testing Precipitation")
    summarize_image(gee_functions.get_precipitation(START_DATE, END_DATE, ROI), "Precipitation")

    logging.info("📦 Testing Land Surface Temperature")
    summarize_image(gee_functions.get_land_surface_temperature(START_DATE, END_DATE, ROI), "Land Surface Temp")

    logging.info("📦 Testing Humidity")
    summarize_image(gee_functions.get_humidity(START_DATE, END_DATE, ROI), "Humidity")

    logging.info("📦 Testing Irradiance")
    summarize_image(gee_functions.get_irradiance(START_DATE, END_DATE, ROI), "Irradiance")

    logging.info("📦 Testing Evapotranspiration")
    summarize_image(gee_functions.get_evapotranspiration(START_DATE, END_DATE, ROI), "Evapotranspiration")

    logging.info("📦 Testing Soil Moisture")
    summarize_image(gee_functions.get_soil_moisture(START_DATE, END_DATE, ROI), "Soil Moisture")

    logging.info("📦 Testing Simulated Hyperspectral")
    summarize_image(gee_functions.get_simulated_hyperspectral(START_DATE, END_DATE, ROI), "Hyperspectral")

    logging.info("📦 Testing Soil Texture")
    summarize_image(gee_functions.get_soil_texture(START_DATE, END_DATE, ROI), "Soil Texture")

    for key in ['ph', 'ocd', 'cec']:
        label = f"Soil Property - {key.upper()}"
        logging.info(f"📦 Testing {label}")
        summarize_image(gee_functions.get_soil_property(key, ROI), label)

if __name__ == '__main__':
    test_all()
