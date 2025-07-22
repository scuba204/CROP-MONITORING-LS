import argparse
import logging
import datetime
from typing import Dict, List, Union # Import Union for type hints

import ee

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------

EE_PROJECT        = 'winged-tenure-464005-p9'
DEFAULT_START     = '2025-06-01'
DEFAULT_END       = '2025-07-22'
DEFAULT_ROI_BBOX  = [26.999, -30.5, 29.5, -28.5]  # [minLon, minLat, maxLon, maxLat]

SOIL_LAYERS = {
    'soil_texture_clay': ('projects/soilgrids-isric/clay_mean', 'clay_0-5cm_mean'),
    'soil_texture_silt': ('projects/soilgrids-isric/silt_mean', 'silt_0-5cm_mean'),
    'soil_texture_sand': ('projects/soilgrids-isric/sand_mean', 'sand_0-5cm_mean'),
    'soil_ph': ('projects/soilgrids-isric/phh2o_mean', 'phh2o_0-5cm_mean'),
    'soil_ocd': ('projects/soilgrids-isric/ocd_mean', 'ocd_0-5cm_mean'),
    'soil_cec': ('projects/soilgrids-isric/cec_mean', 'cec_0-5cm_mean'),
    'soil_nitrogen': ('projects/soilgrids-isric/nitrogen_mean', 'nitrogen_0-5cm_mean'),
}


# -----------------------------------------------------------------------------
# 2. INITIALIZE EARTH ENGINE & LOGGING
# -----------------------------------------------------------------------------

ee.Initialize(project=EE_PROJECT)
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO
)

# -----------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def _prepare_collection(
    col_id: str, start: str, end: str, roi: ee.Geometry
) -> ee.ImageCollection:
    """
    Load an ImageCollection, filter by date and bounds.
    """
    return (
        ee.ImageCollection(col_id)
          .filterDate(start, end)
          .filterBounds(roi)
    )

def _mean_clip(
    col: ee.ImageCollection,
    bands: List[str],
    roi: ee.Geometry
) -> ee.Image:
    """
    Select bands, compute the mean and clip to ROI.
    NOTE: This helper always returns a single mean image.
          Use the `return_collection=True` flag in the main get_ functions
          if you need the full ImageCollection.
    """
    # Defensive check: If the collection is empty, mean() will fail.
    # The individual get_ functions should ideally handle empty collections
    # before calling _mean_clip for map display.
    # However, this acts as a fallback.
    if col.size().getInfo() == 0:
        logging.warning(f"_mean_clip: Empty collection passed for bands {bands}. Returning an empty image.")
        # Create an empty image with the expected band, so it has a band name
        # This helps downstream processing (like bandNames().getInfo())
        # For single-band requests, assume the first band in `bands` is the target
        if bands:
            return ee.Image().rename(bands[0]).clip(roi)
        else:
            return ee.Image().clip(roi) # Just an empty image if no bands requested


    return col.select(bands).mean().clip(roi)

def _log_date_coverage(col: ee.ImageCollection, label: str):
    """
    Logs the date coverage of an ImageCollection if not empty.
    """
    size = col.size().getInfo()
    if size == 0:
        logging.warning(f"{label}: Empty collection")
    else:
        # It's safer to use first()/last() on sorted collection for time_start property
        try:
            first_date = col.sort('system:time_start').first().get('system:time_start').getInfo()
            last_date = col.sort('system:time_start', False).first().get('system:time_start').getInfo()
            first_date_str = ee.Date(first_date).format('YYYY-MM-dd').getInfo()
            last_date_str = ee.Date(last_date).format('YYYY-MM-dd').getInfo()
            logging.info(f"{label}: {size} images from {first_date_str} to {last_date_str}")
        except Exception as e:
            logging.warning(f"{label}: Could not determine date coverage ({size} images). Error: {e}")


def safe_execute(func, **kwargs):
    """
    Call a function, catch exceptions, log stack trace, and return None on failure.
    """
    try:
        return func(**kwargs)
    except Exception:
        logging.exception(f"Failure in {func.__name__} with args={kwargs}")
        return None

# -----------------------------------------------------------------------------
# 4. BASIC ENVIRONMENTAL INDICES
# -----------------------------------------------------------------------------

def get_ndvi(
    start: str,
    end: str,
    roi: ee.Geometry,
    max_expansion_days: int = 30,
    return_collection: bool = False # Parameter to return collection or mean image
) -> Union[ee.Image, ee.ImageCollection]: # Type hint for return type
    """
    NDVI from Sentinel-2 SR Harmonized, expand date range if no images.
    Includes robust cloud masking. Can return a single mean image or a collection.
    """
    start_dt = datetime.datetime.strptime(start, '%Y-%m-%d').date()
    end_dt = datetime.datetime.strptime(end, '%Y-%m-%d').date()

    expanded_start_date_py = start_dt - datetime.timedelta(days=max_expansion_days)
    expanded_end_date_py = end_dt + datetime.timedelta(days=max_expansion_days)

    ee_start_expanded_str = expanded_start_date_py.strftime('%Y-%m-%d')
    ee_end_expanded_str = expanded_end_date_py.strftime('%Y-%m-%d')

    ee_start_expanded = ee.Date(ee_start_expanded_str)
    ee_end_expanded = ee.Date(ee_end_expanded_str)

    s2_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(ee_start_expanded, ee_end_expanded) \
        .filterBounds(roi) \
        .select(['B8', 'B4', 'QA60'])

    _log_date_coverage(s2_collection, 'NDVI (S2_SR_HARMONIZED, raw)')

    def mask_s2_clouds(image):
        qa = image.select('QA60')
        cloud_bit_mask = 1 << 10
        cirrus_bit_mask = 1 << 11
        mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
               qa.bitwiseAnd(cirrus_bit_mask).eq(0))
        return image.updateMask(mask).divide(10000).select(['B8', 'B4']) \
                     .copyProperties(image, ['system:time_start', 'system:index'])

    s2_masked_collection = s2_collection.map(mask_s2_clouds)

    _log_date_coverage(s2_masked_collection, 'NDVI (S2_SR_HARMONIZED, masked)')

    if s2_masked_collection.size().getInfo() == 0:
        logging.warning("NDVI: No images left after cloud masking. Returning an empty image/collection.")
        if return_collection:
            return ee.ImageCollection([]) # Return empty collection
        else:
            return ee.Image().rename('NDVI').clip(roi) # Return empty image for map display

    ndvi_collection = s2_masked_collection.map(lambda img:
        img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        .copyProperties(img, ['system:time_start', 'system:index'])
    )

    if return_collection:
        return ndvi_collection # Return the collection for time series
    else:
        return ndvi_collection.mean().clip(roi) # Return the mean image for map display

def get_precipitation(start: str, end: str, roi: ee.Geometry, return_collection: bool = False) -> Union[ee.Image, ee.ImageCollection]:
    """
    Mean daily precipitation (mm) from CHIRPS.
    """
    col = _prepare_collection('UCSB-CHG/CHIRPS/DAILY', start, end, roi)
    _log_date_coverage(col, 'Precipitation')

    if col.size().getInfo() == 0:
        logging.warning("Precipitation: Empty collection. Returning an empty image/collection.")
        if return_collection:
            return ee.ImageCollection([])
        else:
            return ee.Image().rename('precipitation').clip(roi) # Ensure a band is named

    if return_collection:
        return col.select(['precipitation']) # Return selected collection for time series
    else:
        return _mean_clip(col, ['precipitation'], roi) # Return mean image for map display

def get_land_surface_temperature(start: str, end: str, roi: ee.Geometry, return_collection: bool = False) -> Union[ee.Image, ee.ImageCollection]:
    """
    Mean daytime LST (°C) from MODIS.
    """
    col = _prepare_collection('MODIS/061/MOD11A1', start, end, roi)
    _log_date_coverage(col, 'Land Surface Temperature')

    if col.size().getInfo() == 0:
        logging.warning("Land Surface Temperature: Empty collection. Returning an empty image/collection.")
        if return_collection:
            return ee.ImageCollection([])
        else:
            return ee.Image().rename('LST_C').clip(roi) # Ensure band name matches

    def process_lst(img):
        return img.select('LST_Day_1km') \
                  .multiply(0.02) \
                  .subtract(273.15) \
                  .rename('LST_C') \
                  .copyProperties(img, ['system:time_start', 'system:index'])

    processed_col = col.map(process_lst)

    if return_collection:
        return processed_col # Return processed collection for time series
    else:
        return processed_col.mean().clip(roi) # Return mean image for map display

def get_humidity(start: str, end: str, roi: ee.Geometry, return_collection: bool = False) -> Union[ee.Image, ee.ImageCollection]:
    """
    Mean relative humidity (%) from ERA5-Land.
    """
    col = _prepare_collection('ECMWF/ERA5_LAND/HOURLY', start, end, roi) \
              .select(['dewpoint_temperature_2m', 'temperature_2m'])
    _log_date_coverage(col, 'Relative Humidity')

    if col.size().getInfo() == 0:
        logging.warning("Humidity: Empty collection. Returning an empty image/collection.")
        if return_collection:
            return ee.ImageCollection([])
        else:
            return ee.Image().rename('RH').clip(roi) # Ensure band name matches

    def compute_rh(img):
        Td = img.select('dewpoint_temperature_2m').subtract(273.15)
        T  = img.select('temperature_2m').subtract(273.15)
        num = Td.multiply(17.625).divide(Td.add(243.04)).exp()
        den = T.multiply(17.625).divide(T.add(243.04)).exp()
        return num.divide(den).multiply(100).rename('RH') \
                    .copyProperties(img, ['system:time_start', 'system:index'])

    processed_col = col.map(compute_rh)

    if return_collection:
        return processed_col # Return processed collection for time series
    else:
        return processed_col.mean().clip(roi)

def get_irradiance(start: str, end: str, roi: ee.Geometry, return_collection: bool = False) -> Union[ee.Image, ee.ImageCollection]:
    """
    Mean surface net solar radiation (J/m²) from ERA5-Land.
    """
    col = _prepare_collection('ECMWF/ERA5_LAND/HOURLY', start, end, roi)
    _log_date_coverage(col, 'Irradiance')

    if col.size().getInfo() == 0:
        logging.warning("Irradiance: Empty collection. Returning an empty image/collection.")
        if return_collection:
            return ee.ImageCollection([])
        else:
            return ee.Image().rename('surface_net_solar_radiation').clip(roi)

    if return_collection:
        return col.select(['surface_net_solar_radiation']) # Return selected collection
    else:
        return _mean_clip(col, ['surface_net_solar_radiation'], roi) # Return mean image

def get_evapotranspiration(start_date_str: str, end_date_str: str, geometry: ee.Geometry, return_collection: bool = False) -> Union[ee.Image, ee.ImageCollection]:
    """
    Mean Evapotranspiration from MODIS/061/MOD16A2.
    """
    # NOTE: The dataset ID was 'MODIS/006/MOD16A2' which is deprecated.
    # Changed to 'MODIS/061/MOD16A2' based on your previous logs.
    collection = ee.ImageCollection("MODIS/061/MOD16A2") \
                    .filterDate(start_date_str, end_date_str) \
                    .filterBounds(geometry)

    initial_size = collection.size().getInfo()
    logging.info(f"DEBUG: Evapotranspiration collection size after date & bounds filter ({start_date_str} to {end_date_str}): {initial_size}")

    if initial_size == 0:
        logging.warning(f"Evapotranspiration: Collection is empty after filtering. No data for {start_date_str} to {end_date_str}.")
        if return_collection:
            return ee.ImageCollection([])
        else:
            return ee.Image().rename('ET').clip(geometry)

    # If return_collection is True, return the collection directly
    if return_collection:
        # Select the 'ET' band from the collection
        return collection.select(['ET'])
    else:
        # Otherwise, compute the mean and clip for a single image display
        et = collection.select('ET').mean().clip(geometry)
        return et

def get_simulated_hyperspectral(start: str, end: str, roi: ee.Geometry, return_collection: bool = False) -> Union[ee.Image, ee.ImageCollection]:
    """
    Cloud-filtered median composite of 9 Sentinel-2 bands to mimic hyperspectral.
    """
    bands = ['B2','B3','B4','B5','B6','B7','B8A','B11','B12']
    col = (
        _prepare_collection('COPERNICUS/S2_SR_HARMONIZED', start, end, roi)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    )
    _log_date_coverage(col, 'Simulated Hyperspectral')

    if col.size().getInfo() == 0:
        logging.warning("Simulated Hyperspectral: No images left after filtering. Returning an empty image/collection.")
        if return_collection:
            return ee.ImageCollection([])
        else:
            # Create an empty image that, if possible, has the expected bands, for robustness downstream
            return ee.Image().addBands([ee.Image(0).rename(b) for b in bands]).clip(roi) # A more robust empty image for this case

    if return_collection:
        return col.select(bands) # Return the filtered collection with selected bands
    else:
        return col.median().select(bands).clip(roi)

# -----------------------------------------------------------------------------
# 5. SOIL GRID LAYERS
# -----------------------------------------------------------------------------

def get_soil_moisture(start: str, end: str, roi: ee.Geometry, return_collection: bool = False) -> Union[ee.Image, ee.ImageCollection]:
    """
    Mean soil moisture (0–10 cm) from FLDAS/NOAH.
    """
    logging.info(f"DEBUG: get_soil_moisture called with start={start}, end={end}")

    col = ee.ImageCollection('NASA/FLDAS/NOAH01/C/GL/M/V001')

    logging.info(f"DEBUG: Initial FLDAS/NOAH collection created for Soil Moisture.")

    col_filtered_date = col.filterDate(start, end)
    date_filtered_size = col_filtered_date.size().getInfo()
    logging.info(f"DEBUG: Soil Moisture collection size after date filter ({start} to {end}): {date_filtered_size}")

    if date_filtered_size == 0:
        logging.warning(f"Soil Moisture: Collection is empty after date filtering. No data for {start} to {end}.")
        if return_collection:
            return ee.ImageCollection([])
        else:
            return ee.Image().rename('SoilMoi00_10cm_tavg').clip(roi)

    col_filtered_bounds = col_filtered_date.filterBounds(roi)
    bounds_filtered_size = col_filtered_bounds.size().getInfo()
    logging.info(f"DEBUG: Soil Moisture collection size after bounds filter: {bounds_filtered_size}")

    if bounds_filtered_size == 0:
        logging.warning(f"Soil Moisture: Collection is empty after bounds filtering for ROI. No data in this region for {start} to {end}.")
        if return_collection:
            return ee.ImageCollection([])
        else:
            return ee.Image().rename('SoilMoi00_10cm_tavg').clip(roi)

    _log_date_coverage(col_filtered_bounds, 'Soil Moisture')

    if return_collection:
        return col_filtered_bounds.select(['SoilMoi00_10cm_tavg'])
    else:
        return _mean_clip(col_filtered_bounds, ['SoilMoi00_10cm_tavg'], roi)

def get_soil_property(key: str, roi: ee.Geometry) -> ee.Image:
    """
    Generic loader for SoilGrids properties.
    """
    asset, band = SOIL_LAYERS[key]
    return ee.Image(asset).select(band).clip(roi)

def get_soil_texture(start: str, end: str, roi: ee.Geometry) -> ee.Image:
    """
    Concatenate clay, silt & sand fractions into one 3-band image.
    start/end are ignored.
    """
    # Soil texture is static, so start/end are not relevant here.
    # The functions get_soil_property already handle clipping.
    clay = get_soil_property('soil_texture_clay', roi)
    silt = get_soil_property('soil_texture_silt', roi)
    sand = get_soil_property('soil_texture_sand', roi)
    return ee.Image.cat([clay, silt, sand]).rename(['clay','silt','sand'])