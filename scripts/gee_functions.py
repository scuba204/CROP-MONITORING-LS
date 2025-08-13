import argparse
import logging
import datetime
from typing import Dict, List, Union, Callable

import ee

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------

EE_PROJECT          = 'winged-tenure-464005-p9'
DEFAULT_START       = '2025-06-01'
DEFAULT_END         = '2025-07-22'
DEFAULT_ROI_BBOX    = [26.999, -30.5, 29.5, -28.5]

SOIL_LAYERS = {
    'soil_texture_clay': ('projects/soilgrids-isric/clay_mean', 'clay_0-5cm_mean'),
    'soil_texture_silt': ('projects/soilgrids-isric/silt_mean', 'silt_0-5cm_mean'),
    'soil_texture_sand': ('projects/soilgrids-isric/sand_mean', 'sand_0-5cm_mean'),
    'soil_ph': ('projects/soilgrids-isric/phh2o_mean', 'phh2o_0-5cm_mean'),
    'soil_organic_matter': ('projects/soilgrids-isric/ocd_mean', 'ocd_0-5cm_mean'),
    'soil_cec': ('projects/soilgrids-isric/cec_mean', 'cec_0-5cm_mean'),
    'soil_nitrogen': ('projects/soilgrids-isric/nitrogen_mean', 'nitrogen_0-5cm_mean'),
}

# -----------------------------------------------------------------------------
# 2. INITIALIZE EARTH ENGINE & LOGGING
# -----------------------------------------------------------------------------

ee.Initialize(project=EE_PROJECT)
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)

# -----------------------------------------------------------------------------
# 3. CORE HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def _mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Masks clouds in a Sentinel-2 SR image using the QA60 band."""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000).copyProperties(image, ['system:time_start', 'system:index'])

def _log_date_coverage(col: ee.ImageCollection, label: str):
    """Logs the date coverage of an ImageCollection if not empty."""
    size = col.size().getInfo()
    if size == 0:
        logging.warning(f"{label}: Empty collection")
    else:
        try:
            first_date = ee.Date(col.sort('system:time_start').first().get('system:time_start')).format('YYYY-MM-dd').getInfo()
            last_date = ee.Date(col.sort('system:time_start', False).first().get('system:time_start')).format('YYYY-MM-dd').getInfo()
            logging.info(f"{label}: {size} images from {first_date} to {last_date}")
        except Exception as e:
            logging.warning(f"{label}: Could not determine date coverage ({size} images). Error: {e}")

def _get_s2_derived_product(
    start: str, end: str, roi: ee.Geometry, bands: List[str], index_name: str,
    calculation_function: Callable[[ee.Image], ee.Image],
    return_collection: bool, max_expansion_days: int = 30, reducer: ee.Reducer = ee.Reducer.mean()
) -> Union[ee.Image, ee.ImageCollection]:
    """
    Generic factory for all Sentinel-2 derived products (e.g., NDVI, SAVI, EVI).
    Handles date expansion, cloud masking, calculation, and reduction.
    """
    # CORRECTED SECTION: Convert calculated dates to strings for the API call
    start_dt_obj = datetime.datetime.strptime(start, '%Y-%m-%d').date() - datetime.timedelta(days=max_expansion_days)
    end_dt_obj = datetime.datetime.strptime(end, '%Y-%m-%d').date() + datetime.timedelta(days=max_expansion_days)
    start_date_str = start_dt_obj.strftime('%Y-%m-%d')
    end_date_str = end_dt_obj.strftime('%Y-%m-%d')

    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate(start_date_str, end_date_str).filterBounds(roi)
    _log_date_coverage(s2_col, f'{index_name} (Raw S2)')

    # Select necessary bands plus the QA band for masking
    s2_masked = s2_col.select(bands + ['QA60']).map(_mask_s2_clouds)
    _log_date_coverage(s2_masked, f'{index_name} (Masked S2)')

    if s2_masked.size().getInfo() == 0:
        logging.warning(f"{index_name}: No valid images found. Returning empty object.")
        return ee.ImageCollection([]) if return_collection else ee.Image().rename(index_name).clip(roi)

    processed_col = s2_masked.map(calculation_function)

    if return_collection:
        return processed_col
    else:
        # Use the specified reducer (mean, median, etc.)
        return processed_col.reduce(reducer).rename(index_name).clip(roi)

def _get_generic_product(
    collection_id: str, band: str, start: str, end: str, roi: ee.Geometry,
    return_collection: bool, processing_function: Callable[[ee.Image], ee.Image] = None
) -> Union[ee.Image, ee.ImageCollection]:
    """
    Generic factory for non-S2 time-series products (e.g., CHIRPS, MODIS, ERA5).
    Handles fetching, optional processing, and reduction.
    """
    col = ee.ImageCollection(collection_id).filterDate(start, end).filterBounds(roi)
    _log_date_coverage(col, f'Generic Product: {band}')

    if col.size().getInfo() == 0:
        logging.warning(f"{band}: Empty collection. Returning empty object.")
        return ee.ImageCollection([]) if return_collection else ee.Image().rename(band).clip(roi)

    processed_col = col.map(processing_function) if processing_function else col.select(band)

    if return_collection:
        return processed_col
    else:
        return processed_col.mean().clip(roi)

def safe_execute(func, **kwargs):
    """Calls a function, catches exceptions, and logs them."""
    try:
        return func(**kwargs)
    except Exception:
        logging.exception(f"Failure in {func.__name__} with args={kwargs}")
        return None

# -----------------------------------------------------------------------------
# 4. BASIC ENVIRONMENTAL INDICES (Including all from config.yaml)
# -----------------------------------------------------------------------------

def get_ndvi(start: str, end: str, roi: ee.Geometry, **kwargs) -> Union[ee.Image, ee.ImageCollection]:
    """NDVI from Sentinel-2. Formula: (B8 - B4) / (B8 + B4)"""
    calc = lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI').copyProperties(img, ['system:time_start'])
    return _get_s2_derived_product(start, end, roi, ['B8', 'B4'], 'NDVI', calc, **kwargs)

def get_savi(start: str, end: str, roi: ee.Geometry, L: float = 0.5, **kwargs) -> Union[ee.Image, ee.ImageCollection]:
    """SAVI from Sentinel-2. Formula: ((B8 - B4) / (B8 + B4 + L)) * (1 + L)"""
    calc = lambda img: img.expression('((NIR - RED) * (1 + L)) / (NIR + RED + L)',
        {'NIR': img.select('B8'), 'RED': img.select('B4'), 'L': L}).rename('SAVI').copyProperties(img, ['system:time_start'])
    return _get_s2_derived_product(start, end, roi, ['B8', 'B4'], 'SAVI', calc, **kwargs)

def get_evi(start: str, end: str, roi: ee.Geometry, **kwargs) -> Union[ee.Image, ee.ImageCollection]:
    """EVI from Sentinel-2. Formula: 2.5 * ((B8 - B4) / (B8 + 6 * B4 - 7.5 * B2 + 1))"""
    calc = lambda img: img.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
        {'NIR': img.select('B8'), 'RED': img.select('B4'), 'BLUE': img.select('B2')}).rename('EVI').copyProperties(img, ['system:time_start'])
    return _get_s2_derived_product(start, end, roi, ['B2', 'B4', 'B8'], 'EVI', calc, **kwargs)

def get_ndwi(start: str, end: str, roi: ee.Geometry, **kwargs) -> Union[ee.Image, ee.ImageCollection]:
    """NDWI (McFeeters) from Sentinel-2. Formula: (B3 - B8) / (B3 + B8)"""
    calc = lambda img: img.normalizedDifference(['B3', 'B8']).rename('NDWI').copyProperties(img, ['system:time_start'])
    return _get_s2_derived_product(start, end, roi, ['B3', 'B8'], 'NDWI', calc, **kwargs)

def get_ndmi(start: str, end: str, roi: ee.Geometry, **kwargs) -> Union[ee.Image, ee.ImageCollection]:
    """NDMI from Sentinel-2. Formula: (B8 - B11) / (B8 + B11)"""
    calc = lambda img: img.normalizedDifference(['B8', 'B11']).rename('NDMI').copyProperties(img, ['system:time_start'])
    return _get_s2_derived_product(start, end, roi, ['B8', 'B11'], 'NDMI', calc, **kwargs)

# ----- NEWLY ADDED FUNCTIONS BASED ON config.yaml -----

def get_ndre(start: str, end: str, roi: ee.Geometry, **kwargs) -> Union[ee.Image, ee.ImageCollection]:
    """NDRE from Sentinel-2. Formula: (B8 - B5) / (B8 + B5)"""
    calc = lambda img: img.normalizedDifference(['B8', 'B5']).rename('NDRE').copyProperties(img, ['system:time_start'])
    return _get_s2_derived_product(start, end, roi, ['B8', 'B5'], 'NDRE', calc, **kwargs)

def get_msi(start: str, end: str, roi: ee.Geometry, **kwargs) -> Union[ee.Image, ee.ImageCollection]:
    """MSI from Sentinel-2. Formula: B11 / B8"""
    calc = lambda img: img.expression('B11 / B8',
        {'B11': img.select('B11'), 'B8': img.select('B8')}).rename('MSI').copyProperties(img, ['system:time_start'])
    return _get_s2_derived_product(start, end, roi, ['B11', 'B8'], 'MSI', calc, **kwargs)

def get_osavi(start: str, end: str, roi: ee.Geometry, **kwargs) -> Union[ee.Image, ee.ImageCollection]:
    """OSAVI from Sentinel-2. Formula: (B8 - B4) / (B8 + B4 + 0.16)"""
    calc = lambda img: img.expression('((B8 - B4) / (B8 + B4 + 0.16)) * 1.16',
        {'B8': img.select('B8'), 'B4': img.select('B4')}).rename('OSAVI').copyProperties(img, ['system:time_start'])
    return _get_s2_derived_product(start, end, roi, ['B8', 'B4'], 'OSAVI', calc, **kwargs)

def get_gndvi(start: str, end: str, roi: ee.Geometry, **kwargs) -> Union[ee.Image, ee.ImageCollection]:
    """GNDVI from Sentinel-2. Formula: (B8 - B3) / (B8 + B3)"""
    calc = lambda img: img.normalizedDifference(['B8', 'B3']).rename('GNDVI').copyProperties(img, ['system:time_start'])
    return _get_s2_derived_product(start, end, roi, ['B8', 'B3'], 'GNDVI', calc, **kwargs)

def get_rvi(start: str, end: str, roi: ee.Geometry, **kwargs) -> Union[ee.Image, ee.ImageCollection]:
    """RVI from Sentinel-2. Formula: B8 / B4"""
    calc = lambda img: img.expression('B8 / B4',
        {'B8': img.select('B8'), 'B4': img.select('B4')}).rename('RVI').copyProperties(img, ['system:time_start'])
    return _get_s2_derived_product(start, end, roi, ['B8', 'B4'], 'RVI', calc, **kwargs)

# ----- OTHER GENERIC DATA FUNCTIONS -----

def get_precipitation(start: str, end: str, roi: ee.Geometry, **kwargs) -> Union[ee.Image, ee.ImageCollection]:
    """Mean daily precipitation (mm) from CHIRPS."""
    return _get_generic_product('UCSB-CHG/CHIRPS/DAILY', 'precipitation', start, end, roi, **kwargs)

def get_land_surface_temperature(start: str, end: str, roi: ee.Geometry, **kwargs) -> Union[ee.Image, ee.ImageCollection]:
    """Mean daytime LST (°C) from MODIS."""
    process = lambda img: img.select('LST_Day_1km').multiply(0.02).subtract(273.15).rename('LST_C').copyProperties(img, ['system:time_start'])
    return _get_generic_product('MODIS/061/MOD11A1', 'LST_C', start, end, roi, processing_function=process, **kwargs)

def get_humidity(start: str, end: str, roi: ee.Geometry, **kwargs) -> Union[ee.Image, ee.ImageCollection]:
    """Mean relative humidity (%) from ERA5-Land."""
    def process(img):
        rh = img.expression('100 * exp((17.625 * (Td - 273.15)) / (243.04 + (Td - 273.15))) / exp((17.625 * (T - 273.15)) / (243.04 + (T - 273.15)))',
            {'Td': img.select('dewpoint_temperature_2m'), 'T': img.select('temperature_2m')}).rename('RH')
        return rh.copyProperties(img, ['system:time_start'])
    return _get_generic_product('ECMWF/ERA5_LAND/HOURLY', 'RH', start, end, roi, processing_function=process, **kwargs)

def get_irradiance(start: str, end: str, roi: ee.Geometry, **kwargs) -> Union[ee.Image, ee.ImageCollection]:
    """Mean surface net solar radiation (J/m²) from ERA5-Land."""
    return _get_generic_product('ECMWF/ERA5_LAND/HOURLY', 'surface_net_solar_radiation', start, end, roi, **kwargs)

def get_evapotranspiration(start: str, end: str, roi: ee.Geometry, **kwargs) -> Union[ee.Image, ee.ImageCollection]:
    """Mean Evapotranspiration from MODIS."""
    return _get_generic_product('MODIS/061/MOD16A2', 'ET', start, end, roi, **kwargs)

def get_simulated_hyperspectral(start: str, end: str, roi: ee.Geometry, **kwargs) -> Union[ee.Image, ee.ImageCollection]:
    """Cloud-filtered median composite of 9 Sentinel-2 bands."""
    bands = ['B2','B3','B4','B5','B6','B7','B8A','B11','B12']
    calc = lambda img: img.select(bands) # No calculation needed, just return the bands
    return _get_s2_derived_product(start, end, roi, bands, bands, calc, reducer=ee.Reducer.median(), **kwargs)

# -----------------------------------------------------------------------------
# 5. SOIL GRID LAYERS
# -----------------------------------------------------------------------------

def get_soil_moisture(start: str, end: str, roi: ee.Geometry, **kwargs) -> Union[ee.Image, ee.ImageCollection]:
    """Mean soil moisture (0–10 cm) from FLDAS/NOAH."""
    return _get_generic_product('NASA/FLDAS/NOAH01/C/GL/M/V001', 'SoilMoi00_10cm_tavg', start, end, roi, **kwargs)

def get_soil_property(key: str, roi: ee.Geometry) -> ee.Image:
    """Generic loader for static SoilGrids properties."""
    asset, band = SOIL_LAYERS[key]
    return ee.Image(asset).select(band).clip(roi)

def get_soil_texture(roi: ee.Geometry) -> ee.Image:
    """Concatenates clay, silt & sand fractions into one 3-band image."""
    clay = get_soil_property('soil_texture_clay', roi)
    silt = get_soil_property('soil_texture_silt', roi)
    sand = get_soil_property('soil_texture_sand', roi)
    return ee.Image.cat([clay, silt, sand]).rename(['clay', 'silt', 'sand'])

# -----------------------------------------------------------------------------
# 6. MAIN EXECUTION 
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fetch Geospatial Data from GEE")
    parser.add_argument('--start_date', type=str, default=DEFAULT_START, help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, default=DEFAULT_END, help='End date in YYYY-MM-DD format')
    args = parser.parse_args()

    roi_ee = ee.Geometry.BBox(*DEFAULT_ROI_BBOX)

    logging.info("--- Fetching Mean NDRE Image (Newly Added) ---")
    ndre_image = safe_execute(get_ndre, start=args.start_date, end=args.end_date, roi=roi_ee, return_collection=False)
    if ndre_image:
        logging.info(f"Successfully generated NDRE image with bands: {ndre_image.bandNames().getInfo()}")

    logging.info("\n--- Fetching MSI Time-Series Collection (Newly Added) ---")
    msi_collection = safe_execute(get_msi, start=args.start_date, end=args.end_date, roi=roi_ee, return_collection=True)
    if msi_collection:
        logging.info(f"Successfully generated MSI collection with {msi_collection.size().getInfo()} images.")


    logging.info("--- Fetching Mean OSAVI Image (Newly Added) ---")
    osavi_image = safe_execute(get_osavi, start=args.start_date, end=args.end_date, roi=roi_ee, return_collection=False)
    if osavi_image:
        logging.info(f"Successfully generated OSAVI image with bands: {osavi_image.bandNames().getInfo()}")

    logging.info("---Fetching RVI Time-Series Collection (Newly Added) ---")
    rvi_collection = safe_execute(get_rvi, start=args.start_date, end=args.end_date, roi=roi_ee, return_collection=True)
    if rvi_collection:
        logging.info(f"Successfully generated RVI collection with {rvi_collection.size().getInfo()} images.")

