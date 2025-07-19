import argparse
import logging
from datetime import datetime
from typing import Dict, List

import ee

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------

EE_PROJECT        = 'winged-tenure-464005-p9'
DEFAULT_START     = '2024-07-01'
DEFAULT_END       = '2024-07-10'
DEFAULT_ROI_BBOX  = [26.999, -30.5, 29.5, -28.5]  # [minLon, minLat, maxLon, maxLat]

SOIL_LAYERS = {
    'soil_texture_clay': ('projects/soilgrids-isric/clay_mean', 'clay_0-5cm_mean'),
    'soil_texture_silt': ('projects/soilgrids-isric/silt_mean', 'silt_0-5cm_mean'),
    'soil_texture_sand': ('projects/soilgrids-isric/sand_mean', 'sand_0-5cm_mean'),
    'ph': ('projects/soilgrids-isric/phh2o_mean', 'phh2o_0-5cm_mean'),
    'ocd': ('projects/soilgrids-isric/ocd_mean', 'ocd_0-30cm_mean'),
    'cec': ('projects/soilgrids-isric/cec_mean', 'cec_0-30cm_mean'),
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
    """
    return col.select(bands).mean().clip(roi)

def _log_date_coverage(col: ee.ImageCollection, label: str):
    """
    Logs the date coverage of an ImageCollection if not empty.
    """
    size = col.size().getInfo()
    if size == 0:
        logging.warning(f"{label}: Empty collection")
    else:
        first_date = col.sort('system:time_start').first().date().format('YYYY-MM-dd').getInfo()
        last_date = col.sort('system:time_start', False).first().date().format('YYYY-MM-dd').getInfo()
        logging.info(f"{label}: {size} images from {first_date} to {last_date}")

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
    max_expansion_days: int = 30
) -> ee.Image:
    """
    NDVI from Sentinel-2 SR Harmonized, expand date range if no images.
    """
    base = _prepare_collection(
        'COPERNICUS/S2_SR_HARMONIZED', start, end, roi
    ).select(['B8', 'B4'])

    buffered = _prepare_collection(
        'COPERNICUS/S2_SR_HARMONIZED',
        (datetime.strptime(start, '%Y-%m-%d')
                 .date()
                 .strftime('%Y-%m-%d')),
        (datetime.strptime(end, '%Y-%m-%d')
                 .date()
                 .strftime('%Y-%m-%d')),
        roi
    ).filterDate(
        ee.Date(start).advance(-max_expansion_days, 'day'),
        ee.Date(end).advance(max_expansion_days, 'day')
    ).select(['B8', 'B4'])

    col = ee.ImageCollection(
        ee.Algorithms.If(base.size().gt(0), base, buffered)
    )
    _log_date_coverage(col, 'NDVI')
    return col.map(lambda img:
        img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ).mean().clip(roi)

def get_precipitation(start: str, end: str, roi: ee.Geometry) -> ee.Image:
    """
    Mean daily precipitation (mm) from CHIRPS.
    """
    col = _prepare_collection('UCSB-CHG/CHIRPS/DAILY', start, end, roi)
    _log_date_coverage(col, 'Precipitation')
    return _mean_clip(col, ['precipitation'], roi)

def get_land_surface_temperature(start: str, end: str, roi: ee.Geometry) -> ee.Image:
    """
    Mean daytime LST (°C) from MODIS.
    """
    col = _prepare_collection('MODIS/061/MOD11A1', start, end, roi)
    _log_date_coverage(col, 'Land Surface Temperature')
    return (
        col.select('LST_Day_1km')
           .mean()
           .multiply(0.02)
           .subtract(273.15)
           .rename('LST_C')
           .clip(roi)
    )

def get_humidity(start: str, end: str, roi: ee.Geometry) -> ee.Image:
    """
    Mean relative humidity (%) from ERA5-Land.
    """
    col = _prepare_collection('ECMWF/ERA5_LAND/HOURLY', start, end, roi) \
            .select(['dewpoint_temperature_2m', 'temperature_2m'])
    _log_date_coverage(col, 'Relative Humidity')

    def compute_rh(img):
        Td = img.select('dewpoint_temperature_2m').subtract(273.15)
        T  = img.select('temperature_2m').subtract(273.15)
        num = Td.multiply(17.625).divide(Td.add(243.04)).exp()
        den = T.multiply(17.625).divide(T.add(243.04)).exp()
        return num.divide(den).multiply(100).rename('RH') \
                  .copyProperties(img, img.propertyNames())

    return col.map(compute_rh).mean().clip(roi)

def get_irradiance(start: str, end: str, roi: ee.Geometry) -> ee.Image:
    """
    Mean surface net solar radiation (J/m²) from ERA5-Land.
    """
    col = _prepare_collection('ECMWF/ERA5_LAND/HOURLY', start, end, roi)
    _log_date_coverage(col, 'Irradiance')
    return _mean_clip(col, ['surface_net_solar_radiation'], roi)
import ee

def get_evapotranspiration(start: str, end: str, roi: ee.Geometry) -> ee.Image:
    """
    Fetches mean actual evapotranspiration (ET) from MODIS MOD16A2GF.
    Values are in millimeters (kg/m²) after applying scale factor (0.1).
    
    Args:
        start (str): Start date in 'YYYY-MM-DD'.
        end (str): End date in 'YYYY-MM-DD'.
        roi (ee.Geometry): Region of interest.

    Returns:
        ee.Image: Mean ET image scaled and clipped to ROI.
    """
    dataset_id = 'MODIS/061/MOD16A2GF'
    band = 'ET'
    scale_factor = 0.1

    # Load, filter and clip the image collection
    col = _prepare_collection(dataset_id, start, end, roi)
    _log_date_coverage(col, 'Evapotranspiration')

    # Compute mean, apply scale factor, clip, and rename
    mean_et = _mean_clip(col, [band], roi).multiply(scale_factor).rename('ET')
    return mean_et


def get_simulated_hyperspectral(start: str, end: str, roi: ee.Geometry) -> ee.Image:
    """
    Cloud-filtered median composite of 9 Sentinel-2 bands to mimic hyperspectral.
    """
    bands = ['B2','B3','B4','B5','B6','B7','B8A','B11','B12']
    col = (
        _prepare_collection('COPERNICUS/S2_SR_HARMONIZED', start, end, roi)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    )
    _log_date_coverage(col, 'Simulated Hyperspectral')
    return col.median().select(bands).clip(roi)

# -----------------------------------------------------------------------------
# 5. SOIL GRID LAYERS
# -----------------------------------------------------------------------------

def get_soil_moisture(start: str, end: str, roi: ee.Geometry) -> ee.Image:
    """
    Mean soil moisture (0–10 cm) from FLDAS/NOAH.
    """
    col = ee.ImageCollection('NASA/FLDAS/NOAH01/C/GL/M/V001') \
        .filterDate(start, end) \
        .filterBounds(roi)
    _log_date_coverage(col, 'Soil Moisture')
    return _mean_clip(col, ['SoilMoi00_10cm_tavg'], roi)

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
    clay = get_soil_property('soil_texture_clay', roi)
    silt = get_soil_property('soil_texture_silt', roi)
    sand = get_soil_property('soil_texture_sand', roi)
    return ee.Image.cat([clay, silt, sand]).rename(['clay','silt','sand'])
