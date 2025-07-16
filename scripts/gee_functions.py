#!/usr/bin/env python3
"""
gee_workflow.py

A modular, maintainable GEE functions script with:
  - Centralized configuration
  - DRY date & bounds filtering
  - Docstrings & type hints
  - Static soil layers factory
  - Dynamic dataset registry
  - Error handling & logging
  - CLI via argparse
"""

import argparse
import logging
from typing import Dict, List

import ee

# -----------------------------------------------------------------------------
# 1. CONFIGURATION
# -----------------------------------------------------------------------------

EE_PROJECT        = 'winged-tenure-464005-p9'
DEFAULT_START     = '2024-07-01'
DEFAULT_END       = '2024-07-10'
DEFAULT_ROI_BBOX  = [26.999, -30.5, 29.5, -28.5]  # [minLon, minLat, maxLon, maxLat]

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

def safe_execute(func, **kwargs):
    """
    Call a function, catch exceptions, log errors, and return None on failure.
    """
    try:
        return func(**kwargs)
    except Exception as e:
        logging.error(f"Error in {func.__name__} with args={kwargs}: {e}")
        return None

# -----------------------------------------------------------------------------
# 4. BASIC ENVIRONMENTAL INDICES
# -----------------------------------------------------------------------------

def get_ndvi(start: str, end: str, roi: ee.Geometry) -> ee.Image:
    """
    Mean NDVI from Sentinel-2.
    """
    col = _prepare_collection('COPERNICUS/S2_SR_HARMONIZED', start, end, roi)
    return (
        col
          .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI'))
          .mean()
          .clip(roi)
    )

def get_precipitation(start: str, end: str, roi: ee.Geometry) -> ee.Image:
    """
    Mean daily precipitation (mm) from CHIRPS.
    """
    col = _prepare_collection('UCSB-CHG/CHIRPS/DAILY', start, end, roi)
    return col.select('precipitation').mean().clip(roi)

def get_land_surface_temperature(start: str, end: str, roi: ee.Geometry) -> ee.Image:
    """
    Mean daytime LST (°C) from MODIS.
    """
    col = _prepare_collection('MODIS/061/MOD11A1', start, end, roi)
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

    def compute_rh(img):
        Td = img.select('dewpoint_temperature_2m').subtract(273.15)
        T  = img.select('temperature_2m').subtract(273.15)
        num = Td.multiply(17.625).divide(Td.add(243.04)).exp()
        den = T.multiply(17.625).divide(T.add(243.04)).exp()
        return num.divide(den).multiply(100).rename('RH').copyProperties(img, img.propertyNames())

    return col.map(compute_rh).mean().clip(roi)

def get_irradiance(start: str, end: str, roi: ee.Geometry) -> ee.Image:
    """
    Mean surface net solar radiation (J/m²) from ERA5-Land.
    """
    col = _prepare_collection('ECMWF/ERA5_LAND/HOURLY', start, end, roi)
    return col.select('surface_net_solar_radiation').mean().clip(roi)

def get_evapotranspiration(start: str, end: str, roi: ee.Geometry) -> ee.Image:
    """
    Mean evapotranspiration (kg/m²) from MOD16A2GF.
    """
    col = _prepare_collection('MODIS/061/MOD16A2GF', start, end, roi)
    return col.select('ET').mean().clip(roi)

def get_simulated_hyperspectral(start: str, end: str, roi: ee.Geometry) -> ee.Image:
    """
    Cloud‐filtered median composite of 9 Sentinel-2 bands to mimic hyperspectral.
    """
    bands = ['B2','B3','B4','B5','B6','B7','B8A','B11','B12']
    col = (
        _prepare_collection('COPERNICUS/S2_SR_HARMONIZED', start, end, roi)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    )
    return col.median().select(bands).clip(roi)

# -----------------------------------------------------------------------------
# 5. SOIL GRID LAYERS
# -----------------------------------------------------------------------------

SOIL_LAYERS = {
    'soil_organic_matter': ('projects/soilgrids-isric/soc_mean',     'soc_0-5cm_mean'),
    'soil_ph':             ('projects/soilgrids-isric/phh2o_mean',  'phh2o_0-5cm_mean'),
    'soil_cec':            ('projects/soilgrids-isric/cec_mean',    'cec_0-5cm_mean'),
    'soil_nitrogen':       ('projects/soilgrids-isric/nitrogen_mean','nitrogen_0-5cm_mean'),
}

def get_soil_property(key: str, roi: ee.Geometry) -> ee.Image:
    """
    Generic loader for SoilGrids properties.
    key must be one of SOIL_LAYERS.
    """
    asset, band = SOIL_LAYERS[key]
    return ee.Image(asset).select(band).clip(roi)

def get_soil_texture(roi: ee.Geometry) -> ee.Image:
    """
    Concatenate clay, silt & sand fractions into one 3‐band image.
    """
    clay = get_soil_property('soil_texture_clay', roi)  # if defined in SOIL_LAYERS
    silt = get_soil_property('soil_texture_silt', roi)
    sand= get_soil_property('soil_texture_sand', roi)
    return ee.Image.cat([clay, silt, sand]).rename(['clay','silt','sand'])

# -----------------------------------------------------------------------------
# 6. DYNAMIC DATASET REGISTRY
# -----------------------------------------------------------------------------

# Map display name → (function reference, list of its required params)
DATASETS = {
    'NDVI':                      (get_ndvi,               ['start','end','roi']),
    'Precipitation':             (get_precipitation,      ['start','end','roi']),
    'Land Surface Temperature':  (get_land_surface_temperature, ['start','end','roi']),
    'Relative Humidity':         (get_humidity,           ['start','end','roi']),
    'Irradiance':                (get_irradiance,         ['start','end','roi']),
    'Evapotranspiration':        (get_evapotranspiration, ['start','end','roi']),
    'Simulated Hyperspectral':   (get_simulated_hyperspectral, ['start','end','roi']),
    'Soil Organic Matter':       (lambda start,end,roi: get_soil_property('soil_organic_matter', roi), ['start','end','roi']),
    'Soil pH':                   (lambda start,end,roi: get_soil_property('soil_ph',           roi), ['start','end','roi']),
    'Soil CEC':                  (lambda start,end,roi: get_soil_property('soil_cec',          roi), ['start','end','roi']),
    'Soil Nitrogen':             (lambda start,end,roi: get_soil_property('soil_nitrogen',     roi), ['start','end','roi']),
    # add others as needed
}

def build_collections(
    start: str, end: str, roi: ee.Geometry
) -> Dict[str, ee.Image]:
    """
    Build all requested collections dynamically, handling errors gracefully.
    """
    output: Dict[str, ee.Image] = {}
    param_map = {
        'start': start,
        'end':   end,
        'roi':   roi
    }

    for name, (func, param_keys) in DATASETS.items():
        # Pick only the keys that this function actually wants
        kwargs = {key: param_map[key] for key in param_keys}
        output[name] = safe_execute(func, **kwargs)

    return output

# -----------------------------------------------------------------------------
# 7. COMMAND‐LINE INTERFACE
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Generate GEE ImageCollections for a date range & ROI'
    )
    p.add_argument('--start', default=DEFAULT_START, help='Start date YYYY-MM-DD')
    p.add_argument('--end',   default=DEFAULT_END,   help='End date   YYYY-MM-DD')
    p.add_argument(
        '--bbox', nargs=4, type=float, default=DEFAULT_ROI_BBOX,
        metavar=('MIN_LON','MIN_LAT','MAX_LON','MAX_LAT'),
        help='Bounding box for ROI: minLon minLat maxLon maxLat'
    )
    return p.parse_args()

def main():
    args = parse_args()

    # Build ROI polygon
    min_lon, min_lat, max_lon, max_lat = args.bbox
    roi = ee.Geometry.Polygon([
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat]
    ])

    logging.info(f"Building collections for {args.start} → {args.end}")
    collections = build_collections(args.start, args.end, roi)

    for name, img in collections.items():
        if img:
            bands = img.bandNames().getInfo()
            logging.info(f"{name}: bands={bands}")
        else:
            logging.warning(f"{name}: generation failed.")

if __name__ == '__main__':
    main()
