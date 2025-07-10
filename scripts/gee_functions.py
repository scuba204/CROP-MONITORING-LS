import ee

ee.Initialize(project='winged-tenure-464005-p9')

roi = ee.Geometry.Polygon([
    [
        [26.999, -30.5],
        [29.5, -30.5],
        [29.5, -28.5],
        [26.999, -28.5],
        [26.999, -30.5]
    ]
])

def get_ndvi(start_date, end_date, roi=roi):
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI'))

    return collection.median().clip(roi)

def get_soil_moisture(start_date, end_date, roi=roi):
    sm = ee.ImageCollection('NASA/FLDAS/NOAH01/C/GL/M') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .select('SoilMoi0_10cm_inst') \
        .mean().clip(roi)
    return sm

def get_precipitation(start_date, end_date, roi=roi):
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY") \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .select("precipitation") \
        .sum().clip(roi)
    return chirps


def get_land_surface_temperature(start_date, end_date, roi=roi):
    lst = ee.ImageCollection("MODIS/006/MOD11A2") \
        .filterDate(start_date, end_date) \
        .filterBounds(roi) \
        .select("LST_Day_1km") \
        .mean() \
        .multiply(0.02) \
        .subtract(273.15) \
        .clip(roi)  # Convert from Kelvin to Celsius
    return lst


def get_humidity(start_date, end_date, roi=roi):
    humidity = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .select('dewpoint_temperature_2m') \
        .mean().clip(roi)
    return humidity

def get_irradiance(start_date, end_date, roi=roi):
    irradiance = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
        .filterDate(start_date, end_date) \
        .filterBounds(roi) \
        .select("surface_net_solar_radiation") \
        .mean().clip(roi)
    return irradiance

def get_simulated_hyperspectral(start_date, end_date, roi=roi):
    image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filterBounds(roi) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median() \
        .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']) \
        .clip(roi)
    return image

def get_soil_organic_matter(roi=roi):
    return ee.Image('OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02') \
        .select('oc') \
        .clip(roi)

def get_soil_ph(roi=roi):
    return ee.Image('OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02') \
        .select('phh2o') \
        .clip(roi)

def get_soil_texture(roi=roi):
    return ee.Image('OpenLandMap/SOL/SOL_TEXTURE-USDA-TT_M/v02') \
        .select('usda_texture') \
        .clip(roi)

def get_evapotranspiration(start_date, end_date, roi=roi):
    et = ee.ImageCollection('MODIS/006/MOD16A2') \
        .filterDate(start_date, end_date) \
        .filterBounds(roi) \
        .select('ET') \
        .mean().clip(roi)
    return et


def get_soil_nitrogen(roi):
    return ee.Image("projects/soilgrids-isric/ntd_mean").clip(roi)

def get_soil_phosphorus(roi):
    return ee.Image("projects/soilgrids-isric/phh1_mean").clip(roi)

def get_soil_potassium(roi):
    # SoilGrids does not provide K directly; use CEC as proxy for K-holding capacity
    return ee.Image("projects/soilgrids-isric/cec_mean").clip(roi)

