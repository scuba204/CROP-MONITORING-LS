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

start = '2024-07-01'
end = '2024-07-10'

# Test each function
def get_ndvi(start, end, geom):
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start, end) \
        .filterBounds(geom) \
        .map(lambda img: img.normalizedDifference(['B8', 'B4']).rename('NDVI'))
    ndvi = collection.mean().clip(geom)
    return ndvi

def get_soil_moisture(start, end, geom):
    collection = ee.ImageCollection('NASA/FLDAS/NOAH01/C/GL/M/V001') \
        .filterDate('2024-07-01', '2024-07-31') \
        .filterBounds(geom)
    soil_moisture = collection.select('SoilMoi00_10cm_tavg'
).mean().clip(geom)
    return soil_moisture

def get_precipitation(start, end, geom):
    collection = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY') \
        .filterDate(start, end) \
        .filterBounds(geom)
    precipitation = collection.select('precipitation').mean().clip(geom)
    return precipitation

def get_land_surface_temperature(start, end, geom):
    collection = ee.ImageCollection('MODIS/061/MOD11A1') \
        .filterDate(start, end) \
        .filterBounds(geom)
    # MODIS LST is in Kelvin * 0.02, so convert to Celsius
    lst = collection.select('LST_Day_1km').mean().multiply(0.02).subtract(273.15).clip(geom)
    return lst

def get_humidity(start, end, geom):
    collection = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY') \
        .filterDate(start, end) \
        .filterBounds(geom) \
        .select(['dewpoint_temperature_2m', 'temperature_2m'])

    def compute_rh(image):
        Td = image.select('dewpoint_temperature_2m').subtract(273.15)
        T = image.select('temperature_2m').subtract(273.15)
        numerator = Td.multiply(17.625).divide(Td.add(243.04)).exp()
        denominator = T.multiply(17.625).divide(T.add(243.04)).exp()
        rh = numerator.divide(denominator).multiply(100).rename('RH')
        return rh.copyProperties(image, image.propertyNames())

    rh_collection = collection.map(compute_rh)
    rh_image = rh_collection.mean().clip(geom)
    return rh_image



def get_irradiance(start_date, end_date, roi=roi):
    irradiance = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY") \
        .filterDate(start_date, end_date) \
        .filterBounds(roi) \
        .select("surface_net_solar_radiation") \
        .mean().clip(roi)
    return irradiance


ndvi_img = get_ndvi(start, end, roi)
soil_moisture_img = get_soil_moisture(start, end, roi)
precip_img = get_precipitation(start, end, roi)
lst_img = get_land_surface_temperature(start, end, roi)
humidity_img = get_humidity(start, end, roi)
irradiance_img = get_irradiance(start, end, roi)

# Print type and info
print("NDVI:", type(ndvi_img), ndvi_img.getInfo())
print("Soil Moisture:", type(soil_moisture_img), soil_moisture_img.getInfo())
print("Precipitation:", type(precip_img), precip_img.getInfo())
print("LST:", type(lst_img), lst_img.getInfo())
print("Humidity:", type(humidity_img), humidity_img.getInfo())
print("Irradiance:", type(irradiance_img), irradiance_img.getInfo())