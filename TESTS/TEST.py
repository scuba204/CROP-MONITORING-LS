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

start_date = '2024-07-01'
end_date = '2024-07-10'

def validate_image(img, name):
    print(f"Validating {name}...")
    try:
        bands = img.bandNames().getInfo()
        print(f"  ✅ Bands: {bands}")
        stats = img.reduceRegion(
            reducer=ee.Reducer.first(),
            geometry=roi,
            scale=30,
            maxPixels=1e9
        ).getInfo()
        print(f"  Sample pixel values: {stats}")
        info = img.getInfo()
        print(f"  ℹ️ Image info keys: {list(info.keys())}\n")
    except Exception as e:
        print(f"  ❌ ERROR validating {name}: {e}\n")


# Updated Functions

def get_simulated_hyperspectral(start_date, end_date, roi):
    image = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterDate(start_date, end_date) \
        .filterBounds(roi) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .median() \
        .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']) \
        .clip(roi)
    return image

def get_evapotranspiration(start_date, end_date, roi):
    et = ee.ImageCollection('MODIS/061/MOD16A2GF') \
        .filterDate(start_date, end_date) \
        .filterBounds(roi) \
        .select('ET') \
        .mean().clip(roi)
    return et

def get_soil_organic_matter(roi):
    image = ee.Image("projects/soilgrids-isric/soc_mean")
    return image.select('soc_0-5cm_mean').clip(roi)

def get_soil_ph(roi):
    image = ee.Image("projects/soilgrids-isric/phh2o_mean")
    return image.select('phh2o_0-5cm_mean').clip(roi)

def get_soil_cec(roi):
    image = ee.Image("projects/soilgrids-isric/cec_mean")
    return image.select('cec_0-5cm_mean').clip(roi)

def get_soil_nitrogen(roi):
    image = ee.Image("projects/soilgrids-isric/nitrogen_mean")
    return image.select('nitrogen_0-5cm_mean').clip(roi)

def get_soil_texture(roi):
    clay = ee.Image("projects/soilgrids-isric/clay_mean").select('clay_0-5cm_mean')
    silt = ee.Image("projects/soilgrids-isric/silt_mean").select('silt_0-5cm_mean')
    sand = ee.Image("projects/soilgrids-isric/sand_mean").select('sand_0-5cm_mean')
    return ee.Image.cat([clay, silt, sand]).clip(roi)



datasets = {
    "Simulated Hyperspectral": get_simulated_hyperspectral(start_date, end_date, roi),
    "Evapotranspiration (MOD16A2GF)": get_evapotranspiration(start_date, end_date, roi),
    "Soil Organic Matter (OC)": get_soil_organic_matter(roi),
    "Soil pH": get_soil_ph(roi),
    "Soil Texture": get_soil_texture(roi),
    "Soil CEC": get_soil_cec(roi),
    "Soil Nitrogen (Proxy via OC)": get_soil_nitrogen(roi),
}

for name, img in datasets.items():
    validate_image(img, name)
