import ee
import geemap
import time
import random
import datetime

# Initialize Earth Engine
try:
    ee.Initialize(project="winged-tenure-464005-p9")
except Exception as e:
    print("EE not initialized. Attempting to authenticate...")
    ee.Authenticate()
    ee.Initialize()

# --- PARAMETERS ---
OUTPUT_DRIVE_FOLDER = 'GEE_Nutrient_Dataset'
OUTPUT_FILE_PREFIX = 'nutrient_training'
START_YEAR = 2015
END_YEAR = 2024
SAMPLES_PER_YEAR_TOTAL = 50000  # Increased for better coverage
SCALE = 10  # meters

# Crops and their approximate growing seasons (using Southern Hemisphere seasons)
# Months are based on Northern Hemisphere, so convert to Southern Hemisphere dates
# Note: This is still an approximation, a phenology model would be better.
CROP_SEASONS = {
    "Maize": {"start_month": 11, "end_month": 4},
    "Potato": {"start_month": 9, "end_month": 2},
    "Wheat": {"start_month": 5, "end_month": 10},
    "Barley": {"start_month": 5, "end_month": 9},
    "Sorghum": {"start_month": 11, "end_month": 4},
    "Beans": {"start_month": 11, "end_month": 3},
    "Green Beans": {"start_month": 11, "end_month": 3},
    "Cabbage": {"start_month": 3, "end_month": 9},
    "Tomato": {"start_month": 10, "end_month": 3},
    "Oats": {"start_month": 5, "end_month": 9},
    "Sweet Potato": {"start_month": 10, "end_month": 3}
}

CROPS=list(CROP_SEASONS.keys())
GROWTH_STAGES = ["Vegetative", "Flowering", "Fruiting", "Maturity"]
# Thresholds for each crop and stage (NDVI, NDRE)
# These are still hardcoded but the logic for their application is improved.
THRESHOLDS = {
    "Maize": {"Vegetative": (0.55, 0.35), "Flowering": (0.65, 0.40), "Fruiting": (0.60, 0.38)},
    "Potato": {"Vegetative": (0.50, 0.32), "Flowering": (0.58, 0.35), "Fruiting": (0.55, 0.33)},
    "Wheat": {"Vegetative": (0.50, 0.30), "Flowering": (0.60, 0.35), "Maturity": (0.45, 0.28)},
    "Barley": {"Vegetative": (0.50, 0.30), "Flowering": (0.58, 0.34), "Maturity": (0.44, 0.27)},
    "Sorghum": {"Vegetative": (0.53, 0.34), "Flowering": (0.62, 0.37), "Fruiting": (0.57, 0.35)},
    "Beans": {"Vegetative": (0.52, 0.33), "Flowering": (0.60, 0.36)},
    "Green Beans": {"Vegetative": (0.52, 0.33), "Flowering": (0.60, 0.36)},
    "Cabbage": {"Vegetative": (0.48, 0.30), "Fruiting": (0.55, 0.33)},
    "Tomato": {"Vegetative": (0.50, 0.31), "Flowering": (0.57, 0.34), "Fruiting": (0.53, 0.32)},
    "Oats": {"Vegetative": (0.48, 0.30), "Flowering": (0.56, 0.33)},
    "Sweet Potato": {"Vegetative": (0.49, 0.30), "Fruiting": (0.54, 0.32)}
}

# --- GEE DATASETS ---
S2_COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED'
SOILGRIDS = ee.Image("projects/soilgrids-isric/soc_mean").rename('soil_soc')
CHIRPS = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
SRTM = ee.Image("USGS/SRTMGL1_003")

# --- FUNCTIONS ---
def mask_s2_clouds(image):
    """Masks out clouds and cirrus from a Sentinel-2 image."""
    qa = image.select('QA60')
    cloud_bit_mask = 1 << 10
    cirrus_bit_mask = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
    return image.updateMask(mask).divide(10000)

def add_indices(img):
    """Calculates and adds a set of vegetation indices to an image."""
    nir = img.select('B8')
    red = img.select('B4')
    rededge = img.select('B5')
    green = img.select('B3')
    
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    ndre = nir.subtract(rededge).divide(nir.add(rededge)).rename('NDRE')
    savi = nir.subtract(red).multiply(1.5).divide(nir.add(red).add(0.5)).rename('SAVI')
    mcari = (rededge.subtract(red).subtract(0.2 * (red.subtract(green)))).multiply(rededge.divide(red)).rename('MCARI')
    
    return img.addBands([ndvi, ndre, savi, mcari])

def get_composite_for_period(start_date, end_date):
    """Generates a Sentinel-2 median composite for a specified period."""
    s2_collection = ee.ImageCollection(S2_COLLECTION) \
        .filterDate(start_date, end_date) \
        .filterBounds(lesotho) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70)) \
        .map(mask_s2_clouds) \
        .map(add_indices)
    
    return s2_collection.median().clip(lesotho)

def get_features_for_period(composite, start_date, end_date):
    """Adds soil, rainfall, and terrain features to a composite image."""
    # Add SoilGrids features
    composite = composite.addBands(SOILGRIDS)
    
    # Add rainfall (CHIRPS)
    rainfall = CHIRPS.filterDate(start_date, end_date).filterBounds(lesotho).sum().rename('rainfall')
    composite = composite.addBands(rainfall)
    
    # Add terrain features (SRTM)
    elevation = SRTM.select('elevation').rename('elevation')
    composite = composite.addBands(elevation)
    
    return composite

def generate_samples(params):
    """
    Generates a FeatureCollection of samples for a given crop, year, and stage.
    This function is designed to be mapped over a list of parameters.
    """
    year = ee.Number(params.get('year'))
    crop = ee.String(params.get('crop'))
    stage = ee.String(params.get('stage'))
    
    # Get crop season dates
    season_info = CROP_SEASONS[crop.getInfo()] # Use .getInfo() for client-side access
    start_date = ee.Date.fromYMD(year, season_info['start_month'], 1)
    # Handle cross-year seasons
    end_year = year.add(ee.Number(1) if season_info['end_month'] < season_info['start_month'] else 0)
    end_date = ee.Date.fromYMD(end_year, season_info['end_month'], 28)
    
    composite = get_composite_for_period(start_date, end_date)
    composite = get_features_for_period(composite, start_date, end_date)
    
    # Get thresholds for the current crop and stage
    ndvi_thresh, ndre_thresh = THRESHOLDS.get(crop.getInfo(), {}).get(stage.getInfo(), (0.5, 0.3))
    
    # Create the binary label based on thresholds
    healthy_cond = composite.select('NDVI').gte(ndvi_thresh).And(composite.select('NDRE').gte(ndre_thresh))
    label = ee.Image.constant(0).where(healthy_cond, 1).rename('label')
    composite = composite.addBands(label)
    
    # Sample from the image
    sample = composite.select(['NDVI', 'NDRE', 'SAVI', 'MCARI', 'soil_soc', 'rainfall', 'elevation', 'label']) \
        .stratifiedSample(
            numPoints=SAMPLES_PER_YEAR_TOTAL // (len(CROPS) * len(GROWTH_STAGES)),
            classBand='label',
            region=lesotho,
            scale=SCALE,
            geometries=True,
            seed=year.getInfo()  # Use .getInfo() for seed
        )
    
    # Add metadata
    return sample.map(lambda f: f.set({'year': year, 'crop': crop, 'stage': stage}))

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    # Load ROI
    print("Loading region of interest...")
    countries = ee.FeatureCollection('FAO/GAUL/2015/level0')
    lesotho = countries.filter(ee.Filter.eq('ADM0_NAME', 'Lesotho')).geometry()
    
    # Create a list of parameters for parallel processing
    param_list = []
    for year in range(START_YEAR, END_YEAR + 1):
        for crop in CROP_SEASONS.keys():
            # A more sophisticated approach could use GDD to determine stages
            # For simplicity, we still randomly pick, but now it's part of the
            # server-side mapping, not the client-side loop.
            stages_for_crop = list(THRESHOLDS.get(crop, {}).keys())
            if not stages_for_crop:
                print(f"Skipping {crop} as it has no defined thresholds.")
                continue
            
            for stage in stages_for_crop:
                param_list.append(ee.Dictionary({
                    'year': ee.Number(year),
                    'crop': ee.String(crop),
                    'stage': ee.String(stage)
                }))

    print(f"Generating samples for {len(param_list)} unique crop-year-stage combinations...")
    
    # Use map to run the generation in parallel on GEE's servers
    params_collection = ee.FeatureCollection(param_list)
    all_samples_list = params_collection.map(generate_samples).flatten()
    
    # Export the final dataset
    print(f"Exporting dataset to Google Drive: {OUTPUT_DRIVE_FOLDER}...")
    
    # Ensure all_samples_list is not empty before starting the task
    if all_samples_list.size().getInfo() > 0:
        export_task = ee.batch.Export.table.toDrive(
            collection=all_samples_list,
            description=f'{OUTPUT_FILE_PREFIX}_{START_YEAR}_{END_YEAR}',
            folder=OUTPUT_DRIVE_FOLDER,
            fileNamePrefix=f'{OUTPUT_FILE_PREFIX}_{START_YEAR}_{END_YEAR}',
            fileFormat='CSV'
        )
        export_task.start()
        
        # Monitor the task status
        print('Export task started. Task ID:', export_task.id)
        print('Monitoring task. This may take a while...')
        
        while export_task.active():
            print('Task is still running...', datetime.datetime.now())
            time.sleep(60) # Wait for 1 minute
            
        status = export_task.status()
        if status['state'] == 'COMPLETED':
            print('Export completed successfully!')
        else:
            print(f'Export failed with state: {status["state"]}')
            print('Error message:', status['error_message'])
            
    else:
        print("No samples were generated. Export task was not started.")