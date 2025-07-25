import ee
import logging
import sys
import os

# --- CRITICAL FIX START ---
# Get the absolute path of the directory containing the current script (TEST.py)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the current script's directory (i.e., the project root)
project_root_dir = os.path.abspath(os.path.join(current_script_dir, '..'))

# Add the project root to sys.path if it's not already there.
# Inserting at the beginning gives it priority.
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)
# --- CRITICAL FIX END ---

# --- DEBUGGING LINES (keep for now, then remove if successful) ---
print(f"Current Working Directory (os.getcwd()): {os.getcwd()}")
print("Python System Path (sys.path) AFTER MODIFICATION:")
for p in sys.path:
    print(f"  - {p}")
print("-" * 30) # Separator
# --- DEBUGGING LINES END ---


ee.Initialize(project='winged-tenure-464005-p9')

# This import should now work correctly
from scripts import gee_functions # Import your gee_functions script

# Set up basic logging to see INFO messages from gee_functions
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s',
    level=logging.INFO
)

# Define your Region of Interest
roi = ee.Geometry.Polygon([
    [
        [26.999, -30.5],
        [29.5, -30.5],
        [29.5, -28.5],
        [26.999, -28.5],
        [26.999, -30.5]
    ]
])

# Define date range for temporal indices
start_date = '2025-06-01'
end_date = '2025-06-30'

def validate_image(img_or_col, name, is_collection=False):
    """
    Validates an Earth Engine Image or ImageCollection by trying to get info and sample values.
    Adjusted to handle empty collections gracefully.
    """
    print(f"Validating {name}...")
    try:
        if is_collection:
            size = img_or_col.size().getInfo()
            if size == 0:
                print(f"  ℹ️ Collection is empty: {name}\n")
                return

            first_img = img_or_col.first()
            print(f"  ✅ Collection size: {size}")
            print(f"  ✅ First image in collection properties: {first_img.propertyNames().getInfo()}")
            print(f"  ✅ First image in collection bands: {first_img.bandNames().getInfo()}")

            # Try to get sample pixel values from the first image
            try:
                sample_stats = first_img.reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=roi,
                    scale=30, # Use an appropriate scale for the data
                    maxPixels=1e9
                ).getInfo()
                print(f"  Sample pixel values (first image): {sample_stats}")
            except Exception as sample_e:
                print(f"  ❌ ERROR getting sample pixel values from first image in {name} collection: {sample_e}")
            print("\n")

        else: # It's a single image (mean composite)
            # Check if it's an "empty" image returned by gee_functions for no data
            if 'error' in img_or_col.getInfo():
                print(f"  ℹ️ Image is empty/contains an error: {name}. Error: {img_or_col.getInfo()['error']}\n")
                return

            bands = img_or_col.bandNames().getInfo()
            if not bands:
                print(f"  ℹ️ Image has no bands: {name}. Likely empty or uninitialized.\n")
                return

            print(f"  ✅ Bands: {bands}")
            stats = img_or_col.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=roi,
                scale=30, # Use an appropriate scale for the data
                maxPixels=1e9
            ).getInfo()
            print(f"  Sample pixel values: {stats}")
            print("\n")

    except Exception as e:
        print(f"  ❌ ERROR validating {name}: {e}\n")


print(f"--- Validating Mean Images for ROI {roi.bounds().getInfo()} from {start_date} to {end_date} ---")

# Vegetation Indices (Sentinel-2 based)
ndvi_mean = gee_functions.safe_execute(gee_functions.get_ndvi,
                                       start=start_date, end=end_date, roi=roi,
                                       return_collection=False)
validate_image(ndvi_mean, "NDVI (Mean)")

savi_mean = gee_functions.safe_execute(gee_functions.get_savi,
                                       start=start_date, end=end_date, roi=roi,
                                       L=0.5, # Specify L
                                       return_collection=False)
validate_image(savi_mean, "SAVI (Mean)")

evi_mean = gee_functions.safe_execute(gee_functions.get_evi,
                                      start=start_date, end=end_date, roi=roi,
                                      return_collection=False)
validate_image(evi_mean, "EVI (Mean)")

# Water/Moisture Indices (Sentinel-2 based)
ndwi_mean = gee_functions.safe_execute(gee_functions.get_ndwi,
                                       start=start_date, end=end_date, roi=roi,
                                       return_collection=False)
validate_image(ndwi_mean, "NDWI (Mean)")

ndmi_mean = gee_functions.safe_execute(gee_functions.get_ndmi,
                                       start=start_date, end=end_date, roi=roi,
                                       return_collection=False)
validate_image(ndmi_mean, "NDMI (Mean)")


print(f"\n--- Validating Image Collections for ROI {roi.bounds().getInfo()} from {start_date} to {end_date} ---")

# Vegetation Indices Collections
ndvi_col = gee_functions.safe_execute(gee_functions.get_ndvi,
                                       start=start_date, end=end_date, roi=roi,
                                       return_collection=True)
validate_image(ndvi_col, "NDVI (Collection)", is_collection=True)

savi_col = gee_functions.safe_execute(gee_functions.get_savi,
                                       start=start_date, end=end_date, roi=roi,
                                       L=0.5,
                                       return_collection=True)
validate_image(savi_col, "SAVI (Collection)", is_collection=True)

evi_col = gee_functions.safe_execute(gee_functions.get_evi,
                                      start=start_date, end=end_date, roi=roi,
                                      return_collection=True)
validate_image(evi_col, "EVI (Collection)", is_collection=True)

# Water/Moisture Indices Collections
ndwi_col = gee_functions.safe_execute(gee_functions.get_ndwi,
                                       start=start_date, end=end_date, roi=roi,
                                       return_collection=True)
validate_image(ndwi_col, "NDWI (Collection)", is_collection=True)

ndmi_col = gee_functions.safe_execute(gee_functions.get_ndmi,
                                       start=start_date, end=end_date, roi=roi,
                                       return_collection=True)
validate_image(ndmi_col, "NDMI (Collection)", is_collection=True)

# Existing functions from your gee_functions.py (validation)
et_mean = gee_functions.safe_execute(gee_functions.get_evapotranspiration,
                                     start_date_str=start_date, end_date_str=end_date, geometry=roi,
                                     return_collection=False)
validate_image(et_mean, "Evapotranspiration (MOD16A2, Mean)")

et_col = gee_functions.safe_execute(gee_functions.get_evapotranspiration,
                                    start_date_str=start_date, end_date_str=end_date, geometry=roi,
                                    return_collection=True)
validate_image(et_col, "Evapotranspiration (MOD16A2, Collection)", is_collection=True)

# For static soil properties, get the image directly.
print("\n--- Validating Static Soil Layers ---")

soil_organic_matter_img = gee_functions.safe_execute(gee_functions.get_soil_property,
                                                    key='soil_organic_matter', roi=roi)
validate_image(soil_organic_matter_img, "Soil Organic Matter (OC)")

soil_ph_img = gee_functions.safe_execute(gee_functions.get_soil_property,
                                        key='soil_ph', roi=roi)
validate_image(soil_ph_img, "Soil pH")

soil_cec_img = gee_functions.safe_execute(gee_functions.get_soil_property,
                                         key='soil_cec', roi=roi)
validate_image(soil_cec_img, "Soil CEC")

soil_nitrogen_img = gee_functions.safe_execute(gee_functions.get_soil_property,
                                              key='soil_nitrogen', roi=roi)
validate_image(soil_nitrogen_img, "Soil Nitrogen")

soil_texture_img = gee_functions.safe_execute(gee_functions.get_soil_texture,
                                             start=start_date, end=end_date, roi=roi)
validate_image(soil_texture_img, "Soil Texture (Clay, Silt, Sand)")

# Validate other time-series data
print("\n--- Validating Other Time-Series Data ---")

# Precipitation
precip_mean = gee_functions.safe_execute(gee_functions.get_precipitation,
                                         start=start_date, end=end_date, roi=roi,
                                         return_collection=False)
validate_image(precip_mean, "Precipitation (Mean)")

precip_col = gee_functions.safe_execute(gee_functions.get_precipitation,
                                        start=start_date, end=end_date, roi=roi,
                                        return_collection=True)
validate_image(precip_col, "Precipitation (Collection)", is_collection=True)


# Land Surface Temperature
lst_mean = gee_functions.safe_execute(gee_functions.get_land_surface_temperature,
                                      start=start_date, end=end_date, roi=roi,
                                      return_collection=False)
validate_image(lst_mean, "Land Surface Temperature (Mean)")

lst_col = gee_functions.safe_execute(gee_functions.get_land_surface_temperature,
                                     start=start_date, end=end_date, roi=roi,
                                     return_collection=True)
validate_image(lst_col, "Land Surface Temperature (Collection)", is_collection=True)


# Humidity
humidity_mean = gee_functions.safe_execute(gee_functions.get_humidity,
                                           start=start_date, end=end_date, roi=roi,
                                           return_collection=False)
validate_image(humidity_mean, "Humidity (Mean)")

humidity_col = gee_functions.safe_execute(gee_functions.get_humidity,
                                          start=start_date, end=end_date, roi=roi,
                                          return_collection=True)
validate_image(humidity_col, "Humidity (Collection)", is_collection=True)

# Irradiance
irrad_mean = gee_functions.safe_execute(gee_functions.get_irradiance,
                                        start=start_date, end=end_date, roi=roi,
                                        return_collection=False)
validate_image(irrad_mean, "Irradiance (Mean)")

irrad_col = gee_functions.safe_execute(gee_functions.get_irradiance,
                                       start=start_date, end=end_date, roi=roi,
                                       return_collection=True)
validate_image(irrad_col, "Irradiance (Collection)", is_collection=True)

# Simulated Hyperspectral (already added to gee_functions.py and updated)
sim_hyp_mean = gee_functions.safe_execute(gee_functions.get_simulated_hyperspectral,
                                          start=start_date, end=end_date, roi=roi,
                                          return_collection=False)
validate_image(sim_hyp_mean, "Simulated Hyperspectral (Mean)")

sim_hyp_col = gee_functions.safe_execute(gee_functions.get_simulated_hyperspectral,
                                         start=start_date, end=end_date, roi=roi,
                                         return_collection=True)
validate_image(sim_hyp_col, "Simulated Hyperspectral (Collection)", is_collection=True)