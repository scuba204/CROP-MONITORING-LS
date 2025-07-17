import streamlit as st
import geemap.foliumap as geemap
import ee
import datetime
import tempfile
from streamlit.components.v1 import html

# 1. Initialize EE
try:
    ee.Initialize(project='winged-tenure-464005-p9')
except Exception as e:
    st.error(f"Earth Engine failed to initialize: {e}")
    st.stop()

# 2. Streamlit UI
st.title("🧪 NDVI Map Fixed Test")

start_date = st.date_input("Start Date", value=datetime.date(2024, 6, 1))
end_date = st.date_input("End Date", value=datetime.date(2024, 6, 30))

if start_date > end_date:
    st.error("❌ Start date must be before end date.")
    st.stop()


# 3. Hardcoded geometry for Lesotho bounding box
geom = ee.Geometry.Rectangle([27.5, -29.5, 28.5, -28.5])
st.write("📦 Geometry bounds:", geom.bounds().getInfo())

# 4. Fetch NDVI from Sentinel-2 collection
def get_ndvi_from_collection(start_date, end_date, geom):
    collection = (ee.ImageCollection("COPERNICUS/S2_SR")
                  .filterBounds(geom)
                  .filterDate(str(start_date), str(end_date))
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                  .sort('system:time_start'))

    image = ee.Image(collection.first())
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    return ndvi.clip(geom), image

try:
    ndvi_img, img_obj = get_ndvi_from_collection(start_date, end_date, geom)

    # Get metadata
    acquisition_date = img_obj.date().format('YYYY-MM-dd').getInfo()
    image_id = img_obj.get('system:index').getInfo()
    st.success(f"🛰️ Using image ID: {image_id} acquired on {acquisition_date}")

    # Validate NDVI band
    bands = ndvi_img.bandNames().getInfo()
    if not bands or "NDVI" not in bands:
        st.warning("⚠️ The NDVI band is missing from the image.")
    else:
        st.write("✅ NDVI image bands:", bands)

    # 5. Render the map
    Map = geemap.Map()
    Map.centerObject(geom, zoom=8)
    Map.addLayer(ndvi_img, {'min': 0, 'max': 1, 'palette': ['brown', 'yellow', 'green']}, "NDVI")

    # 6. Export to HTML and embed
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp:
        Map.to_html(tmp.name)
        with open(tmp.name, encoding="utf-8") as f:
            html_content = f.read()
            html(html_content, height=600)

except Exception as e:
    st.error(f"❌ Failed to load or render NDVI image: {e}")
