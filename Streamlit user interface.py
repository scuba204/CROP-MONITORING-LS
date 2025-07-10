import streamlit as st
import datetime
import geopandas as gpd
from scripts.gee_functions import (
    get_ndvi, get_soil_moisture, get_precipitation,
    get_land_surface_temperature, get_humidity, get_irradiance,
    get_evapotranspiration, get_soil_ph, get_soil_texture, get_organic_matter,
    get_soil_nitrogen, get_soil_phosphorus, get_soil_potassium
)
import geemap
import ee
from shapely.geometry import mapping
import pandas as pd

# Initialize
st.set_page_config(layout="wide")
st.title("🌱 Interactive Crop Monitoring Dashboard - Lesotho")

# Load district boundaries
shp_path = "data/districts/LSO_adm1.shp"
district_gdf = gpd.read_file(shp_path)
district_names = district_gdf['NAME_1'].tolist()
selected_district = st.selectbox("Select District", district_names)

start_date = st.date_input("Start Date", datetime.date(2024, 11, 1))
end_date = st.date_input("End Date", datetime.date(2025, 5, 31))

# Dataset options
parameter_options = {
    "NDVI": get_ndvi,
    "Soil Moisture": get_soil_moisture,
    "Precipitation": get_precipitation,
    "LST (Celsius)": get_land_surface_temperature,
    "Humidity": get_humidity,
    "Irradiance": get_irradiance,
    "Evapotranspiration": get_evapotranspiration,
    "Soil pH": get_soil_ph,
    "Soil Texture": get_soil_texture,
    "Organic Matter": get_organic_matter,
    "Soil Nitrogen": get_soil_nitrogen,
    "Soil Phosphorus": get_soil_phosphorus,
    "Soil Potassium": get_soil_potassium
}
selected_param = st.selectbox("Select Parameter to Visualize", list(parameter_options.keys()))

# Get geometry
district_geom = district_gdf[district_gdf['NAME_1'] == selected_district].geometry.values[0]
geom = ee.Geometry(mapping(district_geom))

# Generate
if st.button("Generate Map"):
    with st.spinner("Loading Earth Engine Data..."):
        ee.Initialize(project='winged-tenure-464005-p9')
        img = parameter_options[selected_param](str(start_date), str(end_date), geom)

        # Stats summary
        stat = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=30,
            maxPixels=1e9
        ).getInfo()

        mean_val = list(stat.values())[0] if stat else 0

        # Map rendering
        Map = geemap.Map()
        Map.centerObject(geom, 9)
        vis = {'min': 0, 'max': 1, 'palette': ['brown', 'yellow', 'green']} if selected_param == "NDVI" else {}
        Map.addLayer(img, vis, selected_param)

        # District boundary and label
        Map.addLayer(geemap.geopandas_to_ee(district_gdf), {}, 'District Boundaries')
        for i, row in district_gdf.iterrows():
            Map.add_text(
                xy=[row.geometry.centroid.x, row.geometry.centroid.y],
                text=row['NAME_1'],
                font_size=12,
                text_color='black',
                draggable=False
            )

        html_path = f"outputs/map_exports/{selected_param}_{selected_district}.html"
        Map.to_html(html_path)

        col1, col2 = st.columns([1, 2])
        col1.metric(f"{selected_param} Mean Value", round(mean_val, 3))
        col2.components.v1.iframe(html_path, height=600, scrolling=True)

# Optional trend chart
log_path = 'outputs/district_logs/district_full_log.csv'
if os.path.exists(log_path):
    df = pd.read_csv(log_path)
    param_key = selected_param.lower().replace(" (celsius)", "").replace(" ", "_")
    if param_key in df.columns:
        trend = df[df['district'] == selected_district]
        st.line_chart(trend.set_index('date')[param_key])
