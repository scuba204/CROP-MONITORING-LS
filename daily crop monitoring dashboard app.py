import streamlit as st
import datetime
import geopandas as gpd
from scripts.gee_functions import (
    get_ndvi, get_soil_moisture, get_precipitation,
    get_land_surface_temperature, get_humidity, get_irradiance, get_simulated_hyperspectral,
    get_soil_organic_matter, get_soil_ph, get_soil_texture, get_evapotranspiration,
    get_soil_nitrogen, get_soil_phosphorus, get_soil_potassium
)
import geemap
import ee
from shapely.geometry import mapping
import joblib
import numpy as np

st.set_page_config(layout="wide")
st.title("🌱 District-Level Crop Monitoring Dashboard (Lesotho)")


start_date = st.date_input("Start Date", datetime.date(2024, 11, 1))
end_date = st.date_input("End Date", datetime.date(2025, 5, 31))


shp_path = r'C:\Users\MY PC\Documents\GIS DATA\BOUNDARIES\LSO_adm\LSO_adm1.shp'
district_gdf = gpd.read_file(shp_path)
district_names = district_gdf['NAME_1'].tolist()
selected_district = st.selectbox("Select District", district_names)

# Load ML models
try:
    disease_model = joblib.load("models/disease_risk_model.pkl")
    pest_model = joblib.load("models/pest_risk_model.pkl")
except:
    disease_model = None
    pest_model = None


district_geom = district_gdf[district_gdf['NAME_1'] == selected_district].geometry.values[0]


if st.button("Generate District Map"):
    with st.spinner("Fetching satellite data for selected district..."):
        ee.Initialize(project='winged-tenure-464005-p9')
        geom = ee.Geometry(mapping(district_geom))

        ndvi = get_ndvi(str(start_date), str(end_date), geom)
        soil = get_soil_moisture(str(start_date), str(end_date), geom)
        precip = get_precipitation(str(start_date), str(end_date), geom)
        lst = get_land_surface_temperature(str(start_date), str(end_date), geom)
        humidity = get_humidity(str(start_date), str(end_date), geom)
        irradiance = get_irradiance(str(start_date), str(end_date), geom)
        hyper = get_simulated_hyperspectral(str(start_date), str(end_date), geom)
        som = get_soil_organic_matter(geom)
        soil_ph = get_soil_ph(geom)
        soil_texture = get_soil_texture(geom)
        et = get_evapotranspiration(str(start_date), str(end_date), geom)
        nitrogen = get_soil_nitrogen(geom)
        phosphorus = get_soil_phosphorus(geom)
        potassium = get_soil_potassium(geom)

        b5 = hyper.select("B5")
        b6 = hyper.select("B6")
        b7 = hyper.select("B7")
        b11 = hyper.select("B11")
        b12 = hyper.select("B12")


        stats = ee.Image.cat([
            ndvi.rename('NDVI'),
            soil.rename('Soil'),
            precip.rename('Precip'),
            lst.rename('Temp'),
            humidity.rename('Humidity'),
            irradiance.rename('Irradiance'),
            som.rename('SOM'),
            soil_ph.rename('Soil_pH'),
            soil_texture.rename('Texture'),
            et.rename('ET'),
            nitrogen.rename('Nitrogen'),
            phosphorus.rename('Phosphorus'),
            potassium.rename('Potassium'),
            b5.rename('b5'), b6.rename('b6'), b7.rename('b7'), b11.rename('b11'), b12.rename('b12')
        ]).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=30,
            maxPixels=1e9
        ).getInfo()


        mean_ndvi = stats.get('NDVI', 0)
        b5_val = stats.get('b5', 0)
        b6_val = stats.get('b6', 0)
        b7_val = stats.get('b7', 0)
        b11_val = stats.get('b11', 0)
        b12_val = stats.get('b12', 0)

        disease_risk = 'N/A'
        pest_risk = 'N/A'

        if disease_model:
            disease_risk = disease_model.predict([[b5_val, b6_val, b7_val, b11_val, b12_val, mean_ndvi]])[0]

        if pest_model:
            pest_risk = pest_model.predict([[b5_val, b6_val, b7_val, b11_val, b12_val, mean_ndvi]])[0]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Numerical Summary")
            st.metric("Mean NDVI", round(mean_ndvi, 3))
            st.metric("Soil Moisture", round(stats.get('Soil', 0), 3))
            st.metric("Precipitation (mm)", round(stats.get('Precip', 0), 2))
            st.metric("Land Surface Temp (K)", round(stats.get('Temp', 0), 2))
            st.metric("Humidity (K)", round(stats.get('Humidity', 0), 2))
            st.metric("Irradiance (W/m²)", round(stats.get('Irradiance', 0), 2))
            st.metric("Soil Organic Matter (%)", round(stats.get('SOM', 0), 2))
            st.metric("Soil pH", round(stats.get('Soil_pH', 0), 2))
            st.metric("Soil Texture (sand %)", round(stats.get('Texture', 0), 2))
            st.metric("Evapotranspiration (mm)", round(stats.get('ET', 0), 2))
            st.metric("Soil Nitrogen (g/kg)", round(stats.get('Nitrogen', 0), 2))
            st.metric("Soil Phosphorus (pH proxy)", round(stats.get('Phosphorus', 0), 2))
            st.metric("Soil Potassium (CEC proxy)", round(stats.get('Potassium', 0), 2))
            st.metric("Disease Risk", disease_risk)
            st.metric("Pest Risk", pest_risk)

        with col2:
            Map = geemap.Map()
            Map.centerObject(geom, 9)
            Map.addLayer(ndvi, {'min': 0, 'max': 1, 'palette': ['brown', 'yellow', 'green']}, 'NDVI')
            Map.addLayer(soil, {'min': 0, 'max': 0.5}, 'Soil Moisture')
            Map.addLayer(precip, {'min': 0, 'max': 200}, 'Precipitation')
            Map.addLayer(lst, {'min': 270, 'max': 310}, 'Land Surface Temp')
            Map.addLayer(humidity, {}, 'Humidity')
            Map.addLayer(irradiance, {}, 'Irradiance')
            Map.addLayer(som, {'min': 0, 'max': 10}, 'Soil Organic Matter')
            Map.addLayer(soil_ph, {'min': 4, 'max': 8.5}, 'Soil pH')
            Map.addLayer(soil_texture, {'min': 0, 'max': 100}, 'Soil Texture (Sand %)')
            Map.addLayer(et, {'min': 0, 'max': 100}, 'Evapotranspiration')
            Map.addLayer(nitrogen, {'min': 0, 'max': 10}, 'Soil Nitrogen')
            Map.addLayer(phosphorus, {'min': 4, 'max': 8.5}, 'Soil Phosphorus')
            Map.addLayer(potassium, {'min': 0, 'max': 40}, 'Soil Potassium')

            Map.to_html('outputs/map_exports/district_map.html')
            st.success("Map created!")
            st.components.v1.iframe("outputs/map_exports/district_map.html", height=600, scrolling=True)

