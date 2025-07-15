import streamlit as st
import datetime
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scripts.gee_functions import (
    get_ndvi, get_soil_moisture, get_precipitation,
    get_land_surface_temperature, get_humidity, get_irradiance, get_simulated_hyperspectral,
    get_soil_organic_matter, get_soil_ph, get_soil_texture, get_evapotranspiration,
    get_soil_nitrogen, get_soil_phosphorus, get_soil_potassium
)
import ee
import os
from shapely.geometry import mapping
import geemap.foliumap as geemap

st.set_page_config(layout="wide")
st.title("📍 Daily Crop Monitoring System (District-Level)")

# Initialize Earth Engine
ee.Initialize(project='winged-tenure-464005-p9')

# UI - Date selector
start_date = st.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=7))
end_date = st.date_input("End Date", datetime.date.today())

# Load and select district
shp_path = r'C:\Users\MY PC\Documents\GIS DATA\BOUNDARIES\LSO_adm\LSO_adm1.shp'
district_gdf = gpd.read_file(shp_path)
district_names = district_gdf['NAME_1'].tolist()
selected_district = st.selectbox("Select District", district_names)
district_geom = district_gdf[district_gdf['NAME_1'] == selected_district].geometry.values[0]
geom = ee.Geometry(mapping(district_geom))

# Trigger processing
if st.button("Run Daily Monitoring"):
    with st.spinner("Fetching and processing satellite data..."):
        # Fetch all parameters
        ndvi = get_ndvi(str(start_date), str(end_date), geom)
        soil = get_soil_moisture(str(start_date), str(end_date), geom)
        precip = get_precipitation(str(start_date), str(end_date), geom)
        lst = get_land_surface_temperature(str(start_date), str(end_date), geom)
        humidity = get_humidity(str(start_date), str(end_date), geom)
        irradiance = get_irradiance(str(start_date), str(end_date), geom)
        som = get_soil_organic_matter(geom)
        soil_ph = get_soil_ph(geom)
        texture = get_soil_texture(geom)
        et = get_evapotranspiration(str(start_date), str(end_date), geom)
        nitrogen = get_soil_nitrogen(geom)
        phosphorus = get_soil_phosphorus(geom)
        potassium = get_soil_potassium(geom)
        hyper = get_simulated_hyperspectral(str(start_date), str(end_date), geom)

        b5 = hyper.select("B5")
        b6 = hyper.select("B6")
        b7 = hyper.select("B7")
        b11 = hyper.select("B11")
        b12 = hyper.select("B12")

        image = ee.Image.cat([
            ndvi.rename("NDVI"), soil.rename("SoilMoisture"), precip.rename("Precipitation"),
            lst.rename("LST"), humidity.rename("Humidity"), irradiance.rename("Irradiance"),
            som.rename("SOM"), soil_ph.rename("Soil_pH"), texture.rename("Soil_Texture"),
            et.rename("ET"), nitrogen.rename("Nitrogen"), phosphorus.rename("Phosphorus"),
            potassium.rename("Potassium"), b5.rename("B5"), b6.rename("B6"),
            b7.rename("B7"), b11.rename("B11"), b12.rename("B12")
        ])

        stats = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geom,
            scale=30,
            maxPixels=1e9
        ).getInfo()

        # Display numeric results
        st.subheader("📊 Summary Statistics")
        df = pd.DataFrame.from_dict(stats, orient='index', columns=['Mean']).round(3)
        st.dataframe(df)

        # Option to download
        csv_export_path = f"outputs/daily_summary_{selected_district}.csv"
        df.to_csv(csv_export_path)
        st.download_button("📥 Download Summary CSV", csv_export_path, file_name=f"{selected_district}_summary.csv")

        # Generate trend plots
        log_path = 'outputs/district_logs/district_full_log.csv'
        if os.path.exists(log_path):
            log_df = pd.read_csv(log_path)
            log_df = log_df[log_df['district_id'] == district_gdf[district_gdf['NAME_1'] == selected_district].index[0] + 1]
            st.subheader("📈 Parameter Trends Over Time")
            params = ['NDVI', 'SoilMoisture', 'Precipitation', 'LST', 'ET']
            for param in params:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(pd.to_datetime(log_df['date']), log_df[param], marker='o')
                ax.set_title(f"{param} Trend")
                ax.set_xlabel("Date")
                ax.set_ylabel(param)
                ax.grid(True)
                st.pyplot(fig)

        # Map viewer
        st.subheader("🗺️ Visualize Parameter Maps")
        Map = geemap.Map()
        Map.centerObject(geom, 9)

        vis_options = {
            "NDVI": (ndvi, {'min': 0, 'max': 1, 'palette': ['brown', 'yellow', 'green']}),
            "Soil Moisture": (soil, {'min': 0, 'max': 0.5}),
            "Precipitation": (precip, {'min': 0, 'max': 200}),
            "Land Surface Temp": (lst, {'min': 270, 'max': 310}),
            "Soil pH": (soil_ph, {'min': 4, 'max': 8.5}),
        }

        selected_layer = st.selectbox("Select Layer to View", list(vis_options.keys()))
        Map.addLayer(vis_options[selected_layer][0], vis_options[selected_layer][1], selected_layer)
        map_path = f"outputs/map_{selected_layer}.html"
        Map.to_html(map_path)
        st.components.v1.iframe(map_path, height=600)

        # Export to PDF
        pdf_path = f"outputs/{selected_district}_summary.pdf"
        with PdfPages(pdf_path) as pdf:
            for param in df.index:
                fig, ax = plt.subplots()
                ax.bar(param, df.loc[param, 'Mean'])
                ax.set_ylabel(param)
                pdf.savefig(fig)
                plt.close()
        with open(pdf_path, "rb") as f:
            st.download_button("📄 Download Report PDF", f, file_name=f"{selected_district}_summary.pdf")
