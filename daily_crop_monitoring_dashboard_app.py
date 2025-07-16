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
    get_soil_nitrogen, get_soil_phosphorus, get_soil_cec
)
import ee
import os
from shapely.geometry import mapping
import tempfile
import geemap.foliumap as geemap

st.set_page_config(layout="wide")
st.title("📍 Daily Crop Monitoring System (District-Level)")

# Initialize Earth Engine
ee.Initialize(project='winged-tenure-464005-p9')

# UI - Date selector
start_date = st.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=7))
end_date = st.date_input("End Date", datetime.date.today())

# Load and select districts
shp_path = r'C:\Users\MY PC\Documents\GIS DATA\BOUNDARIES\LSO_adm\LSO_adm1.shp'
district_gdf = gpd.read_file(shp_path)
district_names = district_gdf['NAME_1'].tolist()
selected_districts = st.multiselect("Select District(s)", district_names, default=district_names[:1])

# Parameter selection
param_categories = {
    "Vegetation": ["NDVI"],
    "Climate": ["Precipitation", "Land Surface Temp", "Humidity", "Irradiance"],
    "Soil Properties": ["Soil Moisture", "Soil Organic Matter", "Soil pH", "Soil Texture", "Soil CEC"],
    "Water Use": ["Evapotranspiration"],
    "Nutrients": ["Soil Nitrogen", "Soil Phosphorus"],
    "Hyperspectral Bands": ["B5", "B6", "B7", "B11", "B12"]
}

all_params = sum(param_categories.values(), [])
selected_params = st.multiselect("Select Parameters for Analysis", all_params, default=all_params)

# Modular GEE data fetcher
def fetch_stats(start, end, geom, selected):
    param_map = {
        "NDVI": get_ndvi(start, end, geom).rename("NDVI"),
        "Soil Moisture": get_soil_moisture(start, end, geom).rename("SoilMoisture"),
        "Precipitation": get_precipitation(start, end, geom).rename("Precipitation"),
        "Land Surface Temp": get_land_surface_temperature(start, end, geom).rename("LST"),
        "Humidity": get_humidity(start, end, geom).rename("Humidity"),
        "Irradiance": get_irradiance(start, end, geom).rename("Irradiance"),
        "Soil Organic Matter": get_soil_organic_matter(geom).rename("SOM"),
        "Soil pH": get_soil_ph(geom).rename("Soil_pH"),
        "Soil Texture": get_soil_texture(geom).rename("Soil_Texture"),
        "Evapotranspiration": get_evapotranspiration(start, end, geom).rename("ET"),
        "Soil Nitrogen": get_soil_nitrogen(geom).rename("Nitrogen"),
        "Soil Phosphorus": get_soil_phosphorus(geom).rename("Phosphorus"),
        "Soil CEC": get_soil_cec(geom).rename("CEC")
    }

    if any(p in selected for p in ["B5", "B6", "B7", "B11", "B12"]):
        hyper = get_simulated_hyperspectral(start, end, geom)
        param_map.update({
            "B5": hyper.select("B5"), "B6": hyper.select("B6"), "B7": hyper.select("B7"),
            "B11": hyper.select("B11"), "B12": hyper.select("B12")
        })

    selected_images = [param_map[p] for p in selected if p in param_map]
    image = ee.Image.cat(selected_images)
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=30,
        maxPixels=1e9
    ).getInfo()
    return stats

# Trigger processing
if st.button("Run Monitoring"):
    for district in selected_districts:
        st.markdown(f"### 📍 Results for: **{district}**")
        geom = ee.Geometry(mapping(district_gdf[district_gdf['NAME_1'] == district].geometry.values[0]))

        try:
            with st.spinner("Fetching satellite data..."):
                stats = fetch_stats(str(start_date), str(end_date), geom, selected_params)

            # Display stats
            df = pd.DataFrame.from_dict(stats, orient='index', columns=['Mean']).round(3)
            st.dataframe(df)

            # Trend Plots
            log_path = 'outputs/district_logs/district_full_log.csv'
            if os.path.exists(log_path):
                log_df = pd.read_csv(log_path)
                district_id = district_gdf[district_gdf['NAME_1'] == district].index[0] + 1
                log_df = log_df[log_df['district_id'] == district_id]
                st.subheader("📈 Trends")
                for param in selected_params:
                    if param in log_df.columns:
                        fig, ax = plt.subplots()
                        ax.plot(pd.to_datetime(log_df['date']), log_df[param], marker='o')
                        ax.set_title(f"{param} Trend")
                        ax.set_xlabel("Date")
                        ax.set_ylabel(param)
                        ax.grid(True)
                        st.pyplot(fig)

            # Map viewer
            st.subheader("🗺️ Map Viewer")
            Map = geemap.Map()
            Map.centerObject(geom, 9)

            # Optionally add layers or visualization here

            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as f:
                map_path = f.name
                Map.to_html(map_path)
                st.components.v1.iframe(map_path, height=600)

            # Download buttons
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_csv:
                df.to_csv(tmp_csv.name)
                st.download_button("📥 Download CSV", tmp_csv.name, file_name=f"{district}_summary.csv")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
                with PdfPages(tmp_pdf.name) as pdf:
                    for param in df.index:
                        fig, ax = plt.subplots()
                        ax.bar(param, df.loc[param, 'Mean'])
                        ax.set_ylabel(param)
                        pdf.savefig(fig)
                        plt.close()
                with open(tmp_pdf.name, "rb") as f:
                    st.download_button("📄 Download PDF Report", f, file_name=f"{district}_summary.pdf")

            # Basic interpretation
            if 'NDVI' in stats:
                ndvi_val = stats['NDVI']
                if ndvi_val < 0.3:
                    st.warning("NDVI is low. Crops may be stressed or sparse.")
                elif ndvi_val < 0.6:
                    st.info("NDVI is moderate. Normal early or mixed vegetation conditions.")
                else:
                    st.success("NDVI is high. Likely healthy, dense vegetation.")

        except Exception as e:
            st.error(f"Failed to process {district}: {e}")

    st.write("Selected params:", selected_params)
    st.write("Available params:", list(param_map.keys()))
