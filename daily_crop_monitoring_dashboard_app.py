import streamlit as st
import datetime
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import tempfile
import geemap.foliumap as geemap
import ee
from shapely.geometry import mapping
import os

from scripts.gee_functions import (
    get_ndvi, get_soil_moisture, get_precipitation,
    get_land_surface_temperature, get_humidity, get_irradiance,
    get_simulated_hyperspectral, get_soil_texture, get_evapotranspiration,
    get_soil_property
)

# Initialize
st.set_page_config(layout="wide")
st.title("📍 Daily Crop Monitoring System (District-Level)")
ee.Initialize(project='winged-tenure-464005-p9')

# UI
start_date = st.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=7))
end_date   = st.date_input("End Date",   datetime.date.today())

# Load district shapes
shp_path = r'C:\Users\MY PC\Documents\GIS DATA\BOUNDARIES\LSO_adm\LSO_adm1.shp'
district_gdf = gpd.read_file(shp_path)
district_names = district_gdf['NAME_1'].tolist()
selected_districts = st.multiselect("Select District(s)", district_names, default=district_names[:1])

# Parameter categories
param_categories = {
    "Vegetation":       ["NDVI"],
    "Climate":          ["Precipitation", "Land Surface Temp", "Humidity", "Irradiance"],
    "Soil Properties":  ["Soil Moisture", "Soil Organic Matter", "Soil pH", "Soil Texture - Clay", "Soil Texture - Silt", "Soil Texture - Sand", "Soil CEC", "Soil Nitrogen"],
    "Water Use":        ["Evapotranspiration"],
    "Hyperspectral":    ["B5","B6","B7","B11","B12"]
}
all_params = sum(param_categories.values(), [])
selected_params = st.multiselect("Select Parameters for Analysis", all_params, default=all_params)

# Fetch & assemble layers with exact naming
def fetch_stats_and_layers(start, end, geom, selected):
    layers = {}
    # core indices
    if "NDVI" in selected:
        layers["NDVI"] = get_ndvi(start, end, geom).rename("NDVI")
    if "Soil Moisture" in selected:
        layers["Soil Moisture"] = get_soil_moisture(start, end, geom).rename("Soil Moisture")
    if "Precipitation" in selected:
        layers["Precipitation"] = get_precipitation(start, end, geom).rename("Precipitation")
    if "Land Surface Temp" in selected:
        layers["Land Surface Temp"] = get_land_surface_temperature(start, end, geom).rename("Land Surface Temp")
    if "Humidity" in selected:
        layers["Humidity"] = get_humidity(start, end, geom).rename("Humidity")
    if "Irradiance" in selected:
        layers["Irradiance"] = get_irradiance(start, end, geom).rename("Irradiance")
    if "Evapotranspiration" in selected:
        layers["Evapotranspiration"] = get_evapotranspiration(start, end, geom).rename("Evapotranspiration")
    # soil properties
    if any(p.startswith("Soil") for p in selected):
        if "Soil Organic Matter"  in selected:
            layers["Soil Organic Matter"] = get_soil_property('soil_organic_matter', geom).rename("Soil Organic Matter")
        if "Soil pH"               in selected:
            layers["Soil pH"] = get_soil_property('soil_ph', geom).rename("Soil pH")
        if "Soil CEC"              in selected:
            layers["Soil CEC"] = get_soil_property('soil_cec', geom).rename("Soil CEC")
        if "Soil Nitrogen"         in selected:
            layers["Soil Nitrogen"] = get_soil_property('soil_nitrogen', geom).rename("Soil Nitrogen")
        # soil texture yields 3 bands: clay, silt, sand
        if any(p.startswith("Soil Texture") for p in selected):
            tex = get_soil_texture(geom)
            layers["Soil Texture - Clay"] = tex.select('clay').rename("Soil Texture - Clay")
            layers["Soil Texture - Silt"] = tex.select('silt').rename("Soil Texture - Silt")
            layers["Soil Texture - Sand"] = tex.select('sand').rename("Soil Texture - Sand")
    # hyperspectral
    if any(b in selected for b in ["B5","B6","B7","B11","B12"]):
        hyper = get_simulated_hyperspectral(start, end, geom)
        for b in ["B5","B6","B7","B11","B12"]:
            if b in selected:
                layers[b] = hyper.select(b).rename(b)
    # stack and reduce
    if not layers:
        return {}, {}
    stack = ee.Image.cat(list(layers.values()))
    stats = stack.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=30,
        maxPixels=1e9
    ).getInfo() or {}
    # ensure keys for every selected
    stats = {p: stats.get(p, None) for p in selected}
    return stats, layers

# Main loop
if st.button("Run Monitoring"):
    for district in selected_districts:
        st.header(f"📍 {district}")
        geom = ee.Geometry(mapping(
            district_gdf.loc[district_gdf['NAME_1']==district, 'geometry'].values[0]
        ))

        try:
            with st.spinner("Fetching data…"):
                stats, layers = fetch_stats_and_layers(
                    str(start_date), str(end_date), geom, selected_params
                )

            # show table
            df = pd.DataFrame.from_dict(stats, orient='index', columns=['Mean']).round(3)
            st.dataframe(df)

            # trends
            log_path = 'outputs/district_logs/district_full_log.csv'
            if os.path.exists(log_path):
                log_df = pd.read_csv(log_path)
                did = (district_gdf['NAME_1']==district).idxmax() + 1
                sub = log_df[log_df['district_id']==did]
                if not sub.empty:
                    st.subheader("📈 Trends")
                    for p in selected_params:
                        if p in sub:
                            fig, ax = plt.subplots()
                            ax.plot(pd.to_datetime(sub['date']), sub[p], marker='o')
                            ax.set(title=f"{p} Trend", xlabel="Date", ylabel=p)
                            ax.grid()
                            st.pyplot(fig)

            # map
            st.subheader("🗺️ Map Viewer")
            Map = geemap.Map(center=geom, zoom=8)
            palettes = {
                "NDVI": (0,1,['brown','yellow','green']),
                "Soil Moisture": (0,0.5,['white','blue']),
                "Precipitation": (0,20,['white','blue']),
                "Land Surface Temp": (0,40,['blue','yellow','red']),
                "Humidity": (0,100,['white','green']),
                "Irradiance": (0,300,['white','orange']),
                "Evapotranspiration": (0,50,['white','orange']),
                "Soil Organic Matter": (0,8,['white','black']),
                "Soil pH": (3,9,['red','yellow','green']),
                "Soil CEC": (0,40,['white','blue']),
                "Soil Nitrogen": (0,0.5,['white','green']),
                "Soil Texture - Clay": (0,100,['white','brown']),
                "Soil Texture - Silt": (0,100,['white','grey']),
                "Soil Texture - Sand": (0,100,['white','yellow']),
                "B5": (0,1,['black','white']),
                "B6": (0,1,['black','white']),
                "B7": (0,1,['black','white']),
                "B11": (0,1,['black','white']),
                "B12": (0,1,['black','white'])
            }
            for p in selected_params:
                if p in layers:
                    mn, mx, pal = palettes.get(p, (0,1,['white','black']))
                    Map.addLayer(layers[p], {'min':mn,'max':mx,'palette':pal}, p)
            # embed
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
            Map.to_html(tmp.name)
            st.components.v1.iframe(tmp.name, height=500)

            # downloads
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
                df.to_csv(f.name)
                st.download_button("📥 CSV", f.name, file_name=f"{district}_summary.csv")

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                with PdfPages(f.name) as pdf:
                    for p, row in df.iterrows():
                        fig, ax = plt.subplots()
                        ax.bar(p, row['Mean'])
                        ax.set_ylabel("Mean")
                        pdf.savefig(fig)
                        plt.close(fig)
                with open(f.name, "rb") as ff:
                    st.download_button("📄 PDF Report", ff, file_name=f"{district}_report.pdf")

            # basic NDVI flagging
            if stats.get("NDVI") is not None:
                nd = stats["NDVI"]
                if nd < 0.3:
                    st.warning("Low NDVI – possible crop stress.")
                elif nd < 0.6:
                    st.info("Moderate NDVI – normal vegetation.")
                else:
                    st.success("High NDVI – healthy vegetation.")

        except Exception as e:
            st.error(f"Error for {district}: {e}")

    st.write("Selected parameters:", selected_params)
