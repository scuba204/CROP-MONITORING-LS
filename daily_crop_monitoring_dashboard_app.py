import streamlit as st
import datetime
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import plotly.express as px
import geemap.foliumap as geemap
import ee
import tempfile
from shapely.ops import unary_union
from shapely.geometry import mapping

from scripts.gee_functions import (
    get_ndvi, get_soil_moisture, get_precipitation,
    get_land_surface_temperature, get_humidity, get_irradiance,
    get_simulated_hyperspectral, get_soil_texture,
    get_evapotranspiration, get_soil_property
)

# Initialize
st.set_page_config(layout="wide")
st.title("📍 Daily Crop Monitoring System (Lesotho)")
ee.Initialize(project='winged-tenure-464005-p9')

# Load & union Lesotho geometry
shp_path = r'C:\Users\MY PC\Documents\GIS DATA\BOUNDARIES\LSO_adm\LSO_adm1.shp'
gdf = gpd.read_file(shp_path)
lesotho_shape = unary_union(gdf.geometry)
country_geom = ee.Geometry(mapping(lesotho_shape))

# Parameter categories
param_categories = {
    "Vegetation":       ["NDVI"],
    "Climate":          ["Precipitation", "Land Surface Temp", "Humidity", "Irradiance"],
    "Soil Properties":  ["Soil Moisture", "Soil Organic Matter", "Soil pH",
                         "Soil Texture - Clay", "Soil Texture - Silt", "Soil Texture - Sand",
                         "Soil CEC", "Soil Nitrogen"],
    "Water Use":        ["Evapotranspiration"],
    "Hyperspectral":    ["B5","B6","B7","B11","B12"]
}
all_params = sum(param_categories.values(), [])

# Sidebar UI
with st.sidebar:
    st.header("🧭 Controls")

    with st.expander("📅 Date Range"):
        start_date = st.date_input("Start Date", datetime.date.today() - datetime.timedelta(days=7))
        end_date   = st.date_input("End Date", datetime.date.today())
        if start_date > end_date:
            st.error("Start date must be before end date")
            st.stop()

    with st.expander("🧪 Parameter Selection"):
        category = st.selectbox("Category", list(param_categories.keys()))
        selected_params = st.multiselect("Parameters", param_categories[category], default=param_categories[category])

    with st.expander("📥 Report Settings"):
        filename = st.text_input("Report Name", value="lesotho_report")

# Help panel
with st.expander("ℹ️ How to Use This Dashboard"):
    st.markdown("""
    - Select a date range and parameter category.
    - Choose parameters to analyze.
    - Click 'Run Monitoring' to fetch satellite data.
    - View maps, summary cards, trends, and download reports.
    """)

# Visualization palettes
palettes = {
    "NDVI": (0,1,["brown","yellow","green"]),
    "Precipitation": (0,20,["white","blue"]),
    "Land Surface Temp": (0,40,["blue","yellow","red"]),
    "Humidity": (0,100,["white","green"]),
    "Irradiance": (0,300,["white","orange"]),
    "Evapotranspiration": (0,50,["white","orange"]),
    "Soil Moisture": (0,0.5,["white","blue"]),
    "Soil Organic Matter": (0,8,["white","black"]),
    "Soil pH": (3,9,["red","yellow","green"]),
    "Soil CEC": (0,40,["white","blue"]),
    "Soil Nitrogen": (0,0.5,["white","green"]),
    "Soil Texture - Clay": (0,100,["white","brown"]),
    "Soil Texture - Silt": (0,100,["white","grey"]),
    "Soil Texture - Sand": (0,100,["white","yellow"]),
    "B5": (0,1,["black","white"]),
    "B6": (0,1,["black","white"]),
    "B7": (0,1,["black","white"]),
    "B11": (0,1,["black","white"]),
    "B12": (0,1,["black","white"])
}

# Cache-heavy EE calls
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_layers(start, end, geom, params):
    layers = {}
    if "Soil Moisture" in params:
        layers["Soil Moisture"] = get_soil_moisture(start, end, geom).rename("Soil Moisture")
    if "NDVI" in params:
        layers["NDVI"] = get_ndvi(start, end, geom).rename("NDVI")
    if "Precipitation" in params:
        layers["Precipitation"] = get_precipitation(start, end, geom).rename("Precipitation")
    if "Land Surface Temp" in params:
        layers["Land Surface Temp"] = get_land_surface_temperature(start, end, geom).rename("Land Surface Temp")
    if "Humidity" in params:
        layers["Humidity"] = get_humidity(start, end, geom).rename("Humidity")
    if "Irradiance" in params:
        layers["Irradiance"] = get_irradiance(start, end, geom).rename("Irradiance")
    if "Evapotranspiration" in params:
        layers["Evapotranspiration"] = get_evapotranspiration(start, end, geom).rename("Evapotranspiration")
    if any(p.startswith("Soil") for p in params):
        if "Soil Organic Matter" in params:
            layers["Soil Organic Matter"] = get_soil_property("soil_organic_matter", geom).rename("Soil Organic Matter")
        if "Soil pH" in params:
            layers["Soil pH"] = get_soil_property("soil_ph", geom).rename("Soil pH")
        if "Soil CEC" in params:
            layers["Soil CEC"] = get_soil_property("soil_cec", geom).rename("Soil CEC")
        if "Soil Nitrogen" in params:
            layers["Soil Nitrogen"] = get_soil_property("soil_nitrogen", geom).rename("Soil Nitrogen")
        if any(p.startswith("Soil Texture") for p in params):
            tex = get_soil_texture(geom)
            layers["Soil Texture - Clay"] = tex.select("clay").rename("Soil Texture - Clay")
            layers["Soil Texture - Silt"] = tex.select("silt").rename("Soil Texture - Silt")
            layers["Soil Texture - Sand"] = tex.select("sand").rename("Soil Texture - Sand")
    if any(b in params for b in ["B5","B6","B7","B11","B12"]):
        hyper = get_simulated_hyperspectral(start, end, geom)
        for b in ["B5","B6","B7","B11","B12"]:
            if b in params:
                layers[b] = hyper.select(b).rename(b)
    for name, img in layers.items():
        layers[name] = img.reduceResolution(
            reducer=ee.Reducer.mean(), maxPixels=1024
        ).reproject(crs="EPSG:4326", scale=500)
    return layers

# Run Monitoring
if st.button("Run Monitoring"):
    layers = fetch_layers(str(start_date), str(end_date), country_geom, selected_params)

    # Map Viewer
    st.header("🗺️ National Map Viewer")
    visible_layers = st.multiselect("Show Layers", list(layers.keys()), default=list(layers.keys()))
    m = geemap.Map(center=country_geom, zoom=7)
    for name in visible_layers:
        mn, mx, pal = palettes[name]
        m.addLayer(layers[name], {"min": mn, "max": mx, "palette": pal}, name)
    m.addLayerControl()
    m.addInspector()
    m.to_streamlit(height=600)

    # Statistics
    st.subheader("📊 Parameter Means")
    stats = {
        name: ee.Image(img).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=country_geom,
            scale=500,
            maxPixels=1e9
        ).get(name).getInfo()
        for name, img in layers.items()
    }
    df = pd.DataFrame(stats.items(), columns=["Parameter", "Mean"]).round(3)
    st.dataframe(df)

    # Summary Cards
    st.subheader("📌 Summary")
    cols = st.columns(min(3, len(df)))
    for i, row in enumerate(df.itertuples()):
        cols[i % len(cols)].metric(row.Parameter, row.Mean)

    # Trend Plot (example: NDVI)
    if "NDVI" in layers:
        st.subheader("📈 NDVI Trend (Simulated)")
        dates = pd.date