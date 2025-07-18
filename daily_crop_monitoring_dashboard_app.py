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
import json
from shapely.ops import unary_union
from shapely.geometry import mapping
from concurrent.futures import ThreadPoolExecutor, as_completed

from scripts.gee_functions import (
    get_ndvi, get_soil_moisture, get_precipitation,
    get_land_surface_temperature, get_humidity, get_irradiance,
    get_simulated_hyperspectral, get_soil_texture,
    get_evapotranspiration, get_soil_property
)

# -------------------------------------------------------------------
# CONFIGURATION (can be moved to JSON/YAML)
# -------------------------------------------------------------------
PARAM_CATEGORIES = {
    "Vegetation": {
        "params": ["NDVI"],
        "help": "Normalized Difference Vegetation Index from Sentinel-2"
    },
    "Climate": {
        "params": ["Precipitation", "Land Surface Temp", "Humidity", "Irradiance"],
        "help": "Daily climate variables"
    },
    "Soil Properties": {
        "params": [
            "Soil Moisture", "Soil Organic Matter", "Soil pH",
            "Soil Texture - Clay", "Soil Texture - Silt", "Soil Texture - Sand",
            "Soil CEC", "Soil Nitrogen"
        ],
        "help": "Static soil attributes from SoilGrids"
    },
    "Water Use": {
        "params": ["Evapotranspiration"],
        "help": "Actual Evapotranspiration from MODIS"
    },
    "Hyperspectral": {
        "params": ["B5","B6","B7","B11","B12"],
        "help": "Simulated hyperspectral bands"
    }
}

PALETTES = {
    "NDVI": {"min": 0, "max": 1,   "palette": ["brown", "yellow", "green"]},
    "Precipitation": {"min": 0, "max": 20,  "palette": ["white", "blue"]},
    "Land Surface Temp": {"min": 0, "max": 40, "palette": ["blue","yellow","red"]},
    "Humidity": {"min": 0, "max":100, "palette":["white","green"]},
    "Irradiance": {"min":0, "max":300, "palette":["white","orange"]},
    "Evapotranspiration":{"min":0,"max":50,"palette":["white","orange"]},
    "Soil Moisture":{"min":0,"max":0.5,"palette":["white","blue"]},
    "Soil Organic Matter":{"min":0,"max":8,"palette":["white","black"]},
    "Soil pH":{"min":3,"max":9,"palette":["red","yellow","green"]},
    "Soil CEC":{"min":0,"max":40,"palette":["white","blue"]},
    "Soil Nitrogen":{"min":0,"max":0.5,"palette":["white","green"]},
    "Soil Texture - Clay":{"min":0,"max":100,"palette":["white","brown"]},
    "Soil Texture - Silt":{"min":0,"max":100,"palette":["white","grey"]},
    "Soil Texture - Sand":{"min":0,"max":100,"palette":["white","yellow"]},
    "B5":{"min":0,"max":1,"palette":["black","white"]},
    "B6":{"min":0,"max":1,"palette":["black","white"]},
    "B7":{"min":0,"max":1,"palette":["black","white"]},
    "B11":{"min":0,"max":1,"palette":["black","white"]},
    "B12":{"min":0,"max":1,"palette":["black","white"]}
}

# Dataset availability for date constraints
DATA_AVAILABILITY = {
    "NDVI": datetime.date(2015, 6, 23),
    "Precipitation": datetime.date(1981, 1, 1),
    "Land Surface Temp": datetime.date(2000, 2, 24),
    "Humidity": datetime.date(2017, 1, 1),
    "Irradiance": datetime.date(2017, 1, 1),
    "Evapotranspiration": datetime.date(2000, 2, 24),
    # static layers: no date constraint
}

# -------------------------------------------------------------------
# INITIALIZE
# -------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("📍 Daily Crop Monitoring System (Lesotho)")
ee.Initialize(project='winged-tenure-464005-p9')

# Load, simplify & union Lesotho geometry
shp_path = r'C:\Users\MY PC\Documents\GIS DATA\BOUNDARIES\LSO_adm\LSO_adm1.shp'
gdf = gpd.read_file(shp_path).simplify(0.01)
lesotho_shape = unary_union(gdf.geometry)
country_geom = ee.Geometry(mapping(lesotho_shape))

# -------------------------------------------------------------------
# SIDEBAR: Modular UI Components
# -------------------------------------------------------------------
def select_parameters():
    """Category selector, dynamic search & parameter multiselect with help."""
    st.header("🧩 Controls")
    category = st.selectbox("Parameter Category", list(PARAM_CATEGORIES.keys()))
    st.caption(PARAM_CATEGORIES[category]["help"])
    params = PARAM_CATEGORIES[category]["params"]
    # dynamic filter
    filter_text = st.text_input("🔍 Filter Parameters")
    if filter_text:
        params = [p for p in params if filter_text.lower() in p.lower()]
    selected = st.multiselect("Parameters", params, default=params)
    return selected

def select_date_range(selected_params):
    """Date inputs with dynamic min_date based on dataset availability."""
    min_dates = [
        DATA_AVAILABILITY[p]
        for p in set(selected_params) & set(DATA_AVAILABILITY.keys())
    ]
    if min_dates:
        min_date = min(min_dates)
    else:
        min_date = datetime.date(2000, 1, 1)
    today = datetime.date.today()
    start = st.date_input("Start Date", today - datetime.timedelta(days=7),
                          min_value=min_date, max_value=today)
    end   = st.date_input("End Date", today, min_value=min_date, max_value=today)
    if start > end:
        st.error("Start date must be before end date")
        st.stop()
    return start, end

def select_roi():
    """Region of Interest: country vs district."""
    roi_opt = st.radio("Region of Interest", ["Whole Country", "Select District"])
    roi_geom = country_geom
    if roi_opt == "Select District":
        district = st.selectbox("Choose District", list(gdf.ADM1_NAME.unique()))
        roi_shape = gdf[gdf.ADM1_NAME == district].geometry.unary_union
        roi_geom = ee.Geometry(mapping(roi_shape))
        # highlight district on map later
    return roi_geom, roi_opt

def report_settings():
    """Report filename input."""
    return st.text_input("Report Name", value="lesotho_report")

with st.sidebar:
    selected_params = select_parameters()
    start_date, end_date = select_date_range(selected_params)
    country_geom, roi_option = select_roi()
    filename = report_settings()

# Help panel
with st.sidebar.expander("ℹ️ How to Use"):
    st.markdown("""
    1. Pick parameters & date range (dates constrained by data availability).  
    2. Filter or search parameter lists dynamically.  
    3. Select ROI: whole country or by district (highlighted on map).  
    4. Click **Run Monitoring** to fetch & view results.
    """)

# -------------------------------------------------------------------
# DATA FETCHING: parallel & cached
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_layers(start, end, geom, params):
    layers = {}

    def fetch(param_name):
        """Fetch one layer by name."""
        try:
            if param_name == "NDVI":
                img = get_ndvi(start, end, geom).mean()
            elif param_name == "Precipitation":
                img = get_precipitation(start, end, geom)
            elif param_name == "Land Surface Temp":
                img = get_land_surface_temperature(start, end, geom)
            elif param_name == "Humidity":
                img = get_humidity(start, end, geom)
            elif param_name == "Irradiance":
                img = get_irradiance(start, end, geom)
            elif param_name == "Evapotranspiration":
                img = get_evapotranspiration(start, end, geom)
            elif param_name == "Soil Moisture":
                img = get_soil_moisture(start, end, geom)
            elif param_name in ["Soil Organic Matter","Soil pH","Soil CEC","Soil Nitrogen"]:
                key = param_name.lower().replace(" ", "_")
                img = get_soil_property(key, geom)
            elif param_name.startswith("Soil Texture"):
                tex = get_soil_texture(geom)
                band = param_name.split(" - ")[1].lower()
                img = tex.select(band)
            elif param_name in ["B5","B6","B7","B11","B12"]:
                hyper = get_simulated_hyperspectral(start, end, geom)
                img = hyper.select(param_name)
            else:
                return (param_name, None)
            # reduce & reproject
            img2 = img.rename(param_name) \
                      .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=1024) \
                      .reproject(crs="EPSG:4326", scale=500)
            return (param_name, img2)
        except Exception:
            return (param_name, None)

    # parallel fetch
    with ThreadPoolExecutor(max_workers=6) as exe:
        futures = {exe.submit(fetch, p): p for p in params}
        for future in as_completed(futures):
            name, img = future.result()
            if img is not None:
                layers[name] = img
    return layers

# -------------------------------------------------------------------
# TIME SERIES: cached
# -------------------------------------------------------------------
@st.cache_data(ttl=1800)
def extract_ndvi_timeseries(start, end, geom):
    coll = get_ndvi(start, end, geom)
    def reducer(img):
        mean = img.reduceRegion(ee.Reducer.mean(), geom, 500).get("NDVI")
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        return ee.Feature(None, {"date": date, "NDVI": mean})
    feats = coll.map(reducer)
    dates = feats.aggregate_array("date").getInfo()
    vals  = feats.aggregate_array("NDVI").getInfo()
    df = pd.DataFrame({"Date": pd.to_datetime(dates), "NDVI": vals}).dropna()
    return df

# -------------------------------------------------------------------
# RUN & VISUALIZE
# -------------------------------------------------------------------
if st.button("Run Monitoring"):
    try:
        layers = fetch_layers(str(start_date), str(end_date), country_geom, selected_params)

        # Map Viewer
        st.header("🚨 National Map Viewer")
        visible = st.multiselect("Show Layers", list(layers.keys()), default=list(layers.keys()))
        m = geemap.Map(center=country_geom.centroid().coordinates().getInfo()[::-1], zoom=7)

        # add district outline
        if roi_option == "Select District":
            geo_json = mapping(gpd.GeoSeries(gpd.GeoSeries.unary_union([shp for shp in gdf.geometry if shp])))
            m.addLayer(ee.Geometry(mapping(geo_json)), {"color":"red"}, "District Boundary")

        # add layers + legends
        for name in visible:
            mn, mx, pal = PALETTES[name]["min"], PALETTES[name]["max"], PALETTES[name]["palette"]
            m.addLayer(layers[name], {"min": mn, "max": mx, "palette": pal}, name)
            m.add_legend(title=name, builtin_legend=False,
                         labels=[f"{mn:.2f}", f"{mx:.2f}"],
                         colors=[pal[0], pal[-1]])
        m.addLayerControl()
        m.addInspector()
        m.to_streamlit(height=600)

        # Summary table & cards
        st.subheader("📊 Parameter Means")
        stats = {}
        for name, img in layers.items():
            try:
                val = ee.Image(img).reduceRegion(
                    ee.Reducer.mean(), country_geom, 500, 1e9
                ).get(name).getInfo()
                stats[name] = round(val, 3) if val is not None else "n/a"
            except:
                stats[name] = "n/a"
        df = pd.DataFrame(stats.items(), columns=["Parameter", "Mean"])
        st.dataframe(df)

        st.subheader("📌 Summary")
        cols = st.columns(min(3, len(df)))
        for idx, row in df.iterrows():
            cols[idx % len(cols)].metric(row.Parameter, row.Mean)

        # NDVI Trend + CSV
        if "NDVI" in selected_params:
            st.subheader("📈 NDVI Time Series")
            ndvi_df = extract_ndvi_timeseries(str(start_date), str(end_date), country_geom)
            fig = px.line(ndvi_df, x="Date", y="NDVI",
                          title="Daily NDVI Trend", markers=True)
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ Download NDVI CSV",
                               data=ndvi_df.to_csv(index=False),
                               file_name="ndvi_timeseries.csv")

        # PDF Report
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            with PdfPages(tmp.name) as pdf:
                fig, ax = plt.subplots(figsize=(8,4))
                df.plot(kind="barh", x="Parameter", y="Mean",
                        ax=ax, legend=False, color="skyblue")
                ax.set_title(f"Parameter Means: {start_date} to {end_date}")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            st.download_button("📄 Download PDF Report",
                               data=open(tmp.name, "rb").read(),
                               file_name=f"{filename}.pdf",
                               mime="application/pdf")

    except Exception as e:
        st.error(f"Earth Engine error: {e}")
