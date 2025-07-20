import os
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
from concurrent.futures import ThreadPoolExecutor, as_completed

from scripts.gee_functions import (
    get_ndvi, get_soil_moisture, get_precipitation,
    get_land_surface_temperature, get_humidity, get_irradiance,
    get_simulated_hyperspectral, get_soil_texture,
    get_evapotranspiration, get_soil_property
)
from palettes import get_palettes

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
PARAM_CATEGORIES = {
    "Vegetation":       {"params": ["NDVI"], "help": "NDVI from Sentinel-2"},
    "Climate":          {"params": ["Precipitation","Land Surface Temp","Humidity","Irradiance"],
                         "help": "Daily climate variables"},
    "Soil Properties":  {"params": ["Soil Moisture","Soil Organic Matter","Soil pH",
                                    "Soil Texture - Clay","Soil Texture - Silt","Soil Texture - Sand",
                                    "Soil CEC","Soil Nitrogen"],
                         "help": "Static soil attributes"},
    "Water Use":        {"params": ["Evapotranspiration"], "help": "MODIS ET"},
    "Hyperspectral":    {"params": ["B5","B6","B7","B11","B12"],
                         "help": "Simulated hyperspectral bands"}
}

# Load palettes from external module
PALETTES = get_palettes()

DATA_AVAILABILITY = {
    "NDVI": datetime.date(2015,6,23),
    "Precipitation": datetime.date(1981,1,1),
    "Land Surface Temp": datetime.date(2000,2,24),
    "Humidity": datetime.date(2017,1,1),
    "Irradiance": datetime.date(2017,1,1),
    "Evapotranspiration": datetime.date(2000,2,24),
}

# Parameters that support time series
TIME_SERIES_PARAMS = {
    "NDVI","Precipitation","Land Surface Temp",
    "Humidity","Irradiance","Evapotranspiration","Soil Moisture"
}

# -------------------------------------------------------------------
# INITIALIZE
# -------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("📍 Daily Crop Monitoring System (Lesotho)")
ee.Initialize(project="winged-tenure-464005-p9")

# Load & simplify country geometry
shp = r"C:\Users\MY PC\Documents\GIS DATA\BOUNDARIES\LSO_adm\LSO_adm1.shp"
gdf = gpd.read_file(shp)
gdf["geometry"] = gdf.geometry.simplify(tolerance=0.01)
lesotho_shape = unary_union(gdf.geometry)
country_geom = ee.Geometry(mapping(lesotho_shape))

# -------------------------------------------------------------------
# SIDEBAR COMPONENTS
# -------------------------------------------------------------------
def select_parameters():
    st.header("🧩 Controls")
    cat = st.selectbox("Parameter Category", list(PARAM_CATEGORIES.keys()))
    st.caption(PARAM_CATEGORIES[cat]["help"])
    opts = PARAM_CATEGORIES[cat]["params"]
    q = st.text_input("🔍 Filter Parameters")
    if q:
        opts = [p for p in opts if q.lower() in p.lower()]
    return st.multiselect("Parameters", opts, default=opts)

def select_date_range(params):
    dates = [DATA_AVAILABILITY[p] for p in set(params)&set(DATA_AVAILABILITY)]
    min_date = min(dates) if dates else datetime.date(2000,1,1)
    today = datetime.date.today()
    start = st.date_input("Start Date", today-datetime.timedelta(days=7),
                          min_value=min_date, max_value=today)
    end   = st.date_input("End Date", today,
                          min_value=min_date, max_value=today)
    if start> end:
        st.error("Start date must be before end date"); st.stop()
    return start,end

def select_roi():
    opt = st.radio("Region of Interest",
                   ["Whole Country","Select District","Upload ROI"])
    geom = country_geom
    district = None

    if opt=="Select District":
        district = st.selectbox("Choose District", gdf["NAME_1"].unique())
        shape = gdf.loc[gdf.NAME_1==district,"geometry"].unary_union
        geom = ee.Geometry(mapping(shape))

    if opt=="Upload ROI":
        upl = st.file_uploader("Upload GeoJSON or zipped Shapefile", type=["geojson","zip"])
        if upl:
            if upl.name.endswith(".geojson"):
                gdf_u = gpd.read_file(upl)
            else:
                tmpd = tempfile.mkdtemp()
                path = os.path.join(tmpd,upl.name)
                with open(path,"wb") as f: f.write(upl.read())
                gdf_u = gpd.read_file(f"zip://{path}")
            gdf_u["geometry"] = gdf_u.geometry.simplify(0.001)
            shape = unary_union(gdf_u.geometry)
            geom = ee.Geometry(mapping(shape))

    return geom, opt, district

def report_settings():
    return st.text_input("Report Name", value="lesotho_report")

with st.sidebar:
    selected_params = select_parameters()
    start_date,end_date = select_date_range(selected_params)
    country_geom,roi_option,selected_district = select_roi()
    filename = report_settings()
    ndvi_buffer = st.slider("NDVI Date Buffer (± days)",0,60,30)

with st.sidebar.expander("ℹ️ How to Use"):
    st.markdown("""
    1. Pick parameters & date range.  
    2. Filter list dynamically.  
    3. Select or upload ROI.  
    4. Adjust NDVI buffer if needed.  
    5. Run Monitoring.
    """)

# -------------------------------------------------------------------
# FETCH LAYERS (parallel + error capture)
# -------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_layers(start,end,_geom,params,ndvi_buffer):
    layers, errors = {}, []
    proj500 = ee.Projection("EPSG:4326").atScale(500)

    def fetch_one(p):
        try:
            if p=="NDVI":
                coll = get_ndvi(start,end,_geom,max_expansion_days=ndvi_buffer)
                img  = coll.mean()
            elif p=="Precipitation":
                img = get_precipitation(start,end,_geom)
            elif p=="Land Surface Temp":
                img = get_land_surface_temperature(start,end,_geom)
            elif p=="Humidity":
                img = get_humidity(start,end,_geom)
            elif p=="Irradiance":
                img = get_irradiance(start,end,_geom)
            elif p=="Evapotranspiration":
                img = get_evapotranspiration(start,end,_geom)
            elif p=="Soil Moisture":
                img = get_soil_moisture(start,end,_geom)
            elif p in ["Soil Organic Matter","Soil pH","Soil CEC","Soil Nitrogen"]:
                key = p.lower().replace(" ","_")
                img = get_soil_property(key,_geom)
            elif p.startswith("Soil Texture"):
                tex  = get_soil_texture(_geom)
                band = p.split(" - ")[1].lower()
                img  = tex.select(band)
            elif p in ["B5","B6","B7","B11","B12"]:
                img = get_simulated_hyperspectral(start,end,_geom).select(p)
            else:
                return p,None,"Unknown parameter"

            img2 = (img.rename(p)
                       .setDefaultProjection(proj500)
                       .reduceResolution(ee.Reducer.mean(), maxPixels=1024)
                       .reproject(crs="EPSG:4326", scale=500))
            return p,img2,None
        except Exception as e:
            return p,None,str(e)

    with ThreadPoolExecutor(max_workers=6) as exe:
        futures = {exe.submit(fetch_one,p): p for p in params}
        for f in as_completed(futures):
            name,img,err = f.result()
            if img: layers[name]=img
            else:  errors.append((name,err))

    return layers, errors

# -------------------------------------------------------------------
# GENERIC TIME SERIES EXTRACTION
# -------------------------------------------------------------------
@st.cache_data(ttl=1800)
def extract_timeseries(start,end,_geom,param,ndvi_buffer):
    # fetch collection
    if param=="NDVI":
        coll = get_ndvi(start,end,_geom,max_expansion_days=ndvi_buffer)
    elif param=="Precipitation":
        coll = get_precipitation(start,end,_geom)
    elif param=="Land Surface Temp":
        coll = get_land_surface_temperature(start,end,_geom)
    elif param=="Humidity":
        coll = get_humidity(start,end,_geom)
    elif param=="Irradiance":
        coll = get_irradiance(start,end,_geom)
    elif param=="Evapotranspiration":
        coll = get_evapotranspiration(start,end,_geom)
    elif param=="Soil Moisture":
        coll = get_soil_moisture(start,end,_geom)
    else:
        return pd.DataFrame()

    coll2 = coll.map(lambda img: img.rename(param))
    def to_feat(img):
        val  = img.reduceRegion(ee.Reducer.mean(),_geom,500).get(param)
        date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
        return ee.Feature(None,{"date":date,param:val})

    feats = coll2.map(to_feat)
    dates = feats.aggregate_array("date").getInfo()
    vals  = feats.aggregate_array(param).getInfo()
    df = pd.DataFrame({"Date":pd.to_datetime(dates),param:vals}).dropna()
    return df

# -------------------------------------------------------------------
# RUN & VISUALIZE
# -------------------------------------------------------------------
if st.button("Run Monitoring"):
    try:
        layers, errors = fetch_layers(
            str(start_date),str(end_date),
            country_geom, selected_params, ndvi_buffer
        )

        st.write(f"Fetched layers: {list(layers.keys())}")

        if errors:
            st.error("⚠️ Some layers failed:")
            for p,msg in errors:
                st.write(f"- {p}: {msg}")

        if not layers:
            st.warning("No data layers returned; try a wider range or different ROI.")
            st.stop()

        # Map viewer
        st.header("🚨 Map Viewer")
        visible = st.multiselect("Show Layers", list(layers),
                                 default=list(layers))
        m = geemap.Map(
            center=country_geom.centroid().coordinates().getInfo()[::-1],
            zoom=7
        )

        if roi_option=="Select District" and selected_district:
            shape = gdf.loc[gdf.ADM1_NAME==selected_district,"geometry"].unary_union
            m.addLayer(ee.Geometry(mapping(shape)),
                       {"color":"red","fillOpacity":0},
                       f"{selected_district} Boundary")

        for name in visible:
            cfg = PALETTES[name]
            mn,mx = cfg["min"],cfg["max"]
            pal   = cfg["palette"]
            mid   = (mn+mx)/2
            mid_col = pal[len(pal)//2]
            m.addLayer(layers[name],
                       {"min":mn,"max":mx,"palette":pal},
                       name)
            m.add_legend(title=name,builtin_legend=False,
                         labels=[f"{mn}",f"{mid}",f"{mx}"],
                         colors=[pal[0],mid_col,pal[-1]])

        m.addLayerControl()
        m.to_streamlit(height=600)

        # Parameter means
        st.subheader("📊 Parameter Means")
        stats = {}
        for name,img in layers.items():
            region = ee.Image(img).reduceRegion(
                ee.Reducer.mean(), country_geom, 500, maxPixels=1e9
            ).get(name)
            val=None
            try:
                val = region.getInfo() if region else None
            except:
                val = None
            stats[name] = round(val,3) if isinstance(val,(int,float)) else "n/a"

        df_stats = pd.DataFrame(stats.items(),columns=["Parameter","Mean"])
        st.dataframe(df_stats)

        # Summary metrics
        st.subheader("📌 Summary Metrics")
        cols = st.columns(min(3,len(df_stats)))
        for i,row in df_stats.iterrows():
            cols[i%len(cols)].metric(row.Parameter,row.Mean)

        # Time series for all supported params
        ts_params = [p for p in selected_params if p in TIME_SERIES_PARAMS]
        if ts_params:
            st.subheader("📈 Time Series")
            for p in ts_params:
                df_ts = extract_timeseries(
                    str(start_date),str(end_date),
                    country_geom, p, ndvi_buffer
                )
                if df_ts.empty:
                    st.warning(f"No time series for {p}")
                    continue
                fig = px.line(df_ts,x="Date",y=p,
                              title=f"{p} Trend",markers=True)
                st.plotly_chart(fig,use_container_width=True)
                st.download_button(f"⬇️ Download {p} CSV",
                                   df_ts.to_csv(index=False),
                                   file_name=f"{p.lower()}_timeseries.csv")

        # PDF report
        with tempfile.NamedTemporaryFile(suffix=".pdf",delete=False) as tmp:
            with PdfPages(tmp.name) as pdf:
                fig, ax = plt.subplots(figsize=(8,4))
                df_stats.plot(kind="barh",x="Parameter",y="Mean",
                              ax=ax,legend=False,color="skyblue")
                ax.set_title(f"Parameter Means: {start_date} to {end_date}")
                pdf.savefig(fig,bbox_inches="tight")
                plt.close(fig)
            st.download_button("📄 Download PDF Report",
                               open(tmp.name,"rb").read(),
                               file_name=f"{filename}.pdf",
                               mime="application/pdf")

    except Exception as e:
        st.error(f"Earth Engine error: {e}")
