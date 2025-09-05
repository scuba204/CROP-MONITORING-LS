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
import logging
import tempfile
import folium
from shapely.ops import unary_union
from shapely.geometry import mapping
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Import all necessary GEE functions, including the newly added ones ---
from scripts.gee_functions import (
    get_ndvi, get_savi, get_evi, get_ndwi, get_ndmi,
    get_ndre, get_msi, get_osavi, get_gndvi, get_rvi,
    get_soil_moisture, get_precipitation,
    get_land_surface_temperature, get_humidity, get_irradiance,
    get_soil_texture,
    get_evapotranspiration, get_soil_property
)
from palettes import get_palettes

PALETTES = get_palettes()

# -------------------------------------------------------------------
# CONFIGURATION - CENTRALIZED AND EXTENDED
# -------------------------------------------------------------------
PARAM_CONFIG = {
    "NDVI": {"func": get_ndvi, "args": {"return_collection": False}, "band_name": "NDVI", "type": "time_series", "category": "Vegetation Indices", "help": "Normalized Difference Vegetation Index. A common indicator of live green vegetation. Values typically range from -1 to +1, where higher values indicate more dense and healthy vegetation."},
    "SAVI": {"func": get_savi, "args": {"return_collection": False}, "band_name": "SAVI", "type": "time_series", "category": "Vegetation Indices", "help": "Soil-Adjusted Vegetation Index. Similar to NDVI but corrects for the influence of soil brightness, making it more reliable in areas with sparse vegetation."},
    "EVI": {"func": get_evi, "args": {"return_collection": False}, "band_name": "EVI", "type": "time_series", "category": "Vegetation Indices", "help": "Enhanced Vegetation Index. An optimized index that is more sensitive to changes in high-biomass areas and has reduced atmospheric influence compared to NDVI."},
    "NDWI": {"func": get_ndwi, "args": {"return_collection": False}, "band_name": "NDWI", "type": "time_series", "category": "Water Indices", "help": "Normalized Difference Water Index. Used to monitor changes in water content of leaves and is effective at identifying water bodies. Higher values indicate higher water content or surface water."},
    "NDMI": {"func": get_ndmi, "args": {"return_collection": False}, "band_name": "NDMI", "type": "time_series", "category": "Water Indices", "help": "Normalized Difference Moisture Index. Sensitive to the moisture levels in vegetation. It is often used for monitoring drought and fuel levels in fire-prone areas."},
    "NDRE": {"func": get_ndre, "args": {"return_collection": False}, "band_name": "NDRE", "type": "time_series", "category": "Vegetation Indices", "help": "Normalized Difference Red Edge Index. A good indicator of plant health and nitrogen content, especially in the mid-to-late growth stages when chlorophyll content is high."},
    "MSI": {"func": get_msi, "args": {"return_collection": False}, "band_name": "MSI", "type": "time_series", "category": "Vegetation Indices", "help": "Moisture Stress Index. A ratio-based index sensitive to increasing water content in leaves. Higher values indicate greater water stress and less water content."},
    "OSAVI": {"func": get_osavi, "args": {"return_collection": False}, "band_name": "OSAVI", "type": "time_series", "category": "Vegetation Indices", "help": "Optimized Soil-Adjusted Vegetation Index. A modified version of SAVI that is even more effective at minimizing soil background influence."},
    "GNDVI": {"func": get_gndvi, "args": {"return_collection": False}, "band_name": "GNDVI", "type": "time_series", "category": "Vegetation Indices", "help": "Green Normalized Difference Vegetation Index. Similar to NDVI but uses the green band instead of the red. It is more sensitive to chlorophyll concentration than NDVI."},
    "RVI": {"func": get_rvi, "args": {"return_collection": False}, "band_name": "RVI", "type": "time_series", "category": "Vegetation Indices", "help": "Ratio Vegetation Index. A simple ratio of NIR to Red reflectance. It is sensitive to green vegetation but can be affected by atmospheric conditions."},
    "Soil Moisture": {"func": get_soil_moisture, "args": {"return_collection": False}, "band_name": "SoilMoi00_10cm_tavg", "type": "time_series", "category": "Water & Soil", "help": "Volumetric soil moisture content in the top 10cm of soil, measured in m³/m³. Indicates the amount of water present in the soil."},
    "Precipitation": {"func": get_precipitation, "args": {"return_collection": False}, "band_name": "precipitation", "type": "time_series", "category": "Climate", "help": "Daily accumulated precipitation from the CHIRPS dataset, measured in mm/day. Represents rainfall."},
    "Land Surface Temp": {"func": get_land_surface_temperature, "args": {"return_collection": False}, "band_name": "LST_C", "type": "time_series", "category": "Climate", "help": "The temperature of the Earth's surface in Celsius, as measured by MODIS. It differs from air temperature and is a key indicator of energy balance."},
    "Humidity": {"func": get_humidity, "args": {"return_collection": False}, "band_name": "RH", "type": "time_series", "category": "Climate", "help": "Relative Humidity from ERA5-Land. The amount of water vapor in the air, expressed as a percentage of the maximum amount the air could hold at the given temperature."},
    "Irradiance": {"func": get_irradiance, "args": {"return_collection": False}, "band_name": "surface_net_solar_radiation", "type": "time_series", "category": "Climate", "help": "Daily surface net solar radiation (shortwave) from ERA5-Land. It is the balance between incoming and reflected solar energy, indicating the energy available at the surface."},
    "Evapotranspiration": {"func": get_evapotranspiration, "args": {"return_collection": False}, "band_name": "ET", "type": "time_series", "category": "Water & Soil", "help": "Actual Evapotranspiration from MODIS. The sum of water evaporation from the surface and transpiration from plants."},
    "Soil Organic Matter": {"func": get_soil_property, "args": {"key": "soil_organic_matter"}, "band_name": "ocd_0-5cm_mean", "type": "static", "category": "Soil Properties", "help": "Organic carbon content in the top 5cm of soil, a key indicator of soil health and fertility."},
    "Soil pH": {"func": get_soil_property, "args": {"key": "soil_ph"}, "band_name": "phh2o_0-5cm_mean", "type": "static", "category": "Soil Properties", "help": "The pH level of the soil in the top 5cm, indicating its acidity or alkalinity. A crucial factor for nutrient availability."},
    "Soil CEC": {"func": get_soil_property, "args": {"key": "soil_cec"}, "band_name": "cec_0-5cm_mean", "type": "static", "category": "Soil Properties", "help": "Cation Exchange Capacity of the soil in the top 5cm. Represents the soil's ability to hold onto essential nutrients."},
    "Soil Nitrogen": {"func": get_soil_property, "args": {"key": "soil_nitrogen"}, "band_name": "nitrogen_0-5cm_mean", "type": "static", "category": "Soil Properties", "help": "Total nitrogen concentration in the top 5cm of soil. Nitrogen is a critical nutrient for plant growth."},
    "Soil Texture - Clay": {"func": get_soil_texture, "args": {}, "band_name": "clay", "type": "static", "category": "Soil Texture", "help": "The percentage of clay content in the top 5cm of soil. Affects water retention, drainage, and soil structure."},
    "Soil Texture - Silt": {"func": get_soil_texture, "args": {}, "band_name": "silt", "type": "static", "category": "Soil Texture", "help": "The percentage of silt content in the top 5cm of soil. Affects water retention, drainage, and soil structure."},
    "Soil Texture - Sand": {"func": get_soil_texture, "args": {}, "band_name": "sand", "type": "static", "category": "Soil Texture", "help": "The percentage of sand content in the top 5cm of soil. Affects water retention, drainage, and soil structure."},
}

PARAM_UNITS = {p: "" for p in PARAM_CONFIG.keys()}
PARAM_UNITS.update({
    "Soil Moisture": "m³/m³", "Precipitation": "mm/day", "Land Surface Temp": "°C",
    "Humidity": "%", "Irradiance": "W/m²", "Evapotranspiration": "kg/m²/8day",
    "Soil Organic Matter": "dg/kg", "Soil pH": "pH", "Soil CEC": "cmol(c)/kg",
    "Soil Nitrogen": "cg/kg", "Soil Texture - Clay": "%", "Soil Texture - Silt": "%", "Soil Texture - Sand": "%",
})

PARAM_SCALES = {
    "Soil pH": 0.1,
    "Soil CEC": 0.1,
    "Soil Organic Matter": 0.1,
    "Soil Nitrogen": 0.01,
    "Evapotranspiration": 0.1,
}

PARAM_INFO = {p: data['help'] for p, data in PARAM_CONFIG.items()}

PARAM_CATEGORIES = {}
for param, data in PARAM_CONFIG.items():
    category = data["category"]
    PARAM_CATEGORIES.setdefault(category, {"params": [], "help": f"{category} related parameters."})["params"].append(param)

DATA_AVAILABILITY = {
    "NDVI": datetime.date(2015, 6, 23), "SAVI": datetime.date(2015, 6, 23), "EVI": datetime.date(2015, 6, 23),
    "NDWI": datetime.date(2015, 6, 23), "NDMI": datetime.date(2015, 6, 23),
    "NDRE": datetime.date(2015, 6, 23), "MSI": datetime.date(2015, 6, 23), "OSAVI": datetime.date(2015, 6, 23),
    "GNDVI": datetime.date(2015, 6, 23), "RVI": datetime.date(2015, 6, 23),
    "Precipitation": datetime.date(1981, 1, 1),
    "Land Surface Temp": datetime.date(2000, 2, 24), "Evapotranspiration": datetime.date(2000, 2, 24),
    "Humidity": datetime.date(1981, 1, 1), "Irradiance": datetime.date(1979, 1, 1),
    "Soil Moisture": datetime.date(1981, 1, 1),
}
TIME_SERIES_PARAMS = {p for p, data in PARAM_CONFIG.items() if data["type"] == "time_series"}

# -------------------------------------------------------------------
# INITIALIZE
# -------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("📍 Daily Crop Monitoring System (Lesotho)")
ee.Initialize(project="winged-tenure-464005-p9")

shp = r"data/LSO_adm/LSO_adm1.shp"
gdf = gpd.read_file(shp)
gdf["geometry"] = gdf.geometry.simplify(tolerance=0.01)
# CHANGED: Replaced deprecated unary_union
lesotho_shape = gdf.geometry.union_all()
country_geom = ee.Geometry(mapping(lesotho_shape))

# -------------------------------------------------------------------
# SIDEBAR COMPONENTS
# -------------------------------------------------------------------
if 'selected_params_session_state' not in st.session_state:
    st.session_state.selected_params_session_state = {'last_selected': []}

def select_parameters():
    st.header("🧩 Controls")
    cat = st.selectbox("Parameter Category", list(PARAM_CATEGORIES.keys()))
    
    with st.expander("Category Info"):
        st.caption(PARAM_CATEGORIES[cat]["help"])
    
    opts = PARAM_CATEGORIES[cat]["params"]
    q = st.text_input("🔍 Filter Parameters")
    if q:
        opts = [p for p in opts if q.lower() in p.lower()]

    selected = st.multiselect(
        "Parameters",
        opts,
        default=st.session_state.selected_params_session_state.get('last_selected', [])
    )
    st.session_state.selected_params_session_state['last_selected'] = selected
    
    if selected:
        with st.expander("Parameter Details"):
            st.info(f"**{selected[0]}**: {PARAM_INFO.get(selected[0], 'No information available.')}")

    return selected

def select_date_range(params):
    relevant_dates = [DATA_AVAILABILITY[p] for p in set(params) & set(DATA_AVAILABILITY.keys())]
    min_date = max(relevant_dates) if relevant_dates else datetime.date(2000, 1, 1)

    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=30)
    
    start = st.date_input("Start Date", value=max(default_start, min_date), min_value=min_date, max_value=today)
    end = st.date_input("End Date", value=today, min_value=min_date, max_value=today)
    
    if start > end:
        st.error("Start date must be before or equal to end date")
        st.stop()
    
    return start, end

def select_roi():
    opt = st.radio("Region of Interest", ["Whole Country", "Select District", "Upload ROI"])
    geom = country_geom
    district = None

    if opt == "Select District":
        district = st.selectbox("Choose District", gdf["NAME_1"].unique())
        # CHANGED: Replaced deprecated unary_union
        shape = gdf.loc[gdf.NAME_1 == district, "geometry"].union_all()
        geom = ee.Geometry(mapping(shape))
    elif opt == "Upload ROI":
        upl = st.file_uploader("Upload GeoJSON or zipped Shapefile", type=["geojson", "zip"])
        if not upl:
            st.info("Please upload a file to use this option."); st.stop()
        
        with tempfile.TemporaryDirectory() as tmpd:
            filepath = os.path.join(tmpd, upl.name)
            with open(filepath, "wb") as f:
                f.write(upl.read())
            
            if upl.name.endswith(".geojson"):
                gdf_u = gpd.read_file(filepath)
            else:
                gdf_u = gpd.read_file(f"zip://{filepath}")
            
            gdf_u["geometry"] = gdf_u.geometry.simplify(0.001)
            # CHANGED: Replaced deprecated unary_union
            geom = ee.Geometry(mapping(gdf_u.geometry.union_all()))
            
    return geom, opt, district

def report_settings():
    return st.text_input("Report Name", value="lesotho_report")

with st.sidebar:
    selected_params = select_parameters()
    start_date, end_date = select_date_range(selected_params)
    selected_geom, roi_option, selected_district = select_roi()
    filename = report_settings()
    ndvi_buffer = st.slider("Sentinel-2 Date Buffer (± days)", 0, 60, 30, help="Expands the date range for Sentinel-2 data to find cloud-free images.")

with st.sidebar.expander("ℹ️ How to Use"):
    st.markdown("""
    1. **Select Parameters**: Choose a category, then pick one or more parameters to analyze.
    2. **Set Date Range**: Define the time period for your analysis.
    3. **Define ROI**: Analyze the whole country, a specific district, or upload your own area.
    4. **Run Monitoring**: Click the button to fetch data and generate results.
    """)

# Helper function to get GEE image/collection
def get_gee_data(param_name, start_date_str, end_date_str, geometry, ndvi_buffer, return_collection=False):
    config = PARAM_CONFIG.get(param_name)
    if not config:
        return None, f"Configuration missing for {param_name}"

    gee_func = config["func"]
    param_type = config["type"]

    call_args = {"roi": geometry, **config["args"]}
    
    if param_type == "time_series":
        call_args.update({"start": start_date_str, "end": end_date_str})
        call_args.setdefault("max_expansion_days", ndvi_buffer)
    
    try:
        if return_collection and param_type == "time_series":
             call_args["return_collection"] = True

        result = gee_func(**call_args)
        return result, None

    except ee.EEException as ee_ex:
        return None, f"GEE Error: {ee_ex}"
    except Exception as ex:
        return None, f"General Error: {ex}"

@st.cache_data(show_spinner=False, ttl=1800)
def fetch_layers(start, end, _geom, params, ndvi_buffer):
    layers, errors = {}, []
    proj_crs = "EPSG:4326"
    display_scale = 500

    def fetch_one(p):
        img_or_coll, error_msg = get_gee_data(p, start, end, _geom, ndvi_buffer, return_collection=False)
        if error_msg:
            logging.warning(f"{p}: {error_msg}")
            return p, None, error_msg

        try:
            img = ee.Image(img_or_coll)
            ee_band_name = PARAM_CONFIG[p]["band_name"]
            band_names = img.bandNames().getInfo()

            if ee_band_name not in band_names:
                logging.warning(f"{p}: Expected band '{ee_band_name}' not found. Using first band: {band_names[0]}")
                img = img.select([band_names[0]])
            else:
                img = img.select([ee_band_name])

            final_img = img.rename(p).reproject(crs=proj_crs, scale=display_scale)
            return p, final_img, None
        except Exception as ex:
            logging.error(f"{p}: Error processing fetched image for map: {ex}")
            return p, None, f"Error preparing image for map: {ex}"

    with ThreadPoolExecutor(max_workers=6) as exe:
        futures = {exe.submit(fetch_one, p): p for p in params}
        for f in as_completed(futures):
            name, img, err = f.result()
            if img: layers[name] = img
            else: errors.append((name, err))
    return layers, errors

@st.cache_data(ttl=1800)
def extract_timeseries(start, end, _geom, param, ndvi_buffer):
    if PARAM_CONFIG[param]["type"] != "time_series":
        logging.warning(f"Parameter {param} is not a time series parameter.")
        return pd.DataFrame()

    coll, error_msg = get_gee_data(param, start, end, _geom, ndvi_buffer, return_collection=True)
    if error_msg:
        logging.error(f"Time series for {param}: {error_msg}")
        return pd.DataFrame()
    if not coll or coll.size().getInfo() == 0:
        logging.warning(f"Time series: Empty or no collection for {param} for dates {start} to {end}.")
        return pd.DataFrame()

    band_to_extract = PARAM_CONFIG[param]["band_name"]
    scale_factor = PARAM_SCALES.get(param, 1)

    def to_feat(img):
        try:
            scaled_img = img.multiply(scale_factor)
            val = scaled_img.reduceRegion(ee.Reducer.mean(), _geom, 500).get(band_to_extract)
            date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd")
            return ee.Feature(None, {"date": date, "value": val})
        except Exception as e:
            logging.warning(f"Time series: Error processing image in map for {param}: {e}")
            return ee.Feature(None, {"date": None, "value": None})

    feats=coll.map(to_feat).filter(ee.Filter.notNull(["value"]))
    
    try:
        info = feats.getInfo()['features']
        if not info: return pd.DataFrame()
        
        data = [{'Date': f['properties']['date'], 'Value': f['properties']['value']} for f in info]
        
    except Exception as e:
        logging.error(f"Time series: Error aggregating results for {param}: {e}")
        return pd.DataFrame()
    
    if not data:
        logging.warning(f"No valid data points found for {param} after aggregation.")
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")
    df = df.rename(columns={"Value": param})
    return df

# -------------------------------------------------------------------
# RUN & VISUALIZE
# -------------------------------------------------------------------
if st.button("Run Monitoring"):
    st.markdown("---")
    if not selected_params:
        st.warning("Please select at least one parameter to monitor.")
        st.stop()
        
    try:
        with st.spinner("Fetching Earth Engine data and preparing layers..."):
            layers, errors = fetch_layers(
                str(start_date), str(end_date),
                selected_geom, selected_params, ndvi_buffer
            )

        if not layers:
            st.error("⚠️ No data layers could be returned. Please try a wider date range or a different region.")
            if errors:
                st.info("Details on failed layers:")
                for p, msg in errors:
                    st.write(f"- **{p}**: {msg}")
            st.stop()
        
        # Map viewer
        st.header("🚨 Map Viewer")
        visible_options = list(layers.keys())
        default_visible = visible_options[:1] if visible_options else []
        visible = st.multiselect("Select layers to display on map", visible_options, default=default_visible)

        map_center = selected_geom.centroid().coordinates().getInfo()[::-1]
        m = geemap.Map(center=map_center, zoom=8, plugin_Draw=False, minimap=True)
        m.add_basemap(folium.TileLayer('OpenStreetMap', name='OpenStreetMap'))
        m.add_basemap(folium.TileLayer('Esri.WorldImagery', name='Esri Satellite'))

        if roi_option != "Whole Country":
            roi_name = selected_district if selected_district else "Custom ROI"
            m.addLayer(selected_geom, {"color": "red", "fillOpacity": 0.1, "weight": 3}, f"{roi_name} Boundary")
            m.centerObject(selected_geom)

        legend_html_parts = []
        if visible: legend_html_parts.append('<h4>Legend</h4>')

        for name in visible:
            cfg = PALETTES.get(name)
            if not cfg or not isinstance(cfg.get("palette"), list) or len(cfg["palette"]) < 2:
                logging.warning(f"Invalid palette for {name}. Skipping legend entry.")
                continue
            
            scale_factor = PARAM_SCALES.get(name, 1)
            mn, mx, pal = cfg["min"] * scale_factor, cfg["max"] * scale_factor, cfg["palette"]
            
            m.addLayer(layers[name], {"min": cfg["min"], "max": cfg["max"], "palette": pal}, name)
            
            gradient_css = f"linear-gradient(to right, {pal[0]}, {pal[len(pal)//2]}, {pal[-1]})"
            legend_html_parts.append(f"""
                <p style="margin-bottom: 2px;"><b>{name}:</b></p>
                <div style="width: 100%; height: 15px; background: {gradient_css}; border: 0.5px solid #ccc;"></div>
                <div style="display: flex; justify-content: space-between; font-size:10px;">
                    <span>{round(mn, 2)}</span>
                    <span>{round(mx, 2)}</span>
                </div>
                <br style="margin-top: 5px;">
            """)
        
        if legend_html_parts:
            legend_html = f"""
                <div style="position: fixed; bottom: 50px; left: 10px; width: 200px; max-height: 80%; overflow-y: auto;
                             border:2px solid grey; z-index:9999; font-size:14px;
                             background-color:white; opacity:0.9; padding:10px;">
                    {''.join(legend_html_parts)}
                </div>
            """
            m.get_root().html.add_child(folium.Element(legend_html))
        m.addLayerControl()
        m.to_streamlit(height=600)

        # Parameter means
        st.subheader("📊 Parameter Means")
        stats = {}
        for name, img in layers.items():
            try:
                val = ee.Image(img).reduceRegion(
                    reducer=ee.Reducer.mean(), 
                    geometry=selected_geom, 
                    scale=500, 
                    maxPixels=1e9
                ).get(name).getInfo()
                
                if isinstance(val, (int, float)):
                    scale_factor = PARAM_SCALES.get(name, 1)
                    scaled_val = val * scale_factor
                    stats[name] = round(scaled_val, 2)
                else:
                    stats[name] = "N/A"
            except Exception as e_inner:
                st.warning(f"Failed to calculate mean for {name}: {e_inner}")
                stats[name] = "Error"
        
        df_stats = pd.DataFrame(stats.items(), columns=["Parameter", "Mean"])
        df_stats['Unit'] = df_stats['Parameter'].map(PARAM_UNITS)
        st.dataframe(df_stats)

        # Summary metrics
        st.subheader("📌 Summary Metrics")
        if not df_stats.empty:
            cols = st.columns(min(3, len(df_stats)))
            for i, row in df_stats.iterrows():
                unit = PARAM_UNITS.get(row['Parameter'], '')
                cols[i % len(cols)].metric(
                    f"{row['Parameter']} ({unit})", 
                    f"{row['Mean']}", 
                    help=PARAM_INFO.get(row['Parameter'], '')
                )
        else:
            st.info("No parameters to display summary metrics for.")

        # Time series
        ts_params = [p for p in selected_params if PARAM_CONFIG[p]["type"] == "time_series"]
        if ts_params:
            st.subheader("📈 Time Series")
            for p in ts_params:
                with st.spinner(f"Extracting time series for {p}..."):
                    df_ts = extract_timeseries(str(start_date), str(end_date), selected_geom, p, ndvi_buffer)
                if df_ts.empty:
                    st.warning(f"No time series data available for {p} in the selected range/ROI.")
                    continue
                fig = px.line(df_ts, x="Date", y=p, title=f"{p} Trend", markers=True)
                unit = PARAM_UNITS.get(p, '')
                fig.update_yaxes(title_text=f"{p} ({unit})")
                st.plotly_chart(fig, use_container_width=True)
                st.download_button(f"⬇️ Download {p} CSV", df_ts.to_csv(index=False).encode('utf-8'),
                                    file_name=f"{p.lower().replace(' ', '_')}_timeseries.csv", mime="text/csv")
        else:
            st.info("No time-series parameters selected or available for the given criteria.")

        # --- Glossary section for parameter explanations ---
        st.subheader("📖 Glossary of Selected Parameters")
        for p in selected_params:
            info = PARAM_INFO.get(p, "No information available.")
            st.markdown(f"**{p}**: {info}")
        
        st.markdown("---") 

        # PDF report
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            with PdfPages(tmp.name) as pdf:
                fig, ax = plt.subplots(figsize=(8, 4))
                if not df_stats.empty:
                    df_stats_report = df_stats[df_stats['Mean'] != 'N/A'].set_index('Parameter')
                    df_stats_report['Mean'] = pd.to_numeric(df_stats_report['Mean'])
                    df_stats_report.plot(kind="barh", y="Mean", ax=ax, legend=False, color="skyblue")
                    ax.set_title(f"Parameter Means: {start_date} to {end_date}")
                    pdf.savefig(fig, bbox_inches="tight")
                else:
                    fig.text(0.5, 0.5, "No data for PDF report.", ha='center', va='center')
                    pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
        
        with open(tmp.name, "rb") as f:
            pdf_data = f.read()

        st.download_button(
            "📄 Download PDF Report",
            data=pdf_data,
            file_name=f"{filename}.pdf",
            mime="application/pdf"
        )
        os.unlink(tmp.name)
    except Exception as e:
        st.error(f"An unexpected error occurred: {type(e).__name__}. Error message: {e}")
        st.exception(e)