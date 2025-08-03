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

# --- Import all necessary GEE functions ---
from scripts.gee_functions import (
    get_ndvi, get_savi, get_evi, get_ndwi, get_ndmi,
    get_soil_moisture, get_precipitation,
    get_land_surface_temperature, get_humidity, get_irradiance,
    get_simulated_hyperspectral, get_soil_texture,
    get_evapotranspiration, get_soil_property
)
from palettes import get_palettes

PALETTES = get_palettes()

# -------------------------------------------------------------------
# CONFIGURATION - CENTRALIZED AND EXTENDED
# -------------------------------------------------------------------
# This dictionary now holds all the information for each parameter
# including its GEE function, default parameters for that function,
# and how to map it to an EE band name if different from the display name.
# 'type': 'time_series' or 'static' helps to determine if date ranges apply.
PARAM_CONFIG = {
    "NDVI":              {"func": get_ndvi,               "args": {"return_collection": False}, "band_name": "NDVI", "type": "time_series", "category": "Vegetation", "help": "Vegetation index from Sentinel-2"},
    "SAVI":              {"func": get_savi,               "args": {"return_collection": False}, "band_name": "SAVI", "type": "time_series", "category": "Vegetation", "help": "Soil-adjusted vegetation index from Sentinel-2"},
    "EVI":               {"func": get_evi,                "args": {"return_collection": False}, "band_name": "EVI", "type": "time_series", "category": "Vegetation", "help": "Enhanced vegetation index from Sentinel-2"},
    "NDWI":              {"func": get_ndwi,               "args": {"return_collection": False}, "band_name": "NDWI", "type": "time_series", "category": "Water", "help": "Normalized Difference Water Index"},
    "NDMI":              {"func": get_ndmi,               "args": {"return_collection": False}, "band_name": "NDMI", "type": "time_series", "category": "Water", "help": "Normalized Difference Moisture Index"},
    "Soil Moisture":     {"func": get_soil_moisture,      "args": {"return_collection": False}, "band_name": "SoilMoi00_10cm_tavg", "type": "time_series", "category": "Water", "help": "Soil moisture content (0-10cm)"},
    "Precipitation":     {"func": get_precipitation,      "args": {"return_collection": False}, "band_name": "precipitation", "type": "time_series", "category": "Climate", "help": "Daily accumulated precipitation"},
    "Land Surface Temp": {"func": get_land_surface_temperature, "args": {"return_collection": False}, "band_name": "LST_C", "type": "time_series", "category": "Climate", "help": "Daily land surface temperature in Celsius"},
    "Humidity":          {"func": get_humidity,           "args": {"return_collection": False}, "band_name": "RH", "type": "time_series", "category": "Climate", "help": "Daily relative humidity"},
    "Irradiance":        {"func": get_irradiance,         "args": {"return_collection": False}, "band_name": "surface_net_solar_radiation", "type": "time_series", "category": "Climate", "help": "Daily surface net solar radiation"},
    "Evapotranspiration":{"func": get_evapotranspiration, "args": {}, "band_name": "ET", "type": "time_series", "category": "Climate", "help": "Daily actual evapotranspiration"},
    "Soil Organic Matter": {"func": get_soil_property,    "args": {"property_key": "ocd_0-5cm_mean"}, "band_name": "ocd_0-5cm_mean", "type": "static", "category": "Soil Properties", "help": "Soil organic carbon density (0-5cm)"},
    "Soil pH":           {"func": get_soil_property,      "args": {"property_key": "phh2o_0-5cm_mean"}, "band_name": "phh2o_0-5cm_mean", "type": "static", "category": "Soil Properties", "help": "Soil pH in H2O (0-5cm)"},
    "Soil CEC":          {"func": get_soil_property,      "args": {"property_key": "cec_0-5cm_mean"}, "band_name": "cec_0-5cm_mean", "type": "static", "category": "Soil Properties", "help": "Soil Cation Exchange Capacity (0-5cm)"},
    "Soil Nitrogen":     {"func": get_soil_property,      "args": {"property_key": "nitrogen_0-5cm_mean"}, "band_name": "nitrogen_0-5cm_mean", "type": "static", "category": "Soil Properties", "help": "Soil Nitrogen (0-5cm)"},
    "Soil Texture - Clay": {"func": get_soil_texture,     "args": {}, "band_name": "clay", "type": "static", "category": "Soil Texture", "help": "Clay content of soil"},
    "Soil Texture - Silt": {"func": get_soil_texture,     "args": {}, "band_name": "silt", "type": "static", "category": "Soil Texture", "help": "Silt content of soil"},
    "Soil Texture - Sand": {"func": get_soil_texture,     "args": {}, "band_name": "sand", "type": "static", "category": "Soil Texture", "help": "Sand content of soil"},
    # Simulated hyperspectral bands (assuming these come from get_simulated_hyperspectral)
    "B2": {"func": get_simulated_hyperspectral, "args": {"return_collection": False}, "band_name": "B2", "type": "time_series", "category": "Hyperspectral", "help": "Simulated Sentinel-2 Band 2 (Blue)"},
    "B3": {"func": get_simulated_hyperspectral, "args": {"return_collection": False}, "band_name": "B3", "type": "time_series", "category": "Hyperspectral", "help": "Simulated Sentinel-2 Band 3 (Green)"},
    "B4": {"func": get_simulated_hyperspectral, "args": {"return_collection": False}, "band_name": "B4", "type": "time_series", "category": "Hyperspectral", "help": "Simulated Sentinel-2 Band 4 (Red)"},
    "B5": {"func": get_simulated_hyperspectral, "args": {"return_collection": False}, "band_name": "B5", "type": "time_series", "category": "Hyperspectral", "help": "Simulated Sentinel-2 Band 5 (Red Edge 1)"},
    "B6": {"func": get_simulated_hyperspectral, "args": {"return_collection": False}, "band_name": "B6", "type": "time_series", "category": "Hyperspectral", "help": "Simulated Sentinel-2 Band 6 (Red Edge 2)"},
    "B7": {"func": get_simulated_hyperspectral, "args": {"return_collection": False}, "band_name": "B7", "type": "time_series", "category": "Hyperspectral", "help": "Simulated Sentinel-2 Band 7 (Red Edge 3)"},
    "B8A": {"func": get_simulated_hyperspectral, "args": {"return_collection": False}, "band_name": "B8A", "type": "time_series", "category": "Hyperspectral", "help": "Simulated Sentinel-2 Band 8A (Narrow NIR)"},
    "B11": {"func": get_simulated_hyperspectral, "args": {"return_collection": False}, "band_name": "B11", "type": "time_series", "category": "Hyperspectral", "help": "Simulated Sentinel-2 Band 11 (SWIR 1)"},
    "B12": {"func": get_simulated_hyperspectral, "args": {"return_collection": False}, "band_name": "B12", "type": "time_series", "category": "Hyperspectral", "help": "Simulated Sentinel-2 Band 12 (SWIR 2)"},
}

# Dynamically create PARAM_CATEGORIES from PARAM_CONFIG
PARAM_CATEGORIES = {}
for param, data in PARAM_CONFIG.items():
    category = data["category"]
    if category not in PARAM_CATEGORIES:
        PARAM_CATEGORIES[category] = {"params": [], "help": ""}
    PARAM_CATEGORIES[category]["params"].append(param)
    # Assign help text from the first parameter in a category, or refine manually if needed
    if not PARAM_CATEGORIES[category]["help"]:
        PARAM_CATEGORIES[category]["help"] = f"{category} related parameters."

# Dynamically create DATA_AVAILABILITY and TIME_SERIES_PARAMS from PARAM_CONFIG
DATA_AVAILABILITY = {
    "NDVI": datetime.date(2015, 6, 23),
    "SAVI": datetime.date(2015, 6, 23),
    "EVI": datetime.date(2015, 6, 23),
    "NDWI": datetime.date(2015, 6, 23),
    "NDMI": datetime.date(2015, 6, 23),
    "Precipitation": datetime.date(1981, 1, 1),
    "Land Surface Temp": datetime.date(2000, 2, 24),
    "Humidity": datetime.date(2017, 1, 1),
    "Irradiance": datetime.date(2017, 1, 1),
    "Evapotranspiration": datetime.date(2000, 2, 24),
    "Soil Moisture": datetime.date(2000, 1, 1),
    # Hyperspectral bands are based on Sentinel-2 availability
    "B2": datetime.date(2015, 6, 23), "B3": datetime.date(2015, 6, 23), "B4": datetime.date(2015, 6, 23),
    "B5": datetime.date(2015, 6, 23), "B6": datetime.date(2015, 6, 23), "B7": datetime.date(2015, 6, 23),
    "B8A": datetime.date(2015, 6, 23), "B11": datetime.date(2015, 6, 23), "B12": datetime.date(2015, 6, 23),
}

TIME_SERIES_PARAMS = {p for p, data in PARAM_CONFIG.items() if data["type"] == "time_series"}


# -------------------------------------------------------------------
# INITIALIZE
# -------------------------------------------------------------------
st.set_page_config(layout="wide")
st.title("📍 Daily Crop Monitoring System (Lesotho)")
ee.Initialize(project="winged-tenure-464005-p9")

# Load & simplify country geometry
shp = r"data/LSO_adm/LSO_adm1.shp"
gdf = gpd.read_file(shp)
gdf["geometry"] = gdf.geometry.simplify(tolerance=0.01)
lesotho_shape = unary_union(gdf.geometry)
country_geom = ee.Geometry(mapping(lesotho_shape))

# -------------------------------------------------------------------
# SIDEBAR COMPONENTS
# -------------------------------------------------------------------
if 'selected_params_session_state' not in st.session_state:
    st.session_state.selected_params_session_state = {'last_selected': []}

def select_parameters():
    st.header("🧩 Controls")
    cat = st.selectbox("Parameter Category", list(PARAM_CATEGORIES.keys()))
    st.caption(PARAM_CATEGORIES[cat]["help"])
    opts = PARAM_CATEGORIES[cat]["params"]

    q = st.text_input("🔍 Filter Parameters")
    if q:
        opts = [p for p in opts if q.lower() in p.lower()]

    selected = st.multiselect(
        "Parameters",
        opts,
        default=[p for p in opts if p in st.session_state.selected_params_session_state.get('last_selected', [])]
    )
    st.session_state.selected_params_session_state['last_selected'] = selected
    return selected

def select_date_range(params):
    relevant_dates = [DATA_AVAILABILITY[p] for p in set(params) & set(DATA_AVAILABILITY.keys()) if PARAM_CONFIG[p]["type"] == "time_series"]

    min_date = max(relevant_dates) if relevant_dates else datetime.date(2000, 1, 1)
    if not relevant_dates:
        st.info("No time-series parameters selected. Date range will be broad.")

    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=7)
    start = st.date_input("Start Date", value=max(default_start, min_date),
                          min_value=min_date, max_value=today)
    end   = st.date_input("End Date", value=today,
                          min_value=min_date, max_value=today)

    if start > end:
        st.error("Start date must be before or equal to end date"); st.stop()
    return start, end

def select_roi():
    opt = st.radio("Region of Interest", ["Whole Country","Select District","Upload ROI"])
    geom = country_geom
    district = None

    if opt=="Select District":
        district = st.selectbox("Choose District", gdf["NAME_1"].unique())
        shape = gdf.loc[gdf.NAME_1==district,"geometry"].unary_union
        geom = ee.Geometry(mapping(shape))

    if opt=="Upload ROI":
        upl = st.file_uploader("Upload GeoJSON or zipped Shapefile", type=["geojson","zip"])
        if upl:
            with tempfile.TemporaryDirectory() as tmpd:
                if upl.name.endswith(".geojson"):
                    filepath = os.path.join(tmpd, upl.name)
                    with open(filepath, "wb") as f: f.write(upl.read())
                    gdf_u = gpd.read_file(filepath)
                else: # zipped shapefile
                    filepath = os.path.join(tmpd, upl.name)
                    with open(filepath, "wb") as f: f.write(upl.read())
                    gdf_u = gpd.read_file(f"zip://{filepath}")
            gdf_u["geometry"] = gdf_u.geometry.simplify(0.001)
            geom = ee.Geometry(mapping(unary_union(gdf_u.geometry)))
        else:
            st.info("Please upload a GeoJSON or zipped Shapefile to use 'Upload ROI'.")
            st.stop()
    return geom, opt, district

def report_settings():
    return st.text_input("Report Name", value="lesotho_report")

with st.sidebar:
    selected_params = select_parameters()
    start_date,end_date = select_date_range(selected_params)
    selected_geom, roi_option, selected_district = select_roi()
    filename = report_settings()
    ndvi_buffer = st.slider("NDVI/S2 Date Buffer (± days)",0,60,30)

with st.sidebar.expander("ℹ️ How to Use"):
    st.markdown("""
    1. Pick parameters & date range.
    2. Filter list dynamically.
    3. Select or upload ROI.
    4. Adjust NDVI/S2 buffer if needed.
    5. Run Monitoring.
    """)

# Helper function to get GEE image/collection
# Helper function to get GEE image/collection
# Helper function to get GEE image/collection
def get_gee_data(param_name, start_date_str, end_date_str, geometry, ndvi_buffer, return_collection=False):
    config = PARAM_CONFIG.get(param_name)
    if not config:
        return None, f"Configuration missing for {param_name}"

    gee_func = config["func"]
    func_specific_args = config["args"].copy()
    param_type = config["type"]

    # Initialize call_args with only 'roi'. Dates added conditionally.
    call_args = {"roi": geometry}

    # Add date parameters ONLY if the current parameter is a time_series type
    if param_type == "time_series":
        call_args.update({"start": start_date_str, "end": end_date_str})
        if "max_expansion_days" in func_specific_args:
            call_args["max_expansion_days"] = ndvi_buffer
        if "return_collection" in func_specific_args:
            call_args["return_collection"] = return_collection

    call_args.update(func_specific_args)

    try:
        if gee_func == get_soil_property:
            # Assumes get_soil_property now only needs roi and property_key
            result = gee_func(roi=geometry, property_key=config["args"]["property_key"])
        elif gee_func == get_soil_texture:
            # Now, get_soil_texture only needs roi as per the new definition
            result = gee_func(roi=geometry)
        elif gee_func == get_evapotranspiration:
            collection = gee_func(start=start_date_str, end=end_date_str, roi=geometry)
            if collection.size().getInfo() == 0:
                return None, f"No data available for {param_name} collection."
            result = collection if return_collection else collection.mean()
        else:
            result = gee_func(**call_args)

        if result is None:
            return None, "GEE function returned None"

        if param_type == "static" and gee_func == get_soil_texture and config["band_name"]:
             result = result.select(config["band_name"])

        return result, None

    except ee.EEException as ee_ex:
        return None, f"GEE Error: {ee_ex}"
    except Exception as ex:
        return None, f"General Error: {ex}"

#@st.cache_data(show_spinner=False, ttl=1800)
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
            # Ensure it's an Image for display (get_gee_data already handles collection.mean() for map layers)
            img = ee.Image(img_or_coll)
            ee_band_name = PARAM_CONFIG[p]["band_name"]
            band_names = img.bandNames().getInfo()

            if ee_band_name not in band_names:
                # Fallback to first band if specific band not found (should be rare with good configs)
                logging.warning(f"{p}: Expected band '{ee_band_name}' not found. Using first band: {band_names[0]}")
                img = img.select([band_names[0]])
                ee_band_name = band_names[0]
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

    if coll is None or coll.size().getInfo() == 0:
        logging.warning(f"Time series: Empty or no collection for {param} for dates {start} to {end}.")
        return pd.DataFrame()

    band_to_extract = PARAM_CONFIG[param]["band_name"]

    def to_feat(img):
        try:
            val = img.reduceRegion(ee.Reducer.mean(), _geom, 500).get(band_to_extract).getInfo()
            date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd").getInfo()
            return ee.Feature(None, {"date": date, "value": val})
        except Exception as e:
            logging.warning(f"Time series: Error processing image in map for {param}: {e}")
            return ee.Feature(None, {"date": None, "value": None})

    feats = coll.map(to_feat).filter(ee.Filter.notNull(['value']))

    try:
        dates = feats.aggregate_array("date").getInfo()
        vals = feats.aggregate_array("value").getInfo()
    except Exception as e:
        logging.error(f"Time series: Error aggregating results for {param}: {e}")
        return pd.DataFrame()

    cleaned_data = [(d, v) for d, v in zip(dates, vals) if d is not None and v is not None]
    if not cleaned_data:
        logging.warning(f"No valid data points found for {param} after aggregation.")
        return pd.DataFrame()

    df = pd.DataFrame(cleaned_data, columns=["Date", "Value"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.dropna()
    df = df.rename(columns={"Value": param})
    return df

# -------------------------------------------------------------------
# RUN & VISUALIZE
# -------------------------------------------------------------------
if st.button("Run Monitoring"):
    st.markdown("---")
    try:
        with st.spinner("Fetching Earth Engine data and preparing layers..."):
            layers, errors = fetch_layers(
                str(start_date), str(end_date),
                selected_geom, selected_params, ndvi_buffer
            )

        st.write(f"Layers processed: {list(layers.keys())}")
        if errors:
            st.error("⚠️ Some layers failed to load or had issues:")
            for p, msg in errors: st.write(f"- **{p}**: {msg}")
        if not layers:
            st.warning("No data layers could be returned; try a wider date range, a different ROI, or check your parameters.")
            st.stop()

        # Map viewer
        st.header("🚨 Map Viewer")
        visible_options = list(layers.keys())
        visible = st.multiselect("Select layers to display on map", visible_options, default=visible_options[:1] if visible_options else [])

        map_center = selected_geom.centroid().coordinates().getInfo()[::-1]
        m = geemap.Map(center=map_center, zoom=8, plugin_Draw=False, minimap=True)
        m.add_basemap(folium.TileLayer('OpenStreetMap', name='OpenStreetMap'))
        m.add_basemap(folium.TileLayer('Esri.WorldImagery', name='Esri Satellite'))

        if roi_option != "Whole Country":
            roi_name = selected_district if selected_district else "Custom ROI"
            m.addLayer(selected_geom, {"color":"red","fillOpacity":0.1, "weight": 3}, f"{roi_name} Boundary")
            m.centerObject(selected_geom) # Geemap will automatically determine the best zoom level

        legend_html_parts = []
        if visible: legend_html_parts.append('<h4>Legend</h4>')

        for name in visible:
            if name in PALETTES:
                cfg = PALETTES[name]
                mn, mx, pal = cfg["min"], cfg["max"], cfg["palette"]
                if not isinstance(pal, list) or len(pal) < 2:
                    logging.warning(f"Invalid palette for {name}: {pal}. Skipping legend entry.")
                    continue
                m.addLayer(layers[name], {"min":mn,"max":mx,"palette":pal}, name)
                gradient_css = f"linear-gradient(to right, {pal[0]}, {pal[len(pal)//2]}, {pal[-1]})"
                legend_html_parts.append(f"""
                    <p style="margin-bottom: 2px;"><b>{name}:</b></p>
                    <div style="width: 100%; height: 15px; background: {gradient_css}; border: 0.5px solid #ccc;"></div>
                    <div style="display: flex; justify-content: space-between; font-size:10px;">
                        <span>{mn}</span>
                        <span>{mx}</span>
                    </div>
                    <br style="margin-top: 5px;">
                """)
            else:
                logging.warning(f"No palette defined for {name}. Layer added but no legend entry.")

        if legend_html_parts:
            legend_html = """
            <div style="position: fixed; bottom: 50px; left: 10px; width: 200px; max-height: 80%; overflow-y: auto;
                         border:2px solid grey; z-index:9999; font-size:14px;
                         background-color:white; opacity:0.9; padding:10px;">
                {}
            </div>
            """.format("".join(legend_html_parts))
            m.get_root().html.add_child(folium.Element(legend_html))
        m.addLayerControl()
        m.to_streamlit(height=600)

        # Parameter means
        st.subheader("📊 Parameter Means")
        stats = {}
        for name, img in layers.items():
            try:
                ee_img = ee.Image(img)
                actual_band_name = PARAM_CONFIG[name]["band_name"] # Get the original band name from config
                
                # Check if band exists, if not, fallback (though `fetch_layers` should prevent this)
                if actual_band_name not in ee_img.bandNames().getInfo():
                    actual_band_name = ee_img.bandNames().getInfo()[0]

                val = ee_img.reduceRegion(reducer=ee.Reducer.mean(), geometry=selected_geom, scale=500, maxPixels=1e9).get(actual_band_name).getInfo()
                stats[name] = round(val, 3) if isinstance(val, (int, float)) else "N/A"
            except Exception as e_inner:
                st.warning(f"Failed to calculate mean for {name}: {e_inner}")
                stats[name] = "Error"

        df_stats = pd.DataFrame(stats.items(), columns=["Parameter", "Mean"])
        st.dataframe(df_stats)

        # Summary metrics
        st.subheader("📌 Summary Metrics")
        if not df_stats.empty:
            cols = st.columns(min(3, len(df_stats)))
            for i, row in df_stats.iterrows():
                cols[i % len(cols)].metric(str(row.Parameter), str(row.Mean))
        else:
            st.info("No parameters to display summary metrics for.")

        # Time series
        ts_params = [p for p in selected_params if PARAM_CONFIG[p]["type"] == "time_series"]
        if ts_params:
            st.subheader("📈 Time Series")
            for p in ts_params:
                with st.spinner(f"Extracting time series for {p}..."):
                    df_ts = extract_timeseries(
                        str(start_date), str(end_date),
                        selected_geom, p, ndvi_buffer
                    )
                if df_ts.empty:
                    st.warning(f"No time series data available for {p} in the selected range/ROI.")
                    continue
                fig = px.line(df_ts, x="Date", y=p, title=f"{p} Trend", markers=True)
                st.plotly_chart(fig, use_container_width=True)
                st.download_button(f"⬇️ Download {p} CSV",
                                    df_ts.to_csv(index=False).encode('utf-8'),
                                    file_name=f"{p.lower().replace(' ', '_')}_timeseries.csv",
                                    mime="text/csv")
        else:
            st.info("No time-series parameters selected or available for the given criteria.")

        # PDF report
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            with PdfPages(tmp.name) as pdf:
                fig, ax = plt.subplots(figsize=(8, 4))
                if not df_stats.empty:
                    df_stats.plot(kind="barh", x="Parameter", y="Mean", ax=ax, legend=False, color="skyblue")
                    ax.set_title(f"Parameter Means: {start_date} to {end_date}")
                    pdf.savefig(fig, bbox_inches="tight")
                else:
                    fig.text(0.5, 0.5, "No data for PDF report.", ha='center', va='center')
                    pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            st.download_button("📄 Download PDF Report",
                                open(tmp.name, "rb").read(),
                                file_name=f"{filename}.pdf",
                                mime="application/pdf")
            os.unlink(tmp.name)

    except Exception as e:
        st.error(f"An unexpected error occurred: {type(e).__name__}. Error message: {e}")
        st.exception(e)