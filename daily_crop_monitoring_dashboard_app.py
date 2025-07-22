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

from scripts.gee_functions import (
    get_ndvi, get_soil_moisture, get_precipitation,
    get_land_surface_temperature, get_humidity, get_irradiance,
    get_simulated_hyperspectral, get_soil_texture,
    get_evapotranspiration, get_soil_property
)
from palettes import get_palettes

PALETTES= get_palettes()

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
# EDIT: Changed to a relative path. Place your shapefile in a 'data' subfolder.
shp = r"data/LSO_adm/LSO_adm1.shp"
gdf = gpd.read_file(shp)
gdf["geometry"] = gdf.geometry.simplify(tolerance=0.01)
lesotho_shape = unary_union(gdf.geometry)
country_geom = ee.Geometry(mapping(lesotho_shape))

# Convert hex palettes to RGB

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
#

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
        shape = gdf.loc[gdf.NAME_1==district,"geometry"].union_all()
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
    # EDIT: Renamed the geometry variable for clarity
    selected_geom, roi_option, selected_district = select_roi()
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

# This is crucial for consistent band selection and metric display.
PARAM_BAND_MAPPING = {
    "NDVI": "NDVI",
    "Precipitation": "precipitation",
    "Land Surface Temp": "LST_C", # Based on your test output
    "Humidity": "RH",
    "Irradiance": "surface_net_solar_radiation",
    "Evapotranspiration": "ET",
    "Soil Moisture": "SoilMoi00_10cm_tavg",
    # Soil Properties and Soil Texture might have different band names depending on which is selected
    # For now, we'll assume the get_soil_property and get_soil_texture functions handle their band selection internally
    # For time series, we'll need to be more explicit for these.
}

#@st.cache_data(show_spinner=False, ttl=1800)
def fetch_layers(start, end, _geom, params, ndvi_buffer):
    layers, errors = {}, []
    proj500 = ee.Projection("EPSG:4326").atScale(500)

    def fetch_one(p):
        img = None # Initialize img to None
        try:
            if p == "NDVI":
                # get_ndvi should return an ee.Image (mean/median of collection) for map display
                img = get_ndvi(start, end, _geom, max_expansion_days=ndvi_buffer)
            elif p == "Precipitation":
                img = get_precipitation(start, end, _geom)
            elif p == "Land Surface Temp":
                img = get_land_surface_temperature(start, end, _geom)
            elif p == "Humidity":
                img = get_humidity(start, end, _geom)
            elif p == "Irradiance":
                img = get_irradiance(start, end, _geom)
            elif p == "Evapotranspiration":
                # get_evapotranspiration returns an ImageCollection, so we mean() it here
                coll = get_evapotranspiration(start, end, _geom)
                if coll.size().getInfo() == 0:
                    logging.warning(f"{p}: Empty collection, returning None image.")
                    return p, None, f"No data available for {p}."
                img = coll.mean().rename(PARAM_BAND_MAPPING[p]) # Rename to generic param name
            elif p == "Soil Moisture":
                img = get_soil_moisture(start, end, _geom) # Should return an ee.Image
            elif p in ["Soil Organic Matter", "Soil pH", "Soil CEC", "Soil Nitrogen"]:
                key = p.lower().replace(" ","_")
                img = get_soil_property(key, _geom) # Should return an ee.Image
            elif p.startswith("Soil Texture"):
                tex = get_soil_texture(_geom)
                band = p.split(" - ")[1].lower() # e.g., 'clay', 'silt', 'sand'
                img = tex.select(band) # Select specific band from the texture image
            elif p in ["B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"]: # Include all hyperspectral bands
                # get_simulated_hyperspectral returns an ee.Image (mean/median of collection)
                img = get_simulated_hyperspectral(start, end, _geom).select(p)
            else:
                return p, None, "Unknown parameter"

            # --- Robust checks for the returned image ---
            if img is None:
                logging.warning(f"{p}: get_ function returned None.")
                return p, None, f"No data (None image) for {p}."

            try:
                # Ensure the image actually has bands
                band_names = img.bandNames().getInfo()
                if not band_names:
                    logging.warning(f"{p}: Image has no bands, likely empty data for selected range/ROI.")
                    return p, None, f"No band data available for {p}."
                # If the image has multiple bands but we only want one for display,
                # ensure it's explicitly selected/renamed. This is handled by PARAM_BAND_MAPPING logic.
                if p in PARAM_BAND_MAPPING and PARAM_BAND_MAPPING[p] not in band_names:
                     logging.warning(f"{p}: Expected band '{PARAM_BAND_MAPPING[p]}' not found in image bands: {band_names}")
                     # Try to select the default first band if the specific one isn't found
                     img = img.select([band_names[0]]) # Select the first available band
                     logging.warning(f"{p}: Selected first available band '{band_names[0]}' for display.")

            except ee.EEException as ee_ex:
                 # Catch specific GEE exceptions if bandNames() fails on truly invalid EE objects
                 logging.error(f"{p}: GEE Error getting band names or processing image: {ee_ex}")
                 return p, None, f"GEE Error processing {p} data."
            except Exception as ex:
                 # Catch general Python exceptions
                 logging.error(f"{p}: Unexpected error checking image bands: {ex}")
                 return p, None, f"Unexpected error processing {p} data."

            # Apply common processing (rename, projection, reduceResolution, reproject)
            # Make sure the image is renamed to the generic parameter name for consistency
            final_img = img.rename(p) \
                           .setDefaultProjection(proj500) \
                           .reduceResolution(ee.Reducer.mean(), maxPixels=1024) \
                           .reproject(crs="EPSG:4326", scale=500)

            return p, final_img, None
        except Exception as e:
            # Catch any other unexpected errors during the fetch_one process
            logging.error(f"Error fetching {p}: {e}")
            return p, None, str(e)

    with ThreadPoolExecutor(max_workers=6) as exe:
        futures = {exe.submit(fetch_one, p): p for p in params}
        for f in as_completed(futures):
            name, img, err = f.result()
            if img: layers[name] = img
            else: errors.append((name, err))

    return layers, errors

# -------------------------------------------------------------------
# GENERIC TIME SERIES EXTRACTION
# -------------------------------------------------------------------
@st.cache_data(ttl=1800)
def extract_timeseries(start, end, _geom, param, ndvi_buffer):
    coll = None # Initialize coll

    # Define a mapping for the band name to extract for each parameter
    # This is important because the parameter name (e.g., "Land Surface Temp")
    # might not be the same as the actual band name (e.g., "LST_C")
    band_to_extract = PARAM_BAND_MAPPING.get(param, param) # Default to param if not in map

    if param == "NDVI":
        # For time series, get_ndvi needs to return the collection itself
        coll = get_ndvi(start, end, _geom, max_expansion_days=ndvi_buffer, return_collection=True)
    elif param == "Precipitation":
        coll = get_precipitation(start, end, _geom, return_collection=True) # Assuming get_precipitation also has return_collection
    elif param == "Land Surface Temp":
        coll = get_land_surface_temperature(start, end, _geom, return_collection=True) # Assuming return_collection
    elif param == "Humidity":
        coll = get_humidity(start, end, _geom, return_collection=True) # Assuming return_collection
    elif param == "Irradiance":
        coll = get_irradiance(start, end, _geom, return_collection=True) # Assuming return_collection
    elif param == "Evapotranspiration":
        # get_evapotranspiration already returns an ImageCollection
        coll = get_evapotranspiration(start, end, _geom)
    elif param == "Soil Moisture":
        # get_soil_moisture already returns an ImageCollection (if you modified it to do so for TS)
        # If get_soil_moisture returns an Image for single date/mean, you need a different get_soil_moisture_collection
        coll = get_soil_moisture(start, end, _geom, return_collection=True) # Assuming return_collection
    # Soil properties and texture are usually static or single images, not time series collections.
    # So, they generally won't be in TIME_SERIES_PARAMS. If they are, their handling would be different.
    elif param.startswith("Soil Texture") or param in ["Soil Organic Matter", "Soil pH", "Soil CEC", "Soil Nitrogen"]:
        # These are generally not time series. If param is one of these, it's likely an error
        # in the TIME_SERIES_PARAMS definition or an assumption mismatch.
        # For simplicity, returning empty DataFrame here.
        logging.warning(f"Attempted to extract time series for non-time-series parameter: {param}")
        return pd.DataFrame()
    elif param in ["B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"]: # Hyperspectral bands
        # get_simulated_hyperspectral should return collection for time series
        coll = get_simulated_hyperspectral(start, end, _geom, return_collection=True)
        # If the simulated hyperspectral collection's images' bands are named differently,
        # you'll need to map 'B5' -> 'B5' etc., which is already happening with `coll.select([param])`.
        band_to_extract = param # The band name is directly the parameter name for hyperspectral

    else:
        logging.warning(f"Unknown parameter for time series extraction: {param}")
        return pd.DataFrame() # Should not happen given TIME_SERIES_PARAMS filter

    # --- Robust check for empty collection before mapping ---
    if not coll: # Check if coll is None
        logging.warning(f"Time series: No collection object created for {param}.")
        return pd.DataFrame()

    try:
        if coll.size().getInfo() == 0: # Check if the collection is empty
            logging.warning(f"Time series: Empty collection for {param} for dates {start} to {end}.")
            return pd.DataFrame()
    except Exception as e:
        logging.error(f"Time series: Error checking collection size for {param}: {e}")
        return pd.DataFrame()

    # Ensure the parameter band exists in the collection images
    # Use the mapped band name for selection
    coll_with_param_band = coll.select([band_to_extract])

    def to_feat(img):
        # Use img.get(band_to_extract) to get the value for the specific band
        # Also ensure the scale is appropriate.
        try:
            val = img.reduceRegion(ee.Reducer.mean(), _geom, 500).get(band_to_extract).getInfo()
            date = ee.Date(img.get("system:time_start")).format("YYYY-MM-dd").getInfo()
            return ee.Feature(None, {"date": date, "value": val}) # Use generic "value" key
        except ee.EEException as ee_ex:
            logging.warning(f"Time series: GEE Error processing image in map for {param}: {ee_ex}")
            return ee.Feature(None, {"date": None, "value": None}) # Return a feature that will be filtered out
        except Exception as ex:
            logging.warning(f"Time series: General error processing image in map for {param}: {ex}")
            return ee.Feature(None, {"date": None, "value": None})


    # Apply the map to the collection that has the band selected
    # Filter out features where value might be None from reduction errors
    feats = coll_with_param_band.map(to_feat).filter(ee.Filter.notNull(['value']))

    try:
        # Use aggregate_array on the filtered feature collection
        dates_ee = feats.aggregate_array("date")
        vals_ee = feats.aggregate_array("value")

        # Get results from Earth Engine
        dates = dates_ee.getInfo()
        vals = vals_ee.getInfo()

    except ee.EEException as ee_ex:
        logging.error(f"Time series: GEE Error aggregating results for {param}: {ee_ex}")
        return pd.DataFrame()
    except Exception as ex:
        logging.error(f"Time series: Unexpected error aggregating results for {param}: {ex}")
        return pd.DataFrame()

    cleaned_data = [(d, v) for d, v in zip(dates, vals) if d is not None and v is not None]
    if not cleaned_data:
        logging.warning(f"Time series: No valid data points found for {param} after aggregation.")
        return pd.DataFrame()

    df = pd.DataFrame(cleaned_data, columns=["Date", "Value"])
    df["Date"] = pd.to_datetime(df["Date"]) # Convert Date column
    df = df.dropna() # Drop rows where value might still be NaN
    df = df.rename(columns={"Value": param}) # Rename the 'Value' column to the parameter name

    return df

# -------------------------------------------------------------------
# RUN & VISUALIZE
# -------------------------------------------------------------------
if st.button("Run Monitoring"):
    try:
        # EDIT: Pass the correct user-selected geometry to the fetch function.
        layers, errors = fetch_layers(
            str(start_date),str(end_date),
            selected_geom, selected_params, ndvi_buffer
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
            center=selected_geom.centroid().coordinates().getInfo()[::-1],
            zoom=8
        )

        # EDIT: Optimized map overlay to avoid recalculating geometry.
        # This will draw a boundary unless "Whole Country" is selected.
        if roi_option != "Whole Country":
            roi_name = selected_district if selected_district else "Custom ROI"
            m.addLayer(selected_geom,
                        {"color":"red","fillOpacity":0},
                        f"{roi_name} Boundary")
            

        # Prepare for a single custom HTML legend
        legend_html_parts = []
        if visible: # Only build legend if there are visible layers
            legend_html_parts.append('<h4>Legend</h4>')


        for name in visible:
            cfg = PALETTES[name]
            mn,mx = cfg["min"],cfg["max"]
            pal   = cfg["palette"]
            mid   = (mn+mx)/2
            mid_col = pal[len(pal)//2]

            m.addLayer(layers[name],
                        {"min":mn,"max":mx,"palette":pal},
                        name)

            # Generate HTML for each legend entry
            # Example: <i style="background:rgb(255,0,0)"></i> Red
            # We need a gradient-like representation or min/mid/max for each parameter

            # For simplicity and to avoid the geemap bug, let's create a linear gradient representation
            # This is a common way to display continuous legends in Folium/HTML
            gradient_css = f"linear-gradient(to right, {pal[0]}, {pal[len(pal)//2]}, {pal[-1]})"
            legend_html_parts.append(f"""
                <p>{name}:</p>
                <div style="width: 100%; height: 20px; background: {gradient_css};"></div>
                <div style="display: flex; justify-content: space-between;">
                    <span>{mn}</span>
                    <span>{mid}</span>
                    <span>{mx}</span>
                </div>
                <br>
            """)

            # # COMMENT OUT OR DELETE ALL THE FOLLOWING LINES FOR m.add_legend
            # legend_colors = [hex_to_rgb(pal[0]), hex_to_rgb(mid_col), hex_to_rgb(pal[-1])]
            # m.add_legend(title=name,builtin_legend=False,
            #              labels=[f"{mn}",f"{mid}",f"{mx}"],
            #              colors=legend_colors)

        # Add the overall custom HTML legend to the map after the loop
        if legend_html_parts:
            legend_html = """
            <div style="position: fixed;
                         bottom: 50px; left: 50px; width: 250px; height: auto;
                         border:2px solid grey; z-index:9999; font-size:14px;
                         background-color:white; opacity:0.9; padding:10px;">
                {}
            </div>
            """.format("".join(legend_html_parts))

            m.get_root().html.add_child(folium.Element(legend_html))


        m.addLayerControl() # Keep this as it's useful for toggling layers

        m.to_streamlit(height=600)


        # Parameter means
        st.subheader("📊 Parameter Means")
        stats = {}
        for name,img in layers.items():
            try:
                # Ensure img is an ee.Image, even if it came from a collection
                ee_img = ee.Image(img)
                region_reducer_result = ee_img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=selected_geom,
                    scale=500, # Use scale, not maxPixels for default behavior
                    maxPixels=1e9 # Keep maxPixels for safety
                ).get(name)

                # Get the value, handling potential None from .get()
                val = region_reducer_result.getInfo() if region_reducer_result else None

                if isinstance(val, (int, float)):
                    stats[name] = round(val, 3)
                else:
                    stats[name] = "N/A" # Use 'N/A' as a string here for consistency

            except Exception as e_inner: # Renamed inner exception to avoid conflict
                # Log the specific error if a parameter mean calculation fails
                st.warning(f"Failed to calculate mean for {name}: {e_inner}")
                stats[name] = "Error" # Indicate an error occurred

        df_stats = pd.DataFrame(stats.items(),columns=["Parameter","Mean"])
        st.dataframe(df_stats)

        # Summary metrics
        st.subheader("📌 Summary Metrics")
        cols = st.columns(min(3,len(df_stats)))
        for i,row in df_stats.iterrows():
            cols[i%len(cols)].metric(str(row.Parameter), str(row.Mean)) 
            st.write(f"DEBUG: Parameter={row.Parameter}, Mean={row.Mean}, Type={type(row.Mean)}") # DEBUG LINE STILL HERE

        # Time series for all supported params
        ts_params = [p for p in selected_params if p in TIME_SERIES_PARAMS]
        if ts_params:
            st.subheader("📈 Time Series")
            for p in ts_params:
                # EDIT: Pass the correct user-selected geometry for time series.
                df_ts = extract_timeseries(
                    str(start_date),str(end_date),
                    selected_geom, p, ndvi_buffer
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
        st.error(f"An error occurred: Type of error: {type(e)}. Error message: {e}")
        st.exception(e) # This will print the full traceback for the actual error