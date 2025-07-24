# palettes.py

def get_palettes():
    """
    Returns the color palettes and value ranges for each parameter.
    """

    return {
        # Vegetation Indices
        "NDVI": { "min": 0,   "max": 1,   "palette": ["#A52A2A", "#FFFF00", "#008000"] }, # Brown, Yellow, Green
        "SAVI": { "min": 0,   "max": 1,   "palette": ["#A52A2A", "#FFFF00", "#008000"] }, # Brown, Yellow, Green (similar to NDVI)
        "EVI":  { "min": 0,   "max": 1,   "palette": ["#A52A2A", "#FFFF00", "#008000"] }, # Brown, Yellow, Green (similar to NDVI)

        # Water/Moisture Indices
        "NDWI": { "min": -1,  "max": 1,   "palette": ["#FF0000", "#FFFFFF", "#0000FF"] }, # Red (dry), White, Blue (wet)
        "NDMI": { "min": -1,  "max": 1,   "palette": ["#FF0000", "#FFFFFF", "#0000FF"] }, # Red (dry), White, Blue (wet) (similar to NDWI)
        "Soil Moisture":     { "min": 0,   "max": 0.5, "palette": ["#FFFFFF", "#0000FF"] }, # White, Blue

        # Atmospheric/Climatic Parameters
        "Precipitation": { "min": 0,   "max": 20,  "palette": ["#FFFFFF", "#0000FF"] }, # White, Blue
        "Land Surface Temp": { "min": 0,   "max": 40,  "palette": ["#0000FF", "#FFFF00", "#FF0000"] }, # Blue, Yellow, Red
        "Humidity":          { "min": 0,   "max": 100, "palette": ["#FFFFFF", "#008000"] }, # White, Green
        "Irradiance":        { "min": 0,   "max": 300, "palette": ["#FFFFFF", "#FFA500"] }, # White, Orange
        "Evapotranspiration": { "min": 0,  "max": 50,  "palette": ["#FFFFFF", "#FFA500"] }, # White, Orange

        # Soil Properties
        "Soil Organic Matter": { "min": 0,   "max": 8,   "palette": ["#FFFFFF", "#000000"] }, # White, Black
        "Soil pH":             { "min": 3,   "max": 9,   "palette": ["#FF0000", "#FFFF00", "#008000"] }, # Red, Yellow, Green
        "Soil CEC":            { "min": 0,   "max": 40,  "palette": ["#FFFFFF", "#0000FF"] }, # White, Blue
        "Soil Nitrogen":     { "min": 0,   "max": 0.5, "palette": ["#FFFFFF", "#008000"] }, # White, Green

        # Soil Texture Components
        "Soil Texture - Clay": { "min": 0,   "max": 100, "palette": ["#FFFFFF", "#A52A2A"] }, # White, Brown
        "Soil Texture - Silt": { "min": 0,   "max": 100, "palette": ["#FFFFFF", "#808080"] }, # White, Grey
        "Soil Texture - Sand": { "min": 0,   "max": 100, "palette": ["#FFFFFF", "#FFFF00"] }, # White, Yellow

        # Simulated Hyperspectral (Sentinel-2 bands)
        "B2":  { "min": 0, "max": 0.3, "palette": ["#000000", "#0000FF"] }, # Blue (typical range)
        "B3":  { "min": 0, "max": 0.3, "palette": ["#000000", "#008000"] }, # Green (typical range)
        "B4":  { "min": 0, "max": 0.3, "palette": ["#000000", "#FF0000"] }, # Red (typical range)
        "B5":  { "min": 0, "max": 0.5, "palette": ["#000000", "#FFFFFF"] }, # Black, White
        "B6":  { "min": 0, "max": 0.6, "palette": ["#000000", "#FFFFFF"] }, # Black, White
        "B7":  { "min": 0, "max": 0.7, "palette": ["#000000", "#FFFFFF"] }, # Black, White
        "B8A": { "min": 0, "max": 0.8, "palette": ["#000000", "#FFFFFF"] }, # Black, White (NIR)
        "B11": { "min": 0, "max": 0.5, "palette": ["#000000", "#FFFFFF"] }, # Black, White (SWIR1)
        "B12": { "min": 0, "max": 0.4, "palette": ["#000000", "#FFFFFF"] }, # Black, White (SWIR2)
    }