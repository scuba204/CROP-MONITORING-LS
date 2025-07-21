# palettes.py

def get_palettes():
    """
    Returns the color palettes and value ranges for each parameter.
    """

    return {
        # Changed named colors to hex codes
        "NDVI": { "min": 0,   "max": 1,   "palette": ["#A52A2A", "#FFFF00", "#008000"] }, # Brown, Yellow, Green
        "Precipitation": { "min": 0,   "max": 20,  "palette": ["#FFFFFF", "#0000FF"] }, # White, Blue
        "Land Surface Temp": { "min": 0,   "max": 40,  "palette": ["#0000FF", "#FFFF00", "#FF0000"] }, # Blue, Yellow, Red
        "Humidity":          { "min": 0,   "max": 100, "palette": ["#FFFFFF", "#008000"] }, # White, Green
        "Irradiance":        { "min": 0,   "max": 300, "palette": ["#FFFFFF", "#FFA500"] }, # White, Orange
        "Evapotranspiration": { "min": 0, "max": 50,  "palette": ["#FFFFFF", "#FFA500"] }, # White, Orange

        "Soil Moisture":       { "min": 0,   "max": 0.5, "palette": ["#FFFFFF", "#0000FF"] }, # White, Blue
        "Soil Organic Matter": { "min": 0,   "max": 8,   "palette": ["#FFFFFF", "#000000"] }, # White, Black
        "Soil pH":             { "min": 3,   "max": 9,   "palette": ["#FF0000", "#FFFF00", "#008000"] }, # Red, Yellow, Green
        "Soil CEC":            { "min": 0,   "max": 40,  "palette": ["#FFFFFF", "#0000FF"] }, # White, Blue
        "Soil Nitrogen":       { "min": 0,   "max": 0.5, "palette": ["#FFFFFF", "#008000"] }, # White, Green

        "Soil Texture - Clay": { "min": 0,   "max": 100, "palette": ["#FFFFFF", "#A52A2A"] }, # White, Brown
        "Soil Texture - Silt": { "min": 0,   "max": 100, "palette": ["#FFFFFF", "#808080"] }, # White, Grey
        "Soil Texture - Sand": { "min": 0,   "max": 100, "palette": ["#FFFFFF", "#FFFF00"] }, # White, Yellow

        "B5":  { "min": 0, "max": 1, "palette": ["#000000", "#FFFFFF"] }, # Black, White
        "B6":  { "min": 0, "max": 1, "palette": ["#000000", "#FFFFFF"] }, # Black, White
        "B7":  { "min": 0, "max": 1, "palette": ["#000000", "#FFFFFF"] }, # Black, White
        "B11": { "min": 0, "max": 1, "palette": ["#000000", "#FFFFFF"] }, # Black, White
        "B12": { "min": 0, "max": 1, "palette": ["#000000", "#FFFFFF"] }, # Black, White
    }
