# palettes.py

def get_palettes():
    """
    Returns the color palettes and value ranges for each parameter.
    """

    return {
        "NDVI": { "min": 0,   "max": 1,   "palette": ["brown", "yellow", "green"] },
        "Precipitation": { "min": 0,   "max": 20,  "palette": ["white", "blue"] },
        "Land Surface Temp": { "min": 0,   "max": 40,  "palette": ["blue", "yellow", "red"] },
        "Humidity":        { "min": 0,   "max": 100, "palette": ["white", "green"] },
        "Irradiance":      { "min": 0,   "max": 300, "palette": ["white", "orange"] },
        "Evapotranspiration": { "min": 0, "max": 50,  "palette": ["white", "orange"] },

        "Soil Moisture":       { "min": 0,   "max": 0.5, "palette": ["white", "blue"] },
        "Soil Organic Matter": { "min": 0,   "max": 8,   "palette": ["white", "black"] },
        "Soil pH":             { "min": 3,   "max": 9,   "palette": ["red", "yellow", "green"] },
        "Soil CEC":            { "min": 0,   "max": 40,  "palette": ["white", "blue"] },
        "Soil Nitrogen":       { "min": 0,   "max": 0.5, "palette": ["white", "green"] },

        "Soil Texture - Clay": { "min": 0,   "max": 100, "palette": ["white", "brown"] },
        "Soil Texture - Silt": { "min": 0,   "max": 100, "palette": ["white", "grey"] },
        "Soil Texture - Sand": { "min": 0,   "max": 100, "palette": ["white", "yellow"] },

        "B5":  { "min": 0, "max": 1, "palette": ["black", "white"] },
        "B6":  { "min": 0, "max": 1, "palette": ["black", "white"] },
        "B7":  { "min": 0, "max": 1, "palette": ["black", "white"] },
        "B11": { "min": 0, "max": 1, "palette": ["black", "white"] },
        "B12": { "min": 0, "max": 1, "palette": ["black", "white"] },
    }
