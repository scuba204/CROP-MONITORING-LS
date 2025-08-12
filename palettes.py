# palettes.py
def get_palettes():
    """
    Returns the color palettes and value ranges for each parameter.
    """

    return {
        # Vegetation Indices
        "NDVI": { "min": 0, "max": 1, "palette": ["#A52A2A", "#FFFF00", "#008000"] },  # Brown, Yellow, Green
        "SAVI": { "min": 0, "max": 1, "palette": ["#A52A2A", "#FFFF00", "#008000"] },  # Brown, Yellow, Green (similar to NDVI)
        "EVI":  { "min": 0, "max": 1, "palette": ["#A52A2A", "#FFFF00", "#008000"] },  # Brown, Yellow, Green (similar to NDVI)
        # New Vegetation Indices with distinct palettes
        "NDRE": { "min": 0, "max": 1, "palette": ["#5E3C99", "#B2DF8A", "#1B5E20"] },  # Purple, Light Green, Dark Green
        "MSI":  { "min": 0, "max": 2, "palette": ["#FFFFB3", "#FE9929", "#E31A1C"] },  # Yellow, Orange, Red (Low to High Stress)
        "OSAVI":{ "min": 0, "max": 1, "palette": ["#4292C6", "#2171B5", "#084594"] },  # Light Blue, Medium Blue, Dark Blue
        "GNDVI":{ "min": 0, "max": 1, "palette": ["#D9F0A3", "#78C679", "#238443"] },  # Pale Green, Medium Green, Dark Green
        "RVI":  { "min": 0, "max": 1, "palette": ["#FF0000", "#FFFF00", "#00FF00"] },  # Red, Yellow, Green

        # Water/Moisture Indices
        "NDWI": { "min": -1, "max": 1, "palette": ["#FF0000", "#FFFFFF", "#0000FF"] },  # Red (dry), White, Blue (wet)
        "NDMI": { "min": -1, "max": 1, "palette": ["#FF0000", "#FFFFFF", "#0000FF"] },  # Red (dry), White, Blue (wet) (similar to NDWI)
        "Soil Moisture": { "min": 0, "max": 0.5, "palette": ["#FFFFFF", "#0000FF"] },   # White, Blue

        # Atmospheric/Climatic Parameters
        "Precipitation": { "min": 0, "max": 20, "palette": ["#FFFFFF", "#0000FF"] },  # White, Blue
        "Land Surface Temp": { "min": 0, "max": 40, "palette": ["#0000FF", "#FFFF00", "#FF0000"] },  # Blue, Yellow, Red
        "Humidity": { "min": 0, "max": 100, "palette": ["#FFFFFF", "#008000"] },  # White, Green
        "Irradiance": { "min": 0, "max": 300, "palette": ["#FFFFFF", "#FFA500"] },  # White, Orange
        "Evapotranspiration": { "min": 0, "max": 50, "palette": ["#FFFFFF", "#FFA500"] },  # White, Orange

        # Soil Properties
        "Soil Organic Matter": { "min": 0, "max": 8, "palette": ["#FFFFFF", "#000000"] },  # White, Black
        "Soil pH": { "min": 3, "max": 9, "palette": ["#FF0000", "#FFFF00", "#008000"] },  # Red, Yellow, Green
        "Soil CEC": { "min": 0, "max": 40, "palette": ["#FFFFFF", "#0000FF"] },  # White, Blue
        "Soil Nitrogen": { "min": 0, "max": 0.5, "palette": ["#FFFFFF", "#008000"] },  # White, Green

        # Soil Texture Components
        "Soil Texture - Clay": { "min": 0, "max": 100, "palette": ["#FFFFFF", "#A52A2A"] },  # White, Brown
        "Soil Texture - Silt": { "min": 0, "max": 100, "palette": ["#FFFFFF", "#808080"] },  # White, Grey
        "Soil Texture - Sand": { "min": 0, "max": 100, "palette": ["#FFFFFF", "#FFFF00"] },  # White, Yellow

       
    }