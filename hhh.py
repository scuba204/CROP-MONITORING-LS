import os
import geopandas as gpd

shp = r"C:\Users\MY PC\Documents\GIS DATA\BOUNDARIES\LSO_adm\LSO_adm1.shp"
gdf = gpd.read_file(shp)
print(gdf.columns)