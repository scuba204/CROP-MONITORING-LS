import pandas as pd
import folium
import os
import numpy as np

# Load predictions CSV
csv_path = os.path.join(os.path.dirname(__file__), '../data/predictions_output.csv')
df = pd.read_csv(csv_path)

# # === Apply grid-style spacing simulation ===
# # Approx ~5 meters between rows and columns
# row_spacing = 0.00005
# col_spacing = 0.00005

# df['latitude'] = np.round(df['latitude'] / row_spacing) * row_spacing
# df['longitude'] = np.round(df['longitude'] / col_spacing) * col_spacing

# commented  out this above block to show all points individually

# Center the map on the field
center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=20, tiles='OpenStreetMap')

# Create feature groups
# crop_layer = folium.FeatureGroup(name='Crop', show=True)
# weed_layer = folium.FeatureGroup(name='Weed', show=True)

# Add styled markers to each layer
# for _, row in df.iterrows():
#     color = 'green' if row['predicted_label'] == 'crop' else 'red'
#     marker = folium.CircleMarker(
#         location=[row['latitude'], row['longitude']],
#         radius=4,
#         color='black',
#         weight=0.5,
#         fill=True,
#         fill_color=color,
#         fill_opacity=0.75,
#         popup=f"{row['predicted_label'].capitalize()}"
#     )
    
#     if row['predicted_label'] == 'crop':
#         marker.add_to(crop_layer)
#     else:
#         marker.add_to(weed_layer)

# commented the above block to change visualization

# Create FeatureGroups
crop_layer = folium.FeatureGroup(name='Crop 🥬', show=True)
weed_layer = folium.FeatureGroup(name='Weed 🌿', show=True)

# Add emoji markers
for _, row in df.iterrows():
    if row['predicted_label'] == 'crop':
        emoji = "🥬"
        layer = crop_layer
    else:
        emoji = "🌿"
        layer = weed_layer

    folium.Marker(
        location=[row['latitude'], row['longitude']],
        icon=folium.DivIcon(html=f"""<div style="font-size:20px;">{emoji}</div>"""),
        popup=row['predicted_label'].capitalize()
    ).add_to(layer)

# Add layers to map
crop_layer.add_to(m)
weed_layer.add_to(m)

# Add dropdown filter control
folium.LayerControl(collapsed=False).add_to(m)

# Save HTML map
output_path = os.path.join(os.path.dirname(__file__), 'crop_weed_map_filtered_spaced.html')
m.save(output_path)

print(f"✅ Interactive filtered + spaced map saved as: {output_path}")
