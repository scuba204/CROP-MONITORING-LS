import pandas as pd
import folium
import os

# Load predictions CSV
csv_path = os.path.join(os.path.dirname(__file__), '../data/predictions_output.csv')
df = pd.read_csv(csv_path)

# Center map on field
center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles='OpenStreetMap')

# Create feature groups for dropdown filtering
crop_layer = folium.FeatureGroup(name='Crop', show=True)
weed_layer = folium.FeatureGroup(name='Weed', show=True)

# Add points to respective layers
for _, row in df.iterrows():
    marker = folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=2.5,
        color='green' if row['predicted_label'] == 'crop' else 'red',
        fill=True,
        fill_opacity=0.7,
        popup=f"{row['predicted_label'].capitalize()}"
    )

    if row['predicted_label'] == 'crop':
        marker.add_to(crop_layer)
    else:
        marker.add_to(weed_layer)

# Add layers to map
crop_layer.add_to(m)
weed_layer.add_to(m)

# Add layer control dropdown
folium.LayerControl(collapsed=False).add_to(m)

# Save HTML map
output_path = os.path.join(os.path.dirname(__file__), 'crop_weed_map_filtered.html')
m.save(output_path)
print(f"✅ Interactive filtered map saved as: {output_path}")
