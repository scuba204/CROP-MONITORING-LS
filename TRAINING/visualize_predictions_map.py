import pandas as pd
import folium
import os

# Load the predictions CSV
csv_path = os.path.join(os.path.dirname(__file__), '../data/predictions_output.csv')
df = pd.read_csv(csv_path)

# Create map centered around the mean location
center_lat = df['latitude'].mean()
center_lon = df['longitude'].mean()
m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles='OpenStreetMap')

# Define color mapping
color_map = {'crop': 'green', 'weed': 'red'}

# Add points to map
for _, row in df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=2.5,
        color=color_map.get(row['predicted_label'], 'gray'),
        fill=True,
        fill_opacity=0.7,
        popup=f"{row['predicted_label'].capitalize()}",
    ).add_to(m)

# Save the map
output_path = os.path.join(os.path.dirname(__file__), 'crop_weed_map.html')
m.save(output_path)
print(f"✅ Interactive crop vs weed map saved as: {output_path}")
