import sys
import os
import pandas as pd
import folium

# ==============================================================================
# Step 0: Initial Configuration and Path Setup
# ==============================================================================

# Get the project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Configuration ---
# The path to your prediction results CSV file
predictions_csv_path = "data/predictions_output.csv"
# The name of the output HTML map file
output_map_path = "data/predictions_map.html"

# ==============================================================================
# Main Mapping Pipeline
# ==============================================================================

try:
    print(f"Loading predictions from {predictions_csv_path}...")
    df_predictions = pd.read_csv(predictions_csv_path)
    print(f"Successfully loaded {len(df_predictions)} predictions.")
except FileNotFoundError:
    print(f"Error: Predictions CSV file not found at '{predictions_csv_path}'.")
    print("Please run 'predict_crop_weeds.py' first to generate the predictions.")
    sys.exit(1)

# Check if the required columns exist
if 'latitude' not in df_predictions.columns or 'longitude' not in df_predictions.columns or 'predicted_label' not in df_predictions.columns:
    print("Error: The CSV file is missing 'latitude', 'longitude', or 'predicted_label' columns.")
    sys.exit(1)

# --- Create the Map ---
print("\nCreating the interactive map...")
# Get the average coordinates to center the map
center_lat = df_predictions['latitude'].mean()
center_lon = df_predictions['longitude'].mean()

# Create a Folium map object, centered on your data
predictions_map = folium.Map(location=[center_lat, center_lon], zoom_start=17)

# --- Add Points to the Map ---
# Define colors for each predicted class
color_mapping = {'crop': 'green', 'weed': 'red'}

# Iterate over each row of the DataFrame and add a point to the map
for index, row in df_predictions.iterrows():
    # Set the color based on the predicted label
    marker_color = color_mapping.get(row['predicted_label'], 'blue') # Default to blue if class is unknown
    
    # Create a small circle marker for each point
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=3, # Adjust the size of the points
        color=marker_color,
        fill=True,
        fill_color=marker_color,
        fill_opacity=0.7,
        tooltip=f"Predicted: {row['predicted_label']}" # Show the label on hover
    ).add_to(predictions_map)

# --- Save the Map ---
os.makedirs("data", exist_ok=True)
predictions_map.save(output_map_path)
print(f"\nInteractive map saved to {output_map_path}")
print("You can open this HTML file in your web browser to view the map.")