import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import json

# === STEP 1: Load Labeled Field Data ===
# Assumed format: CSV with lat, lon, label (0=crop, 1=weed), plus optional metadata
csv_path = "data\crop_weed_training.csv"
df = pd.read_csv(csv_path)

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(df.x, df.y)]
# If using lat/lon columns, replace df.x and df.y with df.longitude and df.latitude
gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# === STEP 2: Extract Features from Sentinel-2 or Earth Engine ===
# For now, assume spectral bands already exist in the CSV (e.g., B2, B3, ..., B12)
features = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B11", "B12"]
X = df[features]
y = df["label"]

# === STEP 3: Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === STEP 4: Train Classifier ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === STEP 5: Evaluate Model ===
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["Crop", "Weed"]))

# === STEP 6: Save Model ===
joblib.dump(clf, "models/crop_classifier_model.pkl")
print(f"Model saved to models/crop_classifier_model.pkl")

# Save feature names
with open("models/crop_classifier_features.json", "w") as f:
    json.dump(list(X.columns), f)
