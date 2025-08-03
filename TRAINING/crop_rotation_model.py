# crop_rotation_model.py
import pandas as pd, numpy as np, os, joblib, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("data/rotation_training.csv")

df = pd.get_dummies(df, drop_first=True)

X = df.drop("recommended_crop", axis=1)
y = df["recommended_crop"]

with open("models/crop_rotation_model_features.json", "w") as f:
    json.dump(list(X.columns), f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/crop_rotation_model.pkl")

print("=== Crop Rotation Advisor ===")
print(classification_report(y_test, model.predict(X_test)))
print("Confusion Matrix:\n", confusion_matrix(y_test, model.predict(X_test)))
