# predict_crop.py
import joblib
import json
import pandas as pd
import os

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "crop_rotation_pipeline.joblib")
META_PATH = os.path.join(MODEL_DIR, "crop_rotation_metadata.json")


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")
    if not os.path.exists(META_PATH):
        raise FileNotFoundError(f"Metadata file not found at {META_PATH}")

    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    return model, meta


def predict(last_crop, year, area_ha, yield_kg_ha):
    model, meta = load_model()

    # Build input dataframe (must match training schema)
    input_df = pd.DataFrame([{
        "Year": year,
        "last_crop": last_crop,
        "area_ha": area_ha,
        "yield_kg_ha": yield_kg_ha
    }])

    # Predict
    prediction = model.predict(input_df)[0]

    # Optionally get probabilities
    proba = model.predict_proba(input_df)[0]
    proba_dict = dict(zip(model.classes_, proba))

    return prediction, proba_dict


if __name__ == "__main__":
    # Example usage
    pred, proba = predict(
        last_crop="Maize (corn)",
        year=2020,
        area_ha=5000,
        yield_kg_ha=1200
    )
    print("🌱 Recommended next crop:", pred)
    print("📊 Probabilities:", proba)
