# crop_rotation_model_fixed.py
import os, json, joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def train(
    input_csv="data/rotation_training.csv",
    target_col="recommended_crop",
    model_dir="models",
    test_size=0.2,
    random_state=42,
):
    # Load dataset
    df = pd.read_csv(input_csv)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset!")

    # Separate features & target
    y_raw = df[target_col].astype("category")
    X = df.drop(columns=[target_col])

    # Encode target labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # Identify categorical and numeric features
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    print(f"Categorical columns: {cat_cols}")
    print(f"Numeric columns: {num_cols}")
    print("Target classes:", list(le.classes_))

    # Preprocessing: handle missing + encode categoricals
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Model: simpler RandomForest for small datasets
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,  # limit depth to avoid overfitting small datasets
        min_samples_leaf=1,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state,
    )

    # Full pipeline
    pipe = Pipeline(steps=[("pre", preprocessor), ("clf", clf)])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Train model
    pipe.fit(X_train, y_train)

    # Predictions
    y_pred = pipe.predict(X_test)

    # Evaluation
    print("=== Crop Rotation Advisor ===")
    print(classification_report(y_test, y_pred, target_names=le.classes_, digits=3))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save model + metadata
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "crop_rotation_pipeline.joblib")
    joblib.dump(pipe, model_path)

    meta = {
        "model_path": model_path,
        "target_col": target_col,
        "features": {"numeric": num_cols, "categorical": cat_cols},
        "class_labels": list(le.classes_),
        "label_encoder": {cls: int(i) for i, cls in enumerate(le.classes_)},
    }
    with open(os.path.join(model_dir, "crop_rotation_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"✅ Model saved at: {model_path}")


if __name__ == "__main__":
    train()
