# crop_rotation_model.py
import os, json, joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def train(
    input_csv="data/rotation_training.csv",
    target_col="recommended_crop",
    model_dir="models",
    test_size=0.20,
    random_state=42,
):
    df = pd.read_csv(input_csv)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {input_csv}")

    # Separate target FIRST
    y = df[target_col].astype("category")
    X = df.drop(columns=[target_col])

    # Detect column types
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    # Preprocessing
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        # use sparse=False for broad compatibility across sklearn versions
                        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Model
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=random_state,
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Fit
    pipe.fit(X_train, y_train)

    # Predict and metrics
    y_pred = pipe.predict(X_test)
    metrics = {
        "accuracy": float((y_pred == y_test).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
    }

    # Optional: top-2 accuracy (if proba available)
    try:
        proba = pipe.predict_proba(X_test)
        classes = pipe.named_steps["clf"].classes_
        idx_top2 = np.argsort(proba, axis=1)[:, -2:]
        top2_correct = [
            y_test.iloc[i] in classes[idx_top2[i]] for i in range(len(y_test))
        ]
        metrics["top2_accuracy"] = float(np.mean(top2_correct))
    except Exception:
        pass

    report = classification_report(y_test, y_pred, digits=3)
    cm = confusion_matrix(y_test, y_pred)

    # Persist
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "crop_rotation_pipeline.joblib")
    joblib.dump(pipe, model_path)  # consider compress=3 for smaller size

    meta = {
        "model_path": model_path,
        "target_col": target_col,
        "feature_schema": {"numeric": num_cols, "categorical": cat_cols},
        "class_labels": y.cat.categories.tolist(),
        "metrics": metrics,
    }
    with open(os.path.join(model_dir, "crop_rotation_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("=== Crop Rotation Advisor ===")
    print(report)
    print("Confusion Matrix:\n", cm)


if __name__ == "__main__":
    train()
