# train_rotation_model.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib


def train_and_evaluate_model():
    # === Load dataset ===
    # Correct way to join the directory and filename
    file_path = os.path.join("data", "rotation_training_clean.csv")
    df = pd.read_csv(file_path)

    # === Select features & target ===
    numeric_features = [
        "Year", "area_share", "prev_area_share", "ET", "LST_Day_1km",
        "NDVI", "precipitation", "rel_humidity",
        "surface_solar_radiation_downwards", "soil_ph", "soil_soc",
        "soil_nitrogen", "soil_cec"
    ]
    categorical_features = ["Item"]
    target = "Yield"

    X = df[numeric_features + categorical_features]
    y = df[target]

    # === Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Create preprocessing and model pipeline ===
    # Use ColumnTransformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Create the full pipeline with a regressor
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_jobs=-1, random_state=42))
    ])

    # === Hyperparameter Tuning with RandomizedSearchCV ===
    param_grid = {
        'regressor__n_estimators': [100, 200, 300, 400],
        'regressor__max_depth': [10, 20, 30, None],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4],
    }

    print("🔎 Starting hyperparameter search...")
    random_search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        scoring='r2'
    )

    random_search.fit(X_train, y_train)
    best_pipe = random_search.best_estimator_

    print("\n✅ Model trained and tuned")
    print("Best parameters found:", random_search.best_params_)

    # === Evaluate on the test set ===
    y_pred = best_pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n=== Model Evaluation ===")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R²: {r2:.4f}")

    # === Cross-validation with the best model ===
    cv_scores = cross_val_score(best_pipe, X_train, y_train, cv=5, scoring="r2")
    print("\nCV R² scores:", cv_scores)
    print("Mean CV R²:", cv_scores.mean())

    # === Save the best pipeline model ===
    joblib.dump(best_pipe, "rotation_yield_pipeline.pkl")
    print("\n✅ Best model pipeline saved as 'rotation_yield_pipeline.pkl'")

if __name__ == "__main__":
    train_and_evaluate_model()