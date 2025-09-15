# irrigation_model.py
import pandas as pd, numpy as np, os, joblib, json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging
from datetime import datetime

# --- Configuration (can be moved to a separate file) ---
CONFIG = {
    "data_path": "data/irrigation_training.csv",
    "model_dir": "models",
    "target": "optimal_water_liters",
    "test_size": 0.2,
    "random_state": 42,
    "hyperparameters": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_leaf": [1, 2, 4],
    },
    "version": "1.0.0"
}

# --- Set up logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path):
    """Loads and preprocesses the dataset."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data from {file_path}")
        return df
    except FileNotFoundError:
        logging.error(f"Error: The file {file_path} was not found.")
        return None

def preprocess_data(df, target_column):
    """
    Performs one-hot encoding and separates features from target.
    Handles potential missing values with imputation.
    """
    # Identify categorical columns for one-hot encoding
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Separate features (X) and target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    return X, y

def train_model(X_train, y_train, params):
    """
    Trains a Random Forest Regressor using a pipeline for preprocessing
    and performs hyperparameter tuning with GridSearchCV.
    """
    # Create a pipeline with imputation, scaling, and the model
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')), # Impute missing values with the mean
        ('scaler', StandardScaler()),                 # Scale numerical features
        ('model', RandomForestRegressor(random_state=CONFIG["random_state"], n_jobs=-1))
    ])

    # Use GridSearchCV for hyperparameter tuning
    logging.info("Starting hyperparameter tuning with GridSearchCV...")
    grid_search = GridSearchCV(pipeline, params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logging.info(f"Best hyperparameters found: {grid_search.best_params_}")
    return best_model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and logs key metrics."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logging.info("--- Model Evaluation ---")
    logging.info(f"Mean Squared Error (MSE): {mse:.2f}")
    logging.info(f"Mean Absolute Error (MAE): {mae:.2f}")
    logging.info(f"R2 Score: {r2:.3f}")

def save_artifacts(model, feature_names, version, model_dir):
    """Saves the trained model and feature names with versioning."""
    os.makedirs(model_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model with a versioned filename
    model_filename = os.path.join(model_dir, f"irrigation_optimizer_v{version}_{timestamp}.pkl")
    joblib.dump(model, model_filename)
    logging.info(f"Model saved to {model_filename}")

    # Save feature names with a versioned filename
    features_filename = os.path.join(model_dir, f"irrigation_model_features_v{version}.json")
    with open(features_filename, "w") as f:
        json.dump(feature_names, f)
    logging.info(f"Feature names saved to {features_filename}")

def main():
    """Main function to run the ETL pipeline."""
    # 1. Extract
    df = load_data(CONFIG["data_path"])
    if df is None:
        return

    # 2. Transform
    X, y = preprocess_data(df, CONFIG["target"])

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["test_size"], random_state=CONFIG["random_state"]
    )
    
    # 3. Train and Evaluate
    best_model = train_model(X_train, y_train, {"model__" + k: v for k, v in CONFIG["hyperparameters"].items()})
    evaluate_model(best_model, X_test, y_test)
    
    # 4. Load (Save Artifacts)
    save_artifacts(best_model, list(X.columns), CONFIG["version"], CONFIG["model_dir"])

if __name__ == "__main__":
    main()