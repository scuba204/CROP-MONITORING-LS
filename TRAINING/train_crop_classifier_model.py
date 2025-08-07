#!/usr/bin/env python3
"""
Train a crop-vs-weed RandomForest classifier on multitemporal Sentinel-2 features.

Features:
  - CLI arguments for I/O, test split, random seed, search iterations
  - Modular functions for loading, pivoting, engineering, training, and saving
  - Logging instead of prints for progress tracing
  - RandomizedSearchCV for hyperparameter tuning
  - SHAP-based feature importance summary
  - Persisted classification report and SHAP summary as CSV
"""
print("🚀 Script start! __name__ =", __name__)   # ← sanity check

import os
import sys
import json
import argparse
import logging

import pandas as pd
import numpy as np
import joblib
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, balanced_accuracy_score

# ------------------------------------------------------------------------------
# Default configurations and hyperparameter distributions
# ------------------------------------------------------------------------------

DEFAULT_FEATURE_CSV = "data/features.csv"
DEFAULT_LABEL_CSV   = "data/crop_weed_training.csv"
DEFAULT_OUTDIR      = "models"
DEFAULT_N_ITER      = 20

PARAM_DISTS = {
    "n_estimators":      [100, 200, 500],
    "max_depth":         [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4]
}

# ------------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train crop-vs-weed RF on multitemporal Sentinel-2 features"
    )
    parser.add_argument(
        "--features", "-f",
        default=DEFAULT_FEATURE_CSV,
        help="Path to features CSV (id, date, band/index columns)"
    )
    parser.add_argument(
        "--labels", "-l",
        default=DEFAULT_LABEL_CSV,
        help="Path to labels CSV"
    )
    parser.add_argument(
        "--outdir", "-o",
        default=DEFAULT_OUTDIR,
        help="Directory to save model, feature list, reports, and SHAP summary"
    )
    parser.add_argument(
        "--test-size", "-t",
        type=float,
        default=0.2,
        help="Proportion of data to use for the test split"
    )
    parser.add_argument(
        "--random-state", "-r",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n-iter", "-n",
        type=int,
        default=DEFAULT_N_ITER,
        help="Number of iterations for RandomizedSearchCV"
    )
    return parser.parse_args()

# ------------------------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------------------------

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s ▶ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True                  # ← drop any old handlers
    )
logging.info("✅ setup_logging() was called and logger is live")

# ------------------------------------------------------------------------------
# Data loading, pivoting, and feature engineering
# ------------------------------------------------------------------------------

def load_and_pivot(features_path, labels_path):
    """
    - Loads features with columns [id, date, metric1, metric2, …]
    - Pivots to wide format so each metric_date becomes a column
    - Loads labels, creates an 'id' column if missing
    - Merges labels, then engineers day-to-day delta features
    Returns: X DataFrame, y Series, list of feature columns
    """
    logging.info("Loading features from %s", features_path)
    df = pd.read_csv(features_path, parse_dates=["date"])
    if df.empty:
        raise ValueError("Feature DataFrame is empty!")

    # Pivot from long to wide
    metrics = [c for c in df.columns if c not in ("id", "date")]
    logging.info("Pivoting %d metrics × %d dates", len(metrics), df.date.nunique())
    df_wide = (
        df
        .pivot_table(index="id", columns="date", values=metrics)
        .sort_index(axis=1)
    )

    # Flatten MultiIndex columns
    df_wide.columns = [
        f"{metric}_{dt.strftime('%Y%m%d')}"
        for metric, dt in df_wide.columns
    ]
    df_wide.reset_index(inplace=True)

    # Ensure the 'id' in features is string
    df_wide["id"] = df_wide["id"].astype(str)

    # Load labels
    logging.info("Loading labels from %s", labels_path)
    labels = pd.read_csv(labels_path)

    # If no 'id' column, create from row index
    if "id" not in labels.columns:
        logging.info("No 'id' in labels CSV; creating 'id' from row index")
        labels["id"] = labels.index.astype(str)

    # Label encoding via map (avoids FutureWarning)
    if "label" not in labels.columns:
        raise KeyError("'label' column missing in labels CSV")
    labels["label"] = labels["label"].map({"crop": 0, "weed": 1})

    # Ensure label 'id' is string too
    labels["id"] = labels["id"].astype(str)

    # Keep only id + label
    labels = labels[["id", "label"]]

    # Merge features + labels
    df_final = df_wide.merge(labels, on="id", how="inner")
    lost = df_wide.shape[0] - df_final.shape[0]
    if lost > 0:
        logging.warning("Dropped %d samples during merge (ID mismatch)", lost)

    # Engineer day-to-day deltas
    logging.info("Engineering delta features")
    date_cols = sorted(
        [c for c in df_final.columns if "_" in c and c.split("_")[-1].isdigit()]
    )
    for metric in set(c.rsplit("_", 1)[0] for c in date_cols):
        cols = sorted([c for c in date_cols if c.startswith(metric + "_")])
        for prev, curr in zip(cols, cols[1:]):
            delta_col = f"{metric}_Δ_{curr.split('_')[-1]}"
            df_final[delta_col] = df_final[curr] - df_final[prev]

    feature_cols = [c for c in df_final.columns if c not in ("id", "label")]
    X = df_final[feature_cols]
    y = df_final["label"].astype(int)

    logging.info("Prepared X (%d samples, %d features)", X.shape[0], X.shape[1])
    return X, y, feature_cols

# ------------------------------------------------------------------------------
# Training, tuning, and evaluation
# ------------------------------------------------------------------------------

def train_and_tune(X, y, test_size, random_state, n_iter):
    """
    - Splits into train/test
    - Runs RandomizedSearchCV on RandomForestClassifier
    - Returns the best model, X_test, y_test, and a DataFrame of the classification report
    """
    logging.info("Splitting data (test_size=%.2f)", test_size)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    logging.info(
        "Running RandomizedSearchCV (%d iterations over %d hyperparams)",
        n_iter, len(PARAM_DISTS)
    )
    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(
            class_weight="balanced", random_state=random_state
        ),
        param_distributions=PARAM_DISTS,
        n_iter=n_iter,
        cv=5,
        scoring="balanced_accuracy",
        n_jobs=-1,
        random_state=random_state,
        verbose=2
    )
    search.fit(X_train, y_train)

    best_clf = search.best_estimator_
    logging.info("Best params: %s", search.best_params_)
    logging.info("CV balanced accuracy: %.4f", search.best_score_)

    # Evaluate on test
    y_pred = best_clf.predict(X_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    logging.info("Test balanced accuracy: %.4f", bal_acc)

    report = classification_report(
        y_test, y_pred, target_names=["crop", "weed"], output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    return best_clf, X_test, y_test, report_df

# ------------------------------------------------------------------------------
# SHAP explainability
# ------------------------------------------------------------------------------

def compute_shap_summary(model, X_test, feature_cols, outdir):
    """
    Computes and saves mean absolute SHAP values per feature.
    """
    logging.info("Computing SHAP values")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_abs = np.abs(shap_values[1]).mean(axis=0)
    df_shap = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False)

    path = os.path.join(outdir, "shap_summary.csv")
    df_shap.to_csv(path, index=False)
    logging.info("SHAP summary saved to %s", path)

# ------------------------------------------------------------------------------
# Save all artifacts
# ------------------------------------------------------------------------------

def save_artifacts(model, feature_cols, report_df, outdir):
    os.makedirs(outdir, exist_ok=True)

    # Model
    model_path = os.path.join(outdir, "crop_vs_weed_model.pkl")
    joblib.dump(model, model_path)
    logging.info("Model saved to %s", model_path)

    # Feature list
    feats_path = os.path.join(outdir, "feature_list.json")
    with open(feats_path, "w") as f:
        json.dump(feature_cols, f)
    logging.info("Feature list saved to %s", feats_path)

    # Classification report
    report_path = os.path.join