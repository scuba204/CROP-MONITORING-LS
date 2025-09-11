#!/usr/bin/env python3
"""
train_disease_model.py
Train a disease risk model using extracted Sentinel-2 features + indices.
(With SMOTE for class imbalance)
"""

import os
import sys
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE   # ✅ SMOTE for balancing

# --------------------------------------------------
# Ensure project root is in sys.path
# --------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scripts.data_loader import load_config  # now works anywhere

# --------------------------------------------------
# Load config
# --------------------------------------------------
config_path = os.path.join(project_root,"configs", "config_disease.yaml")
cfg = load_config(config_path)

csv_path = cfg["data"]["csv_path"]
model_out = os.path.join(project_root, "MODELS", cfg["model"]["name"])
features_out = os.path.join(project_root, "MODELS", cfg["model"]["features_file"])

print(f"📂 Loading dataset: {csv_path}")
df = pd.read_csv(csv_path)

# --------------------------------------------------
# Define features & labels
# --------------------------------------------------
label_col = "disease"  # ✅ update if your column is named differently
X = df.drop(columns=["id", "date", "image_id", label_col], errors="ignore")
y = df[label_col]

print(f"✅ Dataset loaded: {df.shape[0]} samples, {X.shape[1]} features")
print("Class distribution:\n", y.value_counts())

# --------------------------------------------------
# Train/test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=cfg["training"]["test_size"],
    random_state=cfg["training"]["random_state"],
    stratify=y
)

# --------------------------------------------------
# Apply SMOTE
# --------------------------------------------------
print("⚖️ Applying SMOTE to balance classes...")
smote = SMOTE(random_state=cfg["training"]["random_state"])
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("After SMOTE balancing:")
print(y_train_res.value_counts())

# --------------------------------------------------
# Model + GridSearch
# --------------------------------------------------
rf = RandomForestClassifier(random_state=cfg["training"]["random_state"], n_jobs=-1)

param_grid = {
    "n_estimators": cfg["grid_search"]["n_estimators"],
    "max_depth": cfg["grid_search"]["max_depth"],
    "min_samples_split": cfg["grid_search"]["min_samples_split"],
    "min_samples_leaf": cfg["grid_search"]["min_samples_leaf"],
}

print("⚙️ Running GridSearchCV...")
grid_search = GridSearchCV(
    rf,
    param_grid,
    cv=cfg["training"]["cv_folds"],
    scoring=cfg["grid_search"]["scoring"],
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_res, y_train_res)

# --------------------------------------------------
# Evaluation
# --------------------------------------------------
print("🏆 Best Params:", grid_search.best_params_)
print("🔢 Best Score:", grid_search.best_score_)

y_pred = grid_search.best_estimator_.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --------------------------------------------------
# Save model + features
# --------------------------------------------------
os.makedirs(os.path.dirname(model_out), exist_ok=True)
joblib.dump(grid_search.best_estimator_, model_out)

with open(features_out, "w") as f:
    json.dump(list(X.columns), f, indent=2)

print(f"💾 Model saved to {model_out}")
print(f"💾 Features saved to {features_out}")
