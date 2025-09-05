# train_disease_model.py
import os
import json
import yaml
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score

# -----------------------------
# Load configuration
# -----------------------------
with open("configs/config_disease.yaml", "r") as f:
    config = yaml.safe_load(f)

csv_path = config["data"]["csv_path"]
model_name = config["model"]["name"]
features_file = config["model"]["features_file"]
test_size = config["training"]["test_size"]
random_state = config["training"]["random_state"]
cv_folds = config["training"]["cv_folds"]
param_grid = config["grid_search"]

# -----------------------------
# Load dataset
# -----------------------------
print(f"📂 Loading dataset: {csv_path}")
df = pd.read_csv(csv_path)

# Expect last column = label
X = df.drop(columns=["disease"])
y = df["disease"]

# Save feature names
feature_names = X.columns.tolist()
with open(features_file, "w") as f:
    json.dump(feature_names, f, indent=2)

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
)

# -----------------------------
# Model & GridSearchCV
# -----------------------------
print("⚙️ Running GridSearchCV...")
rf = RandomForestClassifier(random_state=random_state, class_weight="balanced")

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring=config["grid_search"]["scoring"],
    cv=cv_folds,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

# -----------------------------
# Evaluate best model
# -----------------------------
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n✅ Best Params:", grid_search.best_params_)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))

# -----------------------------
# Save model
# -----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, os.path.join("models", model_name))
print(f"💾 Saved model to models/{model_name}")
