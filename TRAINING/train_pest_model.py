import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import json

# Load the dataset
df = pd.read_csv("data/pest_training.csv")

# One-hot encode crop_type if needed
df = pd.get_dummies(df, columns=["crop_type"], drop_first=True)

# Define feature columns (exclude label)
features = [col for col in df.columns if col != 'label']

X = df[features]
y = df['label']

# Save the features used for training
with open("models/pest_model_features.json", "w") as f:
    json.dump(list(X.columns), f)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/pest_risk_model.pkl")

# Evaluate model
y_pred = model.predict(X_test)
print("===== Pest Model Report =====")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
