import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

# Create output directory
os.makedirs("outputs/evaluation", exist_ok=True)

# ----------------------------
# 1. Disease Detection Model Evaluation
# ----------------------------
print("===== Disease Model Evaluation =====")

df_disease = pd.read_csv("data/disease_training.csv")

# Select features used during training
disease_features = ['b5', 'b6', 'b7', 'b11', 'b12', 'ndvi']
X_disease = df_disease[disease_features]
y_disease = df_disease['label']

# Load model
model_disease = joblib.load("models/disease_risk_model.pkl")

# Train/test split
_, X_test_disease, _, y_test_disease = train_test_split(
    X_disease, y_disease, test_size=0.2, random_state=42
)

# Predict and probabilities
y_pred_disease = model_disease.predict(X_test_disease)
y_prob_disease = model_disease.predict_proba(X_test_disease)[:, 1]

# Metrics
print(classification_report(y_test_disease, y_pred_disease))
print("Confusion Matrix:")
cm_disease = confusion_matrix(y_test_disease, y_pred_disease)
print(cm_disease)
print(f"ROC AUC: {roc_auc_score(y_test_disease, y_prob_disease):.3f}")

# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm_disease, annot=True, fmt='d', cmap='Blues')
plt.title("Disease Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/evaluation/disease_confusion_matrix.png")
plt.close()

# ----------------------------
# 2. Pest Detection Model Evaluation
# ----------------------------
print("\n===== Pest Model Evaluation =====")

df_pest = pd.read_csv("data/pest_training.csv")

# One-hot encode 'crop_type' with drop_first to match training
df_pest = pd.get_dummies(df_pest, columns=["crop_type"], drop_first=True)

# Load model and features used during training
model_pest = joblib.load("models/pest_risk_model.pkl")
with open("models/pest_model_features.json") as f:
    pest_features = json.load(f)

# Filter columns to match training features exactly
X_pest = df_pest[pest_features]
y_pest = df_pest["label"]

# Predict and evaluate
y_pred_pest = model_pest.predict(X_pest)
y_prob_pest = model_pest.predict_proba(X_pest)[:, 1]

print(classification_report(y_pest, y_pred_pest))
print("Confusion Matrix:")
cm_pest = confusion_matrix(y_pest, y_pred_pest)
print(cm_pest)
print(f"ROC AUC: {roc_auc_score(y_pest, y_prob_pest):.3f}")

plt.figure(figsize=(5, 4))
sns.heatmap(cm_pest, annot=True, fmt='d', cmap='Oranges')
plt.title("Pest Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/evaluation/pest_confusion_matrix.png")
plt.close()

# ----------------------------
# 3. Crop vs Weed Classification Model Evaluation
# ----------------------------
print("\n===== Crop/Weed Classification Model Evaluation =====")

df_crop = pd.read_csv("data/crop_weed_training.csv")

# Load features used during training
with open("models/crop_classifier_features.json") as f:
    crop_features = json.load(f)
X_crop = df_crop[crop_features]
y_crop = df_crop['label']

# Train/test split
_, X_test_crop, _, y_test_crop = train_test_split(
    X_crop, y_crop, test_size=0.2, random_state=42
)

# Load crop/weed model
model_crop = joblib.load("models/crop_classifier_model.pkl")

# Predict
y_pred_crop = model_crop.predict(X_test_crop)

print(classification_report(y_test_crop, y_pred_crop))
print("Confusion Matrix:")
cm_crop = confusion_matrix(y_test_crop, y_pred_crop)
print(cm_crop)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_crop, annot=True, fmt='d', cmap='Greens')
plt.title("Crop vs Weed Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/evaluation/crop_weed_confusion_matrix.png")
plt.close()

print("\n✅ Evaluation reports generated. Confusion matrices saved to outputs/evaluation/")
