# ============================
# 1. Disease Detection Model Evaluation
# ============================
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load data and model
df_disease = pd.read_csv("data/disease_training.csv")
X_disease = df_disease[['b5', 'b6', 'b7', 'b11', 'b12', 'ndvi']]
y_disease = df_disease['label']
model_disease = joblib.load("models/disease_risk_model.pkl")

_, X_test_disease, _, y_test_disease = train_test_split(X_disease, y_disease, test_size=0.2, random_state=42)
y_pred_disease = model_disease.predict(X_test_disease)
y_prob_disease = model_disease.predict_proba(X_test_disease)[:, 1]

print("===== Disease Model Report =====")
print(classification_report(y_test_disease, y_pred_disease))
print("Confusion Matrix:")
print(confusion_matrix(y_test_disease, y_pred_disease))
print(f"ROC AUC: {roc_auc_score(y_test_disease, y_prob_disease):.3f}")

# Plot confusion matrix
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test_disease, y_pred_disease), annot=True, fmt='d', cmap='Blues')
plt.title("Disease Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/evaluation/disease_confusion_matrix.png")
plt.close()

# ============================
# 2. Pest Detection Model Evaluation
# ============================
df_pest = pd.read_csv("data/pest_training.csv")
X_pest = df_pest[['b5', 'b6', 'b7', 'b11', 'b12', 'ndvi', 'ndwi', 'lst', 'humidity', 'irradiance']]
y_pest = df_pest['label']
model_pest = joblib.load("models/pest_risk_model.pkl")

_, X_test_pest, _, y_test_pest = train_test_split(X_pest, y_pest, test_size=0.2, random_state=42)
y_pred_pest = model_pest.predict(X_test_pest)
y_prob_pest = model_pest.predict_proba(X_test_pest)[:, 1]

print("\n===== Pest Model Report =====")
print(classification_report(y_test_pest, y_pred_pest))
print("Confusion Matrix:")
print(confusion_matrix(y_test_pest, y_pred_pest))
print(f"ROC AUC: {roc_auc_score(y_test_pest, y_prob_pest):.3f}")

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test_pest, y_pred_pest), annot=True, fmt='d', cmap='Oranges')
plt.title("Pest Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/evaluation/pest_confusion_matrix.png")
plt.close()

# ============================
# 3. Crop/Weed Classification Model Evaluation
# ============================
df_crop = pd.read_csv("data/crop_weed_training.csv")
X_crop = df_crop.drop(columns=['label'])
y_crop = df_crop['label']
model_crop = joblib.load("models/crop_weed_model.pkl")

_, X_test_crop, _, y_test_crop = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)
y_pred_crop = model_crop.predict(X_test_crop)

print("\n===== Crop/Weed Classification Report =====")
print(classification_report(y_test_crop, y_pred_crop))
print("Confusion Matrix:")
print(confusion_matrix(y_test_crop, y_pred_crop))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test_crop, y_pred_crop), annot=True, fmt='d', cmap='Greens')
plt.title("Crop vs Weed Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("outputs/evaluation/crop_weed_confusion_matrix.png")
plt.close()

print("\n✅ Evaluation reports generated. Confusion matrices saved to outputs/evaluation/")
