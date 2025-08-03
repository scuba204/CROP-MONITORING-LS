# nutrient_model.py
import pandas as pd, numpy as np, os, joblib, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df = pd.read_csv("data/nutrient_training.csv")

df = pd.get_dummies(df, drop_first=True)

X = df.drop("label", axis=1)
y = df["label"]

with open("models/nutrient_model_features.json", "w") as f:
    json.dump(list(X.columns), f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/nutrient_deficiency_model.pkl")

print("=== Nutrient Deficiency Model ===")
print(classification_report(y_test, model.predict(X_test)))
print("Confusion Matrix:\n", confusion_matrix(y_test, model.predict(X_test)))
print("ROC AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
