import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Load data
df = pd.read_csv("data/crop_disease_training_data.csv")
X = df[['b5', 'b6', 'b7', 'b11', 'b12', 'ndvi']]
y = df['label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, "models/disease_risk_model.pkl")
print("✅ Model saved to models/disease_risk_model.pkl")
