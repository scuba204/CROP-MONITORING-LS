import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# 1. Generate synthetic data
np.random.seed(42)
n_samples = 500

# Simulate features from Sentinel-2 bands and NDVI
features = pd.DataFrame({
    'b5': np.random.uniform(100, 3000, n_samples),
    'b6': np.random.uniform(100, 3000, n_samples),
    'b7': np.random.uniform(100, 3000, n_samples),
    'b11': np.random.uniform(100, 3000, n_samples),
    'b12': np.random.uniform(100, 3000, n_samples),
    'ndvi': np.random.uniform(0, 1, n_samples)
})

# Simulated pest presence labels (1 = high risk, 0 = low risk)
# This could be based on some rule for realism (e.g., low NDVI + high SWIR = high risk)
features['pest_risk'] = ((features['ndvi'] < 0.4) & (features['b11'] > 1500)).astype(int)

# 2. Prepare training data
X = features[['b5', 'b6', 'b7', 'b11', 'b12', 'ndvi']]
y = features['pest_risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 3. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 5. Save the model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/pest_risk_model.pkl")
print("✅ Model saved to models/pest_risk_model.pkl")
