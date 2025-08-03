# irrigation_model.py
import pandas as pd, numpy as np, os, joblib, json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("data/irrigation_training.csv")

df = pd.get_dummies(df, drop_first=True)

X = df.drop("optimal_water_liters", axis=1)
y = df["optimal_water_liters"]

with open("models/irrigation_model_features.json", "w") as f:
    json.dump(list(X.columns), f)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/irrigation_optimizer.pkl")

print("=== Irrigation Model ===")
print("MSE:", mean_squared_error(y_test, model.predict(X_test)))
print("R2 Score:", r2_score(y_test, model.predict(X_test)))
