import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

# Contoh data dummy WHO-style
data = {
    'age': [30, 45, 50, 60, 25, 35, 70, 40, 28, 52],
    'gender': [1, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    'bmi': [22.0, 28.5, 25.0, 30.2, 18.5, 23.3, 32.0, 27.0, 20.0, 29.1],
    'bp': [120, 140, 135, 150, 110, 125, 160, 138, 115, 142],
    'cholesterol': [180, 220, 210, 240, 170, 190, 260, 215, 175, 230],
    'diabetes': [0, 1, 0, 1, 0, 0, 1, 0, 0, 1],
    'hypertension': [0, 1, 1, 1, 0, 0, 1, 1, 0, 1],
    'smoker': [1, 1, 0, 0, 0, 1, 0, 0, 1, 1],
    'passive_smoker': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1],
    'alcohol': [1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
    'exercise': [2, 0, 3, 1, 4, 3, 0, 1, 5, 2],
    'family_history': [1, 1, 0, 1, 0, 0, 1, 1, 0, 1],
    'income': [15000, 20000, 25000, 18000, 30000, 35000, 12000, 17000, 40000, 16000],
    'education': [12, 10, 15, 8, 16, 14, 7, 9, 18, 11],
    'life_expectancy': [75, 65, 78, 60, 85, 80, 58, 68, 88, 64]
}

df = pd.DataFrame(data)

X = df.drop(columns="life_expectancy")
y = df["life_expectancy"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/life_model.pkl")
print("✅ Model berhasil disimpan di model/life_model.pkl")