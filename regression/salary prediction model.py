# ==================================================
# Linear Regression Example: Salary Prediction
# ==================================================

# Import required libraries
import numpy as np
import pandas as pd

# Scikit-learn utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set random seed for reproducibility
np.random.seed(42)


# ==================================================
# 1) CREATE SYNTHETIC DATA
# ==================================================

n = 200  # Number of employees

# Features
exp = np.random.randint(1, 10, n)                # Years of experience
age = np.random.randint(20, 40, n)               # Age of employee
education_level = np.random.randint(1, 5, n)     # 1: High school ... 4: PhD

# Random noise for realistic variation
noise = np.random.randint(0, 5000, n)

# Target: Salary (in USD)
salary = (
    20000 +                  # Base salary
    (exp * 1500) +           # Experience contribution
    (age * 50) +             # Age contribution
    (education_level * 8000) + # Education contribution
    noise                     # Random noise
)

# Create DataFrame
df = pd.DataFrame({
    "Experience": exp,
    "Age": age,
    "Educational Level": education_level,
    "Salary": salary
})

# Preview the dataset
print(df.head())


# ==================================================
# 2) PREPARE FEATURES AND TARGET
# ==================================================

X = df[["Experience", "Age", "Educational Level"]]
y = df["Salary"]


# ==================================================
# 3) TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==================================================
# 4) FEATURE SCALING
# ==================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==================================================
# 5) TRAIN LINEAR REGRESSION MODEL
# ==================================================

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)


# ==================================================
# 6) EVALUATE MODEL
# ==================================================

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MSE:", mse)
print("MAE:", mae)
print("R2 Score:", r2)


# ==================================================
# 7) FEATURE IMPORTANCE / COEFFICIENTS
# ==================================================

print("\nModel Coefficients (Feature Importance):")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")
