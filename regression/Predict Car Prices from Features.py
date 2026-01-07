# ==================================================
# Linear Regression Example: Car Price Prediction
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

n = 300  # Number of cars

# Features
engine_size = np.random.randint(1500, 2000, n)         # in cc
mileage = np.random.uniform(10, 25, n)                # km/l
age = np.random.randint(1, 15, n)                     # years
seat = np.random.choice([4, 5, 7], n)                 # number of seats
fuel_type = np.random.choice([0, 1], n)               # 0 = Petrol, 1 = Diesel

# Add some random noise to make data realistic
noise = np.random.normal(0, 60000, n)

# Target variable: Price
price = (
    engine_size * 300 + 
    mileage * 12000 - 
    age * 20000 + 
    seat * 15000 + 
    fuel_type * 40000 + 
    noise
)

# Create DataFrame
df = pd.DataFrame({
    "Engine Size": engine_size,
    "Mile Age": mileage,
    "Age": age,
    "Seat": seat,
    "Fuel Type": fuel_type,
    "Price": price
})

# Preview data
print(df.head())


# ==================================================
# 2) PREPARE FEATURES AND TARGET
# ==================================================

X = df[["Engine Size", "Mile Age", "Age", "Seat", "Fuel Type"]]
y = df["Price"]


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
