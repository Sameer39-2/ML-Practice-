# Import required libraries
import numpy as np
import pandas as pd

# Scikit-learn utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -----------------------------
# Step 1: Generate Synthetic Data
# -----------------------------

# Set random seed for reproducibility
np.random.seed(42)

# Generate house features
size = np.random.randint(500, 3000, 150)     # House size in square feet
bedroom = np.random.randint(1, 5, 150)       # Number of bedrooms
age = np.random.randint(1, 30, 150)           # Age of the house (years)

# Random noise to simulate real-world variation
noise = np.random.randint(0, 25, 150)

# Price calculation using a linear formula
price = (size * 0.5) + (bedroom * 10) - (age * 1.2) + noise

# Create DataFrame
df = pd.DataFrame({
    "Size": size,
    "Bedroom": bedroom,
    "Age": age,
    "Price": price
})

# Display first few rows
print(df.head())


# -----------------------------
# Step 2: Feature Selection
# -----------------------------

# Independent variables (features)
x = df[["Size", "Bedroom", "Age"]]

# Dependent variable (target)
y = df["Price"]


# -----------------------------
# Step 3: Train-Test Split
# -----------------------------

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.25, random_state=42
)


# -----------------------------
# Step 4: Feature Scaling
# -----------------------------

# Standardize features to improve model performance
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# -----------------------------
# Step 5: Train Linear Regression Model
# -----------------------------

model = LinearRegression()
model.fit(x_train_scaled, y_train)


# -----------------------------
# Step 6: Make Predictions
# -----------------------------

y_pred = model.predict(x_test_scaled)


# -----------------------------
# Step 7: Model Evaluation
# -----------------------------

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# R-squared score
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("MSE:", mse)
print("R2 Score:", r2)


# ------------------------------------------
# Step 8: Model Coefficients (Feature Importance)
# ------------------------------------------

print("\nModel Coefficients:")
for feature, coef in zip(x.columns, model.coef_):
    print(f"{feature}: {coef}")
