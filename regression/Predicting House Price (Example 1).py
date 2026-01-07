import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1) Data (features + target)
# House sizes (sq ft)
x = np.array([[500], [1000], [1500], [2500], [3000], [3500], [4000]], dtype=float)

# Corresponding house prices (in thousands $)
y = np.array([30, 45, 60, 80, 95, 110, 130], dtype=float)   # FIXED: now 7 values

# 2) Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.25, random_state=1
)

# 3) Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4) Model training
model = LinearRegression()
model.fit(X_train_scaled, Y_train)

# 5) Prediction
y_pred = model.predict(X_test_scaled)

# 6) Evaluation metrics
mse = mean_squared_error(Y_test, y_pred)
mae = mean_absolute_error(Y_test, y_pred)
r2 = r2_score(Y_test, y_pred)

print("Test set actual prices: ", Y_test)
print("Test set predicted prices: ", y_pred)
print("MSE: ", mse)
print("MAE: ", mae)
print("R2 Score: ", r2)
