# Import required libraries
import numpy as np
import pandas as pd

# Scikit-learn utilities
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# Step 1: Data Generation
# -----------------------------

# Set random seed for reproducibility
np.random.seed(42)

# Number of students
n = 100

# Generate random study hours (1–9 hours)
hours = np.random.randint(1, 10, n)

# Generate random attendance percentage (60%–100%)
attendance = np.random.randint(60, 100, n)

# Calculate marks using a weighted formula with random noise
marks = (attendance * 0.8) + (hours * 2) + np.random.randint(-20, 20, n)

# Create DataFrame
df = pd.DataFrame({
    "Hours_Studied": hours,
    "Attendance": attendance,
    "Marks": marks
})

# Create target variable:
# Pass = 1 if marks > 50, else 0
df["Pass"] = (df["Marks"] > 50).astype(int)

# Display first few rows
print(df.head())


# ==================================================
# Model A: Logistic Regression using ONLY Hours Studied
# ==================================================

# Feature selection
X_A = df[["Hours_Studied"]]
y = df["Pass"]

# Split dataset into training and testing sets (80% / 20%)
X_train_A, X_test_A, y_train_A, y_test_A = train_test_split(
    X_A, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler_A = StandardScaler()
X_train_A_scaled = scaler_A.fit_transform(X_train_A)
X_test_A_scaled = scaler_A.transform(X_test_A)

# Initialize Logistic Regression model
# class_weight='balanced' handles class imbalance
model_A = LogisticRegression(class_weight='balanced')

# Train the model
model_A.fit(X_train_A_scaled, y_train_A)

# Make predictions
y_pred_A = model_A.predict(X_test_A_scaled)

# Model evaluation
print("=" * 10, "Model A (Only Hours)", "=" * 10)
print("Accuracy Score:", accuracy_score(y_test_A, y_pred_A))
print("Classification Report:\n",
      classification_report(y_test_A, y_pred_A, zero_division=0))


# ==================================================
# Model B: Logistic Regression using Hours + Attendance
# ==================================================

# Feature selection
X_B = df[["Hours_Studied", "Attendance"]]

# Split dataset
X_train_B, X_test_B, y_train_B, y_test_B = train_test_split(
    X_B, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler_B = StandardScaler()
X_train_B_scaled = scaler_B.fit_transform(X_train_B)
X_test_B_scaled = scaler_B.transform(X_test_B)

# Initialize and train model
model_B = LogisticRegression(class_weight='balanced')
model_B.fit(X_train_B_scaled, y_train_B)

# Make predictions
y_pred_B = model_B.predict(X_test_B_scaled)

# Model evaluation
print("=" * 10, "Model B (Hours + Attendance)", "=" * 10)
print("Accuracy Score:", accuracy_score(y_test_B, y_pred_B))
print("Classification Report:\n",
      classification_report(y_test_B, y_pred_B, zero_division=0))
