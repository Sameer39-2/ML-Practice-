# Import required libraries
import numpy as np
import pandas as pd

# Scikit-learn utilities
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------
# Step 1: Generate Sample Data
# -----------------------------

# Number of students
n = 100

# Random study hours (1–9 hours)
hours = np.random.randint(1, 10, n)

# Random attendance percentage (60%–100%)
attendance = np.random.randint(60, 100, n)

# Calculate marks using a weighted formula with noise
marks = (hours * 2) + (attendance * 0.5) + np.random.randint(-20, 20, n)

# Create DataFrame
df = pd.DataFrame({
    "Hours_Studied": hours,
    "Attendance": attendance,
    "Marks": marks
})

# Create binary target variable:
# Pass = 1 if marks > 50, else 0
df["Pass"] = (df["Marks"] > 50).astype(int)

# Feature matrix and target vector
X = df[["Hours_Studied", "Attendance"]]
y = df["Pass"]

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==================================
# Model A: Logistic Regression WITH Scaling
# ==================================

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train model
model_A = LogisticRegression()
model_A.fit(X_train_scaled, y_train)

# Make predictions
y_pred_A = model_A.predict(X_test_scaled)

# Display results
print(df.head())
print("=" * 10, "With Scaling", "=" * 10)
print("Accuracy:", accuracy_score(y_test, y_pred_A))
print("Classification Report:\n",
      classification_report(y_test, y_pred_A))


# ==================================
# Model B: Logistic Regression WITHOUT Scaling
# ==================================

# Train model on unscaled data
model_B = LogisticRegression()
model_B.fit(X_train, y_train)

# Make predictions
y_pred_B = model_B.predict(X_test)

# Display results
print("=" * 10, "Without Scaling", "=" * 10)
print("Accuracy:", accuracy_score(y_test, y_pred_B))
print("Classification Report:\n",
      classification_report(y_test, y_pred_B))
