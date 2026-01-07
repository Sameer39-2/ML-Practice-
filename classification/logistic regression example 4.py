# ==================================================
# Logistic Regression Example with Threshold Tuning
# ==================================================

# Import required libraries
import numpy as np
import pandas as pd

# Scikit-learn utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Set seed for reproducibility
np.random.seed(42)


# ==================================================
# 1) CREATE SYNTHETIC DATA
# ==================================================

n = 100  # number of students

# Features
hours = np.random.randint(1, 5, n)            # Hours studied
attendance = np.random.randint(60, 100, n)    # Attendance percentage

# Marks formula with some noise
marks = (attendance * 0.8) + (hours * 2) + np.random.randint(-20, 20, n)

# Create DataFrame
df = pd.DataFrame({
    "Hours": hours,
    "Attendance": attendance,
    "Marks": marks
})

# Binary target: Pass if Marks >= 50
df["Pass"] = (df["Marks"] >= 50).astype(int)

# Features and target
X = df[["Hours", "Attendance"]]
y = df["Pass"]

# Display sample data
print(df.head())


# ==================================================
# 2) TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==================================================
# 3) FEATURE SCALING
# ==================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==================================================
# 4) TRAIN LOGISTIC REGRESSION MODEL
# ==================================================

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions using default threshold of 0.5
y_pred = model.predict(X_test_scaled)

# Predicted probabilities for the positive class
y_prob = model.predict_proba(X_test_scaled)[:, -1]


# ==================================================
# 5) EVALUATE MODEL WITH DEFAULT THRESHOLD (0.5)
# ==================================================

print("\n=== Default Threshold (0.5) ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))


# ==================================================
# 6) THRESHOLD TUNING
# ==================================================

# ---- Threshold = 0.3 ----
threshold = 0.3
y_pred_03 = (y_prob >= threshold).astype(int)

print("\n" + "*"*10 + " Threshold = 0.3 " + "*"*10)
print("\nClassification Report:\n", classification_report(y_test, y_pred_03, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_03))


# ---- Threshold = 0.5 ----
threshold = 0.5
y_pred_05 = (y_prob >= threshold).astype(int)

print("\n" + "*"*10 + " Threshold = 0.5 " + "*"*10)
print("Predictions:", y_pred_05)
print("\nClassification Report:\n", classification_report(y_test, y_pred_05, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_05))
