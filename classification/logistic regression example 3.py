# Import required libraries
import numpy as np
import pandas as pd

# Scikit-learn utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    f1_score
)

# Set random seed for reproducibility
np.random.seed(42)


# ==================================================
# 1) CREATE SYNTHETIC DATA
# ==================================================

# Number of students
n = 100

# Generate random features
hours = np.random.randint(1, 10, n)          # Study hours
attendance = np.random.randint(60, 100, n)   # Attendance percentage
pass_score = np.random.randint(40, 100, n)   # Internal score

# Binary target variable
# Pass = 1 if score > 60, else 0
pass_fail = (pass_score > 60).astype(int)

# Create DataFrame
df = pd.DataFrame({
    "Hours_Studied": hours,
    "Attendance": attendance,
    "Pass": pass_fail
})

# Feature matrix and target vector
X = df[["Hours_Studied", "Attendance"]]
y = df["Pass"]


# ==================================================
# 2) TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
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

# Predictions using default threshold (0.5)
y_pred = model.predict(X_test_scaled)

# Predicted probabilities for the positive class
y_prob = model.predict_proba(X_test_scaled)[:, -1]

# Display sample data
print(df.head())


# ==================================================
# 5) EVALUATION WITH DEFAULT THRESHOLD (0.5)
# ==================================================

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

acc = accuracy_score(y_test, y_pred)
print("Accuracy Score:", acc)

re = recall_score(y_test, y_pred)
print("Recall Score:", re)

prec = precision_score(y_test, y_pred)
print("Precision Score:", prec)

f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)


# ==================================================
# 6) THRESHOLD = 0.7
# ==================================================

threshold = 0.7
y_pred_thre = (y_prob >= threshold).astype(int)

print("\n=== Threshold = 0.7 ===")
print("Predictions:", y_pred_thre)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_thre))
print("Classification Report:\n",
      classification_report(y_test, y_pred_thre))


# ==================================================
# 7) THRESHOLD = 0.3
# ==================================================

threshold = 0.3
y_pred_03 = (y_prob >= threshold).astype(int)

print("\n=== Threshold = 0.3 ===")
print("Accuracy:", accuracy_score(y_test, y_pred_03))
print("Precision:", precision_score(y_test, y_pred_03))
print("Recall:", recall_score(y_test, y_pred_03))
print("F1 Score:", f1_score(y_test, y_pred_03))
print("Confusion Matrix:\n",
      confusion_matrix(y_test, y_pred_03))


# ==================================================
# 8) THRESHOLD = 0.5 (EXPLICIT)
# ==================================================

threshold = 0.5
y_pred_05 = (y_prob >= threshold).astype(int)

print("\n=== Threshold = 0.5 ===")
print("Accuracy:", accuracy_score(y_test, y_pred_05))
print("Precision:", precision_score(y_test, y_pred_05))
print("Recall:", recall_score(y_test, y_pred_05))
print("F1 Score:", f1_score(y_test, y_pred_05))
print("Confusion Matrix:\n",
      confusion_matrix(y_test, y_pred_05))
