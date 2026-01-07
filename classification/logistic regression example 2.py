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
    f1_score
)

# Set random seed for reproducibility
np.random.seed(42)


# ==================================================
# 1) CREATE A BALANCED DATASET
# ==================================================

# ---- Fail students (low performance) ----
hours_fail = np.random.randint(0, 4, 50)
score_fail = np.random.randint(20, 50, 50)
att_fail = np.random.randint(40, 70, 50)

# ---- Pass students (high performance) ----
hours_pass = np.random.randint(6, 10, 50)
score_pass = np.random.randint(60, 95, 50)
att_pass = np.random.randint(75, 100, 50)

# Combine both classes
hours = np.concatenate([hours_fail, hours_pass])
scores = np.concatenate([score_fail, score_pass])
attendance = np.concatenate([att_fail, att_pass])

n = len(hours)


# ==================================================
# 1b) ADD NOISE TO LABELS (REALISTIC DATA)
# ==================================================

# Add Gaussian noise to simulate real-world uncertainty
extra_noise = np.random.normal(0, 10, size=n)

# Generate pass/fail labels using a weighted formula
pass_fail = (
    hours * 0.3 +
    scores * 0.5 +
    attendance * 0.2 +
    extra_noise > 50
).astype(int)

# Create DataFrame
df = pd.DataFrame({
    "Hours": hours,
    "Previous Score": scores,
    "Attendance": attendance,
    "Pass_Fail": pass_fail
})

# Display sample data and class balance
print(df.head(), "\n")
print("Class Counts:\n", df["Pass_Fail"].value_counts(), "\n")


# ==================================================
# 2) TRAIN-TEST SPLIT
# ==================================================

X = df[["Hours", "Previous Score", "Attendance"]]
y = df["Pass_Fail"]

# Stratify ensures class balance in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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

# Default predictions (threshold = 0.5)
y_pred = model.predict(X_test_scaled)


# ==================================================
# 5) PERFORMANCE WITH DEFAULT THRESHOLD (0.5)
# ==================================================

print("\n=== Default Threshold (0.5) ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ==================================================
# 6) THRESHOLD TUNING EXPERIMENT
# ==================================================

# Predicted probabilities for positive class
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# ---- Threshold = 0.7 ----
threshold = 0.7
y_pred_07 = (y_prob >= threshold).astype(int)

print("\n=== Threshold = 0.7 ===")
print("Accuracy:", accuracy_score(y_test, y_pred_07))
print("Precision:", precision_score(y_test, y_pred_07))
print("Recall:", recall_score(y_test, y_pred_07))
print("F1 Score:", f1_score(y_test, y_pred_07))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_07))


# ---- Threshold = 0.3 ----
threshold = 0.3
y_pred_03 = (y_prob >= threshold).astype(int)

print("\n=== Threshold = 0.3 ===")
print("Accuracy:", accuracy_score(y_test, y_pred_03))
print("Precision:", precision_score(y_test, y_pred_03))
print("Recall:", recall_score(y_test, y_pred_03))
print("F1 Score:", f1_score(y_test, y_pred_03))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_03))


# ---- Threshold = 0.5 (explicit) ----
threshold = 0.5
y_pred_05 = (y_prob >= threshold).astype(int)

print("\n=== Threshold = 0.5 ===")
print("Accuracy:", accuracy_score(y_test, y_pred_05))
print("Precision:", precision_score(y_test, y_pred_05))
print("Recall:", recall_score(y_test, y_pred_05))
print("F1 Score:", f1_score(y_test, y_pred_05))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_05))


# ---- Threshold = 0.9 ----
threshold = 0.9
y_pred_09 = (y_prob >= threshold).astype(int)

print("\n=== Threshold = 0.9 ===")
print("Accuracy:", accuracy_score(y_test, y_pred_09))
print("Precision:", precision_score(y_test, y_pred_09))
print("Recall:", recall_score(y_test, y_pred_09))
print("F1 Score:", f1_score(y_test, y_pred_09))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_09))
