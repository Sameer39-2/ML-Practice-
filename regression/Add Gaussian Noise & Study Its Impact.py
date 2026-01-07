import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# ======================
# Function: Train + Evaluate Model
# ======================
def train_and_evaluate(X, y, title="Model Results"):
    """Train logistic regression and print evaluation metrics."""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Results
    print("\n", "="*10, title, "="*10)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))


# ======================
# 1. Create Clean Dataset
# ======================
n = 100
hours = np.random.randint(1, 10, n)
attendance = np.random.randint(60, 100, n)

# Marks formula (clean)
marks = (hours * 2) + (attendance * 0.8)

df_clean = pd.DataFrame({
    "Hours_Studied": hours,
    "Attendance": attendance,
    "Marks": marks
})

# Create label (Pass/Fail)
df_clean["Pass"] = (df_clean["Marks"] >= 70).astype(int)

# Inputs: Do NOT include "Marks" because it's the target
X_clean = df_clean[["Hours_Studied", "Attendance"]]
y_clean = df_clean["Pass"]

train_and_evaluate(X_clean, y_clean, "Without Noise (Clean Data)")


# ======================
# 2. Create Noisy Dataset
# ======================

# Add Gaussian noise BEFORE computing the label
marks_noisy = marks + np.random.normal(0, 10, size=n)

df_noisy = pd.DataFrame({
    "Hours_Studied": hours,
    "Attendance": attendance,
    "Marks": marks_noisy
})

df_noisy["Pass"] = (df_noisy["Marks"] >= 70).astype(int)

# Features remain the same — DO NOT include Marks
X_noisy = df_noisy[["Hours_Studied", "Attendance"]]
y_noisy = df_noisy["Pass"]

train_and_evaluate(X_noisy, y_noisy, "With Noise")
