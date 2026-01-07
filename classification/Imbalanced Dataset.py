# Import required libraries
import numpy as np
import pandas as pd

# Scikit-learn utilities
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# -----------------------------
# Step 1: Create Imbalanced Dataset
# -----------------------------

# Number of failed and passed students
n_fail = 90
n_pass = 10

# Generate random study hours and attendance values
hours = np.random.randint(1, 10, n_fail + n_pass)
attendance = np.random.randint(60, 100, n_fail + n_pass)

# Create imbalanced target labels
# 0 = Fail, 1 = Pass
labels = np.array([0] * n_fail + [1] * n_pass)

# Create DataFrame
df = pd.DataFrame({
    "Hours_Studied": hours,
    "Attendance": attendance,
    "Pass": labels
})


# -----------------------------
# Step 2: Feature Selection
# -----------------------------

# Independent variables
X = df[["Hours_Studied", "Attendance"]]

# Target variable
y = df["Pass"]


# -----------------------------
# Step 3: Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# -----------------------------
# Step 4: Train Logistic Regression Model
# -----------------------------

model = LogisticRegression()
model.fit(X_train, y_train)


# -----------------------------
# Step 5: Make Predictions
# -----------------------------

y_pred = model.predict(X_test)


# -----------------------------
# Step 6: Model Evaluation
# -----------------------------

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision Score:", precision_score(y_test, y_pred, zero_division=0))
print("Recall Score:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
