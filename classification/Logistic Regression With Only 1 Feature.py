# ==================================================
# Logistic Regression Example with Synthetic Data
# ==================================================

# Import required libraries
import numpy as np
import pandas as pd

# Scikit-learn utilities
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

# Set random seed for reproducibility
np.random.seed(42)


# ==================================================
# 1) CREATE SYNTHETIC DATA
# ==================================================

n = 50  # number of students

# Generate features
hours = np.random.randint(1, 10, n)   # Hours studied
marks = np.random.randint(40, 100, n)  # Exam marks

# Create DataFrame
df = pd.DataFrame({
    "Hours": hours,
    "Marks": marks
})

# Target: Pass if marks >= 50, else Fail
df["Pass"] = (df["Marks"] >= 50).astype(int)

# Display first few rows
print(df.head())


# ==================================================
# 2) PREPARE FEATURES AND TARGET
# ==================================================

X = df[["Hours", "Marks"]]  # Features
y = df["Pass"]               # Target


# ==================================================
# 3) TRAIN-TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# ==================================================
# 4) FEATURE SCALING
# ==================================================

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==================================================
# 5) TRAIN LOGISTIC REGRESSION MODEL
# ==================================================

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled)


# ==================================================
# 6) EVALUATE MODEL
# ==================================================

print("\n=== Model Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision Score:", precision_score(y_test, y_pred, average='binary'))
print("Recall Score:", recall_score(y_test, y_pred, average='binary'))
print("F1 Score:", f1_score(y_test, y_pred, average='binary'))
