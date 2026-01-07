# Import required libraries
import numpy as np
import pandas as pd

# Scikit-learn utilities for model training and evaluation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


# -----------------------------
# Step 1: Generate Sample Data
# -----------------------------

# Number of students
n = 100

# Randomly generate study hours (1–9 hours)
hours = np.random.randint(1, 10, n)

# Randomly generate attendance percentage (60%–100%)
attendance = np.random.randint(60, 100, n)

# Randomly generate assignment completion (0 = No, 1 = Yes)
assignment = np.random.randint(0, 2, n)

# Calculate marks using a weighted formula + random noise
marks = (
    (hours * 2) +
    (attendance * 0.7) +
    (assignment * 5) +
    np.random.randint(-20, 20, n)
)


# --------------------------------
# Step 2: Create a Pandas DataFrame
# --------------------------------

df = pd.DataFrame({
    "Hours_Studied": hours,
    "Attendance": attendance,
    "Marks": marks,
    "Assignment": assignment
})

# Create target variable:
# Pass = 1 if Marks >= 70, else 0
df["Pass"] = (df["Marks"] >= 70).astype(int)

# Display the dataset
print(df)


# -----------------------------------
# Step 3: Feature Selection & Splitting
# -----------------------------------

# Features (independent variables)
X = df[["Hours_Studied", "Attendance", "Assignment"]]

# Target variable (dependent variable)
y = df["Pass"]

# Split data into training (70%) and testing (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# -----------------------------
# Step 4: Feature Scaling
# -----------------------------

# Standardize features to improve model performance
scaler = StandardScaler()

# Fit scaler on training data and transform
X_train_scaled = scaler.fit_transform(X_train)

# Transform test data using the same scaler
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# Step 5: Train Logistic Regression Model
# -----------------------------

# Initialize Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train_scaled, y_train)


# -----------------------------
# Step 6: Make Predictions
# -----------------------------

# Predict pass/fail on test data
y_pred = model.predict(X_test_scaled)

print("Prediction:", y_pred)


# -----------------------------
# Step 7: Model Evaluation
# -----------------------------

# Evaluate model performance using standard metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision Score:", precision_score(y_test, y_pred))
print("Recall Score:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
