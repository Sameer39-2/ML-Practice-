# Import required libraries
import numpy as np
import pandas as pd

# Scikit-learn utilities
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score


# -----------------------------
# Step 1: Define Dataset
# -----------------------------

# Feature matrix (single feature)
X = np.array([[1], [2], [3], [4], [5]])

# Target labels
y = np.array([0, 0, 0, 1, 1])


# -----------------------------
# Step 2: Train-Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# -----------------------------
# Step 3: Train Logistic Regression Model
# -----------------------------

model = LogisticRegression()
model.fit(X_train, y_train)


# -----------------------------
# Step 4: Probability Prediction
# -----------------------------

# Predict probability of the positive class (class = 1)
y_prob = model.predict_proba(X_test)[:, 1]


# -----------------------------
# Step 5: Threshold Tuning
# -----------------------------

# Different probability thresholds to evaluate
threshold = np.array([0.2, 0.4, 0.5, 0.7, 0.9])

# Store evaluation results
result = []

for t in threshold:
    # Convert probabilities into class labels using threshold
    y_pred_thresh = (y_prob >= t).astype(int)

    # Calculate evaluation metrics
    precision = precision_score(y_test, y_pred_thresh, zero_division=0)
    recall = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)

    # Save results
    result.append([t, precision, recall, f1])


# -----------------------------
# Step 6: Display Results
# -----------------------------

# Create a DataFrame to display metrics for each threshold
df_metrics = pd.DataFrame(
    result,
    columns=["Threshold", "Precision", "Recall", "F1 Score"]
)

print(df_metrics)
