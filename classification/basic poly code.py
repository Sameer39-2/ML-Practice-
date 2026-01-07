# Import evaluation metrics from scikit-learn
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    r2_score,
    precision_score,
    f1_score
)

# Actual labels (ground truth)
y_test = [1, 0, 1, 1, 0, 0]

# Predicted labels from the model
y_pred = [1, 0, 0, 1, 1, 0]

# -----------------------------
# Confusion Matrix Calculation
# -----------------------------

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Extract True Negative, False Positive, False Negative, True Positive
TN, FP, FN, TP = cm.ravel()

# Display confusion matrix and individual values
print("Confusion Matrix:\n", cm)
print(f"TN: {TN}, FP: {FP}, FN: {FN}, TP: {TP}\n")


# -----------------------------
# Model Evaluation Metrics
# -----------------------------

# Accuracy: Overall correctness of the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Precision: How many predicted positives are actually positive
print("Precision Score:", precision_score(y_test, y_pred))

# R2 Score: Measures how well predictions approximate actual values
# (Not commonly used for classification, included for demonstration)
print("R2 Score:", r2_score(y_test, y_pred))

# F1 Score: Harmonic mean of precision and recall
print("F1 Score:", f1_score(y_test, y_pred))
