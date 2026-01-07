import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Fix: Set seed properly
np.random.seed(42)

# Fix: Make y match the length of X
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
pass_fail = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])   # 10 values

df = pd.DataFrame({"Hours": hours, "Pass": pass_fail})
print(df)

# Fix: Use correct column name ('Hours', not 'Hours_Studied')
X = df[["Hours"]]
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_prob = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Precision, Recall, F1
print("Classification Report:\n", classification_report(y_test, y_pred))

# Change threshold to 0.7
y_pred_thresh = (y_prob >= 0.7).astype(int)

print("Predictions with threshold 0.7:", y_pred_thresh)
print("Confusion Matrix (0.7 threshold):\n", confusion_matrix(y_test, y_pred_thresh))
print("Classification Report (0.7 threshold):\n", classification_report(y_test, y_pred_thresh))
