# Credit Card Fraud Detection
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Load dataset
df = pd.read_csv("creditcard.csv")

# Display basic information
print("Dataset Shape:", df.shape)
print("\nClass Distribution:")
print(df["Class"].value_counts())

print("\nMissing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Rows:", df.duplicated().sum())
# Class Distribution Visualization

plt.figure(figsize=(6, 4))

sns.countplot(x="Class", data=df)

plt.title("Normal vs Fraudulent Transactions")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")
plt.xticks([0, 1], ["Normal", "Fraud"])

plt.tight_layout()
plt.savefig("class_distribution.png")
plt.close()
# -------------------------------
# Data Preprocessing
# -------------------------------

# Separate features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Scale Time and Amount
scaler = StandardScaler()

X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Set:", X_train.shape)
print("Testing Set:", X_test.shape)

print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())
# -------------------------------
# Logistic Regression Model
# -------------------------------

model = LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed!")
# -------------------------------
# Model Prediction
# -------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nModel Evaluation")

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# -------------------------------
# Confusion Matrix Visualization
# -------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Fraud"],
    yticklabels=["Normal", "Fraud"]
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

print("\nConfusion matrix saved as confusion_matrix.png")
# -------------------------------
# Random Forest Model
# -------------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Evaluation")

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1 Score:", f1_score(y_test, rf_pred))
print("ROC-AUC:", roc_auc_score(y_test, rf_prob))

print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))
# -------------------------------
# Model Comparison
# -------------------------------

models = ["Logistic Regression", "Random Forest"]

accuracy_values = [
    accuracy_score(y_test, y_pred),
    accuracy_score(y_test, rf_pred)
]

precision_values = [
    precision_score(y_test, y_pred),
    precision_score(y_test, rf_pred)
]

recall_values = [
    recall_score(y_test, y_pred),
    recall_score(y_test, rf_pred)
]

f1_values = [
    f1_score(y_test, y_pred),
    f1_score(y_test, rf_pred)
]

roc_auc_values = [
    roc_auc_score(y_test, y_prob),
    roc_auc_score(y_test, rf_prob)
]

comparison_df = pd.DataFrame({
    "Model": models,
    "Accuracy": accuracy_values,
    "Precision": precision_values,
    "Recall": recall_values,
    "F1 Score": f1_values,
    "ROC-AUC": roc_auc_values
})

print("\nModel Comparison:")
print(comparison_df)

comparison_df.set_index("Model").plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig("model_comparison.png")
plt.close()

print("\nModel comparison graph saved as model_comparison.png")
# -------------------------------
# Save Random Forest Model
# -------------------------------

import joblib

joblib.dump(rf_model, "fraud_detection_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\nRandom Forest model saved as fraud_detection_model.pkl")
print("Scaler saved as scaler.pkl")