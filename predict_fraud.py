import joblib
import numpy as np
import pandas as pd

# Load saved model and scaler
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

print("Credit Card Fraud Detection")
print("----------------------------")

# Enter transaction details
time = float(input("Enter Transaction Time: "))
amount = float(input("Enter Transaction Amount: "))

# Enter V1 to V28 values
v_values = []

for i in range(1, 29):
    value = float(input(f"Enter V{i}: "))
    v_values.append(value)

# Create input DataFrame with the same feature names used during training
columns = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
    "V15", "V16", "V17", "V18", "V19", "V20", "V21",
    "V22", "V23", "V24", "V25", "V26", "V27", "V28",
    "Amount"
]

input_data = pd.DataFrame(
    [[time] + v_values + [amount]],
    columns=columns
)

# Scale Time and Amount
input_data[["Time", "Amount"]] = scaler.transform(
    input_data[["Time", "Amount"]]
)

# Make prediction
prediction = model.predict(input_data)[0]
probability = model.predict_proba(input_data)[0][1]

print("\nPrediction Result:")

if prediction == 1:
    print("⚠️ FRAUDULENT TRANSACTION")
else:
    print("✅ NORMAL TRANSACTION")

print(f"Fraud Probability: {probability:.2%}")