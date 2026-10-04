import streamlit as st
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")

# Page settings
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Enter transaction details and click Predict Fraud."
)

st.divider()

with st.form("fraud_form"):

    # Basic transaction details
    st.subheader("Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        time = st.number_input(
            "Transaction Time",
            value=0.0
        )

    with col2:
        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=0.0
        )

    st.subheader("Transaction Features")

    # Create three columns
    columns = st.columns(3)

    v_values = []

    for i in range(1, 29):
        with columns[(i - 1) % 3]:
            value = st.number_input(
                f"V{i}",
                value=0.0,
                format="%.6f"
            )
            v_values.append(value)

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Fraud",
        type="primary",
        use_container_width=True
    )


if submitted:

    feature_columns = [
        "Time",
        "V1", "V2", "V3", "V4", "V5", "V6", "V7",
        "V8", "V9", "V10", "V11", "V12", "V13", "V14",
        "V15", "V16", "V17", "V18", "V19", "V20", "V21",
        "V22", "V23", "V24", "V25", "V26", "V27", "V28",
        "Amount"
    ]

    input_data = pd.DataFrame(
        [[time] + v_values + [amount]],
        columns=feature_columns
    )

    # Scale Time and Amount
    input_data[["Time", "Amount"]] = scaler.transform(
        input_data[["Time", "Amount"]]
    )

    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ FRAUDULENT TRANSACTION")
    else:
        st.success("✅ NORMAL TRANSACTION")

    st.info(f"Fraud Probability: {probability:.2%}")