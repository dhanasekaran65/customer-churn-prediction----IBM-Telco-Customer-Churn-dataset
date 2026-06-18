"""
predict_new_customer.py

Loads the trained churn model and scores a single new customer.
This simulates how the retention team would use the model day-to-day:
feed in a customer's current account details, get back a churn risk score.

Usage:
    python src/predict_new_customer.py
"""

import pandas as pd
import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "churn_model.pkl")


def predict_churn(customer: dict) -> dict:
    """Predict churn risk for a single new customer.

    Parameters
    ----------
    customer : dict
        Raw customer fields, matching the original dataset columns
        (excluding customerID and Churn).

    Returns
    -------
    dict with churn_probability (0-1) and a plain-English prediction label.
    """
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_columns = artifact["feature_columns"]
    numeric_columns = artifact["numeric_columns"]

    row = pd.DataFrame([customer])
    row_encoded = pd.get_dummies(row)
    row_encoded = row_encoded.reindex(columns=feature_columns, fill_value=0)
    row_encoded[numeric_columns] = scaler.transform(row_encoded[numeric_columns])

    proba = model.predict_proba(row_encoded)[0][1]
    label = "High risk \u2014 likely to churn" if proba >= 0.5 else "Low risk \u2014 likely to stay"

    return {"churn_probability": round(float(proba), 3), "prediction": label}


if __name__ == "__main__":
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 190.0,
    }

    result = predict_churn(sample_customer)
    print("Sample new customer (2 months tenure, month-to-month, fiber optic, high monthly bill):")
    print(result)
