import streamlit as st
import pandas as pd
import joblib


# Page configuration

st.set_page_config(page_title="Telco Customer Churn Predictor", page_icon="📡", layout="centered")


# Load model, scaler, and the exact column order used at training time

@st.cache_resource
def load_artifacts():
    model = joblib.load("Telco_Churn_Model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, scaler, feature_columns

model, scaler, FEATURE_COLUMNS = load_artifacts()

INTERNET_DEPENDENT_FIELDS = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

DEFAULTS = {
    "tenure": 12,
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
    "SeniorCitizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "PaperlessBilling": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check",
}


# Session state init / reset

def init_state():
    for key, val in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = val
    if "prediction_result" not in st.session_state:
        st.session_state["prediction_result"] = None

def reset_form():
    for key, val in DEFAULTS.items():
        st.session_state[key] = val
    st.session_state["prediction_result"] = None

init_state()

# Detect "InternetService" is "No" and set its dependent fields automatically into "No internet service"
def sync_internet_fields():
    if st.session_state["InternetService"] == "No":
        for f in INTERNET_DEPENDENT_FIELDS:
            st.session_state[f] = "No internet service"
    else:
        # If a field was previously forced to "No internet service", reset it to "No"
        for f in INTERNET_DEPENDENT_FIELDS:
            if st.session_state[f] == "No internet service":
                st.session_state[f] = "No"


# Header

st.title("📡 Telco Customer Churn Predictor")
st.write(
    "Fill your details below and click **Predict** to estimate "
    "the likelihood that this you will churn."
)

st.divider()


# Numerical inputs(Tenure, MonthlyCharges and TotalCharges)

st.subheader("Account Details")

col1, col2, col3 = st.columns(3)
with col1:
    tenure = st.number_input(
        "Tenure (months)", min_value=0, max_value=100, step=1, key="tenure"
    )
with col2:
    monthly_charges = st.number_input(
        "Monthly Charges ($)", min_value=0.0, max_value=1000.0, step=0.5, key="MonthlyCharges"
    )
with col3:
    total_charges = st.number_input(
        "Total Charges ($)", min_value=0.0, max_value=20000.0, step=1.0, key="TotalCharges"
    )

st.divider()


# Demographic / account dropdowns

st.subheader("Customer Profile")

col1, col2, col3 = st.columns(3)
with col1:
    senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"], key="SeniorCitizen")
with col2:
    partner = st.selectbox("Partner", ["Yes", "No"], key="Partner")
with col3:
    dependents = st.selectbox("Dependents", ["Yes", "No"], key="Dependents")

col1, col2 = st.columns(2)
with col1:
    contract = st.selectbox(
        "Contract", ["Month-to-month", "One year", "Two year"], key="Contract"
    )
with col2:
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        key="PaymentMethod",
    )

paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"], key="PaperlessBilling")

st.divider()


# Phone services

st.subheader("Phone Services")

col1, col2 = st.columns(2)
with col1:
    phone_service = st.selectbox("Phone Service", ["Yes", "No"], key="PhoneService")
with col2:
    multiple_lines_options = (
        ["No phone service"] if phone_service == "No" else ["Yes", "No"]
    )
    if phone_service == "No":
        st.session_state["MultipleLines"] = "No phone service"
    elif st.session_state["MultipleLines"] == "No phone service":
        st.session_state["MultipleLines"] = "No"
    multiple_lines = st.selectbox(
        "Multiple Lines",
        multiple_lines_options,
        key="MultipleLines",
        disabled=(phone_service == "No"),
    )

st.divider()


# Internet services

st.subheader("Internet Services")

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"],
    key="InternetService",
    on_change=sync_internet_fields,
)

internet_disabled = internet_service == "No"
if internet_disabled:
    st.caption("ℹ️ Internet-related services are disabled because this customer has no internet service.")

internet_options = ["No internet service"] if internet_disabled else ["Yes", "No"]

col1, col2 = st.columns(2)
with col1:
    online_security = st.selectbox(
        "Online Security", internet_options, key="OnlineSecurity", disabled=internet_disabled
    )
    device_protection = st.selectbox(
        "Device Protection", internet_options, key="DeviceProtection", disabled=internet_disabled
    )
    streaming_tv = st.selectbox(
        "Streaming TV", internet_options, key="StreamingTV", disabled=internet_disabled
    )
with col2:
    online_backup = st.selectbox(
        "Online Backup", internet_options, key="OnlineBackup", disabled=internet_disabled
    )
    tech_support = st.selectbox(
        "Tech Support", internet_options, key="TechSupport", disabled=internet_disabled
    )
    streaming_movies = st.selectbox(
        "Streaming Movies", internet_options, key="StreamingMovies", disabled=internet_disabled
    )

st.divider()


# tenure * MonthlyCharges (allow some tolerance for discounts/price changes)

TOLERANCE = 1.5  # allow up to 50% above the expected total

def check_charges_consistency(tenure_val, monthly_val, total_val):
    expected_max = tenure_val * monthly_val * TOLERANCE
    # Special case: tenure = 0 (brand new customer) should have ~0 total charges
    if tenure_val == 0:
        if total_val > monthly_val:
            return False, expected_max
        return True, expected_max
    if total_val > expected_max:
        return False, expected_max
    return True, expected_max

is_consistent, expected_max = check_charges_consistency(
    st.session_state["tenure"], st.session_state["MonthlyCharges"], st.session_state["TotalCharges"]
)

if not is_consistent:
    st.warning(
        f"⚠️ **Total Charges looks inconsistent.** With a tenure of "
        f"{st.session_state['tenure']} month(s) and Monthly Charges of "
        f"${st.session_state['MonthlyCharges']:.2f}, Total Charges shouldn't "
        f"reasonably exceed about **${expected_max:.2f}**. Please double-check "
        f"these values — Predict is disabled until this is fixed."
    )


# Buttons

btn_col1, btn_col2 = st.columns(2)
predict_clicked = btn_col1.button(
    "🔮 Predict", use_container_width=True, type="primary", disabled=not is_consistent
)
reset_clicked = btn_col2.button("🔄 Reset", use_container_width=True, on_click=reset_form)


# Build the model-ready feature row from current inputs

def build_input_row():
    raw = {
        "SeniorCitizen": 1 if st.session_state["SeniorCitizen"] == "Yes" else 0,
        "Partner": 1 if st.session_state["Partner"] == "Yes" else 0,
        "Dependents": 1 if st.session_state["Dependents"] == "Yes" else 0,
        "tenure": st.session_state["tenure"],
        "PhoneService": 1 if st.session_state["PhoneService"] == "Yes" else 0,
        "PaperlessBilling": 1 if st.session_state["PaperlessBilling"] == "Yes" else 0,
        "MonthlyCharges": st.session_state["MonthlyCharges"],
        "TotalCharges": st.session_state["TotalCharges"],
        "MultipleLines": st.session_state["MultipleLines"],
        "InternetService": st.session_state["InternetService"],
        "OnlineSecurity": st.session_state["OnlineSecurity"],
        "OnlineBackup": st.session_state["OnlineBackup"],
        "DeviceProtection": st.session_state["DeviceProtection"],
        "TechSupport": st.session_state["TechSupport"],
        "StreamingTV": st.session_state["StreamingTV"],
        "StreamingMovies": st.session_state["StreamingMovies"],
        "Contract": st.session_state["Contract"],
        "PaymentMethod": st.session_state["PaymentMethod"],
    }
    df_row = pd.DataFrame([raw])

    categorical_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod",
    ]
    df_dummies = pd.get_dummies(df_row, columns=categorical_cols)

    # Ensure every column the model expects exists, in the right order
    for col in FEATURE_COLUMNS:
        if col not in df_dummies.columns:
            df_dummies[col] = 0
    df_dummies = df_dummies[FEATURE_COLUMNS]

    return df_dummies


# Predict

if predict_clicked:
    input_df = build_input_row()
    input_scaled = scaler.transform(input_df)
    pred = int(model.predict(input_scaled)[0])
    proba = float(model.predict_proba(input_scaled)[0][1])
    st.session_state["prediction_result"] = (pred, proba)

if st.session_state["prediction_result"] is not None:
    pred, proba = st.session_state["prediction_result"]
    st.divider()
    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"⚠️ This customer is **likely to churn**.\n\nChurn probability: **{proba:.1%}**")
    else:
        st.success(f"✅ This customer is **likely to stay**.\n\nChurn probability: **{proba:.1%}**")
    st.progress(min(max(proba, 0.0), 1.0))
