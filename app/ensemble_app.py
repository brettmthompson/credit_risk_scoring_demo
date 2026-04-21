import streamlit as st
import requests
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(
    page_title="Credit Risk Ensemble Predictor",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Risk Ensemble Predictor")
st.markdown("**Predict loan default risk using an ensemble of XGBoost, LightGBM, and Scikit-learn models**")

# Model configuration (defaults from environment variables, editable in UI)
st.sidebar.header("⚙️ Model Configuration")

st.sidebar.subheader("Endpoints")
xgb_endpoint = st.sidebar.text_input(
    "XGBoost Endpoint",
    value=os.getenv("XGBOOST_ENDPOINT", ""),
    help="XGBoost InferenceService endpoint"
)
lgb_endpoint = st.sidebar.text_input(
    "LightGBM Endpoint",
    value=os.getenv("LIGHTGBM_ENDPOINT", ""),
    help="LightGBM InferenceService endpoint"
)
sklearn_endpoint = st.sidebar.text_input(
    "Sklearn Endpoint",
    value=os.getenv("SKLEARN_ENDPOINT", ""),
    help="Sklearn InferenceService endpoint"
)

st.sidebar.subheader("Model Names")
xgb_model = st.sidebar.text_input(
    "XGBoost Model Name",
    value=os.getenv("XGBOOST_MODEL_NAME", ""),
    help="Model name as configured in MLServer"
)
lgb_model = st.sidebar.text_input(
    "LightGBM Model Name",
    value=os.getenv("LIGHTGBM_MODEL_NAME", ""),
    help="Model name as configured in MLServer"
)
sklearn_model = st.sidebar.text_input(
    "Sklearn Model Name",
    value=os.getenv("SKLEARN_MODEL_NAME", ""),
    help="Model name as configured in MLServer"
)

st.sidebar.markdown("---")
st.sidebar.subheader("Decision Threshold")
decision_threshold = st.sidebar.slider(
    "Default Probability Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Deny loan if default probability exceeds this value. Lower threshold = stricter (deny more), higher threshold = lenient (approve more)."
)
st.sidebar.caption(f"Current: {decision_threshold:.0%} - Probability > {decision_threshold:.0%} = DENY")

st.sidebar.markdown("---")
st.sidebar.markdown("*Edit values above or set via environment variables*")

# Feature inputs
st.header("Applicant Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Details")
    person_age = st.slider("Age", 18, 100, 35, help="Applicant's age in years")
    person_income = st.number_input("Annual Income ($)", 0, 500000, 50000, step=5000, help="Annual income in USD")
    person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    person_emp_length = st.slider("Employment Length (years)", 0, 50, 5, help="Years at current employer")

with col2:
    st.subheader("Loan Details")
    loan_intent = st.selectbox("Loan Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"])
    loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    loan_amnt = st.number_input("Loan Amount ($)", 0, 100000, 10000, step=1000, help="Requested loan amount")
    loan_int_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 10.0, 0.5, help="Loan interest rate")

with col3:
    st.subheader("Credit History")
    loan_percent_income = st.slider("Loan % of Income", 0.0, 1.0, 0.2, 0.01, help="Loan amount as % of annual income")
    cb_person_default_on_file = st.selectbox("Previous Default on File", ["Y", "N"])
    cb_person_cred_hist_length = st.slider("Credit History Length (years)", 0, 50, 10, help="Years of credit history")

# Encode categorical features (same as training)
home_ownership_map = {"RENT": 0, "OWN": 1, "MORTGAGE": 2, "OTHER": 3}
loan_intent_map = {"EDUCATION": 0, "MEDICAL": 1, "VENTURE": 2, "PERSONAL": 3, "DEBTCONSOLIDATION": 4, "HOMEIMPROVEMENT": 5}
loan_grade_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6}
default_map = {"Y": 1, "N": 0}

# Prepare feature vector (order must match training)
features = [
    person_age,
    person_income,
    home_ownership_map[person_home_ownership],
    person_emp_length,
    loan_intent_map[loan_intent],
    loan_grade_map[loan_grade],
    loan_amnt,
    loan_int_rate,
    loan_percent_income,
    default_map[cb_person_default_on_file],
    cb_person_cred_hist_length
]

feature_names = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_default_on_file", "cb_person_cred_hist_length"
]

st.divider()

# Prediction button
if st.button("🔮 Predict Default Risk", type="primary", use_container_width=True):
    with st.spinner("Calling model endpoints..."):
        try:
            # MLServer V2 inference protocol format
            inference_request = {
                "inputs": [
                    {
                        "name": "input-0",
                        "shape": [1, 11],
                        "datatype": "FP64",
                        "data": [features]
                    }
                ]
            }

            results = {}
            errors = []

            # Define inference function for parallel execution
            def call_model(name, endpoint, model_name):
                try:
                    response = requests.post(
                        f"{endpoint}/v2/models/{model_name}/infer",
                        json=inference_request,
                        verify=False,  # Skip SSL verification for self-signed certs
                        timeout=5
                    )
                    response.raise_for_status()
                    # Handle different formats: sklearn=[P(class0), P(class1)], others=[P(class1)]
                    # Take last element to get positive class probability in both cases
                    probability = response.json()["outputs"][0]["data"][-1]
                    return name, probability, None
                except Exception as e:
                    return name, None, str(e)

            # Call all models in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(call_model, "XGBoost", xgb_endpoint, xgb_model): "XGBoost",
                    executor.submit(call_model, "LightGBM", lgb_endpoint, lgb_model): "LightGBM",
                    executor.submit(call_model, "Sklearn", sklearn_endpoint, sklearn_model): "Sklearn"
                }

                for future in as_completed(futures):
                    name, result, error = future.result()
                    if error:
                        errors.append(f"{name}: {error}")
                    else:
                        results[name] = result

            # Display errors if any
            if errors:
                st.error("⚠️ Some models failed:")
                for error in errors:
                    st.error(error)

            # Display results if we have any
            if results:
                st.header("📊 Prediction Results")

                # Individual model predictions (ordered consistently)
                model_order = ["XGBoost", "LightGBM", "Sklearn"]
                ordered_results = [(name, results[name]) for name in model_order if name in results]

                cols = st.columns(len(ordered_results))
                for idx, (model_name, prob) in enumerate(ordered_results):
                    with cols[idx]:
                        st.metric(
                            label=f"{model_name}",
                            value=f"{prob:.1%}",
                            delta=None
                        )
                        if prob > decision_threshold:
                            st.error("🔴 High Risk")
                        else:
                            st.success("🟢 Low Risk")

                # Ensemble prediction (average)
                st.divider()
                ensemble_prob = sum(results.values()) / len(results)

                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.subheader("🎯 Ensemble Prediction")
                    st.metric(
                        label="Default Probability",
                        value=f"{ensemble_prob:.1%}",
                        delta=None
                    )

                    if ensemble_prob > decision_threshold:
                        st.error("### ❌ DENY LOAN")
                        st.markdown(f"**Risk Level:** HIGH ({ensemble_prob:.1%} probability of default)")
                        st.caption(f"Threshold: {decision_threshold:.0%}")
                    else:
                        st.success("### ✅ APPROVE LOAN")
                        st.markdown(f"**Risk Level:** LOW ({ensemble_prob:.1%} probability of default)")
                        st.caption(f"Threshold: {decision_threshold:.0%}")

                    # Progress bar visualization
                    st.progress(ensemble_prob)

                # Model agreement analysis
                st.divider()
                st.subheader("Model Agreement")

                predictions = [1 if p > decision_threshold else 0 for p in results.values()]
                agreement = sum(predictions)

                if agreement == 0:
                    st.success("✅ All models agree: LOW RISK")
                elif agreement == len(predictions):
                    st.error("❌ All models agree: HIGH RISK")
                else:
                    st.warning(f"⚠️ Mixed predictions: {agreement}/{len(predictions)} models predict default")

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

# Feature vector display (for debugging)
with st.expander("🔍 View Feature Vector"):
    st.json({name: val for name, val in zip(feature_names, features)})
