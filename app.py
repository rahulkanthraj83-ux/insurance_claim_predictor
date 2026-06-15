import streamlit as st
import pickle
import pandas as pd
import json

st.set_page_config(page_title="Insurance Claim Predictor", page_icon="", layout="centered")

@st.cache_resource
def load_artifacts():
    model    = pickle.load(open("model.pkl", "rb"))
    encoders = pickle.load(open("encoders.pkl", "rb"))
    columns  = pickle.load(open("columns.pkl", "rb"))
    meta     = json.load(open("meta.json", "r"))
    return model, encoders, columns, meta

model, encoders, columns, meta = load_artifacts()
cat_options = meta["cat_options"]

st.title("Insurance Claim Predictor")
st.write("Fill in 5 details to predict if your claim will be Approved or Rejected")
st.markdown("---")

with st.form("claim_form"):

    c1, c2 = st.columns(2)
    claim_amount    = c1.number_input("Claim Amount (Rs)", 0, 10000000, 150000, step=5000)
    sum_insured     = c2.number_input("Sum Insured (Rs)", 0, 10000000, 500000, step=10000)

    c1, c2 = st.columns(2)
    coverage_months = c1.number_input("Coverage Duration (months)", 0, 360, 24)
    doc_status      = c2.selectbox("Document Status", cat_options["document_status"])

    pre_auth        = st.selectbox("Pre-Authorization Status", cat_options["pre_auth_status"])

    st.markdown("---")
    submitted = st.form_submit_button("Predict Claim Status", use_container_width=True, type="primary")

if submitted:
    defaults = {col: 0 for col in columns}

    updates = {
        "claim_amount_inr": claim_amount,
        "sum_insured_inr": sum_insured,
        "continuous_coverage_months": coverage_months,
        "document_status": encoders["document_status"].transform([doc_status])[0],
        "pre_auth_status": encoders["pre_auth_status"].transform([pre_auth])[0],
        "claim_coverage_ratio": round((claim_amount / sum_insured * 100), 2) if sum_insured > 0 else 0,
    }
    defaults.update(updates)

    input_df   = pd.DataFrame([defaults])[columns]
    pred       = model.predict(input_df)[0]
    proba      = model.predict_proba(input_df)[0]
    confidence = round(max(proba) * 100, 1)

    st.markdown("---")
    if pred == 1:
        st.success("CLAIM APPROVED - Confidence: " + str(confidence) + "%")
    else:
        st.error("CLAIM REJECTED - Confidence: " + str(confidence) + "%")

    c1, c2 = st.columns(2)
    c1.metric("Approved Probability", str(round(proba[1]*100,1)) + "%")
    c2.metric("Rejected Probability", str(round(proba[0]*100,1)) + "%")
