
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.pipeline.column_mapping import ColumnMapping
import matplotlib.pyplot as plt
import shap

# Load model and SHAP image
model_pipeline = joblib.load("xgb_midwest_cost_model.pkl")




# Set page config
st.set_page_config(page_title="Midwest Patient Cost Estimator", layout="wide")
st.title("🏥 Midwest Patient Cost Estimator")
st.markdown("Estimate patient hospital visit cost based on demographics, visit reason, and medical details.")

# About section
with st.expander("ℹ️ About This Model"):
    st.markdown("""
    This estimator uses an XGBoost regression model trained on a synthetic dataset of hospital visits across the Midwest. 
    The model considers patient demographics, comorbidities, and visit-specific details like reason for visit and whether imaging or labs were required.

    **Model Performance Metrics:**
    - **MAE (Mean Absolute Error):** $31.15
    - **RMSE (Root Mean Squared Error):** $47.35
    - **R² Score:** 0.957

    These metrics indicate the model is highly accurate, with predictions on average within $31 of actual values.
    """)

# Sidebar: Patient Demographics
st.sidebar.header("🧑‍⚕️ Patient Information")
age = st.sidebar.slider("Age", 10, 95, 50)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
state = st.sidebar.selectbox("State", ["MN", "WI", "IA", "IL", "IN", "MI", "OH", "ND", "SD", "NE"])

# Sidebar: Visit Details
st.sidebar.header("📋 Visit Details")
reason_for_visit = st.sidebar.selectbox("Reason for Visit", [
    "Checkup", "Cough/Fever", "Chest Pain", "Cardiology Consult", "Surgical Consult",
    "Neurology Referral", "GI Complaint", "Fracture Evaluation", "Prenatal Visit", "Urinary Issue",
    "Back Pain", "Skin Rash", "Mental Health", "Diabetes Management", "Follow-up Visit",
    "Medication Refill", "Physical Therapy", "Ear Infection", "Annual Physical", "Sports Injury"
])
lab_required = st.sidebar.checkbox("Lab Work Required")
imaging_required = st.sidebar.checkbox("Imaging Required")

# Sidebar: Comorbidities
st.sidebar.header("🩺 Comorbidities")
comorbidities = {
    "diabetic": st.sidebar.checkbox("Diabetic"),
    "obese": st.sidebar.checkbox("Obese"),
    "smoker": st.sidebar.checkbox("Smoker"),
    "hypertensive": st.sidebar.checkbox("Hypertensive"),
    "asthmatic": st.sidebar.checkbox("Asthmatic")
}

# Create DataFrame for prediction
input_data = pd.DataFrame([{
    "age": age,
    "gender": gender,
    "hospital_state": state,
    "hospital_name": "Generic Midwest Hospital",  # placeholder
    "reason_for_visit": reason_for_visit,
    "lab_required": lab_required,
    "imaging_required": imaging_required,
    **comorbidities
}])

# Predict and display result
predicted_cost = model_pipeline.predict(input_data)[0]
st.subheader(f"💵 Estimated Cost: ${predicted_cost:,.2f}")

# Log input data to current_data.csv
log_path = "current_data.csv"
input_data.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)

# SHAP Explanation Section 
with st.expander("📊 Cost Drivers Explanation (SHAP)"):
    st.markdown("This plot shows how the top 10 features influence cost predictions.")

    try:
        # Display SHAP summary image
        st.image("shap_summary_cleaned.png", caption="Top Cost Drivers (Cleaned Names)", use_column_width=True)

        # Add explanation of top drivers (you can hardcode or parse from the plot code if needed)
        st.markdown("**🧠 Top Cost Drivers:**")
        st.markdown("""
        - **Imaging Required** with average SHAP impact of `92.29`  
        - **Lab Work Required** with average SHAP impact of `53.74`  
        - **Visit: Surgical Consult** with average SHAP impact of `39.45`  
        - **Visit: Neurology Referral** with average SHAP impact of `31.85`  
        - **Visit: GI Complaint** with average SHAP impact of `30.86`
        """)
    except Exception as e:
        st.warning(f"SHAP explanation could not be displayed: {e}")


# Drift Report 
with st.expander("📉 Data Drift Report (Evidently)"):
    try:
        ref_df = pd.read_csv("reference_data.csv")
        curr_df = pd.read_csv("current_data.csv")

        # Show data sample sizes
        st.markdown(f"**Reference data size:** {len(ref_df):,} rows | **Current data size:** {len(curr_df):,} rows")

        # Define column mapping
        column_mapping = ColumnMapping()
        column_mapping.categorical_features = ["gender", "hospital_state", "reason_for_visit"]
        column_mapping.numerical_features = [
            "age", "lab_required", "imaging_required",
            "diabetic", "obese", "smoker", "hypertensive", "asthmatic"
        ]
        column_mapping.target = None

        # Generate Evidently report
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=ref_df,
            current_data=curr_df,
            column_mapping=column_mapping
        )
        report.save_html("drift_report.html")

        with open("drift_report.html", "r", encoding="utf-8") as f:
            st.components.v1.html(f.read(), height=600, scrolling=True)

        # Extract drifted columns
        drift_results = report.as_dict()
        drifted_columns = []
        try:
            drift_by_col = drift_results["metrics"][0]["result"]["drift_by_columns"]
            drifted_columns = [col for col, res in drift_by_col.items() if res.get("drift_detected") is True]
        except Exception:
            pass

        if drifted_columns:
            st.markdown("### 🚨 Drift Detected In:")
            for col in drifted_columns:
                st.markdown(f"- **{col}**")
        else:
            st.markdown("✅ No significant column drift detected.")

    except Exception as e:
        st.warning(f"Drift report could not be generated: {e}")

