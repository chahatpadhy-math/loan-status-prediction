import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yaml


# Page Config

st.set_page_config(
    page_title="WSHG Loan Approval Prediction",
    layout="centered"
)


# Load Config

with open("config.yaml") as f:
    config = yaml.safe_load(f)


# Load Model

@st.cache_resource
def load_model():
    model = joblib.load("models/wshg_best_model.joblib")
    return model

model = load_model()


# Title

st.title("🏦 WSHG Loan Approval Prediction")
st.markdown(
    "Fill in the applicant and WSHG group details below to predict loan approval status."
)

st.divider()


# Form

with st.form("loan_form"):

    st.subheader("👤 Applicant Information")

    col1, col2 = st.columns(2)

    with col1:
        married = st.selectbox("Married", ["Yes", "No"])

        dependents = st.selectbox(
            "Dependents",
            [0, 1, 2, 4]
        )

        education = st.selectbox(
            "Education",
            ["Graduate", "Not Graduate"]
        )

        self_employed = st.selectbox(
            "Self Employed",
            ["Yes", "No"]
        )

        property_area = st.selectbox(
            "Property Area",
            ["Rural", "Semiurban", "Urban"]
        )

    with col2:

        applicant_income = st.number_input(
            "Applicant Income (₹)",
            min_value=0,
            value=5000,
            step=500
        )

        coapplicant_income = st.number_input(
            "Co-applicant Income (₹)",
            min_value=0,
            value=0,
            step=500
        )

        loan_amount = st.number_input(
            "Loan Amount (₹ thousands)",
            min_value=1,
            value=100,
            step=10
        )

        loan_amount_term = st.selectbox(
            "Loan Term (months)",
            [120, 180, 240, 300, 360, 480]
        )

        credit_history = st.selectbox(
            "Credit History",
            [1.0, 0.0],
            format_func=lambda x:
            "Good (1)" if x == 1.0 else "Bad (0)"
        )

    st.divider()

    st.subheader("👥 WSHG Group Information")

    col3, col4 = st.columns(2)

    with col3:

        group_size = st.slider(
            "Group Size",
            min_value=5,
            max_value=25,
            value=12
        )

        years_in_group = st.slider(
            "Years in Group",
            min_value=1,
            max_value=15,
            value=3
        )

        meeting_attendance_pct = st.slider(
            "Meeting Attendance (%)",
            min_value=0,
            max_value=100,
            value=75
        )

    with col4:

        group_savings = st.number_input(
            "Group Savings (₹)",
            min_value=0,
            value=25000,
            step=1000
        )

        group_loan_repayment_rate = st.slider(
            "Group Loan Repayment Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.85,
            step=0.01
        )

    st.divider()

    submitted = st.form_submit_button(
        "🔍 Predict Loan Status",
        use_container_width=True
    )


# Prediction

if submitted:

    # Encode categorical variables
    married_enc = 1 if married == "Yes" else 0

    education_enc = 1 if education == "Graduate" else 0

    self_employed_enc = 1 if self_employed == "Yes" else 0

    property_enc = {
        "Rural": 0,
        "Semiurban": 1,
        "Urban": 2
    }[property_area]

    # Feature Engineering
    total_income = applicant_income + coapplicant_income

    loan_income_ratio = (
        loan_amount / (total_income + 1)
    )

    group_strength_score = round(
        group_savings / 10000 +
        group_loan_repayment_rate * 10 +
        years_in_group * 0.5 +
        meeting_attendance_pct / 20,
        2
    )

    # Final Input DataFrame
    input_data = pd.DataFrame([{
        'Group_Size': group_size,
        'Years_in_Group': years_in_group,
        'Group_Savings': group_savings,
        'Group_Loan_Repayment_Rate': group_loan_repayment_rate,
        'Meeting_Attendance_Pct': meeting_attendance_pct,
        'Married': married_enc,
        'Dependents': dependents,
        'Education': education_enc,
        'Self_Employed': self_employed_enc,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_enc,
        'Total_Income': total_income,
        'Loan_Income_Ratio': loan_income_ratio,
        'Group_Strength_Score': group_strength_score
    }])

    # Prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    st.divider()

    if prediction == 1:

        st.success("✅ Loan Approved")

        st.metric(
            label="Approval Confidence",
            value=f"{probability[1] * 100:.1f}%"
        )

    else:

        st.error("❌ Loan Rejected")

        st.metric(
            label="Rejection Confidence",
            value=f"{probability[0] * 100:.1f}%"
        )

 
    # Input Summary

    st.divider()

    st.subheader("📋 Input Summary")

    summary = {
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self Employed": self_employed,
        "Property Area": property_area,
        "Applicant Income": f"₹{applicant_income:,}",
        "Co-applicant Income": f"₹{coapplicant_income:,}",
        "Loan Amount": f"₹{loan_amount}k",
        "Loan Term": f"{loan_amount_term} months",
        "Credit History":
            "Good" if credit_history == 1.0 else "Bad",
        "Group Size": group_size,
        "Years in Group": years_in_group,
        "Group Savings": f"₹{group_savings:,}",
        "Repayment Rate":
            f"{group_loan_repayment_rate * 100:.0f}%",
        "Meeting Attendance":
            f"{meeting_attendance_pct}%",
        "Total Income": f"₹{total_income:,}",
        "Loan-Income Ratio":
            f"{loan_income_ratio:.3f}",
        "Group Strength Score":
            group_strength_score
    }

    st.table(
        pd.DataFrame(
            summary.items(),
            columns=["Field", "Value"]
        )
    )


# Footer

st.divider()

st.caption(
    "WSHG Loan Approval Prediction System • Built with Streamlit"
)