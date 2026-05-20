import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('models/loan_model.pkl', 'rb'))

st.title('Loan Status Prediction App')

# Inputs
Gender = st.selectbox('Gender', ['Male', 'Female'])

Married = st.selectbox('Married', ['Yes', 'No'])

Dependents = st.selectbox('Dependents', ['0', '1', '2', '3+'])

Education = st.selectbox('Education', ['Graduate', 'Not Graduate'])

Self_Employed = st.selectbox('Self Employed', ['Yes', 'No'])

ApplicantIncome = st.number_input('Applicant Income')

CoapplicantIncome = st.number_input('Coapplicant Income')

LoanAmount = st.number_input('Loan Amount')

Loan_Amount_Term = st.selectbox(
    'Loan Amount Term',
    [120, 180, 240, 300, 360]
)

Credit_History = st.selectbox('Credit History', [1, 0])

Property_Area = st.selectbox(
    'Property Area',
    ['Urban', 'Semiurban', 'Rural']
)

# Prediction
if st.button('Predict'):

    gender = 1 if Gender == 'Male' else 0

    married = 1 if Married == 'Yes' else 0

    if Dependents == '3+':
        dependents = 3
    else:
        dependents = int(Dependents)

    education = 1 if Education == 'Graduate' else 0

    self_employed = 1 if Self_Employed == 'Yes' else 0

    if Property_Area == 'Urban':
        property_area = 2
    elif Property_Area == 'Semiurban':
        property_area = 1
    else:
        property_area = 0

    # SAME NUMBER OF FEATURES AS TRAINING
    features = np.array([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        ApplicantIncome,
        CoapplicantIncome,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        property_area
    ]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.success('Loan Approved')
    else:
        st.error('Loan Rejected')