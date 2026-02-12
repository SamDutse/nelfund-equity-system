import streamlit as st
import requests
import pandas as pd

API_URL = "https://nelfund-equity-system.onrender.com/predict"

st.title("NELFUND Eligibility Checker")

tuition = st.number_input("Tuition Amount", 0)
income = st.number_input("Household Income (Monthly)", 0)
cgpa = st.number_input("CGPA", 0.0, 5.0, step=0.1)
dependents = st.number_input("Number of Dependents", 0)

if st.button("Check Eligibility"):

    payload = {
        "school_id": "UNI_001",
        "programme_level": "Undergraduate",
        "tuition_amount": tuition,
        "household_income_monthly": income,
        "number_of_dependents": dependents,
        "orphan_status": 0,
        "disability_status": 0,
        "sponsor_type": "None",
        "has_other_scholarship": 0,
        "employment_status": "Unemployed",
        "cgpa": cgpa,
        "academic_standing": "Good",
        "years_remaining": 2,
        "state_of_origin": "Kaduna",
        "socioeconomic_index": 0.5,
        "income_bracket": "Low",
        "tuition_income_ratio": 2.5
    }

    response = requests.post(API_URL, json=payload)

    if response.status_code == 200:
        result = response.json()
        st.success(f"Approval Probability: {result['approval_probability']:.2f}")
    else:
        st.error("Error connecting to API")
