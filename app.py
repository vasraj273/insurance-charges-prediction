import streamlit as st
import joblib
import numpy as np

st.markdown("---")
st.write("This application predicts insurance charges using a trained Random Forest Regression model.")
st.markdown("---")

# Load trained model
model = joblib.load("insurance_model.pkl")

st.title("Insurance Charges Prediction App")

st.write("Enter the details below to predict insurance charges.")

# User Inputs
age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)

sex = st.selectbox("Sex", ["female", "male"])
smoker = st.selectbox("Smoker", ["no", "yes"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

# Convert categorical inputs into encoded format
sex_male = 1 if sex == "male" else 0
smoker_yes = 1 if smoker == "yes" else 0

region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0

# Arrange input in same order as training
input_data = np.array([[age, bmi, children,
                        sex_male, smoker_yes,
                        region_northwest, region_southeast, region_southwest]])

if st.button("Predict Insurance Charges"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Insurance Charges: ₹{prediction[0]:,.2f}")

st.info("Note: This prediction is based on historical data patterns and may vary in real-world scenarios.")