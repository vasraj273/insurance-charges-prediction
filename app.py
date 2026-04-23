import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Insurance Charges Prediction",
    page_icon="💰",
    layout="wide"
)

# Load trained model
model = joblib.load("insurance_model.pkl")

# Load dataset for EDA
df = pd.read_csv("insurance.csv")

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Predict Charges", "EDA Dashboard", "Bulk Prediction"]
)

# ===============================
# Overview Page
# ===============================

if page == "Overview":

    st.title("Insurance Charges Prediction App")
    st.markdown("---")

    st.markdown("""
    ### Project Overview

    This application predicts medical insurance charges using a trained
    **Random Forest Regression model**.

    The prediction is based on:

    - Age
    - BMI
    - Number of Dependents
    - Gender
    - Smoking Status

    Feature engineering was applied using:

    BMI × Age interaction feature
    """)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)


# ===============================
# Prediction Page
# ===============================

elif page == "Predict Charges":

    st.title("Predict Insurance Charges")
    st.markdown("---")

    age = st.number_input("Age", 18, 100, 30)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    dependents = st.number_input("Number of Dependents", 0, 10, 0)

    sex = st.selectbox("Sex", ["female", "male"])
    smoker = st.selectbox("Smoker", ["no", "yes"])

    sex_male = 1 if sex == "male" else 0
    smoker_yes = 1 if smoker == "yes" else 0

    bmi_age = bmi * age

    input_data = np.array([[
        age,
        bmi,
        dependents,
        bmi_age,
        sex_male,
        smoker_yes
    ]])

    if st.button("Predict Insurance Charges"):

        prediction = model.predict(input_data)

        st.markdown(f"""
        ### 💰 Estimated Insurance Charges

        ## $ {prediction[0]:,.2f}
        """)

        st.info("Prediction based on trained Random Forest model.")


# ===============================
# EDA Dashboard Page
# ===============================

elif page == "EDA Dashboard":

    st.title("Exploratory Data Analysis")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age vs Charges")
        fig, ax = plt.subplots(figsize=(5,3))
        sns.scatterplot(data=df, x="age", y="charges", ax=ax)
        st.pyplot(fig)

    with col2:
        st.subheader("BMI vs Charges")
        fig, ax = plt.subplots(figsize=(5,3))
        sns.scatterplot(data=df, x="bmi", y="charges", ax=ax)
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Smoking Status Impact")
        fig, ax = plt.subplots(figsize=(5,3))
        sns.boxplot(data=df, x="smoker", y="charges", ax=ax)
        st.pyplot(fig)

    with col4:
        st.subheader("Dependents Distribution")
        fig, ax = plt.subplots(figsize=(5,3))
        sns.countplot(data=df, x="dependents", ax=ax)
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(df.corr(numeric_only=True), annot=True,
                cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# ===============================
# Bulk Prediction Page
# ===============================

elif page == "Bulk Prediction":

    st.title("Bulk Insurance Charges Prediction")
    st.markdown("---")

    required_columns = [
        "age",
        "bmi",
        "dependents",
        "sex",
        "smoker"
    ]

    st.download_button(
        "📥 Download Sample CSV Template",
        "age,bmi,dependents,sex,smoker\n30,25,1,male,no",
        "insurance_sample.csv"
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:

        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset Preview")
        st.dataframe(data.head())

        data.columns = data.columns.str.lower()

        missing_cols = [
            col for col in required_columns
            if col not in data.columns
        ]

        if missing_cols:

            st.error(
                f"Missing required columns: {missing_cols}"
            )

        else:

            data = data[required_columns]

            data["sex_male"] = data["sex"].apply(
                lambda x: 1 if x == "male" else 0
            )

            data["smoker_yes"] = data["smoker"].apply(
                lambda x: 1 if x == "yes" else 0
            )

            data["bmi_age"] = data["bmi"] * data["age"]

            input_data = data[[
                "age",
                "bmi",
                "dependents",
                "bmi_age",
                "sex_male",
                "smoker_yes"
            ]]

            predictions = model.predict(input_data)

            data["Predicted Charges"] = predictions

            st.subheader("Prediction Results")
            st.dataframe(data)

            csv = data.to_csv(index=False)

            st.download_button(
                "📥 Download Results CSV",
                csv,
                "predictions.csv"
            )