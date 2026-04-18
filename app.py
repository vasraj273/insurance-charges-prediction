import streamlit as st
st.set_page_config(
    page_title="Insurance Charges Prediction",
    page_icon="💰",
    layout="wide"
)
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model
model = joblib.load("insurance_model.pkl")

# Load dataset for EDA
df = pd.read_csv("insurance.csv")

st.set_page_config(page_title="Insurance Charges Prediction App", layout="wide")

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

    This application predicts medical insurance charges using a trained **Random Forest Regression model**.

    The prediction is based on the following input features:

    - Age
    - BMI
    - Number of Children
    - Gender
    - Smoking Status
    - Region

    The model was trained using historical insurance dataset and evaluated using:

    - MAE
    - RMSE
    - R² Score

    This tool helps estimate expected insurance costs quickly and efficiently.
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
    children = st.number_input("Number of Dependents", 0, 10, 0)

    sex = st.selectbox("Sex", ["female", "male"])
    smoker = st.selectbox("Smoker", ["no", "yes"])
    region = st.selectbox(
        "Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )

    sex_male = 1 if sex == "male" else 0
    smoker_yes = 1 if smoker == "yes" else 0

    region_northwest = 1 if region == "northwest" else 0
    region_southeast = 1 if region == "southeast" else 0
    region_southwest = 1 if region == "southwest" else 0

    input_data = np.array([[age, bmi, children,
                            sex_male, smoker_yes,
                            region_northwest,
                            region_southeast,
                            region_southwest]])

    if st.button("Predict Insurance Charges"):

        prediction = model.predict(input_data)

        st.markdown(
    f"""
    ### 💰 Estimated Insurance Charges

    ## ₹ {prediction[0]:,.2f}
    """
    )

    st.info("Prediction based on trained Random Forest model.")


# ===============================
# EDA Dashboard Page
# ===============================

elif page == "EDA Dashboard":

    st.title("Exploratory Data Analysis")
    st.markdown("---")
    
    # First row
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


    # Second row
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Smoking Status Impact")
        fig, ax = plt.subplots(figsize=(5,3))
        sns.boxplot(data=df, x="smoker", y="charges", ax=ax)
        st.pyplot(fig)

    with col4:
        st.subheader("Region Distribution")
        fig, ax = plt.subplots(figsize=(5,3))
        sns.countplot(data=df, x="region", ax=ax)
        st.pyplot(fig)


    # Third row (heatmap full width)
    st.subheader("Correlation Heatmap")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
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
        "children",
        "sex",
        "smoker",
        "region"
    ]

    st.download_button(
    "📥 Download Sample CSV Template",
    "age,bmi,children,sex,smoker,region\n30,25,1,male,no,northeast",
    "insurance.csv"
    )
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:

        data = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Dataset Preview")
        st.dataframe(data.head())

        # Convert column names to lowercase
        data.columns = data.columns.str.lower()

        # Check missing columns
        missing_cols = [
            col for col in required_columns
            if col not in data.columns
        ]

        if missing_cols:

            st.error(
                f"Missing required columns: {missing_cols}"
            )

        else:

            # Ignore extra columns safely
            data = data[required_columns]

            # Encoding categorical variables
            data["sex_male"] = data["sex"].apply(
                lambda x: 1 if x == "male" else 0
            )

            data["smoker_yes"] = data["smoker"].apply(
                lambda x: 1 if x == "yes" else 0
            )

            data["region_northwest"] = data["region"].apply(
                lambda x: 1 if x == "northwest" else 0
            )

            data["region_southeast"] = data["region"].apply(
                lambda x: 1 if x == "southeast" else 0
            )

            data["region_southwest"] = data["region"].apply(
                lambda x: 1 if x == "southwest" else 0
            )

            input_data = data[[
                "age",
                "bmi",
                "children",
                "sex_male",
                "smoker_yes",
                "region_northwest",
                "region_southeast",
                "region_southwest"
            ]]

            predictions = model.predict(input_data)

            data["predicted_charges"] = predictions

            st.success("Predictions completed successfully")

            st.dataframe(data)

            csv = data.to_csv(index=False)

            st.download_button(
                "Download Predictions CSV",
                csv,
                "predicted_results.csv",
                "text/csv"
            )

st.markdown("---")
st.caption("Developed by Vashisht Rajpurohit | Machine Learning Deployment Project")