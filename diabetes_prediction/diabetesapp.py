import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.title("🩺 Diabetes Prediction App")

uploaded_file = st.file_uploader("Upload Diabetes Dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    if 'Outcome' not in df.columns:
        st.error("⚠️ Uploaded CSV must have an 'Outcome' column.")
    else:
        st.success("✅ File uploaded successfully!")
        st.dataframe(df.head())

        # Features and labels
        X = df.drop("Outcome", axis=1)
        y = df["Outcome"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model training
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)

        # Sidebar Inputs
        st.sidebar.header("Enter Patient Health Info:")
        Pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
        Glucose = st.sidebar.number_input("Glucose", 0, 200, 120)
        BloodPressure = st.sidebar.number_input("Blood Pressure", 0, 150, 70)
        SkinThickness = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
        Insulin = st.sidebar.number_input("Insulin", 0, 900, 85)
        BMI = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
        DiabetesPedigreeFunction = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        Age = st.sidebar.number_input("Age", 0, 120, 30)

        # Prediction button
        if st.sidebar.button("Predict"):
            input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                    Insulin, BMI, DiabetesPedigreeFunction, Age]])
            prediction = model.predict(input_data)[0]

            if prediction == 1:
                st.error("⚠️ The model predicts the patient is **Diabetic**.")
            else:
                st.success("✅ The model predicts the patient is **Not Diabetic**.")

