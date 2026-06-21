import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open('titanic_model.pkl', 'rb'))

st.title("Titanic Survival Predictor")
st.write("Enter passenger details to predict survival")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouse count", 0, 8, 0)
parch = st.number_input("Parents/Children count", 0, 6, 0)
fare = st.number_input("Fare paid", 0.0, 500.0, 50.0)
embarked = st.selectbox("Embarked from", ["S", "C", "Q"])

# Encode inputs
sex = 0 if sex == "Male" else 1
embarked = 0 if embarked == "S" else 1 if embarked == "C" else 2
family_size = sibsp + parch + 1

# Predict button
if st.button("Predict"):
    input_data = np.array([[pclass, sex, age, sibsp, 
                            parch, fare, embarked, family_size]])
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("✅ Passenger SURVIVED")
    else:
        st.error("❌ Passenger did NOT survive")