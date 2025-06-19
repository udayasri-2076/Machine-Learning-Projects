import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load and preprocess dataset
@st.cache_data
def load_data():
    car = pd.read_csv("quikr_car.csv")
    car.columns = ['name', 'company', 'year', 'price', 'kms_driven', 'fuel_type']
    car = car[car['price'] != "Ask For Price"]
    car.dropna(inplace=True)

    car['price'] = car['price'].str.replace('Rs.', '').str.replace(',', '').astype(int)
    car['kms_driven'] = car['kms_driven'].str.replace(' kms', '').str.replace(',', '')
    car = car[car['kms_driven'].str.isnumeric()]
    car['kms_driven'] = car['kms_driven'].astype(int)
    car['year'] = car['year'].astype(int)
    car['name'] = car['name'].str.split().str[0]

    car = car[['name', 'company', 'year', 'kms_driven', 'fuel_type', 'price']]
    car_encoded = pd.get_dummies(car, drop_first=True)

    X = car_encoded.drop('price', axis=1)
    y = car_encoded['price']
    return X, y, car_encoded

X, y, car_encoded = load_data()

# Train model
model = LinearRegression()
model.fit(X, y)

# App UI
st.title("🚗 Car Price Predictor App")
st.markdown("Enter car details to estimate the price (based on Quikr data)")

# Input form
col1, col2 = st.columns(2)

with col1:
    year = st.selectbox("Year of Purchase", sorted(car_encoded['year'].unique()))
    kms_driven = st.number_input("Kilometers Driven", min_value=0, step=500)
    fuel_type = st.selectbox("Fuel Type", sorted([col.replace("fuel_type_", "") for col in car_encoded.columns if "fuel_type_" in col]))

with col2:
    company = st.selectbox("Company", sorted([col.replace("company_", "") for col in car_encoded.columns if "company_" in col]))
    car_name = st.selectbox("Car Brand", sorted([col.replace("name_", "") for col in car_encoded.columns if "name_" in col]))

# Predict button
if st.button("Predict Price"):
    # Create a zero row with same features as X
    input_data = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)

    # Fill in user inputs
    input_data['year'] = year
    input_data['kms_driven'] = kms_driven

    fuel_col = f"fuel_type_{fuel_type}"
    company_col = f"company_{company}"
    name_col = f"name_{car_name}"

    if fuel_col in input_data.columns:
        input_data[fuel_col] = 1
    if company_col in input_data.columns:
        input_data[company_col] = 1
    if name_col in input_data.columns:
        input_data[name_col] = 1

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Price: ₹{int(prediction):,}")
