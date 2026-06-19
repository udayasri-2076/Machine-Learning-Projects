import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import streamlit as st

# Load dataset
csv_path = os.path.join(os.path.dirname(__file__), "Advertising.csv")
sdf = pd.read_csv(csv_path)

# Drop unnecessary column if present
if "Unnamed: 0" in sdf.columns:
    sdf = sdf.drop(columns=["Unnamed: 0"])

# Features and target
x = sdf[["TV", "Radio", "Newspaper"]]
y = sdf["Sales"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=50
)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Streamlit UI
st.title("Sales Prediction using Multiple Linear Regression")

st.write(
    "Enter the advertising budgets for TV, Radio, and Newspaper to predict sales."
)

# Sidebar inputs
st.sidebar.header("Enter Advertising Budget")

tv_budget = st.sidebar.number_input(
    "TV Budget", min_value=0.0, value=100.0
)

radio_budget = st.sidebar.number_input(
    "Radio Budget", min_value=0.0, value=50.0
)

newspaper_budget = st.sidebar.number_input(
    "Newspaper Budget", min_value=0.0, value=25.0
)

# Prediction
if st.sidebar.button("Predict Sales"):
    sample_input = np.array(
        [[tv_budget, radio_budget, newspaper_budget]]
    )

    predicted_value = model.predict(sample_input)

    st.success(
        f"Predicted Sales: {predicted_value[0]:.2f}"
    )
