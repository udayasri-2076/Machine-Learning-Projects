import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Page Config
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")
st.title("🧬 Breast Cancer Prediction")
st.markdown("This app uses Machine Learning to predict whether a tumor is **Benign** or **Malignant** based on medical data.")

# Load Data
df = pd.read_csv("D:\\MLProjects\\Datasets\\breast cancer.csv")
if 'id' in df.columns:
    df = df.drop(['id'], axis=1)

# Encode Target
df['diagnosis'] = LabelEncoder().fit_transform(df['diagnosis'])

# Features and Labels
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# Sidebar Options
st.sidebar.header("Input Options")
input_method = st.sidebar.radio("Choose Input Method", ["Use Random Test Data", "Enter Manually"])

# Function: Predict
def predict(data):
    prediction = model.predict(data)
    proba = model.predict_proba(data)[0][prediction[0]]
    return prediction[0], proba

# Input
if input_method == "Use Random Test Data":
    random_index = st.sidebar.slider("Pick a sample index", 0, X_test.shape[0] - 1)
    input_data = X_test.iloc[random_index:random_index + 1]
    st.subheader("📊 Selected Test Sample Data")
    st.dataframe(input_data)
else:
    st.subheader("🖊️ Enter Values Manually")
    input_data = {}
    col1, col2 = st.columns(2)
    for i, feature in enumerate(X.columns):
        if i % 2 == 0:
            input_data[feature] = col1.slider(feature, float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))
        else:
            input_data[feature] = col2.slider(feature, float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))
    input_data = pd.DataFrame([input_data])

# Predict Button
if st.button("🔍 Predict"):
    result, confidence = predict(input_data)
    st.subheader("🧾 Result")
    if result == 1:
        st.error(f"Malignant Tumor (Cancerous) with {confidence * 100:.2f}% confidence.")
    else:
        st.success(f"Benign Tumor (Non-Cancerous) with {confidence * 100:.2f}% confidence.")

# Show Accuracy
st.sidebar.markdown(f"🔢 Model Accuracy: **{acc * 100:.2f}%**")
