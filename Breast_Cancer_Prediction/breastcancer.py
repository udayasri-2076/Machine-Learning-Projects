import os
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

# --- FIX 1: PATH HANDLING ---
# This ensures Streamlit finds the CSV right next to this script file inside your folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "breast cancer.csv")

# Load Data
df = pd.read_csv(csv_path)

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

# --- FIX 2: PREDICT INDEXING ---
# Function: Predict (Flattened array to handle manual vs random data shapes cleanly)
def predict(data):
    pred_array = model.predict(data)
    prediction = int(pred_array)
    
    probabilities = model.predict_proba(data).flatten()
    proba = float(probabilities[prediction])
    
    return prediction, proba

# Input Selection
if input_method == "Use Random Test Data":
    # --- FIX 3: CORRECTED SHAPE INDEXING ---
    # X_test.shape explicitly pulls out the number of rows as a single integer
    max_index = int(X_test.shape) - 1
    random_index = st.sidebar.slider("Pick a sample index", 0, max_index)
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
