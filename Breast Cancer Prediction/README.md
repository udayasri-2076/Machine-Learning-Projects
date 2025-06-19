#  Breast Cancer Prediction using Logistic Regression

This project uses a machine learning model to predict whether a tumor is **malignant** (cancerous) or **benign** (non-cancerous) using medical diagnostic data. The model is built using **Logistic Regression**, a powerful yet interpretable classification algorithm.

---

##  Project Structure

- `Breast cancer prediction.ipynb` – Jupyter notebook with data analysis, model training & evaluation  
- `data.csv` – Breast cancer dataset used for the project  
- `streamlit_app.py` – (optional) Streamlit web app for live predictions  
- `README.md` – This file

---

##  Dataset Info

- 📍 Source: Breast Cancer Wisconsin Dataset (from `sklearn.datasets`)
- ✅ Features: 30 numerical columns such as:
  - `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, etc.
- 🎯 Target:
  - `0` – Malignant (Cancerous)
  - `1` – Benign (Non-cancerous)

---

##  What’s Inside the Notebook?

### 🔹 Data Loading & Preprocessing
- Loaded dataset using `sklearn.datasets.load_breast_cancer`
- Converted to pandas DataFrame
- Checked for missing values
- Normalized features using `StandardScaler`

### 🔹 Model Building
- Split data into training and testing sets (80/20)
- Trained a **Logistic Regression** classifier using `sklearn.linear_model`
- Achieved accuracy of **~96–97%** on test set

### 🔹 Evaluation
- Evaluated using accuracy score, confusion matrix, and classification report
- Performed predictions on custom input data

---

##  Streamlit Web App (Optional)

The app allows users to:
- Input 30 feature values
- Instantly get a prediction: **Benign** or **Malignant**

To run the app:

```bash
streamlit run breast cancer.py

