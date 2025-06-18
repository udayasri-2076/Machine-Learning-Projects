# Diabetes Prediction ML Project

This machine learning project predicts whether a person is diabetic based on medical information such as Glucose levels, Blood Pressure, BMI, etc. The model is built using Support Vector Machine (SVM) and includes a Streamlit web application for live predictions.


# Project Files

- `Diabetes Prediction.ipynb` – Jupyter Notebook for full model building and evaluation
- `diabetes.csv` – Input dataset used to train the model
- `streamlit_app.py` – Web app for real-time diabetes prediction
- `README.md` – This documentation


##  Dataset Description

This dataset contains the following features:

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`
- `Outcome` (0 = Non-Diabetic, 1 = Diabetic)

Source: PIMA Indian Diabetes Dataset


## Jupyter Notebook Highlights (`Diabetes Prediction.ipynb`)

- ✅ Loaded and explored the dataset using `pandas`
- ✅ Checked for missing values and basic statistical summaries
- 🔍 Split the dataset using `train_test_split`
- ⚖️ Scaled features using `StandardScaler` for SVM model performance
- 🤖 Built and trained an **SVM Classifier** using `sklearn.svm.SVC`
- 📊 Evaluated using `accuracy_score` and tested predictions

> Final accuracy: **~78%**


## Streamlit Web App (`streamlit_app.py`)

This app allows users to:
- Input personal health details like glucose level, BMI, insulin level, etc.
- Click a button to predict if the person is likely diabetic

The app uses the trained **SVM model** and displays the result instantly.

---

## How to Run the App

Install Streamlit first:

```bash
pip install streamlit

Open the anaconda powershell prompt and paste the path of ur project then tun the code as

streamlit run diabetesapp.py
