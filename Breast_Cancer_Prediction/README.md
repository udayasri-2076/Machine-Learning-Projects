# Breast Cancer Prediction

A Machine Learning project that predicts whether a tumor is **Malignant (Cancerous)** or **Benign (Non-Cancerous)** using medical diagnostic data. Built using Logistic Regression and deployed as an interactive Streamlit web application.

## 🚀 Live Demo

**Streamlit App:**
https://machine-learning-projects-juunphabldu7nv6rhzatqy.streamlit.app/

## 📁 GitHub Repository

Repository:
https://github.com/udayasri-2076/Machine-Learning-Projects

---

## 📌 Project Overview

Early detection of breast cancer significantly improves treatment outcomes. This project uses the Breast Cancer Wisconsin Dataset to classify tumors as malignant or benign based on 30 medical diagnostic features.

---

## 📊 Dataset

* **Source:** Breast Cancer Wisconsin Dataset (`sklearn.datasets`)
* **Samples:** 569
* **Features:** 30 numerical features including `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, etc.
* **Target:**

  * `0` — Malignant (Cancerous)
  * `1` — Benign (Non-Cancerous)

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit

---

## ⚙️ ML Workflow

### 1. Data Preprocessing

* Loaded dataset from `sklearn.datasets`
* Converted to Pandas DataFrame
* Checked for missing values
* Standardized features using `StandardScaler`

### 2. Model Building

* Performed an 80/20 Train-Test Split
* Trained a Logistic Regression classifier using Scikit-Learn

### 3. Model Evaluation

* Achieved **96%+ Accuracy**
* Evaluated using:

  * Accuracy Score
  * Confusion Matrix
  * Classification Report
* Tested on custom input data

---

## 🌐 Streamlit App Features

Users can:

* Select a random test sample
* Enter custom diagnostic values
* Predict whether the tumor is Benign or Malignant
* View prediction confidence percentage instantly

---

## 📂 Project Structure

```text
Breast_Cancer_Prediction/
│
├── breastcancer.py
├── breast cancer.csv
├── Breast cancer prediction.ipynb
├── requirements.txt
└── README.md
```

---

## ▶️ Run Locally

```bash
# Clone repository
git clone https://github.com/udayasri-2076/Machine-Learning-Projects.git

# Navigate to project folder
cd Breast_Cancer_Prediction

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run breastcancer.py
```

---

## 🎯 Key Learnings

* Data Preprocessing and Feature Engineering
* Logistic Regression Classification
* Model Evaluation Techniques
* Streamlit Application Development
* GitHub Project Management
* Cloud Deployment using Streamlit Community Cloud

---

## 👩‍💻 Author

**Udayasri Simma**

* GitHub: https://github.com/udayasri-2076
* LinkedIn: https://www.linkedin.com/in/udayasrisimma-b0541b331

