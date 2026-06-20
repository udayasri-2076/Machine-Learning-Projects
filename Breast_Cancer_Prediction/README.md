Breast Cancer Prediction

A Machine Learning project that predicts whether a tumor is Malignant (Cancerous) or Benign (Non-Cancerous) using medical diagnostic data. Built using Logistic Regression and deployed as an interactive Streamlit web application.

🚀 Live Demo

Streamlit App: Click here to try it

📁 GitHub Repository

github.com/udayasri-2076/Machine-Learning-Projects


📌 Project Overview

Early detection of breast cancer significantly improves treatment outcomes. This project uses the Breast Cancer Wisconsin Dataset to classify tumors as malignant or benign based on 30 medical diagnostic features.


📊 Dataset


Source: Breast Cancer Wisconsin Dataset (sklearn.datasets)
Samples: 569
Features: 30 numerical features including radius_mean, texture_mean, perimeter_mean, area_mean, etc.
Target:

0 — Malignant (Cancerous)
1 — Benign (Non-Cancerous)






🛠️ Technologies Used


Python
Pandas, NumPy
Scikit-Learn
Streamlit



⚙️ ML Workflow

1. Data Preprocessing


Loaded dataset from sklearn.datasets
Converted to pandas DataFrame
Checked for missing values
Normalized features using StandardScaler


2. Model Building


80/20 Train-Test Split
Trained a Logistic Regression classifier using Scikit-Learn


3. Model Evaluation


Accuracy: 96%+
Evaluated using accuracy score, confusion matrix, and classification report
Tested on custom input data



🌐 Streamlit App Features

Users can:


Choose between random test sample or manual input
Input 30 feature values using sliders
Instantly get prediction: Benign or Malignant
See confidence percentage of the prediction



📂 Project Structure

Breast_Cancer_Prediction/
│
├── breastcancer.py
├── breast cancer.csv
├── Breast cancer prediction.ipynb
├── requirements.txt
└── README.md


▶️ Run Locally

bash# Clone repository
git clone https://github.com/udayasri-2076/Machine-Learning-Projects.git

# Navigate to project folder
cd Breast_Cancer_Prediction

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run breastcancer.py


👩‍💻 Author

Udayasri Simma


GitHub: udayasri-2076
LinkedIn: udayasrisimma
