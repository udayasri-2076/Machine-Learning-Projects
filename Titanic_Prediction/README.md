
# Titanic Survival Predictor 🚢

A Machine Learning web application that predicts whether a Titanic passenger would have survived based on passenger details such as age, gender, ticket class, fare, and family information. Built using Logistic Regression and deployed as an interactive Streamlit web application.

## 🚀 Live Demo

**Streamlit App:**
https://machine-learning-projects-kkx3db2ndasus6cuaosx8e.streamlit.app/

## 📁 GitHub Repository

Repository:
https://github.com/udayasri-2076/Machine-Learning-Projects

---

## 📌 Project Overview

The sinking of the RMS Titanic is one of the most well-known maritime disasters in history. This project uses Machine Learning to predict passenger survival based on demographic and travel-related information.

The model analyzes passenger attributes and predicts whether a passenger would have:

* Survived (1)
* Not Survived (0)

---

## 📊 Dataset

* **Source:** Kaggle Titanic Dataset
* **Rows:** 891
* **Columns:** 12
* **Target Variable:** `Survived`

### Key Features

* Passenger Class (Pclass)
* Sex
* Age
* Fare
* SibSp (Siblings/Spouses aboard)
* Parch (Parents/Children aboard)
* Embarked Port

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Streamlit

---

## ⚙️ ML Workflow

### 1. Data Preprocessing

* Loaded Titanic dataset
* Handled missing values using median and mode imputation
* Removed irrelevant columns
* Encoded categorical features
* Created new features such as:

  * FamilySize
  * IsAlone

### 2. Model Building

* Performed an 80/20 Train-Test Split
* Applied StandardScaler normalization
* Trained a Logistic Regression classifier

### 3. Model Evaluation

* Achieved **72% Accuracy**
* Evaluated using:

  * Accuracy Score
  * Precision Score
  * Recall Score
  * F1 Score
  * Confusion Matrix

---

## 🌐 Streamlit App Features

Users can:

* Enter passenger information
* Select passenger class
* Choose gender
* Input age and fare details
* Enter family information
* Predict survival probability instantly

---

## 📈 Model Performance

| Metric              | Score |
| ------------------- | ----- |
| Accuracy            | 72%   |
| Precision (Class 0) | 71%   |
| Precision (Class 1) | 74%   |
| F1 Score            | 0.71  |

---

## 🔍 Key Insights From EDA

* Female passengers had a significantly higher survival rate than males.
* First-class passengers survived more frequently than third-class passengers.
* Younger passengers generally had higher survival chances.
* Passengers traveling alone showed different survival patterns compared to families.
* Fare amount had a positive relationship with survival probability.

---

## 📂 Project Structure

```text
Titanic_Survival_Predictor/
│
├── app.py
├── Titanic.csv
├── Titanic Survival Predictor.ipynb
├── requirements.txt
└── README.md
```

---

## ▶️ Run Locally

```bash
# Clone repository
git clone https://github.com/udayasri-2076/Machine-Learning-Projects.git

# Navigate to project folder
cd Titanic_Survival_Predictor

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

---

## 🎯 Key Learnings

* Data Cleaning and Preprocessing
* Feature Engineering
* Logistic Regression Classification
* Model Evaluation Metrics
* Exploratory Data Analysis (EDA)
* Streamlit Application Development
* Cloud Deployment using Streamlit Community Cloud

---

## 👩‍💻 Author

**Udayasri Simma**

* GitHub: https://github.com/udayasri-2076
* LinkedIn: https://www.linkedin.com/in/udayasrisimma-b0541b331
