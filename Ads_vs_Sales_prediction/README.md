# Ads vs Sales Prediction

This Machine Learning project predicts product sales based on advertising budgets spent across different marketing channels such as TV, Radio, and Newspaper. A Multiple Linear Regression model is trained using historical advertising data and deployed as an interactive Streamlit web application.

---

## Live Demo

**Streamlit App:**
https://machine-learning-projects-8wcu9vepkzqkjjapm8gdgc.streamlit.app/

---

## GitHub Repository

Repository:
https://github.com/udayasri-2076/Machine-Learning-Projects

---

## Project Overview

Businesses invest in multiple advertising channels to increase product sales. This project analyzes the relationship between advertising spend and sales performance using Machine Learning.

The trained model predicts expected sales based on:

* TV advertising budget
* Radio advertising budget
* Newspaper advertising budget

---

## Dataset Description

The dataset contains the following features:

| Feature   | Description                           |
| --------- | ------------------------------------- |
| TV        | Advertising budget spent on TV        |
| Radio     | Advertising budget spent on Radio     |
| Newspaper | Advertising budget spent on Newspaper |
| Sales     | Product sales (Target Variable)       |

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Streamlit

---

## Machine Learning Workflow

### Data Preprocessing

* Loaded dataset using Pandas
* Removed unnecessary columns
* Checked data quality

### Exploratory Data Analysis

* Correlation Analysis
* Pair Plots
* Feature Relationship Visualization
* Sales Trend Analysis

### Model Building

* Multiple Linear Regression
* Train-Test Split
* Model Training using Scikit-Learn

### Model Evaluation

* Prediction Analysis
* R² Score Evaluation
* Performance Validation

---

## Streamlit Application Features

The deployed web application allows users to:

* Enter TV advertising budget
* Enter Radio advertising budget
* Enter Newspaper advertising budget
* Predict expected sales instantly

---

## Project Structure

```text
Ads_vs_Sales_prediction/
│
├── app.py
├── Advertising.csv
├── requirements.txt
├── README.md
└── Ads vs Sales Prediction.ipynb
```

---

## Run Locally

### 1. Clone Repository

```bash
git clone https://github.com/udayasri-2076/Machine-Learning-Projects.git
```

### 2. Navigate to Project Folder

```bash
cd Ads_vs_Sales_prediction
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Streamlit App

```bash
streamlit run app.py
```

---

## Sample Prediction

Input:

* TV Budget = 100
* Radio Budget = 50
* Newspaper Budget = 25

Output:

```text
Predicted Sales: XX.XX
```

---

## Key Learnings

* Data Preprocessing using Pandas
* Exploratory Data Analysis (EDA)
* Multiple Linear Regression
* Model Evaluation
* Streamlit Application Development
* GitHub Project Management
* Cloud Deployment using Streamlit Community Cloud

---

## Author

**Udayasri Simma**

GitHub: https://github.com/udayasri-2076

LinkedIn: Add your LinkedIn profile link here
