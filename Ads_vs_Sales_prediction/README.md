# Ads vs Sales Prediction

This machine learning project explores how advertising budgets across different channels (TV, Radio, Newspaper) impact product sales. A linear regression model is trained to predict sales based on advertising spend.

---

## Project Files

- `Ads vs Sales Prediction.ipynb` – Jupyter notebook with full analysis and ML pipeline
- `advertising.csv` – Dataset containing ad spend vs sales data
- `app.py` – Web app to predict sales based on custom ad spend
- `README.md` – Project documentation

---

##  Dataset Description

The dataset has 4 columns:
- `TV`: Ad budget spent on TV (in thousands of dollars)
- `Radio`: Ad budget spent on Radio
- `Newspaper`: Ad budget spent on Newspaper
- `Sales`: Number of units sold (target)

---

## What’s in the Jupyter Notebook?

- ✅ Loaded and explored the dataset with Pandas
- 📊 Visualized feature relationships using Seaborn pair plots and correlation heatmaps
- 🧼 Performed feature scaling (if needed)
- 🤖 Built a **Linear Regression** model
- 📈 Evaluated the model with R² score and prediction plots

> The model showed a strong correlation between TV + Radio ads and increased sales.

---

## Streamlit Web App

This app allows users to:
- Input ad budgets (TV, Radio, Newspaper)
- Get instant predicted sales using the trained linear regression model

To run the app:

```bash
Open Anaconda powershell prompt and add path to the file where you saved your streamlit code, you can write your streamlit code in pycharm or in notepad but u need to install pycharm for the code and then give instruction in the powershell prompt as given below.
streamlit run app.py

