#  Car Price Prediction using Linear Regression

This machine learning project predicts the **price of a car** based on various features like year, fuel type, seller type, kilometers driven, etc. The model is trained using **Linear Regression**, which is ideal for predicting continuous numerical values like price.

---

##  Project Files

- `Car price predictor.ipynb` – Jupyter notebook for model training and evaluation  
- `car_data.csv` – Dataset containing car details and price  
- `carapp.py` – (optional) Streamlit web app to predict car price interactively  
- `README.md` – Project overview and documentation

---

##  Dataset Description

Key features in the dataset include:

- `Present_Price`: Current ex-showroom price
- `Kms_Driven`: Total kilometers the car has driven
- `Fuel_Type`: Petrol / Diesel / CNG
- `Seller_Type`: Dealer / Individual
- `Transmission`: Manual / Automatic
- `Owner`: Number of previous owners
- `Year`: Year of purchase
- `Selling_Price`: Price the car was sold for (target)

---

##  Notebook Workflow

### 🔹 Data Exploration
- Loaded dataset using pandas
- Visualized correlations using heatmap
- Checked distributions of numerical and categorical features

### 🔹 Feature Engineering
- Calculated `car_age` from `Year`
- Dropped irrelevant columns like `Car_Name`
- Applied **one-hot encoding** to categorical columns
- Normalized feature scale using `StandardScaler`

### 🔹 Model Building
- Trained a **Linear Regression** model using `sklearn.linear_model`
- Used `train_test_split` for testing and validation
- Evaluated model with **R² score**, MAE, and MSE

> Achieved high accuracy in predicting car prices

---

##  Streamlit Web App (Optional)

A Streamlit app can be built where users enter:
- Car age, kilometers driven, fuel type, etc.

And get:
- Predicted price using the trained Linear Regression model

### To run the app:

```bash
streamlit run carapp.py

