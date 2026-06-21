import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

@st.cache_resource
def train_model():
    df = pd.read_csv('train.csv')
    
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    if 'Cabin' in df.columns:
        df.drop(columns=['Cabin'], inplace=True)
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    df.drop(columns=['Name', 'Ticket', 'PassengerId'], inplace=True)
    
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return model, scaler

model, scaler = train_model()

st.title("Titanic Survival Predictor 🚢")
st.write("Enter passenger details to predict survival")

pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
age = st.slider("Age", 1, 80, 25)
sibsp = st.number_input("Siblings/Spouse count", 0, 8, 0)
parch = st.number_input("Parents/Children count", 0, 6, 0)
fare = st.number_input("Fare paid", 0.0, 500.0, 50.0)
embarked = st.selectbox("Embarked from", ["S", "C", "Q"])

sex = 0 if sex == "Male" else 1
embarked = 0 if embarked == "S" else 1 if embarked == "C" else 2
family_size = sibsp + parch + 1

if st.button("Predict"):
    input_data = np.array([[pclass, sex, age, sibsp,
                            parch, fare, embarked, family_size]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.success("✅ Passenger SURVIVED")
    else:
        st.error("❌ Passenger did NOT survive")
