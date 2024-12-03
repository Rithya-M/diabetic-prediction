# Import necessary libraries
import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Web app title
st.title("Diabetic Prediction App")

# Step 1: Load the dataset
file_path = r"C:\Users\HP\Downloads\Diabetic Prediction Project\Diabetic Prediction Project\diabetes.csv"

# Column names as per the Pima Indians Diabetes dataset
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI','DiabetesPedigreeFunction', 'Age', 'Outcome']

# Check if the file exists and load it
try:
    df = pd.read_csv(file_path, names=columns, header=0)
except FileNotFoundError:
    st.error("Dataset not found! Please check the file path.")
    st.stop()

# Preprocessing
X = df.drop(columns='Outcome')
y = df['Outcome']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))
st.write(f"Model Accuracy: {accuracy*100:.2f}%")

# Step 2: User Input
st.header("Enter Your Details for Prediction")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=300, step=1)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, step=1)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, step=1)
bmi = st.number_input("BMI (Body Mass Index)", min_value=0.0, max_value=70.0, step=0.1)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.01)
age = st.number_input("Age", min_value=0, max_value=120, step=1)

# Predict button
if st.button("Predict"):
    # Normalize the input values
    input_data = scaler.transform([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])

    # Make prediction
    prediction = model.predict(input_data)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    st.subheader(f"Prediction: You are {result}.")
