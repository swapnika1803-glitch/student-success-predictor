import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

# Title
st.title("Study Hours Predictor")

st.write("Enter study hours to predict marks")

# Sample dataset
hours = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape(-1, 1)
marks = np.array([10, 20, 30, 40, 50, 60, 70, 80])

# Train model
model = LinearRegression()
model.fit(hours, marks)

# User input
study_hours = st.number_input("Enter study hours", min_value=0.0, max_value=12.0, step=0.5)

# Prediction button
if st.button("Predict"):
    prediction = model.predict([[study_hours]])
    st.success(f"Predicted Marks: {prediction[0]:.2f}")