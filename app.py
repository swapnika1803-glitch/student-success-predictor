import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Title
st.title("Student Success Predictor")

st.write("Enter student details to predict final marks")

# Sample dataset
data = {
    'study_hours': [2, 3, 4, 5, 6, 7, 8, 9],
    'sleep_hours': [6, 7, 6, 7, 8, 7, 8, 6],
    'attendance': [60, 65, 70, 75, 80, 85, 90, 95],
    'previous_marks': [50, 55, 60, 65, 70, 75, 80, 85],
    'final_marks': [55, 60, 65, 70, 75, 80, 85, 90]
}

df = pd.DataFrame(data)

# Features and target
X = df[['study_hours', 'sleep_hours', 'attendance', 'previous_marks']]
y = df['final_marks']

# Train model
model = LinearRegression()
model.fit(X, y)

# User Inputs
study_hours = st.number_input("Study Hours", 0.0, 12.0)
sleep_hours = st.number_input("Sleep Hours", 0.0, 12.0)
attendance = st.number_input("Attendance (%)", 0.0, 100.0)
previous_marks = st.number_input("Previous Marks", 0.0, 100.0)

# Prediction
if st.button("Predict Final Marks"):
    prediction = model.predict([[study_hours, sleep_hours, attendance, previous_marks]])
    st.success(f"Predicted Final Marks: {prediction[0]:.2f}")