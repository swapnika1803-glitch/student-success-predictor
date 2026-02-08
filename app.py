import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Load Dataset
# -----------------------------
data = pd.read_csv("student_data.csv")

# Convert Pass/Fail to numeric
data["Result"] = data["Result"].map({"Fail": 0, "Pass": 1})

# Features and Target
X = data[["StudyHours", "SleepHours", "Attendance", "PreviousMarks"]]
y = data["Result"]

# -----------------------------
# Train Model
# -----------------------------
model = LogisticRegression()
model.fit(X, y)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Student Success Predictor")

st.write("Enter student details to predict result:")

study_hours = st.number_input("Study Hours", min_value=0, max_value=12, value=4)
sleep_hours = st.number_input("Sleep Hours", min_value=0, max_value=12, value=6)
attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=70)
previous_marks = st.number_input("Previous Marks", min_value=0, max_value=100, value=50)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    input_data = np.array([[study_hours, sleep_hours, attendance, previous_marks]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Prediction: Student will PASS")
    else:
        st.error("Prediction: Student may FAIL")