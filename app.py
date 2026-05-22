import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data = pd.read_csv("Salary_Data.csv")

# Independent and dependent variables
X = data[['YearsExperience']]
y = data['Salary']

# Train model
model = LinearRegression()
model.fit(X, y)

# Streamlit UI
st.title("💼 Salary Prediction App")

st.write("Enter Years of Experience to Predict Salary")

# User input
experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    step=0.1
)

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict([[experience]])

    st.success(f"Predicted Salary: ₹ {prediction[0]:,.2f}")
