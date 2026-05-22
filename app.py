import streamlit as st
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load trained data from pkl file
with open("salary_data.pkl", "rb") as file:
    data = pickle.load(file)

# Prepare data
X = data[["YearsExperience"]]
y = data["Salary"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Streamlit page config
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="centered"
)

# Title
st.title("💼 Salary Prediction App")

st.write("Predict salary based on years of experience")

# User input
experience = st.slider(
    "Years of Experience",
    min_value=0.0,
    max_value=20.0,
    step=0.1
)

# Predict button
if st.button("Predict Salary"):

    prediction = model.predict([[experience]])

    st.success(f"Estimated Salary: ₹ {prediction[0]:,.2f}")

    st.balloons()

# Dataset preview
with st.expander("Show Dataset"):
    st.dataframe(data)
