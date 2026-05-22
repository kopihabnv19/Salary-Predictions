import streamlit as st
import pickle
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(
    page_title="Salary Prediction App",
    page_icon="💼",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    font-size: 55px;
    font-weight: bold;
    color: white;
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 20px;
    color: #A0A0A0;
    text-align: center;
    margin-bottom: 40px;
}

.stButton>button {
    width: 100%;
    background: linear-gradient(90deg, #ff4b4b, #ff6b6b);
    color: white;
    border-radius: 12px;
    height: 3.2em;
    font-size: 20px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(90deg, #ff6b6b, #ff4b4b);
}

.result-box {
    background-color: #132A13;
    padding: 25px;
    border-radius: 15px;
    text-align: center;
    color: #4ADE80;
    font-size: 30px;
    font-weight: bold;
    margin-top: 25px;
    box-shadow: 0px 0px 15px rgba(74,222,128,0.3);
}

</style>
""", unsafe_allow_html=True)

# Load pickle file
with open("salary_data.pkl", "rb") as file:
    data = pickle.load(file)

# Prepare model
X = data[["YearsExperience"]]
y = data["Salary"]

model = LinearRegression()
model.fit(X, y)

# UI
st.markdown('<p class="title">💼 Salary Prediction App</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Predict your estimated salary based on experience</p>',
    unsafe_allow_html=True
)

# Experience Slider
experience = st.slider(
    "Years of Experience",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.1
)

# Predict Button
if st.button("Predict Salary"):

    prediction = model.predict([[experience]])

    st.markdown(
        f"""
        <div class="result-box">
            Estimated Salary <br><br>
            ₹ {prediction[0]:,.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.balloons()
