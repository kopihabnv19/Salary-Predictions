import streamlit as st
import pickle
from sklearn.linear_model import LinearRegression

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Salary Predictor",
    page_icon="💸",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.main {
    background: linear-gradient(135deg, #0f172a, #111827);
    color: white;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-card {
    background: rgba(255,255,255,0.05);
    padding: 40px;
    border-radius: 25px;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0,0,0,0.35);
}

.title {
    text-align: center;
    font-size: 58px;
    font-weight: 800;
    color: white;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 20px;
    color: #9CA3AF;
    margin-bottom: 40px;
}

.stSlider label {
    color: white !important;
    font-size: 18px !important;
    font-weight: 600;
}

.stButton > button {
    width: 100%;
    height: 60px;
    border-radius: 16px;
    border: none;
    background: linear-gradient(90deg,#7c3aed,#ec4899);
    color: white;
    font-size: 22px;
    font-weight: bold;
    transition: all 0.3s ease;
    margin-top: 20px;
}

.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0px 0px 20px rgba(236,72,153,0.6);
}

.result-container {
    margin-top: 35px;
    padding: 35px;
    border-radius: 20px;
    text-align: center;
    background: linear-gradient(135deg,#16a34a,#22c55e);
    box-shadow: 0px 0px 25px rgba(34,197,94,0.45);
    animation: fadeIn 0.7s ease-in-out;
}

.result-label {
    font-size: 24px;
    color: white;
    margin-bottom: 15px;
}

.result-value {
    font-size: 42px;
    font-weight: 800;
    color: white;
}

.bottom-text {
    text-align: center;
    margin-top: 30px;
    color: #9CA3AF;
    font-size: 15px;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(15px);
    }
    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
with open("salary_data.pkl", "rb") as file:
    data = pickle.load(file)

# ---------------- MODEL ----------------
X = data[["YearsExperience"]]
y = data["Salary"]

model = LinearRegression()
model.fit(X, y)

# ---------------- UI ----------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.markdown(
    '<div class="title">💸 Salary Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered salary estimation based on your experience</div>',
    unsafe_allow_html=True
)

# Slider
experience = st.slider(
    "Years of Experience",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.1
)

# Predict Button
if st.button("🚀 Predict Salary"):

    prediction = model.predict([[experience]])

    st.markdown(f"""
        <div class="result-container">
            <div class="result-label">
                Estimated Salary
            </div>

            <div class="result-value">
                ₹ {prediction[0]:,.2f}
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown(
    '<div class="bottom-text">Built with Streamlit & Machine Learning</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
