import streamlit as st
import pandas as pd
import joblib

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Dengue Prediction App", page_icon="🦟", layout="centered")

# ---- HEADER ----
st.markdown("""
    <h1 style='text-align: center; color: #FF5733;'>Dengue Prediction App</h1>
    <p style='text-align: center;'>Predict whether a person has dengue based on given parameters.</p>
    <hr style='border: 1px solid #FF5733;'>
""", unsafe_allow_html=True)

# ---- SIDEBAR ----
st.sidebar.header("User Input Parameters")

Gender = st.sidebar.selectbox("Gender", ["Male", "Female"], index=0)
Age = st.sidebar.slider("Age", min_value=0, max_value=100, value=25)
NS1 = st.sidebar.selectbox("NS1 (Non-structural Protein 1)", [0, 1], index=0)
IgG = st.sidebar.selectbox("IgG (Immunoglobulin G)", [0, 1], index=0)
IgM = st.sidebar.selectbox("IgM (Immunoglobulin M)", [0, 1], index=0)

Area = st.sidebar.selectbox("Area", ["Mirpur", "Chawkbazar", "Paltan", "Motijheel", "Gendaria",
       "Dhanmondi", "New Market", "Sher-e-Bangla Nagar", "Kafrul",
       "Pallabi", "Mohammadpur", "Shahbagh", "Shyampur", "Kalabagan",
       "Bosila", "Jatrabari", "Adabor", "Kamrangirchar", "Biman Bandar",
       "Ramna", "Badda", "Bangshal", "Sabujbagh", "Hazaribagh",
       "Sutrapur", "Lalbagh", "Demra", "Banasree", "Cantonment",
       "Keraniganj", "Tejgaon", "Khilkhet", "Kadamtali", "Gulshan",
       "Rampura", "Khilgaon"], index=0)

AreaType = st.sidebar.selectbox("Area Type", ["Undeveloped", "Developed"], index=1)
HouseType = st.sidebar.selectbox("House Type", ["Building", "Other", "Tinshed"], index=0)
District = st.sidebar.selectbox("District", ["Dhaka"], index=0)

# ---- INPUT DATA ----
input_data = {
    "Gender": Gender,
    "Age": Age,
    "NS1": NS1,
    "IgG": IgG,
    "IgM": IgM,
    "Area": Area,
    "AreaType": AreaType,
    "HouseType": HouseType,
    "District": District
}
input_data_df = pd.DataFrame([input_data])

# ---- LOAD MODEL & PREDICT ----
model = joblib.load("models/log_reg_with_pipeline.pkl")
result = model.predict(input_data_df)[0]

# ---- DISPLAY RESULTS ----
st.markdown("""
    <h3 style='text-align: center; color: #3498db;'>User Input Data</h3>
""", unsafe_allow_html=True)
st.table(input_data_df)

# ---- RESULT DESIGN ----
prediction_text = "Dengue Positive 🦟" if result == 1 else "Dengue Negative ✅"
prediction_color = "#e74c3c" if result == 1 else "#2ecc71"

st.markdown(f"""
    <div style='text-align: center; padding: 20px; background-color: {prediction_color}; color: white; border-radius: 10px;'>
        <h2>{prediction_text}</h2>
    </div>
""", unsafe_allow_html=True)
