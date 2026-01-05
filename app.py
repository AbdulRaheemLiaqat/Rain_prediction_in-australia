import streamlit as st
import numpy as np
import joblib

# ================================
# LOAD SAVED OBJECTS
# ================================
model = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.set_page_config(page_title="Rain Prediction", layout="centered")

st.title("🌧️ Rain Tomorrow Prediction")
st.write("This app predicts whether it will rain tomorrow using a trained ML model.")

# ================================
# USER INPUTS (NUMERIC FEATURES)
# ================================
min_temp = st.number_input("Min Temperature (°C)", value=10.0)
max_temp = st.number_input("Max Temperature (°C)", value=25.0)
rainfall = st.number_input("Rainfall (mm)", value=0.0)
evaporation = st.number_input("Evaporation", value=5.0)
sunshine = st.number_input("Sunshine (hours)", value=8.0)
humidity9am = st.number_input("Humidity at 9am (%)", value=70.0)
humidity3pm = st.number_input("Humidity at 3pm (%)", value=50.0)
pressure9am = st.number_input("Pressure at 9am", value=1015.0)
pressure3pm = st.number_input("Pressure at 3pm", value=1013.0)
temp9am = st.number_input("Temperature at 9am", value=18.0)
temp3pm = st.number_input("Temperature at 3pm", value=24.0)

wind_gust_dir = label_encoders["WindGustDir"].transform(["W"])[0]
wind_dir9am = label_encoders["WindDir9am"].transform(["W"])[0]
wind_dir3pm = label_encoders["WindDir3pm"].transform(["W"])[0]
location = label_encoders["Location"].transform(["Sydney"])[0]
raintoday = label_encoders["RainToday"].transform(["No"])[0]

wind_gust_speed = 40
wind_speed9am = 15
wind_speed3pm = 20
cloud9am = 5
cloud3pm = 5
year = 2015
month = 6
day = 15


if st.button("Predict"):
    input_data = np.array([[
        location, min_temp, max_temp, rainfall, evaporation, sunshine,
        wind_gust_dir, wind_gust
