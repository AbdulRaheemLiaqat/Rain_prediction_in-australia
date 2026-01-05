import streamlit as st
import numpy as np
import joblib

model = joblib.load("random_forest.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🌧️ Rain Prediction in Australia")

st.write("Enter weather values to predict rain tomorrow")

min_temp = st.number_input("Min Temperature (°C)", value=10.0)
max_temp = st.number_input("Max Temperature (°C)", value=25.0)
rainfall = st.number_input("Rainfall (mm)", value=0.0)
humidity_3pm = st.number_input("Humidity at 3PM (%)", value=50.0)
pressure_3pm = st.number_input("Pressure at 3PM (hPa)", value=1015.0)

if st.button("Predict"):
    input_data = np.array([
        [min_temp, max_temp, rainfall, humidity_3pm, pressure_3pm]])

    total_features = scaler.mean_.shape[0]
    if input_data.shape[1] < total_features:
        padding = total_features - input_data.shape[1]
        input_data = np.pad(input_data, ((0, 0), (0, padding)))

    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("☔ Rain Tomorrow")
    else:
        st.success("☀️ No Rain Tomorrow")
