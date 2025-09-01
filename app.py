import streamlit as st
import pandas as pd
import joblib

# Load model and feature list
model, model_features = joblib.load("pest_model.pkl")

# Inputs
temperature = st.number_input("🌡️ Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)
rainfall = st.number_input("💧 Rainfall (mm)", min_value=0.0, max_value=500.0, value=50.0)
humidity = st.number_input("💦 Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
crop = st.selectbox("🌾 Crop Type", ["Wheat", "Rice", "Cotton", "Maize"])
season = st.selectbox("☀️ Season", ["Rainy", "Summer", "Winter"])

# Prepare DataFrame
input_data = pd.DataFrame({
    "Temperature": [temperature],
    "Rainfall": [rainfall],
    "Humidity": [humidity],
    "Crop": [crop],
    "Season": [season],
})

# One-hot encode with same training columns
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=model_features, fill_value=0)

# Predict
if st.button("🔮 Predict Pest Risk"):
    pred = model.predict(input_data)[0]
    if pred == 1:
        st.error("⚠️ High Risk of Pest Attack! Take Preventive Action.")
    else:
        st.success("✅ Low Risk of Pest Attack. Safe for now.")
