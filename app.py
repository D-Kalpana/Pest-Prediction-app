import streamlit as st
import pandas as pd
import joblib

# Load model + feature names
model = joblib.load("pest_model.pkl")
model_features = joblib.load("model_features.pkl")

st.set_page_config(page_title="🌾 Pest Attack Prediction", layout="centered")
st.title("🌾 Pest Attack Prediction on Farmlands")

temperature = st.number_input("🌡️ Temperature (°C)", 0, 50, 30)
rainfall = st.number_input("💧 Rainfall (mm)", 0, 500, 50)
humidity = st.number_input("🌫️ Humidity (%)", 0, 100, 60)
crop = st.selectbox("🌱 Crop Type", ["Wheat", "Rice", "Cotton", "Maize"])
season = st.selectbox("🗓️ Season", ["Kharif", "Rabi", "Zaid"])

# Convert input into dataframe
input_data = pd.DataFrame({
    "Temperature": [temperature],
    "Rainfall": [rainfall],
    "Humidity": [humidity],
    "Crop": [crop],
    "Season": [season]
})

# Match training features
input_data = pd.get_dummies(input_data)
input_data = input_data.reindex(columns=model_features, fill_value=0)

if st.button("🔮 Predict Pest Risk"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High Risk of Pest Attack! Take Preventive Action.")
    else:
        st.success("✅ Low Risk of Pest Attack. Safe for now.")
