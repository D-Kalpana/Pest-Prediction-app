import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Page config
st.set_page_config(page_title="🌾 Pest Attack Prediction", layout="centered")

st.title("🌾 Pest Attack Prediction App")
st.write("Upload crop details to predict whether a pest attack is likely.")

# ---------------------------
# Train model (cached)
# ---------------------------
@st.cache_resource
def load_model():
    df = pd.read_csv("pest_attack_dataset.csv")  # dataset in GitHub repo
    X = df.drop("Pest_Attack", axis=1)
    y = df["Pest_Attack"]

    # Train-test split
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    return model, X.columns.tolist()

model, model_features = load_model()

# ---------------------------
# User Input
# ---------------------------
st.subheader("Enter Crop Data")

user_input = []
for feature in model_features:
    value = st.number_input(f"Enter {feature}:", min_value=0.0, step=1.0)
    user_input.append(value)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict"):
    input_df = pd.DataFrame([user_input], columns=model_features)
    prediction = model.predict(input_df)[0]
    prediction_text = "⚠️ Pest Attack Likely!" if prediction == 1 else "✅ No Pest Attack"
    st.subheader("Prediction Result")
    st.success(prediction_text
