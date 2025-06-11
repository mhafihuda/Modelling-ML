import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("mental_health_risk_model.h5")

st.set_page_config(page_title="Maternal Health Risk Prediction", layout="centered")
st.title("🩺 Maternal Health Risk Prediction")
st.markdown("Masukkan data di bawah untuk memprediksi tingkat risiko kesehatan ibu hamil.")

with st.form("prediction_form"):
    age = st.number_input("Age (years)", min_value=10, max_value=60, value=25)
    systolic = st.number_input("Systolic Blood Pressure (mmHg)", 70, 200, value=120)
    diastolic = st.number_input("Diastolic Blood Pressure (mmHg)", 40, 140, value=80)
    bs = st.number_input("Blood Sugar (mg/dL)", 3.0, 20.0, value=7.0)
    temp = st.number_input("Body Temperature (°F)", 95.0, 105.0, value=98.0)
    hr = st.number_input("Heart Rate (bpm)", 50, 150, value=85)
    bmi = st.number_input("BMI", 10.0, 50.0, value=24.0)
    hypertension = st.selectbox("Hypertension", ["No", "Yes"])
    diabetes = st.selectbox("Diabetes", ["No", "Yes"])
    prev_comp = st.selectbox("Previous Complications", ["No", "Yes"])
    medication = st.selectbox("Currently on Medication", ["No", "Yes"])

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    input_data = np.array([[
        age,
        systolic,
        diastolic,
        bs,
        temp,
        hr,
        bmi,
        1 if hypertension == "Yes" else 0,
        1 if diabetes == "Yes" else 0,
        1 if prev_comp == "Yes" else 0,
        1 if medication == "Yes" else 0
    ]])

    prediction = model.predict(input_data)
    pred_class = np.argmax(prediction, axis=1)[0]

    label_map = ["Low Risk", "Mid Risk", "High Risk"]
    icon_map = ["🟢", "🟡", "🔴"]
    label = label_map[pred_class]
    icon = icon_map[pred_class]

    st.subheader("📊 Predicted Risk Level:")
    st.markdown(f"### {icon} **{label}**")

    if pred_class == 2:
        st.error("⚠️ Risiko tinggi! Harap segera konsultasi ke dokter.")
    elif pred_class == 1:
        st.warning("Perlu perhatian dan kontrol lebih lanjut.")
    else:
        st.success("Risiko rendah. Tetap jaga kesehatan dan rutin kontrol.")
