
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model_status_gizi.h5")

st.set_page_config(page_title="Prediksi Status Gizi Ibu Hamil", layout="centered")
st.title("🥗 Prediksi Status Gizi Ibu Hamil")
st.markdown("Masukkan data berikut untuk memprediksi status gizi ibu hamil.")

with st.form("prediction_form"):
    usia = st.number_input("Usia (tahun)", min_value=10, max_value=60, value=28)
    bb_dulu = st.number_input("Berat Badan Sebelum Hamil (kg)", min_value=30.0, max_value=150.0, value=55.0)
    bb_sekarang = st.number_input("Berat Badan Sekarang (kg)", min_value=30.0, max_value=150.0, value=60.0)
    tinggi_badan = st.number_input("Tinggi Badan (cm)", min_value=130.0, max_value=200.0, value=160.0)
    lingkar_lengan = st.number_input("Lingkar Lengan Atas (cm)", min_value=15.0, max_value=45.0, value=25.0)
    hb = st.number_input("Kadar Hb (g/dL)", min_value=5.0, max_value=15.0, value=11.0)
    tekanan_darah = st.number_input("Tekanan Darah (mmHg)", min_value=80, max_value=180, value=120)

    submitted = st.form_submit_button("🔍 Prediksi")

if submitted:
    tinggi_m = tinggi_badan / 100
    imt = bb_sekarang / (tinggi_m ** 2)

    input_data = np.array([[usia, bb_dulu, bb_sekarang, tinggi_badan, tekanan_darah, lingkar_lengan, hb, imt]])
    prediction = model.predict(input_data)
    pred_class = np.argmax(prediction, axis=1)[0]

    label_map = ["Gizi Kurang", "Gizi Normal", "Gizi Lebih"]
    icon_map = ["🔴", "🟢", "🟡"]
    label = label_map[pred_class]
    icon = icon_map[pred_class]

    st.subheader("📊 Hasil Prediksi Status Gizi:")
    st.markdown(f"### {icon} **{label}**")

    if pred_class == 0:
        st.error("⚠️ Gizi Kurang! Perlu intervensi gizi dan konsultasi.")
    elif pred_class == 2:
        st.warning("🍟 Perlu pengawasan karena kelebihan gizi.")
    else:
        st.success("👍 Status gizi normal. Tetap jaga pola makan dan kontrol rutin.")
