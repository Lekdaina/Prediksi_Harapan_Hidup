import streamlit as st
import numpy as np
import joblib
import os

st.set_page_config(page_title="Prediksi Harapan Hidup", page_icon="🧬", layout="centered")

# === DARK THEME CSS ===
st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #90caf9;
        text-align: center;
    }
    .stButton > button {
        background-color: #1565c0;
        color: white;
        border: none;
    }
    .stSidebar {
        background-color: #1e1e1e;
    }
    </style>
""", unsafe_allow_html=True)

# === Judul Aplikasi ===
st.markdown("<h1>🔮 Prediksi Harapan Hidup</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#bbbbbb;'>Berdasarkan indikator kesehatan WHO</p><hr>", unsafe_allow_html=True)

# === Sidebar Input ===
st.sidebar.header("🩺 Masukkan Data Anda")

age = st.sidebar.slider("Usia", 18, 100, 30)
gender = st.sidebar.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
bmi = st.sidebar.slider("BMI", 10.0, 50.0, 22.0)
bp = st.sidebar.slider("Tekanan Darah (mmHg)", 90, 200, 120)
cholesterol = st.sidebar.slider("Kolesterol (mg/dL)", 100, 350, 180)
diabetes = st.sidebar.selectbox("Diabetes?", ["Ya", "Tidak"])
hypertension = st.sidebar.selectbox("Hipertensi?", ["Ya", "Tidak"])
smoker = st.sidebar.selectbox("Perokok?", ["Ya", "Tidak"])
passive_smoker = st.sidebar.selectbox("Terpapar Asap Rokok?", ["Ya", "Tidak"])
alcohol = st.sidebar.selectbox("Konsumsi Alkohol?", ["Ya", "Tidak"])
exercise = st.sidebar.slider("Olahraga per Minggu", 0, 7, 3)
family_history = st.sidebar.selectbox("Riwayat Penyakit Keluarga?", ["Ya", "Tidak"])
income = st.sidebar.slider("Pendapatan per Tahun (USD)", 500, 100000, 20000)
education = st.sidebar.slider("Tingkat Pendidikan (Tahun)", 0, 20, 12)

def to_binary(val): return 1 if val == "Ya" else 0
gender_val = 1 if gender == "Perempuan" else 0

input_data = np.array([[
    age, gender_val, bmi, bp, cholesterol,
    to_binary(diabetes), to_binary(hypertension),
    to_binary(smoker), to_binary(passive_smoker),
    to_binary(alcohol), exercise, to_binary(family_history),
    income, education
]])

# === Load Model ===
model_path = "model/life_model.pkl"
if not os.path.exists(model_path):
    st.error("❌ Model belum ditemukan. Jalankan train_model.py terlebih dahulu.")
else:
    model = joblib.load(model_path)

    # === Disclaimer Checkbox ===
    st.markdown("### 📜 Disclaimer")
    agree = st.checkbox("Saya memahami bahwa aplikasi ini hanya untuk simulasi dan edukasi, bukan pengganti konsultasi medis.")

    if not agree:
        st.info("✅ Silakan centang kotak di atas untuk melanjutkan prediksi.")
    else:
        if st.button("🔍 Prediksi Sekarang"):
            prediction = model.predict(input_data)[0]
            st.success(f"🎯 Estimasi Harapan Hidup Anda: **{prediction:.1f} tahun**")

            # Saran Otomatis
            st.subheader("📋 Saran Kesehatan")
            if bmi < 18.5:
                st.warning("🍽️ Berat badan terlalu rendah.")
            elif bmi > 30:
                st.warning("⚖️ Berat badan berlebih/obesitas.")
            if cholesterol > 240:
                st.warning("🥩 Kolesterol sangat tinggi.")
            if bp > 140:
                st.warning("🩸 Tekanan darah tinggi.")
            if exercise < 2:
                st.info("🤸 Tambahkan aktivitas fisik.")
            if to_binary(smoker):
                st.warning("🚬 Merokok memperpendek usia.")
            if to_binary(passive_smoker):
                st.info("😷 Hindari asap rokok.")

# === Footer Disclaimer ===
st.markdown("""<hr>""", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #888888; font-size: 0.85em;'>
⚠️ <strong>Disclaimer:</strong> Aplikasi ini bersifat simulatif dan tidak menggantikan diagnosis profesional.
</div>
""", unsafe_allow_html=True)