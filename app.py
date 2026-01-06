import streamlit as st 
from model import load_model
from predict import predict_image

st.set_page_config(page_title="AI Görsel Tespiti")

st.title("AI Görsel Tespit Sistemi")
st.write("Yüklediğiniz görselin yapay zeka ile üretilip üretilmediğini tahmin eder.")

model = load_model()

file = st.file_uploader("📂 Bir görsel yükleyin", type=["jpg", "png", "jpeg"])

if file:
    st.image(file, caption="Yüklenen görsel", use_column_width=True)

    real, ai = predict_image(model, file)

    st.subheader("📊 Tahmin Sonuçları")
    st.write(f"🧑 Gerçek Fotoğraf: %{real*100:.2f}")
    st.write(f"🤖 Yapay Zeka: %{ai*100:.2f}")