import streamlit as st
from model import load_model
from predict import predict_image
from gradcam_utilitys import generate_gradcam
from face_utils import extract_faces

from PIL import Image
import numpy as np
import cv2
import torch

# ----------------- Page -----------------
st.set_page_config(
    page_title="Görselde Yapay Zeka Analizi",
    layout="wide"
)

# ----------------- Sidebar -----------------
st.sidebar.title("ℹ️ Proje Hakkında")
st.sidebar.write("""
Bu sistem, yüklenen görsellerin  
yapay zeka ile üretilip üretilmediğini  
derin öğrenme kullanarak tahmin eder.
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Kullanılan Teknolojiler")
st.sidebar.write("""
- Python  
- PyTorch  
- ResNet18  
- Streamlit  
- Grad-CAM (XAI)
""")

# ----------------- Hero -----------------
st.markdown("""
# 🧠 Yapay Zeka Görsel Analizi
Yüklediğiniz görselin **gerçek mi yapay mı** olduğunu analiz eder  
ve modelin **nereye baktığını** gösterir.
""")

# ----------------- Model -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model().to(device)
model.eval()

# ----------------- Upload -----------------
file = st.file_uploader("📂 Görsel yükleyin", type=["jpg", "png", "jpeg"])

if file is not None:
    # 1️⃣ PIL (Streamlit)
    pil_image = Image.open(file).convert("RGB")

    # 2️⃣ OpenCV (face detection)
    cv_image = np.array(pil_image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    col_img, col_info = st.columns([2, 1])

    with col_img:
        st.image(pil_image, caption="Yüklenen Görsel", use_container_width=True)

    # ---------- Prediction ----------
    real, ai = predict_image(model, file)

    with col_info:
        st.metric("🧑 Gerçek", f"%{real*100:.2f}")
        st.metric("🤖 Yapay Zeka", f"%{ai*100:.2f}")

        if ai > 0.75:
            st.error("⚠️ Büyük ihtimalle yapay zeka")
        elif ai > 0.5:
            st.warning("⚠️ Kararsız sonuç")
        else:
            st.success("✅ Büyük ihtimalle gerçek")

    st.markdown("---")

    # ---------- Face-based Grad-CAM ----------
    st.markdown("## 🧑‍🦱 Yüz Odaklı Deepfake Analizi")

    faces = extract_faces(cv_image)

    if len(faces) == 0:
        st.warning("Yüz tespit edilemedi.")
    else:
        for i, face in enumerate(faces):
            st.markdown(f"### Yüz {i+1}")

            col1, col2 = st.columns(2)

            with col1:
                st.image(face, caption="Tespit Edilen Yüz", use_container_width=True)

            with col2:
                cam = generate_gradcam(model, face, device)
                st.image(cam, caption="Grad-CAM (Modelin Baktığı Yer)", use_container_width=True)


# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "👨‍💻 Geliştirici: **[Arda24](https://github.com/ardaa24)** | AI Image Detector © 2026",
    unsafe_allow_html=True
)
