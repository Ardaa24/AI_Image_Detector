import streamlit as st
from model import load_model
from predict import predict_image
from gradcam_utilitys import generate_gradcam
from PIL import Image
import torch

st.set_page_config(
    page_title="Görselde Yapay Zeka Analizi",
    layout="wide"
)

# ---------- Sidebar ----------
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

st.sidebar.markdown("---")

# ---------- Hero ----------
st.markdown("""
# 🧠 Yapay Zeka Görsel Analizi
Yüklediğiniz görselin **gerçek mi yapay mı** olduğunu analiz eder  
ve modelin **nereye baktığını** gösterir.
""")

# ---------- Model ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model().to(device)
st.write("MODEL YÜKLENMEDİ - TEST")
model.eval()

# ---------- Upload ----------
file = st.file_uploader("📂 Görsel yükleyin", type=["jpg","png","jpeg"])

if file:
    image = Image.open(file).convert("RGB")

    col_img, col_info = st.columns([2,1])

    with col_img:
        st.image(image, caption="Yüklenen Görsel", use_container_width=True)

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

    # ---------- Grad-CAM ----------
    st.subheader("🔥 Model Nereye Baktı? (Grad-CAM)")

    with st.spinner("Grad-CAM oluşturuluyor..."):
        cam_image = generate_gradcam(model, image, device)

    st.image(
        cam_image,
        caption="Kırmızı alanlar modelin karar verirken en çok odaklandığı bölgeler",
        use_container_width=True
    )

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "👨‍💻 Geliştirici: **Arda24** | AI Image Detector © 2026",
    unsafe_allow_html=True
)
