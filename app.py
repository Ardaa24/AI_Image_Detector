import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

# Kendi modüllerimiz
from model import load_model
from predict import predict_image
from face_utils import extract_faces
from gradcam_utilitys import generate_gradcam

# ----------------- Page Config -----------------
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

# ----------------- Model Yükleme -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
@st.cache_resource # Modeli her seferinde tekrar yüklememek için cache
def get_model():
    model = load_model()
    model.to(device)
    model.eval()
    return model

model = get_model()

# ----------------- Yüz Analizi İçin Ayarlar -----------------
face_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ----------------- Upload & İşlemler -----------------
file = st.file_uploader("📂 Görsel yükleyin", type=["jpg", "png", "jpeg"])

if file:
    # 1. Görseli Yükle (Standart isim: 'image')
    image = Image.open(file).convert("RGB")

    col_img, col_info = st.columns([2, 1])

    with col_img:
        st.image(image, caption="Yüklenen Görsel", use_container_width=True)

    # 2. Genel Tahmin (Tüm Resim)
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

    st.divider()

    # 3. Yüz Odaklı Analiz
    st.markdown("## 🧑‍🦱 Yüz Odaklı Deepfake Analizi")

    # Değişken ismi artık uyumlu: 'image' gönderiyoruz
    faces = extract_faces(image)

    if not faces:
        st.warning("Görselde yüz tespit edilemedi.")
    else:
        for i, face in enumerate(faces):
            st.markdown(f"### 👤 Yüz {i+1} Analizi")

            col1, col2 = st.columns(2)

            # --- Sol Sütun: Yüz ve Tahmin ---
            with col1:
                st.image(face, caption="Tespit Edilen Yüz", use_container_width=True)
                
                try:
                    # Yüzü tensor formatına çevir
                    face_tensor = face_transform(face).unsqueeze(0).to(device)
                    
                    # Tahmin yap
                    with torch.no_grad():
                        output = model(face_tensor)
                        probs = F.softmax(output, dim=1)
                        real_score = probs[0][0].item()
                        ai_score = probs[0][1].item()

                    # Sonucu Yazdır
                    if ai_score > 0.5:
                        st.error(f"🚨 **DEEPFAKE TESPİT EDİLDİ.**\n\nOran: %{ai_score*100:.2f} Yapay Zeka")
                    else:
                        st.success(f"✅ **DEEPFAKE TESPİT EDİLMEDİ.**\n\nOran: %{real_score*100:.2f} Orijinal")
                
                except Exception as e:
                    st.error(f"Tahmin hatası: {e}")

            # --- Sağ Sütun: Grad-CAM ---
            with col2:
                # Grad-CAM oluştur
                cam_face = generate_gradcam(model, face, device)
                
                st.image(cam_face, caption="Modelin Odaklandığı Bölge", use_container_width=True)
                st.info("Kırmızı alanlar, modelin kararı verirken en çok dikkat ettiği bölgelerdir.")

            st.divider()

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "👨‍💻 Geliştirici: **[Arda24](https://github.com/ardaa24)** | AI Image Detector © 2026",
    unsafe_allow_html=True
)