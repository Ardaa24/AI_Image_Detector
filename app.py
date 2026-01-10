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

    # ---------- Gerekli Ek Kütüphaneler ----------
from torchvision import transforms
import torch.nn.functional as F

# Model için görüntü işleme ayarları (Yüz tahmini için gerekli)
face_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------- Grad-CAM ve Yüz Analizi ----------
from face_utils import extract_faces
# Eğer önceki adımda gradcam fonksiyonunu güncellediysen (model, image, device) parametreleriyle çağır
# Güncellemediysen eski haliyle (sadece image) kalabilir. Aşağıdaki kod en güncel halidir:
from gradcam_utilitys import generate_gradcam

st.markdown("## 🧑‍🦱 Yüz Odaklı Deepfake Analizi")

# Yüzleri bul
faces = extract_faces(cv_image)

if not faces:
    st.warning("Görselde yüz tespit edilemedi.")
else:
    for i, face in enumerate(faces):
        st.markdown(f"### 👤 Yüz {i+1} Analizi")

        col1, col2 = st.columns(2)

        # 1. Sütun: Yüz ve Tahmin
        with col1:
            st.image(face, caption="Tespit Edilen Yüz", use_container_width=True)
            
            # --- YÜZ TAHMİNİ BAŞLANGIÇ ---
            # Yüzü modele uygun hale getir
            face_tensor = face_transform(face).unsqueeze(0).to(device)
            
            # Tahmin yap
            with torch.no_grad():
                output = model(face_tensor)
                probs = F.softmax(output, dim=1)
                
                # Sınıf 0: Gerçek, Sınıf 1: AI (Eğitim sırasına göre değişebilir, senin projende genelde böyledir)
                real_score = probs[0][0].item()
                ai_score = probs[0][1].item()

            # Sonucu Yazdır
            if ai_score > 0.5:
                st.error(f"🚨 **DEEPFAKE TESPİT EDİLDİ**\n\nOran: %{ai_score*100:.2f} Yapay Zeka")
            else:
                st.success(f"✅ **GERÇEK YÜZ**\n\nOran: %{real_score*100:.2f} Orijinal")
            # --- YÜZ TAHMİNİ BİTİŞ ---

        # 2. Sütun: Grad-CAM (Nereye Odaklandı?)
        with col2:
            # Önceki adımda gradcam kodunu güncellediysen bu satırı kullan:
            cam_face = generate_gradcam(model, face, device)
            
            # Eğer gradcam kodunu güncellemediysen eski hali: cam_face = generate_gradcam(face)
            
            st.image(cam_face, caption="Modelin Odaklandığı Bölge (Isı Haritası)", use_container_width=True)
            st.info("Kırmızı alanlar, modelin 'sahte' veya 'gerçek' kararı verirken en çok şüphelendiği bölgelerdir.")

        st.divider() # Araya çizgi çek

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    "👨‍💻 Geliştirici: **[Arda24](https://github.com/ardaa24)** | AI Image Detector © 2026",
    unsafe_allow_html=True
)