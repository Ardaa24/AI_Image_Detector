import cv2
import numpy as np
from PIL import Image
import os

# --- GÜVENLİ MODEL YÜKLEME ---
def load_face_cascade():
    # Denenecek potansiyel yollar
    paths_to_try = [
        # 1. Öncelik: OpenCV'nin kendi içindeki hazır modeli (En sağlıklısı)
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"),
        # 2. Öncelik: Proje klasöründeki har.xml
        "har.xml",
        # 3. Öncelik: Proje klasöründeki orijinal isimli dosya
        "haarcascade_frontalface_default.xml"
    ]

    # Boş bir classifier oluşturuyoruz (Hata vermemesi için)
    cascade = cv2.CascadeClassifier()

    for path in paths_to_try:
        # Dosya var mı kontrol et
        if os.path.exists(path):
            # Yüklemeyi dene (load fonksiyonu True/False döner, çökmez)
            if cascade.load(path):
                print(f"✅ Model başarıyla yüklendi: {path}")
                return cascade
    
    print("⚠️ HATA: Yüz tanıma modeli (XML) bulunamadı veya yüklenemedi!")
    return None

# Modeli başlat
face_cascade = load_face_cascade()

def extract_faces(image):
    faces = []
    
    # Eğer model yüklenemediyse işlemi iptal et ama programı çökertme
    if face_cascade is None or face_cascade.empty():
        return faces

    # 1. PIL görselini Numpy array'e çevir
    image_np = np.array(image)

    # 2. Gri tona çevir (Hata korumalı)
    try:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    except:
        gray = image_np

    # 3. Yüzleri tespit et
    try:
        detections = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(64, 64)
        )
    except Exception as e:
        print(f"Tespit sırasında hata: {e}")
        return []

    # 4. Yüzleri kesip listeye ekle
    for (x, y, w, h) in detections:
        face_np = image_np[y:y+h, x:x+w]
        
        if face_np.size > 0:
            # Grad-CAM için tekrar PIL formatına çevir
            face_pil = Image.fromarray(face_np)
            faces.append(face_pil)

    return faces