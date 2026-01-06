import streamlit as st 
from model import load_model
from predict import predict_image

st.set_page_config(page_title="Görselde Yapay Zeka Analizi", layout="wide") #Title

#Sidebar Start
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
""")
st.sidebar.markdown("---")

#sidebar end

#hero start
st.markdown("""
# 🧠 Yapay Zeka Görsel Analizi
Yüklediğiniz görselin gerçek mi yapay mı olduğunu analiz eder.
""")

model = load_model()

file = st.file_uploader("📂 Görsel yükleyin", type=["jpg","png","jpeg"])

if file:
    st.image(file, use_container_width=True)
    real, ai = predict_image(model, file)

    col1, col2 = st.columns(2)
    col1.metric("🧑 Gerçek", f"%{real*100:.2f}")
    col2.metric("🤖 Yapay Zeka", f"%{ai*100:.2f}")

    if ai > 0.75:
        st.error("⚠️ Büyük ihtimalle yapay zeka")
    elif ai > 0.5:
        st.warning("⚠️ Kararsız sonuç")
    else:
        st.success("✅ Büyük ihtimalle gerçek")


st.markdown("---")

#hero end

#footer start
st.markdown(
    "👨‍💻 Geliştirici: **Arda24** | AI Image Detector © 2026",
    unsafe_allow_html=True
)

#footer end