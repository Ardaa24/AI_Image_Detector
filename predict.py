import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import sys

# ---------- Transform ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------- Streamlit için ----------
def predict_image(model, image_file):
    image = Image.open(image_file).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)[0]

    real = probs[1].item()
    ai = probs[0].item()

    return real, ai


# ---------- CLI için ----------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Kullanım: python predict.py <gorsel_yolu>")
        exit()

    from model import load_model

    model = load_model()
    image_path = sys.argv[1]

    real, ai = predict_image(model, image_path)

    if ai > real:
        print("Tahmin: AI Generated")
        print(f"Güven Oranı: %{ai*100:.2f}")
    else:
        print("Tahmin: Real Photo")
        print(f"Güven Oranı: %{real*100:.2f}")
