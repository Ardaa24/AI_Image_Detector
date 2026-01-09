import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import sys


#  Cihaz seçimi

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transform 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


#  Modeli yükle

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("ai_image_detector.pth", map_location=device))
model = model.to(device)
model.eval()


#  Görsel yükleme

if len(sys.argv) < 2:
    print("Kullanım: python predict.py <gorsel_yolu>")
    sys.exit()

image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)


#  Tahmin

with torch.no_grad():
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)
    confidence, predicted = torch.max(probs, 1)

class_names = ["AI Generated", "Real Photo"]

print(f"Tahmin: {class_names[predicted.item()]}")
print(f"Güven Oranı: %{confidence.item() * 100:.2f}")
