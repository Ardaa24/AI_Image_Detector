import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Model ----------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# ⬇⬇⬇ KRİTİK DÜZELTME
model.load_state_dict(torch.load("ai_image_detector.pth", map_location=device))

model.to(device)
model.eval()

# ---------- Grad-CAM Layer ----------
target_layer = model.layer4[-1]

gradients = None
activations = None

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def forward_hook(module, input, output):
    global activations
    activations = output

target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# ---------- Transform ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------- Grad-CAM Fonksiyonu ----------
def generate_gradcam(image: Image.Image):
    img = image.convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    output = model(x)
    pred = output.argmax(dim=1)

    model.zero_grad()
    output[0, pred].backward()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1).squeeze()
    cam = F.relu(cam)

    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = cam / cam.max()

    original = np.array(img.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    return overlay
