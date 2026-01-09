import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import sys
import torch.nn as nn

# ---------- Device ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Model ----------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

checkpoint = torch.load("ai_image_detector.pth", map_location=device)

model.to(device)
model.eval()

# ---------- Target Layer ----------
target_layer = model.layer4[-1]

# ---------- Hooks ----------
gradients = None
activations = None

def forward_hook(module, input, output):
    global activations
    activations = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)

# ---------- Image ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device)

# ---------- Forward ----------
output = model(input_tensor)
pred_class = output.argmax(dim=1)

# ---------- Backward ----------
model.zero_grad()
output[0, pred_class].backward()

# ---------- Grad-CAM ----------
weights = gradients.mean(dim=(2, 3), keepdim=True)
cam = (weights * activations).sum(dim=1).squeeze()
cam = F.relu(cam)

cam = cam.detach().cpu().numpy()
cam = cv2.resize(cam, (224, 224))
cam = cam / (cam.max() + 1e-8)

# ---------- Overlay ----------
original = np.array(image.resize((224, 224)))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

cv2.imwrite("gradcam_output.jpg", overlay)
print("Grad-CAM oluşturuldu → gradcam_output.jpg")
