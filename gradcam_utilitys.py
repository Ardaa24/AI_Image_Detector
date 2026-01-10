import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------- Model --------
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2)

checkpoint = torch.load("ai_image_detector.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

model.to(device)
model.eval()

# -------- Target Layer --------
target_layer = model.layer4[-1]

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

# -------- Transform --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# -------- Main Function --------
def predict_with_gradcam(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(device)

    output = model(img_tensor)
    probs = torch.softmax(output, dim=1)
    pred = probs.argmax(dim=1).item()
    confidence = probs[0, pred].item()

    model.zero_grad()
    output[0, pred].backward()

    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * activations).sum(dim=1).squeeze()
    cam = F.relu(cam)

    cam = cam.detach().cpu().numpy()
    cam = cv2.resize(cam, (224, 224))
    cam = cam / (cam.max() + 1e-8)

    image_np = np.array(image.resize((224, 224)))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)

    label = "AI Generated" if pred == 0 else "Real Photo"
    return label, confidence, overlay
