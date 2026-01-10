import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# Global değişkenler (Hook verilerini tutmak için)
gradients = None
activations = None

def backward_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

def forward_hook(module, input, output):
    global activations
    activations = output

# ---------- Transform ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------- Grad-CAM Fonksiyonu (GÜNCELLENDİ) ----------
def generate_gradcam(model, image, device):
    """
    Artık 3 parametre alıyor:
    1. model: app.py'da yüklü olan model
    2. image: İşlenecek yüz görseli
    3. device: cuda veya cpu
    """
    global gradients, activations
    
    # 1. Hook'ları anlık olarak kaydet
    # ResNet18'in son katmanına kanca atıyoruz
    target_layer = model.layer4[-1]
    
    # Hook'ları ekle ve handle'ları sakla (iş bitince silmek için)
    handle_f = target_layer.register_forward_hook(forward_hook)
    handle_b = target_layer.register_full_backward_hook(backward_hook)

    try:
        # 2. Görseli hazırla
        # Eğer görsel zaten PIL değilse (örn numpy array), PIL'e çevir
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
            
        img = image.convert("RGB")
        x = transform(img).unsqueeze(0).to(device)

        # 3. Forward Pass
        output = model(x)
        pred = output.argmax(dim=1)

        # 4. Backward Pass
        model.zero_grad()
        output[0, pred].backward()

        # Gradient veya Activation yakalanamadıysa boş dön
        if gradients is None or activations is None:
            return np.array(img.resize((224, 224)))

        # 5. Grad-CAM Hesapla
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1).squeeze()
        cam = F.relu(cam)

        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))

        # Sıfıra bölünme hatası önlemi
        if cam.max() > 0:
            cam = cam / cam.max()
        else:
            cam = np.zeros_like(cam)

        # 6. Görseli Birleştir (Overlay)
        original = np.array(img.resize((224, 224)))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        
        return overlay

    except Exception as e:
        print(f"Grad-CAM hatası: {e}")
        # Hata durumunda orijinal resmi geri döndür
        return np.array(img.resize((224, 224)))

    finally:
        # 7. Hook'ları temizle (ÇOK ÖNEMLİ)
        # Bunu yapmazsak her karede bellekte yeni bir kanca birikir ve sistem yavaşlar
        handle_f.remove()
        handle_b.remove()