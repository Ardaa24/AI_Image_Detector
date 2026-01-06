import torch 
from PIL import Image 
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(model, image_file):
    image = Image.open(image_file).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)


    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)

    real_prob = probs[0][0].item()
    ai_prob = probs[0][1].item()

    return real_prob, ai_prob