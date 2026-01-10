import torch
from torchvision.models import resnet18

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)

    state = torch.load(
        "ai_image_detector.pth",
        map_location=device
    )
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model
