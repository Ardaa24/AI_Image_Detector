import torch
from torchvision.models import resnet18

def load_model():
    model = resnet18(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.eval()
    return model