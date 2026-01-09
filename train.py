import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# =============================
# 1️⃣ Cihaz seçimi (CPU / GPU)
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Kullanılan cihaz:", device)

# =============================
# 2️⃣ Görsel dönüşümleri
# =============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =============================
# 3️⃣ Dataset yükleme
# =============================
train_dataset = datasets.ImageFolder("dataset/train", transform=transform)
val_dataset   = datasets.ImageFolder("dataset/val", transform=transform)

print("Sınıf isimleri:", train_dataset.classes)

# =============================
# 4️⃣ DataLoader
# =============================
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False)

# =============================
# 5️⃣ Model (ResNet18)
# =============================
model = models.resnet18(pretrained=True)

# Son katmanı değiştiriyoruz (2 sınıf: ai / real)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# =============================
# 6️⃣ Kayıp fonksiyonu & optimizer
# =============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# =============================
# 7️⃣ Eğitim döngüsü
# =============================
epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # =============================
    # Validation
    # =============================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Loss: {avg_loss:.4f} "
          f"Val Accuracy: {accuracy:.2f}%")

# =============================
# 8️⃣ Modeli kaydet
# =============================
torch.save(model.state_dict(), "ai_image_detector.pth")
print("Model kaydedildi.")
