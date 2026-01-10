import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import copy

# 1. Cihaz Ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Kullanılan Cihaz: {device}")

# 2. Transformlar (Veri Çoğaltma)
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Dataset Yükleme
train_dataset = datasets.ImageFolder("dataset/train", transform=train_transform)
val_dataset = datasets.ImageFolder("dataset/val", transform=val_transform)

# --- KRİTİK KONTROL ---
print(f"📂 Sınıf Haritası: {train_dataset.class_to_idx}")
# Çıktıda {'0_real': 0, '1_fake': 1} görmelisin. 
# Eğer {'fake': 0, 'real': 1} görüyorsan klasör adlarını değiştir!

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 4. Model Kurulumu
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.to(device)

# 5. Ayarlar
criterion = nn.CrossEntropyLoss()
# Learning rate düşürüldü (daha hassas öğrenme için)
optimizer = optim.Adam(model.parameters(), lr=0.00005) 

epochs = 15 # Epoch artırıldı
best_acc = 0.0
best_model_wts = copy.deepcopy(model.state_dict())

# 6. Eğitim Döngüsü
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    print("-" * 10)

    # --- TRAIN ---
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_train += torch.sum(preds == labels.data)
        total_train += labels.size(0)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_train.double() / total_train
    print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # --- VALIDATION ---
    model.eval()
    running_corrects = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    val_acc = running_corrects.double() / len(val_dataset)
    print(f"Val Acc: {val_acc:.4f}")

    # En iyi modeli hafızada tut
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        print("⭐ Yeni en iyi model bulundu!")

# 7. En İyi Modeli Kaydet
print(f"\nEn yüksek Validation Accuracy: {best_acc:.4f}")
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "ai_image_detector.pth")
print("✅ En başarılı model 'ai_image_detector.pth' olarak kaydedildi.")