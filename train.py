from torchvision.datasets import ImageFolder

dataset = ImageFolder("dataset/train")

print(dataset.classes)
print(dataset[0])
