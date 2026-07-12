import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import random

# same dataset folder as before
data_dir = r"C:\Users\sijip\.cache\kagglehub\datasets\emmarex\plantdisease\versions\1\PlantVillage"

# basic transforms - resizing everything to the same size and turning into tensors
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# loading the full dataset
full_dataset = datasets.ImageFolder(data_dir, transform=transform)
print("total classes:", len(full_dataset.classes))
print("classes:", full_dataset.classes)

# my laptop doesn't have a GPU, so using the full 54k images would take forever
# just grabbing a random subset for now to get a real baseline number
random.seed(42)
subset_size = 2000
indices = random.sample(range(len(full_dataset)), subset_size)
subset = Subset(full_dataset, indices)

# splitting into train/test (80/20)
train_size = int(0.8 * len(subset))
test_size = len(subset) - train_size
train_data, test_data = torch.utils.data.random_split(subset, [train_size, test_size])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False)

print("training on", len(train_data), "images, testing on", len(test_data))

# using a pretrained resnet18 (lighter than resnet50, easier to train on CPU)
# and fine-tuning just the last layer, like the pitch mentioned
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_classes = len(full_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# just doing a few epochs since this is a baseline, not the final model
epochs = 3
for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    print(f"epoch {epoch+1}/{epochs} - loss: {running_loss:.2f} - train acc: {train_acc:.2f}%")

# now checking on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"baseline test accuracy: {test_acc:.2f}%")