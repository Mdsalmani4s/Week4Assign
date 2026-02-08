"""
Week 4 Assignment: Neural Networks for Vision Tasks
Script 3: Convolutional Neural Network for CIFAR-10 Classification
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Commit: "Loaded and preprocessed CIFAR-10 dataset"
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# CIFAR-10 class names
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Display and save sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    image, label = trainset[i]
    # Denormalize for display
    img = image / 2 + 0.5
    img = np.transpose(img.numpy(), (1, 2, 0))
    ax.imshow(img)
    ax.set_title(f"{classes[label]}")
    ax.axis("off")
plt.suptitle("CIFAR-10 Sample Images", fontsize=16)
plt.tight_layout()
plt.savefig('cifar10_sample.png', dpi=150, bbox_inches='tight')
print("✓ Saved: cifar10_sample.png")



# Commit: "Implemented CNN for CIFAR-10 classification"
# Define a simple CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * 16 * 16, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc1(x)
        return x

# Initialize model, loss function, and optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
train_losses = []
print("\n" + "=" * 50)
print("Training CNN on CIFAR-10")
print("=" * 50)

for epoch in range(5):
    epoch_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/5, Loss: {avg_loss:.4f}")

# Evaluate model
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

accuracy = 100 * correct / total
print(f"\nTest Accuracy: {accuracy:.2f}%")
print("=" * 50)

# Save training loss plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), train_losses, marker='o', linewidth=2, markersize=8, color='green')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('CNN Training Loss on CIFAR-10', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cifar10_training_loss.png', dpi=150, bbox_inches='tight')
print("✓ Saved: cifar10_training_loss.png")

# Generate and save confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes, cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('CIFAR-10 Confusion Matrix (CNN)', fontsize=14)
plt.tight_layout()
plt.savefig('cifar10_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Saved: cifar10_confusion_matrix.png")

# Generate classification report
report = classification_report(all_labels, all_preds, target_names=classes)
print("\nClassification Report:")
print(report)

# Save classification report to file
with open('cifar10_classification_report.txt', 'w') as f:
    f.write("CIFAR-10 Classification Report (CNN)\n")
    f.write("=" * 50 + "\n")
    f.write(report)
print("✓ Saved: cifar10_classification_report.txt")

print("\n✅ CIFAR-10 CNN Training Complete!")