"""
Week 4 Assignment: Neural Networks for Vision Tasks
Script 2: Fully Connected Neural Network for MNIST Classification
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

# Commit: "Loaded and preprocessed MNIST dataset"
# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Display and save sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    image, label = trainset[i]
    ax.imshow(image.squeeze(), cmap='gray')
    ax.set_title(f"Label: {label}")
    ax.axis("off")
plt.suptitle("MNIST Sample Images", fontsize=16)
plt.tight_layout()
plt.savefig('mnist_sample.png', dpi=150, bbox_inches='tight')
print("✓ Saved: mnist_sample.png")


# Commit: "Implemented FCNN for MNIST classification"
# Define a simple fully connected network
class FCNN(nn.Module):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer
model = FCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
train_losses = []
print("\n" + "=" * 50)
print("Training FCNN on MNIST")
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
plt.plot(range(1, 6), train_losses, marker='o', linewidth=2, markersize=8)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Training Loss', fontsize=12)
plt.title('FCNN Training Loss on MNIST', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mnist_training_loss.png', dpi=150, bbox_inches='tight')
print("✓ Saved: mnist_training_loss.png")

# Generate and save confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('MNIST Confusion Matrix (FCNN)', fontsize=14)
plt.tight_layout()
plt.savefig('mnist_confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✓ Saved: mnist_confusion_matrix.png")

# Generate classification report
report = classification_report(all_labels, all_preds, target_names=[str(i) for i in range(10)])
print("\nClassification Report:")
print(report)

# Save classification report to file
with open('mnist_classification_report.txt', 'w') as f:
    f.write("MNIST Classification Report (FCNN)\n")
    f.write("=" * 50 + "\n")
    f.write(report)
print("✓ Saved: mnist_classification_report.txt")

print("\n✅ MNIST FCNN Training Complete!")