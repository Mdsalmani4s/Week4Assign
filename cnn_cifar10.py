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