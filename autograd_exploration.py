"""
Week 4 Assignment: Neural Networks for Vision Tasks
Script 1: Exploring PyTorch Autograd Mechanism
"""

import torch

# Commit: "Explored PyTorch autograd mechanism"
# Define tensors with requires_grad=True to track computation
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Define an operation: z = x^2 + y^3
z = x ** 2 + y ** 3

# Compute gradients automatically
z.backward()

# Print gradients
print("=" * 50)
print("PyTorch Autograd Exploration")
print("=" * 50)
print(f"Function: z = x^2 + y^3")
print(f"x = {x.item()}, y = {y.item()}")
print(f"z = {z.item()}")
print(f"\nGradient of x (dz/dx = 2x): {x.grad.item()}")
print(f"Gradient of y (dz/dy = 3y^2): {y.grad.item()}")
print("=" * 50)