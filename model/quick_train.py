#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

# At the top of quick_train.py
import sys
from pathlib import Path

# Add cosim folder to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'cosim'))

from run_cosim import MNISTModel
  # reuse your model definition

def train():
    print("=== TRAINING MODEL (PHASE 3 REBUILD) ===")

    model = MNISTModel()
    model.train()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = torchvision.datasets.MNIST(
        root="../data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=64, shuffle=True
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 3

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

    # Save weights
    model_dir = Path(__file__).parent
    save_path = model_dir / "model_weights.pth"
    torch.save(model.state_dict(), save_path)

    print(f"\n✓ Training complete. Saved model: {save_path}")

if __name__ == "__main__":
    train()
