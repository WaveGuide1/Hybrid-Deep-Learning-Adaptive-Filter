import os
import sys

# Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.hybrid_cnn import HybridCNN

"""
Configurations
"""
processed_dir = "data/processed_data"
noisy_data_dir = os.path.join(processed_dir, "noisy")
save_model_path = "experiments/cnn_model.pth"
snr_for_training = 5

batch_size = 64
epochs = 50
learning_rate = 0.001

"""
# Data Preparation
"""
train_clean = np.load(os.path.join(processed_dir, "train_normalized.npy"))
train_noisy = np.load(os.path.join(noisy_data_dir, f"train_noisy_{snr_for_training}dB.npy"))

"""
Convert to torch tensors
"""
train_clean_tensor = torch.tensor(train_clean, dtype=torch.float32)
train_noisy_tensor = torch.tensor(train_noisy, dtype=torch.float32)

"""
Create DataLoader
"""
train_dataset = TensorDataset(train_noisy_tensor, train_clean_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

"""
Model, Loss, Optimizer
"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

"""
Training Loop
"""
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for noisy_batch, clean_batch in train_loader:
        noisy_batch = noisy_batch.to(device)
        clean_batch = clean_batch.to(device)

        optimizer.zero_grad()
        outputs = model(noisy_batch)
        loss = criterion(outputs, clean_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")



# Save trained model
os.makedirs("experiments", exist_ok=True)
torch.save(model.state_dict(), save_model_path)
print(f"Model saved to {save_model_path}")
