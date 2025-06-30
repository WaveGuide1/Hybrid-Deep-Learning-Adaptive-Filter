import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.hybrid_cnn import HybridCNN

"""
Configurations
"""
processed_dir = "data/processed_data"
noisy_data_dir = os.path.join(processed_dir, "noisy")
save_model_path = "experiments/cnn_model.pth"
snr_for_training = 5

batch_size = 512
epochs = 50
learning_rate = 0.1

def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

"""
Data Preparation
"""
log("Loading data...")
sample_count = 20000
clean_data = np.load(os.path.join(processed_dir, "train_normalized.npy"))[:sample_count, :187]
noisy_data = np.load(os.path.join(noisy_data_dir, f"train_noisy_{snr_for_training}dB.npy"))[:sample_count, :187]

"""Vectorized normalization"""
log("Normalizing data...")
min_vals = np.min(noisy_data, axis=1, keepdims=True)
max_vals = np.max(noisy_data, axis=1, keepdims=True)
ranges = max_vals - min_vals
ranges[ranges == 0] = 1
train_noisy = 2 * (noisy_data - min_vals) / ranges - 1

min_vals_c = np.min(clean_data, axis=1, keepdims=True)
max_vals_c = np.max(clean_data, axis=1, keepdims=True)
ranges_c = max_vals_c - min_vals_c
ranges_c[ranges_c == 0] = 1
train_clean = 2 * (clean_data - min_vals_c) / ranges_c - 1

"""
Create tensors and DataLoader
"""
log("Creating DataLoader...")
train_clean_tensor = torch.tensor(train_clean, dtype=torch.float32)
train_noisy_tensor = torch.tensor(train_noisy, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(train_noisy_tensor, train_clean_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

"""Model setup"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f"Using device: {device}")
model = HybridCNN().to(device)
criterion = nn.MSELoss()

"""Use SGD with momentum for faster convergence"""
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

"""Training Loop"""
log("Starting training...")
start_time = time.time()

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    
    for noisy_batch, clean_batch in train_loader:
        noisy_batch = noisy_batch.to(device)
        clean_batch = clean_batch.to(device)

        optimizer.zero_grad()
        outputs = model(noisy_batch)
        loss = criterion(outputs, clean_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    """Calculate average epoch loss"""
    epoch_loss /= len(train_loader)
    log(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.6f}")

"""Save model"""
os.makedirs("experiments", exist_ok=True)
torch.save(model.state_dict(), save_model_path)
training_time = (time.time() - start_time) / 60
log(f"Training completed in {training_time:.2f} minutes")
log(f"Final model saved to {save_model_path}")