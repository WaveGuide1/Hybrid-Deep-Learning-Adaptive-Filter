# File: src/hybrid_cnn.py

import torch
import torch.nn as nn

class HybridCNN(nn.Module):
    """Hybrid CNN model for processing 1D time series data.
    """
    def __init__(self):
        super(HybridCNN, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=16, out_channels=8, kernel_size=3, stride=1),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(8 * 89, 128),
            nn.ReLU(),
            nn.Linear(128, 187)
        )

    def forward(self, x):
        
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = HybridCNN()
    print(model)
