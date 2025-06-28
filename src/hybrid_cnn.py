import torch
import torch.nn as nn
import torch.nn.init as init

class HybridCNN(nn.Module):
    """
    Simple and Fast 1D CNN for ECG Denoising
    """
    def __init__(self):
        super(HybridCNN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.Conv1d(32, 1, kernel_size=1)
        )
        
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
        
    def forward(self, x):
        return self.net(x).squeeze(1)