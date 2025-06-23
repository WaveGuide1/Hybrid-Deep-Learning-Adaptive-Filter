import numpy as np
import torch
from src.nlms_filter import NLMSFilter
from src.hybrid_cnn import HybridCNN

class HybridFilterSystem:
    def __init__(self, cnn_model_path, lr=0.01, sample_size=16, device='cpu'):
        self.device = torch.device(device)
        self.nlms = NLMSFilter(lr=lr, sample_size=sample_size)

        self.cnn = HybridCNN().to(self.device)
        self.cnn.load_state_dict(torch.load(cnn_model_path, map_location=self.device))
        self.cnn.eval()

    def apply_hybrid_filter(self, noisy_signal, desired_signal):
    
        # NLMS filtering
        nlms_output = self.nlms.filter(noisy_signal, desired_signal)

        # CNN refinement
        input_tensor = torch.tensor(nlms_output, dtype=torch.float32).unsqueeze(0).to(self.device)
        cnn_output = self.cnn(input_tensor).squeeze().cpu().detach().numpy()

        return cnn_output

    def batch_hybrid_filter(self, noisy_batch, desired_batch):
        """
        Apply hybrid filter to batch of segments.
        """
        results = []
        for noisy, desired in zip(noisy_batch, desired_batch):
            filtered = self.apply_hybrid_filter(noisy, desired)
            results.append(filtered)
        return np.array(results)
