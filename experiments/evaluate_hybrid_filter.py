import os
import sys

# Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from src.hybrid_filter import HybridFilterSystem

# Load noisy and clean test data
noisy_batch = np.load("data/processed_data/noisy/test_noisy_5dB.npy")
clean_batch = np.load("data/processed_data/test_normalized.npy")

# Initialize system
system = HybridFilterSystem(cnn_model_path="experiments/cnn_model.pth", device="cpu")

# Run hybrid filtering
filtered_batch = system.batch_hybrid_filter(noisy_batch, clean_batch)

