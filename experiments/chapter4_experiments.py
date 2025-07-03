import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlms_filter import NLMSFilter
from src.hybrid_cnn import HybridCNN
from evaluation_metrics import calculate_snr, calculate_mse, estimate_cnn_flops

import torch

def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

"""
Configurations
"""
processed_dir = "data/processed_data"
noisy_data_dir = os.path.join(processed_dir, "noisy")
results_dir = "results"
figures_dir = os.path.join(results_dir, "figures")
tables_dir = os.path.join(results_dir, "tables")

os.makedirs(figures_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)

standalone_cnn_model_path = "experiments/cnn_model.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
log(f"Using device: {device}")
snr_levels = [0, 5, 10, 15, 20]

"""
Load Clean Test Dataset
"""
log("Loading test data...")
clean_test = np.load(os.path.join(processed_dir, "test_normalized.npy"))[:, :187]

"""Load Model"""
log("Loading CNN model...")
cnn_model = HybridCNN().to(device)
cnn_model.load_state_dict(torch.load(standalone_cnn_model_path, map_location=device))
cnn_model.eval()

"""
Calculate CNN FLOPs
"""
log("Calculating CNN FLOPs...")
cnn_flops = estimate_cnn_flops(cnn_model, input_shape=(1, 1, 187))
log(f"CNN FLOPs: {cnn_flops}")

"""
Initialize NLMS filter
"""
nlms = NLMSFilter(lr=0.01, sample_size=16)

"""
Experiment
"""
log("Starting experiments...")
results = []
time_metrics = []

for snr in snr_levels:
    log(f"Processing SNR: {snr}dB")
    noisy_path = os.path.join(noisy_data_dir, f"test_noisy_{snr}dB.npy")
    noisy_test = np.load(noisy_path)[:, :187]
    
    """
    NLMS Processing
    """
    log("  Running NLMS...")
    nlms_outputs = []
    nlms_times = []
    
    start_time = time.time()
    for noisy, clean in zip(noisy_test, clean_test):
        """Time individual NLMS processing"""
        sample_start = time.time()
        output = nlms.filter(noisy, clean)[:187]
        nlms_times.append(time.time() - sample_start)
        nlms_outputs.append(output)
    
    nlms_total_time = time.time() - start_time
    nlms_outputs = np.array(nlms_outputs)
    
    """
    CNN Processing
    """
    log("  Running CNN...")
    cnn_outputs = []
    cnn_times = [] 
    
    start_time = time.time()
    for noisy_signal in noisy_test:
        input_tensor = torch.tensor(noisy_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        
        """Time individual CNN processing """
        sample_start = time.time()
        output = cnn_model(input_tensor).squeeze().cpu().detach().numpy()
        if device == 'cuda':
            torch.cuda.synchronize()
        cnn_times.append(time.time() - sample_start)
        cnn_outputs.append(output)
    
    cnn_total_time = time.time() - start_time
    cnn_outputs = np.array(cnn_outputs)
    

    """
    Calculate Metrics
    """
    nlms_snr = calculate_snr(clean_test, nlms_outputs)
    nlms_mse = calculate_mse(clean_test, nlms_outputs)
    cnn_snr = calculate_snr(clean_test, cnn_outputs)
    cnn_mse = calculate_mse(clean_test, cnn_outputs)
    
    log(f"  NLMS: SNR={nlms_snr:.2f}dB, MSE={nlms_mse:.6f}")
    log(f"  CNN: SNR={cnn_snr:.2f}dB, MSE={cnn_mse:.6f}")
    

    """
    Store Results
    """
    results.append({
        'SNR (Input)': snr,
        'SNR (NLMS)': nlms_snr,
        'SNR (CNN)': cnn_snr,
        'MSE (NLMS)': nlms_mse,
        'MSE (CNN)': cnn_mse
    })
    
    time_metrics.append({
        'SNR (Input)': snr,
        'NLMS Avg Time (ms)': np.mean(nlms_times) * 1000,
        'CNN Avg Time (ms)': np.mean(cnn_times) * 1000,
        'NLMS Total Time (s)': nlms_total_time,
        'CNN Total Time (s)': cnn_total_time,
        'Speedup Factor': nlms_total_time / cnn_total_time
    })

    """Save results"""
log("Saving results...")
results_df = pd.DataFrame(results)
time_df = pd.DataFrame(time_metrics)

results_df.to_csv(os.path.join(tables_dir, "performance.csv"), index=False)
time_df.to_csv(os.path.join(tables_dir, "timing_metrics.csv"), index=False)

"""Save FLOPs separately"""
with open(os.path.join(tables_dir, "flops.txt"), "w") as f:
    f.write(f"CNN FLOPs: {cnn_flops}\n")
    f.write(f"Device: {device}\n")

"""
Create plots
"""
plt.figure(figsize=(10, 6))
plt.plot(results_df['SNR (Input)'], results_df['SNR (NLMS)'], 'b-o', label="NLMS")
plt.plot(results_df['SNR (Input)'], results_df['SNR (CNN)'], 'r-s', label="CNN")
plt.xlabel("Input SNR (dB)")
plt.ylabel("Output SNR (dB)")
plt.title("SNR Improvement")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(figures_dir, "snr_comparison.png"))

plt.figure(figsize=(10, 6))
plt.semilogy(results_df['SNR (Input)'], results_df['MSE (NLMS)'], 'b-o', label="NLMS")
plt.semilogy(results_df['SNR (Input)'], results_df['MSE (CNN)'], 'r-s', label="CNN")
plt.xlabel("Input SNR (dB)")
plt.ylabel("MSE (log scale)")
plt.title("MSE Comparison")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(figures_dir, "mse_comparison.png"))

"""
Create timing comparison plot
"""
plt.figure(figsize=(10, 6))
plt.plot(time_df['SNR (Input)'], time_df['NLMS Avg Time (ms)'], 'b-o', label="NLMS")
plt.plot(time_df['SNR (Input)'], time_df['CNN Avg Time (ms)'], 'r-s', label="CNN")
plt.xlabel("Input SNR (dB)")
plt.ylabel("Processing Time per Sample (ms)")
plt.title("Inference Time Comparison")
plt.grid(True)
plt.legend()
plt.savefig(os.path.join(figures_dir, "time_comparison.png"))

log("Experiments completed!")
log(f"CNN FLOPs: {cnn_flops} | Device: {device}")