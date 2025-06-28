import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.signal import find_peaks
import time
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.nlms_filter import NLMSFilter
from src.hybrid_cnn import HybridCNN
from evaluation_metrics import estimate_cnn_flops

"""
Configurations
"""
processed_dir = "data/processed_data"
noisy_data_dir = os.path.join(processed_dir, "noisy")
results_dir = "results/figures"
os.makedirs(results_dir, exist_ok=True)

cnn_model_path = "experiments/cnn_model.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

"""Initialize CNN model for FLOPs calculation"""
cnn_model = HybridCNN().to(device)
if os.path.exists(cnn_model_path):
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=device))

"""FLOPs estimation"""
input_shape = (1, 1, 187)
flops = estimate_cnn_flops(cnn_model, input_shape)
print(f"Estimated CNN FLOPs: {flops}")

"""SNR levels to visualize"""
snr_levels = [0, 5, 10, 15, 20]

"""
Load test data
"""
clean_test = np.load(os.path.join(processed_dir, "test_normalized.npy"))
noisy_test = {}
for snr in snr_levels:
    noisy_path = os.path.join(noisy_data_dir, f"test_noisy_{snr}dB.npy")
    if os.path.exists(noisy_path):
        noisy_test[snr] = np.load(noisy_path)

"""R-peak detection"""
def detect_r_peaks(signal, fs=360):
    """Handle inverted signals"""
    if np.mean(signal) < 0:
        signal = -signal
    
    """Adaptive thresholds based on signal range"""
    max_val = np.max(signal)
    min_val = np.min(signal)
    signal_range = max_val - min_val
    
    """Skip if signal is flat"""
    if signal_range < 0.1:
        return np.array([])
    
    """Set thresholds relative to signal range"""
    height_threshold = min_val + 0.7 * signal_range
    prominence_threshold = 0.3 * signal_range
    
    """Find peaks with prominence requirement"""
    peaks, _ = find_peaks(signal, 
                          height=height_threshold,
                          distance=fs//10,
                          prominence=prominence_threshold)
    return peaks

"""Simplified inference time measurement"""
def measure_cnn_inference(model, signal, device):
    """Measure inference time for a single signal"""
    model.eval()
    
    with torch.no_grad():
        """CUDA is initialized"""
        if device == 'cuda':
            warmup = torch.randn(1, 1, len(signal)).to(device)
            _ = model(warmup)
            torch.cuda.synchronize()
        
        """Measure actual inference"""
        start_time = time.perf_counter()
        input_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        _ = model(input_tensor)
        if device == 'cuda':
            torch.cuda.synchronize()
        end_time = time.perf_counter()
    
    return (end_time - start_time) * 1000

"""
Visualization for each SNR
"""
for snr in snr_levels:
    if snr not in noisy_test:
        print(f"Skipping SNR {snr}dB - data not found")
        continue
        
    print(f"\nProcessing SNR: {snr}dB")
    
    """Pick a sample with clear R-peaks"""
    valid_sample_found = False
    for attempt in range(10):
        idx = random.randint(0, len(clean_test)-1)
        clean_signal = clean_test[idx]
        noisy_signal = noisy_test[snr][idx]
        
        """Find R-peaks in clean signal"""
        r_peaks = detect_r_peaks(clean_signal)
        
        if len(r_peaks) >= 1:
            valid_sample_found = True
            break
    
    if not valid_sample_found:
        print(f"Warning: Couldn't find sample with clear R-peaks at {snr}dB")
        """Use sample with highest amplitude as fallback"""
        idx = np.argmax(np.max(clean_test, axis=1) - np.min(clean_test, axis=1))
        clean_signal = clean_test[idx]
        noisy_signal = noisy_test[snr][idx]
        r_peaks = detect_r_peaks(clean_signal)

    print(f"Using sample {idx} with {len(r_peaks)} R-peaks")
    

    """
    NLMS processing
    """
    nlms = NLMSFilter(lr=0.01, sample_size=16)
    start_time = time.perf_counter()
    nlms_output = nlms.filter(noisy_signal, noisy_signal)
    nlms_time = (time.perf_counter() - start_time) * 1000
    

    """
    CNN processing
    """
    cnn = HybridCNN().to(device)
    cnn.load_state_dict(torch.load(cnn_model_path, map_location=device))
    cnn.eval()
    
    """Measure CNN inference time"""
    cnn_time = measure_cnn_inference(cnn, noisy_signal, device)
    
    """Get CNN output"""
    noisy_tensor = torch.tensor(noisy_signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    cnn_output = cnn(noisy_tensor).squeeze().detach().cpu().numpy()
    

    """Create plots"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    """Signal comparison"""
    ax1.plot(clean_signal, label='Clean Signal', linewidth=2, color='blue')
    ax1.plot(noisy_signal, label=f'Noisy Input ({snr}dB)', alpha=0.4, color='red')
    ax1.plot(nlms_output, label=f'NLMS Output ({nlms_time:.2f}ms)', color='green')
    ax1.plot(cnn_output, label=f'CNN Output ({cnn_time:.2f}ms)', color='purple')
    
    """Mark R-peaks"""
    if len(r_peaks) > 0:
        ax1.scatter(r_peaks, clean_signal[r_peaks], marker='*', s=100, 
                   color='gold', zorder=3, label='R-peaks')
    
    ax1.set_title(f"ECG Signal Denoising (Sample {idx}, SNR={snr}dB)")
    ax1.set_xlabel("Sample Index")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    """Residual comparison"""
    nlms_residual = clean_signal - nlms_output
    cnn_residual = clean_signal - cnn_output
    
    ax2.plot(nlms_residual, label='NLMS Residual', color='green', alpha=0.7)
    ax2.plot(cnn_residual, label='CNN Residual', color='purple', alpha=0.7)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    """Mark R-peak positions"""
    if len(r_peaks) > 0:
        for peak in r_peaks:
            ax2.axvline(peak, color='gold', alpha=0.3, linestyle=':')
    
    ax2.set_title("Residual Noise Comparison")
    ax2.set_xlabel("Sample Index")
    ax2.set_ylabel("Error Amplitude")
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    """Add FLOPs info"""
    plt.figtext(0.5, 0.01, f"CNN FLOPs: {flops} | Device: {device}", 
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(os.path.join(results_dir, f"denoising_analysis_{snr}dB.png"), dpi=300)
    print(f"Saved visualization: {os.path.join(results_dir, f'denoising_analysis_{snr}dB.png')}")
    plt.close()

print("\nAll visualizations completed!")