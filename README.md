# Comparative Evaluation of Deep Learning and NLMS Adaptive Filtering for ECG Denoising in Resource Constrained Environments.

![ECG Denoising Results](results/figures/)

## Abstract
The demand for real-time signal processing in embedded systems, particularly for healthcare applications, has led to the exploration of efficient noise reduction techniques. This study presents a comparative evaluation of two distinct approaches to ECG signal denoising: Normalized Least Mean Squares (NLMS) adaptive filtering and a lightweight Convolutional Neural Network (CNN). The ECG5000 dataset was normalized, augmented with additive white Gaussian noise (AWGN) at varying signal-to-noise ratio (SNR) levels, and processed by both models. Quantitative analysis revealed the CNN achieved 6.51 dB output SNR at 5 dB input (vs NLMS' 1.42 dB, p<0.01) but required 239× more FLOPs (119.9K vs 0.5K). Results show that while NLMS has lower computational complexity, CNN provides superior noise suppression and faster inference times (0.94ms vs 2.47ms) when optimized. This work offers insights into the implementation-dependent trade-offs between traditional and learning-based approaches under resource constraints typical of edge devices. 

## The implementation includes:

-  **Hybrid CNN architecture** optimized for real-time ECG denoising
-  **NLMS adaptive filter** implementation with configurable parameters
- ⚡ **Performance benchmarking** across multiple noise levels (SNR 0-20dB)
- 📊 **Visualization tools** for qualitative and quantitative analysis

The CNN model demonstrates **>10× speedup** compared to NLMS while maintaining superior denoising performance across all tested SNR levels.

## Key Features

- **End-to-end pipeline** from dataset acquisition to performance evaluation
- **Noise injection** at controlled SNR levels (0-20dB)
- **Computational efficiency metrics** (FLOPs, inference time)
- **Signal quality metrics** (SNR improvement, MSE)
- **Visualization tools** for waveform and residual analysis

## Installation

```bash
# Clone repository
git clone https://github.com/WaveGuide1/Hybrid-Deep-Learning-Adaptive-Filter
cd Hybrid-Deep-Learning-Adaptive-Filter

# Create virtual environment
python -m venv ecg-env
source ecg-env/bin/activate  # Linux/Mac
.\ecg-env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
