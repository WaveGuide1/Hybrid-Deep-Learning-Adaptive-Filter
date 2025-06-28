import numpy as np
import torch

"""
SNR Calculation
"""
def calculate_snr(clean, denoised):
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean((clean - denoised) ** 2)
    snr = 10 * np.log10(signal_power / noise_power + 1e-8)
    return snr

"""
MSE Calculation
"""
def calculate_mse(clean, denoised):
    return np.mean((clean - denoised) ** 2)

"""
FLOPs Estimation (Simple Rough Estimate)
"""
def estimate_cnn_flops(model, input_shape=(1, 1, 187)):
    total_flops = 0

    dummy_input = torch.randn(*input_shape)
    hooks = []

    def count_flops(module, input, output):
        nonlocal total_flops
        if isinstance(module, torch.nn.Conv1d):
            out_channels, kernel_size = module.weight.shape[0], module.kernel_size[0]
            out_length = output.shape[-1]
            flops = out_channels * kernel_size * out_length
            total_flops += flops

        elif isinstance(module, torch.nn.Linear):
            flops = module.in_features * module.out_features
            total_flops += flops

    for layer in model.modules():
        hooks.append(layer.register_forward_hook(count_flops))

    model(dummy_input)
    for hook in hooks:
        hook.remove()

    return total_flops

