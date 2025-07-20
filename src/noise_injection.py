import os
import numpy as np


class NoiseInjector:
    """Class to inject noise into clean signals to create noisy datasets for training and testing CNN models.
    This class generates noisy datasets by adding additive white Gaussian noise (AWGN) to clean signals"""
    
    def __init__(self, processed_data_path, snr_levels=[0, 5, 10, 15, 20]):
        self.processed_dir = processed_data_path
        self.snr_levels = snr_levels
        self.noisy_data_path = os.path.join(processed_data_path, "noisy")
        os.makedirs(self.noisy_data_path, exist_ok=True)

    """Adds additive white Gaussian noise to a clean signal based on the target SNR in dB."""
    def additive_white_gaussian_noise(self, clean_signal, target_snr_db):
        signal_power = np.mean(clean_signal**2)
        snr_linear = 10**(target_snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = np.random.normal(0, np.sqrt(noise_power), clean_signal.shape)
        return clean_signal + noise
    
    """ Generates noisy datasets for training and testing by applying AWGN at specified SNR levels."""
    def generate_noisy_dataset(self, dataset_type):
        clean_path = os.path.join(self.processed_dir, f"{dataset_type}_normalized.npy")
        clean_data = np.load(clean_path)

        for snr in self.snr_levels:
            noisy_segments = []
            for signal in clean_data:
                noisy_signal = self.additive_white_gaussian_noise(signal, snr)
                noisy_segments.append(noisy_signal)
            noisy_segments = np.array(noisy_segments)

            noisy_save_path = os.path.join(
                self.noisy_data_path, f"{dataset_type}_noisy_{snr}dB.npy"
            )
            np.save(noisy_save_path, noisy_segments)
            print(f"Saved noisy {dataset_type} at {snr}dB → {noisy_save_path}")


if __name__ == "__main__":
    processed_data_path = "data/processed_data"
    injector = NoiseInjector(processed_data_path)
    
    injector.generate_noisy_dataset("train")
    injector.generate_noisy_dataset("test")
