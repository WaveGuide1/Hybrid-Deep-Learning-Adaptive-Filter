import os
import numpy as np
import pandas as pd  # type: ignore

class DataProcessor:
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path

        os.makedirs(self.processed_data_path, exist_ok=True)

    def load_data(self, file_path):
        """
        Load data from a CSV file.
        """
        df = pd.read_csv(file_path, header=None)
        print(f"Loaded {df.shape[0]} signals with {df.shape[1]} columns from {file_path}")
        signals = df.iloc[:, :-1].values.astype(np.float32)  # exclude label
        return signals

    def clean_data(self, signal):
        """
        Clean the data by removing trailing zeros.
        """
        non_zero_indices = np.where(signal != 0)[0]
        if len(non_zero_indices) == 0:
            return None
        last_index = non_zero_indices[-1]
        return signal[:last_index + 1]

    def normalize_data(self, signal):
        """
        Normalize signal to range [-1, 1].
        """
        min_val = np.min(signal)
        max_val = np.max(signal)
        if max_val == min_val:
            return signal
        normalized = 2 * (signal - min_val) / (max_val - min_val) - 1
        return normalized

    def process_file(self, csv_file, dataset_type):
        """
        Load, clean, normalize each signal and save full-length version (no segmentation).
        """
        signals = self.load_data(csv_file)
        processed_signals = []

        for idx, signal in enumerate(signals):
            cleaned = self.clean_data(signal)
            if cleaned is None:
                continue
            normalized = self.normalize_data(cleaned)
            # Ensure fixed length (e.g., pad to 187 if needed)
            if len(normalized) < 187:
                padded = np.pad(normalized, (0, 187 - len(normalized)))
                processed_signals.append(padded)
            elif len(normalized) > 187:
                truncated = normalized[:187]
                processed_signals.append(truncated)
            else:
                processed_signals.append(normalized)

        processed_signals = np.array(processed_signals, dtype=np.float32)
        save_path = os.path.join(self.processed_data_path, f"{dataset_type}_normalized.npy")
        np.save(save_path, processed_signals)
        print(f"Saved {processed_signals.shape[0]} full-length signals to {save_path}")

    def process_dataset(self):
        train_file = os.path.join(self.raw_data_path, "mitbih_train.csv")
        test_file  = os.path.join(self.raw_data_path, "mitbih_test.csv")

        self.process_file(train_file, "train")
        self.process_file(test_file, "test")


if __name__ == "__main__":
    raw_data_path = "data/ecg"
    processed_data_path = "data/processed_data"

    processor = DataProcessor(raw_data_path, processed_data_path)
    processor.process_dataset()
