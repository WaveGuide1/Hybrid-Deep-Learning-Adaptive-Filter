import os
import numpy as np
import pandas as pd # type: ignore

class DataProcessor:
    def __init__(self, raw_data_path, processed_data_path, window_size=128, overlap=0.5):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.window_size = window_size
        self.overlap = overlap

        os.makedirs(self.processed_data_path, exist_ok=True)

    def load_data(self, file_path):
        """
        Load data from a CSV file.
        """
        df = pd.read_csv(file_path, header=None)
        print(f"Loaded {df.shape[0]} signals with {df.shape[1]} columns from {file_path}")
        signals = df.iloc[:, :-1].values.astype(np.float32)
        return signals
    
    def clean_data(self, signals):
        """
        Clean the data by removing NaN values.
        """
        non_zero_indices = np.where(signals != 0)[0]
        if len(non_zero_indices) == 0:
            return None
        last_index = non_zero_indices[-1]
        return signals[:last_index + 1]
    
    def normalize_data(self, signals):
        """
        Normalize the data to the range [0, 1].
        """
        min_val = np.min(signals)
        max_val = np.max(signals)
        if max_val == min_val:
            return signals
        normalized =  2 * (signals - min_val) / (max_val - min_val) - 1
        return normalized
    
    def segment_data(self, signals):
        """
        Segment the data into overlapping windows.
        """
        segments = []
        step = int(self.window_size * (1 - self.overlap))
        for start in range(0, len(signals) - self.window_size + 1, step):
            window = signals[start:start + self.window_size]
            segments.append(window)
        return np.array(segments)
    
    def process_file(self, csv_file, dataset_type):
        """
        Process a single file: load, clean, normalize, and segment the data.
        """
        signals = self.load_data(csv_file)
        all_segments = []

        for idx, signal in enumerate(signals):
            cleaned = self.clean_data(signal)
            if cleaned is None or len(cleaned) < self.window_size:
                continue
            normalized = self.normalize_data(cleaned)
            segments = self.segment_data(normalized)
            all_segments.append(segments)

        all_segments = np.concatenate(all_segments, axis=0)
        save_path = os.path.join(self.processed_data_path, f"{dataset_type}_segments.npy")
        np.save(save_path, all_segments)
        print(f"Saved {all_segments.shape[0]} segments to {save_path}")

    def process_dataset(self):
        train_file = os.path.join(self.raw_data_path, "mitbih_train.csv")
        test_file  = os.path.join(self.raw_data_path, "mitbih_test.csv")

        self.process_file(train_file, "train")
        self.process_file(test_file, "test")


if __name__ == "__main__":
    raw_data_path = "data/ecg"          
    processed_data_path = "data/processed"

    processor = DataProcessor(raw_data_path, processed_data_path)
    processor.process_dataset()