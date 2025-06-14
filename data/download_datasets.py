import kagglehub
import os
import shutil

def download_dataset(dataset_names, path):
    """
    Downloads datasets from Kaggle using kagglehub.
    """
    print("Downloading {} datasets from Kaggle...".format(dataset_names))
    dataset_path = kagglehub.dataset_download(dataset_names)
    os.makedirs(path, exist_ok=True)
    
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        dst = os.path.join(path, item)
        if os.path.isdir(item_path):
            shutil.copytree(item_path, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item_path, path)
    
    print("Datasets {} save to {}".format(dataset_names, path))

if __name__ == "__main__":
    # ECG dataset
    # download_dataset("shayanfazeli/heartbeat", "data/ecg")

    # Vibration dataset
    # download_dataset("dnkumars/industrial-equipment-monitoring-dataset", "data/vibration")

    # Audio dataset
    # download_dataset("chrisfilo/urbansound8k", "data/audio")

    pass
