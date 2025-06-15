import kagglehub
import os
import shutil
import requests
import zipfile

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


def download_github_zip(url, extract_to):
    """
    Download dataset from GitHub.
    """
    print("Downloading dataset from {}...".format(url))
    response = requests.get(url, stream=True)
    response.raise_for_status()

    zip_path = os.path.join(extract_to, "temp_download.zip")
    os.makedirs(extract_to, exist_ok=True)

    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print("Download complete. Extracting...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    os.remove(zip_path)
    print("Extraction complete: {}".format(extract_to))

if __name__ == "__main__":
    # ECG dataset
    # download_dataset("shayanfazeli/heartbeat", "data/ecg")

    # Vibration dataset
    # download_dataset("dnkumars/industrial-equipment-monitoring-dataset", "data/vibration")

    # Audio dataset
    # download_dataset("chrisfilo/urbansound8k", "data/audio")

    # Audio dataset from GitHub
    # download_github_zip("https://github.com/karoldvl/ESC-50/archive/master.zip", "data/audio")
    pass

