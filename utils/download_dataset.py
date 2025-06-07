import os
import urllib.request
import zipfile
from pathlib import Path

# Dataset URL (official RAVDESS audio speech subset)
DATASET_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
ZIP_FILENAME = "Audio_Speech_Actors_01-24.zip"
TARGET_FOLDER = Path("data\\ravdess")
EXTRACTED_FOLDER = TARGET_FOLDER / "Audio_Speech_Actors_01-24"

def download_file(url, save_path):
    print(f"Downloading dataset from {url}...")
    urllib.request.urlretrieve(url, save_path)
    print("Download completed.")

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("Extraction completed.")

def main():
    TARGET_FOLDER.mkdir(parents=True, exist_ok=True)
    zip_path = TARGET_FOLDER / ZIP_FILENAME

    # Step 1: Download
    if not zip_path.exists():
        download_file(DATASET_URL, zip_path)
    else:
        print("Zip file already exists. Skipping download.")

    # Step 2: Extract
    if not EXTRACTED_FOLDER.exists():
        extract_zip(zip_path, TARGET_FOLDER)
    else:
        print("Dataset already extracted. Skipping extraction.")

    # Optional: remove zip to save space
    zip_path.unlink()

if __name__ == "__main__":
    main()