import base64
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from zipfile import ZipFile
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

def download_dataset(dataset: str, output_dir: Path) -> Path:
    """Downloads and extracts a single Kaggle dataset using safe file handling."""
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_API")
    
    if not (username and key):
        config = Path(os.getenv("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
        if config.exists():
            data = json.loads(config.read_text())
            username, key = data.get("username"), data.get("key")
    
    if not (username and key):
        raise RuntimeError("Kaggle credentials not found.")

    dataset_dir = output_dir / dataset.rsplit("/", 1)[-1]
    
    if dataset_dir.exists() and any(dataset_dir.iterdir()):
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)
    auth = base64.b64encode(f"{username}:{key}".encode()).decode()
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset}"
    req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})

    # Use mkstemp to avoid Windows file locking issues
    fd, tmp_path = tempfile.mkstemp(suffix=".zip")
    tmp_path = Path(tmp_path)
    
    try:
        # Open file descriptor for writing
        with os.fdopen(fd, 'wb') as tmp:
            with urllib.request.urlopen(req) as res, tqdm(
                total=int(res.headers.get("Content-Length", 0)), 
                unit="B", unit_scale=True, 
                desc=f"Downloading {dataset.split('/')[-1]}"
            ) as pbar:
                while chunk := res.read(1024 * 1024):
                    tmp.write(chunk)
                    pbar.update(len(chunk))
        
        # Extraction
        with ZipFile(tmp_path) as zf:
            zf.extractall(dataset_dir)
            
    finally:
        # Safe cleanup
        if tmp_path.exists():
            os.remove(tmp_path)
            
    return dataset_dir

def download_datasets(datasets: list[str], output_dir: Path):
    """Call this to download all datasets in the list."""
    for ds in datasets:
        try:
            path = download_dataset(ds, output_dir)
            print(f"Dataset: {path}")
        except Exception as e:
            print(f"Failed to download {ds}: {e}")