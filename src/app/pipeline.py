import os
from pathlib import Path
import sys
import dotenv

dotenv.load_dotenv()

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import MY_DATASETS
from src.utils.data import download_datasets as downloder  # type: ignore

def data_init():
    """Downloads and extracts datasets from Kaggle."""
    output_dir = Path(os.getenv("OUTPUT_DIR") or "")
    downloder(MY_DATASETS, output_dir)

if __name__ == "__main__":
    data_init()



