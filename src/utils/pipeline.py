import os
from pathlib import Path
import sys
import dotenv

dotenv.load_dotenv()

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.dataset import download_datasets
from src.utils.eda_imp import eda_imp
from src.utils.balance import data_split


def _data_init():
    """Downloads and extracts datasets from Kaggle."""
    output_dir = Path(os.getenv("OUTPUT_DIR") or "")
    MY_DATASETS = ["muhammadzamancuiisb/casme2-preprocessed-v2","daremobolaji/ckplusferdata","iconistt/sammassamexpression"]
    print("+" * 50)
    print("Step 01: [data init]")
    print("+" * 50)
    download_datasets(MY_DATASETS, output_dir)
    
def _run_eda(num: int):
    if num == 1:
        DATA_PATH = './data/ckplusferdata'
        ck_plus = eda_imp(
            DATA_PATH=DATA_PATH,
            OUTPUT_DIR=DATA_PATH,
            NAME="eda.csv"
        )
        print(f"[9]EDA for ckplus saved successfully.")
        data_split(ck_plus, output_dir=DATA_PATH, TARGET='class')
        print(f"[10]Data Balance for ckplus saved successfully.")
        
    elif num ==2:
        samm = eda_imp(
        DATA_PATH='./data/sammassamexpression',
        OUTPUT_DIR='./data/sammassamexpression',
        NAME="eda.csv"
        )
        print(f"[9]EDA for samm saved successfully.")
        data_split(samm, output_dir='./data/sammassamexpression', TARGET='class')
        print(f"[10]Data Balance for samm saved successfully.")
    elif num ==3:
        Data_PATH = './data/casme2-preprocessed-v2'
        casme2 = eda_imp(
        DATA_PATH=Data_PATH,
        OUTPUT_DIR=Data_PATH,
        NAME="eda.csv"
        )
        print(f"[9]EDA for casme2 saved successfully.")
        data_split(casme2, output_dir=Data_PATH, TARGET='class')
        print(f"[10]Data Balance for casme2 saved successfully.")
    elif num > 3 or num < 1:
        print("Invalid input. Please enter a number between 1 and 3.")
        

if __name__ == "__main__":
    
    # 1. Data Download and Extraction
    _data_init()
    # 2. Perform EDA on (ckplus, samm, casme2) datasets

    _run_eda(1)
    _run_eda(2)
    _run_eda(3)
    
    # 3. Dataset Balancing and Splitting
    



