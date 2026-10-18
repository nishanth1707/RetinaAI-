import kagglehub
from pathlib import Path
import sys
import shutil

sys.path.append(str(Path(__file__).resolve().parent))
from src.config import APTOS_DIR, IDRiD_DIR

DDR_DIR = Path(__file__).resolve().parent / "data" / "ddr"


def download_aptos():
    print("=" * 50)
    print("Downloading APTOS 2019 (for classification)...")
    print("=" * 50)
    try:
        path = kagglehub.competition_download("aptos2019-blindness-detection")
        print(f"APTOS downloaded to: {path}")
        print(f"Structure: organize into data/aptos/train_images/*.png")
        print(f"           with data/aptos/train.csv")
        return path
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print()
        print("Manual download:")
        print("  1. https://www.kaggle.com/c/aptos2019-blindness-detection")
        print("  2. Download & extract to:", APTOS_DIR)
        return None


def download_ddr():
    print()
    print("=" * 50)
    print("Downloading DDR dataset (for segmentation)...")
    print("=" * 50)
    print("Size: ~2.98 GB (may take 15-30 min)")
    print()
    try:
        path = kagglehub.dataset_download("mariaherrerot/ddrdataset")
        print(f"DDR downloaded to: {path}")
        print(f"Copying to {DDR_DIR}...")
        DDR_DIR.mkdir(parents=True, exist_ok=True)
        for item in Path(path).iterdir():
            dest = DDR_DIR / item.name
            if not dest.exists():
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
        print(f"DDR ready at: {DDR_DIR}")
        print("Structure: data/ddr/train/ (images + masks)")
        return path
    except Exception as e:
        print(f"Kaggle download failed: {e}")
        print()
        print("Alternative sources:")
        print("  Option 1 - DDR from HuggingFace:")
        print("    from huggingface_hub import snapshot_download")
        print("    path = snapshot_download('ctmedtech/DDR-dataset')")
        print()
        print("  Option 2 - IDRiD from IEEE Dataport:")
        print("    https://ieee-dataport.org/open-access/")
        print("    indian-diabetic-retinopathy-image-dataset-idrid")
        print()
        print("  Option 3 - Use Demo mode (no training needed)")
        print("    streamlit run app/dashboard.py  (select 'Demo' mode)")
        return None


if __name__ == "__main__":
    download_aptos()
    download_ddr()
    print()
    print("=" * 50)
    print("Done! Next: train models with:")
    print("  .\\venv\\Scripts\\python.exe run_training.py")
    print("=" * 50)
