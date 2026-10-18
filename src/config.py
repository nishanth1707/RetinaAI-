import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
APP_DIR = BASE_DIR / "app"

APTOS_DIR = DATA_DIR / "aptos"
IDRiD_DIR = DATA_DIR / "idrid"
DDR_SEG_DIR = DATA_DIR / "ddr_seg" / "lesion_segmentation"

CLASSIFIER_PATH = MODELS_DIR / "dr_classifier.pth"
SEGMENTER_PATH = MODELS_DIR / "dr_segmenter.pth"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

DR_CLASSES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]

DR_CLASSES_SHORT = ["No_DR", "Mild", "Moderate", "Severe", "PDR"]

LESION_NAMES = ["Microaneurysms", "Hemorrhages", "Hard Exudates", "Cotton Wool Spots"]

LESION_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (255, 255, 0),
    (0, 0, 255),
]

IMG_SIZE_CLASSIFIER = 224
IMG_SIZE_SEGMENTER = 512

NUM_CLASSES = 5
NUM_LESIONS = 4

QUADRANT_NAMES = ["Superior Temporal", "Superior Nasal", "Inferior Temporal", "Inferior Nasal"]

SEVERITY_LABELS = ["Low Risk", "Moderate Risk", "High Risk"]
SEVERITY_THRESHOLDS = [0.30, 0.70]
