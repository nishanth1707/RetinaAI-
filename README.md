# RetinaAI: Diabetic Retinopathy Detection System

> **Final Year Engineering Project** | AI-Powered Fundus Image Analysis

An end-to-end deep learning system for automated detection, segmentation, and analysis of Diabetic Retinopathy from retinal fundus images. The system integrates 8 modules covering classification, lesion segmentation, damage estimation, explainable AI, and risk prediction.

---

## Features (8 Modules)

| Module | Description | Output |
|--------|-------------|--------|
| **1. DR Classification** | 5-stage classification using EfficientNet-B0 | No DR, Mild NPDR, Moderate NPDR, Severe NPDR, PDR |
| **2. Lesion Segmentation** | U-Net with EfficientNet encoder | Microaneurysms, Hemorrhages, Hard Exudates, Cotton Wool Spots |
| **3. Damage Estimation** | Pixel-level damage calculation + quadrant mapping | Damage %, 4-quadrant lesion density |
| **4. Severity Index** | Risk scoring weighted by classification + damage | Low/Moderate/High Risk with confidence |
| **5. Explainable AI** | Grad-CAM heatmap visualization | Overlay highlighting model focus areas |
| **6. Interactive Dashboard** | Streamlit web application | Upload, visualize, download reports |
| **7. Report Generator** | Automated clinical PDF reports | Structured ophthalmology report |
| **8. Progression Predictor** | 1-3 year PDR risk estimation | Risk percentages + survival curve |

---

## Architecture

```
DR-Detection/
├── src/
│   ├── config.py                 # Constants, paths, hyperparameters
│   ├── utils.py                  # Image processing utilities
│   ├── model_classifier.py       # EfficientNet-B0 classifier
│   ├── model_segmenter.py        # U-Net segmenter
│   ├── train_classifier.py       # Classification training pipeline
│   ├── train_segmenter.py        # Segmentation training pipeline
│   ├── inference.py              # Unified inference pipeline
│   ├── damage_calculator.py      # Damage % + quadrant mapping
│   ├── severity_index.py         # Risk scoring
│   ├── grad_cam.py               # Grad-CAM explainable AI
│   ├── report_generator.py       # PDF report generation
│   └── progression_predictor.py  # Progression risk prediction
├── app/
│   └── dashboard.py              # Streamlit UI
├── data/                         # Datasets (APTOS, DDR/IDRiD)
├── models/                       # Trained weights
├── notebooks/                    # Jupyter notebooks
├── requirements.txt
└── run_training.py               # One-command training
```

---

## Installation

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 12.4+ (recommended) or CPU
- Windows/Linux/macOS

### Setup

```powershell
# Clone or navigate to project
cd DR-Detection

# Create virtual environment (Python 3.10)
py -3.10 -m venv venv

# Activate
.\venv\Scripts\Activate.ps1     # Windows
# source venv/bin/activate       # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Preparation

### 1. APTOS 2019 (Classification)
- Download from: https://www.kaggle.com/datasets/mariaherrerot/aptos2019
- Extract to: `data/aptos/`
```
data/aptos/
├── train.csv
├── test.csv
└── train_images/
```

### 2. DDR Dataset (Segmentation)
- Download from: https://www.kaggle.com/datasets/mariaherrerot/ddrdataset
- Extract to: `data/ddr/`
```
data/ddr/
├── train/
├── val/
└── test/
```

---

## Training

```powershell
# Train both models
python run_training.py --epochs-cls 20 --epochs-seg 30

# Train only classifier
python run_training.py --classifier-only --epochs-cls 20

# Train only segmenter
python run_training.py --segmenter-only --epochs-seg 30
```

Training times on RTX 3050 (4GB):
- Classifier (EfficientNet-B0): ~1-2 hours
- Segmenter (U-Net): ~2-3 hours

---

## Usage

### Launch Dashboard

```powershell
streamlit run app/dashboard.py
```

### Demo Mode
The dashboard runs in **Demo (Simulated)** mode by default, showing realistic results without trained models. Switch to **Local (Trained Models)** in the sidebar after training.

---

## Model Details

### Classifier (Module 1)
- **Architecture**: EfficientNet-B0 pretrained on ImageNet
- **Input**: 224x224 RGB fundus images
- **Output**: 5-class softmax probabilities
- **Training**: Cross-entropy loss, AdamW optimizer, Cosine annealing LR

### Segmenter (Module 2)
- **Architecture**: U-Net with EfficientNet-B0 encoder (via SMP)
- **Input**: 512x512 RGB fundus images
- **Output**: 4-channel binary masks (MA, HE, EX, CWS)
- **Training**: BCEWithLogitsLoss, AdamW optimizer

---

## Addressing Known Limitations

| Limitation (from IEEE papers) | Our Solution |
|-------------------------------|--------------|
| Low accuracy on poor quality images | Grad-CAM visual validation (Module 5) |
| Large datasets are cumbersome | Automated pipeline with organized data structure |
| Patients don't trust AI | Explainable AI with confidence scores (Module 5) |
| Complex implementation | Modular architecture, one-command training |
| Focus on single disease only | Multi-lesion detection + potential other disease flags |
| Pre-processing required | Built-in preprocessing pipeline |
| Deployment uncertainty | Streamlit dashboard, Docker support |

---

## Project Report

See [REPORT.md](REPORT.md) for the detailed project report template suitable for final year submission.

---



## License

MIT License

## Acknowledgements

- APTOS 2019 Blindness Detection (Kaggle)
- DDR Dataset (Chinese Academy of Sciences)
- IDRiD Dataset (IEEE Dataport)
