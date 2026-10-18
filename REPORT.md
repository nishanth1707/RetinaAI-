# RetinaAI: Diabetic Retinopathy Detection System
## Final Year Engineering Project Report

---

## Abstract

Diabetic Retinopathy (DR) is a leading cause of preventable blindness among working-age adults worldwide. Early detection and treatment can prevent vision loss, but manual screening of fundus images is time-consuming and requires specialized expertise. This project presents **RetinaAI**, an end-to-end deep learning system that automates DR detection, lesion segmentation, damage estimation, and risk prediction from retinal fundus images. The system integrates eight comprehensive modules: (1) DR classification across 5 stages, (2) pixel-level lesion segmentation for 4 lesion types, (3) retinal damage percentage estimation with quadrant mapping, (4) DR severity indexing with confidence scoring, (5) Grad-CAM explainable AI visualization, (6) an interactive Streamlit dashboard, (7) automated clinical report generation, and (8) disease progression prediction. The system addresses key limitations identified in IEEE literature including trust deficits through explainable AI, limited lesion scope through multi-lesion segmentation, and deployment complexity through a modular, production-ready architecture.

---

## 1. Introduction

### 1.1 Background
Diabetic Retinopathy is a microvascular complication of diabetes that affects the retina. According to the International Diabetes Federation, approximately 425 million adults have diabetes globally, with one in three expected to develop some form of DR. The disease progresses through five stages: No DR, Mild Non-Proliferative DR (NPDR), Moderate NPDR, Severe NPDR, and Proliferative DR (PDR).

### 1.2 Problem Statement
Manual DR screening faces several challenges:
- Shortage of ophthalmologists in rural and underserved areas
- Subjectivity in human interpretation
- Time-consuming manual analysis
- High false-negative rates in early-stage detection

### 1.3 Objectives
1. Develop a robust deep learning model for 5-stage DR classification
2. Implement pixel-level segmentation of 4 lesion types
3. Calculate quantitative retinal damage percentage
4. Generate severity risk indices with confidence scores
5. Provide explainable AI visualizations for clinical trust
6. Build an interactive dashboard for clinical deployment
7. Generate automated ophthalmology reports
8. Predict disease progression risk over 1-3 years

### 1.4 Scope
The system focuses on color fundus photography analysis, the most common and accessible retinal imaging modality. It covers both non-proliferative and proliferative stages of DR.

---

## 2. Literature Review

### 2.1 Existing Approaches
[Summarize 5-8 IEEE papers on DR detection]
- Paper 1: [Author, Year] - [Method] - [Accuracy/Limitation]
- Paper 2: [Author, Year] - [Method] - [Accuracy/Limitation]
- ...

### 2.2 Identified Limitations
Based on our review of IEEE papers, the following limitations were identified:

| # | Limitation | Our Solution |
|---|-----------|--------------|
| 1 | Accuracy varies with image quality | Grad-CAM validation + confidence scoring |
| 2 | Large dataset management burden | Automated organized data pipeline |
| 3 | Patient/clinician trust deficit | Explainable AI (Module 5) |
| 4 | Complex implementation | Modular architecture |
| 5 | Complex architecture overhead | EfficientNet + U-Net balance |
| 6 | Single disease focus | Multi-lesion detection (Module 2) |
| 7 | Pre-processing dependency | Built-in preprocessing |
| 8 | Deployment uncertainty | Streamlit + Docker support |

---

## 3. System Architecture

### 3.1 Overall Architecture

```
Input Image
    |
    ├──> Preprocessing (resize, normalize)
    |
    ├──> [Module 1] DR Classifier (EfficientNet-B0)
    |       └──> DR Stage + Confidence
    |
    ├──> [Module 2] Lesion Segmenter (U-Net)
    |       └──> 4 Lesion Masks
    |
    ├──> [Module 3] Damage Calculator
    |       └──> Damage % + Quadrant Map
    |
    ├──> [Module 4] Severity Index
    |       └──> Risk Level + Score
    |
    ├──> [Module 5] Grad-CAM
    |       └──> Heatmap Overlay
    |
    ├──> [Module 8] Progression Predictor
    |       └──> 1-3 Year Risk
    |
    └──> [Module 6+7] Dashboard + Report
            └──> Interactive UI + PDF
```

### 3.2 Technology Stack
- **Framework**: PyTorch 2.5.1
- **Classification**: EfficientNet-B0 (timm)
- **Segmentation**: U-Net (Segmentation Models PyTorch)
- **Explainable AI**: Grad-CAM
- **Backend**: Python 3.10
- **Frontend**: Streamlit 1.28+
- **Visualization**: Plotly, Matplotlib, OpenCV
- **Reporting**: FPDF2

---

## 4. Methodology

### 4.1 Dataset

| Dataset | Use | Images | Source |
|---------|-----|--------|--------|
| APTOS 2019 | Classification | 3,662 | Kaggle |
| DDR | Segmentation | 757 | Kaggle/HuggingFace |

**Data Split**: 85% training, 15% validation

### 4.2 Preprocessing
- Resize to 224x224 (classifier) / 512x512 (segmenter)
- Normalization: ImageNet mean/std
- Augmentation: Random rotation, flip, brightness/contrast, Gaussian blur

### 4.3 Classification Model
- **Architecture**: EfficientNet-B0
- **Pretrained**: ImageNet
- **Fine-tuning**: All layers
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-4)
- **Loss**: Weighted cross-entropy + label smoothing (0.1)
- **Scheduler**: ReduceLROnPlateau (factor 0.5, patience 3)
- **Epochs**: 30
- **Batch size**: 16

### 4.4 Segmentation Model
- **Architecture**: U-Net with EfficientNet-B0 encoder
- **Pretrained**: ImageNet
- **Optimizer**: AdamW (lr=1e-4)
- **Loss**: Focal loss + Dice loss (combined)
- **Epochs**: 100
- **Batch size**: 4

### 4.5 Damage Calculation
- Retina mask via thresholding + morphological ops
- Lesion area = sum of segmented lesion pixels
- Damage % = (Lesion Area / Retina Area) × 100
- Quadrant division at retinal center point

### 4.6 Severity Index
- Weighted combination: 30% damage + 70% classifier probability
- Thresholds: <30 = Low Risk, 30-70 = Moderate Risk, >70 = High Risk

### 4.7 Grad-CAM
- Gradient-weighted Class Activation Mapping
- Targets final convolutional layer of EfficientNet
- Overlay: jet colormap at α=0.5 transparency

### 4.8 Progression Prediction
- Risk = 0.5×base_risk + 0.3×severity + 0.2×damage
- Base risk from DR stage: 0.02 (No DR) to 0.60 (PDR)
- Multi-year projection with survival curve

---

## 5. Implementation

### 5.1 Module 1: DR Classification
[Implementation details, code structure, key classes]

### 5.2 Module 2: Lesion Segmentation
[Implementation details]

### 5.3 Module 3: Damage Estimation
[Implementation details]

### 5.4 Module 4: Severity Index
[Implementation details]

### 5.5 Module 5: Explainable AI
[Implementation details]

### 5.6 Module 6: Interactive Dashboard
[Implementation details]

### 5.7 Module 7: Report Generator
[Implementation details]

### 5.8 Module 8: Progression Predictor
[Implementation details]

---

## 6. Results and Analysis

### 6.1 Classification Performance
| Metric | Value |
|--------|-------|
| Validation Accuracy | 81.97% |
| Macro AUC-ROC | 0.8921 |
| Precision (macro) | 68.30% |
| Recall (macro) | 69.41% |
| F1-Score (macro) | 68.72% |

**Per-class metrics (APTOS validation set, n=366):**

| DR Class | Precision | Recall | F1-Score |
|----------|-----------|--------|----------|
| No DR | 98.27% | 98.84% | 98.55% |
| Mild NPDR | 65.91% | 72.50% | 69.05% |
| Moderate NPDR | 77.32% | 72.12% | 74.63% |
| Severe NPDR | 42.31% | 50.00% | 45.83% |
| Proliferative DR | 57.69% | 53.57% | 55.56% |

### 6.2 Segmentation Performance
| Lesion Type | Dice Coefficient | IoU |
|-------------|-----------------|-----|
| Microaneurysms | 0.1795 | 0.1220 |
| Hemorrhages | 0.2990 | 0.2023 |
| Hard Exudates | 0.0963 | 0.0567 |
| Cotton Wool Spots | 0.3221 | 0.2410 |
| **Average** | **0.2242** | **0.1555** |

### 6.3 Confusion Matrix

```
        No DR  Mild   Moder  Severe Prolif
No DR     170     2      0      0      0
Mild        1    29      9      0      1
Moderate    2    11     75     11      5
Severe      0     1      5     11      5
Prolif      0     1      8      4     15
```

### 6.4 Sample Results
- **Classifier**: 81.97% accuracy on 366 held-out fundus images (81.98% on APTOS test split)
- **Segmenter**: average Dice 0.224 across 4 lesion types on 149 DDR validation images
- **Grad-CAM**: heatmaps consistently highlight exudates and hemorrhages in affected regions
- **Dashboard**: full pipeline runs in <3 seconds per image on RTX 3050 GPU

---

## 7. Addressing Limitations

### 7.1 Trust and Transparency (Module 5)
Grad-CAM heatmaps provide visual explanations of model decisions, allowing clinicians to verify that the model focuses on clinically relevant regions.

### 7.2 Multi-Lesion Detection (Module 2)
Unlike systems that detect only one lesion type, our system segments four distinct lesion categories simultaneously.

### 7.3 Confidence Scoring (Module 4)
Every prediction includes a confidence score, enabling clinicians to assess reliability.

---

## 8. Conclusion and Future Work

### 8.1 Conclusion
The RetinaAI system successfully demonstrates a comprehensive, modular approach to automated DR detection. The integration of explainable AI addresses the critical trust barrier identified in existing literature.

### 8.2 Future Work
1. Integration with OCT imaging modality
2. Federated learning for privacy-preserving training
3. Mobile deployment for rural screening
4. Real-time video fundus analysis
5. Integration with electronic health records

---

## 9. References

1. Porwal, P., et al. "Indian Diabetic Retinopathy Image Dataset (IDRiD)." Data, 2018.
2. [Reference 2]
3. [Reference 3]
...
[Include 10-15 IEEE/Scopus indexed references]

---

## Appendices

### A. Source Code Structure
[Project tree]

### B. Installation Guide
[Step-by-step setup]

### C. User Manual
[Dashboard usage guide]
