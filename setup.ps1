Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  DR Detection System - Setup Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$ErrorActionPreference = "Stop"

Write-Host "[1/4] Creating virtual environment..." -ForegroundColor Yellow
py -3.10 -m venv venv
if (-not $?) { throw "Failed to create virtual environment" }

Write-Host "[2/4] Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

Write-Host "[3/4] Installing dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
if (-not $?) {
    Write-Host "CUDA PyTorch failed, installing CPU version..." -ForegroundColor Yellow
    pip install torch torchvision torchaudio
}
pip install -r requirements.txt

Write-Host "[4/4] Verifying installation..." -ForegroundColor Yellow
python -c "import torch; import torchvision; import streamlit; import cv2; print('All imports successful!')"
if (-not $?) { throw "Import verification failed" }

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  Setup Complete!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Download datasets from Kaggle:"
Write-Host "     - APTOS 2019: https://www.kaggle.com/c/aptos2019-blindness-detection"
Write-Host "     - IDRiD: https://www.kaggle.com/datasets/balraj98/idrid-dataset"
Write-Host "  2. Place datasets in data/aptos and data/idrid folders"
Write-Host "  3. Train models: python run_training.py"
Write-Host "  4. Launch dashboard: streamlit run app/dashboard.py"
Write-Host ""
