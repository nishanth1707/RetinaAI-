import argparse
import csv
import json
import glob
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import timm
from tqdm import tqdm

os.makedirs("reports", exist_ok=True)
os.makedirs("models/comparison", exist_ok=True)

IMG_SIZE = 224
TRAIN_CSV = "data/aptos/train_1.csv"
VALID_CSV = "data/aptos/valid.csv"
TRAIN_DIR = "data/aptos/train_images/train_images"
VAL_DIR = "data/aptos/val_images/val_images"
RESULT_JSON = "reports/comparison_results.json"


class FundusDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.transform(image=img)
        return aug["image"], self.labels[idx]


def get_transforms():
    train_tf = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    val_tf = A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return train_tf, val_tf


def load_split(csv_path, img_dir):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            stem = r["id_code"]
            for ext in (".png", ".jpg", ".jpeg"):
                p = Path(img_dir) / f"{stem}{ext}"
                if p.exists():
                    rows.append((p, int(r["diagnosis"])))
                    break
    paths = [r[0] for r in rows]
    labels = [r[1] for r in rows]
    return paths, labels


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cm = np.zeros((5, 5), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = np.trace(cm) / cm.sum()
    prec, rec, f1 = [], [], []
    for i in range(5):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p_ = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f_ = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0.0
        prec.append(p_)
        rec.append(r_)
        f1.append(f_)
    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(float(np.mean(prec)) * 100, 2),
        "recall": round(float(np.mean(rec)) * 100, 2),
        "f1": round(float(np.mean(f1)) * 100, 2),
        "per_class_f1": [round(x * 100, 2) for x in f1],
        "per_class_precision": [round(x * 100, 2) for x in prec],
        "per_class_recall": [round(x * 100, 2) for x in rec],
    }


def train_one(model_name, epochs, lr, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_paths, train_labels = load_split(TRAIN_CSV, TRAIN_DIR)
    val_paths, val_labels = load_split(VALID_CSV, VAL_DIR)
    print(f"[{model_name}] train={len(train_paths)} val={len(val_paths)} device={device}")

    train_tf, val_tf = get_transforms()
    train_loader = DataLoader(FundusDataset(train_paths, train_labels, train_tf),
                              batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(FundusDataset(val_paths, val_labels, val_tf),
                            batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = timm.create_model(model_name, pretrained=True, num_classes=5).to(device)

    counts = [0] * 5
    for l in train_labels:
        counts[l] += 1
    total = sum(counts)
    weights = torch.tensor([total / (5 * c) for c in counts], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    best_acc = 0.0
    best_metrics = None
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"[{model_name}] ep{epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

        metrics = evaluate(model, val_loader, device)
        scheduler.step(metrics["accuracy"])
        print(f"[{model_name}] epoch {epoch+1}: acc={metrics['accuracy']:.2f}% f1={metrics['f1']:.2f}%")
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_metrics = metrics
            torch.save({"model_state_dict": model.state_dict(),
                        "val_accuracy": best_acc,
                        "arch": model_name},
                       f"models/comparison/{model_name}.pth")

    elapsed = round((time.time() - t0) / 60, 1)
    result = {"arch": model_name, "epochs": epochs, "params_m": round(
        sum(p.numel() for p in model.parameters()) / 1e6, 2), "best": best_metrics, "minutes": elapsed}

    results = {}
    if os.path.exists(RESULT_JSON):
        with open(RESULT_JSON) as f:
            results = json.load(f)
    results[model_name] = result
    with open(RESULT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[{model_name}] DONE best_acc={best_acc:.2f}% ({elapsed} min) -> saved to {RESULT_JSON}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    train_one(args.model, args.epochs, args.lr, args.batch_size)
