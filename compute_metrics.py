import glob
import csv
import os
import numpy as np
import torch
from pathlib import Path

os.makedirs("reports", exist_ok=True)

from src.utils import preprocess_image
from src.model_classifier import load_classifier, predict_classifier
from src.config import DR_CLASSES

CSV_PATH = "data/aptos/valid.csv"
IMG_DIR = "data/aptos/val_images/val_images"

labels = {}
with open(CSV_PATH, newline="") as f:
    for row in csv.DictReader(f):
        labels[row["id_code"]] = int(row["diagnosis"])

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_classifier("models/dr_classifier.pth", device=device)
print("Loaded classifier on", device)

y_true, y_pred, confs = [], [], []
paths = sorted(glob.glob(f"{IMG_DIR}/*.png"))
for p in paths:
    stem = Path(p).stem
    if stem not in labels:
        continue
    tensor, _ = preprocess_image(p)
    pred, conf, probs = predict_classifier(model, tensor, device)
    y_true.append(labels[stem])
    y_pred.append(pred)
    confs.append(conf)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
print(f"Evaluated {len(y_true)} images\n")

n = 5
cm = np.zeros((n, n), dtype=int)
for t, p in zip(y_true, y_pred):
    cm[t, p] += 1

acc = np.trace(cm) / cm.sum()
print(f"Overall Accuracy: {acc*100:.2f}%\n")

print(f"{'Class':<16}{'Precision':>10}{'Recall':>10}{'F1':>10}{'Support':>10}")
prec_list, rec_list, f1_list = [], [], []
for i in range(n):
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    sup = cm[i, :].sum()
    prec_list.append(prec)
    rec_list.append(rec)
    f1_list.append(f1)
    print(f"{DR_CLASSES[i]:<16}{prec*100:>9.1f}%{rec*100:>9.1f}%{f1*100:>9.1f}%{sup:>10}")

print(f"\n{'MACRO AVG':<16}{np.mean(prec_list)*100:>9.1f}%{np.mean(rec_list)*100:>9.1f}%{np.mean(f1_list)*100:>9.1f}%{len(y_true):>10}")
print(f"Mean confidence: {np.mean(confs)*100:.1f}%")

with open("reports/classifier_metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc*100:.2f}%\n")
    f.write(f"Macro Precision: {np.mean(prec_list)*100:.2f}%\n")
    f.write(f"Macro Recall: {np.mean(rec_list)*100:.2f}%\n")
    f.write(f"Macro F1: {np.mean(f1_list)*100:.2f}%\n")
    for i in range(n):
        f.write(f"{DR_CLASSES[i]}: P={prec_list[i]*100:.1f} R={rec_list[i]*100:.1f} F1={f1_list[i]*100:.1f} n={cm[i,:].sum()}\n")

np.save("reports/confusion_matrix_eval.npy", cm)
print("\nSaved reports/classifier_metrics.txt and reports/confusion_matrix_eval.npy")
