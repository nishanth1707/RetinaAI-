import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

os.makedirs("poster_assets", exist_ok=True)
OUT = Path("poster_assets")

# --- Figure 1: Confusion matrix ---
confusion = np.load("models/confusion_matrix.npy")
labels = ["No DR", "Mild", "Moderate", "Severe", "PDR"]
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(confusion, cmap="Blues")
ax.set_xticks(range(5)); ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticks(range(5)); ax.set_yticklabels(labels)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix - DR Classification", fontweight="bold")
for i in range(5):
    for j in range(5):
        ax.text(j, i, str(confusion[i, j]), ha="center", va="center",
                color="white" if confusion[i, j] > confusion.max()/2 else "black")
fig.colorbar(im, shrink=0.8)
plt.tight_layout()
fig.savefig(OUT / "confusion_matrix.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved confusion_matrix.png")

# --- Figure 2: Sample fundus + segmentation overlay ---
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.model_segmenter import DRSegmenter
from src.utils import preprocess_image_segmentation, resize_to_original, overlay_multiple_masks
from src.config import SEGMENTER_PATH, IMG_SIZE_SEGMENTER, LESION_NAMES, LESION_COLORS

device = "cuda" if torch.cuda.is_available() else "cpu"
seg = DRSegmenter(num_lesions=4)
st_seg = torch.load(str(SEGMENTER_PATH), map_location=device, weights_only=True)
seg.load_state_dict(st_seg["model_state_dict"])
seg.to(device); seg.eval()

# pick a few test images with real lesions
test_dir = Path("data/ddr_seg/lesion_segmentation/images/train")
img = test_dir / "007-1774-100.jpg"
tensor, orig, orig_size = preprocess_image_segmentation(img, IMG_SIZE_SEGMENTER)
with torch.no_grad():
    out = torch.sigmoid(seg(tensor.to(device)))[0].cpu().numpy()
masks = [resize_to_original((out[i] > 0.3).astype(np.uint8), orig_size) for i in range(4)]
overlay = overlay_multiple_masks(orig, masks, LESION_COLORS, alpha=0.4)

fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
axs[0].imshow(orig); axs[0].set_title("Original Fundus", fontweight="bold"); axs[0].axis("off")
axs[1].imshow(overlay); axs[1].set_title("Lesion Segmentation Overlay", fontweight="bold"); axs[1].axis("off")
plt.tight_layout()
fig.savefig(OUT / "segmentation.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved segmentation.png")

# --- Figure 3: Grad-CAM heatmap ---
from src.model_classifier import DRClassifier, load_classifier
from src.grad_cam import GradCAM, apply_gradcam_heatmap
from src.config import CLASSIFIER_PATH, IMG_SIZE_CLASSIFIER
from src.utils import preprocess_image

cls_model = load_classifier(CLASSIFIER_PATH, device=device)
gradcam = GradCAM(cls_model)
cls_tensor, pil_img = preprocess_image(img, IMG_SIZE_CLASSIFIER)
cam = gradcam.generate(cls_tensor.to(device), class_idx=None)
orig_for_cam = cv2.cvtColor(orig, cv2.COLOR_RGB2BGR)
orig_for_cam = cv2.resize(orig_for_cam, (IMG_SIZE_CLASSIFIER, IMG_SIZE_CLASSIFIER))
orig_for_cam = cv2.cvtColor(orig_for_cam, cv2.COLOR_BGR2RGB)
heatmap_overlay, heatmap_raw = apply_gradcam_heatmap(orig_for_cam, cam)

fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
axs[0].imshow(orig_for_cam); axs[0].set_title("Original (224x224)", fontweight="bold"); axs[0].axis("off")
axs[1].imshow(heatmap_overlay); axs[1].set_title("Grad-CAM Heatmap", fontweight="bold"); axs[1].axis("off")
plt.tight_layout()
fig.savefig(OUT / "gradcam.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved gradcam.png")

# --- Figure 4: Results summary bar chart ---
fig, ax = plt.subplots(figsize=(7, 4))
cls_scores = {"Accuracy": 81.97, "Macro AUC-ROC": 89.21}
ax.bar(list(cls_scores.keys()), list(cls_scores.values()), color=["#2a9d8f", "#264653"])
ax.set_ylim(0, 100); ax.set_ylabel("Percentage (%)")
ax.set_title("Classification Performance", fontweight="bold")
for i, v in enumerate(cls_scores.values()):
    ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "classification_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved classification_results.png")

# --- Figure 5: Per-lesion Dice bar chart ---
lesion_names = ["Micro-\naneurysms", "Hemorr-\nhages", "Hard\nExudates", "CWS"]
dice = [0.18, 0.30, 0.10, 0.32]
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(lesion_names, dice, color=["#e63946", "#f4a261", "#2a9d8f", "#e9c46a"])
ax.set_ylim(0, 0.5); ax.set_ylabel("Dice Coefficient")
ax.set_title("Segmentation Performance by Lesion Type", fontweight="bold")
for b, v in zip(bars, dice):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}", ha="center", fontweight="bold")
ax.axhline(0.224, color="black", linestyle="--", linewidth=1.5, label="Average = 0.224")
ax.legend()
plt.tight_layout()
fig.savefig(OUT / "segmentation_results.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved segmentation_results.png")

print("\nAll poster assets saved to:", OUT.resolve())
