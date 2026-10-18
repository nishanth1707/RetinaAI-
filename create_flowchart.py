import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 18), dpi=150)
ax.set_xlim(0, 14)
ax.set_ylim(0, 18)
ax.axis("off")
fig.patch.set_facecolor("#f0f4f8")

# --- Color palette ---
C_INPUT = "#4a90d9"
C_VALID = "#e8a838"
C_MODEL = "#2d6a4f"
C_SEG = "#7b2d8b"
C_CALC = "#d62828"
C_OUTPUT = "#1a2a3a"
C_REPORT = "#0e7c7b"
WHITE = "#ffffff"
TXT = "#ffffff"


def box(x, y, w, h, text, color, fontsize=9, bold=True):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.12",
        facecolor=color, edgecolor="white", linewidth=1.5,
        zorder=3,
    )
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(
        x, y, text, ha="center", va="center",
        fontsize=fontsize, color=WHITE, fontweight=weight,
        zorder=4, linespacing=1.3,
    )


def arrow(x1, y1, x2, y2, color="#555555"):
    ax.annotate(
        "",
        xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=2,
            connectionstyle="arc3,rad=0",
        ),
        zorder=2,
    )


def diamond(x, y, w, h, text, color, fontsize=8.5):
    pts = [
        [x, y + h / 2],
        [x + w / 2, y],
        [x, y - h / 2],
        [x - w / 2, y],
    ]
    poly = plt.Polygon(pts, facecolor=color, edgecolor="white", linewidth=1.5, zorder=3)
    ax.add_patch(poly)
    ax.text(
        x, y, text, ha="center", va="center",
        fontsize=fontsize, color=WHITE, fontweight="bold",
        zorder=4, linespacing=1.2,
    )


# --- Title ---
ax.text(
    7, 17.6, "RetinaAI — System Architecture Flowchart",
    ha="center", va="center", fontsize=16, fontweight="bold",
    color="#1a2a3a",
)
ax.text(
    7, 17.15, "Diabetic Retinopathy Detection Using Deep Learning",
    ha="center", va="center", fontsize=11, color="#6b7c93",
)

# =====================  COLUMN LAYOUT  =====================
# Left col x=3.5, Right col x=10.5

# --- Step 1: Upload ---
box(7, 16.0, 5.5, 0.85, "USER UPLOADS\nFundus Image (JPG / PNG / TIFF)", C_INPUT, fontsize=10)
arrow(7, 15.55, 7, 15.1)

# --- Step 2: Validation ---
diamond(7, 14.4, 5.0, 1.4, "Input Validation\nIs it a fundus image?", C_VALID, fontsize=9)

# Reject path
arrow(4.5, 14.4, 2.0, 14.4, color="#d62828")
box(1.5, 14.4, 2.6, 0.7, "REJECT\nShow error message", "#d62828", fontsize=8)

# Pass path
arrow(7, 13.7, 7, 13.15)
ax.text(7.4, 13.4, "Yes", fontsize=8, color="#2d6a4f", fontweight="bold")

# --- Step 3: Preprocessing ---
box(7, 12.7, 5.5, 0.8, "PRE-PROCESSING\nResize / Normalise / Augment", "#5b7fa5", fontsize=9)
arrow(7, 12.25, 7, 11.75)

# --- Split into two models ---
# Left branch: Classifier
arrow(7, 11.75, 3.5, 11.75)
arrow(3.5, 11.75, 3.5, 11.2)

# Right branch: Segmenter
arrow(7, 11.75, 10.5, 11.75)
arrow(10.5, 11.75, 10.5, 11.2)

ax.text(4.8, 11.9, "Branch A", fontsize=8, color="#4a90d9", fontweight="bold", ha="center")
ax.text(9.2, 11.9, "Branch B", fontsize=8, color="#7b2d8b", fontweight="bold", ha="center")

# --- Step 4A: Classifier ---
box(3.5, 10.5, 5.0, 1.2, "CLASSIFICATION MODEL\nEfficientNet-B0 (CNN)\n5-Class DR Grading", C_MODEL, fontsize=8.5)
arrow(3.5, 9.85, 3.5, 9.3)

# --- Step 4B: Segmenter ---
box(10.5, 10.5, 5.0, 1.2, "SEGMENTATION MODEL\nU-Net + EfficientNet Encoder\n4 Lesion Types (Pixel-level)", C_SEG, fontsize=8.5)
arrow(10.5, 9.85, 10.5, 9.3)

# --- Step 5A: Classification output ---
box(3.5, 8.65, 5.0, 1.0, "Predicted DR Stage +\nConfidence + Probabilities", "#2d6a4f", fontsize=8.5)
arrow(3.5, 8.1, 3.5, 7.55)

# --- Step 5B: Segmentation output ---
box(10.5, 8.65, 5.0, 1.0, "4 Lesion Masks\n(MA / HE / EX / SE)", "#7b2d8b", fontsize=8.5)
arrow(10.5, 8.1, 10.5, 7.55)

# --- Step 6: Damage Calculation (from segmenter) ---
box(10.5, 6.9, 5.0, 1.0, "DAMAGE QUANTIFICATION\nDamage % / Quadrant Analysis", C_CALC, fontsize=8.5)
arrow(10.5, 6.35, 10.5, 5.85)

# --- Step 6B: Severity Index ---
box(10.5, 5.3, 5.0, 0.9, "SEVERITY INDEX\n0-100 Composite Score", "#c1440e", fontsize=8.5)

# --- Step 6C: Grad-CAM (from classifier) ---
box(3.5, 6.9, 5.0, 1.0, "GRAD-CAM EXPLAINABILITY\nHeatmap Overlay", "#1a5276", fontsize=8.5)
arrow(3.5, 6.35, 3.5, 5.85)

# --- Step 7: Progression ---
box(3.5, 5.3, 5.0, 0.9, "PROGRESSION FORECAST\n1-3 Year Risk + Survival Curve", "#8e44ad", fontsize=8.5)

# --- Merge both branches ---
arrow(3.5, 4.8, 3.5, 4.25)
arrow(10.5, 4.8, 10.5, 4.25)
arrow(3.5, 4.25, 7, 4.25)
arrow(10.5, 4.25, 7, 4.25)
arrow(7, 4.25, 7, 3.85)

# --- Step 8: Report ---
box(7, 3.35, 7.0, 0.9, "REPORT GENERATION\nPDF + Text Clinical Report (fpdf2)", C_REPORT, fontsize=9)
arrow(7, 2.85, 7, 2.35)

# --- Step 9: Dashboard ---
box(7, 1.8, 9.5, 1.1, "INTERACTIVE DASHBOARD (Streamlit)\nClassification  |  Segmentation  |  Damage  |  Grad-CAM  |  Progression  |  Report", C_OUTPUT, fontsize=8.5)
arrow(7, 1.2, 7, 0.8)

# --- Output ---
box(7, 0.4, 6.5, 0.65, "CLINICIAN REVIEWS RESULTS & DOWNLOADS REPORT", "#2c3e50", fontsize=9)

# --- Legend ---
legend_items = [
    mpatches.Patch(color=C_INPUT, label="User Input"),
    mpatches.Patch(color=C_VALID, label="Validation"),
    mpatches.Patch(color=C_MODEL, label="Classification (EfficientNet-B0)"),
    mpatches.Patch(color=C_SEG, label="Segmentation (U-Net)"),
    mpatches.Patch(color=C_CALC, label="Damage Analysis"),
    mpatches.Patch(color="#1a5276", label="Grad-CAM (Explainability)"),
    mpatches.Patch(color="#8e44ad", label="Progression Forecast"),
    mpatches.Patch(color=C_REPORT, label="Report Generation"),
    mpatches.Patch(color=C_OUTPUT, label="Dashboard Output"),
]
ax.legend(
    handles=legend_items,
    loc="upper left",
    bbox_to_anchor=(0.0, 0.14),
    fontsize=7.5,
    frameon=True,
    facecolor="white",
    edgecolor="#cccccc",
    ncol=3,
)

plt.tight_layout()
output_path = "poster_assets/system_flowchart.png"
import os
os.makedirs("poster_assets", exist_ok=True)
plt.savefig(output_path, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Flowchart saved to {output_path}")
