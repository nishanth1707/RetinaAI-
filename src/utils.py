import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


def preprocess_image(image_path, size=224):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor, img


def preprocess_image_segmentation(image_path, size=512):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_size = img.shape[:2]
    img_resized = cv2.resize(img, (size, size))
    img_normalized = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_normalized - mean) / std
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor, img, original_size


def create_retina_mask(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [largest], -1, 255, -1)
    return mask


def is_fundus_image(image):
    """Heuristic check whether an image looks like a retinal fundus photo."""
    if image is None or len(image.shape) != 3 or min(image.shape[:2]) < 50:
        return False

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    coverage = area / (h * w)
    perimeter = cv2.arcLength(largest, True)
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0

    r_mean = float(image[:, :, 0].mean())
    b_mean = float(image[:, :, 2].mean())
    red_ratio = r_mean / max(b_mean, 1.0)

    return (0.20 <= coverage <= 1.0) and circularity > 0.65 and red_ratio > 1.4


def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.5):
    overlay = image.copy()
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay


def overlay_multiple_masks(image, masks, colors, alpha=0.4):
    overlay = image.copy()
    for mask, color in zip(masks, colors):
        if mask is not None and mask.max() > 0:
            colored = np.zeros_like(image)
            colored[mask > 0] = color
            cv2.addWeighted(colored, alpha, overlay, 1 - alpha, 0, overlay)
    return overlay


def resize_to_original(pred_mask, original_size):
    h, w = original_size
    return cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)


def fig_to_image(fig):
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img
