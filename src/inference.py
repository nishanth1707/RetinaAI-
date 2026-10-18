import torch
import numpy as np
import cv2
from pathlib import Path

from .config import (
    DR_CLASSES, LESION_NAMES, LESION_COLORS,
    IMG_SIZE_CLASSIFIER, IMG_SIZE_SEGMENTER,
)
from .utils import (
    preprocess_image, preprocess_image_segmentation,
    create_retina_mask, overlay_multiple_masks, resize_to_original,
    is_fundus_image,
)
from .model_classifier import load_classifier, predict_classifier
from .model_segmenter import load_segmenter, predict_segmenter
from .damage_calculator import (
    compute_retina_area, compute_lesion_area,
    compute_damage_percentage, compute_quadrant_damage,
    create_quadrant_map_visual,
)
from .severity_index import compute_severity_index
from .grad_cam import GradCAM, apply_gradcam_heatmap
from .progression_predictor import predict_progression


class DRInference:
    def __init__(self, classifier_path, segmenter_path, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading models on {self.device}...")
        self.classifier = load_classifier(classifier_path, device=self.device)
        self.segmenter = load_segmenter(segmenter_path, device=self.device)
        self.gradcam = GradCAM(self.classifier)
        print("Models loaded successfully.")

    def analyze(self, image_path):
        results = {}
        image_path = Path(image_path)

        cls_tensor, pil_img = preprocess_image(image_path, IMG_SIZE_CLASSIFIER)
        seg_tensor, orig_img, orig_size = preprocess_image_segmentation(image_path, IMG_SIZE_SEGMENTER)

        if not is_fundus_image(orig_img):
            raise ValueError(
                "The image does not appear to be a retinal fundus image. "
                "Please upload a fundus/retina photo."
            )

        dr_class_idx, confidence, class_probs = predict_classifier(
            self.classifier, cls_tensor, self.device
        )
        results["dr_class"] = DR_CLASSES[dr_class_idx]
        results["dr_class_idx"] = dr_class_idx
        results["confidence"] = confidence
        results["class_probs"] = class_probs

        seg_masks = predict_segmenter(self.segmenter, seg_tensor, self.device)
        seg_masks_resized = np.array([resize_to_original(m, orig_size) for m in seg_masks])

        retina_mask = create_retina_mask(orig_img)
        retina_area = compute_retina_area(retina_mask)

        total_lesion_area, per_lesion_areas = compute_lesion_area(seg_masks_resized)
        damage_pct = compute_damage_percentage(retina_area, total_lesion_area)

        results["retina_area"] = retina_area
        results["total_lesion_area"] = total_lesion_area
        results["damage_percentage"] = damage_pct
        results["lesion_areas"] = per_lesion_areas
        results["lesion_names"] = LESION_NAMES

        quadrant_damage, most_affected = compute_quadrant_damage(seg_masks_resized, retina_mask)
        results["quadrant_damage"] = quadrant_damage
        results["most_affected_quadrant"] = most_affected

        severity = compute_severity_index(damage_pct, class_probs)
        results["severity_score"] = severity["score"]
        results["severity_label"] = severity["label"]
        results["severity_icon"] = severity["icon"]
        results["severity_level"] = severity["level"]

        progression = predict_progression(dr_class_idx, severity["score"], damage_pct)
        results["progression"] = progression
        results["progression_risk"] = progression["overall_risk"]

        seg_overlay = overlay_multiple_masks(orig_img, seg_masks_resized, LESION_COLORS, alpha=0.4)
        results["segmentation_image"] = seg_overlay

        cam = self.gradcam.generate(cls_tensor.to(self.device), class_idx=dr_class_idx)
        orig_for_cam = cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR)
        orig_for_cam = cv2.resize(orig_for_cam, (IMG_SIZE_CLASSIFIER, IMG_SIZE_CLASSIFIER))
        orig_for_cam = cv2.cvtColor(orig_for_cam, cv2.COLOR_BGR2RGB)
        heatmap_overlay, heatmap_raw = apply_gradcam_heatmap(orig_for_cam, cam)
        results["heatmap_image"] = heatmap_overlay
        results["heatmap_raw"] = heatmap_raw

        quadrant_map = create_quadrant_map_visual(retina_mask, seg_masks_resized, orig_img)
        results["quadrant_map"] = quadrant_map
        results["retina_mask"] = retina_mask
        results["orig_image"] = orig_img

        return results
