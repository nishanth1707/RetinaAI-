import numpy as np
import cv2


def compute_retina_area(mask):
    return np.sum(mask > 0)


def compute_lesion_area(segmentation_masks):
    total = 0
    per_lesion = []
    for mask in segmentation_masks:
        area = np.sum(mask > 0)
        per_lesion.append(area)
        total += area
    return total, per_lesion


def compute_damage_percentage(retina_area, total_lesion_area):
    if retina_area == 0:
        return 0.0
    return (total_lesion_area / retina_area) * 100


def divide_into_quadrants(mask):
    h, w = mask.shape
    cy, cx = h // 2, w // 2

    quadrants = {}
    qmask = np.zeros_like(mask)
    qmask[:cy, cx:] = 1
    quadrants["Superior Temporal"] = qmask
    qmask = np.zeros_like(mask)
    qmask[:cy, :cx] = 1
    quadrants["Superior Nasal"] = qmask
    qmask = np.zeros_like(mask)
    qmask[cy:, cx:] = 1
    quadrants["Inferior Temporal"] = qmask
    qmask = np.zeros_like(mask)
    qmask[cy:, :cx] = 1
    quadrants["Inferior Nasal"] = qmask
    return quadrants


def compute_quadrant_damage(segmentation_masks, retina_mask):
    retina_quadrants = divide_into_quadrants(retina_mask)
    quadrant_damage = {}
    quadrant_lesion_details = {}
    combined_lesion = np.max(segmentation_masks, axis=0) if len(segmentation_masks) > 0 else np.zeros_like(retina_mask)

    for qname, qmask in retina_quadrants.items():
        q_lesion = combined_lesion * (qmask > 0).astype(np.uint8)
        q_retina_area = np.sum(qmask > 0)
        q_lesion_area = np.sum(q_lesion > 0)
        if q_retina_area > 0:
            damage_pct = (q_lesion_area / q_retina_area) * 100
        else:
            damage_pct = 0.0
        quadrant_damage[qname] = round(damage_pct, 2)
        quadrant_lesion_details[qname] = q_lesion_area

    most_affected = max(quadrant_damage, key=quadrant_damage.get)
    return quadrant_damage, most_affected


def create_quadrant_map_visual(retina_mask, segmentation_masks, original_image=None):
    h, w = retina_mask.shape
    cy, cx = h // 2, w // 2

    if original_image is not None:
        vis = original_image.copy()
    else:
        vis = np.zeros((h, w, 3), dtype=np.uint8)

    combined_lesion = np.max(segmentation_masks, axis=0) if len(segmentation_masks) > 0 else np.zeros((h, w), dtype=np.uint8)

    overlay = vis.copy()
    colored = np.zeros_like(vis)
    colored[combined_lesion > 0] = (255, 0, 0)
    cv2.addWeighted(colored, 0.5, overlay, 0.5, 0, overlay)

    cv2.line(overlay, (cx, 0), (cx, h), (255, 255, 255), 2)
    cv2.line(overlay, (0, cy), (w, cy), (255, 255, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    labels = {
        (cx // 4, cy // 4): "Superior Nasal",
        (3 * cx // 2, cy // 4): "Superior Temporal",
        (cx // 4, 3 * cy // 2): "Inferior Nasal",
        (3 * cx // 2, 3 * cy // 2): "Inferior Temporal",
    }

    for (x, y), label in labels.items():
        if x < w and y < h:
            cv2.putText(overlay, label, (x, y), font, 0.4, (255, 255, 255), 1)

    return overlay
