import numpy as np


def compute_severity_index(damage_percentage, class_probs=None):
    if class_probs is not None:
        weighted = np.sum(class_probs * np.array([0, 25, 50, 75, 100]))
        severity_score = 0.3 * damage_percentage + 0.7 * weighted
    else:
        severity_score = damage_percentage

    severity_score = min(severity_score, 100.0)

    if severity_score < 30:
        level = 0
        label = "Low Risk"
        icon = "🟢"
    elif severity_score < 70:
        level = 1
        label = "Moderate Risk"
        icon = "🟡"
    else:
        level = 2
        label = "High Risk"
        icon = "🔴"

    return {
        "score": round(severity_score, 2),
        "level": level,
        "label": label,
        "icon": icon,
        "damage_contribution": round(damage_percentage, 2),
        "class_contribution": round(weighted if class_probs is not None else 0, 2),
    }


def get_severity_color(level):
    colors = [(0, 200, 0), (200, 200, 0), (200, 0, 0)]
    return colors[level]
