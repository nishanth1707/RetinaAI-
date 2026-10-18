import numpy as np


def predict_progression(dr_class_idx, severity_score, damage_percentage):
    dr_class_idx = int(dr_class_idx)

    base_risks = {
        0: 0.02,
        1: 0.08,
        2: 0.15,
        3: 0.35,
        4: 0.60,
    }
    base_risk = base_risks.get(dr_class_idx, 0.10)
    severity_factor = severity_score / 100.0
    damage_factor = min(damage_percentage / 20.0, 1.0)

    combined_risk = 0.5 * base_risk + 0.3 * severity_factor + 0.2 * damage_factor
    combined_risk = min(combined_risk, 0.95)

    yr1 = combined_risk * 0.4
    yr2 = combined_risk * 0.65
    yr3 = combined_risk * 0.85

    survival = [1.0, 1.0 - yr1, 1.0 - yr2, 1.0 - yr3]
    survival = [max(s, 0.05) for s in survival]

    return {
        "risk_1yr": round(yr1, 3),
        "risk_2yr": round(yr2, 3),
        "risk_3yr": round(yr3, 3),
        "overall_risk": round(combined_risk, 3),
        "survival_curve": survival,
    }


def get_progression_description(risk):
    if risk < 0.1:
        return "Low risk of progression. Regular annual screening recommended."
    elif risk < 0.3:
        return "Moderate risk of progression. Consider 6-month follow-up."
    elif risk < 0.5:
        return "High risk of progression. Recommend 3-month follow-up and tighter glycemic control."
    else:
        return "Very high risk of progression. Urgent ophthalmology referral needed. Consider pan-retinal photocoagulation."
