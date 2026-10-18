import os
from datetime import datetime
from fpdf import FPDF
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO


class DRReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "Diabetic Retinopathy Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def _sanitize(text):
    if isinstance(text, (np.integer, np.floating)):
        text = str(text)
    return str(text).encode("latin-1", errors="replace").decode("latin-1")


def generate_report(results, output_path):
    pdf = DRReport()
    pdf.alias_nb_pages()
    pdf.add_page()

    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "1. Classification Result", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Diagnosis: {results['dr_class']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"Confidence: {results['confidence']*100:.1f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "2. Retinal Damage Assessment", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Estimated Retinal Damage: {results['damage_percentage']:.1f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"Most Affected Region: {results['most_affected_quadrant']}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "3. Quadrant-wise Damage", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    for qname, dmg in results['quadrant_damage'].items():
        pdf.cell(0, 10, f"  {qname}: {dmg:.1f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "4. DR Severity Index", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Severity: {results['severity_label']}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 10, f"Severity Score: {results['severity_score']}/100", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "5. Lesion Analysis", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    for i, (lname, area) in enumerate(zip(results['lesion_names'], results['lesion_areas'])):
        pct = (area / max(results['retina_area'], 1)) * 100
        if pct > 0.1:
            pdf.cell(0, 10, f"  {lname}: {pct:.2f}% of retina", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "6. Progression Risk (1-3 Years)", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Risk of progression to PDR: {results['progression_risk']*100:.1f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 8, "Disclaimer: This report is AI-generated and should be verified by a qualified ophthalmologist. It is not a substitute for professional medical advice.")
    pdf.ln(5)

    for img_key in ['segmentation_image', 'heatmap_image']:
        if img_key in results and results[img_key] is not None:
            try:
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 14)
                pdf.cell(0, 10, img_key.replace("_", " ").title(), new_x="LMARGIN", new_y="NEXT")
                img_path = f"temp_{img_key}.png"
                plt.imsave(img_path, results[img_key])
                pdf.image(img_path, x=20, w=pdf.w - 40)
                os.remove(img_path)
            except:
                pass

    pdf.output(str(output_path))
    return output_path


def generate_report_text(results):
    lines = []
    lines.append("=" * 60)
    lines.append("DIABETIC RETINOPATHY ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append(f"Diagnosis: {results['dr_class']}")
    lines.append(f"Confidence: {results['confidence']*100:.1f}%")
    lines.append("")
    lines.append(f"Retinal Damage: {results['damage_percentage']:.1f}%")
    lines.append(f"Most Affected Region: {results['most_affected_quadrant']}")
    lines.append("")
    lines.append("Quadrant-wise Damage:")
    for qname, dmg in results['quadrant_damage'].items():
        lines.append(f"  {qname}: {dmg:.1f}%")
    lines.append("")
    lines.append(f"Severity: {results['severity_icon']} {results['severity_label']} ({results['severity_score']}/100)")
    lines.append(f"Progression Risk (1-3 yr): {results['progression_risk']*100:.1f}%")
    lines.append("")
    lines.append("-" * 60)
    lines.append("Lesion Breakdown:")
    for i, (lname, area) in enumerate(zip(results['lesion_names'], results['lesion_areas'])):
        pct = (area / max(results['retina_area'], 1)) * 100
        if pct > 0.1:
            lines.append(f"  {lname}: {pct:.2f}% of retina")
    lines.append("=" * 60)
    lines.append("Disclaimer: AI-generated report. Verify with ophthalmologist.")
    return "\n".join(lines)
