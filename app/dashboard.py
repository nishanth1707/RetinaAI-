import streamlit as st
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import base64
import tempfile
import os
import json
from datetime import datetime

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import (
    DR_CLASSES, LESION_NAMES, LESION_COLORS, QUADRANT_NAMES,
    CLASSIFIER_PATH, SEGMENTER_PATH,
)
from src.inference import DRInference
from src.report_generator import generate_report, generate_report_text
from src.damage_calculator import compute_retina_area
from src.utils import is_fundus_image

st.set_page_config(
    page_title="RetinaAI - Diabetic Retinopathy Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROFESSIONAL_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f0 100%);
    }

    .main-header {
        background: linear-gradient(135deg, #0f1923 0%, #1a2a3a 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
        color: white;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        font-size: 0.9rem;
        opacity: 0.8;
        margin: 0.3rem 0 0 0;
        font-weight: 300;
    }

    .module-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #e8edf2;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 0.8rem;
    }
    .module-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .module-card h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #1a2a3a;
        margin: 0 0 0.4rem 0;
    }
    .module-card p {
        font-size: 0.85rem;
        color: #6b7c93;
        margin: 0;
    }

    .metric-box {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        border: 1px solid #e8edf2;
    }
    .metric-box .label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #6b7c93;
        font-weight: 500;
    }
    .metric-box .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a2a3a;
    }

    .severity-indicator {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .severity-low { background: #e6f7e6; color: #1a7d1a; }
    .severity-moderate { background: #fff3e0; color: #cc7a00; }
    .severity-high { background: #ffe6e6; color: #cc0000; }

    .result-card {
        background: white;
        border-radius: 14px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        border: 1px solid #e8edf2;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #e8edf2;
        border-radius: 10px;
        padding: 3px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
        font-size: 0.85rem;
    }
    .stTabs [aria-selected="true"] {
        background: white;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.2s;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1a2a3a, #2c3e50);
        border: none;
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(26,42,58,0.3);
    }

    .report-text {
        font-family: 'Inter', monospace;
        white-space: pre-wrap;
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e8edf2;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .upload-area {
        border: 2px dashed #c0ccda;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        background: white;
        transition: all 0.3s;
    }
    .upload-area:hover {
        border-color: #1a2a3a;
        background: #f8f9fa;
    }

    .sidebar-header {
        background: linear-gradient(135deg, #0f1923, #1a2a3a);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sidebar-header h2 {
        font-size: 1.2rem;
        font-weight: 700;
        margin: 0;
    }
    .sidebar-header p {
        font-size: 0.75rem;
        opacity: 0.7;
        margin: 0.2rem 0 0 0;
    }

    @media (max-width: 768px) {
        .main-header h1 { font-size: 1.4rem; }
        .metric-box .value { font-size: 1.2rem; }
    }

    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-blue { background: #e3ecfa; color: #1a56db; }
    .badge-green { background: #e6f7e6; color: #1a7d1a; }
    .badge-orange { background: #fff3e0; color: #cc7a00; }
    .badge-red { background: #ffe6e6; color: #cc0000; }

    .progress-bar-container {
        background: #e8edf2;
        border-radius: 8px;
        height: 8px;
        overflow: hidden;
    }
    .progress-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.6s ease;
    }
</style>
"""

st.markdown(PROFESSIONAL_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_models():
    if not CLASSIFIER_PATH.exists():
        return None
    if not SEGMENTER_PATH.exists():
        return None
    return DRInference(CLASSIFIER_PATH, SEGMENTER_PATH)


def plot_gauge(value, title, max_val=100, color_ranges=None):
    if color_ranges is None:
        color_ranges = [(0, 30, "green"), (30, 70, "orange"), (70, 100, "red")]
    steps = []
    for lo, hi, clr in color_ranges:
        steps.append({"range": [lo, hi], "color": clr})

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=min(value, max_val),
        title={"text": title, "font": {"size": 14, "color": "#4a5568"}},
        number={"font": {"size": 28, "color": "#1a2a3a"}, "suffix": "%" if "Damage" in title else ""},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, max_val], "tickwidth": 1, "tickcolor": "#c0ccda"},
            "bar": {"color": "#1a2a3a", "thickness": 0.3},
            "bgcolor": "#f8f9fa",
            "borderwidth": 0,
            "steps": steps,
            "threshold": {
                "line": {"color": "#1a2a3a", "width": 3},
                "thickness": 0.7,
                "value": min(value, max_val),
            },
        },
    ))
    fig.update_layout(
        height=220, margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#4a5568"}
    )
    return fig


def plot_bar_chart(x, y, title, xlabel, ylabel, colors=None):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=y, marker_color=colors or "#1a2a3a",
        text=[f"{v:.1f}" if isinstance(v, float) else str(v) for v in y],
        textposition="outside", textfont={"size": 11}
    ))
    fig.update_layout(
        title=title, xaxis_title=xlabel, yaxis_title=ylabel,
        height=280, margin=dict(l=10, r=10, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#4a5568", "size": 11},
        xaxis={"gridcolor": "#e8edf2", "tickangle": -20},
        yaxis={"gridcolor": "#e8edf2"},
    )
    return fig


def plot_line_chart(x, y_sets, title, xlabel, ylabel):
    fig = go.Figure()
    for name, y, color in y_sets:
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers", name=name,
            line=dict(color=color, width=3),
            marker=dict(size=9)
        ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#1a2a3a")),
        xaxis_title=dict(text=xlabel, font=dict(size=13, color="#1a2a3a")),
        yaxis_title=dict(text=ylabel, font=dict(size=13, color="#1a2a3a")),
        height=320, margin=dict(l=40, r=20, t=60, b=50),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#4a5568", "size": 12},
        xaxis=dict(gridcolor="#e8edf2"),
        yaxis=dict(
            gridcolor="#e8edf2", range=[0, 1],
            tickformat=".0%", title_standoff=10,
        ),
        legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center",
                    bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    return fig


def plot_pie_chart(values, labels, title, colors):
    fig = go.Figure(go.Pie(
        values=values, labels=labels, marker=dict(colors=colors),
        hole=0.45, textinfo="label+percent", textfont={"size": 11}
    ))
    fig.update_layout(
        title=title, height=280, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#4a5568", "size": 11},
        showlegend=False,
    )
    return fig


def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>RetinaAI · Diabetic Retinopathy Detection System</h1>
        <p>AI-Powered Fundus Analysis &mdash; Classification &middot; Segmentation &middot; Grad-CAM &middot; Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <h2>RetinaAI</h2>
            <p>v2.0 · DR Detection System</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Analysis Pipeline")
        st.markdown("""
        <div class="module-card">
            <h3>1 &nbsp; Classification</h3>
            <p>5-stage DR grading (No DR → PDR)</p>
        </div>
        <div class="module-card">
            <h3>2 &nbsp; Segmentation</h3>
            <p>4 lesion types identified</p>
        </div>
        <div class="module-card">
            <h3>3 &nbsp; Damage Analysis</h3>
            <p>Retina damage % + quadrant mapping</p>
        </div>
        <div class="module-card">
            <h3>4 &nbsp; Severity Index</h3>
            <p>Risk scoring with confidence</p>
        </div>
        <div class="module-card">
            <h3>5 &nbsp; Explainable AI</h3>
            <p>Grad-CAM heatmap visualization</p>
        </div>
        <div class="module-card">
            <h3>6 &nbsp; Progression Risk</h3>
            <p>1-3 year PDR prediction</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        operation_mode = st.radio(
            "Operation Mode",
            ["Demo (Simulated)", "Local (Trained Models)"],
            index=0,
            help="Demo mode works immediately. Local mode requires trained models.",
        )

        st.markdown("---")
        st.markdown(
            "<p style='text-align:center;font-size:0.75rem;color:#6b7c93;'>"
            "RetinaAI Screening System</p>",
            unsafe_allow_html=True,
        )

    return operation_mode


def render_upload_section():
    st.markdown('<div class="upload-area">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a retinal fundus image",
        type=["jpg", "jpeg", "png", "tif", "bmp"],
        label_visibility="collapsed",
    )
    if uploaded is None:
        st.markdown(
            "<p style='color:#6b7c93;margin-top:0.5rem;'>"
            "Supported formats: JPG, PNG, TIFF, BMP</p>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)
    return uploaded


def simulate_results(image):
    h, w = image.shape[:2]
    results = {
        "dr_class": "Moderate NPDR",
        "dr_class_idx": 2,
        "confidence": 0.912,
        "class_probs": np.array([0.008, 0.032, 0.912, 0.035, 0.013]),
        "retina_area": h * w,
        "total_lesion_area": int(h * w * 0.063),
        "damage_percentage": 6.3,
        "lesion_areas": [int(h * w * 0.008), int(h * w * 0.025), int(h * w * 0.028), int(h * w * 0.002)],
        "lesion_names": LESION_NAMES,
        "quadrant_damage": {
            "Superior Temporal": 8.9,
            "Superior Nasal": 3.2,
            "Inferior Temporal": 7.1,
            "Inferior Nasal": 4.8,
        },
        "most_affected_quadrant": "Superior Temporal",
        "severity_score": 42.5,
        "severity_label": "Moderate Risk",
        "severity_icon": "🟡",
        "severity_level": 1,
        "progression": {
            "risk_1yr": 0.12,
            "risk_2yr": 0.28,
            "risk_3yr": 0.45,
            "overall_risk": 0.32,
            "survival_curve": [1.0, 0.88, 0.72, 0.55],
        },
        "progression_risk": 0.32,
        "orig_image": image,
        "retina_mask": np.ones((h, w), dtype=np.uint8) * 255,
        "segmentation_image": image.copy(),
        "heatmap_image": image.copy(),
        "heatmap_raw": np.zeros((224, 224), dtype=np.float32),
        "quadrant_map": image.copy(),
    }
    return results


def render_classification_tab(results):
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.5])

    with col1:
        severity = results["severity_label"]
        sev_class = severity.lower().replace(" ", "-")
        icon = results["severity_icon"]

        st.markdown(f"""
        <div style="margin-bottom:1.5rem;">
            <div style="font-size:0.8rem;color:#6b7c93;text-transform:uppercase;letter-spacing:0.5px;">Diagnosis</div>
            <div style="font-size:1.8rem;font-weight:700;color:#1a2a3a;">{results['dr_class']}</div>
        </div>
        <div style="margin-bottom:1.5rem;">
            <div style="font-size:0.8rem;color:#6b7c93;text-transform:uppercase;letter-spacing:0.5px;">Confidence</div>
            <div style="font-size:1.8rem;font-weight:700;color:#1a2a3a;">{results['confidence']*100:.1f}%</div>
        </div>
        <div>
            <div style="font-size:0.8rem;color:#6b7c93;text-transform:uppercase;letter-spacing:0.5px;">Severity Index</div>
            <div style="margin-top:0.4rem;">
                <span class="severity-indicator severity-{sev_class}">
                    {icon} {severity}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        probs = results["class_probs"] * 100
        colors = ["#2d6a4f", "#52b788", "#f4a261", "#e76f51", "#d62828"]
        fig = plot_bar_chart(
            DR_CLASSES, probs,
            "Class Probability Distribution", "DR Stage", "Probability (%)",
            colors=[colors[i] for i in range(5)],
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
    cols = st.columns(5)
    class_colors = ["#2d6a4f", "#52b788", "#f4a261", "#e76f51", "#d62828"]
    class_bg = ["#e6f7e6", "#edfaf0", "#fff8f0", "#fef2f0", "#fef0f0"]
    for i, (dr_class, color, bg) in enumerate(zip(DR_CLASSES, class_colors, class_bg)):
        prob = results["class_probs"][i] * 100
        is_selected = i == results["dr_class_idx"]
        border = f"2px solid {color}" if is_selected else "1px solid transparent"
        with cols[i]:
            st.markdown(f"""
            <div style="text-align:center;padding:0.8rem 0.4rem;border-radius:10px;
                        background:{bg};border:{border};">
                <div style="font-size:0.7rem;color:#4a5568;font-weight:500;">{dr_class}</div>
                <div style="font-size:1.4rem;font-weight:700;color:{color};">{prob:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_segmentation_tab(results):
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        orig = results["orig_image"]
        st.image(orig, caption="Original Fundus Image", use_container_width=True)

    with col2:
        seg = results["segmentation_image"]
        st.image(seg, caption="Lesion Segmentation Overlay", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
    st.markdown("### Lesion Breakdown")
    cols = st.columns(4)
    for i, (lname, color) in enumerate(zip(LESION_NAMES, LESION_COLORS)):
        r, g, b = color
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        with cols[i]:
            st.markdown(f"""
            <div style="text-align:center;padding:1rem;background:white;border-radius:10px;
                        border:1px solid #e8edf2;">
                <div style="width:24px;height:24px;background:{hex_color};border-radius:6px;margin:0 auto 0.5rem;"></div>
                <div style="font-weight:600;font-size:0.85rem;">{lname}</div>
                <div style="font-size:0.8rem;color:#6b7c93;">{results['lesion_areas'][i]:,} px</div>
            </div>
            """, unsafe_allow_html=True)

    fig = plot_pie_chart(
        [max(a, 0) for a in results["lesion_areas"]],
        results["lesion_names"],
        "Lesion Composition", ["#ff0000", "#00ff00", "#ffff00", "#0000ff"]
    )
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_damage_tab(results):
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        fig1 = plot_gauge(results["damage_percentage"], "Retinal Damage")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = plot_gauge(
            results["severity_score"],
            "DR Severity Index",
            color_ranges=[(0, 30, "#2d6a4f"), (30, 70, "#f4a261"), (70, 100, "#d62828")],
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Retinal Damage</div>
            <div class="value">{results['damage_percentage']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Most Affected Quadrant</div>
            <div class="value" style="font-size:1rem;">{results['most_affected_quadrant']}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        sev = results["severity_label"]
        sev_class = sev.lower().replace(" ", "-")
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Severity Score</div>
            <div class="value">{results['severity_score']:.1f} <span style="font-size:0.8rem;">/ 100</span></div>
            <div style="margin-top:0.3rem;"><span class="severity-indicator severity-{sev_class}">{sev}</span></div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
    st.markdown("### Quadrant-wise Analysis")
    col1, col2 = st.columns(2)

    with col1:
        qnames = list(results["quadrant_damage"].keys())
        qvalues = list(results["quadrant_damage"].values())
        colors = ["#ff6b6b" if v == max(qvalues) else "#4ecdc4" for v in qvalues]
        fig = plot_bar_chart(qnames, qvalues, "", "Quadrant", "Damage (%)", colors)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        if "quadrant_map" in results:
            st.image(results["quadrant_map"], caption="Quadrant Map", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
    cols = st.columns(4)
    for i, (qname, dmg) in enumerate(results["quadrant_damage"].items()):
        is_highest = qname == results["most_affected_quadrant"]
        border = "2px solid #d62828" if is_highest else "1px solid #e8edf2"
        bg = "#fff5f5" if is_highest else "white"
        with cols[i]:
            st.markdown(f"""
            <div style="text-align:center;padding:1rem;border:{border};border-radius:10px;background:{bg};">
                <div style="font-size:0.8rem;color:#4a5568;font-weight:500;">{qname}</div>
                <div style="font-size:2rem;font-weight:700;color:#1a2a3a;">{dmg:.1f}%</div>
                <div style="font-size:0.7rem;color:#d62828;">{'&#9650; Most affected' if is_highest else ''}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_gradcam_tab(results):
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.image(results["orig_image"], caption="Original Fundus Image", use_container_width=True)

    with col2:
        st.image(results["heatmap_image"], caption="Grad-CAM Heatmap Overlay", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
    st.markdown("### How to Interpret")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div style="background:white;border-radius:10px;padding:1rem;border:1px solid #e8edf2;text-align:center;">
            <div style="font-size:2rem;">&#128308;</div>
            <div style="font-weight:600;font-size:0.9rem;">High Focus</div>
            <div style="font-size:0.8rem;color:#6b7c93;">Red/yellow regions the model focuses on</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:white;border-radius:10px;padding:1rem;border:1px solid #e8edf2;text-align:center;">
            <div style="font-size:2rem;">&#128992;</div>
            <div style="font-weight:600;font-size:0.9rem;">Lesion Areas</div>
            <div style="font-size:0.8rem;color:#6b7c93;">Heatmap highlights lesions & exudates</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div style="background:white;border-radius:10px;padding:1rem;border:1px solid #e8edf2;text-align:center;">
            <div style="font-size:2rem;">&#128994;</div>
            <div style="font-weight:600;font-size:0.9rem;">Low Focus</div>
            <div style="font-size:0.8rem;color:#6b7c93;">Blue/green = less relevant areas</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
    st.markdown("### Addressing Trust & Transparency")
    st.markdown("""
    | Challenge | Solution |
    |---|---|
    | Black-box AI not trusted by clinicians | Grad-CAM provides **visual explanation** of model decisions |
    | Patients skeptical of automated diagnosis | Heatmaps show exactly **which regions** influenced the prediction |
    | Need for clinical validation | Model focus areas can be **verified by ophthalmologists** |
    """)
    st.markdown("</div>", unsafe_allow_html=True)


def render_progression_tab(results):
    prog = results["progression"]

    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">1-Year Risk</div>
            <div class="value" style="color:#2d6a4f;">{prog['risk_1yr']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">2-Year Risk</div>
            <div class="value" style="color:#f4a261;">{prog['risk_2yr']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">3-Year Risk</div>
            <div class="value" style="color:#d62828;">{prog['risk_3yr']*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
    years = ["Current", "Year 1", "Year 2", "Year 3"]
    risk_vals = [0, prog["risk_1yr"], prog["risk_2yr"], prog["risk_3yr"]]
    survival = prog["survival_curve"]

    fig = plot_line_chart(
        years,
        [("Progression Risk", risk_vals, "#d62828"), ("Survival Probability", survival, "#2d6a4f")],
        "Disease Progression Prediction", "Time", "Probability"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem;'>", unsafe_allow_html=True)
    risk = prog["overall_risk"]
    if risk < 0.1:
        msg = "Low risk of progression. Regular annual screening recommended."
        st.success(msg)
    elif risk < 0.3:
        msg = "Moderate risk of progression. Consider 6-month follow-up."
        st.info(msg)
    elif risk < 0.5:
        msg = "High risk of progression. Recommend 3-month follow-up and tighter glycemic control."
        st.warning(msg)
    else:
        msg = "Very high risk of progression. Urgent ophthalmology referral needed."
        st.error(msg)
    st.markdown("</div>", unsafe_allow_html=True)


def render_report_tab(results):
    report_text = generate_report_text(results)

    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    st.markdown("### Automated Clinical Report")
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;font-size:0.8rem;"
        f"color:#6b7c93;margin-bottom:0.75rem;'>"
        f"<span>Generated: {datetime.now().strftime('%d %b %Y, %I:%M %p')}</span>"
        f"<span>Patient ID: —</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="report-text">{report_text}</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        with st.spinner("Generating PDF report..."):
            try:
                fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
                os.close(fd)
                try:
                    results_for_pdf = results.copy()
                    for img_key in ("segmentation_image", "heatmap_image"):
                        if img_key not in results_for_pdf:
                            results_for_pdf[img_key] = results.get(img_key)
                    generate_report(results_for_pdf, tmp_path)
                    with open(tmp_path, "rb") as f:
                        pdf_bytes = f.read()
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                st.success("PDF report ready.")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")
                pdf_bytes = None

        if pdf_bytes is not None:
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="RetinaAI_DR_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with col2:
        st.download_button(
            label="Download Text Report",
            data=report_text.encode(),
            file_name="RetinaAI_DR_Report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.markdown(
        "<div style='margin-top:1rem;padding:0.75rem 1rem;background:#fff7e6;"
        "border-left:4px solid #f0ad4e;border-radius:6px;font-size:0.8rem;color:#6b5b1e;'>"
        "<b>Important Notice:</b> This report is AI-assisted and intended for screening "
        "support only. All findings must be reviewed and confirmed by a qualified "
        "ophthalmologist. It does not replace professional medical diagnosis.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    render_header()
    operation_mode = render_sidebar()

    uploaded_file = st.file_uploader(
        "Upload Fundus Image",
        type=["jpg", "jpeg", "png", "tif", "bmp"],
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown('<div class="upload-area">', unsafe_allow_html=True)
        st.markdown("""
        <div style="padding:2rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">&#128065;</div>
            <h3 style="color:#1a2a3a;margin:0 0 0.5rem 0;">Upload a Retinal Fundus Image</h3>
            <p style="color:#6b7c93;margin:0;">Drag & drop or click to browse</p>
            <p style="color:#6b7c93;font-size:0.8rem;margin-top:0.5rem;">
                Supported: JPG, PNG, TIFF, BMP
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:2rem;">
            <h3 style="color:#1a2a3a;">System Modules</h3>
        </div>
        """, unsafe_allow_html=True)

        cols = st.columns(3)
        modules = [
            ("DR Classification", "5-stage detection: No DR, Mild, Moderate, Severe, Proliferative DR", "1"),
            ("Lesion Segmentation", "Pixel-level identification of Microaneurysms, Hemorrhages, Hard Exudates, CWS", "2"),
            ("Damage Estimation", "Retinal damage percentage with quadrant-wise lesion density mapping", "3"),
            ("Severity Index", "Risk scoring with confidence levels and clinical recommendations", "4"),
            ("Explainable AI", "Grad-CAM heatmaps highlighting model focus areas for transparency", "5"),
            ("Progression Risk", "1-3 year PDR progression prediction with survival curves", "6"),
        ]
        for i, (title, desc, num) in enumerate(modules):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="module-card">
                    <h3>{num}. {title}</h3>
                    <p>{desc}</p>
                </div>
                """, unsafe_allow_html=True)

        return

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    uploaded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    uploaded_image = cv2.cvtColor(uploaded_image, cv2.COLOR_BGR2RGB)

    if not is_fundus_image(uploaded_image):
        st.error(
            "The uploaded image does not appear to be a retinal fundus image. "
            "Please upload a fundus/retina photo (the circular orange-red eye interior view)."
        )
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(uploaded_image, caption="Uploaded Fundus Image", use_container_width=True)

    with col2:
        st.markdown(f"""
        <div style="background:white;border-radius:12px;padding:1.2rem;border:1px solid #e8edf2;">
            <h4 style="margin:0 0 0.8rem 0;color:#1a2a3a;">Image Details</h4>
            <table style="width:100%;font-size:0.9rem;">
                <tr><td style="color:#6b7c93;padding:0.2rem 0;">Dimensions</td>
                    <td style="font-weight:500;">{uploaded_image.shape[1]} &times; {uploaded_image.shape[0]}</td></tr>
                <tr><td style="color:#6b7c93;padding:0.2rem 0;">Channels</td>
                    <td style="font-weight:500;">RGB ({uploaded_image.shape[2]})</td></tr>
                <tr><td style="color:#6b7c93;padding:0.2rem 0;">File</td>
                    <td style="font-weight:500;">{uploaded_file.name}</td></tr>
                <tr><td style="color:#6b7c93;padding:0.2rem 0;">Mode</td>
                    <td style="font-weight:500;">{operation_mode}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

        analyze_btn = st.button(
            "Analyze Image",
            type="primary",
            use_container_width=True,
        )

    if analyze_btn:
        with st.spinner("Analyzing fundus image..."):
            if operation_mode == "Local (Trained Models)":
                infer = load_models()
                if infer is None:
                    st.warning("Models not found. Using simulated results.")
                    results = simulate_results(uploaded_image)
                else:
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        cv2.imwrite(tmp.name, cv2.cvtColor(uploaded_image, cv2.COLOR_RGB2BGR))
                        try:
                            results = infer.analyze(tmp.name)
                        except ValueError as e:
                            st.error(str(e))
                            return
                    os.unlink(tmp.name)
            else:
                results = simulate_results(uploaded_image)

            st.success("Analysis complete.")

        tab_labels = ["Classification", "Segmentation", "Damage Analysis", "Grad-CAM", "Progression", "Report"]
        tab_icons = ["1", "2", "3", "4", "5", "6"]
        tabs = st.tabs([f"{icon}. {label}" for icon, label in zip(tab_icons, tab_labels)])

        with tabs[0]:
            render_classification_tab(results)
        with tabs[1]:
            render_segmentation_tab(results)
        with tabs[2]:
            render_damage_tab(results)
        with tabs[3]:
            render_gradcam_tab(results)
        with tabs[4]:
            render_progression_tab(results)
        with tabs[5]:
            render_report_tab(results)


if __name__ == "__main__":
    main()
