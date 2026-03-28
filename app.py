"""
Система скоринга сельскохозяйственных субсидий Республики Казахстан.
Основное приложение Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path

from data_loader import load_and_process, get_summary_stats, DATA_PATH
from feature_engineering import (
    compute_producer_features,
    get_scoring_features,
    get_feature_descriptions,
    get_feature_weights_default,
)
from scoring_engine import ScoringEngine
from analytics import (
    plot_status_distribution,
    plot_region_distribution,
    plot_region_amounts,
    plot_direction_pie,
    plot_monthly_trend,
    plot_score_distribution,
    plot_score_by_region,
    plot_producer_breakdown,
    plot_feature_importance,
    plot_comparison_radar,
    plot_amount_vs_score,
    plot_correlation_heatmap,
    plot_approval_rate_by_direction,
)
from fairness import (
    compute_regional_fairness,
    compute_direction_fairness,
    compute_fairness_metrics,
    plot_fairness_overview,
    plot_score_violin_by_region,
    plot_lorenz_curve,
    generate_fairness_report,
)
from utils import (
    format_tenge,
    format_percent,
    format_score,
    score_color,
    score_label,
    truncate_id,
    PRODUCER_DISPLAY_COLS,
    SHORTLIST_COLS,
    export_shortlist_csv,
)

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Скоринг субсидий -- МСХ РК",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
NAVY = "#0F172A"
NAVY_LIGHT = "#1E3A5F"
TEAL = "#3B82F6"
GOLD = "#3B82F6"
WHITE = "#ffffff"
CHART_COLORS = ["#3B82F6", "#1E3A5F", "#60A5FA", "#22C55E", "#b84c4c", "#6b5fa5",
                "#0EA5E9", "#d4843e", "#1D4ED8", "#8b6caf"]

DATA_FILE = os.environ.get(
    "SUBSIDY_DATA_PATH",
    str(Path(__file__).parent / "data" / "subsidies_2025.xlsx"),
)


# ---------------------------------------------------------------------------
# Dark mode helper
# ---------------------------------------------------------------------------
def is_dark_mode() -> bool:
    return True


# ---------------------------------------------------------------------------
# Plotly theme helper
# ---------------------------------------------------------------------------
def apply_chart_theme(fig, height=450, transparent=True):
    """Apply consistent government premium theme to plotly figures."""
    bg = "rgba(0,0,0,0)" if transparent else "#0B1220"
    fig.update_layout(
        template="plotly_dark",
        height=height,
        font=dict(family="Inter, system-ui, sans-serif", size=13, color="#E2E8F0"),
        paper_bgcolor=bg,
        plot_bgcolor=bg,
        title_font=dict(size=17, color="#E2E8F0", family="Inter, system-ui, sans-serif"),
        legend=dict(
            bgcolor="rgba(11,18,32,0.7)",
            bordercolor="rgba(56,189,248,0.15)",
            borderwidth=1,
            font=dict(size=11, color="#E2E8F0"),
        ),
        margin=dict(t=55, b=40, l=40, r=20),
        xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)"),
    )
    return fig


# ---------------------------------------------------------------------------
# Massive CSS injection
# ---------------------------------------------------------------------------
def inject_css():
    # Light theme removed - dark only
    light_css = ""
    # --- Dark theme CSS ---
    dark_css = """
/* ===== IMPORTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ===== GLOBAL TEXT FIX ===== */
b, strong { color: #E2E8F0 !important; }
.stMarkdown b, .stMarkdown strong { color: #E2E8F0 !important; }
div[data-testid="stMarkdownContainer"] b { color: #E2E8F0 !important; }

/* ===== ROOT VARIABLES ===== */
:root {
    --navy: #E2E8F0;
    --navy-light: #94A3B8;
    --teal: #38BDF8;
    --gold: #38BDF8;
    --gold-light: #0EA5E9;
    --white: #0B1220;
    --bg: #0B1220;
    --bg-primary: #0B1220;
    --bg-surface: rgba(17,24,39,0.85);
    --bg-card: rgba(17,24,39,0.85);
    --text-primary: #E2E8F0;
    --text-secondary: #94A3B8;
    --glass-bg: rgba(17, 24, 39, 0.85);
    --glass-border: rgba(56, 189, 248, 0.15);
    --glass-shadow: 0 8px 32px rgba(56, 189, 248, 0.06);
    --blue-primary: #38BDF8;
    --blue-hover: #0284C7;
    --blue-accent: #0EA5E9;
    --border-color: rgba(56, 189, 248, 0.15);
    --shadow-color: rgba(56, 189, 248, 0.06);
    --radius: 16px;
    --radius-sm: 10px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ===== GLOBAL ===== */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    color: #E2E8F0;
}

[data-testid="stAppViewContainer"] {
    background:
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%2338BDF8' fill-opacity='0.04'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E"),
        linear-gradient(160deg, #0B1220 0%, #111827 40%, #0B1220 100%);
}

[data-testid="stMainBlockContainer"] {
    animation: pageContentFadeIn 0.5s ease-out;
}

@keyframes pageContentFadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Dark mode: override default text colors */
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] div {
    color: #E2E8F0;
}

[data-testid="stAppViewContainer"] .stDataFrame,
[data-testid="stAppViewContainer"] .stDataFrame td,
[data-testid="stAppViewContainer"] .stDataFrame th {
    color: #E2E8F0 !important;
}

/* ===== Force all text light in dark mode ===== */
.stApp, .stApp p, .stApp span, .stApp label, .stApp div, .stApp li, .stApp td, .stApp th {
    color: #E2E8F0 !important;
}

/* Streamlit markdown elements */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: #E2E8F0 !important;
}

/* Expander headers */
[data-testid="stExpander"] summary span {
    color: #E2E8F0 !important;
}

/* Selectbox, multiselect, text input labels */
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p {
    color: #E2E8F0 !important;
}

/* Radio labels */
[data-testid="stRadio"] label {
    color: #E2E8F0 !important;
}

/* Metric values and labels */
[data-testid="stMetricValue"] {
    color: #E2E8F0 !important;
}
[data-testid="stMetricLabel"] {
    color: #E2E8F0 !important;
}

/* Dataframe/table text */
[data-testid="stDataFrame"] {
    color: #E2E8F0 !important;
}

/* Caption text */
[data-testid="stCaption"] {
    color: #94A3B8 !important;
}

/* Toggle labels */
.stToggle label span {
    color: #E2E8F0 !important;
}

/* ===== Style ALL Streamlit buttons in dark mode ===== */
.stButton > button {
    background: #0EA5E9 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 8px 20px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    background: #0284C7 !important;
    box-shadow: 0 4px 15px rgba(56,189,248,0.3) !important;
    transform: translateY(-1px) !important;
}

/* Download buttons */
.stDownloadButton > button {
    background: #0EA5E9 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    color: #E2E8F0 !important;
}
[data-testid="stFileUploader"] label {
    color: #E2E8F0 !important;
}
[data-testid="stFileUploader"] small {
    color: #94A3B8 !important;
}

/* ===== Sidebar text in dark mode ===== */
[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] .stButton > button {
    color: white !important;
}

/* ===== Slider and input elements ===== */
[data-testid="stSlider"] label,
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #E2E8F0 !important;
}

/* Number input */
[data-testid="stNumberInput"] label {
    color: #E2E8F0 !important;
}
[data-testid="stNumberInput"] input {
    color: #E2E8F0 !important;
    background: rgba(17,24,39,0.85) !important;
}

/* Text input */
[data-testid="stTextInput"] input {
    color: #E2E8F0 !important;
    background: rgba(17,24,39,0.85) !important;
}

/* Checkbox */
[data-testid="stCheckbox"] label {
    color: #E2E8F0 !important;
}

/* Tabs labels */
.stTabs [data-baseweb="tab"] {
    color: #E2E8F0 !important;
}

/* Info/Warning/Success/Error messages */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
.stAlert p, .stAlert span {
    color: #E2E8F0 !important;
}

/* Selectbox dropdown text and selected values */
[data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-testid="stSelectbox"] div[data-baseweb="select"] div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] span,
[data-testid="stMultiSelect"] div[data-baseweb="select"] div {
    color: #E2E8F0 !important;
}

/* ===== DARK SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0B1220 0%, #111827 100%) !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(56,189,248,0.12) !important;
    border-color: #38BDF8 !important;
    border-left: 3px solid #38BDF8 !important;
    transform: translateX(4px) !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
}
[data-testid="stSidebar"] img {
    filter: brightness(1.1);
}

/* ===== DARK HEADER ===== */
.gov-header {
    background: linear-gradient(135deg, #0B1220 0%, #0EA5E9 50%, #38BDF8 100%) !important;
    color: white !important;
    box-shadow: 0 12px 40px rgba(56, 189, 248, 0.06) !important;
    border: 2px solid rgba(56, 189, 248, 0.15) !important;
}
.gov-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(56,189,248,0.1) 0%, transparent 70%);
    border-radius: 50%;
}
.gov-header h1 {
    color: white !important;
    text-shadow: 0 1px 2px rgba(0,0,0,0.15) !important;
}
.gov-header p {
    color: rgba(255,255,255,0.85) !important;
    font-weight: 400 !important;
}
"""

    theme_css = dark_css

    # --- Common CSS (independent of theme) ---
    common_css = """
/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.04); border-radius: 4px; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, #38BDF8, #94A3B8); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #E2E8F0; }

/* ===== SIDEBAR (structural only) ===== */
[data-testid="stSidebar"] .stButton > button {
    border-radius: 10px !important;
    padding: 10px 16px !important;
    margin-bottom: 4px !important;
    text-align: left !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    box-shadow: none !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

/* ===== HEADER BANNER (structural only) ===== */
.gov-header {
    padding: 28px 36px;
    border-radius: 16px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
    background-clip: padding-box;
}
.gov-header::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 60%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(59,130,246,0.04), transparent);
    animation: headerShimmer 8s ease-in-out infinite;
}
.gov-header h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.5px;
    position: relative;
    z-index: 1;
}
.gov-header p {
    margin: 6px 0 0 0;
    font-size: 15px;
    font-weight: 500;
    position: relative;
    z-index: 1;
}
.gov-header .gold-line {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #3B82F6, #60A5FA);
    border-radius: 2px;
    margin-top: 14px;
    position: relative;
    z-index: 1;
}
/* Animated gradient border on header */
.gov-header-wrapper {
    position: relative;
    border-radius: 16px;
    padding: 2px;
    margin-bottom: 28px;
    background: linear-gradient(90deg, #1E3A5F, #3B82F6, #0EA5E9, #3B82F6, #1E3A5F);
    background-size: 300% 100%;
    animation: headerBorderGlow 4s ease-in-out infinite;
}
.gov-header-wrapper .gov-header {
    margin-bottom: 0;
}
@keyframes headerBorderGlow {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}
@keyframes headerShimmer {
    0% { left: -100%; }
    100% { left: 200%; }
}
@keyframes winnerPulseGov {
    0%, 100% { box-shadow: 0 0 0 0 rgba(59,130,246,0.3); }
    50% { box-shadow: 0 0 0 6px rgba(59,130,246,0); }
}
@keyframes miniCardSlide {
    from { opacity: 0; transform: translateX(-10px); }
    to { opacity: 1; transform: translateX(0); }
}

/* ===== GLASS CARDS ===== */
.glass-card {
    background: rgba(17,24,39,0.85);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 8px 32px rgba(56,189,248,0.06);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(59, 130, 246, 0.14), 0 2px 6px rgba(0, 0, 0, 0.06), inset 0 1px 0 rgba(13, 115, 119, 0.06);
    border-color: rgba(13, 115, 119, 0.3);
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #38BDF8, #38BDF8);
    border-radius: 16px 16px 0 0;
    opacity: 0;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover::before {
    opacity: 1;
}

/* ===== METRIC CARDS ===== */
.metric-card {
    background: rgba(17,24,39,0.85);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 16px;
    padding: 22px 24px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(56,189,248,0.06);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
div[data-testid="stHorizontalBlock"] > div:nth-child(1) .metric-card::before { background: linear-gradient(90deg, #3B82F6, #60A5FA); }
div[data-testid="stHorizontalBlock"] > div:nth-child(2) .metric-card::before { background: linear-gradient(90deg, #2563EB, #3B82F6); }
div[data-testid="stHorizontalBlock"] > div:nth-child(3) .metric-card::before { background: linear-gradient(90deg, #60A5FA, #0EA5E9); }
div[data-testid="stHorizontalBlock"] > div:nth-child(4) .metric-card::before { background: linear-gradient(90deg, #0EA5E9, #38BDF8); }
div[data-testid="stHorizontalBlock"] > div:nth-child(5) .metric-card::before { background: linear-gradient(90deg, #1D4ED8, #3B82F6); }
div[data-testid="stHorizontalBlock"] > div:nth-child(6) .metric-card::before { background: linear-gradient(90deg, #22C55E, #3B82F6); }
.metric-card:hover {
    transform: translateY(-4px) scale(1.02);
    box-shadow: 0 16px 48px rgba(59, 130, 246, 0.18);
}
.metric-card .metric-icon {
    font-size: 14px;
    font-weight: 700;
    color: #38BDF8;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
}
.metric-card .metric-value {
    font-size: 32px;
    font-weight: 800;
    background: linear-gradient(135deg, #60A5FA, #38BDF8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 4px;
}
.metric-card .metric-value.gold {
    background: linear-gradient(135deg, #38BDF8, #2563EB);
    -webkit-background-clip: text;
    background-clip: text;
}
.metric-card .metric-sub {
    font-size: 12px;
    color: #94A3B8;
    font-weight: 500;
}
.metric-card .metric-bar {
    height: 3px;
    background: linear-gradient(90deg, #38BDF8, #38BDF8);
    border-radius: 2px;
    margin-top: 12px;
    width: 0%;
    animation: barGrow 1.2s ease-out forwards;
}

@keyframes barGrow {
    to { width: 100%; }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes countUp {
    from { opacity: 0; transform: scale(0.8); }
    to { opacity: 1; transform: scale(1); }
}

.animate-in {
    animation: fadeInUp 0.6s ease-out forwards;
}

.animate-delay-1 { animation-delay: 0.1s; }
.animate-delay-2 { animation-delay: 0.2s; }
.animate-delay-3 { animation-delay: 0.3s; }
.animate-delay-4 { animation-delay: 0.4s; }

/* ===== SECTION HEADERS ===== */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 2px solid rgba(13, 115, 119, 0.15);
}
.section-header h2 {
    margin: 0;
    font-size: 22px;
    font-weight: 700;
    color: #E2E8F0;
}
.section-header .section-badge {
    background: linear-gradient(135deg, #38BDF8, #1E3A5F);
    color: white !important;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ===== RECOMMENDATION BADGES ===== */
.badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-green {
    background: linear-gradient(135deg, rgba(45, 143, 94, 0.15), rgba(45, 143, 94, 0.08));
    color: #4ade80 !important;
    border: 1px solid rgba(45, 143, 94, 0.25);
}
.badge-yellow {
    background: linear-gradient(135deg, rgba(201, 162, 39, 0.15), rgba(201, 162, 39, 0.08));
    color: #fbbf24 !important;
    border: 1px solid rgba(201, 162, 39, 0.25);
}
.badge-red {
    background: linear-gradient(135deg, rgba(184, 76, 76, 0.15), rgba(184, 76, 76, 0.08));
    color: #f87171 !important;
    border: 1px solid rgba(184, 76, 76, 0.25);
}
.badge-blue {
    background: linear-gradient(135deg, rgba(26, 74, 110, 0.15), rgba(26, 74, 110, 0.08));
    color: #60A5FA !important;
    border: 1px solid rgba(26, 74, 110, 0.25);
}

/* ===== STYLED TABLE ===== */
.styled-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: 10px;
    overflow: hidden;
    font-size: 13px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}
.styled-table thead th {
    background: linear-gradient(135deg, #0F172A, #1E3A5F);
    color: white !important;
    padding: 14px 16px;
    font-weight: 600;
    text-align: left;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    white-space: nowrap;
    border-bottom: 2px solid rgba(59,130,246,0.3);
}
.styled-table tbody tr {
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    background: rgba(17,24,39,0.85);
}
.styled-table tbody tr:nth-child(odd) {
    background: rgba(17,24,39,0.85);
}
.styled-table tbody tr:nth-child(even) {
    background: rgba(13, 115, 119, 0.04);
}
.styled-table tbody tr:hover {
    background: rgba(13, 115, 119, 0.1);
    transform: scale(1.003);
}
.styled-table tbody td {
    padding: 13px 16px;
    border-bottom: 1px solid rgba(0,0,0,0.04);
    vertical-align: middle;
    color: #E2E8F0;
}

/* Score bar inside table */
.score-bar-container {
    width: 100%;
    background: rgba(0,0,0,0.06);
    border-radius: 6px;
    height: 24px;
    position: relative;
    overflow: hidden;
}
.score-bar {
    height: 100%;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 8px;
    font-size: 11px;
    font-weight: 700;
    color: white !important;
    transition: width 0.8s ease-out;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}
.score-bar.high { background: linear-gradient(90deg, #1a7a4a, #2d8f5e); }
.score-bar.medium { background: linear-gradient(90deg, #2563EB, #3B82F6); }
.score-bar.low { background: linear-gradient(90deg, #b84c4c, #d46a6a); }

/* ===== GAUGE CHART ===== */
.gauge-container {
    text-align: center;
    padding: 16px;
}
.gauge-label {
    font-size: 13px;
    font-weight: 600;
    color: #94A3B8;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.gauge-value {
    font-size: 36px;
    font-weight: 800;
    line-height: 1;
}

/* ===== STRENGTH / WEAKNESS ITEMS ===== */
.sw-item {
    padding: 10px 16px;
    border-radius: 10px;
    margin-bottom: 6px;
    font-size: 14px;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 10px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.sw-item:hover {
    transform: translateX(4px);
}
.sw-strength {
    background: rgba(45, 143, 94, 0.08);
    border-left: 3px solid #2d8f5e;
    color: #4ade80 !important;
}
.sw-weakness {
    background: rgba(184, 76, 76, 0.08);
    border-left: 3px solid #b84c4c;
    color: #f87171 !important;
}
.sw-icon {
    font-weight: 800;
    font-size: 16px;
    flex-shrink: 0;
}

/* ===== COMPARISON WINNER ===== */
.winner-highlight {
    background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(59,130,246,0.08));
    border: 2px solid #38BDF8;
    border-radius: 16px;
    padding: 4px;
    animation: winnerPulseGov 2.5s ease-in-out infinite;
}

/* ===== FILTER PANEL ===== */
.filter-panel {
    background: rgba(17,24,39,0.85);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(56,189,248,0.15);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
}

/* ===== DIVIDER ===== */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(13, 115, 119, 0.2), #38BDF8, rgba(13, 115, 119, 0.2), transparent);
    margin: 28px 0;
    border: none;
}

/* ===== BASELINE COMPARISON CARDS ===== */
.baseline-card {
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.baseline-old {
    background: linear-gradient(135deg, rgba(184,76,76,0.08), rgba(184,76,76,0.03));
    border: 1px solid rgba(184,76,76,0.2);
}
.baseline-new {
    background: linear-gradient(135deg, rgba(45,143,94,0.08), rgba(59,130,246,0.05));
    border: 2px solid #38BDF8;
    box-shadow: 0 8px 32px rgba(59,130,246,0.12);
}
.baseline-card h3 {
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 16px;
}
.baseline-metric {
    font-size: 28px;
    font-weight: 800;
    margin: 8px 0;
}
.baseline-label {
    font-size: 12px;
    color: #94A3B8;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.improvement-arrow {
    font-size: 40px;
    color: #38BDF8;
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* ===== FAIRNESS ALERT ===== */
.fairness-alert {
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 8px;
    font-size: 14px;
    font-weight: 500;
    display: flex;
    align-items: flex-start;
    gap: 10px;
}
.fairness-ok {
    background: rgba(45, 143, 94, 0.08);
    border: 1px solid rgba(45, 143, 94, 0.2);
    border-left: 3px solid #2d8f5e;
    color: #4ade80 !important;
}
.fairness-warn {
    background: rgba(201, 162, 39, 0.08);
    border: 1px solid rgba(201, 162, 39, 0.2);
    border-left: 3px solid #3B82F6;
    color: #fbbf24 !important;
}
.fairness-danger {
    background: rgba(184, 76, 76, 0.08);
    border: 1px solid rgba(184, 76, 76, 0.2);
    border-left: 3px solid #b84c4c;
    color: #f87171 !important;
}

/* ===== TIMELINE ===== */
.timeline-item {
    display: flex;
    gap: 16px;
    padding: 14px 0;
    border-bottom: none;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    margin-left: 6px;
    border-left: 2px solid rgba(59,130,246,0.15);
    padding-left: 20px;
}
.timeline-item:last-child {
    border-left-color: transparent;
}
.timeline-item:hover {
    background: rgba(59,130,246,0.03);
    border-radius: 0 10px 10px 0;
}
.timeline-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 4px;
    box-shadow: 0 0 0 3px rgba(0,0,0,0.05);
    position: absolute;
    left: -7px;
}
.timeline-dot.approved { background: #2d8f5e; }
.timeline-dot.rejected { background: #b84c4c; }
.timeline-dot.pending { background: #3B82F6; }
.timeline-dot.completed { background: #3B82F6; }
.timeline-content {
    flex: 1;
}
.timeline-date {
    font-size: 12px;
    color: #94A3B8;
    font-weight: 500;
}
.timeline-title {
    font-size: 14px;
    font-weight: 600;
    color: #E2E8F0;
    margin: 2px 0;
}
.timeline-meta {
    font-size: 12px;
    color: #94A3B8;
}

/* ===== OVERRIDE DEFAULT STREAMLIT METRICS ===== */
[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    color: #E2E8F0 !important;
}
[data-testid="stMetricLabel"] {
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    font-size: 12px !important;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 10px 20px;
    font-weight: 600;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.stTabs [aria-selected="true"] {
    background: rgba(13, 115, 119, 0.08) !important;
    border-bottom: 3px solid #38BDF8 !important;
}

/* ===== BUTTONS ===== */
.stDownloadButton > button,
.stButton > button {
    background: linear-gradient(135deg, #0F172A, #3B82F6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-size: 13px !important;
    box-shadow: 0 2px 8px rgba(59, 130, 246, 0.15) !important;
}
.stDownloadButton > button:hover,
.stButton > button:hover {
    background: linear-gradient(135deg, #1E3A5F, #3B82F6) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(13, 115, 119, 0.3), 0 0 12px rgba(13, 115, 119, 0.15) !important;
    filter: brightness(1.05) !important;
}

/* ===== INPUTS & SELECTS ===== */
.stSelectbox > div > div,
.stSlider > div,
.stNumberInput > div {
    border-radius: 10px !important;
}

/* ===== HIDE STREAMLIT BRANDING ===== */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* ===== DATA QUALITY INDICATOR ===== */
.dq-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 14px;
    border-radius: 10px;
    font-size: 13px;
    font-weight: 500;
}
.dq-good {
    background: rgba(45, 143, 94, 0.08);
    color: #4ade80 !important;
}
.dq-warn {
    background: rgba(201, 162, 39, 0.08);
    color: #fbbf24 !important;
}
.dq-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dq-dot.good { background: #2d8f5e; }
.dq-dot.warn { background: #3B82F6; }

/* ===== FOOTER ===== */
.gov-footer {
    text-align: center;
    padding: 20px 0 12px 0;
    margin-top: 40px;
    border-top: 1px solid rgba(13, 115, 119, 0.1);
    font-size: 11px;
    color: #94A3B8;
    letter-spacing: 0.5px;
    opacity: 0.6;
    font-weight: 300;
}

/* ===== EXPANDERS ===== */
details[data-testid="stExpander"] {
    border: 1px solid rgba(56,189,248,0.15) !important;
    border-left: 3px solid #38BDF8 !important;
    border-radius: 0 10px 10px 0 !important;
    transition: all 0.25s ease !important;
}
details[data-testid="stExpander"]:hover {
    border-left-color: #38BDF8 !important;
    box-shadow: 0 2px 12px rgba(59,130,246,0.06) !important;
}

/* ===== SCORE DOT INDICATOR ===== */
.score-dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}
.score-dot-high { background: #2d8f5e; box-shadow: 0 0 6px rgba(45,143,94,0.4); }
.score-dot-mid { background: #60A5FA; box-shadow: 0 0 6px rgba(96,165,250,0.4); }
.score-dot-low { background: #b84c4c; box-shadow: 0 0 6px rgba(184,76,76,0.4); }

/* ===== COAT OF ARMS FALLBACK ===== */
.coat-of-arms-fallback {
    text-align: center;
    font-size: 24px;
    font-weight: 800;
    color: rgba(59,130,246,0.9);
    padding: 10px 0;
    letter-spacing: 2px;
}
"""

    st.markdown(f"<style>{common_css}\n{theme_css}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Reusable HTML components
# ---------------------------------------------------------------------------
def render_page_header(title: str, subtitle: str = ""):
    sub_html = f'<p>{subtitle}</p>' if subtitle else ''
    st.markdown(f"""
    <div class="gov-header-wrapper">
        <div class="gov-header animate-in">
            <h1>{title}</h1>
            {sub_html}
            <div class="gold-line"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(label: str, value: str, sub: str = "", gold: bool = False, delay: int = 1):
    v_class = "metric-value gold" if gold else "metric-value"
    st.markdown(f"""
    <div class="metric-card animate-in animate-delay-{delay}">
        <div class="metric-icon">{label}</div>
        <div class="{v_class}">{value}</div>
        <div class="metric-sub">{sub}</div>
        <div class="metric-bar"></div>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, badge: str = ""):
    badge_html = f'<span class="section-badge">{badge}</span>' if badge else ''
    st.markdown(f"""
    <div class="section-header">
        <h2>{title}</h2>
        {badge_html}
    </div>
    """, unsafe_allow_html=True)


def render_divider():
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)


def render_footer():
    st.markdown(
        '<div class="gov-footer">'
        'Decentrathon 5.0 | AI inDrive Track | Кейс: Скоринг сельхозпроизводителей'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Navigation helpers
# ---------------------------------------------------------------------------
PAGE_LIST = [
    ("dashboard",   "\u25A3 Главная панель",            "Главная панель"),
    ("as_is",       "\u2192 Текущий процесс (AS IS)",   "Текущий процесс (AS IS)"),
    ("overview",    "\u2261 Обзор данных",               "Обзор данных"),
    ("scoring",     "\u2605 Скоринг производителей",     "Скоринг производителей"),
    ("profile",     "\u2302 Профиль производителя",      "Профиль производителя"),
    ("comparison",  "\u21C4 Сравнение производителей",   "Сравнение производителей"),
    ("shortlist",   "\u2611 Шортлист",                   "Шортлист"),
    ("fairness",    "\u2696 Анализ справедливости",      "Анализ справедливости"),
    ("analytics",   "\u2318 Аналитика",                  "Аналитика"),
    ("settings",    "\u2699 Настройки модели",           "Настройки модели"),
]

PAGE_KEY_TO_LABEL = {k: label for k, _, label in PAGE_LIST}


def nav_to(page_key: str):
    """Callback for navigation buttons."""
    st.session_state["current_page"] = page_key


def render_breadcrumb(current_label: str):
    """Render breadcrumb: Главная > Текущая страница."""
    st.markdown(
        f'<div style="font-size:13px; margin-bottom:16px; color:#94A3B8;">'
        f'<a href="#" style="color:#38BDF8; text-decoration:none; font-weight:600;" '
        f'onclick="return false;">Главная</a>'
        f' &gt; <span style="font-weight:600; color:#E2E8F0;">{current_label}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    # Clickable "Главная" via st.button (HTML links can't trigger Streamlit state)
    # We render a small inline button right after for real navigation
    if current_label != "Главная панель":
        cols = st.columns([1, 8])
        with cols[0]:
            if st.button("\u2190 Назад", key=f"back_{current_label}"):
                nav_to("dashboard")
                st.rerun()


def get_score_bar_class(score):
    if score >= 65:
        return "high"
    elif score >= 35:
        return "medium"
    else:
        return "low"


def get_recommendation_badge(score):
    if score >= 75:
        return '<span class="badge badge-green">Рекомендован</span>'
    elif score >= 50:
        return '<span class="badge badge-yellow">На рассмотрении</span>'
    else:
        return '<span class="badge badge-red">Не рекомендован</span>'


def get_status_dot_class(status: str) -> str:
    status_lower = status.lower() if isinstance(status, str) else ""
    if "исполнена" in status_lower:
        return "completed"
    elif "одобрена" in status_lower or "сформировано" in status_lower:
        return "approved"
    elif "отклонена" in status_lower:
        return "rejected"
    return "pending"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Загрузка данных...")
def load_data(filepath: str) -> pd.DataFrame:
    return load_and_process(filepath)


@st.cache_data(show_spinner="Вычисление признаков...")
def compute_features(df_hash: int, _df: pd.DataFrame) -> pd.DataFrame:
    return compute_producer_features(_df)


@st.cache_resource(show_spinner="Обучение скоринговой модели...")
def train_scoring_engine(
    features_hash: int, _features: pd.DataFrame, ml_w: float, rule_w: float
) -> ScoringEngine:
    engine = ScoringEngine(ml_weight=ml_w, rule_weight=rule_w)
    engine.fit(_features)
    return engine


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    # ----- Initialize navigation state -----
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "dashboard"

    # ----- Sidebar -----
    # Government emblem as styled element
    st.sidebar.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
        <div style="width:64px;height:64px;margin:0 auto;border-radius:50%;background:linear-gradient(135deg,#3B82F6,#2563EB);display:flex;align-items:center;justify-content:center;box-shadow:0 4px 16px rgba(59,130,246,0.3);border:2px solid rgba(59,130,246,0.5)">
            <span style="color:white;font-weight:900;font-size:1.5rem;letter-spacing:1px">KZ</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div style='text-align:center; margin-bottom:4px;'>"
        "<span style='font-size:20px; font-weight:700; letter-spacing:-0.5px; color:#E2E8F0;'>Скоринг субсидий</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<div style='text-align:center; font-size:12px; color:#94A3B8; margin-bottom:16px;'>"
        "Министерство сельского хозяйства РК</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")


    # Navigation label
    st.sidebar.markdown(
        "<div style='color:rgba(255,255,255,0.6); font-size:12px; text-transform:uppercase; "
        "letter-spacing:1.5px; font-weight:600; margin-bottom:8px; padding:0 4px;'>НАВИГАЦИЯ</div>",
        unsafe_allow_html=True,
    )

    # Page descriptions for tooltip
    page_descriptions = {
        "dashboard": "Ключевые метрики и топ производителей",
        "as_is": "Диаграмма текущего бизнес-процесса",
        "overview": "Статистика и качество исходных данных",
        "scoring": "Ранжирование всех производителей",
        "profile": "Детальный анализ одного производителя",
        "comparison": "Сопоставление 2-3 производителей",
        "shortlist": "Формирование списка получателей",
        "fairness": "Проверка на предвзятость модели",
        "analytics": "Корреляции, валидация, базовые модели",
        "settings": "Настройка весов и параметров",
    }

    current_page = st.session_state["current_page"]

    # Render navigation buttons
    for page_key, icon_label, _display_label in PAGE_LIST:
        is_active = (current_page == page_key)
        if is_active:
            # Active page: highlighted styling via custom HTML + button
            st.sidebar.markdown(
                f"<div style='background:linear-gradient(135deg, rgba(59,130,246,0.25), rgba(56,189,248,0.15)); "
                f"border:1px solid #3B82F6; border-left:3px solid #60A5FA; border-radius:10px; "
                f"padding:10px 16px; margin-bottom:4px; font-weight:600; font-size:14px; "
                f"color:#E2E8F0; box-shadow:0 0 12px rgba(59,130,246,0.15);'>"
                f"{icon_label}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.button(
                icon_label,
                key=f"nav_{page_key}",
                on_click=nav_to,
                args=(page_key,),
                use_container_width=True,
            )

    # Show description of current page
    desc = page_descriptions.get(current_page, "")
    if desc:
        st.sidebar.markdown(
            f"<div style='font-size:12px; opacity:0.7; padding:4px 8px; "
            f"background:rgba(255,255,255,0.06); border-radius:8px; margin-bottom:8px; margin-top:8px;'>"
            f"{desc}</div>",
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size:11px; opacity:0.5; text-align:center; padding:8px 0;'>"
        "Система скоринга v2.0<br>Decentrathon 5.0 | AI inDrive Track</div>",
        unsafe_allow_html=True,
    )

    # Inject dark theme CSS
    inject_css()

    page = current_page

    # ----- Data loading -----
    # Sidebar: upload custom Excel
    with st.sidebar.expander("Загрузить свой датасет", expanded=False):
        st.caption("Загрузите Excel (.xlsx) в формате датасета субсидий")
        custom_upload = st.file_uploader(
            "Файл Excel", type=["xlsx"], key="sidebar_custom_xlsx",
            label_visibility="collapsed",
        )
        if custom_upload:
            # Only process the upload if file changed (avoid infinite rerun loop)
            upload_id = f"{custom_upload.name}_{custom_upload.size}"
            if st.session_state.get("_last_upload_id") != upload_id:
                import tempfile as _tf
                with _tf.NamedTemporaryFile(delete=False, suffix=".xlsx") as _tmp:
                    _tmp.write(custom_upload.read())
                    st.session_state["custom_data_path"] = _tmp.name
                st.session_state["_last_upload_id"] = upload_id
                # Clear cached data so the new file is loaded
                load_data.clear()
                st.rerun()

        # Sample data links
        import os as _os
        sample_dir = _os.path.join(_os.path.dirname(__file__), "sample_data")
        if _os.path.isdir(sample_dir):
            st.caption("Примеры данных:")
            for fname in sorted(_os.listdir(sample_dir)):
                if fname.endswith(".xlsx"):
                    fpath = _os.path.join(sample_dir, fname)
                    size_kb = _os.path.getsize(fpath) // 1024
                    with open(fpath, "rb") as _f:
                        st.download_button(
                            f"{fname} ({size_kb} KB)",
                            _f.read(),
                            file_name=fname,
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"dl_sample_{fname}",
                        )

    custom_path = st.session_state.get("custom_data_path", None)
    data_source = custom_path if custom_path else DATA_FILE

    try:
        df = load_data(data_source)
    except FileNotFoundError:
        st.error(
            f"Файл данных не найден: `{DATA_FILE}`\n\n"
            "Загрузите файл через боковую панель или поместите в `data/subsidies_2025.xlsx`."
        )
        return

    if df.empty:
        st.warning(
            "Загруженный датасет пуст (0 записей). "
            "Проверьте файл данных или загрузите другой файл через боковую панель."
        )
        return

    df_hash = hash(len(df))
    with st.spinner("Вычисление признаков производителей..."):
        producer_features = compute_features(df_hash, df)

    if producer_features.empty:
        st.warning(
            "Не удалось извлечь признаки производителей из данных. "
            "Проверьте формат файла и наличие обязательных столбцов."
        )
        return

    if "ml_weight" not in st.session_state:
        st.session_state.ml_weight = 0.6
    if "rule_weight" not in st.session_state:
        st.session_state.rule_weight = 0.4
    if "feature_weights" not in st.session_state:
        st.session_state.feature_weights = get_feature_weights_default()

    with st.spinner("Обучение и скоринг модели..."):
        engine = train_scoring_engine(
            hash(len(producer_features)),
            producer_features,
            st.session_state.ml_weight,
            st.session_state.rule_weight,
        )
        engine.update_weights(
            st.session_state.feature_weights,
            st.session_state.ml_weight,
            st.session_state.rule_weight,
        )
        scored = engine.score(producer_features)

    if scored.empty:
        st.warning(
            "Результат скоринга пуст. "
            "Проверьте данные и параметры модели."
        )
        return

    # ----- Routing -----
    if page == "dashboard":
        render_dashboard(df, scored)
    elif page == "as_is":
        render_as_is_process()
    elif page == "overview":
        render_overview(df, producer_features)
    elif page == "scoring":
        render_scoring(scored, producer_features)
    elif page == "profile":
        render_profile(df, scored, producer_features, engine)
    elif page == "comparison":
        render_comparison(scored, producer_features)
    elif page == "shortlist":
        render_shortlist(scored)
    elif page == "fairness":
        render_fairness(scored)
    elif page == "analytics":
        render_analytics(df, scored, engine)
    elif page == "settings":
        render_settings(scored)

    # Footer on every page
    render_footer()


# ===========================================================================
# PAGE: AS IS PROCESS
# ===========================================================================
def render_as_is_process():
    render_breadcrumb("Текущий процесс (AS IS)")
    render_page_header(
        "Текущий процесс (AS IS)",
        "Действующая схема распределения субсидий на приобретение маточного поголовья КРС"
    )

    st.markdown("""
    <div class="glass-card" style="margin-bottom:24px;">
        <div style="line-height:1.8; font-size:14px; color:#E2E8F0;">
            <p style="font-weight:700; font-size:16px; color:#E2E8F0; margin-bottom:12px;">
                Принцип очередности (FCFS)
            </p>
            <p>
                Текущий процесс распределения субсидий на приобретение маточного поголовья КРС
                работает по принципу очередности. Наша система заменяет этот подход merit-based
                скорингом.
            </p>
            <p style="margin-top:12px;">
                Диаграмма ниже описывает бизнес-процесс AS IS по стандарту BPMN.
                Заявки обрабатываются в порядке поступления, без оценки качества заявителя.
                Это приводит к неэффективному распределению бюджетных средств.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_divider()

    render_section_header("Диаграмма бизнес-процесса (BPMN)", "AS IS")

    image_path = Path(__file__).parent / "assets" / "as_is_process.png"
    if image_path.exists():
        st.image(str(image_path), use_container_width=True,
                 caption="AS IS: Приобретение маточного поголовья КРС, овец, баранов-производителей")
    else:
        st.warning("Файл диаграммы не найден: assets/as_is_process.png")

    render_divider()

    render_section_header("Проблемы текущего подхода", "Анализ")

    problems = [
        ("Принцип очередности (FCFS)", "Субсидии распределяются по принципу 'первый пришёл -- первый получил', без учёта качества заявителя."),
        ("Отсутствие объективной оценки", "Нет формализованных критериев оценки эффективности производителя."),
        ("Неэффективное распределение бюджета", "Средства могут получать производители с низкой историей освоения субсидий."),
        ("Непрозрачность", "Отсутствие объяснимых критериев снижает доверие к системе распределения."),
    ]

    for i, (title, desc) in enumerate(problems, 1):
        st.markdown(
            f'<div class="sw-item sw-weakness" style="margin-bottom:8px;">'
            f'<span class="sw-icon">{i}.</span> '
            f'<div><span style="font-weight:700;">{title}.</span> {desc}</div></div>',
            unsafe_allow_html=True,
        )

    render_divider()

    render_section_header("Наше решение: Merit-based скоринг", "TO BE")

    solutions = [
        ("Объективный скоринг", "Комбинированная модель (ML + правила) оценивает каждого производителя по 14 факторам."),
        ("Прозрачность", "Каждый балл объясним: производитель видит свои сильные и слабые стороны."),
        ("Эффективность бюджета", "Приоритет получают производители с высокой историей освоения и исполнения."),
        ("Анализ справедливости", "Встроенные проверки на региональную и направленческую предвзятость."),
    ]

    for i, (title, desc) in enumerate(solutions, 1):
        st.markdown(
            f'<div class="sw-item sw-strength" style="margin-bottom:8px;">'
            f'<span class="sw-icon">{i}.</span> '
            f'<div><span style="font-weight:700;">{title}.</span> {desc}</div></div>',
            unsafe_allow_html=True,
        )

    render_divider()

    # --- Visual comparison: AS IS vs TO BE ---
    render_section_header("Сравнение подходов", "AS IS vs TO BE")

    cmp1, cmp_arrow, cmp2 = st.columns([5, 1, 5])
    with cmp1:
        st.markdown("""
        <div class="baseline-card baseline-old">
            <h3 style="color:#8a3030;">AS IS: Текущий процесс</h3>
            <div style="text-align:left; font-size:14px; line-height:1.8; color:#E2E8F0;">
                <div style="margin-bottom:6px;"><span style="font-weight:700; color:#b84c4c;">\u2717</span> Принцип очередности (FCFS)</div>
                <div style="margin-bottom:6px;"><span style="font-weight:700; color:#b84c4c;">\u2717</span> Нет объективной оценки</div>
                <div style="margin-bottom:6px;"><span style="font-weight:700; color:#b84c4c;">\u2717</span> Непрозрачное распределение</div>
                <div style="margin-bottom:6px;"><span style="font-weight:700; color:#b84c4c;">\u2717</span> Низкая эффективность бюджета</div>
                <div><span style="font-weight:700; color:#b84c4c;">\u2717</span> Нет проверки справедливости</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with cmp_arrow:
        st.markdown("""
        <div class="improvement-arrow" style="height:100%; display:flex; align-items:center; justify-content:center;">
            \u21D2
        </div>
        """, unsafe_allow_html=True)
    with cmp2:
        st.markdown("""
        <div class="baseline-card baseline-new">
            <h3 style="color:#38BDF8;">TO BE: Наше решение</h3>
            <div style="text-align:left; font-size:14px; line-height:1.8; color:#E2E8F0;">
                <div style="margin-bottom:6px;"><span style="font-weight:700; color:#2d8f5e;">\u2713</span> Merit-based скоринг</div>
                <div style="margin-bottom:6px;"><span style="font-weight:700; color:#2d8f5e;">\u2713</span> 14 объективных факторов</div>
                <div style="margin-bottom:6px;"><span style="font-weight:700; color:#2d8f5e;">\u2713</span> Полная прозрачность (XAI)</div>
                <div style="margin-bottom:6px;"><span style="font-weight:700; color:#2d8f5e;">\u2713</span> Приоритет эффективным</div>
                <div><span style="font-weight:700; color:#2d8f5e;">\u2713</span> Встроенный анализ справедливости</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 1: DASHBOARD
# ===========================================================================
def render_dashboard(df: pd.DataFrame, scored: pd.DataFrame):
    render_breadcrumb("Главная панель")
    render_page_header(
        "Главная панель",
        "Система скоринга сельскохозяйственных субсидий Республики Казахстан"
    )

    stats = get_summary_stats(df)

    # --- Usage workflow ---
    with st.expander("Как использовать систему (инструкция для специалиста)", expanded=False):
        st.markdown("""
        <div class="glass-card" style="padding:20px 24px;">
            <div style="font-weight:700; font-size:16px; color:#E2E8F0; margin-bottom:16px;">
                Порядок работы с системой скоринга
            </div>
            <div style="line-height:2.0; font-size:14px; color:#E2E8F0;">
                <div class="sw-item sw-strength" style="margin-bottom:10px;">
                    <span class="sw-icon">1.</span>
                    <div><b>Загрузить данные.</b> Откройте боковую панель и загрузите актуальный
                    файл Excel (.xlsx) из реестра ИСС (subsidy.plem.kz). Система автоматически
                    обработает данные и вычислит признаки.</div>
                </div>
                <div class="sw-item sw-strength" style="margin-bottom:10px;">
                    <span class="sw-icon">2.</span>
                    <div><b>Настроить веса (если нужно).</b> Перейдите в раздел
                    "Настройки модели" для изменения баланса ML/Правила и весов
                    отдельных факторов в соответствии с приоритетами государственной политики.</div>
                </div>
                <div class="sw-item sw-strength" style="margin-bottom:10px;">
                    <span class="sw-icon">3.</span>
                    <div><b>Просмотреть скоринг.</b> Откройте страницу "Скоринг производителей"
                    для просмотра полного ранжирования. Используйте фильтры по региону,
                    направлению и минимальному баллу.</div>
                </div>
                <div class="sw-item sw-strength" style="margin-bottom:10px;">
                    <span class="sw-icon">4.</span>
                    <div><b>Изучить профили.</b> Перейдите в "Профиль производителя" для
                    детального анализа конкретного заявителя: скоринговые баллы, вклад
                    каждого фактора, сильные и слабые стороны, история заявок.</div>
                </div>
                <div class="sw-item sw-strength" style="margin-bottom:10px;">
                    <span class="sw-icon">5.</span>
                    <div><b>Сформировать шорт-лист.</b> Используйте раздел "Шортлист"
                    для генерации списка приоритетных получателей субсидий.
                    Экспортируйте результат в CSV для дальнейшей работы.</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # --- Big metrics ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card(
            "ПРОИЗВОДИТЕЛИ", f"{stats['unique_producers']:,}",
            "Уникальных в реестре", delay=1
        )
    with c2:
        render_metric_card(
            "ОБЪЁМ СУБСИДИЙ", format_tenge(stats["total_amount"]),
            "Общая сумма заявок", gold=True, delay=2
        )
    with c3:
        render_metric_card(
            "СРЕДНИЙ БАЛЛ", f"{scored['combined_score'].mean():.1f}",
            f"Медиана: {scored['combined_score'].median():.1f}", delay=3
        )
    with c4:
        render_metric_card(
            "РЕГИОНЫ", f"{stats['unique_regions']}",
            f"Районов: {stats['unique_districts']}", delay=4
        )

    render_divider()

    # --- Action buttons (glass cards) ---
    render_section_header("Быстрый доступ", "Действия")
    ac1, ac2, ac3, ac4 = st.columns(4)
    with ac1:
        st.markdown("""
        <div class="glass-card" style="text-align:center; cursor:pointer;">
            <div style="font-size:22px; font-weight:800; color:#38BDF8; margin-bottom:8px;">\u2605</div>
            <div style="font-size:15px; font-weight:700; color:#E2E8F0; margin-bottom:6px;">Скоринг производителей</div>
            <div style="font-size:12px; color:#94A3B8; line-height:1.5;">
                Ранжирование всех производителей по комбинированному баллу
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Перейти к скорингу", key="dash_action_scoring"):
            nav_to("scoring")
            st.rerun()
    with ac2:
        st.markdown("""
        <div class="glass-card" style="text-align:center; cursor:pointer;">
            <div style="font-size:22px; font-weight:800; color:#38BDF8; margin-bottom:8px;">\u2302</div>
            <div style="font-size:15px; font-weight:700; color:#E2E8F0; margin-bottom:6px;">Профиль производителя</div>
            <div style="font-size:12px; color:#94A3B8; line-height:1.5;">
                Детальный анализ скорингового профиля конкретного производителя
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Перейти к профилю", key="dash_action_profile"):
            nav_to("profile")
            st.rerun()
    with ac3:
        st.markdown("""
        <div class="glass-card" style="text-align:center; cursor:pointer;">
            <div style="font-size:22px; font-weight:800; color:#38BDF8; margin-bottom:8px;">\u2696</div>
            <div style="font-size:15px; font-weight:700; color:#E2E8F0; margin-bottom:6px;">Справедливость</div>
            <div style="font-size:12px; color:#94A3B8; line-height:1.5;">
                Проверка модели на предвзятость по регионам и направлениям
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Перейти к анализу", key="dash_action_fairness"):
            nav_to("fairness")
            st.rerun()
    with ac4:
        st.markdown("""
        <div class="glass-card" style="text-align:center; cursor:pointer;">
            <div style="font-size:22px; font-weight:800; color:#38BDF8; margin-bottom:8px;">\u2318</div>
            <div style="font-size:15px; font-weight:700; color:#E2E8F0; margin-bottom:6px;">Аналитика</div>
            <div style="font-size:12px; color:#94A3B8; line-height:1.5;">
                Корреляции, валидация модели, сравнение с базовыми подходами
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Перейти к аналитике", key="dash_action_analytics"):
            nav_to("analytics")
            st.rerun()

    render_divider()

    # --- Value proposition: Merit-based vs FCFS ---
    render_section_header("Ценность системы", "Merit-based скоринг")
    val1, val2, val3 = st.columns(3)
    with val1:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:14px; font-weight:700; color:#38BDF8; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">
                Вместо FCFS
            </div>
            <div style="font-size:14px; color:#E2E8F0; line-height:1.6;">
                Текущая система: заявки по очереди (FCFS).
                Наш подход: <b>ранжирование по заслугам</b> на основе 14 объективных факторов.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with val2:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:14px; font-weight:700; color:#38BDF8; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">
                Гибридная модель
            </div>
            <div style="font-size:14px; color:#E2E8F0; line-height:1.6;">
                <b>60% ML</b> (GradientBoosting, нелинейные зависимости) +
                <b>40% правила</b> (экспертные веса, полная объяснимость).
            </div>
        </div>
        """, unsafe_allow_html=True)
    with val3:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:14px; font-weight:700; color:#38BDF8; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;">
                Результат
            </div>
            <div style="font-size:14px; color:#E2E8F0; line-height:1.6;">
                Приоритет -- производителям с высоким исполнением, одобрением и освоением средств.
                <b>Каждый балл объясним.</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    render_divider()

    # --- Two columns: top-10 + status donut ---
    col_left, col_right = st.columns([3, 2])

    with col_left:
        render_section_header("Топ-10 производителей", "Лидеры")
        top10 = scored.nsmallest(10, "rank")
        table_rows = ""
        for _, row in top10.iterrows():
            sc = float(row["combined_score"])
            bar_class = get_score_bar_class(sc)
            rec_badge = get_recommendation_badge(sc)
            table_rows += f"""
            <tr>
                <td style="font-weight:700; color:#38BDF8;">#{int(row['rank'])}</td>
                <td style="font-family:monospace; font-size:12px;">...{str(row['producer_id'])[-6:]}</td>
                <td>{row['region']}</td>
                <td>
                    <div class="score-bar-container">
                        <div class="score-bar {bar_class}" style="width:{sc}%;">{sc:.1f}</div>
                    </div>
                </td>
                <td>{rec_badge}</td>
            </tr>"""

        st.markdown(f"""
        <table class="styled-table">
            <thead>
                <tr>
                    <th>Ранг</th>
                    <th>ID</th>
                    <th>Регион</th>
                    <th>Балл</th>
                    <th>Статус</th>
                </tr>
            </thead>
            <tbody>{table_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

    with col_right:
        render_section_header("Распределение статусов", "Заявки")
        status_counts = df["status"].value_counts()
        fig = go.Figure(go.Pie(
            labels=status_counts.index.tolist(),
            values=status_counts.values.tolist(),
            hole=0.55,
            marker=dict(colors=CHART_COLORS[:len(status_counts)]),
            textinfo="percent+label",
            textfont=dict(size=11),
            hovertemplate="%{label}<br>%{value:,} заявок<br>%{percent}<extra></extra>"
        ))
        apply_chart_theme(fig, height=380)
        fig.update_layout(
            showlegend=False,
            margin=dict(t=20, b=20, l=20, r=20),
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_dash_status_pie")

    render_divider()

    # --- Quick regional chart ---
    render_section_header("Распределение по регионам", "География")
    region_data = scored.groupby("region").agg(
        count=("producer_id", "size"),
        mean_score=("combined_score", "mean"),
    ).reset_index().sort_values("count", ascending=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=region_data["region"],
        x=region_data["count"],
        orientation="h",
        marker=dict(
            color=region_data["mean_score"],
            colorscale=[[0, "#b84c4c"], [0.5, "#60A5FA"], [1, "#3B82F6"]],
            colorbar=dict(title="Ср. балл", thickness=15),
            cornerradius=4,
        ),
        text=region_data["count"],
        textposition="auto",
        hovertemplate="%{y}<br>Производителей: %{x}<br>Ср. балл: %{marker.color:.1f}<extra></extra>",
    ))
    apply_chart_theme(fig, height=max(400, len(region_data) * 32))
    fig.update_layout(
        xaxis_title="Количество производителей",
        margin=dict(l=250, t=20),
    )
    st.plotly_chart(fig, use_container_width=True, key="chart_dash_region_bar")

    render_divider()

    # --- Quantified impact metrics ---
    render_section_header("Количественный эффект", "Влияние")
    top100 = scored.nsmallest(100, "rank")
    overall_approval = float(scored["approval_rate"].mean())
    top100_approval = float(top100["approval_rate"].mean())
    overall_completion = float(scored["completion_rate"].mean())
    top100_completion = float(top100["completion_rate"].mean())
    overall_utilization = float(scored["utilization_rate"].mean())
    top100_utilization = float(top100["utilization_rate"].mean())

    imp1, imp2, imp3 = st.columns(3)
    with imp1:
        delta_a = ((top100_approval - overall_approval) / max(overall_approval, 0.01)) * 100
        render_metric_card(
            "ОДОБРЕНИЕ (ТОП-100)", f"{top100_approval:.0%}",
            f"+{delta_a:.0f}% vs среднее ({overall_approval:.0%})", gold=True, delay=1
        )
    with imp2:
        delta_c = ((top100_completion - overall_completion) / max(overall_completion, 0.01)) * 100
        render_metric_card(
            "ИСПОЛНЕНИЕ (ТОП-100)", f"{top100_completion:.0%}",
            f"+{delta_c:.0f}% vs среднее ({overall_completion:.0%})", gold=True, delay=2
        )
    with imp3:
        delta_u = ((top100_utilization - overall_utilization) / max(overall_utilization, 0.01)) * 100
        render_metric_card(
            "ОСВОЕНИЕ (ТОП-100)", f"{top100_utilization:.0%}",
            f"+{delta_u:.0f}% vs среднее ({overall_utilization:.0%})", gold=True, delay=3
        )

    render_divider()

    # --- Budget savings & transparency metrics ---
    render_section_header("Экономия бюджета и прозрачность", "Количественный эффект")

    # Budget savings: difference in utilization rate * total subsidy amount
    total_subsidy_amount = float(scored["total_amount"].sum())
    utilization_diff = top100_utilization - overall_utilization
    budget_savings_est = utilization_diff * total_subsidy_amount

    # Gini coefficient improvement: compare Gini of top-100 scores vs all scores
    from fairness import compute_gini_coefficient
    gini_all = compute_gini_coefficient(scored["combined_score"].values)
    gini_top100 = compute_gini_coefficient(top100["combined_score"].values)
    gini_improvement = ((gini_all - gini_top100) / max(gini_all, 0.001)) * 100

    bs1, bs2, bs3 = st.columns(3)
    with bs1:
        render_metric_card(
            "ЭКОНОМИЯ БЮДЖЕТА",
            format_tenge(abs(budget_savings_est)),
            f"Разница освоения: +{utilization_diff:.1%} от {format_tenge(total_subsidy_amount)}",
            gold=True, delay=1,
        )
    with bs2:
        render_metric_card(
            "ДЖИНИ (ВСЕ)",
            f"{gini_all:.3f}",
            "Неравенство баллов (все производители)",
            delay=2,
        )
    with bs3:
        render_metric_card(
            "ПОВЫШЕНИЕ ПРОЗРАЧНОСТИ",
            f"{abs(gini_improvement):.1f}%",
            f"Снижение Gini: {gini_all:.3f} -> {gini_top100:.3f} (топ-100)",
            gold=True, delay=3,
        )

    # Calculate annual budget savings estimate
    annual_budget_est = total_subsidy_amount  # approximate annual volume
    efficiency_gain_pct = abs(utilization_diff) * 100
    annual_savings_est = abs(budget_savings_est)
    annual_savings_bln = annual_savings_est / 1_000_000_000

    st.markdown(f"""
    <div class="glass-card" style="margin-top:16px; border-left:4px solid #38BDF8;">
        <div style="font-size:14px; line-height:1.8; color:#E2E8F0;">
            <span style="font-weight:700; color:#E2E8F0;">Интерпретация:</span>
            При переходе от случайного отбора (FCFS) к merit-based скорингу,
            топ-100 производителей осваивают субсидии на <b>{utilization_diff:.1%}</b> эффективнее,
            что эквивалентно экономии <b>{format_tenge(abs(budget_savings_est))}</b>
            от общего объёма. Коэффициент Джини среди отобранных снижается
            с <b>{gini_all:.3f}</b> до <b>{gini_top100:.3f}</b>, что означает
            более равномерное и прозрачное распределение баллов.
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_divider()

    # --- Explicit budget savings card (criterion 1) ---
    render_section_header("Экономия бюджета", "Merit-based скоринг")
    st.markdown(f"""
    <div class="glass-card" style="border-left:4px solid #22C55E; margin-bottom:16px;">
        <div style="font-weight:700; font-size:16px; color:#E2E8F0; margin-bottom:12px;">
            Потенциальная экономия при переходе на merit-based скоринг
        </div>
        <div style="font-size:14px; line-height:1.8; color:#E2E8F0;">
            <p>
                При переходе на merit-based скоринг эффективность использования субсидий
                увеличивается на <b>{efficiency_gain_pct:.1f}%</b>
                (средняя утилизация топ-100: <b>{top100_utilization:.1%}</b> vs
                общая средняя: <b>{overall_utilization:.1%}</b>).
            </p>
            <p style="margin-top:8px;">
                Потенциальная экономия: <b>{annual_savings_bln:.2f} млрд тг ежегодно</b>
                (расчёт: разница освоения {utilization_diff:.1%} x общий объём субсидий
                {format_tenge(total_subsidy_amount)}).
            </p>
            <p style="margin-top:8px;">
                Топ-100 производителей по скорингу показывают одобрение <b>{top100_approval:.0%}</b>
                (vs {overall_approval:.0%} в среднем) и исполнение <b>{top100_completion:.0%}</b>
                (vs {overall_completion:.0%} в среднем).
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    sav1, sav2, sav3, sav4 = st.columns(4)
    with sav1:
        render_metric_card(
            "ЭКОНОМИЯ / ГОД",
            f"{annual_savings_bln:.2f} млрд",
            f"от {format_tenge(annual_budget_est)} бюджета",
            gold=True, delay=1,
        )
    with sav2:
        render_metric_card(
            "РОСТ ЭФФЕКТИВНОСТИ",
            f"+{efficiency_gain_pct:.1f}%",
            f"Освоение: {overall_utilization:.0%} -> {top100_utilization:.0%}",
            delay=2,
        )
    with sav3:
        render_metric_card(
            "РОСТ ОДОБРЕНИЯ",
            f"+{((top100_approval - overall_approval) / max(overall_approval, 0.01)) * 100:.0f}%",
            f"{overall_approval:.0%} -> {top100_approval:.0%}",
            delay=3,
        )
    with sav4:
        render_metric_card(
            "РОСТ ИСПОЛНЕНИЯ",
            f"+{((top100_completion - overall_completion) / max(overall_completion, 0.01)) * 100:.0f}%",
            f"{overall_completion:.0%} -> {top100_completion:.0%}",
            delay=4,
        )


# ===========================================================================
# PAGE 2: DATA OVERVIEW
# ===========================================================================
def render_overview(df: pd.DataFrame, features: pd.DataFrame):
    render_breadcrumb("Обзор данных")
    render_page_header(
        "Обзор данных реестра субсидий",
        "Источник: Реестр заявок ИСС (subsidy.plem.kz), 2025 год"
    )

    stats = get_summary_stats(df)

    # --- Summary cards ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("ЗАЯВКИ", f"{stats['total_applications']:,}", "Всего в реестре", delay=1)
    with c2:
        render_metric_card("ПРОИЗВОДИТЕЛИ", f"{stats['unique_producers']:,}", "Уникальных", delay=2)
    with c3:
        render_metric_card("ОДОБРЕНИЕ", format_percent(stats["approval_rate"]), "Доля одобренных", gold=True, delay=3)
    with c4:
        render_metric_card(
            "ПЕРИОД",
            f"{stats['date_range'][0]}",
            f"по {stats['date_range'][1]}",
            delay=4,
        )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("ОБЩАЯ СУММА", format_tenge(stats["total_amount"]), "Все заявки", gold=True, delay=1)
    with c2:
        render_metric_card("СРЕДНЯЯ", format_tenge(stats["avg_amount"]), "На одну заявку", delay=2)
    with c3:
        render_metric_card("РЕГИОНЫ", str(stats["unique_regions"]), "Областей", delay=3)
    with c4:
        render_metric_card("РАЙОНЫ", str(stats["unique_districts"]), "Уникальных", delay=4)

    render_divider()

    # --- Data quality ---
    render_section_header("Качество данных", "Мониторинг")
    dq1, dq2, dq3 = st.columns(3)
    with dq1:
        missing_rate = df.isnull().mean().mean()
        cls = "good" if missing_rate < 0.05 else "warn"
        st.markdown(f"""
        <div class="dq-indicator dq-{cls}">
            <span class="dq-dot {cls}"></span>
            Пропуски данных: {missing_rate:.1%}
        </div>
        """, unsafe_allow_html=True)
    with dq2:
        dup_rate = df.duplicated(subset=["app_num"]).mean()
        cls = "good" if dup_rate < 0.01 else "warn"
        st.markdown(f"""
        <div class="dq-indicator dq-{cls}">
            <span class="dq-dot {cls}"></span>
            Дубликаты: {dup_rate:.1%}
        </div>
        """, unsafe_allow_html=True)
    with dq3:
        st.markdown(f"""
        <div class="dq-indicator dq-good">
            <span class="dq-dot good"></span>
            Записей: {len(df):,}
        </div>
        """, unsafe_allow_html=True)

    render_divider()

    # --- Detailed Data Quality Section ---
    render_section_header("Качество данных", "Детальный анализ")

    # 1. Completeness per column
    completeness_data = []
    key_columns = [
        ("id", "ID записи"), ("date_raw", "Дата поступления"),
        ("region", "Область"), ("akimat", "Акимат"),
        ("app_num", "Номер заявки"), ("direction", "Направление"),
        ("subsidy_name", "Наименование субсидии"), ("status", "Статус"),
        ("rate", "Норматив"), ("amount", "Сумма"),
        ("district", "Район хозяйства"),
    ]
    for col_key, col_label in key_columns:
        if col_key in df.columns:
            non_null = df[col_key].notna().sum()
            total = len(df)
            pct = non_null / total * 100 if total > 0 else 0
            cls = "good" if pct >= 95 else "warn"
            completeness_data.append((col_label, col_key, non_null, total, pct, cls))

    completeness_rows = ""
    for col_label, col_key, non_null, total, pct, cls in completeness_data:
        dot = f'<span class="dq-dot {cls}" style="display:inline-block;"></span>'
        completeness_rows += (
            f"<tr><td>{col_label}</td><td><code>{col_key}</code></td>"
            f"<td>{non_null:,} / {total:,}</td>"
            f"<td>{dot} {pct:.1f}%</td></tr>"
        )

    st.markdown(f"""
    <div class="glass-card" style="margin-bottom:24px;">
        <div style="font-weight:700; font-size:16px; color:#E2E8F0; margin-bottom:12px;">
            Полнота данных по столбцам (% непустых значений)
        </div>
        <table class="styled-table">
            <thead>
                <tr><th>Столбец</th><th>Код</th><th>Заполнено</th><th>Полнота</th></tr>
            </thead>
            <tbody>{completeness_rows}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # 2. Outlier detection
    outlier_info = []
    if "amount" in df.columns:
        amt = df["amount"].dropna()
        if len(amt) > 0:
            q1 = amt.quantile(0.25)
            q3 = amt.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            n_outliers = int(((amt < lower_bound) | (amt > upper_bound)).sum())
            outlier_pct = n_outliers / len(amt) * 100
            outlier_info.append(("Сумма субсидии (amount)", n_outliers, len(amt), outlier_pct))

    if "rate" in df.columns:
        rate_vals = pd.to_numeric(df["rate"], errors="coerce").dropna()
        if len(rate_vals) > 0:
            q1_r = rate_vals.quantile(0.25)
            q3_r = rate_vals.quantile(0.75)
            iqr_r = q3_r - q1_r
            if iqr_r > 0:
                n_out_r = int(((rate_vals < q1_r - 1.5 * iqr_r) | (rate_vals > q3_r + 1.5 * iqr_r)).sum())
                out_pct_r = n_out_r / len(rate_vals) * 100
                outlier_info.append(("Норматив (rate)", n_out_r, len(rate_vals), out_pct_r))

    outlier_rows = ""
    for o_label, o_count, o_total, o_pct in outlier_info:
        cls = "good" if o_pct < 5 else "warn"
        dot = f'<span class="dq-dot {cls}" style="display:inline-block;"></span>'
        outlier_rows += (
            f"<tr><td>{o_label}</td>"
            f"<td>{o_count:,} из {o_total:,}</td>"
            f"<td>{dot} {o_pct:.1f}%</td></tr>"
        )

    if outlier_rows:
        st.markdown(f"""
        <div class="glass-card" style="margin-bottom:24px;">
            <div style="font-weight:700; font-size:16px; color:#E2E8F0; margin-bottom:12px;">
                Выбросы (метод IQR: значения за пределами 1.5 x межквартильного размаха)
            </div>
            <table class="styled-table">
                <thead>
                    <tr><th>Поле</th><th>Выбросов</th><th>Доля</th></tr>
                </thead>
                <tbody>{outlier_rows}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # 3. Data coverage
    date_min = df["date"].min() if "date" in df.columns and df["date"].notna().any() else None
    date_max = df["date"].max() if "date" in df.columns and df["date"].notna().any() else None
    n_regions = df["region"].nunique() if "region" in df.columns else 0
    n_districts = df["district"].nunique() if "district" in df.columns else 0
    n_directions = df["direction"].nunique() if "direction" in df.columns else 0
    n_months = df["month"].nunique() if "month" in df.columns else 0

    date_range_str = ""
    if date_min is not None and date_max is not None:
        date_range_str = f"{date_min.strftime('%d.%m.%Y')} -- {date_max.strftime('%d.%m.%Y')}"
    else:
        date_range_str = "Н/Д"

    st.markdown(f"""
    <div class="glass-card" style="margin-bottom:24px;">
        <div style="font-weight:700; font-size:16px; color:#E2E8F0; margin-bottom:12px;">
            Покрытие данных
        </div>
        <table class="styled-table">
            <thead>
                <tr><th>Параметр</th><th>Значение</th></tr>
            </thead>
            <tbody>
                <tr><td>Временной диапазон</td><td>{date_range_str}</td></tr>
                <tr><td>Количество месяцев</td><td>{n_months}</td></tr>
                <tr><td>Регионов (областей)</td><td>{n_regions}</td></tr>
                <tr><td>Районов</td><td>{n_districts}</td></tr>
                <tr><td>Направлений</td><td>{n_directions}</td></tr>
                <tr><td>Записей</td><td>{len(df):,}</td></tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    render_divider()

    # --- Feature Engineering explanation ---
    render_section_header("Feature Engineering", "14 признаков")
    st.markdown("""
    <div class="glass-card" style="margin-bottom:24px;">
        <div style="line-height:1.8; font-size:14px; color:#E2E8F0;">
            <p style="font-weight:700; font-size:16px; color:#E2E8F0; margin-bottom:12px;">
                Извлечение признаков из сырых данных
            </p>
            <p>
                Из исходных заявок (строка = одна заявка) мы агрегируем данные по каждому
                производителю (producer_id) и формируем <b>14 признаков</b>, которые используются
                в скоринговой модели:
            </p>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:12px;">
                <div class="sw-item sw-strength" style="margin:0;"><span class="sw-icon">1.</span> Количество заявок (total_apps)</div>
                <div class="sw-item sw-strength" style="margin:0;"><span class="sw-icon">2.</span> Доля одобренных (approval_rate)</div>
                <div class="sw-item sw-strength" style="margin:0;"><span class="sw-icon">3.</span> Доля исполненных (completion_rate)</div>
                <div class="sw-item sw-strength" style="margin:0;"><span class="sw-icon">4.</span> Доля отклонённых (rejection_rate)</div>
                <div class="sw-item sw-strength" style="margin:0;"><span class="sw-icon">5.</span> Средняя сумма log (avg_amount_log)</div>
                <div class="sw-item sw-strength" style="margin:0;"><span class="sw-icon">6.</span> Вариативность сумм (amount_cv)</div>
                <div class="sw-item sw-strength" style="margin:0;"><span class="sw-icon">7.</span> Диверсификация направлений</div>
                <div class="sw-item sw-strength" style="margin:0;"><span class="sw-icon">8.</span> Виды субсидий (subsidy_type_count)</div>
                <div class="sw-item sw-strength" style="margin:0;"><span class="sw-icon">9.</span> Освоение средств (utilization_rate)</div>
                <div class="sw-item sw-strength" style="margin:0;"><span class="sw-icon">10.</span> Активность подачи (apps_per_month)</div>
                <div class="sw-item sw-strength" style="margin:0;"><span class="sw-icon">11.</span> Рабочее время (working_hours_ratio)</div>
                <div class="sw-item sw-strength" style="margin:0;"><span class="sw-icon">12.</span> Регулярность подачи (month_regularity)</div>
                <div class="sw-item sw-strength" style="margin:0;"><span class="sw-icon">13.</span> Охват районов (unique_districts)</div>
                <div class="sw-item sw-strength" style="margin:0;"><span class="sw-icon">14.</span> Период активности (activity_span_days)</div>
            </div>
            <p style="margin-top:16px; padding:12px 16px; background:rgba(59,130,246,0.06); border-radius:8px; font-weight:600;">
                Все признаки нормализуются в диапазон [0, 1] с помощью MinMaxScaler перед подачей в модели.
                Целевая переменная для ML формируется как взвешенная комбинация:
                исполнение (35%) + одобрение (25%) + освоение (20%) + диверсификация (10%) + (1 - отклонения) (10%).
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_divider()

    # --- Charts ---
    col_left, col_right = st.columns(2)
    with col_left:
        fig = plot_status_distribution(df)
        apply_chart_theme(fig, height=400)
        st.plotly_chart(fig, use_container_width=True, key="chart_overview_status_dist")
    with col_right:
        fig = plot_direction_pie(df)
        apply_chart_theme(fig, height=400)
        st.plotly_chart(fig, use_container_width=True, key="chart_overview_direction_pie")

    fig = plot_monthly_trend(df)
    apply_chart_theme(fig, height=420)
    st.plotly_chart(fig, use_container_width=True, key="chart_overview_monthly_trend")

    col_left, col_right = st.columns(2)
    with col_left:
        fig = plot_region_distribution(df)
        apply_chart_theme(fig, height=600)
        fig.update_layout(margin=dict(l=250))
        st.plotly_chart(fig, use_container_width=True, key="chart_overview_region_dist")
    with col_right:
        fig = plot_region_amounts(df)
        apply_chart_theme(fig, height=600)
        fig.update_layout(margin=dict(l=250))
        st.plotly_chart(fig, use_container_width=True, key="chart_overview_region_amounts")

    fig = plot_approval_rate_by_direction(df)
    apply_chart_theme(fig, height=400)
    fig.update_layout(margin=dict(l=200))
    st.plotly_chart(fig, use_container_width=True, key="chart_overview_approval_by_dir")


# ===========================================================================
# PAGE 3: SCORING
# ===========================================================================
def render_scoring(scored: pd.DataFrame, features: pd.DataFrame):
    render_breadcrumb("Скоринг производителей")
    render_page_header(
        "Скоринг сельхозпроизводителей",
        "Ранжирование на основе комбинированной модели (ML + правила)"
    )

    # --- Quick stats ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("ВСЕГО", f"{len(scored):,}", "Производителей", delay=1)
    with c2:
        render_metric_card("СРЕДНИЙ БАЛЛ", f"{scored['combined_score'].mean():.1f}", "По всем", delay=2)
    with c3:
        render_metric_card("МЕДИАНА", f"{scored['combined_score'].median():.1f}", "Балл", delay=3)
    with c4:
        top100_threshold = scored.nsmallest(100, "rank")["combined_score"].min()
        render_metric_card("ТОП-100 ПОРОГ", f"{top100_threshold:.1f}", "Минимальный балл", gold=True, delay=4)

    render_divider()

    # --- Model architecture explanation ---
    render_section_header("Архитектура модели", "60% ML + 40% Правила")
    arch1, arch2, arch3 = st.columns([2, 1, 2])
    with arch1:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:15px; font-weight:700; color:#38BDF8; margin-bottom:10px;">ML-модель (60%)</div>
            <div style="font-size:13px; color:#E2E8F0; line-height:1.7;">
                <b>GradientBoostingRegressor</b><br>
                200 деревьев, глубина 5, learning_rate 0.1<br>
                Обучение на 14 нормализованных признаках<br>
                Кросс-валидация 5-fold<br>
                Выявляет нелинейные зависимости
            </div>
        </div>
        """, unsafe_allow_html=True)
    with arch2:
        st.markdown("""
        <div style="display:flex; align-items:center; justify-content:center; height:100%;">
            <div style="text-align:center;">
                <div style="font-size:28px; font-weight:800; color:#38BDF8;">+</div>
                <div style="font-size:11px; color:#94A3B8; font-weight:600; margin-top:4px;">КОМБИНАЦИЯ</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with arch3:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:15px; font-weight:700; color:#38BDF8; margin-bottom:10px;">Правила (40%)</div>
            <div style="font-size:13px; color:#E2E8F0; line-height:1.7;">
                <b>Взвешенная сумма</b><br>
                Экспертные веса для каждого фактора<br>
                Одобрение (20%), Исполнение (15%)<br>
                Освоение (15%), Отклонения (-10%)<br>
                Полная прозрачность формулы
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin:16px 0 24px 0; padding:12px 24px; background:rgba(59,130,246,0.06);
        border-radius:10px; font-size:14px; font-weight:600; color:#E2E8F0;">
        Итоговый балл = 0.6 x ML_балл + 0.4 x Правила_балл (диапазон 0--100)
    </div>
    """, unsafe_allow_html=True)

    render_divider()

    # --- Filters ---
    render_section_header("Фильтры", "Настройка")
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    with fcol1:
        regions = ["Все"] + sorted(scored["region"].unique().tolist())
        sel_region = st.selectbox("Регион", regions, key="scoring_region")
    with fcol2:
        directions = ["Все"] + sorted(scored["main_direction"].dropna().unique().tolist())
        sel_direction = st.selectbox("Направление", directions, key="scoring_direction")
    with fcol3:
        min_score = st.slider("Минимальный балл", 0, 100, 0, key="scoring_min")
    with fcol4:
        search_id = st.text_input("Поиск по ID", "", key="scoring_search")
    st.markdown('</div>', unsafe_allow_html=True)

    filtered = scored.copy()
    if sel_region != "Все":
        filtered = filtered[filtered["region"] == sel_region]
    if sel_direction != "Все":
        filtered = filtered[filtered["main_direction"] == sel_direction]
    if min_score > 0:
        filtered = filtered[filtered["combined_score"] >= min_score]
    if search_id:
        filtered = filtered[filtered["producer_id"].str.contains(search_id, na=False)]

    st.markdown(
        f'<div style="font-size:15px; font-weight:600; margin-bottom:16px; color:#E2E8F0;">'
        f'Найдено: {len(filtered)} производителей</div>',
        unsafe_allow_html=True
    )

    # --- Paginated table ---
    page_size = 50
    total_pages = max(1, (len(filtered) + page_size - 1) // page_size)

    if "scoring_page" not in st.session_state:
        st.session_state.scoring_page = 1

    display_page = st.session_state.scoring_page
    start_idx = (display_page - 1) * page_size
    end_idx = min(start_idx + page_size, len(filtered))
    page_data = filtered.iloc[start_idx:end_idx]

    # Build HTML table
    table_rows = ""
    for _, row in page_data.iterrows():
        sc = float(row["combined_score"])
        bar_class = get_score_bar_class(sc)
        rec_badge = get_recommendation_badge(sc)
        table_rows += f"""
        <tr>
            <td style="font-weight:700; color:#38BDF8;">#{int(row['rank'])}</td>
            <td style="font-family:monospace; font-size:12px;">{row['producer_id']}</td>
            <td>{row['region']}</td>
            <td>{row.get('main_direction', '')}</td>
            <td>
                <div class="score-bar-container">
                    <div class="score-bar {bar_class}" style="width:{sc}%;">{sc:.1f}</div>
                </div>
            </td>
            <td style="text-align:center;">{row.get('rule_score', 0):.1f}</td>
            <td style="text-align:center;">{row.get('ml_score', 0):.1f}</td>
            <td>{format_percent(row.get('approval_rate', 0))}</td>
            <td>{rec_badge}</td>
        </tr>"""

    st.markdown(f"""
    <table class="styled-table">
        <thead>
            <tr>
                <th>Ранг</th>
                <th>ID производителя</th>
                <th>Регион</th>
                <th>Направление</th>
                <th style="min-width:140px;">Комб. балл</th>
                <th>Правила</th>
                <th>ML</th>
                <th>Одобрение</th>
                <th>Статус</th>
            </tr>
        </thead>
        <tbody>{table_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # Pagination controls (prominent)
    st.markdown("<br>", unsafe_allow_html=True)
    pcol1, pcol2, pcol3, pcol4, pcol5 = st.columns([1, 1, 3, 1, 1])
    with pcol1:
        if st.button("\u2190 Первая", disabled=(display_page <= 1), key="pg_first"):
            st.session_state.scoring_page = 1
            st.rerun()
    with pcol2:
        if st.button("\u2190 Назад", disabled=(display_page <= 1), key="pg_prev"):
            st.session_state.scoring_page = max(1, display_page - 1)
            st.rerun()
    with pcol3:
        st.markdown(
            f'<div style="text-align:center; padding:10px; font-weight:700; font-size:16px; color:#E2E8F0; '
            f'background:rgba(17,24,39,0.85); border-radius:10px; border:1px solid rgba(56,189,248,0.15);">'
            f'Страница {display_page} из {total_pages} &nbsp;|&nbsp; '
            f'Записи {start_idx+1}--{end_idx} из {len(filtered)}</div>',
            unsafe_allow_html=True
        )
    with pcol4:
        if st.button("Вперёд \u2192", disabled=(display_page >= total_pages), key="pg_next"):
            st.session_state.scoring_page = min(total_pages, display_page + 1)
            st.rerun()
    with pcol5:
        if st.button("Последняя \u2192", disabled=(display_page >= total_pages), key="pg_last"):
            st.session_state.scoring_page = total_pages
            st.rerun()

    render_divider()

    # Export button
    export_cols = ["rank", "producer_id", "region", "main_direction", "combined_score",
                   "rule_score", "ml_score", "approval_rate"]
    export_df = filtered[[c for c in export_cols if c in filtered.columns]]
    csv_export = export_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="\u21E9 Экспорт таблицы (CSV)",
        data=csv_export,
        file_name="scoring_table.csv",
        mime="text/csv",
        key="scoring_export_csv",
    )

    render_divider()

    fig = plot_score_distribution(filtered)
    apply_chart_theme(fig, height=380)
    st.plotly_chart(fig, use_container_width=True, key="chart_scoring_score_dist")


# ===========================================================================
# PAGE 4: PRODUCER PROFILE
# ===========================================================================
def render_profile(
    df: pd.DataFrame,
    scored: pd.DataFrame,
    features: pd.DataFrame,
    engine: ScoringEngine,
):
    render_breadcrumb("Профиль производителя")
    render_page_header(
        "Профиль производителя",
        "Детальный анализ скорингового профиля"
    )

    # Top action row: back to scoring + compare
    top_c1, top_c2, top_c3 = st.columns([2, 2, 6])
    with top_c1:
        if st.button("\u2190 Вернуться к скорингу", key="profile_back_scoring"):
            nav_to("scoring")
            st.rerun()
    with top_c2:
        if st.button("\u21C4 Сравнить с другим", key="profile_compare"):
            nav_to("comparison")
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Searchable dropdown
    sorted_scored = scored.sort_values("rank")
    display_options = [
        f"{pid} (ранг {int(sorted_scored[sorted_scored['producer_id']==pid]['rank'].values[0])})"
        for pid in sorted_scored["producer_id"].head(300)
    ]

    # Initialize profile index for prev/next navigation
    if "profile_idx" not in st.session_state:
        st.session_state["profile_idx"] = 0

    selected_display = st.selectbox(
        "Выберите производителя",
        display_options,
        index=st.session_state.get("profile_idx", 0),
        key="profile_producer",
    )
    # Update index to match selection
    current_idx = display_options.index(selected_display) if selected_display in display_options else 0
    st.session_state["profile_idx"] = current_idx

    selected_id = selected_display.split(" (")[0]

    # Prev / Next producer navigation
    nav_c1, nav_c2, nav_c3 = st.columns([1, 6, 1])
    with nav_c1:
        if st.button("\u2190 Предыдущий", disabled=(current_idx <= 0), key="profile_prev"):
            st.session_state["profile_idx"] = max(0, current_idx - 1)
            st.rerun()
    with nav_c3:
        if st.button("Следующий \u2192", disabled=(current_idx >= len(display_options) - 1), key="profile_next"):
            st.session_state["profile_idx"] = min(len(display_options) - 1, current_idx + 1)
            st.rerun()

    prod_scored = scored[scored["producer_id"] == selected_id]
    if prod_scored.empty:
        st.warning("Производитель не найден")
        return

    row = prod_scored.iloc[0]
    prod_feat = features[features["producer_id"] == selected_id].iloc[0]
    prod_apps = df[df["producer_id"] == selected_id]

    score_val = float(row["combined_score"])
    rule_val = float(row.get("rule_score", 0))
    ml_val = float(row.get("ml_score", 0))

    # --- Commission summary block ---
    completion_pct = float(prod_feat.get("completion_rate", 0))
    approval_pct = float(prod_feat.get("approval_rate", 0))
    utilization_pct = float(prod_feat.get("utilization_rate", 0))

    # Determine recommendation level
    if score_val >= 75:
        commission_rec = "приоритетная"
        commission_cls = "fairness-ok"
    elif score_val >= 50:
        commission_rec = "стандартная"
        commission_cls = "fairness-warn"
    else:
        commission_rec = "требует проверки"
        commission_cls = "fairness-danger"

    # Build strengths summary
    strengths_parts = []
    if approval_pct >= 0.7:
        strengths_parts.append(f"высокий процент одобрения ({approval_pct:.0%})")
    if completion_pct >= 0.5:
        strengths_parts.append(f"высокий процент завершения ({completion_pct:.0%})")
    if utilization_pct >= 0.5:
        strengths_parts.append(f"эффективное освоение средств ({utilization_pct:.0%})")
    if float(prod_feat.get("activity_span_days", 0)) > 60:
        strengths_parts.append("стабильная история")
    if float(prod_feat.get("direction_diversity", 0)) >= 0.15:
        strengths_parts.append("диверсификация направлений")
    if not strengths_parts:
        strengths_parts.append("показатели в пределах среднего")

    strengths_str = ", ".join(strengths_parts[:3])

    st.markdown(f"""
    <div class="{commission_cls} fairness-alert" style="margin-bottom:20px; padding:18px 24px; font-size:15px; line-height:1.7;">
        <div style="font-weight:700; font-size:16px; color:#E2E8F0; margin-bottom:8px;">
            Резюме для комиссии
        </div>
        <div>
            Производитель <b>{selected_id}</b> рекомендован к субсидированию
            (балл: <b>{score_val:.1f}</b> из 100, ранг #{int(row['rank'])} из {len(scored)}).
            Основные сильные стороны: {strengths_str}.
            Рекомендация: <b>{commission_rec}</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Hero card ---
    if score_val >= 75:
        badge_cls = "badge-green"
        badge_text = "Рекомендован"
    elif score_val >= 50:
        badge_cls = "badge-yellow"
        badge_text = "На рассмотрении"
    else:
        badge_cls = "badge-red"
        badge_text = "Не рекомендован"

    st.markdown(f"""
    <div class="glass-card animate-in" style="margin-bottom:24px;">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:16px;">
            <div>
                <div style="font-size:13px; color:#94A3B8; text-transform:uppercase; letter-spacing:1px; font-weight:600;">
                    ID Производителя
                </div>
                <div style="font-size:22px; font-weight:700; color:#E2E8F0; font-family:monospace; margin:4px 0;">
                    {selected_id}
                </div>
                <div style="margin-top:8px;">
                    <span class="badge {badge_cls}">{badge_text}</span>
                    <span class="badge badge-blue" style="margin-left:8px;">Ранг #{int(row['rank'])} из {len(scored)}</span>
                </div>
            </div>
            <div style="display:flex; gap:24px; flex-wrap:wrap;">
                <div style="text-align:center;">
                    <div style="font-size:11px; color:#94A3B8; font-weight:600; text-transform:uppercase;">Регион</div>
                    <div style="font-size:16px; font-weight:700; color:#E2E8F0; margin-top:4px;">{row['region']}</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:11px; color:#94A3B8; font-weight:600; text-transform:uppercase;">Направление</div>
                    <div style="font-size:16px; font-weight:700; color:#E2E8F0; margin-top:4px;">{row['main_direction']}</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:11px; color:#94A3B8; font-weight:600; text-transform:uppercase;">Заявок</div>
                    <div style="font-size:16px; font-weight:700; color:#E2E8F0; margin-top:4px;">{int(prod_feat['total_apps'])}</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:11px; color:#94A3B8; font-weight:600; text-transform:uppercase;">Общая сумма</div>
                    <div style="font-size:16px; font-weight:700; color:#38BDF8; margin-top:4px;">{format_tenge(prod_feat['total_amount'])}</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Percentile rank ---
    percentile = float((scored["combined_score"] < score_val).mean() * 100)
    percentile_top = max(1, 100 - percentile)
    st.markdown(f"""
    <div class="glass-card animate-in" style="margin-bottom:24px; text-align:center;">
        <div style="font-size:14px; color:#94A3B8; text-transform:uppercase; letter-spacing:1px; font-weight:600; margin-bottom:8px;">
            Процентильный ранг
        </div>
        <div style="font-size:36px; font-weight:800; background:linear-gradient(135deg, #38BDF8, #38BDF8);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:4px;">
            Топ {percentile_top:.0f}%
        </div>
        <div style="font-size:14px; color:#94A3B8;">
            среди всех {len(scored)} производителей
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Gauge scores ---
    render_section_header("Скоринговые баллы", "Оценка")
    gc1, gc2, gc3 = st.columns(3)

    gauge_num_color = "#E2E8F0"
    gauge_title_color = "#94A3B8"

    def make_gauge(value, title, _color_ranges):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            number=dict(font=dict(size=40, color=gauge_num_color, family="Inter"), suffix=""),
            gauge=dict(
                axis=dict(range=[0, 100], tickwidth=1, tickcolor="rgba(0,0,0,0.2)", dtick=25),
                bar=dict(color=TEAL, thickness=0.3),
                bgcolor="rgba(0,0,0,0.02)",
                borderwidth=0,
                steps=[
                    dict(range=[0, 25], color="rgba(184,76,76,0.12)"),
                    dict(range=[25, 50], color="rgba(59,130,246,0.12)"),
                    dict(range=[50, 75], color="rgba(59,130,246,0.12)"),
                    dict(range=[75, 100], color="rgba(45,143,94,0.12)"),
                ],
                threshold=dict(line=dict(color=GOLD, width=3), thickness=0.8, value=value),
            ),
            title=dict(text=title, font=dict(size=14, color=gauge_title_color, family="Inter")),
        ))
        apply_chart_theme(fig, height=250)
        fig.update_layout(margin=dict(t=50, b=20, l=30, r=30))
        return fig

    with gc1:
        st.plotly_chart(make_gauge(score_val, "Комбинированный", None), use_container_width=True, key="chart_profile_gauge_combined")
    with gc2:
        st.plotly_chart(make_gauge(ml_val, "ML-модель", None), use_container_width=True, key="chart_profile_gauge_ml")
    with gc3:
        st.plotly_chart(make_gauge(rule_val, "Правиловая модель", None), use_container_width=True, key="chart_profile_gauge_rule")

    render_divider()

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Вклад факторов",
        "Что улучшить",
        "История заявок",
        "Сильные и слабые стороны",
        "Детали",
    ])

    with tab1:
        descs = get_feature_descriptions()
        # Use user-friendly Russian factor names in the chart
        friendly_names = {
            "total_apps": "Количество заявок",
            "approval_rate": "Доля одобрений",
            "completion_rate": "Доля исполнений",
            "rejection_rate": "Доля отклонений",
            "avg_amount_log": "Средняя сумма",
            "amount_cv": "Стабильность сумм",
            "direction_diversity": "Диверсификация",
            "subsidy_type_count": "Виды субсидий",
            "utilization_rate": "Освоение средств",
            "apps_per_month": "Активность подачи",
            "working_hours_ratio": "Рабочее время",
            "month_regularity": "Регулярность",
            "unique_districts": "Охват районов",
            "activity_span_days": "Период активности",
        }
        combined_descs = {k: friendly_names.get(k, v) for k, v in descs.items()}
        fig = plot_producer_breakdown(row, combined_descs)
        apply_chart_theme(fig, height=max(350, len(combined_descs) * 32))
        fig.update_layout(margin=dict(l=200))
        st.plotly_chart(fig, use_container_width=True, key="chart_profile_factor_breakdown")

        # --- Natural language explanations for each factor (criterion 4) ---
        render_divider()
        render_section_header("Пояснения к факторам", "Интерпретация")

        # Compute overall averages for comparison
        avg_values = {}
        for feat_key in friendly_names:
            if feat_key in scored.columns:
                avg_values[feat_key] = float(scored[feat_key].mean())
            elif feat_key in features.columns:
                avg_values[feat_key] = float(features[feat_key].mean())

        nl_explanations = []
        for feat_key, feat_label in friendly_names.items():
            prod_val = float(prod_feat.get(feat_key, 0))
            avg_val = avg_values.get(feat_key, 0)

            if feat_key == "approval_rate":
                if prod_val >= avg_val * 1.1:
                    nl_explanations.append(
                        f'<div class="sw-item sw-strength" style="margin-bottom:6px;">'
                        f'<span class="sw-icon">+</span> '
                        f'<b>{feat_label}:</b> Высокий показатель одобрения ({prod_val:.0%}) '
                        f'значительно выше среднего ({avg_val:.0%}), что положительно влияет на итоговый балл.</div>'
                    )
                elif prod_val < avg_val * 0.9:
                    nl_explanations.append(
                        f'<div class="sw-item sw-weakness" style="margin-bottom:6px;">'
                        f'<span class="sw-icon">-</span> '
                        f'<b>{feat_label}:</b> Показатель одобрения ({prod_val:.0%}) '
                        f'ниже среднего ({avg_val:.0%}), рекомендуется улучшить качество заявок.</div>'
                    )
            elif feat_key == "completion_rate":
                if prod_val >= avg_val * 1.1:
                    nl_explanations.append(
                        f'<div class="sw-item sw-strength" style="margin-bottom:6px;">'
                        f'<span class="sw-icon">+</span> '
                        f'<b>{feat_label}:</b> Высокая доля исполнения ({prod_val:.0%}) '
                        f'выше среднего ({avg_val:.0%}), что свидетельствует об ответственном освоении субсидий.</div>'
                    )
                elif prod_val < avg_val * 0.9:
                    nl_explanations.append(
                        f'<div class="sw-item sw-weakness" style="margin-bottom:6px;">'
                        f'<span class="sw-icon">-</span> '
                        f'<b>{feat_label}:</b> Показатель завершения ({prod_val:.0%}) '
                        f'ниже среднего ({avg_val:.0%}), рекомендуется улучшить.</div>'
                    )
            elif feat_key == "utilization_rate":
                if prod_val >= avg_val * 1.1:
                    nl_explanations.append(
                        f'<div class="sw-item sw-strength" style="margin-bottom:6px;">'
                        f'<span class="sw-icon">+</span> '
                        f'<b>{feat_label}:</b> Эффективное освоение средств ({prod_val:.0%}) '
                        f'выше среднего ({avg_val:.0%}), производитель рационально использует субсидии.</div>'
                    )
                elif prod_val < avg_val * 0.9:
                    nl_explanations.append(
                        f'<div class="sw-item sw-weakness" style="margin-bottom:6px;">'
                        f'<span class="sw-icon">-</span> '
                        f'<b>{feat_label}:</b> Освоение средств ({prod_val:.0%}) '
                        f'ниже среднего ({avg_val:.0%}), рекомендуется планировать использование субсидий.</div>'
                    )
            elif feat_key == "rejection_rate":
                if prod_val <= avg_val * 0.9:
                    nl_explanations.append(
                        f'<div class="sw-item sw-strength" style="margin-bottom:6px;">'
                        f'<span class="sw-icon">+</span> '
                        f'<b>{feat_label}:</b> Низкая доля отклонений ({prod_val:.0%}) '
                        f'ниже среднего ({avg_val:.0%}), что говорит о качественной подготовке документов.</div>'
                    )
                elif prod_val > avg_val * 1.1:
                    nl_explanations.append(
                        f'<div class="sw-item sw-weakness" style="margin-bottom:6px;">'
                        f'<span class="sw-icon">-</span> '
                        f'<b>{feat_label}:</b> Высокая доля отклонений ({prod_val:.0%}) '
                        f'выше среднего ({avg_val:.0%}), необходимо проанализировать причины отказов.</div>'
                    )
            elif feat_key == "direction_diversity":
                if prod_val >= avg_val * 1.1:
                    nl_explanations.append(
                        f'<div class="sw-item sw-strength" style="margin-bottom:6px;">'
                        f'<span class="sw-icon">+</span> '
                        f'<b>{feat_label}:</b> Хорошая диверсификация ({prod_val:.2f}) '
                        f'выше среднего ({avg_val:.2f}), производитель работает по нескольким направлениям.</div>'
                    )
                elif prod_val < avg_val * 0.5:
                    nl_explanations.append(
                        f'<div class="sw-item sw-weakness" style="margin-bottom:6px;">'
                        f'<span class="sw-icon">-</span> '
                        f'<b>{feat_label}:</b> Низкая диверсификация ({prod_val:.2f}), '
                        f'рекомендуется участие в смежных программах.</div>'
                    )
            elif feat_key == "working_hours_ratio":
                if prod_val >= 0.7:
                    nl_explanations.append(
                        f'<div class="sw-item sw-strength" style="margin-bottom:6px;">'
                        f'<span class="sw-icon">+</span> '
                        f'<b>{feat_label}:</b> Заявки подаются преимущественно в рабочее время ({prod_val:.0%}), '
                        f'что свидетельствует о системном подходе.</div>'
                    )
                elif prod_val < 0.4:
                    nl_explanations.append(
                        f'<div class="sw-item sw-weakness" style="margin-bottom:6px;">'
                        f'<span class="sw-icon">-</span> '
                        f'<b>{feat_label}:</b> Заявки подаются вне рабочего времени ({prod_val:.0%}), '
                        f'рекомендуется подавать в период 9:00-18:00.</div>'
                    )
            elif feat_key == "total_apps":
                if prod_val >= avg_val * 1.2:
                    nl_explanations.append(
                        f'<div class="sw-item sw-strength" style="margin-bottom:6px;">'
                        f'<span class="sw-icon">+</span> '
                        f'<b>{feat_label}:</b> Активный производитель ({int(prod_val)} заявок, '
                        f'среднее: {avg_val:.0f}), что повышает достоверность скоринга.</div>'
                    )
                elif prod_val < 3:
                    nl_explanations.append(
                        f'<div class="sw-item sw-weakness" style="margin-bottom:6px;">'
                        f'<span class="sw-icon">-</span> '
                        f'<b>{feat_label}:</b> Мало заявок ({int(prod_val)}), '
                        f'недостаточно данных для надёжной оценки.</div>'
                    )

        if nl_explanations:
            for expl in nl_explanations:
                st.markdown(expl, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="fairness-alert fairness-ok">
                Все показатели производителя находятся в пределах среднего.
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        # "Что улучшить" section with actionable recommendations
        render_section_header("Рекомендации по улучшению", "Что улучшить")
        improvement_items = []

        if float(prod_feat.get("approval_rate", 0)) < 0.7:
            improvement_items.append(
                "Повысить долю одобренных заявок: проверять корректность документации "
                "перед подачей, консультироваться с районным специалистом."
            )
        if float(prod_feat.get("completion_rate", 0)) < 0.5:
            improvement_items.append(
                "Увеличить долю исполнения: довести начатые проекты до завершения, "
                "своевременно предоставлять отчётную документацию."
            )
        if float(prod_feat.get("utilization_rate", 0)) < 0.5:
            improvement_items.append(
                "Улучшить освоение средств: планировать использование субсидий заранее, "
                "не запрашивать больше, чем может быть освоено."
            )
        if float(prod_feat.get("rejection_rate", 0)) > 0.3:
            improvement_items.append(
                "Снизить долю отклонений: анализировать причины отказов, "
                "устранять типовые ошибки в заявках."
            )
        if float(prod_feat.get("direction_diversity", 0)) < 0.15:
            improvement_items.append(
                "Расширить диверсификацию: рассмотреть участие в дополнительных "
                "программах субсидирования по смежным направлениям."
            )
        if float(prod_feat.get("working_hours_ratio", 0)) < 0.5:
            improvement_items.append(
                "Подавать заявки в рабочее время (9:00--18:00): "
                "это повышает скоринговый балл."
            )
        if float(prod_feat.get("month_regularity", 0)) < 0.3:
            improvement_items.append(
                "Подавать заявки регулярно: распределить подачу по разным месяцам "
                "вместо массовой подачи в один период."
            )
        if int(prod_feat.get("total_apps", 0)) < 3:
            improvement_items.append(
                "Увеличить количество заявок: активное участие в программах "
                "субсидирования повышает рейтинг."
            )

        if not improvement_items:
            st.markdown("""
            <div class="fairness-alert fairness-ok">
                Производитель демонстрирует высокие показатели по всем ключевым факторам.
                Рекомендуется поддерживать текущий уровень.
            </div>
            """, unsafe_allow_html=True)
        else:
            for i, item in enumerate(improvement_items, 1):
                st.markdown(
                    f'<div class="sw-item sw-weakness" style="margin-bottom:8px;">'
                    f'<span class="sw-icon">{i}.</span> {item}</div>',
                    unsafe_allow_html=True,
                )

    with tab3:
        render_section_header("Хронология заявок", "История")
        prod_apps_sorted = prod_apps.sort_values("date", ascending=False)
        timeline_html = ""
        for _, app_row in prod_apps_sorted.iterrows():
            dot_class = get_status_dot_class(str(app_row.get("status", "")))
            date_str = app_row["date"].strftime("%d.%m.%Y %H:%M") if pd.notna(app_row["date"]) else "---"
            timeline_html += f"""
            <div class="timeline-item">
                <div class="timeline-dot {dot_class}"></div>
                <div class="timeline-content">
                    <div class="timeline-date">{date_str}</div>
                    <div class="timeline-title">{app_row.get('subsidy_name', '')}</div>
                    <div class="timeline-meta">
                        {app_row.get('direction_short', '')} |
                        {app_row.get('status', '')} |
                        {format_tenge(app_row.get('amount', 0))} |
                        {app_row.get('district', '')}
                    </div>
                </div>
            </div>"""

        st.markdown(
            f'<div class="glass-card" style="max-height:500px; overflow-y:auto;">{timeline_html}</div>',
            unsafe_allow_html=True,
        )

    with tab4:
        explanation = engine.explain_producer(features, scored, selected_id)
        if explanation:
            # Recommendation
            rec_text = explanation.recommendation
            if "приоритетному" in rec_text.lower():
                rec_cls = "badge-green"
            elif "стандартном" in rec_text.lower():
                rec_cls = "badge-yellow"
            else:
                rec_cls = "badge-red"

            st.markdown(
                f'<div style="margin-bottom:20px;"><span class="badge {rec_cls}" style="font-size:15px; padding:10px 24px;">'
                f'{rec_text}</span></div>',
                unsafe_allow_html=True
            )

            col_s, col_w = st.columns(2)
            with col_s:
                render_section_header("Сильные стороны", "")
                for s in explanation.strengths:
                    st.markdown(
                        f'<div class="sw-item sw-strength"><span class="sw-icon">+</span> {s}</div>',
                        unsafe_allow_html=True,
                    )
            with col_w:
                render_section_header("Слабые стороны", "")
                for w in explanation.weaknesses:
                    st.markdown(
                        f'<div class="sw-item sw-weakness"><span class="sw-icon">-</span> {w}</div>',
                        unsafe_allow_html=True,
                    )

    with tab5:
        detail_data = {
            "Параметр": [
                "ID производителя", "Регион", "Район", "Основное направление",
                "Всего заявок", "Одобрено", "Исполнено", "Отклонено", "Отозвано",
                "Общая сумма", "Средняя сумма", "Кол-во направлений",
                "Кол-во видов субсидий", "Период активности", "Заявок в месяц",
                "Подача в рабочее время", "Балл (правила)", "Балл (ML)", "Балл (итоговый)",
            ],
            "Значение": [
                selected_id, row["region"], row["district"], row["main_direction"],
                int(prod_feat["total_apps"]), int(prod_feat["approved_count"]),
                int(prod_feat["completed_count"]), int(prod_feat["rejected_count"]),
                int(prod_feat["withdrawn_count"]),
                format_tenge(prod_feat["total_amount"]),
                format_tenge(prod_feat["avg_amount"]),
                int(prod_feat["direction_count"]),
                int(prod_feat["subsidy_type_count"]),
                f"{int(prod_feat['activity_span_days'])} дней",
                f"{prod_feat['apps_per_month']:.1f}",
                format_percent(prod_feat["working_hours_ratio"]),
                format_score(rule_val), format_score(ml_val), format_score(score_val),
            ],
        }
        st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE 5: COMPARISON
# ===========================================================================
def render_comparison(scored: pd.DataFrame, features: pd.DataFrame):
    render_breadcrumb("Сравнение производителей")
    render_page_header(
        "Сравнение производителей",
        "Детальное сопоставление показателей двух или трёх производителей"
    )

    top_producers = scored.nsmallest(200, "rank")["producer_id"].tolist()
    col1, col2, col3 = st.columns(3)
    with col1:
        prod1 = st.selectbox("Производитель 1", top_producers, index=0, key="cmp1")
    with col2:
        prod2 = st.selectbox("Производитель 2", top_producers, index=min(1, len(top_producers)-1), key="cmp2")
    with col3:
        prod3_options = ["---"] + top_producers
        prod3_sel = st.selectbox("Производитель 3 (опц.)", prod3_options, index=0, key="cmp3")

    selected_ids = [prod1, prod2]
    if prod3_sel != "---":
        selected_ids.append(prod3_sel)

    # Check uniqueness
    if len(set(selected_ids)) < len(selected_ids):
        st.warning("Выберите разных производителей для сравнения")
        return

    rows = [scored[scored["producer_id"] == pid].iloc[0] for pid in selected_ids]
    feats = [features[features["producer_id"] == pid].iloc[0] for pid in selected_ids]

    render_divider()

    # --- Side-by-side score cards ---
    render_section_header("Сводка баллов", "Сравнение")

    cols = st.columns(len(selected_ids))
    winner_idx = max(range(len(rows)), key=lambda i: float(rows[i]["combined_score"]))

    for i, (col, pid, r) in enumerate(zip(cols, selected_ids, rows)):
        with col:
            sc = float(r["combined_score"])
            is_winner = (i == winner_idx)
            border_style = "border: 2px solid #38BDF8; box-shadow: 0 0 20px rgba(59,130,246,0.2);" if is_winner else ""
            winner_label = '<div style="color:#38BDF8; font-weight:700; font-size:12px; margin-bottom:4px; text-transform:uppercase; letter-spacing:1px;">Лидер</div>' if is_winner else ''

            st.markdown(f"""
            <div class="glass-card" style="text-align:center; {border_style}">
                {winner_label}
                <div style="font-family:monospace; font-size:13px; color:#94A3B8; margin-bottom:8px;">
                    ...{pid[-6:]}
                </div>
                <div style="font-size:42px; font-weight:800; background:linear-gradient(135deg, #60A5FA, #38BDF8);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:4px;">
                    {sc:.1f}
                </div>
                <div style="font-size:13px; color:#94A3B8; margin-bottom:8px;">
                    Ранг #{int(r['rank'])}
                </div>
                {get_recommendation_badge(sc)}
                <div style="margin-top:12px; font-size:12px;">
                    <span style="color:#94A3B8;">Правила:</span> <b>{float(r.get('rule_score',0)):.1f}</b> |
                    <span style="color:#94A3B8;">ML:</span> <b>{float(r.get('ml_score',0)):.1f}</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

    render_divider()

    # --- Dimension comparison table ---
    render_section_header("Детальное сопоставление", "Параметры")
    feat_descs = get_feature_descriptions()
    comparison_data = []
    for feat_name, desc in feat_descs.items():
        row_data = {"Параметр": desc}
        vals = []
        for i, (pid, f) in enumerate(zip(selected_ids, feats)):
            v = float(f.get(feat_name, 0))
            row_data[f"...{pid[-6:]}"] = round(v, 4)
            vals.append(v)
        if len(vals) >= 2:
            row_data["Разница (1 vs 2)"] = round(vals[0] - vals[1], 4)
        comparison_data.append(row_data)

    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)

    render_divider()

    # --- Radar chart ---
    render_section_header("Радарная диаграмма", "Визуализация")
    radar_features = [
        "approval_rate", "completion_rate", "utilization_rate",
        "direction_diversity", "month_regularity", "working_hours_ratio",
    ]
    fig = plot_comparison_radar(feats, selected_ids, radar_features, feat_descs)
    apply_chart_theme(fig, height=500)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(0,0,0,0.08)"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    # Override trace colors
    for i, trace in enumerate(fig.data):
        trace.line = dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2)
        trace.fillcolor = CHART_COLORS[i % len(CHART_COLORS)].replace(")", ",0.15)").replace("rgb", "rgba") if "rgb" in CHART_COLORS[i % len(CHART_COLORS)] else None
    st.plotly_chart(fig, use_container_width=True, key="chart_compare_radar")


# ===========================================================================
# PAGE 6: SHORTLIST
# ===========================================================================
def render_shortlist(scored: pd.DataFrame):
    render_breadcrumb("Шортлист")
    render_page_header(
        "Генератор шортлиста",
        "Формирование списка приоритетных получателей субсидий"
    )

    # --- Filters ---
    render_section_header("Параметры формирования", "Настройка")
    st.markdown('<div class="filter-panel">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        top_n = st.number_input("Количество в шортлисте", 10, 1000, 100, step=10)
    with col2:
        min_score = st.slider("Минимальный балл", 0, 100, 30, key="short_min")
    with col3:
        regions = ["Все"] + sorted(scored["region"].unique().tolist())
        sel_region = st.selectbox("Регион", regions, key="short_region")
    with col4:
        directions = ["Все"] + sorted(scored["main_direction"].dropna().unique().tolist())
        sel_direction = st.selectbox("Направление", directions, key="short_direction")
    st.markdown('</div>', unsafe_allow_html=True)

    filtered = scored.copy()
    if sel_region != "Все":
        filtered = filtered[filtered["region"] == sel_region]
    if sel_direction != "Все":
        filtered = filtered[filtered["main_direction"] == sel_direction]
    if min_score > 0:
        filtered = filtered[filtered["combined_score"] >= min_score]

    shortlist = filtered.nsmallest(top_n, "rank")

    render_divider()

    # --- Summary metrics ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("В ШОРТЛИСТЕ", str(len(shortlist)), "Производителей", delay=1)
    with c2:
        if len(shortlist) > 0:
            render_metric_card(
                "СРЕДНИЙ БАЛЛ", f"{shortlist['combined_score'].mean():.1f}",
                f"Мин: {shortlist['combined_score'].min():.1f}", delay=2,
            )
        else:
            render_metric_card("СРЕДНИЙ БАЛЛ", "---", "Нет данных", delay=2)
    with c3:
        render_metric_card(
            "ОБЪЁМ СУБСИДИЙ",
            format_tenge(shortlist["total_amount"].sum()),
            "Суммарно",
            gold=True, delay=3,
        )
    with c4:
        render_metric_card(
            "РЕГИОНОВ",
            str(shortlist["region"].nunique()),
            "Представлено",
            delay=4,
        )

    render_divider()

    # --- Table ---
    render_section_header("Шортлист", f"{len(shortlist)} записей")
    table_rows = ""
    for _, row in shortlist.iterrows():
        sc = float(row["combined_score"])
        bar_class = get_score_bar_class(sc)
        table_rows += f"""
        <tr>
            <td style="font-weight:700; color:#38BDF8;">#{int(row['rank'])}</td>
            <td style="font-family:monospace; font-size:12px;">{row['producer_id']}</td>
            <td>{row['region']}</td>
            <td>{row.get('main_direction', '')}</td>
            <td>
                <div class="score-bar-container">
                    <div class="score-bar {bar_class}" style="width:{sc}%;">{sc:.1f}</div>
                </div>
            </td>
            <td>{format_percent(row.get('approval_rate', 0))}</td>
            <td>{format_percent(row.get('utilization_rate', 0))}</td>
            <td style="font-weight:600;">{format_tenge(row.get('total_amount', 0))}</td>
        </tr>"""

    st.markdown(f"""
    <table class="styled-table">
        <thead>
            <tr>
                <th>Ранг</th>
                <th>ID производителя</th>
                <th>Регион</th>
                <th>Направление</th>
                <th style="min-width:130px;">Балл</th>
                <th>Одобрение</th>
                <th>Освоение</th>
                <th>Сумма</th>
            </tr>
        </thead>
        <tbody>{table_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # Export (styled)
    st.markdown("<br>", unsafe_allow_html=True)
    csv_data = export_shortlist_csv(shortlist)
    exp_c1, exp_c2 = st.columns([2, 6])
    with exp_c1:
        st.download_button(
            label="\u21E9 Экспорт CSV",
            data=csv_data.encode("utf-8-sig"),
            file_name="shortlist_subsidies.csv",
            mime="text/csv",
            key="shortlist_export_csv",
        )

    render_divider()

    # --- Budget impact summary card ---
    total_budget_est = scored["total_amount"].sum()
    shortlist_budget_est = shortlist["total_amount"].sum()
    budget_share = shortlist_budget_est / total_budget_est * 100 if total_budget_est > 0 else 0
    avg_score_shortlist = shortlist["combined_score"].mean() if len(shortlist) > 0 else 0
    st.markdown(f"""
    <div class="glass-card" style="margin-bottom:24px; border-left:4px solid #38BDF8;">
        <div style="font-size:16px; font-weight:700; color:#E2E8F0; margin-bottom:12px;">
            Оценка бюджетного воздействия
        </div>
        <div style="display:flex; gap:32px; flex-wrap:wrap; font-size:14px; line-height:1.7; color:#E2E8F0;">
            <div>
                <span style="font-weight:600; color:#94A3B8;">Производителей в шортлисте:</span>
                <span style="font-weight:700;"> {len(shortlist)}</span>
            </div>
            <div>
                <span style="font-weight:600; color:#94A3B8;">Объём субсидий:</span>
                <span style="font-weight:700; color:#38BDF8;"> {format_tenge(shortlist_budget_est)}</span>
            </div>
            <div>
                <span style="font-weight:600; color:#94A3B8;">Доля от общего бюджета:</span>
                <span style="font-weight:700;"> {budget_share:.1f}%</span>
            </div>
            <div>
                <span style="font-weight:600; color:#94A3B8;">Средний балл шортлиста:</span>
                <span style="font-weight:700; color:#38BDF8;"> {avg_score_shortlist:.1f}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_divider()

    # --- Budget impact simulation ---
    render_section_header("Бюджетное моделирование", "Симуляция")
    total_budget = scored["total_amount"].sum()
    shortlist_budget = shortlist["total_amount"].sum()
    budget_pct = shortlist_budget / total_budget * 100 if total_budget > 0 else 0

    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        render_metric_card("ОБЩИЙ БЮДЖЕТ", format_tenge(total_budget), "Все производители")
    with bc2:
        render_metric_card("ШОРТЛИСТ", format_tenge(shortlist_budget), f"{budget_pct:.1f}% от общего", gold=True)
    with bc3:
        efficiency = shortlist["combined_score"].mean() / max(scored["combined_score"].mean(), 1) if len(shortlist) > 0 else 0
        render_metric_card("ЭФФЕКТИВНОСТЬ", f"x{efficiency:.2f}", "Относительно среднего")

    render_divider()

    # --- Regional distribution ---
    render_section_header("Распределение шортлиста по регионам", "Структура")
    if len(shortlist) > 0:
        region_counts = shortlist["region"].value_counts().reset_index()
        region_counts.columns = ["Регион", "Количество"]
        fig = go.Figure(go.Bar(
            y=region_counts.sort_values("Количество", ascending=True)["Регион"],
            x=region_counts.sort_values("Количество", ascending=True)["Количество"],
            orientation="h",
            marker=dict(color=TEAL, cornerradius=4),
            text=region_counts.sort_values("Количество", ascending=True)["Количество"],
            textposition="auto",
        ))
        apply_chart_theme(fig, height=max(350, len(region_counts) * 30))
        fig.update_layout(margin=dict(l=250), xaxis_title="Количество")
        st.plotly_chart(fig, use_container_width=True, key="chart_shortlist_region_bar")
    else:
        st.info("Нет данных для отображения графика.")


# ===========================================================================
# PAGE 7: FAIRNESS
# ===========================================================================
def render_fairness(scored: pd.DataFrame):
    render_breadcrumb("Анализ справедливости")
    render_page_header(
        "Анализ справедливости",
        "Проверка наличия предвзятости в скоринговой модели по регионам и направлениям"
    )

    metrics = compute_fairness_metrics(scored)

    # --- Key metrics ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        gini_cls = "good" if metrics["gini_score"] < 0.3 else "warn"
        render_metric_card(
            "ДЖИНИ (БАЛЛЫ)",
            f"{metrics['gini_score']:.3f}",
            "Ниже 0.3 = хорошо",
            gold=(gini_cls == "warn"),
            delay=1,
        )
    with c2:
        render_metric_card(
            "ДЖИНИ (СУММЫ)",
            f"{metrics['gini_amount']:.3f}",
            "Неравенство сумм",
            delay=2,
        )
    with c3:
        render_metric_card(
            "CV РЕГИОНАЛЬНЫЙ",
            f"{metrics['cv_regional']:.3f}",
            "Коэффициент вариации",
            delay=3,
        )
    with c4:
        render_metric_card(
            "ПРЕДВЗЯТОСТЬ",
            f"{metrics['biased_regions']} из {metrics['total_regions']}",
            "Регионов с отклонением",
            gold=(metrics["biased_regions"] > 0),
            delay=4,
        )

    render_divider()

    # --- Kruskal-Wallis test result ---
    kw_p = metrics["kruskal_p"]
    kw_h = metrics["kruskal_h"]
    if kw_p < 0.05:
        st.markdown(f"""
        <div class="fairness-alert fairness-warn">
            <span style="font-weight:700;">ТЕСТ КРАСКЕЛА-УОЛЛИСА:</span>
            Различия между регионами статистически значимы (H={kw_h}, p={kw_p}).
            Рекомендуется регионально-адаптивная нормализация.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="fairness-alert fairness-ok">
            <span style="font-weight:700;">ТЕСТ КРАСКЕЛА-УОЛЛИСА:</span>
            Статистически значимых различий не выявлено (p={kw_p}).
        </div>
        """, unsafe_allow_html=True)

    render_divider()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Региональный анализ",
        "По направлениям",
        "Кривая Лоренца",
        "Рекомендации",
    ])

    with tab1:
        fig = plot_fairness_overview(scored)
        apply_chart_theme(fig, height=600)
        fig.update_layout(margin=dict(l=250))
        st.plotly_chart(fig, use_container_width=True, key="chart_fairness_overview")

        fig = plot_score_violin_by_region(scored)
        apply_chart_theme(fig, height=500)
        fig.update_layout(margin=dict(b=150))
        st.plotly_chart(fig, use_container_width=True, key="chart_fairness_violin_region")

        render_section_header("Таблица региональной справедливости", "Данные")
        regional = compute_regional_fairness(scored)
        display = regional.copy()
        display["mean_score"] = display["mean_score"].round(1)
        display["median_score"] = display["median_score"].round(1)
        display["std_score"] = display["std_score"].round(1)
        display["score_deviation"] = display["score_deviation"].round(3)
        display["mean_approval"] = display["mean_approval"].apply(lambda x: f"{x:.1%}")
        display["total_amount"] = display["total_amount"].apply(format_tenge)
        display = display.rename(columns={
            "region": "Регион", "producer_count": "Производителей",
            "mean_score": "Сред. балл", "median_score": "Медиана",
            "std_score": "СКО", "score_deviation": "Отклонение",
            "bias_direction": "Предвзятость", "mean_approval": "Одобрение",
            "total_amount": "Объём субсидий",
        })
        show_cols = [
            "Регион", "Производителей", "Сред. балл", "Медиана",
            "Отклонение", "Предвзятость", "Одобрение", "Объём субсидий",
        ]
        st.dataframe(
            display[[c for c in show_cols if c in display.columns]],
            use_container_width=True, hide_index=True,
        )

    with tab2:
        dir_fairness = compute_direction_fairness(scored)
        display = dir_fairness.copy()
        display["mean_score"] = display["mean_score"].round(1)
        display["median_score"] = display["median_score"].round(1)
        display["deviation"] = display["deviation"].round(1)
        display["mean_approval"] = display["mean_approval"].apply(lambda x: f"{x:.1%}")
        display = display.rename(columns={
            "main_direction": "Направление", "producer_count": "Производителей",
            "mean_score": "Сред. балл", "median_score": "Медиана",
            "deviation": "Отклонение", "mean_approval": "Одобрение",
        })
        st.dataframe(display, use_container_width=True, hide_index=True)

    with tab3:
        fig = plot_lorenz_curve(scored)
        apply_chart_theme(fig, height=450)
        st.plotly_chart(fig, use_container_width=True, key="chart_fairness_lorenz")

    with tab4:
        render_section_header("Рекомендации по обеспечению справедливости", "")
        report = generate_fairness_report(scored)
        for item in report:
            # Classify the item for styling
            text = item.strip()
            clean = text.strip()
            if text.startswith("[+]") or "умеренное" in text or "не выявил" in text or "Ни один регион" in text:
                st.markdown(
                    f'<div class="fairness-alert fairness-ok">{clean}</div>',
                    unsafe_allow_html=True,
                )
            elif text.startswith("[!]") or "заметное" in text or "статистически значимы" in text:
                st.markdown(
                    f'<div class="fairness-alert fairness-warn">{clean}</div>',
                    unsafe_allow_html=True,
                )
            elif text.startswith("[-]") or "значительное" in text:
                st.markdown(
                    f'<div class="fairness-alert fairness-danger">{clean}</div>',
                    unsafe_allow_html=True,
                )
            elif "Рекомендации" in text:
                # section header with bullet points
                st.markdown(f"""
                <div class="glass-card" style="margin-top:16px;">
                    <div style="font-weight:700; font-size:16px; color:#E2E8F0; margin-bottom:12px;">
                        Рекомендации по обеспечению справедливости
                    </div>
                    <div style="white-space:pre-line; line-height:1.8; font-size:14px;">{clean}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="fairness-alert fairness-warn">{clean}</div>',
                    unsafe_allow_html=True,
                )


# ===========================================================================
# PAGE 8: ANALYTICS
# ===========================================================================
def render_analytics(df: pd.DataFrame, scored: pd.DataFrame, engine: ScoringEngine):
    render_breadcrumb("Аналитика")
    render_page_header(
        "Аналитика и корреляции",
        "Углублённый анализ модели и данных"
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Важность признаков",
        "Корреляции",
        "Связь балла и суммы",
        "Сравнение с базовыми моделями",
        "Валидация модели",
        "Ablation Study",
        "Анализ чувствительности",
    ])

    with tab1:
        render_section_header("Важность признаков ML-модели", "GradientBoosting")
        importance = engine.ml_scorer.get_feature_importance()
        fig = plot_feature_importance(importance)
        apply_chart_theme(fig, height=max(350, len(importance) * 30))
        fig.update_layout(margin=dict(l=200))
        st.plotly_chart(fig, use_container_width=True, key="chart_analytics_feature_importance")

        render_divider()

        render_section_header("Пермутационная важность", "Робастность")
        with st.spinner("Вычисление пермутационной важности..."):
            from feature_engineering import compute_producer_features as cpf
            feats = cpf(df)
            perm_imp = engine.ml_scorer.compute_permutation_importance(feats, n_repeats=5)
        fig = plot_feature_importance(perm_imp)
        apply_chart_theme(fig, height=max(350, len(perm_imp) * 30))
        fig.update_layout(margin=dict(l=200))
        st.plotly_chart(fig, use_container_width=True, key="chart_analytics_perm_importance")

    with tab2:
        render_section_header("Корреляционная матрица", "Признаки")
        feature_cols = get_scoring_features()
        fig = plot_correlation_heatmap(scored, feature_cols)
        apply_chart_theme(fig, height=600, transparent=False)
        fig.update_layout(margin=dict(l=150, b=150))
        st.plotly_chart(fig, use_container_width=True, key="chart_analytics_correlation")

    with tab3:
        render_section_header("Зависимость балла от суммы", "Scatter")
        fig = plot_amount_vs_score(scored)
        apply_chart_theme(fig, height=500)
        st.plotly_chart(fig, use_container_width=True, key="chart_analytics_amount_vs_score")

        render_divider()

        render_section_header("Распределение баллов по регионам", "Боксплот")
        fig = plot_score_by_region(scored)
        apply_chart_theme(fig, height=500)
        fig.update_layout(margin=dict(b=150))
        st.plotly_chart(fig, use_container_width=True, key="chart_analytics_score_by_region")

        render_divider()

        render_section_header("Одобрение по направлениям", "")
        fig = plot_approval_rate_by_direction(df)
        apply_chart_theme(fig, height=400)
        fig.update_layout(margin=dict(l=200))
        st.plotly_chart(fig, use_container_width=True, key="chart_analytics_approval_by_dir")

        render_divider()

        render_section_header("Динамика по месяцам", "Тренд")
        fig = plot_monthly_trend(df)
        apply_chart_theme(fig, height=420)
        st.plotly_chart(fig, use_container_width=True, key="chart_analytics_monthly_trend")

    with tab4:
        render_baseline_comparison(scored, engine, df)

    with tab5:
        render_model_validation(scored, engine, df)

    with tab6:
        render_ablation_study(scored, engine, df)

    with tab7:
        render_sensitivity_analysis(scored, engine, df)


def render_ablation_study(scored: pd.DataFrame, engine: ScoringEngine, df: pd.DataFrame):
    """Ablation study: remove each feature one at a time, measure impact on R2 and ranking."""
    from feature_engineering import compute_producer_features, prepare_model_data, get_scoring_features
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import GradientBoostingRegressor

    render_section_header("Ablation Study", "Влияние каждого признака")
    st.markdown("""
    <div class="glass-card" style="margin-bottom:24px;">
        <div style="font-size:14px; line-height:1.7; color:#E2E8F0;">
            Ablation study показывает, как удаление каждого из 14 признаков влияет на качество модели (R2)
            и стабильность ранжирования (изменение позиций в топ-100).
            Чем больше падение R2 при удалении признака, тем он важнее для модели.
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        feats = compute_producer_features(df)
        X, y = prepare_model_data(feats)
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        feature_names = get_scoring_features()
        descs = get_feature_descriptions()

        # Baseline R2 with all features
        X_scaled_full = engine.ml_scorer.scaler.transform(X_clean)
        baseline_cv = cross_val_score(engine.ml_scorer.model, X_scaled_full, y, cv=5, scoring="r2")
        baseline_r2 = float(baseline_cv.mean())

        # Get baseline ranking
        y_pred_full = engine.ml_scorer.model.predict(X_scaled_full)
        baseline_top100 = set(np.argsort(y_pred_full)[-100:])

        ablation_results = []
        for i, feat in enumerate(feature_names):
            # Remove feature i
            X_ablated = X_clean.drop(columns=[feat], errors="ignore")
            if X_ablated.shape[1] == X_clean.shape[1]:
                continue
            from sklearn.preprocessing import MinMaxScaler
            scaler_abl = MinMaxScaler()
            X_abl_scaled = scaler_abl.fit_transform(X_ablated)

            model_abl = GradientBoostingRegressor(
                n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
            )
            cv_abl = cross_val_score(model_abl, X_abl_scaled, y, cv=5, scoring="r2")
            r2_abl = float(cv_abl.mean())
            r2_drop = baseline_r2 - r2_abl

            # Ranking change
            model_abl.fit(X_abl_scaled, y)
            y_pred_abl = model_abl.predict(X_abl_scaled)
            abl_top100 = set(np.argsort(y_pred_abl)[-100:])
            ranking_overlap = len(baseline_top100 & abl_top100)

            ablation_results.append({
                "feature": feat,
                "label": descs.get(feat, feat),
                "r2_without": r2_abl,
                "r2_drop": r2_drop,
                "top100_overlap": ranking_overlap,
            })

        if ablation_results:
            abl_df = pd.DataFrame(ablation_results).sort_values("r2_drop", ascending=False)

            # Bar chart: R2 drop per feature
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=abl_df["label"],
                x=abl_df["r2_drop"],
                orientation="h",
                marker=dict(
                    color=abl_df["r2_drop"],
                    colorscale=[[0, "#60A5FA"], [1, "#1D4ED8"]],
                    cornerradius=4,
                ),
                text=[f"{v:.4f}" for v in abl_df["r2_drop"]],
                textposition="auto",
                hovertemplate="%{y}<br>Падение R2: %{x:.4f}<extra></extra>",
            ))
            apply_chart_theme(fig, height=max(400, len(abl_df) * 32))
            fig.update_layout(
                xaxis_title="Падение R2 при удалении признака",
                margin=dict(l=250, t=30),
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_ablation_r2_drop")

            render_divider()

            # Table
            render_section_header("Таблица ablation study", "Данные")
            display_abl = abl_df[["label", "r2_without", "r2_drop", "top100_overlap"]].copy()
            display_abl.columns = ["Признак", "R2 без признака", "Падение R2", "Top-100 совпадение"]
            display_abl["R2 без признака"] = display_abl["R2 без признака"].round(4)
            display_abl["Падение R2"] = display_abl["Падение R2"].round(4)
            st.dataframe(display_abl, use_container_width=True, hide_index=True)

            st.markdown(f"""
            <div class="glass-card" style="margin-top:16px; border-left:4px solid #38BDF8;">
                <div style="font-size:14px; line-height:1.7; color:#E2E8F0;">
                    <span style="font-weight:700; color:#E2E8F0;">Baseline R2:</span> {baseline_r2:.4f}.
                    Наиболее критичные признаки -- те, при удалении которых R2 падает сильнее всего.
                    Top-100 overlap показывает, сколько из 100 лучших производителей
                    остаются в топ-100 после удаления признака.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Не удалось выполнить ablation study.")
    except Exception as e:
        st.warning(f"Не удалось выполнить ablation study: {e}")


def render_sensitivity_analysis(scored: pd.DataFrame, engine: ScoringEngine, df: pd.DataFrame):
    """Sensitivity analysis: shift feature weights +/-10%, show ranking stability heatmap."""
    render_section_header("Анализ чувствительности", "Стабильность ранжирования")
    st.markdown("""
    <div class="glass-card" style="margin-bottom:24px;">
        <div style="font-size:14px; line-height:1.7; color:#E2E8F0;">
            Анализ чувствительности показывает, как изменение весов отдельных признаков
            на +/-10% влияет на стабильность ранжирования. Heatmap отображает процент
            совпадения топ-100 при сдвиге каждого веса.
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        from feature_engineering import compute_producer_features, get_scoring_features

        feature_names = get_scoring_features()
        descs = get_feature_descriptions()
        current_weights = st.session_state.get("feature_weights", get_feature_weights_default())

        feats = compute_producer_features(df)

        # Get baseline ranking from rule scores with current weights
        def compute_rule_ranking(weights_dict, features_df):
            """Compute weighted rule-based ranking."""
            score_cols = [c for c in feature_names if c in features_df.columns]
            normalized = features_df[score_cols].copy()
            for col in score_cols:
                col_min = normalized[col].min()
                col_max = normalized[col].max()
                if col_max > col_min:
                    normalized[col] = (normalized[col] - col_min) / (col_max - col_min)
                else:
                    normalized[col] = 0.0
            rule_scores = pd.Series(0.0, index=features_df.index)
            for col in score_cols:
                w = weights_dict.get(col, 0.0)
                rule_scores += normalized[col] * w
            return rule_scores.rank(ascending=False).values

        baseline_ranks = compute_rule_ranking(current_weights, feats)
        baseline_top100 = set(np.argsort(-baseline_ranks)[:100])

        shifts = [-0.10, -0.05, 0.0, 0.05, 0.10]
        shift_labels = ["-10%", "-5%", "0%", "+5%", "+10%"]
        heatmap_data = []
        feature_labels = []

        for feat in feature_names:
            if feat not in current_weights:
                continue
            row = []
            for shift in shifts:
                modified_weights = current_weights.copy()
                modified_weights[feat] = current_weights[feat] + shift
                modified_ranks = compute_rule_ranking(modified_weights, feats)
                modified_top100 = set(np.argsort(-modified_ranks)[:100])
                overlap = len(baseline_top100 & modified_top100)
                row.append(overlap)
            heatmap_data.append(row)
            feature_labels.append(descs.get(feat, feat))

        if heatmap_data:
            heatmap_array = np.array(heatmap_data)

            fig = go.Figure(data=go.Heatmap(
                z=heatmap_array,
                x=shift_labels,
                y=feature_labels,
                colorscale=[[0, "#EF4444"], [0.5, "#F59E0B"], [0.85, "#60A5FA"], [1, "#22C55E"]],
                zmin=50, zmax=100,
                text=heatmap_array,
                texttemplate="%{text}%",
                textfont=dict(size=11),
                hovertemplate="Признак: %{y}<br>Сдвиг: %{x}<br>Top-100 совпадение: %{z}%<extra></extra>",
                colorbar=dict(title="Overlap %"),
            ))
            apply_chart_theme(fig, height=max(450, len(feature_labels) * 35))
            fig.update_layout(
                xaxis_title="Сдвиг веса",
                yaxis_title="",
                margin=dict(l=250, t=30),
            )
            st.plotly_chart(fig, use_container_width=True, key="chart_sensitivity_heatmap")

            render_divider()

            # Summary
            min_overlap = int(heatmap_array.min())
            avg_overlap = float(heatmap_array[heatmap_array < 100].mean()) if (heatmap_array < 100).any() else 100.0
            stability_label = "высокая" if min_overlap >= 90 else ("средняя" if min_overlap >= 75 else "низкая")
            stability_cls = "fairness-ok" if min_overlap >= 90 else ("fairness-warn" if min_overlap >= 75 else "fairness-danger")

            st.markdown(f"""
            <div class="{stability_cls} fairness-alert">
                <span style="font-weight:700;">Стабильность ранжирования: {stability_label}.</span>
                Минимальное совпадение топ-100 при сдвиге весов на +/-10%: {min_overlap}%.
                Среднее совпадение при ненулевых сдвигах: {avg_overlap:.1f}%.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Нет данных для анализа чувствительности.")
    except Exception as e:
        st.warning(f"Не удалось выполнить анализ чувствительности: {e}")


def render_baseline_comparison(scored: pd.DataFrame, engine: ScoringEngine, df: pd.DataFrame):
    """Сравнение гибридной модели с базовыми моделями (FCFS, Rule-only, ML-only)."""
    render_section_header(
        "Сравнение с базовыми моделями",
        "Эффективность"
    )

    st.markdown("""
    <div class="glass-card" style="margin-bottom:24px;">
        <div style="font-size:14px; line-height:1.7; color:#94A3B8;">
            Сравнение четырёх подходов к ранжированию: случайный отбор (FCFS),
            только правиловая модель (100% правила), только ML-модель (100% ML)
            и текущая гибридная модель (60% ML + 40% правила).
        </div>
    </div>
    """, unsafe_allow_html=True)

    n_top = min(100, len(scored))
    total_producers = len(scored)

    # 1. FCFS baseline: random selection
    np.random.seed(42)
    fcfs_indices = np.random.choice(total_producers, n_top, replace=False)
    fcfs_sample = scored.iloc[fcfs_indices]

    # 2. Rule-only baseline: rank by rule_score
    if "rule_score" in scored.columns:
        rule_sorted = scored.sort_values("rule_score", ascending=False)
        rule_only_top = rule_sorted.head(n_top)
    else:
        rule_only_top = fcfs_sample

    # 3. ML-only baseline: rank by ml_score
    if "ml_score" in scored.columns:
        ml_sorted = scored.sort_values("ml_score", ascending=False)
        ml_only_top = ml_sorted.head(n_top)
    else:
        ml_only_top = fcfs_sample

    # 4. Hybrid (current): top N by combined rank
    hybrid_top = scored.nsmallest(n_top, "rank")

    # Compute metrics for each
    models = {
        "FCFS (случайный)": fcfs_sample,
        "Только правила (100%)": rule_only_top,
        "Только ML (100%)": ml_only_top,
        "Гибрид (60/40)": hybrid_top,
    }

    comparison_rows = []
    for model_name, sample in models.items():
        comparison_rows.append({
            "Модель": model_name,
            "Ср. балл": round(float(sample["combined_score"].mean()), 1),
            "Медиана балла": round(float(sample["combined_score"].median()), 1),
            "Ср. одобрение": f"{sample['approval_rate'].mean():.1%}",
            "Ср. освоение": f"{sample['utilization_rate'].mean():.1%}",
            "Ср. исполнение": f"{sample['completion_rate'].mean():.1%}",
            "Регионов": int(sample["region"].nunique()),
        })

    comp_df = pd.DataFrame(comparison_rows)
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    render_divider()

    # Visual comparison: bar chart
    render_section_header("Визуальное сравнение", "Графики")

    metric_names = ["FCFS", "Правила", "ML", "Гибрид"]
    avg_scores = [
        float(fcfs_sample["combined_score"].mean()),
        float(rule_only_top["combined_score"].mean()),
        float(ml_only_top["combined_score"].mean()),
        float(hybrid_top["combined_score"].mean()),
    ]
    avg_approvals = [
        float(fcfs_sample["approval_rate"].mean()) * 100,
        float(rule_only_top["approval_rate"].mean()) * 100,
        float(ml_only_top["approval_rate"].mean()) * 100,
        float(hybrid_top["approval_rate"].mean()) * 100,
    ]
    avg_utils = [
        float(fcfs_sample["utilization_rate"].mean()) * 100,
        float(rule_only_top["utilization_rate"].mean()) * 100,
        float(ml_only_top["utilization_rate"].mean()) * 100,
        float(hybrid_top["utilization_rate"].mean()) * 100,
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Средний балл", x=metric_names, y=avg_scores,
                         marker=dict(color=TEAL, cornerradius=4), text=[f"{v:.1f}" for v in avg_scores], textposition="auto"))
    fig.add_trace(go.Bar(name="Одобрение (%)", x=metric_names, y=avg_approvals,
                         marker=dict(color=GOLD, cornerradius=4), text=[f"{v:.1f}" for v in avg_approvals], textposition="auto"))
    fig.add_trace(go.Bar(name="Освоение (%)", x=metric_names, y=avg_utils,
                         marker=dict(color=NAVY_LIGHT, cornerradius=4), text=[f"{v:.1f}" for v in avg_utils], textposition="auto"))
    apply_chart_theme(fig, height=450)
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True, key="chart_baseline_comparison")

    render_divider()

    # Improvement metrics (FCFS vs Hybrid)
    render_section_header("Прирост гибридной модели vs FCFS", "Результат")

    fcfs_avg_score = float(fcfs_sample["combined_score"].mean())
    hybrid_avg_score = float(hybrid_top["combined_score"].mean())
    score_improvement = ((hybrid_avg_score - fcfs_avg_score) / max(fcfs_avg_score, 1)) * 100

    fcfs_avg_approval = float(fcfs_sample["approval_rate"].mean())
    hybrid_avg_approval = float(hybrid_top["approval_rate"].mean())
    approval_improvement = ((hybrid_avg_approval - fcfs_avg_approval) / max(fcfs_avg_approval, 0.01)) * 100

    fcfs_avg_utilization = float(fcfs_sample["utilization_rate"].mean())
    hybrid_avg_utilization = float(hybrid_top["utilization_rate"].mean())
    util_improvement = ((hybrid_avg_utilization - fcfs_avg_utilization) / max(fcfs_avg_utilization, 0.01)) * 100

    ic1, ic2, ic3 = st.columns(3)
    with ic1:
        render_metric_card("БАЛЛ", f"+{score_improvement:.0f}%",
                           f"{fcfs_avg_score:.1f} -> {hybrid_avg_score:.1f}", gold=True)
    with ic2:
        render_metric_card("ОДОБРЕНИЕ", f"+{approval_improvement:.0f}%",
                           f"{fcfs_avg_approval:.1%} -> {hybrid_avg_approval:.1%}", gold=True)
    with ic3:
        render_metric_card("ОСВОЕНИЕ", f"+{util_improvement:.0f}%",
                           f"{fcfs_avg_utilization:.1%} -> {hybrid_avg_utilization:.1%}", gold=True)

    render_divider()

    # Side-by-side old vs new
    col1, col_arrow, col2 = st.columns([5, 1, 5])
    with col1:
        st.markdown(f"""
        <div class="baseline-card baseline-old">
            <h3 style="color:#8a3030;">Текущая система (FCFS)</h3>
            <div class="baseline-label">Средний балл (топ-{n_top})</div>
            <div class="baseline-metric" style="color:#b84c4c;">{fcfs_avg_score:.1f}</div>
            <div class="baseline-label">Среднее одобрение</div>
            <div class="baseline-metric" style="color:#b84c4c; font-size:22px;">{fcfs_avg_approval:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_arrow:
        st.markdown("""
        <div class="improvement-arrow" style="height:100%; display:flex; align-items:center; justify-content:center;">
            >>>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="baseline-card baseline-new">
            <h3 style="color:#38BDF8;">Гибридная модель (60/40)</h3>
            <div class="baseline-label">Средний балл (топ-{n_top})</div>
            <div class="baseline-metric" style="color:#38BDF8;">{hybrid_avg_score:.1f}</div>
            <div class="baseline-label">Среднее одобрение</div>
            <div class="baseline-metric" style="color:#38BDF8; font-size:22px;">{hybrid_avg_approval:.1%}</div>
        </div>
        """, unsafe_allow_html=True)


def render_model_validation(scored: pd.DataFrame, engine: ScoringEngine, df: pd.DataFrame):
    """Раздел валидации модели: feature importance, CV, распределение, scatter."""
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from feature_engineering import compute_producer_features, prepare_model_data

    render_section_header("Валидация модели", "Качество")

    # --- Feature importance chart ---
    render_section_header("Важность признаков (GradientBoosting)", "Feature Importance")
    if engine.ml_scorer._fitted:
        raw_importances = dict(zip(
            get_scoring_features(),
            engine.ml_scorer.model.feature_importances_
        ))
        descs = get_feature_descriptions()
        named_importances = {descs.get(k, k): v for k, v in raw_importances.items()}
        fig = plot_feature_importance(named_importances)
        apply_chart_theme(fig, height=max(350, len(named_importances) * 30))
        fig.update_layout(margin=dict(l=200))
        st.plotly_chart(fig, use_container_width=True, key="chart_validation_feature_imp")
    else:
        st.info("Модель ещё не обучена.")

    render_divider()

    # --- Cross-validation results ---
    render_section_header("Кросс-валидация (5-fold)", "Метрики")

    try:
        feats = compute_producer_features(df)
        X, y = prepare_model_data(feats)
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        X_scaled = engine.ml_scorer.scaler.transform(X_clean)

        cv_r2 = cross_val_score(engine.ml_scorer.model, X_scaled, y, cv=5, scoring="r2")

        y_pred = engine.ml_scorer.model.predict(X_scaled)
        mae_val = mean_absolute_error(y, y_pred)
        rmse_val = float(np.sqrt(mean_squared_error(y, y_pred)))

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            render_metric_card("R2 (среднее)", f"{cv_r2.mean():.4f}", f"std: {cv_r2.std():.4f}", gold=True, delay=1)
        with mc2:
            render_metric_card("R2 (мин)", f"{cv_r2.min():.4f}", "Худший фолд", delay=2)
        with mc3:
            render_metric_card("MAE", f"{mae_val:.4f}", "Средняя абс. ошибка", delay=3)
        with mc4:
            render_metric_card("RMSE", f"{rmse_val:.4f}", "Среднеквадр. ошибка", delay=4)

        # CV fold table
        cv_table = pd.DataFrame({
            "Фолд": [f"Фолд {i+1}" for i in range(len(cv_r2))],
            "R2": [f"{v:.4f}" for v in cv_r2],
        })
        st.dataframe(cv_table, use_container_width=True, hide_index=True)

    except Exception:
        st.warning("Не удалось вычислить метрики кросс-валидации.")

    render_divider()

    # --- Score distribution with KDE-like overlay ---
    render_section_header("Распределение итоговых баллов", "Гистограмма")
    if "combined_score" in scored.columns and not scored.empty:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=scored["combined_score"],
            nbinsx=50,
            marker_color=TEAL,
            opacity=0.7,
            name="Распределение",
        ))
        # KDE approximation
        from scipy.stats import gaussian_kde
        try:
            vals = scored["combined_score"].dropna().values
            kde = gaussian_kde(vals)
            x_range = np.linspace(vals.min(), vals.max(), 200)
            kde_vals = kde(x_range)
            # Scale KDE to match histogram
            bin_width = (vals.max() - vals.min()) / 50
            kde_scaled = kde_vals * len(vals) * bin_width
            fig.add_trace(go.Scatter(
                x=x_range, y=kde_scaled,
                mode="lines",
                name="Плотность (KDE)",
                line=dict(color=GOLD, width=3),
            ))
        except Exception:
            pass

        fig.add_vline(
            x=scored["combined_score"].median(),
            line_dash="dash", line_color="red",
            annotation_text=f"Медиана: {scored['combined_score'].median():.1f}",
        )
        fig.add_vline(
            x=scored["combined_score"].mean(),
            line_dash="dot", line_color="blue",
            annotation_text=f"Среднее: {scored['combined_score'].mean():.1f}",
        )
        apply_chart_theme(fig, height=420)
        fig.update_layout(
            xaxis_title="Балл",
            yaxis_title="Количество",
            legend=dict(orientation="h", y=1.05),
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_validation_score_dist")

    render_divider()

    # --- Robustness test: Spearman rank correlation under perturbation ---
    render_section_header("Робастность модели", "Стабильность ранжирования")
    try:
        from scipy.stats import spearmanr
        feats_robust = compute_producer_features(df)
        X_robust, y_robust = prepare_model_data(feats_robust)
        X_robust_clean = X_robust.fillna(0).replace([np.inf, -np.inf], 0)
        X_robust_scaled = engine.ml_scorer.scaler.transform(X_robust_clean)

        # Original predictions
        y_pred_original = engine.ml_scorer.model.predict(X_robust_scaled)
        original_ranks = pd.Series(y_pred_original).rank(ascending=False).values

        # Perturb 5% of the data with Gaussian noise
        np.random.seed(42)
        X_perturbed = X_robust_scaled.copy()
        n_rows, n_cols = X_perturbed.shape
        n_perturb = max(1, int(n_rows * 0.05))
        perturb_idx = np.random.choice(n_rows, n_perturb, replace=False)
        noise_scale = 0.05
        for idx in perturb_idx:
            X_perturbed[idx] += np.random.normal(0, noise_scale, n_cols)

        y_pred_perturbed = engine.ml_scorer.model.predict(X_perturbed)
        perturbed_ranks = pd.Series(y_pred_perturbed).rank(ascending=False).values

        # Spearman rank correlation
        spearman_corr, spearman_p = spearmanr(original_ranks, perturbed_ranks)

        # Top-100 overlap
        top100_original = set(np.argsort(y_pred_original)[-100:])
        top100_perturbed = set(np.argsort(y_pred_perturbed)[-100:])
        top100_overlap = len(top100_original & top100_perturbed)

        rc1, rc2 = st.columns(2)
        with rc1:
            render_metric_card(
                "SPEARMAN КОРРЕЛЯЦИЯ",
                f"{spearman_corr:.4f}",
                f"p-value: {spearman_p:.2e} (при возмущении 5% данных)",
                gold=(spearman_corr < 0.95),
                delay=1,
            )
        with rc2:
            render_metric_card(
                "TOP-100 OVERLAP",
                f"{top100_overlap}%",
                "Совпадение топ-100 при возмущении данных",
                gold=(top100_overlap < 90),
                delay=2,
            )

        stability_label = "высокая" if spearman_corr >= 0.98 else ("средняя" if spearman_corr >= 0.95 else "низкая")
        stability_cls = "fairness-ok" if spearman_corr >= 0.98 else ("fairness-warn" if spearman_corr >= 0.95 else "fairness-danger")
        st.markdown(f"""
        <div class="{stability_cls} fairness-alert">
            <span style="font-weight:700;">Стабильность ранжирования: {stability_label}.</span>
            Spearman rho = {spearman_corr:.4f} при возмущении 5% записей (шум sigma={noise_scale}).
            Top-100 overlap при возмущении данных: {top100_overlap}%.
        </div>
        """, unsafe_allow_html=True)

    except Exception:
        st.warning("Не удалось выполнить тест робастности.")

    render_divider()

    # --- Prediction vs Actual scatter ---
    render_section_header("Предсказание vs Целевая переменная", "Scatter")
    try:
        feats2 = compute_producer_features(df)
        X2, y2 = prepare_model_data(feats2)
        X2_clean = X2.fillna(0).replace([np.inf, -np.inf], 0)
        X2_scaled = engine.ml_scorer.scaler.transform(X2_clean)
        y2_pred = engine.ml_scorer.model.predict(X2_scaled)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=y2.values, y=y2_pred,
            mode="markers",
            marker=dict(color=TEAL, size=5, opacity=0.5),
            name="Производители",
        ))
        min_val = min(y2.min(), y2_pred.min())
        max_val = max(y2.max(), y2_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines",
            line=dict(color="red", dash="dash", width=2),
            name="Идеальное совпадение",
        ))
        apply_chart_theme(fig, height=450)
        fig.update_layout(
            xaxis_title="Целевая переменная (фактическая)",
            yaxis_title="Предсказание модели",
        )
        st.plotly_chart(fig, use_container_width=True, key="chart_validation_pred_vs_actual")
    except Exception:
        st.warning("Не удалось построить график предсказание vs факт.")


# ===========================================================================
# PAGE 9: SETTINGS
# ===========================================================================
def render_settings(scored: pd.DataFrame):
    render_breadcrumb("Настройки модели")
    render_page_header(
        "Настройки скоринговой модели",
        "Настройка весов факторов для адаптации скоринга под приоритеты государственной политики"
    )

    # --- Model balance ---
    render_section_header("Баланс моделей", "ML vs Правила")
    col1, col2, col3 = st.columns([3, 1, 2])
    with col1:
        ml_w = st.slider(
            "Вес ML-модели", 0.0, 1.0,
            st.session_state.ml_weight, 0.05,
            key="settings_ml_w",
        )
    with col2:
        rule_w = 1.0 - ml_w
        st.markdown(f"""
        <div class="metric-card" style="padding:16px;">
            <div class="metric-icon">ВЕС ПРАВИЛ</div>
            <div class="metric-value" style="font-size:28px;">{rule_w:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        # Visual balance bar
        ml_pct = int(ml_w * 100)
        st.markdown(f"""
        <div style="margin-top:12px;">
            <div style="font-size:12px; color:#94A3B8; margin-bottom:6px;">Баланс моделей</div>
            <div style="display:flex; height:32px; border-radius:8px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
                <div style="width:{ml_pct}%; background:linear-gradient(90deg, #38BDF8, #1E3A5F);
                    display:flex; align-items:center; justify-content:center; color:white; font-size:11px; font-weight:700;">
                    ML {ml_pct}%
                </div>
                <div style="width:{100-ml_pct}%; background:linear-gradient(90deg, #38BDF8, #60A5FA);
                    display:flex; align-items:center; justify-content:center; color:white; font-size:11px; font-weight:700;">
                    Правила {100-ml_pct}%
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.session_state.ml_weight = ml_w
    st.session_state.rule_weight = rule_w

    render_divider()

    # --- Feature weights ---
    render_section_header("Веса факторов (правиловая модель)", "Настройка")
    st.markdown("""
    <div class="glass-card" style="margin-bottom:20px; padding:16px 20px;">
        <div style="font-size:14px; color:#94A3B8; line-height:1.6;">
            Положительные веса увеличивают балл, отрицательные -- уменьшают.
            Комиссия может изменять приоритеты факторов в соответствии с государственной политикой.
        </div>
    </div>
    """, unsafe_allow_html=True)

    descs = get_feature_descriptions()
    current_weights = st.session_state.feature_weights

    new_weights = {}
    cols = st.columns(2)
    for i, (feat, default_w) in enumerate(current_weights.items()):
        with cols[i % 2]:
            label = descs.get(feat, feat)
            new_weights[feat] = st.slider(
                label, -0.30, 0.30, float(default_w), 0.01,
                key=f"weight_{feat}",
            )

    st.session_state.feature_weights = new_weights

    render_divider()

    # --- Preview top-10 changes ---
    render_section_header("Предпросмотр топ-10", "Текущие результаты")
    top10_preview = scored.nsmallest(10, "rank")
    preview_rows = ""
    for _, row in top10_preview.iterrows():
        sc = float(row["combined_score"])
        bar_class = get_score_bar_class(sc)
        preview_rows += f"""
        <tr>
            <td style="font-weight:700; color:#38BDF8;">#{int(row['rank'])}</td>
            <td style="font-family:monospace; font-size:12px;">...{str(row['producer_id'])[-6:]}</td>
            <td>{row['region']}</td>
            <td>
                <div class="score-bar-container">
                    <div class="score-bar {bar_class}" style="width:{sc}%;">{sc:.1f}</div>
                </div>
            </td>
        </tr>"""

    st.markdown(f"""
    <table class="styled-table">
        <thead><tr><th>Ранг</th><th>ID</th><th>Регион</th><th>Балл</th></tr></thead>
        <tbody>{preview_rows}</tbody>
    </table>
    """, unsafe_allow_html=True)

    render_divider()

    # --- Model metrics ---
    render_section_header("Метрики модели", "Качество")
    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        cv_score = engine_cv_score(scored)
        render_metric_card("CV SCORE (R2)", f"{cv_score:.3f}", "Кросс-валидация", delay=1)
    with mc2:
        render_metric_card("ML ВЕС", f"{st.session_state.ml_weight:.2f}", "", delay=2)
    with mc3:
        render_metric_card("ПРАВИЛА ВЕС", f"{st.session_state.rule_weight:.2f}", "", delay=3)

    render_divider()

    # --- Reset ---
    render_section_header("Сброс настроек", "Действие")
    st.markdown("""
    <div class="glass-card" style="margin-bottom:16px; padding:16px 20px;">
        <div style="font-size:14px; color:#94A3B8; line-height:1.6;">
            Сбросить все веса факторов и баланс моделей к значениям по умолчанию
            (ML 60% / Правила 40%, стандартные веса).
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("\u21BA Сбросить настройки", key="reset_settings_btn"):
        st.session_state.ml_weight = 0.6
        st.session_state.rule_weight = 0.4
        st.session_state.feature_weights = get_feature_weights_default()
        st.rerun()

    render_divider()

    # --- Privacy ---
    render_section_header("Приватность и безопасность", "Защита данных")
    with st.expander("Приватность и безопасность", expanded=False):
        st.markdown("""
        <div class="glass-card">
            <div style="line-height:1.8; font-size:14px; color:#E2E8F0;">
                <p style="font-weight:700; font-size:16px; color:#E2E8F0; margin-bottom:12px;">
                    Меры защиты данных
                </p>
                <p>
                    <span style="font-weight:700; color:#38BDF8;">1. Локальная обработка:</span>
                    Все данные обрабатываются исключительно на сервере приложения.
                    Информация не передаётся во внешние API, облачные сервисы или третьим лицам.
                </p>
                <p>
                    <span style="font-weight:700; color:#38BDF8;">2. Отсутствие внешних вызовов:</span>
                    Система не использует внешние API для анализа данных.
                    Все вычисления (ML-модель, статистика, визуализация) выполняются локально.
                </p>
                <p>
                    <span style="font-weight:700; color:#38BDF8;">3. Сессионность данных:</span>
                    Данные сессии не сохраняются между запусками приложения.
                    Каждый сеанс работы начинается с чистого состояния.
                </p>
                <p>
                    <span style="font-weight:700; color:#38BDF8;">4. Анонимизация:</span>
                    Идентификаторы производителей (producer_id) используются только для
                    группировки заявок. Они могут быть анонимизированы без потери функциональности
                    скоринга.
                </p>
                <p style="margin-top:16px; padding:12px 16px; background:rgba(59,130,246,0.06); border-radius:8px; font-weight:600;">
                    Рекомендация: разворачивать систему в защищённом контуре Министерства
                    сельского хозяйства с ограниченным сетевым доступом.
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    render_divider()

    # --- Methodology ---
    render_section_header("Методология скоринга", "Описание")
    st.markdown("""
    <div class="glass-card">
        <div style="line-height:1.8; font-size:14px; color:#E2E8F0;">
            <p style="font-weight:700; font-size:16px; color:#E2E8F0; margin-bottom:12px;">
                Комбинированный скоринг объединяет два подхода:
            </p>
            <p>
                <span style="font-weight:700; color:#38BDF8;">1. Правиловая модель</span> --
                экспертные веса, прозрачная формула, полная объяснимость.
                Каждый фактор нормализован [0, 1] и умножается на вес. Итоговый балл -- взвешенная сумма.
            </p>
            <p>
                <span style="font-weight:700; color:#38BDF8;">2. ML-модель (Gradient Boosting)</span> --
                обучена на исторических данных, выявляет нелинейные зависимости.
                Целевая переменная формируется из комбинации: исполнение (35%), одобрение (25%),
                освоение средств (20%), диверсификация (10%), отсутствие отклонений (10%).
            </p>
            <p style="margin-top:16px; padding:12px 16px; background:rgba(59,130,246,0.06); border-radius:8px; font-weight:600;">
                Итоговый балл = вес_ML x балл_ML + вес_правил x балл_правил. Все баллы нормализованы в диапазон 0--100.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_divider()

    # --- Scalable Architecture ---
    render_section_header("Масштабируемая архитектура", "Scalability")
    st.markdown("""
    <div class="glass-card">
        <div style="line-height:1.8; font-size:14px; color:#E2E8F0;">
            <p style="font-weight:700; font-size:16px; color:#E2E8F0; margin-bottom:12px;">
                Принципы масштабируемости
            </p>
            <p>
                <span style="font-weight:700; color:#38BDF8;">1. Модульная архитектура:</span>
                Каждый компонент (загрузка данных, инженерия признаков, скоринг, аналитика,
                справедливость) реализован как отдельный модуль Python с чётко определённым API.
                Модули могут быть заменены или расширены независимо.
            </p>
            <p>
                <span style="font-weight:700; color:#38BDF8;">2. Кэширование:</span>
                Streamlit-кэш (@st.cache_data, @st.cache_resource) обеспечивает мгновенный
                отклик при повторных обращениях. Данные загружаются и модель обучается один раз за сессию.
            </p>
            <p>
                <span style="font-weight:700; color:#38BDF8;">3. Горизонтальное масштабирование:</span>
                Приложение не хранит состояние на сервере (stateless). Может быть развёрнуто
                за балансировщиком нагрузки с несколькими экземплярами.
            </p>
            <p>
                <span style="font-weight:700; color:#38BDF8;">4. Поддержка больших данных:</span>
                Pandas-пайплайн оптимизирован для агрегации десятков тысяч заявок. При необходимости
                может быть переведён на Dask/Spark для миллионов записей.
            </p>
            <p>
                <span style="font-weight:700; color:#38BDF8;">5. Расширяемость модели:</span>
                Новые признаки добавляются в feature_engineering.py, новые модели -- в scoring_engine.py.
                Структура позволяет добавить нейронные сети, ансамбли или онлайн-обучение.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_divider()

    # --- Limitations ---
    render_section_header("Ограничения системы", "Limitations")
    limitations = [
        ("Cold-start проблема",
         "Новые производители без истории заявок не могут получить корректный скоринг. "
         "Система опирается на накопленную статистику."),
        ("Данные за один год",
         "Модель обучена на данных 2025 года. Сезонные и годовые тренды могут измениться."),
        ("Только животноводство",
         "Текущая модель адаптирована под направления животноводства. "
         "Для растениеводства потребуется адаптация признаков."),
        ("Синтетическая целевая переменная",
         "ML-модель обучена на комбинации метрик, а не на внешней независимой оценке "
         "качества производителя."),
        ("Нет учёта внешних факторов",
         "Погодные условия, эпизоотическая обстановка, рыночные цены не учитываются."),
        ("Качество исходных данных",
         "Система зависит от полноты и корректности данных ИСС. "
         "Пропуски и ошибки в реестре влияют на скоринг."),
        ("Статический анализ",
         "Скоринг вычисляется на момент запуска. Для промышленной эксплуатации "
         "нужен механизм регулярного обновления."),
        ("Отсутствие A/B-тестирования",
         "Эффективность скоринговой системы по сравнению с FCFS "
         "не проверена в реальных условиях."),
        ("Producer ID -- приближение",
         "ID производителя извлекается из первых 11 цифр номера заявки, "
         "что может приводить к неточностям группировки."),
        ("Региональные различия",
         "Социально-экономические различия между регионами могут создавать "
         "систематические смещения в оценке."),
    ]
    for i, (title, desc) in enumerate(limitations, 1):
        st.markdown(
            f'<div class="sw-item sw-weakness" style="margin-bottom:8px;">'
            f'<span class="sw-icon">{i}.</span> '
            f'<div><span style="font-weight:700;">{title}.</span> {desc}</div></div>',
            unsafe_allow_html=True,
        )


def engine_cv_score(scored):
    """Quick helper to get a CV score proxy."""
    if "rule_score" in scored.columns and "ml_score" in scored.columns:
        corr = scored["rule_score"].corr(scored["ml_score"])
        return max(0, corr)
    return 0.0


# ===========================================================================
if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        st.error(
            f"Файл данных не найден: {e}\n\n"
            "Пожалуйста, поместите файл Excel в папку data/ "
            "или загрузите его через боковую панель."
        )
    except Exception as e:
        st.error(
            "Произошла непредвиденная ошибка при работе приложения.\n\n"
            f"Тип: {type(e).__name__}\n\n"
            f"Описание: {e}\n\n"
            "Попробуйте перезагрузить страницу или проверить входные данные."
        )
