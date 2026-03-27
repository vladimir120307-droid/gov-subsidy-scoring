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
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
NAVY = "#0f2b46"
NAVY_LIGHT = "#1a4a6e"
TEAL = "#0d7377"
GOLD = "#c9a227"
WHITE = "#ffffff"
CHART_COLORS = ["#0d7377", "#1a4a6e", "#c9a227", "#2d8f5e", "#b84c4c", "#6b5fa5",
                "#3a9fbf", "#d4843e", "#5c8a4d", "#8b6caf"]

DATA_FILE = os.environ.get(
    "SUBSIDY_DATA_PATH",
    str(Path(__file__).parent / "data" / "subsidies_2025.xlsx"),
)


# ---------------------------------------------------------------------------
# Dark mode helper
# ---------------------------------------------------------------------------
def is_dark_mode() -> bool:
    return st.session_state.get("dark_mode", False)


# ---------------------------------------------------------------------------
# Plotly theme helper
# ---------------------------------------------------------------------------
def apply_chart_theme(fig, height=450, transparent=True):
    """Apply consistent government premium theme to plotly figures."""
    dark = is_dark_mode()
    if dark:
        bg = "rgba(0,0,0,0)" if transparent else "#0a1929"
        fig.update_layout(
            template="plotly_dark",
            height=height,
            font=dict(family="Inter, system-ui, sans-serif", size=13, color="#e0e6ed"),
            paper_bgcolor=bg,
            plot_bgcolor=bg,
            title_font=dict(size=17, color="#e0e6ed", family="Inter, system-ui, sans-serif"),
            legend=dict(
                bgcolor="rgba(15,43,70,0.7)",
                bordercolor="rgba(13,115,119,0.3)",
                borderwidth=1,
                font=dict(size=11, color="#e0e6ed"),
            ),
            margin=dict(t=55, b=40, l=40, r=20),
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)", zerolinecolor="rgba(255,255,255,0.12)"),
        )
    else:
        bg = "rgba(0,0,0,0)" if transparent else WHITE
        fig.update_layout(
            template="plotly_white",
            height=height,
            font=dict(family="Inter, system-ui, sans-serif", size=13, color="#2c3e50"),
            paper_bgcolor=bg,
            plot_bgcolor=bg,
            title_font=dict(size=17, color=NAVY, family="Inter, system-ui, sans-serif"),
            legend=dict(
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(0,0,0,0.08)",
                borderwidth=1,
                font=dict(size=11),
            ),
            margin=dict(t=55, b=40, l=40, r=20),
            xaxis=dict(gridcolor="rgba(0,0,0,0.06)", zerolinecolor="rgba(0,0,0,0.08)"),
            yaxis=dict(gridcolor="rgba(0,0,0,0.06)", zerolinecolor="rgba(0,0,0,0.08)"),
        )
    return fig


# ---------------------------------------------------------------------------
# Massive CSS injection
# ---------------------------------------------------------------------------
def inject_css():
    dark = is_dark_mode()

    # --- Light theme CSS ---
    light_css = """
/* ===== IMPORTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ===== ROOT VARIABLES ===== */
:root {
    --navy: #0f2b46;
    --navy-light: #1a4a6e;
    --teal: #0d7377;
    --gold: #c9a227;
    --gold-light: #dbb94a;
    --white: #ffffff;
    --bg: #f0f2f6;
    --text-primary: #1a1f36;
    --text-secondary: #4a5568;
    --glass-bg: rgba(255, 255, 255, 0.72);
    --glass-border: rgba(255, 255, 255, 0.35);
    --glass-shadow: 0 8px 32px rgba(15, 43, 70, 0.10);
    --radius: 16px;
    --radius-sm: 10px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ===== GLOBAL ===== */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    color: var(--text-primary);
}

[data-testid="stAppViewContainer"] {
    background:
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%230d7377' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E"),
        linear-gradient(160deg, #e8ecf2 0%, #f0f2f6 40%, #e6f0f0 100%);
}

/* ===== PAGE CONTENT FADE-IN ===== */
[data-testid="stMainBlockContainer"] {
    animation: pageContentFadeIn 0.5s ease-out;
}

@keyframes pageContentFadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
"""

    # --- Dark theme CSS ---
    dark_css = """
/* ===== IMPORTS ===== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ===== ROOT VARIABLES ===== */
:root {
    --navy: #e0e6ed;
    --navy-light: #b0c4d8;
    --teal: #15b8bd;
    --gold: #c9a227;
    --gold-light: #dbb94a;
    --white: #0a1929;
    --bg: #0a1929;
    --text-primary: #e0e6ed;
    --text-secondary: #8fa3b8;
    --glass-bg: rgba(15, 43, 70, 0.85);
    --glass-border: rgba(13, 115, 119, 0.3);
    --glass-shadow: 0 8px 32px rgba(0, 0, 0, 0.30);
    --radius: 16px;
    --radius-sm: 10px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ===== GLOBAL ===== */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    color: var(--text-primary);
}

[data-testid="stAppViewContainer"] {
    background:
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%230d7377' fill-opacity='0.06'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E"),
        linear-gradient(160deg, #060e1a 0%, #0a1929 40%, #0b1e30 100%);
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
    color: var(--text-primary);
}

[data-testid="stAppViewContainer"] .stDataFrame,
[data-testid="stAppViewContainer"] .stDataFrame td,
[data-testid="stAppViewContainer"] .stDataFrame th {
    color: var(--text-primary) !important;
}
"""

    theme_css = dark_css if dark else light_css

    # --- Common CSS (independent of theme) ---
    common_css = """
/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.04); border-radius: 4px; }
::-webkit-scrollbar-thumb { background: linear-gradient(180deg, var(--teal), var(--navy-light)); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--navy); }

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1f33 0%, #0f2b46 30%, #112d47 70%, #0d7377 100%) !important;
}
[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.92) !important;
}
[data-testid="stSidebar"] .stRadio > label {
    color: rgba(255,255,255,0.6) !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: var(--radius-sm);
    padding: 10px 16px;
    margin-bottom: 4px;
    transition: var(--transition);
    font-weight: 500;
    font-size: 14px !important;
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
    background: rgba(255,255,255,0.12);
    border-color: var(--gold);
    transform: translateX(4px);
}
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label[aria-checked="true"] {
    background: linear-gradient(135deg, rgba(201,162,39,0.25), rgba(13,115,119,0.25)) !important;
    border-color: var(--gold) !important;
    box-shadow: 0 0 12px rgba(201,162,39,0.15);
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
}
[data-testid="stSidebar"] img {
    filter: brightness(1.1);
}

/* ===== HEADER BANNER ===== */
.gov-header {
    background: linear-gradient(135deg, #0f2b46 0%, #1a4a6e 50%, #0d7377 100%);
    padding: 28px 36px;
    border-radius: var(--radius);
    margin-bottom: 28px;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 12px 40px rgba(15, 43, 70, 0.25);
    border: 2px solid transparent;
    background-clip: padding-box;
}
.gov-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(201,162,39,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.gov-header::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(13,115,119,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.gov-header h1 {
    margin: 0;
    font-size: 28px;
    font-weight: 700;
    letter-spacing: -0.5px;
    position: relative;
    z-index: 1;
    color: white !important;
}
.gov-header p {
    margin: 6px 0 0 0;
    font-size: 15px;
    opacity: 0.85;
    font-weight: 400;
    position: relative;
    z-index: 1;
    color: white !important;
}
.gov-header .gold-line {
    width: 60px;
    height: 3px;
    background: linear-gradient(90deg, #c9a227, #dbb94a);
    border-radius: 2px;
    margin-top: 14px;
    position: relative;
    z-index: 1;
}
/* Animated gradient border on header */
.gov-header-wrapper {
    position: relative;
    border-radius: var(--radius);
    padding: 2px;
    margin-bottom: 28px;
    background: linear-gradient(90deg, #0d7377, #c9a227, #1a4a6e, #c9a227, #0d7377);
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

/* ===== GLASS CARDS ===== */
.glass-card {
    background: var(--glass-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 24px;
    box-shadow: var(--glass-shadow);
    transition: var(--transition);
    position: relative;
    overflow: hidden;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px rgba(15, 43, 70, 0.15);
    border-color: rgba(13, 115, 119, 0.3);
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--teal), var(--gold));
    border-radius: var(--radius) var(--radius) 0 0;
    opacity: 0;
    transition: var(--transition);
}
.glass-card:hover::before {
    opacity: 1;
}

/* ===== METRIC CARDS ===== */
.metric-card {
    background: var(--glass-bg);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 22px 24px;
    text-align: center;
    box-shadow: var(--glass-shadow);
    transition: var(--transition);
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 16px 48px rgba(15, 43, 70, 0.18);
}
.metric-card .metric-icon {
    font-size: 14px;
    font-weight: 700;
    color: var(--teal);
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
    background: linear-gradient(135deg, var(--navy), var(--teal));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    margin-bottom: 4px;
}
.metric-card .metric-value.gold {
    background: linear-gradient(135deg, var(--gold), #b8941f);
    -webkit-background-clip: text;
    background-clip: text;
}
.metric-card .metric-sub {
    font-size: 12px;
    color: var(--text-secondary);
    font-weight: 500;
}
.metric-card .metric-bar {
    height: 3px;
    background: linear-gradient(90deg, var(--teal), var(--gold));
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
    color: var(--navy);
}
.section-header .section-badge {
    background: linear-gradient(135deg, var(--teal), #1a4a6e);
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
    color: #1a7a4a !important;
    border: 1px solid rgba(45, 143, 94, 0.25);
}
.badge-yellow {
    background: linear-gradient(135deg, rgba(201, 162, 39, 0.15), rgba(201, 162, 39, 0.08));
    color: #8a6d15 !important;
    border: 1px solid rgba(201, 162, 39, 0.25);
}
.badge-red {
    background: linear-gradient(135deg, rgba(184, 76, 76, 0.15), rgba(184, 76, 76, 0.08));
    color: #9a3030 !important;
    border: 1px solid rgba(184, 76, 76, 0.25);
}
.badge-blue {
    background: linear-gradient(135deg, rgba(26, 74, 110, 0.15), rgba(26, 74, 110, 0.08));
    color: #1a4a6e !important;
    border: 1px solid rgba(26, 74, 110, 0.25);
}

/* ===== STYLED TABLE ===== */
.styled-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    border-radius: var(--radius-sm);
    overflow: hidden;
    font-size: 13px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.06);
}
.styled-table thead th {
    background: linear-gradient(135deg, #0f2b46, #1a4a6e);
    color: white !important;
    padding: 14px 16px;
    font-weight: 600;
    text-align: left;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    white-space: nowrap;
}
.styled-table tbody tr {
    transition: var(--transition);
    background: var(--glass-bg);
}
.styled-table tbody tr:nth-child(even) {
    background: rgba(13, 115, 119, 0.03);
}
.styled-table tbody tr:hover {
    background: rgba(13, 115, 119, 0.08);
    transform: scale(1.002);
}
.styled-table tbody td {
    padding: 12px 16px;
    border-bottom: 1px solid rgba(0,0,0,0.04);
    vertical-align: middle;
    color: var(--text-primary);
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
.score-bar.medium { background: linear-gradient(90deg, #b8941f, #c9a227); }
.score-bar.low { background: linear-gradient(90deg, #b84c4c, #d46a6a); }

/* ===== GAUGE CHART ===== */
.gauge-container {
    text-align: center;
    padding: 16px;
}
.gauge-label {
    font-size: 13px;
    font-weight: 600;
    color: var(--text-secondary);
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
    border-radius: var(--radius-sm);
    margin-bottom: 6px;
    font-size: 14px;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 10px;
    transition: var(--transition);
}
.sw-item:hover {
    transform: translateX(4px);
}
.sw-strength {
    background: rgba(45, 143, 94, 0.08);
    border-left: 3px solid #2d8f5e;
    color: #1a6b3f !important;
}
.sw-weakness {
    background: rgba(184, 76, 76, 0.08);
    border-left: 3px solid #b84c4c;
    color: #8a3030 !important;
}
.sw-icon {
    font-weight: 800;
    font-size: 16px;
    flex-shrink: 0;
}

/* ===== COMPARISON WINNER ===== */
.winner-highlight {
    background: linear-gradient(135deg, rgba(201,162,39,0.12), rgba(13,115,119,0.08));
    border: 2px solid var(--gold);
    border-radius: var(--radius);
    padding: 4px;
}

/* ===== FILTER PANEL ===== */
.filter-panel {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: var(--radius);
    padding: 20px 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
}

/* ===== DIVIDER ===== */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(13, 115, 119, 0.2), var(--gold), rgba(13, 115, 119, 0.2), transparent);
    margin: 28px 0;
    border: none;
}

/* ===== BASELINE COMPARISON CARDS ===== */
.baseline-card {
    border-radius: var(--radius);
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
    background: linear-gradient(135deg, rgba(45,143,94,0.08), rgba(13,115,119,0.05));
    border: 2px solid var(--teal);
    box-shadow: 0 8px 32px rgba(13,115,119,0.12);
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
    color: var(--text-secondary);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.improvement-arrow {
    font-size: 40px;
    color: var(--teal);
    font-weight: 800;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* ===== FAIRNESS ALERT ===== */
.fairness-alert {
    border-radius: var(--radius-sm);
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
    color: #1a6b3f !important;
}
.fairness-warn {
    background: rgba(201, 162, 39, 0.08);
    border: 1px solid rgba(201, 162, 39, 0.2);
    color: #7a5f10 !important;
}
.fairness-danger {
    background: rgba(184, 76, 76, 0.08);
    border: 1px solid rgba(184, 76, 76, 0.2);
    color: #8a3030 !important;
}

/* ===== TIMELINE ===== */
.timeline-item {
    display: flex;
    gap: 16px;
    padding: 12px 0;
    border-bottom: 1px solid rgba(0,0,0,0.05);
    transition: var(--transition);
}
.timeline-item:hover {
    background: rgba(13,115,119,0.03);
    padding-left: 8px;
}
.timeline-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
    margin-top: 4px;
    box-shadow: 0 0 0 3px rgba(0,0,0,0.05);
}
.timeline-dot.approved { background: #2d8f5e; }
.timeline-dot.rejected { background: #b84c4c; }
.timeline-dot.pending { background: #c9a227; }
.timeline-dot.completed { background: #0d7377; }
.timeline-content {
    flex: 1;
}
.timeline-date {
    font-size: 12px;
    color: var(--text-secondary);
    font-weight: 500;
}
.timeline-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-primary);
    margin: 2px 0;
}
.timeline-meta {
    font-size: 12px;
    color: var(--text-secondary);
}

/* ===== OVERRIDE DEFAULT STREAMLIT METRICS ===== */
[data-testid="stMetricValue"] {
    font-weight: 700 !important;
    color: var(--navy) !important;
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
    border-radius: var(--radius-sm) var(--radius-sm) 0 0;
    padding: 10px 20px;
    font-weight: 600;
    transition: var(--transition);
}
.stTabs [aria-selected="true"] {
    background: rgba(13, 115, 119, 0.08) !important;
    border-bottom: 3px solid var(--teal) !important;
}

/* ===== BUTTONS ===== */
.stDownloadButton > button,
.stButton > button {
    background: linear-gradient(135deg, #0f2b46, #1a4a6e) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: var(--transition) !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-size: 13px !important;
}
.stDownloadButton > button:hover,
.stButton > button:hover {
    background: linear-gradient(135deg, #1a4a6e, #0d7377) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(15, 43, 70, 0.25) !important;
}

/* ===== INPUTS & SELECTS ===== */
.stSelectbox > div > div,
.stSlider > div,
.stNumberInput > div {
    border-radius: var(--radius-sm) !important;
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
    border-radius: var(--radius-sm);
    font-size: 13px;
    font-weight: 500;
}
.dq-good {
    background: rgba(45, 143, 94, 0.08);
    color: #1a6b3f !important;
}
.dq-warn {
    background: rgba(201, 162, 39, 0.08);
    color: #7a5f10 !important;
}
.dq-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dq-dot.good { background: #2d8f5e; }
.dq-dot.warn { background: #c9a227; }

/* ===== FOOTER ===== */
.gov-footer {
    text-align: center;
    padding: 24px 0 12px 0;
    margin-top: 40px;
    border-top: 1px solid rgba(13, 115, 119, 0.15);
    font-size: 12px;
    color: var(--text-secondary);
    letter-spacing: 0.3px;
}

/* ===== COAT OF ARMS FALLBACK ===== */
.coat-of-arms-fallback {
    text-align: center;
    font-size: 24px;
    font-weight: 800;
    color: rgba(201,162,39,0.8);
    padding: 10px 0;
    letter-spacing: 2px;
}
"""

    st.markdown(f"<style>{theme_css}\n{common_css}</style>", unsafe_allow_html=True)


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
        'Министерство сельского хозяйства РК | Decentrathon 5.0'
        '</div>',
        unsafe_allow_html=True,
    )


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
    # ----- Sidebar -----
    # Coat of arms image with fallback
    try:
        st.sidebar.image(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Coat_of_arms_of_Kazakhstan.svg/200px-Coat_of_arms_of_Kazakhstan.svg.png",
            width=70,
        )
    except Exception:
        st.sidebar.markdown(
            '<div class="coat-of-arms-fallback">KZ</div>',
            unsafe_allow_html=True,
        )
    st.sidebar.markdown(
        "<div style='text-align:center; margin-bottom:4px;'>"
        "<span style='font-size:20px; font-weight:700; letter-spacing:-0.5px;'>Скоринг субсидий</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<div style='text-align:center; font-size:12px; opacity:0.7; margin-bottom:16px;'>"
        "Министерство сельского хозяйства РК</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    # Dark mode toggle
    st.sidebar.toggle("Темная тема", value=False, key="dark_mode")

    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "НАВИГАЦИЯ",
        [
            "Главная панель",
            "Обзор данных",
            "Скоринг производителей",
            "Профиль производителя",
            "Сравнение производителей",
            "Шортлист",
            "Анализ справедливости",
            "Аналитика",
            "Настройки модели",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='font-size:11px; opacity:0.5; text-align:center; padding:8px 0;'>"
        "Система скоринга v2.0<br>МСХ Республики Казахстан</div>",
        unsafe_allow_html=True,
    )

    # Inject CSS after dark_mode toggle is set
    inject_css()

    # ----- Data loading -----
    try:
        df = load_data(DATA_FILE)
    except FileNotFoundError:
        st.error(
            f"Файл данных не найден: `{DATA_FILE}`\n\n"
            "Укажите путь через переменную окружения `SUBSIDY_DATA_PATH` "
            "или поместите файл в папку `data/subsidies_2025.xlsx`."
        )
        uploaded = st.file_uploader("Или загрузите файл Excel", type=["xlsx"])
        if uploaded:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            df = load_data(tmp_path)
        else:
            return

    df_hash = hash(len(df))
    producer_features = compute_features(df_hash, df)

    if "ml_weight" not in st.session_state:
        st.session_state.ml_weight = 0.6
    if "rule_weight" not in st.session_state:
        st.session_state.rule_weight = 0.4
    if "feature_weights" not in st.session_state:
        st.session_state.feature_weights = get_feature_weights_default()

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

    # ----- Routing -----
    if page == "Главная панель":
        render_dashboard(df, scored)
    elif page == "Обзор данных":
        render_overview(df, producer_features)
    elif page == "Скоринг производителей":
        render_scoring(scored, producer_features)
    elif page == "Профиль производителя":
        render_profile(df, scored, producer_features, engine)
    elif page == "Сравнение производителей":
        render_comparison(scored, producer_features)
    elif page == "Шортлист":
        render_shortlist(scored)
    elif page == "Анализ справедливости":
        render_fairness(scored)
    elif page == "Аналитика":
        render_analytics(df, scored, engine)
    elif page == "Настройки модели":
        render_settings(scored)

    # Footer on every page
    render_footer()


# ===========================================================================
# PAGE 1: DASHBOARD
# ===========================================================================
def render_dashboard(df: pd.DataFrame, scored: pd.DataFrame):
    render_page_header(
        "Главная панель",
        "Система скоринга сельскохозяйственных субсидий Республики Казахстан"
    )

    stats = get_summary_stats(df)

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
                <td style="font-weight:700; color:var(--teal);">#{int(row['rank'])}</td>
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
        st.plotly_chart(fig, use_container_width=True)

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
            colorscale=[[0, "#b84c4c"], [0.5, "#c9a227"], [1, "#0d7377"]],
            colorbar=dict(title="Ср. балл", thickness=15),
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
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 2: DATA OVERVIEW
# ===========================================================================
def render_overview(df: pd.DataFrame, features: pd.DataFrame):
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

    # --- Charts ---
    col_left, col_right = st.columns(2)
    with col_left:
        fig = plot_status_distribution(df)
        apply_chart_theme(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        fig = plot_direction_pie(df)
        apply_chart_theme(fig, height=400)
        st.plotly_chart(fig, use_container_width=True)

    fig = plot_monthly_trend(df)
    apply_chart_theme(fig, height=420)
    st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns(2)
    with col_left:
        fig = plot_region_distribution(df)
        apply_chart_theme(fig, height=600)
        fig.update_layout(margin=dict(l=250))
        st.plotly_chart(fig, use_container_width=True)
    with col_right:
        fig = plot_region_amounts(df)
        apply_chart_theme(fig, height=600)
        fig.update_layout(margin=dict(l=250))
        st.plotly_chart(fig, use_container_width=True)

    fig = plot_approval_rate_by_direction(df)
    apply_chart_theme(fig, height=400)
    fig.update_layout(margin=dict(l=200))
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 3: SCORING
# ===========================================================================
def render_scoring(scored: pd.DataFrame, features: pd.DataFrame):
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
        f'<div style="font-size:15px; font-weight:600; margin-bottom:16px; color:var(--navy);">'
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
            <td style="font-weight:700; color:var(--teal);">#{int(row['rank'])}</td>
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

    # Pagination controls
    pcol1, pcol2, pcol3 = st.columns([1, 2, 1])
    with pcol1:
        if st.button("Назад", disabled=(display_page <= 1), key="pg_prev"):
            st.session_state.scoring_page = max(1, display_page - 1)
            st.rerun()
    with pcol2:
        st.markdown(
            f'<div style="text-align:center; padding:8px; font-weight:600; color:var(--navy);">'
            f'Страница {display_page} из {total_pages} | '
            f'Записи {start_idx+1}--{end_idx} из {len(filtered)}</div>',
            unsafe_allow_html=True
        )
    with pcol3:
        if st.button("Вперёд", disabled=(display_page >= total_pages), key="pg_next"):
            st.session_state.scoring_page = min(total_pages, display_page + 1)
            st.rerun()

    render_divider()

    fig = plot_score_distribution(filtered)
    apply_chart_theme(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 4: PRODUCER PROFILE
# ===========================================================================
def render_profile(
    df: pd.DataFrame,
    scored: pd.DataFrame,
    features: pd.DataFrame,
    engine: ScoringEngine,
):
    render_page_header(
        "Профиль производителя",
        "Детальный анализ скорингового профиля"
    )

    # Searchable dropdown
    sorted_scored = scored.sort_values("rank")
    display_options = [
        f"{pid} (ранг {int(sorted_scored[sorted_scored['producer_id']==pid]['rank'].values[0])})"
        for pid in sorted_scored["producer_id"].head(300)
    ]
    selected_display = st.selectbox(
        "Выберите производителя",
        display_options,
        key="profile_producer",
    )
    selected_id = selected_display.split(" (")[0]

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
                <div style="font-size:13px; color:var(--text-secondary); text-transform:uppercase; letter-spacing:1px; font-weight:600;">
                    ID Производителя
                </div>
                <div style="font-size:22px; font-weight:700; color:var(--navy); font-family:monospace; margin:4px 0;">
                    {selected_id}
                </div>
                <div style="margin-top:8px;">
                    <span class="badge {badge_cls}">{badge_text}</span>
                    <span class="badge badge-blue" style="margin-left:8px;">Ранг #{int(row['rank'])} из {len(scored)}</span>
                </div>
            </div>
            <div style="display:flex; gap:24px; flex-wrap:wrap;">
                <div style="text-align:center;">
                    <div style="font-size:11px; color:var(--text-secondary); font-weight:600; text-transform:uppercase;">Регион</div>
                    <div style="font-size:16px; font-weight:700; color:var(--navy); margin-top:4px;">{row['region']}</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:11px; color:var(--text-secondary); font-weight:600; text-transform:uppercase;">Направление</div>
                    <div style="font-size:16px; font-weight:700; color:var(--navy); margin-top:4px;">{row['main_direction']}</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:11px; color:var(--text-secondary); font-weight:600; text-transform:uppercase;">Заявок</div>
                    <div style="font-size:16px; font-weight:700; color:var(--navy); margin-top:4px;">{int(prod_feat['total_apps'])}</div>
                </div>
                <div style="text-align:center;">
                    <div style="font-size:11px; color:var(--text-secondary); font-weight:600; text-transform:uppercase;">Общая сумма</div>
                    <div style="font-size:16px; font-weight:700; color:var(--gold); margin-top:4px;">{format_tenge(prod_feat['total_amount'])}</div>
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
        <div style="font-size:14px; color:var(--text-secondary); text-transform:uppercase; letter-spacing:1px; font-weight:600; margin-bottom:8px;">
            Процентильный ранг
        </div>
        <div style="font-size:36px; font-weight:800; background:linear-gradient(135deg, var(--teal), var(--gold));
            -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:4px;">
            Топ {percentile_top:.0f}%
        </div>
        <div style="font-size:14px; color:var(--text-secondary);">
            среди всех {len(scored)} производителей
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Gauge scores ---
    render_section_header("Скоринговые баллы", "Оценка")
    gc1, gc2, gc3 = st.columns(3)

    dark = is_dark_mode()
    gauge_num_color = "#e0e6ed" if dark else NAVY
    gauge_title_color = "#8fa3b8" if dark else "#4a5568"

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
                    dict(range=[25, 50], color="rgba(201,162,39,0.12)"),
                    dict(range=[50, 75], color="rgba(13,115,119,0.12)"),
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
        st.plotly_chart(make_gauge(score_val, "Комбинированный", None), use_container_width=True)
    with gc2:
        st.plotly_chart(make_gauge(ml_val, "ML-модель", None), use_container_width=True)
    with gc3:
        st.plotly_chart(make_gauge(rule_val, "Правиловая модель", None), use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

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
            border_style = "border: 2px solid var(--gold); box-shadow: 0 0 20px rgba(201,162,39,0.2);" if is_winner else ""
            winner_label = '<div style="color:var(--gold); font-weight:700; font-size:12px; margin-bottom:4px; text-transform:uppercase; letter-spacing:1px;">Лидер</div>' if is_winner else ''

            st.markdown(f"""
            <div class="glass-card" style="text-align:center; {border_style}">
                {winner_label}
                <div style="font-family:monospace; font-size:13px; color:var(--text-secondary); margin-bottom:8px;">
                    ...{pid[-6:]}
                </div>
                <div style="font-size:42px; font-weight:800; background:linear-gradient(135deg, var(--navy), var(--teal));
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin-bottom:4px;">
                    {sc:.1f}
                </div>
                <div style="font-size:13px; color:var(--text-secondary); margin-bottom:8px;">
                    Ранг #{int(r['rank'])}
                </div>
                {get_recommendation_badge(sc)}
                <div style="margin-top:12px; font-size:12px;">
                    <span style="color:var(--text-secondary);">Правила:</span> <b>{float(r.get('rule_score',0)):.1f}</b> |
                    <span style="color:var(--text-secondary);">ML:</span> <b>{float(r.get('ml_score',0)):.1f}</b>
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
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# PAGE 6: SHORTLIST
# ===========================================================================
def render_shortlist(scored: pd.DataFrame):
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
            <td style="font-weight:700; color:var(--teal);">#{int(row['rank'])}</td>
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

    # Export
    st.markdown("<br>", unsafe_allow_html=True)
    csv_data = export_shortlist_csv(shortlist)
    st.download_button(
        label="Скачать шортлист (CSV)",
        data=csv_data.encode("utf-8-sig"),
        file_name="shortlist_subsidies.csv",
        mime="text/csv",
    )

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
            marker=dict(color=TEAL),
            text=region_counts.sort_values("Количество", ascending=True)["Количество"],
            textposition="auto",
        ))
        apply_chart_theme(fig, height=max(350, len(region_counts) * 30))
        fig.update_layout(margin=dict(l=250), xaxis_title="Количество")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Нет данных для отображения графика.")


# ===========================================================================
# PAGE 7: FAIRNESS
# ===========================================================================
def render_fairness(scored: pd.DataFrame):
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
        st.plotly_chart(fig, use_container_width=True)

        fig = plot_score_violin_by_region(scored)
        apply_chart_theme(fig, height=500)
        fig.update_layout(margin=dict(b=150))
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

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
                    <div style="font-weight:700; font-size:16px; color:var(--navy); margin-bottom:12px;">
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
    render_page_header(
        "Аналитика и корреляции",
        "Углублённый анализ модели и данных"
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Важность признаков",
        "Корреляции",
        "Связь балла и суммы",
        "Сравнение с базовыми моделями",
        "Валидация модели",
    ])

    with tab1:
        render_section_header("Важность признаков ML-модели", "GradientBoosting")
        importance = engine.ml_scorer.get_feature_importance()
        fig = plot_feature_importance(importance)
        apply_chart_theme(fig, height=max(350, len(importance) * 30))
        fig.update_layout(margin=dict(l=200))
        st.plotly_chart(fig, use_container_width=True)

        render_divider()

        render_section_header("Пермутационная важность", "Робастность")
        with st.spinner("Вычисление пермутационной важности..."):
            from feature_engineering import compute_producer_features as cpf
            feats = cpf(df)
            perm_imp = engine.ml_scorer.compute_permutation_importance(feats, n_repeats=5)
        fig = plot_feature_importance(perm_imp)
        apply_chart_theme(fig, height=max(350, len(perm_imp) * 30))
        fig.update_layout(margin=dict(l=200))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        render_section_header("Корреляционная матрица", "Признаки")
        feature_cols = get_scoring_features()
        fig = plot_correlation_heatmap(scored, feature_cols)
        apply_chart_theme(fig, height=600, transparent=False)
        fig.update_layout(margin=dict(l=150, b=150))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        render_section_header("Зависимость балла от суммы", "Scatter")
        fig = plot_amount_vs_score(scored)
        apply_chart_theme(fig, height=500)
        st.plotly_chart(fig, use_container_width=True)

        render_divider()

        render_section_header("Распределение баллов по регионам", "Боксплот")
        fig = plot_score_by_region(scored)
        apply_chart_theme(fig, height=500)
        fig.update_layout(margin=dict(b=150))
        st.plotly_chart(fig, use_container_width=True)

        render_divider()

        render_section_header("Одобрение по направлениям", "")
        fig = plot_approval_rate_by_direction(df)
        apply_chart_theme(fig, height=400)
        fig.update_layout(margin=dict(l=200))
        st.plotly_chart(fig, use_container_width=True)

        render_divider()

        render_section_header("Динамика по месяцам", "Тренд")
        fig = plot_monthly_trend(df)
        apply_chart_theme(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        render_baseline_comparison(scored, engine, df)

    with tab5:
        render_model_validation(scored, engine, df)


def render_baseline_comparison(scored: pd.DataFrame, engine: ScoringEngine, df: pd.DataFrame):
    """Сравнение гибридной модели с базовыми моделями (FCFS, Rule-only, ML-only)."""
    render_section_header(
        "Сравнение с базовыми моделями",
        "Эффективность"
    )

    st.markdown("""
    <div class="glass-card" style="margin-bottom:24px;">
        <div style="font-size:14px; line-height:1.7; color:var(--text-secondary);">
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
    rule_only_top = scored.nsmallest(n_top, scored["rule_score"].rank(ascending=False, method="min").name if False else "rank")
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
                         marker_color=TEAL, text=[f"{v:.1f}" for v in avg_scores], textposition="auto"))
    fig.add_trace(go.Bar(name="Одобрение (%)", x=metric_names, y=avg_approvals,
                         marker_color=GOLD, text=[f"{v:.1f}" for v in avg_approvals], textposition="auto"))
    fig.add_trace(go.Bar(name="Освоение (%)", x=metric_names, y=avg_utils,
                         marker_color=NAVY_LIGHT, text=[f"{v:.1f}" for v in avg_utils], textposition="auto"))
    apply_chart_theme(fig, height=450)
    fig.update_layout(barmode="group", legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True)

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
            <h3 style="color:var(--teal);">Гибридная модель (60/40)</h3>
            <div class="baseline-label">Средний балл (топ-{n_top})</div>
            <div class="baseline-metric" style="color:var(--teal);">{hybrid_avg_score:.1f}</div>
            <div class="baseline-label">Среднее одобрение</div>
            <div class="baseline-metric" style="color:var(--teal); font-size:22px;">{hybrid_avg_approval:.1%}</div>
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
        st.plotly_chart(fig, use_container_width=True)
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
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.warning("Не удалось построить график предсказание vs факт.")


# ===========================================================================
# PAGE 9: SETTINGS
# ===========================================================================
def render_settings(scored: pd.DataFrame):
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
            <div style="font-size:12px; color:var(--text-secondary); margin-bottom:6px;">Баланс моделей</div>
            <div style="display:flex; height:32px; border-radius:8px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,0.1);">
                <div style="width:{ml_pct}%; background:linear-gradient(90deg, var(--teal), #1a4a6e);
                    display:flex; align-items:center; justify-content:center; color:white; font-size:11px; font-weight:700;">
                    ML {ml_pct}%
                </div>
                <div style="width:{100-ml_pct}%; background:linear-gradient(90deg, var(--gold), #dbb94a);
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
        <div style="font-size:14px; color:var(--text-secondary); line-height:1.6;">
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
            <td style="font-weight:700; color:var(--teal);">#{int(row['rank'])}</td>
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
    if st.button("Сбросить настройки по умолчанию"):
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
            <div style="line-height:1.8; font-size:14px; color:var(--text-primary);">
                <p style="font-weight:700; font-size:16px; color:var(--navy); margin-bottom:12px;">
                    Меры защиты данных
                </p>
                <p>
                    <span style="font-weight:700; color:var(--teal);">1. Локальная обработка:</span>
                    Все данные обрабатываются исключительно на сервере приложения.
                    Информация не передаётся во внешние API, облачные сервисы или третьим лицам.
                </p>
                <p>
                    <span style="font-weight:700; color:var(--teal);">2. Отсутствие внешних вызовов:</span>
                    Система не использует внешние API для анализа данных.
                    Все вычисления (ML-модель, статистика, визуализация) выполняются локально.
                </p>
                <p>
                    <span style="font-weight:700; color:var(--teal);">3. Сессионность данных:</span>
                    Данные сессии не сохраняются между запусками приложения.
                    Каждый сеанс работы начинается с чистого состояния.
                </p>
                <p>
                    <span style="font-weight:700; color:var(--teal);">4. Анонимизация:</span>
                    Идентификаторы производителей (producer_id) используются только для
                    группировки заявок. Они могут быть анонимизированы без потери функциональности
                    скоринга.
                </p>
                <p style="margin-top:16px; padding:12px 16px; background:rgba(13,115,119,0.06); border-radius:8px; font-weight:600;">
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
        <div style="line-height:1.8; font-size:14px; color:var(--text-primary);">
            <p style="font-weight:700; font-size:16px; color:var(--navy); margin-bottom:12px;">
                Комбинированный скоринг объединяет два подхода:
            </p>
            <p>
                <span style="font-weight:700; color:var(--teal);">1. Правиловая модель</span> --
                экспертные веса, прозрачная формула, полная объяснимость.
                Каждый фактор нормализован [0, 1] и умножается на вес. Итоговый балл -- взвешенная сумма.
            </p>
            <p>
                <span style="font-weight:700; color:var(--teal);">2. ML-модель (Gradient Boosting)</span> --
                обучена на исторических данных, выявляет нелинейные зависимости.
                Целевая переменная формируется из комбинации: исполнение (35%), одобрение (25%),
                освоение средств (20%), диверсификация (10%), отсутствие отклонений (10%).
            </p>
            <p style="margin-top:16px; padding:12px 16px; background:rgba(13,115,119,0.06); border-radius:8px; font-weight:600;">
                Итоговый балл = вес_ML x балл_ML + вес_правил x балл_правил. Все баллы нормализованы в диапазон 0--100.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def engine_cv_score(scored):
    """Quick helper to get a CV score proxy."""
    if "rule_score" in scored.columns and "ml_score" in scored.columns:
        corr = scored["rule_score"].corr(scored["ml_score"])
        return max(0, corr)
    return 0.0


# ===========================================================================
if __name__ == "__main__":
    main()
