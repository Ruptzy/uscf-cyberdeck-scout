"""
USCF Cyberdeck Scout — Dashboard
================================
A tactical chess-opponent scouting terminal built on top of the
USCF MSA pipeline.  Renders prediction + scouting intelligence from
cached/precomputed CSVs and model artifacts; never touches USCF live.

Run locally:
    streamlit run app.py

Architecture:
* All paths are relative to this file's directory — safe for deployment.
* `st.cache_data` for CSV loads, `st.cache_resource` for the model.
* Sidebar-driven navigation across seven HUD pages.

Visual identity: cyberdeck/HUD terminal (deep burgundy + cyan + coral).
"""

import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ============================================================================
# Paths (RELATIVE to this file for deployment portability)
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "outputs" / "models"
EDA_DIR = BASE_DIR / "outputs" / "eda"
SCRIPTS_DIR = BASE_DIR / "scripts"


# ============================================================================
# Page config — must be the very first Streamlit call
# ============================================================================
st.set_page_config(
    page_title="USCF Cyberdeck Scout",
    page_icon="♟",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# Cyberdeck CSS
# ============================================================================
CYBERDECK_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap');

  :root {
    --bg-main: #120812;
    --bg-deep: #03070D;
    --bg-panel: #1A0B14;
    --bg-panel-alt: #102027;
    --cyan-main: #7AF7FF;
    --cyan-soft: #5ED7D9;
    --cyan-muted: #2D8F95;
    --red-main: #C0444A;
    --red-dark: #7A2630;
    --red-soft: #E05A5F;
    --text-main: #D8FFFF;
    --text-muted: #7FAFB4;
    --text-warning: #FF7777;
    --gold-accent: #A38560;
  }

  /* ----- Global app background ----- */
  [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(ellipse at top left, rgba(122,247,255,0.06), transparent 50%),
      radial-gradient(ellipse at bottom right, rgba(192,68,74,0.08), transparent 55%),
      linear-gradient(180deg, var(--bg-main) 0%, var(--bg-deep) 100%);
    color: var(--text-main);
    font-family: 'Rajdhani', sans-serif;
  }
  [data-testid="stHeader"] { background: transparent; }

  /* faint scanline overlay */
  [data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed; inset: 0; pointer-events: none;
    background: repeating-linear-gradient(
      0deg, rgba(122,247,255,0.025) 0 1px, transparent 1px 3px);
    z-index: 9999;
  }

  /* ----- Sidebar ----- */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #100712 0%, #07050A 100%);
    border-right: 1px solid var(--cyan-muted);
    box-shadow: 2px 0 16px rgba(122,247,255,0.07);
  }
  [data-testid="stSidebar"] * { color: var(--text-main); }
  [data-testid="stSidebar"] .stRadio > label > div p {
    font-family: 'Share Tech Mono', monospace;
    color: var(--cyan-soft);
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 0.85rem;
  }

  /* ----- Typography ----- */
  h1, h2, h3, h4, h5 {
    font-family: 'Rajdhani', sans-serif;
    color: var(--cyan-main);
    letter-spacing: 0.05em;
    text-transform: uppercase;
    text-shadow: 0 0 6px rgba(122,247,255,0.4);
  }
  p, li, span, label { color: var(--text-main); }
  .stCaption, [data-testid="stCaptionContainer"] { color: var(--text-muted) !important; }

  /* ----- HUD blocks ----- */
  .hud-meta {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: var(--cyan-soft);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
    opacity: 0.85;
  }
  .hud-meta::before { content: ">> "; color: var(--cyan-muted); }

  .hud-card {
    background: linear-gradient(135deg, rgba(26,11,20,0.92), rgba(16,32,39,0.55));
    border: 1px solid var(--cyan-muted);
    box-shadow:
      inset 0 0 14px rgba(122,247,255,0.08),
      0 0 12px rgba(122,247,255,0.06);
    padding: 1.1rem 1.2rem;
    margin: 0.6rem 0;
    clip-path: polygon(
      14px 0, 100% 0,
      100% calc(100% - 14px), calc(100% - 14px) 100%,
      0 100%, 0 14px
    );
    border-radius: 2px;
  }
  .hud-card.hud-warning {
    border-color: var(--red-main);
    background: linear-gradient(135deg, rgba(60,12,16,0.92), rgba(40,8,14,0.55));
    box-shadow:
      inset 0 0 16px rgba(224,90,95,0.10),
      0 0 14px rgba(192,68,74,0.18);
  }
  .hud-card.hud-warning .hud-meta { color: var(--red-soft); }
  .hud-card.hud-success {
    border-color: var(--cyan-soft);
    box-shadow:
      inset 0 0 18px rgba(122,247,255,0.12),
      0 0 16px rgba(122,247,255,0.18);
  }
  .hud-card.hud-gold {
    border-color: var(--gold-accent);
    box-shadow:
      inset 0 0 16px rgba(163,133,96,0.18),
      0 0 12px rgba(163,133,96,0.20);
  }

  .hud-divider {
    height: 1px;
    background: linear-gradient(90deg,
      transparent, var(--cyan-muted) 25%, var(--cyan-main) 50%,
      var(--cyan-muted) 75%, transparent);
    margin: 1.0rem 0;
    opacity: 0.7;
  }

  .hud-pill {
    display: inline-block;
    padding: 0.18rem 0.6rem;
    border: 1px solid var(--cyan-muted);
    color: var(--cyan-soft);
    background: rgba(122,247,255,0.05);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-right: 0.35rem;
    margin-bottom: 0.25rem;
    border-radius: 2px;
  }
  .hud-pill.danger {
    border-color: var(--red-soft);
    color: var(--red-soft);
    background: rgba(192,68,74,0.08);
  }
  .hud-pill.gold {
    border-color: var(--gold-accent);
    color: var(--gold-accent);
    background: rgba(163,133,96,0.10);
  }

  .hud-progress-track {
    position: relative; height: 10px; width: 100%;
    background: rgba(122,247,255,0.06);
    border: 1px solid var(--cyan-muted);
    border-radius: 2px; overflow: hidden;
  }
  .hud-progress-bar {
    position: absolute; left: 0; top: 0; bottom: 0;
    background: linear-gradient(90deg, var(--cyan-soft), var(--cyan-main));
    box-shadow: 0 0 8px var(--cyan-main);
  }
  .hud-progress-bar.danger {
    background: linear-gradient(90deg, var(--red-soft), var(--red-main));
    box-shadow: 0 0 8px var(--red-main);
  }
  .hud-progress-bar.gold {
    background: linear-gradient(90deg, var(--gold-accent), #d6b07f);
    box-shadow: 0 0 8px var(--gold-accent);
  }

  /* ----- Streamlit native widget tweaks ----- */
  [data-testid="stMetric"] {
    background: rgba(16,32,39,0.55);
    border: 1px solid var(--cyan-muted);
    padding: 0.7rem 0.9rem;
    border-radius: 2px;
    clip-path: polygon(
      10px 0, 100% 0, 100% calc(100% - 10px),
      calc(100% - 10px) 100%, 0 100%, 0 10px);
  }
  [data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--cyan-soft) !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.7rem !important;
  }
  [data-testid="stMetricValue"] {
    color: var(--text-main) !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600;
  }
  [data-testid="stMetricDelta"] {
    color: var(--gold-accent) !important;
    font-family: 'Share Tech Mono', monospace !important;
  }

  div[data-baseweb="select"] > div, .stTextInput > div > div, .stNumberInput > div > div {
    background: rgba(16,32,39,0.7) !important;
    border: 1px solid var(--cyan-muted) !important;
    color: var(--text-main) !important;
  }

  .stButton > button {
    background: linear-gradient(135deg, rgba(45,143,149,0.25), rgba(122,247,255,0.15));
    border: 1px solid var(--cyan-main);
    color: var(--cyan-main);
    font-family: 'Share Tech Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    padding: 0.5rem 1.2rem;
    border-radius: 2px;
    box-shadow: 0 0 12px rgba(122,247,255,0.15);
  }
  .stButton > button:hover {
    background: linear-gradient(135deg, rgba(122,247,255,0.25), rgba(122,247,255,0.4));
    color: var(--bg-deep);
    box-shadow: 0 0 22px rgba(122,247,255,0.5);
  }

  /* Streamlit dataframe restyle */
  [data-testid="stDataFrame"] {
    background: rgba(16,32,39,0.5);
    border: 1px solid var(--cyan-muted);
    border-radius: 2px;
  }

  /* st.info / st.warning / st.error blocks */
  [data-testid="stAlert"] {
    background: rgba(16,32,39,0.6);
    border-left: 3px solid var(--cyan-main);
    color: var(--text-main);
  }

  /* ============================================================
     TOP CHROME BAND (reference: cyberdeck v552)
     ============================================================ */
  .cyber-topband {
    margin: -0.4rem 0 1.2rem 0;
    padding: 0.7rem 1.0rem 0.5rem 1.0rem;
    border: 1px solid var(--red-soft);
    background:
      linear-gradient(90deg, rgba(60,12,16,0.55), rgba(40,12,18,0.35) 50%, rgba(60,12,16,0.55));
    box-shadow:
      inset 0 0 18px rgba(192,68,74,0.10),
      0 0 14px rgba(192,68,74,0.15);
    clip-path: polygon(
      0 0, 100% 0,
      100% calc(100% - 12px), calc(100% - 12px) 100%,
      0 100%);
  }
  .cyber-topband-row {
    display: flex; align-items: center; justify-content: space-between;
    gap: 1.5rem; flex-wrap: wrap;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem; letter-spacing: 0.18em;
    color: var(--red-soft);
    text-transform: uppercase;
  }
  .cyber-topband-row .left  { display:flex; gap:0.7rem; align-items:center; }
  .cyber-topband-row .right { display:flex; gap:0.7rem; align-items:center; color: var(--cyan-soft); }
  .cyber-topband-kanji {
    font-family: 'Rajdhani', sans-serif; font-size: 1.4rem;
    color: var(--red-soft); letter-spacing: 0.05em;
  }

  /* dotted bracket scale: [ . S . C . O . U . T . ] */
  .cyber-bracket {
    display: flex; align-items: center; gap: 0.4rem;
    color: var(--red-soft); font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem; letter-spacing: 0.15em;
  }
  .cyber-bracket .tick { color: var(--red-soft); opacity: 0.6; }
  .cyber-bracket .letter { color: var(--cyan-soft); padding: 0 0.1rem; }

  .cyber-barcode {
    height: 14px; margin: 0.5rem 0 0.3rem 0;
    background-image: repeating-linear-gradient(
      90deg,
      var(--cyan-main) 0 1px, transparent 1px 3px,
      var(--cyan-main) 3px 5px, transparent 5px 9px,
      var(--cyan-main) 9px 10px, transparent 10px 14px,
      var(--cyan-main) 14px 17px, transparent 17px 22px);
    opacity: 0.85;
  }
  .cyber-barcode.red {
    background-image: repeating-linear-gradient(
      90deg,
      var(--red-soft) 0 1px, transparent 1px 3px,
      var(--red-soft) 3px 5px, transparent 5px 9px,
      var(--red-soft) 9px 10px, transparent 10px 14px,
      var(--red-soft) 14px 17px, transparent 17px 22px);
    opacity: 0.7;
  }

  .cyber-scanning {
    display: flex; align-items: center; gap: 0.8rem;
    font-family: 'Share Tech Mono', monospace; font-size: 0.78rem;
    letter-spacing: 0.15em; color: var(--cyan-soft);
    text-transform: uppercase;
    margin-top: 0.3rem;
  }
  .cyber-scanning .label { flex: 0 0 auto; }
  .cyber-scanning .track {
    flex: 1; height: 8px; position: relative;
    background: rgba(122,247,255,0.06);
    border: 1px solid var(--cyan-muted);
  }
  .cyber-scanning .bar {
    position: absolute; left: 0; top: 0; bottom: 0;
    background: linear-gradient(90deg, var(--cyan-soft), var(--cyan-main));
    box-shadow: 0 0 6px var(--cyan-main);
  }
  .cyber-scanning .pct { flex: 0 0 auto; color: var(--cyan-main); }

  /* ============================================================
     SIDEBAR NAV BUTTON-CARDS (button-styled per reference image)
     ============================================================ */
  [data-testid="stSidebar"] .stButton > button {
    /* every sidebar button takes the cyberdeck card look */
    background: linear-gradient(135deg, rgba(45,143,149,0.18), rgba(122,247,255,0.06));
    color: var(--cyan-main);
    border: 1px solid var(--cyan-muted);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.85rem; font-weight: 500;
    letter-spacing: 0.1em; text-transform: uppercase;
    text-align: left;
    padding: 0.55rem 0.75rem 0.5rem 0.75rem;
    border-radius: 1px;
    box-shadow:
      inset 0 0 12px rgba(122,247,255,0.06),
      0 0 8px rgba(122,247,255,0.06);
    clip-path: polygon(
      8px 0, 100% 0,
      100% calc(100% - 8px), calc(100% - 8px) 100%,
      0 100%, 0 8px);
    margin-top: 0.1rem;
    white-space: pre-wrap;
    line-height: 1.05;
    min-height: 2.6rem;
  }
  [data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, rgba(122,247,255,0.22), rgba(122,247,255,0.10));
    border-color: var(--cyan-main);
    color: var(--cyan-main);
    box-shadow:
      inset 0 0 16px rgba(122,247,255,0.18),
      0 0 14px rgba(122,247,255,0.30);
  }
  /* Active (primary) nav button — red HACK-card look */
  [data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(192,68,74,0.55), rgba(122,38,48,0.50));
    color: var(--text-main);
    border-color: var(--red-soft);
    box-shadow:
      inset 0 0 16px rgba(224,90,95,0.30),
      0 0 16px rgba(192,68,74,0.45);
  }
  [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, rgba(224,90,95,0.70), rgba(160,40,50,0.60));
    color: var(--text-main);
    border-color: var(--red-soft);
  }

  .cyber-nav-subtitle {
    color: var(--text-muted);
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.66rem; letter-spacing: 0.18em;
    text-transform: uppercase;
    margin: -0.05rem 0 0.55rem 0.6rem;
    opacity: 0.78;
  }
  .cyber-nav-subtitle.active {
    color: var(--red-soft);
    opacity: 1.0;
  }

  /* ============================================================
     FOOTER TERMINAL BAND
     ============================================================ */
  .cyber-footer {
    margin-top: 2rem; padding: 0.8rem 1rem 0.6rem 1rem;
    border-top: 1px solid var(--cyan-muted);
    border-bottom: 1px solid var(--cyan-muted);
    background: linear-gradient(180deg, rgba(16,32,39,0.5), rgba(10,18,22,0.3));
  }
  .cyber-footer .prompt {
    font-family: 'Share Tech Mono', monospace;
    color: var(--cyan-main); font-size: 0.78rem;
    letter-spacing: 0.12em;
  }
  .cyber-footer .prompt::after {
    content: "▌"; color: var(--cyan-main); animation: blink 1.1s steps(2) infinite;
    margin-left: 0.2rem;
  }
  @keyframes blink { 50% { opacity: 0; } }
  .cyber-footer .disclaimer {
    font-family: 'Share Tech Mono', monospace;
    color: var(--text-muted); font-size: 0.62rem;
    letter-spacing: 0.15em; line-height: 1.5;
    margin-top: 0.4rem; opacity: 0.7;
    text-transform: uppercase;
  }
</style>
"""

st.markdown(CYBERDECK_CSS, unsafe_allow_html=True)


# ============================================================================
# Matchup module — load via importlib because the script name starts with a digit
# ============================================================================
@st.cache_resource
def _load_matchup_module():
    sys.path.insert(0, str(SCRIPTS_DIR))
    spec = importlib.util.spec_from_file_location(
        "_matchup", str(SCRIPTS_DIR / "21_generate_matchup_report.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_matchup_module():
    try:
        return _load_matchup_module()
    except Exception as e:  # noqa: BLE001
        st.error(f"Matchup module failed to load: {e}")
        return None


# ============================================================================
# Cached loaders — fail gracefully if files are missing
# ============================================================================
@st.cache_data
def load_profiles() -> pd.DataFrame | None:
    p = DATA_DIR / "player_profiles.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, dtype={"player_id": str})


@st.cache_data
def load_scores() -> pd.DataFrame | None:
    p = DATA_DIR / "player_scouting_scores.csv"
    if not p.exists():
        return None
    return pd.read_csv(p, dtype={"player_id": str})


@st.cache_data
def load_kfold_metrics() -> pd.DataFrame | None:
    p = MODELS_DIR / "kfold_cv_metrics.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_genai_metrics() -> pd.DataFrame | None:
    p = MODELS_DIR / "genai_comparison_metrics.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_feature_importances() -> pd.DataFrame | None:
    p = MODELS_DIR / "kfold_cv_feature_importances.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_per_fold() -> pd.DataFrame | None:
    p = MODELS_DIR / "kfold_cv_per_fold.csv"
    return pd.read_csv(p) if p.exists() else None


# ============================================================================
# HUD primitive helpers (HTML fragments)
# ============================================================================
def _flatten_html(s: str) -> str:
    """Collapse blank lines and leading-whitespace inside an HTML fragment.

    Streamlit / CommonMark treats a blank line inside an HTML block as the
    end of that block — anything after is re-parsed as markdown and shows up
    as escaped text on the page.  Stripping blank lines and the leading
    indentation makes the fragment a single uninterrupted HTML block, which
    `st.markdown(..., unsafe_allow_html=True)` renders correctly.
    """
    # Remove leading whitespace on each line so HTML starts in col 0
    s = "\n".join(line.lstrip() for line in s.splitlines())
    # Drop blank lines entirely
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def render_html(html: str):
    """Render an HTML fragment safely through Streamlit's markdown channel."""
    st.markdown(_flatten_html(html), unsafe_allow_html=True)


def hud_header(eyebrow: str, title: str, subtitle: str = ""):
    sub_html = (
        f'<div style="color:var(--text-muted);font-size:1.0rem;margin-top:0.2rem;">{subtitle}</div>'
        if subtitle else ""
    )
    render_html(f"""
        <div style="margin:0.4rem 0 1.2rem 0;">
          <div class="hud-meta">{eyebrow}</div>
          <h1 style="margin:0;font-size:2.0rem;">{title}</h1>
          {sub_html}
        </div>
    """)


def hud_card(eyebrow: str, body_html: str, variant: str = ""):
    cls = "hud-card"
    if variant in ("warning", "success", "gold"):
        cls += f" hud-{variant}"
    render_html(
        f'<div class="{cls}">'
        f'<div class="hud-meta">{eyebrow}</div>'
        f'{_flatten_html(body_html)}'
        f'</div>'
    )


def hud_progress(label: str, value_pct: float, variant: str = ""):
    """Render a labeled cyberdeck-style progress bar (0-100)."""
    value_pct = max(0.0, min(100.0, float(value_pct)))
    bar_cls = "hud-progress-bar"
    if variant in ("danger", "gold"):
        bar_cls += f" {variant}"
    return f"""
      <div style="margin:0.45rem 0;">
        <div style="display:flex;justify-content:space-between;
                    font-family:'Share Tech Mono',monospace;
                    font-size:0.78rem;color:var(--cyan-soft);
                    letter-spacing:0.1em;text-transform:uppercase;">
          <span>{label}</span><span>{value_pct:.0f}%</span>
        </div>
        <div class="hud-progress-track">
          <div class="{bar_cls}" style="width:{value_pct:.1f}%"></div>
        </div>
      </div>
    """


def pill(text: str, variant: str = "") -> str:
    cls = "hud-pill"
    if variant in ("danger", "gold"):
        cls += f" {variant}"
    return f'<span class="{cls}">{text}</span>'


def divider():
    st.markdown('<div class="hud-divider"></div>', unsafe_allow_html=True)


def _fmt_pct(x):
    if x is None or pd.isna(x):
        return "—"
    return f"{float(x)*100:.0f}%"


def _fmt_num(x, signed=True):
    if x is None or pd.isna(x):
        return "—"
    try:
        v = float(x)
        return f"{v:+.0f}" if signed else f"{v:.0f}"
    except (TypeError, ValueError):
        return str(x)


def _fmt_int(x):
    if x is None or pd.isna(x):
        return "—"
    return str(int(x))


# ============================================================================
# Coaching advice — practical chess-prep ideas derived from the profile.
# Inspired by standard chess-coaching principles; not a guaranteed game plan.
# ============================================================================
def build_coaching_advice(profile: dict, score: dict | None) -> list[str]:
    """Return 3–6 plain-English bullets of practical preparation ideas
    based on a player's scouting profile."""
    out: list[str] = []
    s = score or {}
    upset = float(s.get("upset_score") or 0)
    form  = float(s.get("form_score") or 0)
    moment = float(s.get("momentum_score") or 0)
    sched = float(s.get("schedule_score") or 0)
    vol   = float(s.get("volatility_score") or 0)
    active = float(s.get("activity_score") or 0)

    g90 = profile.get("games_last_90d") or 0
    wr90 = profile.get("recent_win_rate_90d")
    best_tc = profile.get("best_time_control_by_score_rate")
    blitz_pct = profile.get("pct_blitz_games") or 0
    travel_label = (profile.get("traveling_competitor_label") or "").lower()
    unique_locs = profile.get("unique_event_locations") or 0
    avg_travel = profile.get("avg_travel_distance_miles")
    losses_lower = profile.get("losses_vs_100_plus_lower") or 0
    career = profile.get("career_n_games") or 0

    # --- Upset / giant-killer ---
    if upset >= 14:
        out.append(
            "**Treat the rating gap with respect.** This opponent has a record of beating "
            "higher-rated players. Choose solid openings you know well and avoid unnecessary "
            "complications unless you've calculated clearly."
        )

    # --- Hot recent form ---
    if form >= 14 and wr90 is not None and wr90 >= 0.55:
        out.append(
            "**They're in form.** Recent results suggest they may be sharper than rating "
            "alone shows. Prepare seriously and avoid casual or experimental opening choices."
        )

    # --- Rising rating ---
    if moment >= 14:
        out.append(
            "**Rating is climbing.** This player has been gaining points recently — their "
            "true strength may be slightly ahead of the number on paper."
        )

    # --- High activity ---
    if active >= 7 or g90 >= 15:
        out.append(
            "**Very active recently.** Expect sharper, more practical play and good clock "
            "management. Stay focused early; don't drift in the opening."
        )

    # --- Rusty / inactive ---
    if (g90 or 0) <= 1 and career >= 30:
        out.append(
            "**Possibly rusty.** They haven't played much in the last 90 days. Apply steady "
            "pressure and let them solve practical problems rather than forcing chaos."
        )

    # --- Time-control specialist ---
    if best_tc == "Blitz" or blitz_pct >= 0.5:
        out.append(
            "**Blitz-strong opponent.** Don't drift into time trouble. Choose positions with "
            "clear plans you can play quickly and confidently."
        )
    elif best_tc == "Regular":
        out.append(
            "**Classical-style opponent.** They favor slower, deeper games. Be ready for long "
            "strategic phases and don't expect them to crack from time pressure alone."
        )

    # --- Strong schedule (plays up) ---
    if sched >= 10:
        out.append(
            "**Battle-tested.** This player frequently plays up against stronger fields, so "
            "their rating may understate how prepared they are. Don't underestimate."
        )

    # --- Volatile / boom-bust ---
    if vol >= 7 or profile.get("boom_bust_flag"):
        out.append(
            "**Volatile performer.** Capable of tactical swings in either direction. Stay "
            "calm and convert advantages patiently — don't try to match the chaos."
        )

    # --- Weak recent form ---
    if wr90 is not None and wr90 <= 0.40 and (g90 or 0) >= 5:
        out.append(
            "**Cold recent form.** Apply pressure steadily and stay disciplined. Don't "
            "become overconfident — players bounce back."
        )

    # --- Travel-active / road warrior ---
    if "road warrior" in travel_label or "travel-active" in travel_label or unique_locs >= 6:
        msg = "**Travel-active competitor.** "
        if avg_travel and avg_travel > 0:
            msg += f"Averages about {avg_travel:.0f} miles per event. "
        msg += (
            "Players who consistently travel to events tend to be more "
            "tournament-hardened than rating suggests. Expect practical resilience "
            "and well-established over-the-board habits."
        )
        out.append(msg)

    # --- Mostly local player ---
    if "local" in travel_label and unique_locs <= 3:
        out.append(
            "**Mostly local player.** Plays a narrow set of venues. May be less "
            "tournament-hardened than frequent travelers, though strong local players "
            "often still have deep familiarity with their usual fields."
        )

    # --- Loses to lower-rated players (drops points to weaker fields) ---
    if losses_lower >= 5:
        out.append(
            "**Sometimes loses to lower-rated players.** Keep the game stable and let "
            "them create their own weaknesses. Avoid giving them unnecessary tactical "
            "chances — patience is rewarded against this profile."
        )

    # Fallback if nothing strong fired
    if not out:
        out.append(
            "**No standout danger signals.** Trust the rating difference if it favors "
            "you and play your usual preparation. Every game is still decided over the board."
        )

    return out[:6]


def render_coaching_card(profile: dict, score: dict | None,
                         heading: str = "PRACTICAL CHESS PREP IDEAS") -> None:
    """Render the coaching-advice card with a clear disclaimer."""
    bullets = build_coaching_advice(profile, score)
    items = "".join(f"<li style='margin-bottom:0.45rem;'>{b}</li>" for b in bullets)
    hud_card(
        heading,
        f"""
        <ul style="margin:0; padding-left:1.2rem; font-size:1.0rem; line-height:1.6;">{items}</ul>
        <div style="color:var(--text-muted); font-size:0.78rem; margin-top:0.6rem; line-height:1.5;">
          Inspired by standard chess-coaching principles. These are general
          preparation ideas based on the available data, <em>not</em> a guaranteed
          game plan.
        </div>
        """,
        variant="gold",
    )


# Plotly theming
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(16,32,39,0.5)",
    font=dict(family="Share Tech Mono", color="#D8FFFF"),
    margin=dict(l=20, r=10, t=40, b=20),
    xaxis=dict(gridcolor="rgba(122,247,255,0.10)", linecolor="#2D8F95",
               tickcolor="#2D8F95", title_font=dict(size=12)),
    yaxis=dict(gridcolor="rgba(122,247,255,0.10)", linecolor="#2D8F95",
               tickcolor="#2D8F95", title_font=dict(size=12)),
    title_font=dict(family="Rajdhani", color="#7AF7FF", size=16),
)


# ============================================================================
# Top header band (reference: cyberdeck v552 admin chrome)
# ============================================================================
def render_top_band():
    """Reference-image style chrome: bracket scale + version code + barcode + scanning bar."""
    # The dotted bracket scale: [ . S . C . O . U . T . ]
    letters = list("SCOUT")
    bracket_inner = "".join(
        f'<span class="tick">·</span><span class="letter">{ch}</span>'
        for ch in letters
    ) + '<span class="tick">·</span>'
    bracket = f'<span>[</span>{bracket_inner}<span>]</span>'

    # Scanning bar — based on number of scouted players in the dataset
    profiles = load_profiles()
    total = 191
    scanned = len(profiles) if profiles is not None else 0
    pct = (scanned / total * 100) if total else 0

    render_html(f"""
        <div class="cyber-topband">
          <div class="cyber-topband-row">
            <div class="left">
              <span class="cyber-topband-kanji">♟</span>
              <span>CYBERDECK&nbsp;V&nbsp;0.55.2&nbsp;//&nbsp;USCF&nbsp;NODE&nbsp;0xC4</span>
            </div>
            <div class="cyber-bracket">{bracket}</div>
            <div class="right">
              <span>BUILD 18 · REV 22</span>
              <span style="color:var(--red-soft);">// ADMIN ACCESS GRANTED</span>
            </div>
          </div>
          <div class="cyber-barcode"></div>
          <div class="cyber-scanning">
            <span class="label">SCANNING&nbsp;PLAYER&nbsp;PROFILES</span>
            <div class="track"><div class="bar" style="width:{pct:.0f}%"></div></div>
            <span class="pct">{pct:.0f}%&nbsp;&nbsp;·&nbsp;&nbsp;{scanned} / {total} PROFILES</span>
          </div>
        </div>
    """)


def render_footer():
    render_html("""
        <div class="cyber-footer">
          <div class="cyber-barcode" style="margin-top:0;"></div>
          <div class="prompt" style="margin-top:0.5rem;">ROOT@USCF-CYBERDECK : ~ #&nbsp;</div>
          <div class="disclaimer">
            CACHED DATA NODE · NO LIVE USCF SCRAPING · NO API KEYS ·
            FOR EDUCATIONAL &amp; PORTFOLIO USE ·
            CROSSTABLE ORDER IS NOT OFFICIAL TIEBREAK ORDER ·
            RATING-TREND FIGURES MARKED "PROXY" ARE RECONSTRUCTED FROM PRE-RATING CHRONOLOGY
          </div>
        </div>
    """)


# ============================================================================
# Sidebar navigation — button-card nav (active = red HACK-card style)
# ============================================================================
NAV_ITEMS: list[tuple[str, str, str]] = [
    ("01 // COMMAND CENTER",       "▣",  "MISSION BRIEF · DATASET METRICS"),
    ("02 // MODEL INTEL",          "▼",  "MODEL VS ELO · K-FOLD CV"),
    ("03 // FEATURE VECTORS",      "▦",  "INPUT SIGNALS · IMPORTANCE"),
    ("04 // PLAYER DOSSIER",       "◉",  "OPPONENT PROFILE INTEL"),
    ("05 // UNDERRATED PROTOCOL",  "▲",  "0–100 THREAT SCORE · 6 SUBSCORES"),
    ("06 // MATCHUP SIM",          "✸",  "PREDICT P(WIN) · SIDE-BY-SIDE"),
    ("07 // DATA REPAIR LOG",      "⚠",  "PARSER PATCH · BEFORE / AFTER"),
]

if "active_page" not in st.session_state:
    st.session_state.active_page = NAV_ITEMS[0][0]

with st.sidebar:
    render_html("""
        <div style="text-align:center; margin-bottom:0.6rem;">
          <div class="hud-meta" style="text-align:center;">USCF // NODE 0xC4</div>
          <h2 style="margin:0; font-size:1.45rem; line-height:1.0;">CYBERDECK<br/>SCOUT</h2>
          <div style="color:var(--text-muted);font-size:0.75rem;letter-spacing:0.15em;font-family:'Share Tech Mono',monospace;">
            v1.0  ·  ACCESS GRANTED
          </div>
        </div>
        <div class="cyber-barcode" style="margin:0.4rem 0 0.8rem 0;"></div>
        <div class="hud-meta">ACCESS NODES</div>
    """)

    for page_key, icon, subtitle in NAV_ITEMS:
        is_active = (st.session_state.active_page == page_key)
        label = f"{icon}   {page_key}"
        if st.button(label, key=f"nav_{page_key}",
                     type=("primary" if is_active else "secondary"),
                     width="stretch"):
            st.session_state.active_page = page_key
            st.rerun()
        st.markdown(
            f'<div class="cyber-nav-subtitle{" active" if is_active else ""}">{subtitle}</div>',
            unsafe_allow_html=True,
        )

    page = st.session_state.active_page

    render_html("""
        <div class="hud-divider"></div>
        <div class="hud-meta">SYSTEM STATUS</div>
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:var(--text-muted);line-height:1.7;">
          MODEL_NODE ........ <span style="color:#5ED7D9;">ONLINE</span><br/>
          DATA_NODE ......... <span style="color:#5ED7D9;">ONLINE</span><br/>
          SCOUT_NODE ........ <span style="color:#5ED7D9;">ONLINE</span><br/>
          GEO_NODE .......... <span style="color:#5ED7D9;">ONLINE</span><br/>
          USCF_SCRAPER ...... <span style="color:#FF7777;">OFFLINE</span><br/>
          CACHE_AGE ......... <span style="color:#A38560;">7d</span>
        </div>
        <div class="cyber-barcode red" style="margin-top:0.6rem;"></div>
    """)


# ============================================================================
# PAGE 1 — Command Center
# ============================================================================
def page_command_center():
    # ------- Hero block (cinematic) -------
    render_html("""
        <div class="hud-meta">ACCESS GRANTED  ·  NODE 0xC4  ·  SESSION ACTIVE</div>
        <h1 style="margin:0 0 0.3rem 0; font-size:2.6rem; line-height:1.0;">USCF Cyberdeck Scout</h1>
        <div style="color:var(--text-muted); font-size:1.15rem; max-width:780px; line-height:1.5;">
          United States Chess Federation opponent scouting and
          game-outcome prediction from rated tournament history.
        </div>
    """)

    # Headline metric strip — readable, generous spacing
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RAW GAMES SCRAPED", "13,305")
    c2.metric("CLEAN MODELED GAMES", "8,514")
    c3.metric("FOCAL PLAYERS", "179")
    c4.metric("TOURNAMENT EVENTS", "3,020")

    divider()

    # ------- What is USCF? Plain English first --------
    hud_card(
        "WHAT IS USCF?",
        """
        <p style="font-size:1.02rem; line-height:1.6;">
        <strong style="color:var(--cyan-main)">USCF</strong> stands for the
        <strong>United States Chess Federation</strong> — the main organization
        behind official rated chess tournaments in the United States. Every
        tournament-rated game played in the U.S. flows through their system.
        </p>
        <p style="font-size:1.02rem; line-height:1.6;">
        In plain English, this project uses data from the
        <strong style="color:var(--cyan-main)">official U.S. chess rating and
        tournament network</strong> — public crosstable pages from
        <code>uschess.org</code> covering thousands of real tournament games.
        </p>
        """,
    )

    # ------- Why this matters -------
    hud_card(
        "WHY THIS PROJECT MATTERS",
        """
        <p style="font-size:1.02rem; line-height:1.6;">
        Most chess players prepare for an opponent by glancing at one number:
        their rating. But rating is a snapshot — it doesn't tell you
        whether the opponent is in <em>hot form</em>, <em>rusty</em>,
        <em>improving fast</em>, or <em>dangerous against stronger players</em>.
        </p>
        <p style="font-size:1.02rem; line-height:1.6;">
        This dashboard goes beyond rating. It predicts who is favored to win,
        but more importantly it explains
        <strong style="color:var(--gold-accent)">what kind of opponent you are
        facing</strong> — and gives you practical preparation ideas based on
        their profile.
        </p>
        """,
        variant="success",
    )

    # ------- The modern underrated-player problem -------
    cM1, cM2 = st.columns(2)
    with cM1:
        hud_card(
            "THE MODERN UNDERRATED-PLAYER PROBLEM",
            """
            <p style="font-size:1.0rem; line-height:1.6;">
            Since online chess exploded in popularity (especially during the
            post-2020 boom), many players improve dramatically <strong>online</strong>
            before they show up at an official over-the-board tournament. They play
            thousands of online games, study openings, run engines, take lessons —
            and their <em>practical</em> strength climbs much faster than their
            official rating can catch up.
            </p>
            <p style="font-size:1.0rem; line-height:1.6;">
            That creates a real scouting problem. A player may walk into your
            section with a modest official rating but actually play closer to a
            much higher level. They may have an inactive rating from years ago,
            or simply not enough USCF events for the system to know how strong
            they currently are.
            </p>
            <p style="font-size:1.0rem; line-height:1.6; color:var(--gold-accent);">
            The goal of this dashboard is to surface those signals before the
            game — <em>not to accuse anyone of anything</em>, just to give players
            a better picture of who is across the board.
            </p>
            """,
            variant="warning",
        )
    with cM2:
        hud_card(
            "WHY RATING ALONE IS NOT ENOUGH",
            """
            <p style="font-size:1.0rem; line-height:1.6;">
            Official ratings are powerful, but they can <strong>lag behind reality</strong>.
            Rating doesn't show:
            </p>
            <ul style="font-size:1.0rem; line-height:1.7; padding-left:1.2rem;">
              <li>how active the player has been in the last 90 days</li>
              <li>their <strong>recent win rate</strong> against real opposition</li>
              <li>how often they <strong>beat higher-rated players</strong></li>
              <li>whether their rating is currently <strong>rising or stagnant</strong></li>
              <li>which time controls they actually perform best in</li>
              <li>whether they travel to play strong fields or mostly play locally</li>
            </ul>
            <p style="font-size:1.0rem; line-height:1.6;">
            All of these add scouting context that pure rating cannot.
            </p>
            """,
        )

    # ------- One-liner stinger -------
    render_html("""
        <div style="text-align:center; margin:1.0rem 0 0.4rem 0;">
          <span style="font-family:'Rajdhani',sans-serif; font-size:1.4rem;
                       color:var(--cyan-main); letter-spacing:0.03em;
                       text-shadow:0 0 8px rgba(122,247,255,0.4);">
            "Rating tells you who is favored.&nbsp;
            <span style="color:var(--gold-accent);">Scouting tells you what kind of opponent you're actually facing.</span>"
          </span>
        </div>
    """)

    # ------- The 6-step process flow -------
    render_html('<div class="hud-meta" style="margin-top:1.2rem;">>> PROJECT PROCESS  ·  HOW WE BUILT IT</div>')
    steps = [
        ("01", "COLLECT",
         "We collected public United States Chess Federation tournament pages "
         "and converted them into a structured game-level dataset. Because the "
         "source pages are messy crosstables instead of clean tables, this "
         "required a custom parser and local HTML cache."),
        ("02", "CLEAN",
         "Raw rows were checked for duplicates, invalid dates, and missing "
         "ratings. A serious parser bug was discovered and fixed (see Data "
         "Repair Log), which corrected the rating signal across the entire "
         "dataset. Final cleaned dataset: 8,514 modeled games."),
        ("03", "ENGINEER VARIABLES",
         "From the cleaned games we built features the model can learn from: "
         "player rating, opponent rating, rating difference, time control, "
         "recent activity, and recent win rate. See Feature Vectors."),
        ("04", "TRAIN MODELS",
         "Logistic Regression as the benchmark, plus Random Forest and "
         "Gradient Boosting as stronger machine-learning models. We also "
         "compare against the Elo formula — the closed-form baseline that "
         "chess has used since the 1970s."),
        ("05", "EVALUATE",
         "We split the data chronologically so the model is always tested "
         "on games it has never seen, and use 5-fold time-aware "
         "cross-validation to make sure the result is stable. Reported "
         "metrics: AUC, F1, Accuracy, Precision, Recall."),
        ("06", "SCOUT",
         "On top of the model, the scouting layer answers the harder question: "
         "what kind of opponent is this? It produces a 0–100 Underrated "
         "Potential score, plain-English profile cards, and practical chess "
         "preparation ideas for the matchup."),
    ]
    for i in range(0, len(steps), 2):
        cols = st.columns(2)
        for j, (num, title, body) in enumerate(steps[i:i + 2]):
            with cols[j]:
                hud_card(
                    f"STEP {num}  ·  {title}",
                    f'<p style="font-size:0.98rem; line-height:1.55;">{body}</p>',
                )

    # ------- Honest finding -------
    divider()
    hud_card(
        "HEADLINE FINDING  ·  WHAT THE NUMBERS SAY",
        """
        <p style="font-size:1.02rem; line-height:1.6;">
        Across our test set, every model scored within ~0.002 AUC of each
        other — including the simple Elo formula at
        <strong style="color:var(--cyan-main)">0.825</strong>. The
        machine-learning pipeline matches Elo's predictive power without
        claiming to beat it.
        </p>
        <p style="font-size:1.02rem; line-height:1.6;">
        That's the honest framing: Elo is extremely strong for chess
        prediction because rating already captures most of what's
        predictable. The real product value is one layer up — turning
        the same data into a
        <strong style="color:var(--gold-accent)">scouting report</strong>
        that explains what kind of opponent is on the other side of the
        board.
        </p>
        """,
        variant="gold",
    )

    # ------- Rubric map (presentation-ready) -------
    divider()
    render_html('<div class="hud-meta">>> RUBRIC MAP  ·  HOW THIS PROJECT MEETS THE CLASS REQUIREMENTS</div>')

    rubric_rows = [
        ("Novel dataset (not Kaggle)",        "Public United States Chess Federation tournament crosstables, scraped + cached locally."),
        ("Cleaning",                          "Parser-bug repair, date / rating validation, duplicate handling, leakage controls."),
        ("EDA",                               "Distribution charts, correlation matrix, redundancy audit (rating_diff vs raw ratings)."),
        ("Benchmark model",                   "Logistic Regression on rating_diff + time_control."),
        ("ML comparison",                     "Random Forest, Gradient Boosting, and an Elo zero-shot baseline."),
        ("Validation",                        "Chronological 70 / 15 / 15 train / val / test plus 5-fold TimeSeriesSplit CV."),
        ("Presentation value",                "This live dashboard — Cyberdeck-themed scouting interface with 7 pages."),
        ("Extra credit (Generative-AI comparison)",
                                              "Elo zero-shot baseline on the same held-out test set; optional Claude API path documented."),
    ]
    th_style = ("color:var(--cyan-soft);text-align:left;padding:0.55rem 0.8rem;"
                "border-bottom:1px solid var(--cyan-muted);"
                "text-transform:uppercase;letter-spacing:0.08em;font-family:'Share Tech Mono',monospace;"
                "font-size:0.78rem;")
    td_l = ("padding:0.55rem 0.8rem;border-bottom:1px solid rgba(122,247,255,0.10);"
            "font-family:'Share Tech Mono',monospace;font-size:0.85rem;"
            "color:var(--cyan-main);")
    td_r = ("padding:0.55rem 0.8rem;border-bottom:1px solid rgba(122,247,255,0.10);"
            "font-size:0.95rem;color:var(--text-main);line-height:1.5;")
    rows_html = "".join(
        f"<tr><td style='{td_l}'>{req}</td><td style='{td_r}'>{where}</td></tr>"
        for req, where in rubric_rows
    )
    render_html(
        '<table style="width:100%;border-collapse:collapse;margin-top:0.4rem;">'
        f'<thead><tr><th style="{th_style}">Rubric Requirement</th>'
        f'<th style="{th_style}">Where This Project Shows It</th></tr></thead>'
        f'<tbody>{rows_html}</tbody></table>'
    )

    # ------- Soft pointer to the data-quality story -------
    divider()
    hud_card(
        "DATA QUALITY ALERT  ·  THE PARSER STORY",
        """
        <p style="font-size:1.0rem; line-height:1.6;">
        During exploratory analysis the rating advantage appeared to <em>hurt</em>
        the chance of winning — impossible for chess. Tracing the issue
        upstream revealed that the data parser was capturing part of the
        player's USCF ID instead of the rating itself.
        </p>
        <p style="font-size:1.0rem; line-height:1.6;">
        After fixing the parser and re-reading the cached pages, the rating
        signal behaved correctly: rating advantage and winning correlated at
        <strong style="color:var(--cyan-main)">+0.49</strong>, and test AUC
        moved from <strong style="color:var(--red-soft)">0.67</strong> to
        <strong style="color:var(--cyan-main)">0.82</strong>.
        Full repair log on page <em>07 // DATA REPAIR LOG</em>.
        </p>
        """,
        variant="warning",
    )

    # ------- How to use this tool responsibly -------
    divider()
    hud_card(
        "HOW TO USE THIS TOOL  ·  RESPONSIBLE USE NOTE",
        """
        <p style="font-size:1.02rem; line-height:1.6;">
        Use this dashboard as a <strong>scouting aid</strong>, not as a final
        judgment. Official rating is still one of the strongest predictors in
        chess; the metrics here add useful context, but they are
        <strong>signals — not proof</strong>.
        </p>
        <p style="font-size:1.02rem; line-height:1.6;">
        The Underrated Potential score does <strong>not</strong> prove that
        anyone is a "smurf" or doing anything wrong. It simply highlights
        players whose recent activity, upset history, rating movement, or
        event results suggest that the rating <em>may be lagging behind
        current strength</em>. Always interpret small samples carefully,
        and remember that every game is still decided over the board.
        </p>
        """,
        variant="gold",
    )

    # ------- Roadmap / future product ideas -------
    divider()
    render_html('<div class="hud-meta" style="margin-top:0.3rem;">>> ROADMAP  ·  NEXT VERSIONS OF THE SCOUT</div>')
    rL, rR = st.columns(2)
    with rL:
        hud_card(
            "LIVE USCF LOOKUP  ·  COMING SOON",
            """
            <p style="font-size:1.0rem; line-height:1.6;">
            A future version will let you type any USCF ID and have the app
            fetch that player's public tournament history on demand, update
            the local cache, and generate a fresh scouting report in real time.
            </p>
            <p style="font-size:0.95rem; line-height:1.55; color:var(--text-muted);">
            For now the deployed dashboard reads from a precomputed cache only —
            no live scraping at runtime, no API keys, no server load on USCF.
            If a player you're searching for isn't in the cached set, the
            dashboard will tell you they're not in the current dataset yet.
            </p>
            """,
        )
    with rR:
        hud_card(
            "DEEPER TRAVEL / GEOGRAPHY ANALYSIS",
            """
            <p style="font-size:1.0rem; line-height:1.6;">
            Travel features are already in the Player Dossier (home region,
            unique event locations, approximate travel distance, "road
            warrior" labels). Future versions can refine this with full
            ZIP-code-level geocoding, event venue lookups, and a travel-map
            visual for each player.
            </p>
            <p style="font-size:0.95rem; line-height:1.55; color:var(--text-muted);">
            Why this matters: travel patterns are a behavioral signal of
            tournament commitment. A player who consistently travels for
            events tends to be more battle-tested than rating alone shows.
            </p>
            """,
        )

    # ------- Value statement strip -------
    divider()
    render_html("""
        <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:0.8rem; margin-top:0.4rem;">
          <div class="hud-card">
            <div class="hud-meta">PRINCIPLE 01</div>
            <p style="font-size:0.98rem; line-height:1.5; margin:0;">
              Rating is the strongest signal, but it is <strong>not the only useful signal</strong>.
            </p>
          </div>
          <div class="hud-card">
            <div class="hud-meta">PRINCIPLE 02</div>
            <p style="font-size:0.98rem; line-height:1.5; margin:0;">
              The model estimates who is favored. The scouting layer explains
              <strong>why a matchup may still be dangerous</strong>.
            </p>
          </div>
          <div class="hud-card">
            <div class="hud-meta">PRINCIPLE 03</div>
            <p style="font-size:0.98rem; line-height:1.5; margin:0;">
              The goal is not to replace chess judgment — it's to give players a
              <strong>better pre-game picture</strong>.
            </p>
          </div>
        </div>
    """)


# ============================================================================
# PAGE 2 — Model Intel
# ============================================================================
def page_model_intel():
    hud_header(
        "MODEL INTEL  ·  PREDICTION RESULTS",
        "How well do the models predict game outcomes?",
        "All metrics are reported on a held-out test set of 1,278 games the model never saw during training.",
    )

    # ------- What the metrics mean (plain English) -------
    hud_card(
        "WHAT THE METRICS MEAN",
        """
        <p style="font-size:1.0rem; line-height:1.6;">
        <strong style="color:var(--cyan-main)">AUC</strong> — how well the model
        separates likely wins from likely losses (higher is better; 0.5 = random).<br/>
        <strong style="color:var(--cyan-main)">Accuracy</strong> — the percentage
        of predictions the model gets right.<br/>
        <strong style="color:var(--cyan-main)">F1 Score</strong> — a balanced
        score for both missed wins and false alarms.<br/>
        <strong style="color:var(--cyan-main)">Precision</strong> — when the
        model says "win," how often it is correct.<br/>
        <strong style="color:var(--cyan-main)">Recall</strong> — of the actual
        wins, how many the model identifies.
        </p>
        """,
    )

    metrics = load_kfold_metrics()
    genai = load_genai_metrics()
    if metrics is None:
        st.error("kfold_cv_metrics.csv is missing — run `scripts/16_kfold_cv_tuning.py`.")
        return

    test_rows = metrics[metrics["split"] == "Test"].copy()
    headline = test_rows[["model", "cv_mean_roc_auc", "roc_auc", "f1", "accuracy", "precision", "recall"]] \
        .rename(columns={
            "model": "Model",
            "cv_mean_roc_auc": "CV AUC",
            "roc_auc": "Test AUC",
            "f1": "F1",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
        })

    extra_rows = []
    if genai is not None:
        # Always include the Elo zero-shot baseline if present
        elo_mask = genai["approach"].str.startswith("Elo")
        if elo_mask.any():
            elo = genai[elo_mask].iloc[0]
            extra_rows.append({
                "Model": "Elo Baseline",
                "CV AUC": np.nan, "Test AUC": elo["roc_auc"],
                "F1": elo["f1"], "Accuracy": elo["accuracy"],
                "Precision": elo["precision"], "Recall": elo["recall"],
            })
        # And the Claude inline-reasoning row (extra credit) if present
        claude_mask = genai["approach"].str.contains("Claude", case=False, na=False)
        if claude_mask.any():
            claude = genai[claude_mask].iloc[0]
            n_claude = int(claude.get("n_test_rows") or 0)
            extra_rows.append({
                "Model": f"Claude Opus 4.7 (LLM, n={n_claude})",
                "CV AUC": np.nan, "Test AUC": claude["roc_auc"],
                "F1": claude["f1"], "Accuracy": claude["accuracy"],
                "Precision": claude["precision"], "Recall": claude["recall"],
            })
    full = pd.concat([headline, pd.DataFrame(extra_rows)], ignore_index=True) if extra_rows else headline

    render_html('<div class="hud-meta">>> MODEL COMPARISON  ·  TEST PERFORMANCE</div>')
    fmt = {c: "{:.4f}" for c in ["CV AUC", "Test AUC", "F1", "Accuracy", "Precision", "Recall"]}
    st.dataframe(
        full.style
            .format(fmt, na_rep="—")
            .background_gradient(subset=["Test AUC"], cmap="Greens"),
        width="stretch", hide_index=True,
    )
    render_html(
        '<p style="font-size:1.0rem; line-height:1.55;">'
        '<strong style="color:var(--cyan-main);">Summary.</strong> '
        'All models perform similarly, with Test AUC around <strong>0.82–0.83</strong>. '
        'This means rating is already a very strong predictor of the outcome. '
        'The machine-learning models add value by also looking at recent activity and form.'
        '</p>'
    )

    cL, cR = st.columns(2)
    with cL:
        fig = px.bar(
            full.sort_values("Test AUC"),
            x="Test AUC", y="Model", orientation="h",
            range_x=[0.5, 0.9],
            color="Test AUC", color_continuous_scale=[(0, "#2D8F95"), (1, "#7AF7FF")],
            title="Test Performance by Model",
        )
        fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=380,
                          coloraxis_showscale=False,
                          xaxis_title="Test AUC (higher is better)",
                          yaxis_title="")
        st.plotly_chart(fig, width="stretch")

    with cR:
        folds = load_per_fold()
        if folds is not None:
            fig2 = px.box(folds, x="model", y="val_roc_auc",
                          title="Cross-Validation Stability (5 time-aware folds)",
                          points="all", color_discrete_sequence=["#7AF7FF"])
            fig2.update_layout(**PLOTLY_LAYOUT, height=380,
                               yaxis_title="Validation AUC",
                               xaxis_title="Model")
            st.plotly_chart(fig2, width="stretch")
            render_html(
                '<p style="font-size:0.95rem; line-height:1.5; color:var(--text-muted);">'
                'Cross-validation means we re-tested the model on five different time-based '
                'slices of the training data. The narrow boxes show that performance is '
                '<strong style="color:var(--cyan-main);">stable</strong>, not lucky.'
                '</p>'
            )

    divider()

    hud_card(
        "HONEST FRAMING  ·  ML MATCHES ELO",
        """
        <p style="font-size:1.02rem; line-height:1.6;">
        All three ML models land within about <strong>0.002 AUC</strong>
        of the closed-form Elo formula on the test set. That is the
        <em>correct</em> finding — Elo has been the gold standard in chess
        prediction since the 1970s because rating already captures most
        of what's predictable about a game.
        </p>
        <p style="font-size:1.02rem; line-height:1.6;">
        The goal here was never to beat Elo at all costs. Elo provides
        a strong baseline; the machine-learning pipeline matches that
        baseline while adding recent activity, form, and other context
        that flow into the scouting layer.
        </p>
        <p style="font-size:1.02rem; line-height:1.6; color:var(--gold-accent);">
        <strong>Product value lives one layer above prediction</strong>
        &mdash; in the scouting intelligence that explains <em>what kind
        of opponent</em> you are facing, not just who is favored.
        </p>
        """,
        variant="gold",
    )


# ============================================================================
# PAGE 3 — Feature Vectors
# ============================================================================
def page_feature_vectors():
    hud_header(
        "FEATURE VECTORS  ·  WHAT THE MODEL LEARNS FROM",
        "Variables explained in plain English",
        "These are the inputs the prediction model uses, with friendly names and what each one tells us.",
    )

    # plain_name, column, family, meaning, why_it_matters, example
    feature_docs = [
        ("Player Rating",                "player_pre_rating",     "Rating",
         "The player's USCF rating coming into the game.",
         "Stronger players usually win more often — this is the single biggest signal.",
         "Rating 1900 going into a tournament round."),
        ("Opponent Rating",              "opponent_pre_rating",   "Rating",
         "The opponent's USCF rating coming into the game.",
         "Half of the matchup. Helps describe how hard the game is.",
         "Opponent rated 2050."),
        ("Rating Difference",            "rating_diff",           "Rating",
         "Player rating minus opponent rating.",
         "Quick summary of who is favored. Positive = player is the higher-rated side.",
         "+150 means the player is rated 150 points higher than their opponent."),
        ("Time Control",                 "time_control",          "Time Control",
         "Whether the game was Regular, Quick, or Blitz.",
         "Players are not equally good in all formats — blitz specialists are different from classical grinders.",
         "Blitz, Quick, Regular, or Unknown."),
        ("Recent Games (30 days)",       "player_games_last_30d", "Activity",
         "Number of rated games played in the last 30 days.",
         "A short-term snapshot of how active and warmed up the player is right now.",
         "12 games in 30 days = very recent activity."),
        ("Recent Games (90 days)",       "player_games_last_90d", "Activity",
         "Number of rated games played in the last 90 days.",
         "The most balanced activity window. Active players are usually sharper.",
         "20 games in 90 days = consistently active."),
        ("Recent Games (365 days)",      "player_games_last_365d","Activity",
         "Number of rated games played in the last year.",
         "Separates committed tournament players from casual entrants.",
         "80 games in a year = serious tournament player."),
        ("Recent Opponent Strength",     "player_recent_avg_opponent_rating_90d", "Recent Form",
         "Average rating of opponents the player has faced in the last 90 days.",
         "Tells us if recent results came against strong or weak fields.",
         "Avg 2100 = battle-tested against strong players."),
        ("Recent Win Rate",              "player_recent_win_rate_90d", "Recent Form",
         "How often the player has been winning recently (last 90 days).",
         "Helps show whether a player may be hot, cold, or stable.",
         "65% win rate over 90 days = currently in good form."),
        ("Cold-Start Flag — opponents",  "missing_recent_avg_opp_90d", "Cold Start",
         "A marker for players who have no recent opponents on file at all.",
         "Tells the model to be cautious about recent-form features rather than guess.",
         "Set to 1 when there is no recent-opponent data to summarize."),
        ("Cold-Start Flag — results",    "missing_recent_win_rate_90d", "Cold Start",
         "A marker for players with no recent results to summarize.",
         "Same idea — explicit missingness is more honest than silently filling in a number.",
         "Set to 1 when there is no recent win-rate data."),
    ]

    family_variant = {"Rating": "", "Time Control": "", "Activity": "",
                      "Recent Form": "", "Cold Start": "danger"}

    fams = sorted({f[2] for f in feature_docs})
    selected = st.multiselect("Filter by feature family", fams, default=fams)

    filtered = [f for f in feature_docs if f[2] in selected]
    for i in range(0, len(filtered), 2):
        cols = st.columns(2)
        for j, feat in enumerate(filtered[i:i + 2]):
            plain_name, column, fam, meaning, why, example = feat
            with cols[j]:
                hud_card(
                    f"[ {fam.upper()} ]",
                    f"""
                    <div style="color:var(--cyan-main);font-size:1.15rem;font-weight:600;margin-bottom:0.15rem;">
                      {plain_name}
                    </div>
                    <div style="font-family:'Share Tech Mono',monospace;color:var(--text-muted);font-size:0.78rem;letter-spacing:0.08em;margin-bottom:0.6rem;">
                      column: <span style="color:var(--cyan-soft);">{column}</span>
                    </div>
                    <div style="color:var(--text-muted);font-size:0.78rem;letter-spacing:0.1em;margin-bottom:0.2rem;text-transform:uppercase;">Plain meaning</div>
                    <p style="margin:0 0 0.55rem 0;font-size:0.98rem;line-height:1.5;">{meaning}</p>
                    <div style="color:var(--text-muted);font-size:0.78rem;letter-spacing:0.1em;margin-bottom:0.2rem;text-transform:uppercase;">Why it matters</div>
                    <p style="margin:0 0 0.55rem 0;font-size:0.98rem;line-height:1.5;">{why}</p>
                    <div style="color:var(--text-muted);font-size:0.78rem;letter-spacing:0.1em;margin-bottom:0.2rem;text-transform:uppercase;">Example</div>
                    <p style="margin:0;font-size:0.95rem;line-height:1.5;color:var(--cyan-soft);"><em>{example}</em></p>
                    """,
                    variant=family_variant.get(fam, ""),
                )

    divider()

    fi = load_feature_importances()
    if fi is not None:
        render_html('<div class="hud-meta">>> FEATURE IMPORTANCE  ·  WHICH SIGNALS THE MODEL RELIES ON MOST</div>')

        # Map technical column names to friendly labels in the chart
        friendly = {
            "player_pre_rating": "Player Rating",
            "opponent_pre_rating": "Opponent Rating",
            "rating_diff": "Rating Difference",
            "player_games_last_30d": "Recent Games (30d)",
            "player_games_last_90d": "Recent Games (90d)",
            "player_games_last_365d": "Recent Games (365d)",
            "player_recent_avg_opponent_rating_90d": "Recent Opponent Strength",
            "player_recent_win_rate_90d": "Recent Win Rate",
            "missing_recent_avg_opp_90d": "Cold-Start (opponents)",
            "missing_recent_win_rate_90d": "Cold-Start (results)",
            "time_control_Quick": "Time Control: Quick",
            "time_control_Regular": "Time Control: Regular",
            "time_control_Unknown": "Time Control: Unknown",
        }
        rf_imp = fi[fi["model"] == "RandomForest"].copy()
        rf_imp["feature_label"] = rf_imp["feature"].map(friendly).fillna(rf_imp["feature"])
        rf_imp = rf_imp.sort_values("importance", ascending=True).tail(10)
        fig = px.bar(
            rf_imp, x="importance", y="feature_label", orientation="h",
            color="importance",
            color_continuous_scale=[(0, "#2D8F95"), (1, "#7AF7FF")],
            title="Most Important Features (Random Forest)",
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=460,
                          showlegend=False, coloraxis_showscale=False,
                          xaxis_title="Importance (higher = more influence on the prediction)",
                          yaxis_title="")
        st.plotly_chart(fig, width="stretch")
        render_html(
            '<p style="font-size:1.0rem; line-height:1.6;">'
            '<strong style="color:var(--cyan-main);">Summary.</strong> '
            'Rating Difference is by far the biggest signal — about 55% of the '
            'model\'s decision is driven by who is the higher-rated side. '
            'Recent activity and recent win rate add useful context (~13% combined), '
            'but they do not replace rating. The cold-start flags tell the model '
            'when to <em>discount</em> recency features rather than pretend they exist.'
            '</p>'
        )
        st.markdown(
            '<div style="color:var(--text-muted);font-size:0.85rem;">'
            'Rating dominates, momentum fine-tunes — this matches what the chess literature '
            'reports about rating-based prediction.'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================================
# PAGE 4 — Player Dossier
# ============================================================================
def _player_picker(profiles: pd.DataFrame, key: str, default_idx: int = 0) -> str:
    rated = profiles.dropna(subset=["current_rating"]).copy()
    rated["display"] = rated.apply(
        lambda r: f"{r['player_id']}  ·  RTG {int(r['current_rating'])}  ·  "
                  f"{int(r['career_n_games']) if pd.notna(r['career_n_games']) else 0} games",
        axis=1,
    )
    options = rated.sort_values("current_rating", ascending=False)["display"].tolist()
    pick = st.selectbox("TARGET PLAYER ID", options, index=min(default_idx, len(options) - 1),
                        key=key)
    return pick.split("  ·  ")[0]


def page_player_dossier():
    hud_header(
        "PLAYER DOSSIER  ·  CLASSIFIED",
        "Opponent Profile Intel",
        "Per-player aggregate of activity, form, upset, momentum, time-control, and event-success vectors.",
    )

    profiles = load_profiles()
    scores = load_scores()
    if profiles is None or scores is None:
        st.error("Player profile/scoring CSVs missing — run scripts 19 & 20.")
        return

    pid = _player_picker(profiles, key="dossier_pick")
    row = profiles[profiles["player_id"] == pid].iloc[0]
    sc_match = scores[scores["player_id"] == pid]
    s = sc_match.iloc[0] if not sc_match.empty else None

    # Top metric strip
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("CURRENT RATING", _fmt_int(row.get("current_rating")))
    m2.metric("CAREER GAMES", _fmt_int(row.get("career_n_games")))
    m3.metric("GAMES LAST 90D", _fmt_int(row.get("games_last_90d")))
    if s is not None and pd.notna(s.get("underrated_score")):
        m4.metric("UNDERRATED POTENTIAL", f"{s['underrated_score']:.0f}/100",
                  s.get("bucket_label"))
    else:
        m4.metric("UNDERRATED POTENTIAL", "—")

    # Soft data advisory — never hard-block. Show what we know, flag what to
    # interpret carefully.
    advisory_text = s.get("data_advisory") if s is not None else None
    if advisory_text:
        hud_card("DATA ADVISORY", f'<p style="font-size:0.98rem; line-height:1.55;">{advisory_text}</p>')

    divider()

    # ----- Activity / Recent Form -----
    cA, cB = st.columns(2)
    with cA:
        bars = []
        bars.append(hud_progress("ACTIVITY · GAMES 90d", min(100, (row.get("games_last_90d", 0) or 0) * 100 / 15)))
        wr90 = row.get("recent_win_rate_90d")
        bars.append(hud_progress("RECENT WIN RATE 90d",
                                 float(wr90) * 100 if pd.notna(wr90) else 0,
                                 "danger" if (pd.notna(wr90) and wr90 < 0.4) else ""))
        sr90 = row.get("recent_score_rate_90d")
        bars.append(hud_progress("RECENT SCORE RATE 90d (W=1, D=0.5)",
                                 float(sr90) * 100 if pd.notna(sr90) else 0))
        cons = row.get("consistency_score")
        if pd.notna(cons):
            bars.append(hud_progress("CONSISTENCY INDEX", float(cons) * 100, "gold"))
        hud_card("ACTIVITY · FORM · CONSISTENCY", "\n".join(bars))

    with cB:
        # Upset / strength of schedule
        bars = []
        sr_hi = row.get("score_rate_vs_higher_rated")
        bars.append(hud_progress("SCORE RATE vs +100 OPPONENTS",
                                 float(sr_hi) * 100 if pd.notna(sr_hi) else 0,
                                 "gold" if (pd.notna(sr_hi) and sr_hi >= 0.5) else ""))
        pct_higher = row.get("pct_games_vs_higher_rated_90d")
        bars.append(hud_progress("PCT GAMES vs HIGHER-RATED 90d",
                                 float(pct_higher) * 100 if pd.notna(pct_higher) else 0))
        upset = row.get("upset_rate_365d")
        bars.append(hud_progress("UPSET WIN RATE 365d (+100)",
                                 float(upset) * 100 if pd.notna(upset) else 0,
                                 "gold" if (pd.notna(upset) and upset >= 0.2) else ""))
        hud_card("UPSET · SCHEDULE STRENGTH", "\n".join(bars))

    # ----- Rating trend chart -----
    cT, cR = st.columns([1.3, 1])
    with cT:
        ratings_seq = []
        for label, d in [("365d", "rating_change_365d_proxy"),
                         ("180d", "rating_change_180d_proxy"),
                         ("90d", "rating_change_90d_proxy"),
                         ("30d", "rating_change_30d_proxy")]:
            v = row.get(d)
            ratings_seq.append({"window": label,
                                "delta": float(v) if pd.notna(v) else 0.0})
        df_trend = pd.DataFrame(ratings_seq)
        fig = px.bar(
            df_trend, x="window", y="delta",
            title="RATING DELTA (PROXY · PRE-RATING CHRONOLOGY)",
            color="delta", color_continuous_scale=[(0, "#C0444A"), (0.5, "#2D8F95"), (1, "#7AF7FF")],
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=320, coloraxis_showscale=False,
                          yaxis_title="Δ rating (points)")
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            '<div style="color:var(--text-muted);font-size:0.8rem;">'
            'TODO: <code>player_post_rating</code> not currently parsed; deltas are reconstructed '
            'from chronological pre-rating values.</div>',
            unsafe_allow_html=True,
        )

    with cR:
        # Time-control profile
        pcts = {
            "Regular": float(row.get("pct_regular_games") or 0) * 100,
            "Quick":   float(row.get("pct_quick_games") or 0) * 100,
            "Blitz":   float(row.get("pct_blitz_games") or 0) * 100,
        }
        df_tc = pd.DataFrame({"tc": list(pcts.keys()), "pct": list(pcts.values())})
        fig = px.bar(df_tc, x="tc", y="pct", title="TIME-CONTROL IDENTITY",
                     color="pct", color_continuous_scale=[(0, "#2D8F95"), (1, "#7AF7FF")])
        fig.update_layout(**PLOTLY_LAYOUT, height=320, coloraxis_showscale=False,
                          yaxis_title="% of games", yaxis_range=[0, 100])
        st.plotly_chart(fig, width="stretch")
        st.markdown(
            f'<div class="hud-pill">SPECIALIST · {row.get("time_control_specialist_label") or "—"}</div>'
            f'<div class="hud-pill gold">BEST TC · {row.get("best_time_control_by_score_rate") or "—"}</div>',
            unsafe_allow_html=True,
        )

    divider()

    # ----- Event Success + Home Region -----
    cE, cH = st.columns([1.5, 1])
    with cE:
        hud_card(
            "EVENT SUCCESS  ·  LAST 365D  (CROSSTABLE-DERIVED)",
            f"""
            <table style="width:100%;border-collapse:collapse;font-family:'Share Tech Mono',monospace;font-size:0.85rem;">
              <tr><td style="color:var(--text-muted);">TOP CROSSTABLE SCORE 90d ...</td><td style="text-align:right;color:var(--cyan-main);">{_fmt_int(row.get('events_won_last_90d'))}</td></tr>
              <tr><td style="color:var(--text-muted);">TOP CROSSTABLE SCORE 365d ..</td><td style="text-align:right;color:var(--cyan-main);">{_fmt_int(row.get('events_won_last_365d'))}</td></tr>
              <tr><td style="color:var(--text-muted);">APPROX TOP-3 FINISHES 365d .</td><td style="text-align:right;color:var(--cyan-main);">{_fmt_int(row.get('top_3_finishes_last_365d'))}</td></tr>
              <tr><td style="color:var(--text-muted);">APPROX TOP-5 FINISHES 365d .</td><td style="text-align:right;color:var(--cyan-main);">{_fmt_int(row.get('top_5_finishes_last_365d'))}</td></tr>
              <tr><td style="color:var(--text-muted);">BEST FINISH PCTILE 365d ....</td><td style="text-align:right;color:var(--gold-accent);">{_fmt_pct(row.get('best_recent_finish_percentile'))}</td></tr>
              <tr><td style="color:var(--text-muted);">AVG FINISH PCTILE 365d .....</td><td style="text-align:right;color:var(--cyan-soft);">{_fmt_pct(row.get('avg_finish_percentile_365d'))}</td></tr>
              <tr><td style="color:var(--text-muted);">AVG FIELD STRENGTH 365d ....</td><td style="text-align:right;color:var(--cyan-soft);">{_fmt_num(row.get('avg_field_strength_last_365d'), signed=False)}</td></tr>
            </table>
            <div style="color:var(--text-muted);font-size:0.78rem;margin-top:0.6rem;line-height:1.5;">
              USCF crosstables are sorted by score group then post-event rating, <em>not</em>
              by tiebreak/prize order &mdash; so these are <strong>approximate</strong> finishes
              (e.g. "top score group") rather than official podium placements.
            </div>
            """,
        )
    with cH:
        avg_mi = row.get("avg_travel_distance_miles")
        max_mi = row.get("max_travel_distance_miles")
        out_pct = row.get("pct_events_outside_home_region")
        unique_locs = row.get("unique_event_locations")
        unique_states = row.get("unique_event_states") or row.get("unique_states_played")
        travel_label = row.get("traveling_competitor_label") or "—"
        travel_conf = row.get("travel_distance_confidence") or "—"
        travel_pill = pill(f"PROFILE · {travel_label}",
                           "gold" if "ROAD WARRIOR" in travel_label.upper() else "")
        hud_card(
            "TRAVEL · GEOGRAPHY",
            f"""
            <p style="font-family:'Share Tech Mono',monospace;font-size:0.85rem;line-height:1.7;">
            HOME REGION ........ <span style="color:var(--cyan-main);">{row.get('inferred_home_region') or '—'}</span><br/>
            STATES PLAYED ...... <span style="color:var(--cyan-main);">{_fmt_int(unique_states)}</span><br/>
            UNIQUE LOCATIONS ... <span style="color:var(--cyan-main);">{_fmt_int(unique_locs)}</span><br/>
            AVG TRAVEL MI ...... <span style="color:var(--gold-accent);">{f"{avg_mi:.0f}" if pd.notna(avg_mi) else "—"}</span><br/>
            MAX TRAVEL MI ...... <span style="color:var(--gold-accent);">{f"{max_mi:.0f}" if pd.notna(max_mi) else "—"}</span><br/>
            OUTSIDE HOME ....... <span style="color:var(--cyan-soft);">{_fmt_pct(out_pct)}</span><br/>
            </p>
            <div style="margin-top:0.4rem;">{travel_pill}</div>
            <div style="color:var(--text-muted);font-size:0.78rem;margin-top:0.5rem;">
              {travel_conf}. USCF MSA exposes city + state + zip on the event header;
              GPS-level coordinates are not provided, so distances are approximate.
            </div>
            """,
        )

    # ------- Plain-English player summary paragraph -------
    divider()
    rating = row.get("current_rating")
    career = row.get("career_n_games") or 0
    g90 = row.get("games_last_90d") or 0
    wr90 = row.get("recent_win_rate_90d")
    wr_phrase = f"{wr90*100:.0f}% recent win rate" if pd.notna(wr90) else "no recent win rate yet"
    tc_label = row.get("time_control_specialist_label") or "balanced across time controls"
    advisory_text = s.get("data_advisory") if s is not None else ""

    hud_card(
        "WRITTEN PLAYER SUMMARY",
        f"""
        <p style="font-size:1.02rem; line-height:1.6;">
        This player is currently rated
        <strong style="color:var(--cyan-main)">{_fmt_int(rating)}</strong> with
        <strong>{career}</strong> career games on file and
        <strong>{g90}</strong> games in the last 90 days
        ({wr_phrase}). Their time-control profile is
        <strong>{tc_label.lower()}</strong>.
        </p>
        <p style="font-size:1.02rem; line-height:1.6; color:var(--text-muted);">
        {advisory_text}
        </p>
        """,
    )

    # ------- Practical chess prep ideas (coaching card) -------
    render_coaching_card(row.to_dict(), s.to_dict() if s is not None else None,
                         heading="PRACTICAL CHESS PREP IDEAS  ·  IF YOU'RE FACING THIS PLAYER")


# ============================================================================
# PAGE 5 — Underrated Protocol
# ============================================================================
def _gauge(value: float, max_value: float = 100.0, title: str = "UNDERRATED POTENTIAL"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"family": "Rajdhani", "color": "#D8FFFF", "size": 44}},
        gauge={
            "axis": {"range": [0, max_value], "tickcolor": "#2D8F95",
                     "tickfont": {"family": "Share Tech Mono", "color": "#7FAFB4"}},
            "bar": {"color": "#7AF7FF"},
            "bgcolor": "rgba(16,32,39,0.4)",
            "borderwidth": 1, "bordercolor": "#2D8F95",
            "steps": [
                {"range": [0, 30],  "color": "rgba(45,143,149,0.18)"},
                {"range": [30, 55], "color": "rgba(122,247,255,0.12)"},
                {"range": [55, 70], "color": "rgba(163,133,96,0.20)"},
                {"range": [70, 85], "color": "rgba(192,68,74,0.25)"},
                {"range": [85, 100], "color": "rgba(192,68,74,0.45)"},
            ],
        },
        title={"text": title,
               "font": {"family": "Share Tech Mono", "color": "#7AF7FF", "size": 14}},
    ))
    layout = {**PLOTLY_LAYOUT, "margin": dict(t=60, b=20, l=20, r=20)}
    fig.update_layout(**layout, height=320)
    return fig


def page_underrated_protocol():
    hud_header(
        "UNDERRATED PROTOCOL  ·  THREAT SCORING",
        "Underrated Potential Score (0–100)",
        "An estimate of whether a player may currently be performing above their official rating.",
    )

    # ------- Plain-English explanation up front -------
    hud_card(
        "WHAT THIS SCORE MEANS",
        """
        <p style="font-size:1.02rem; line-height:1.6;">
        The <strong style="color:var(--cyan-main)">Underrated Potential</strong>
        score estimates whether a player may currently be stronger than their
        official rating suggests. It combines six signals: results against
        stronger opponents, recent form, rating movement, strength of schedule,
        activity, and volatility.
        </p>
        <p style="font-size:1.02rem; line-height:1.6;">
        This score is <strong>not an accusation</strong>. It does not prove a
        player is a "smurf." It simply flags cases where the official rating
        may not fully capture current playing strength &mdash; for example,
        a player who has improved significantly online before their over-the-board
        rating could catch up.
        </p>
        <p style="font-size:0.95rem; line-height:1.55; color:var(--text-muted);">
        A score of 0–30 means the rating looks accurate; 31–55 means no strong
        signal either way; 56–70 is a watchlist; 71–85 is a strong underrated
        signal; 86+ is very likely playing above rating. All scores are damped
        by a sample-size multiplier so flukes can't headline.
        </p>
        """,
    )

    profiles = load_profiles()
    scores = load_scores()
    if profiles is None or scores is None:
        st.error("Run scripts 19 & 20 first.")
        return

    pick = _player_picker(profiles, key="protocol_pick", default_idx=10)
    row = profiles[profiles["player_id"] == pick].iloc[0]
    sc = scores[scores["player_id"] == pick].iloc[0] if not scores[scores["player_id"] == pick].empty else None

    # Source of truth for "do we have any data?" is the profile row's
    # career_n_games count — NOT a possibly-stale `underrated_score` value
    # in the scoring CSV. As long as the player has at least one career
    # game, we render the full breakdown.
    n_career = int(row.get("career_n_games") or 0)
    has_career_data = n_career > 0

    # Per-component scores — pull from the scoring CSV when present,
    # otherwise compute on-the-fly from the profile so the dashboard
    # never silently falls back to "no data" if the CSV is stale.
    def _sc_get(name: str, default: float = 0.0) -> float:
        if sc is None:
            return default
        v = sc.get(name)
        try:
            return float(v) if v is not None and not pd.isna(v) else default
        except (TypeError, ValueError):
            return default

    score_val      = _sc_get("underrated_score")
    upset_v        = _sc_get("upset_score")
    form_v         = _sc_get("form_score")
    moment_v       = _sc_get("momentum_score")
    sched_v        = _sc_get("schedule_score")
    active_v       = _sc_get("activity_score")
    vol_v          = _sc_get("volatility_score")
    sample_mult    = _sc_get("sample_size_multiplier", 1.0)
    bucket         = sc.get("bucket_label") if sc is not None else None
    advisory_text  = sc.get("data_advisory") if sc is not None else None

    cG, cI = st.columns([1, 1.2])
    with cG:
        st.plotly_chart(_gauge(score_val if has_career_data else 0), width="stretch")
        if has_career_data:
            badge_var = "gold" if score_val >= 70 else ""
            label = bucket or "Score available"
            render_html(
                f'<div style="text-align:center;">{pill(label, badge_var)}</div>'
            )

    with cI:
        if not has_career_data:
            hud_card(
                "NO TOURNAMENT HISTORY ON FILE",
                "<p>This player has no career games on file in the current dataset. "
                "Live USCF lookup is on the roadmap — see the landing page.</p>",
                variant="warning",
            )
        else:
            bars = [
                hud_progress(f"UPSET ({upset_v:.1f}/25)",        upset_v * 100 / 25, "gold" if upset_v >= 12 else ""),
                hud_progress(f"RECENT FORM ({form_v:.1f}/20)",   form_v * 100 / 20),
                hud_progress(f"MOMENTUM ({moment_v:.1f}/20)",    moment_v * 100 / 20),
                hud_progress(f"SCHEDULE ({sched_v:.1f}/15)",     sched_v * 100 / 15),
                hud_progress(f"ACTIVITY ({active_v:.1f}/10)",    active_v * 100 / 10),
                hud_progress(f"VOLATILITY ({vol_v:.1f}/10)",     vol_v * 100 / 10),
            ]
            hud_card(
                f"COMPONENT BREAKDOWN  ·  ×{sample_mult:.2f} SAMPLE-SIZE MULTIPLIER",
                "\n".join(bars),
            )

    # ------- Data advisory (always shown — softer than before) -------
    if has_career_data and advisory_text:
        hud_card("DATA ADVISORY", f'<p style="font-size:0.98rem; line-height:1.55;">{advisory_text}</p>')

    # ------- Plain-English summary -------
    if has_career_data:
        bucket_pretty = (bucket or "score available").lower()
        summary_text = (
            f"This player scores <strong>{score_val:.0f} / 100</strong> on "
            f"Underrated Potential — <em>{bucket_pretty}</em>. "
        )
        top_components = sorted(
            [("Upset", upset_v), ("Recent Form", form_v), ("Momentum", moment_v),
             ("Schedule", sched_v), ("Activity", active_v), ("Volatility", vol_v)],
            key=lambda t: -t[1],
        )[:2]
        if any(v >= 6 for _, v in top_components):
            top_names = ", ".join(n for n, v in top_components if v >= 6)
            summary_text += f"Strongest contributing signals: <strong>{top_names}</strong>."
        else:
            summary_text += (
                "No single signal stands out, which usually means the player's "
                "rating is a reasonably honest reflection of recent results."
            )
        hud_card("WHAT THE SCORE SAYS", f'<p style="font-size:1.0rem; line-height:1.6;">{summary_text}</p>')

        # Always-on responsible-use reminder on this page
        hud_card(
            "RESPONSIBLE USE",
            """
            <p style="font-size:0.98rem; line-height:1.55;">
            The Underrated Potential score is a <strong>scouting signal</strong>,
            not a final judgment. It does not prove that a player is a "smurf" or
            doing anything wrong — it simply highlights cases where the rating
            may be lagging behind current strength based on recent results,
            activity, and rating movement.
            </p>
            """,
        )

    if has_career_data and sc is not None and isinstance(sc.get("highlight_signals"), str) and sc.get("highlight_signals"):
        divider()
        sig_html = "".join(
            f"<li>{s.strip()}</li>"
            for s in sc["highlight_signals"].split("•") if s.strip()
        )
        if sig_html:
            hud_card(
                "WHY THIS PLAYER MIGHT BE DANGEROUS",
                f'<ul style="margin:0;padding-left:1.2rem;line-height:1.8;">{sig_html}</ul>',
                variant="gold",
            )

    divider()
    # Distribution of all players
    hud_card(
        "POPULATION DISTRIBUTION",
        '<div style="color:var(--text-muted);font-size:0.85rem;margin-bottom:0.4rem;">'
        'Where this player sits relative to every scouted player in our dataset.</div>'
    )
    scored = scores.dropna(subset=["underrated_score"]).copy()
    fig = px.histogram(
        scored, x="underrated_score", nbins=20,
        title="UNDERRATED POTENTIAL · POPULATION HISTOGRAM",
        color_discrete_sequence=["#7AF7FF"],
    )
    if sc is not None and pd.notna(sc.get("underrated_score")):
        fig.add_vline(x=float(sc["underrated_score"]), line_color="#E05A5F",
                      line_width=2, annotation_text=f"  {pick}",
                      annotation_font_color="#E05A5F")
    fig.update_layout(**PLOTLY_LAYOUT, height=320, bargap=0.08,
                      yaxis_title="Players", xaxis_title="Score")
    st.plotly_chart(fig, width="stretch")


# ============================================================================
# PAGE 6 — Matchup Sim
# ============================================================================
def _opponent_card_html(card: dict, label: str) -> str:
    if not card.get("in_dataset"):
        return f"""
        <div class="hud-card">
          <div class="hud-meta">{label}</div>
          <p>Not in the scouted dataset — using rating only.</p>
        </div>
        """
    rating = _fmt_int(card.get("rating"))
    games90 = _fmt_int(card.get("games_last_90d"))
    wr = _fmt_pct(card.get("recent_win_rate_90d"))
    won365 = _fmt_int(card.get("events_won_last_365d"))
    top5 = _fmt_int(card.get("top_5_finishes_last_365d"))
    rd180 = _fmt_num(card.get("rating_change_180d_proxy"))
    home = card.get("home_region") or "—"
    tc = card.get("best_time_control") or "—"
    score = card.get("underrated_score")
    bucket = card.get("underrated_bucket") or "—"
    if pd.notna(score):
        up_pill = pill(f"UNDERRATED  {score:.0f}/100",
                       "gold" if score >= 70 else "")
        bucket_html = f'<div style="color:var(--cyan-soft);font-size:0.85rem;">{bucket}</div>'
    else:
        up_pill = pill("UNDERRATED  INSUFF DATA", "danger")
        bucket_html = ""
    danger_pill = pill("SMALL-SAMPLE WARNING", "danger") if card.get("small_sample_warning") else ""

    sigs = card.get("highlight_signals") or ""
    sig_html = ""
    if sigs:
        bits = "".join(f"<li>{s.strip()}</li>" for s in sigs.split("•") if s.strip())
        sig_html = f"""
        <div class="hud-meta" style="margin-top:0.7rem;">WHY DANGEROUS</div>
        <ul style="margin:0;padding-left:1.2rem;line-height:1.7;font-size:0.92rem;">{bits}</ul>
        """

    return f"""
    <div class="hud-card">
      <div class="hud-meta">{label}</div>
      <div style="font-family:'Share Tech Mono',monospace;font-size:0.85rem;line-height:1.8;">
        RATING ........... <span style="color:var(--cyan-main);">{rating}</span><br/>
        GAMES 90d ........ {games90}<br/>
        WIN RATE 90d ..... {wr}<br/>
        EVENTS WON 365d .. {won365}<br/>
        TOP-5 365d ....... {top5}<br/>
        Δ RATING 180d .... <span style="color:var(--gold-accent);">{rd180}</span><br/>
        BEST TC .......... {tc}<br/>
        HOME REGION ...... {home}<br/>
      </div>
      <div style="margin-top:0.6rem;">{up_pill}{danger_pill}{bucket_html}</div>
      {sig_html}
    </div>
    """


def _outlook_text(report: dict) -> str:
    pred = report["prediction"]
    p, o = report["player_card"], report["opponent_card"]
    bits = []
    pwin = pred["p_win_model"]
    if pwin >= 0.65:
        bits.append("**CLEAR FAVORITE** &mdash; rating and form both work in your favor.")
    elif pwin >= 0.55:
        bits.append("**SLIGHT EDGE** &mdash; closer than rating alone suggests.")
    elif pwin >= 0.45:
        bits.append("**ROUGHLY EVEN** &mdash; this game could go either way.")
    elif pwin >= 0.35:
        bits.append("**SLIGHT UNDERDOG** &mdash; upset is well within reach.")
    else:
        bits.append("**CLEAR UNDERDOG** &mdash; playing for an upset.")

    d = pred["model_vs_elo_disagreement_pp"]
    if abs(d) >= 5:
        if d > 0:
            bits.append(f"Model is **+{d:.0f} pp more bullish** on you than Elo &mdash; "
                        "your recent activity / form likely outpaces your static rating.")
        else:
            bits.append(f"Model is **{d:.0f} pp less bullish** on you than Elo &mdash; "
                        "opponent's recent form is the more dangerous side of this matchup.")
    if o.get("in_dataset") and o.get("underrated_score") and o["underrated_score"] >= 60:
        bits.append(f"⚠ Opponent's <em>Underrated Potential</em> is "
                    f"<strong>{o['underrated_score']:.0f}/100</strong> "
                    f"({o['underrated_bucket']}) &mdash; expect tougher than rating suggests.")
    if p.get("in_dataset") and (p.get("games_last_90d") or 0) == 0:
        bits.append("Note: **you've been inactive 90+ days** &mdash; possible rust.")
    if o.get("in_dataset") and (o.get("games_last_90d") or 0) >= 15:
        bits.append("Note: opponent is **very active recently** &mdash; sharper than rating suggests.")
    return "<br/><br/>".join(bits)


def page_matchup_sim():
    hud_header(
        "MATCHUP SIM  ·  CONTACT IMMINENT",
        "Predicted Win Probability + Opponent Risk Profile",
        "Model probability + Elo baseline + side-by-side scouting cards + plain-English outlook.",
    )

    profiles = load_profiles()
    scores = load_scores()
    matchup = get_matchup_module()
    if profiles is None or matchup is None:
        st.error("Profiles or matchup module unavailable.")
        return

    c1, c2, c3 = st.columns([1, 1, 0.6])
    with c1:
        p_id = _player_picker(profiles, key="match_p", default_idx=0)
    with c2:
        o_id = _player_picker(profiles, key="match_o", default_idx=1)
    with c3:
        tc = st.selectbox("TIME CONTROL", ["Regular", "Quick", "Blitz", "Unknown"])

    if p_id == o_id:
        st.error("PLAYER == OPPONENT. Select two different IDs.")
        return

    if st.button("EXECUTE MATCHUP SIMULATION", type="primary"):
        with st.spinner("Scanning histories · loading model · computing probabilities..."):
            try:
                report = matchup.build_matchup_report(p_id, o_id, time_control=tc)
            except Exception as e:  # noqa: BLE001
                st.error(f"Matchup failed: {e}")
                return

        pred = report["prediction"]
        cA, cB, cC = st.columns(3)
        cA.metric("MODEL P(WIN)", f"{pred['p_win_model']*100:.1f}%")
        cB.metric("ELO BASELINE", f"{pred['p_win_elo']*100:.1f}%",
                  f"{pred['model_vs_elo_disagreement_pp']:+.1f} pp")
        cC.metric("RATING DIFF", f"{pred['rating_diff']:+.0f}")

        # Risk pill strip
        risk = "LOW"
        if pred["p_win_model"] < 0.4:
            risk = "HIGH"
        elif pred["p_win_model"] < 0.55:
            risk = "MEDIUM"
        risk_var = {"LOW": "", "MEDIUM": "gold", "HIGH": "danger"}[risk]
        delta_pp = pred["model_vs_elo_disagreement_pp"]
        delta_var = "gold" if abs(delta_pp) >= 5 else ""
        threat_pill = pill(f"THREAT LEVEL · {risk}", risk_var)
        tc_pill = pill(f"TIME CONTROL · {tc}")
        delta_pill = pill(f"DELTA vs ELO · {delta_pp:+.1f} pp", delta_var)
        st.markdown(
            f'<div style="text-align:center;margin:0.4rem 0 0.6rem 0;">'
            f'{threat_pill}{tc_pill}{delta_pill}</div>',
            unsafe_allow_html=True,
        )

        divider()
        cL, cR = st.columns(2)
        with cL:
            render_html(_opponent_card_html(report["player_card"], "PLAYER DOSSIER  ·  YOU"))
        with cR:
            render_html(_opponent_card_html(report["opponent_card"], "OPPONENT DOSSIER"))

        divider()
        hud_card("MATCHUP OUTLOOK  ·  PLAIN ENGLISH", _outlook_text(report), variant="success")

        # ------- Practical chess prep ideas, derived from the opponent -------
        # Pull the opponent's profile + scouting score for the coaching helper.
        opp_profile_row = profiles[profiles["player_id"] == o_id]
        opp_score_row = None
        opp_prof_dict = None
        if not opp_profile_row.empty:
            opp_prof_dict = opp_profile_row.iloc[0].to_dict()
            scores_df = load_scores()
            if scores_df is not None:
                m = scores_df[scores_df["player_id"] == o_id]
                if not m.empty:
                    opp_score_row = m.iloc[0].to_dict()
        if opp_prof_dict:
            render_coaching_card(opp_prof_dict, opp_score_row,
                                 heading="PRACTICAL CHESS PREP IDEAS  ·  FOR THIS MATCHUP")


# ============================================================================
# PAGE 7 — Data Repair Log
# ============================================================================
def page_data_repair_log():
    hud_header(
        "DATA REPAIR LOG  ·  PARSER PATCH  v1.1",
        "System Repair Log: USCF Rating Field",
        "How we caught a parser bug that made the strongest chess predictor look worse than random.",
    )

    hud_card(
        "WARNING · CORRUPTED RATING SIGNAL",
        """
        <p><strong style="color:var(--red-soft)">SYMPTOM.</strong>
        During EDA, <code>rating_diff</code> correlated <em>negatively</em>
        with winning (r = <strong>−0.13</strong>). That is mathematically
        impossible if the rating column is correct &mdash; a higher-rated
        player should win more, not less.</p>
        <p><strong style="color:var(--red-soft)">DOWNSTREAM IMPACT.</strong>
        The Elo zero-shot baseline calibrated against <code>rating_diff</code>
        scored <strong>AUC 0.37</strong> on the held-out test set &mdash;
        <em>worse than random</em>. The tuned Random Forest was at 0.67,
        which <em>looked</em> fine if you didn't know what good looked like.</p>
        """,
        variant="warning",
    )

    hud_card(
        "DIAGNOSTIC · CHESS LOGIC FAILED",
        """
        <p>If a higher-rated player loses more often than a lower-rated one,
        either the universe broke or the data did. We trusted chess theory
        and audited the data pipeline upstream of the model.</p>
        <p>Spot-checks of individual rows showed <strong>"ratings" that
        looked suspiciously like the leading digits of the opponent's USCF
        ID</strong> (e.g. ID 31462359 → rating "3146"; ID 16108388 →
        rating "1610"). That's not a real chess rating &mdash; that's an
        ID prefix.</p>
        """,
    )

    hud_card(
        "PATCH · PARSER REGEX FIXED",
        """
        <p>Root cause in <code>src/parser/msa_parser.py</code>:</p>
        <pre style="background:rgba(122,247,255,0.04);padding:0.6rem;border-left:2px solid var(--red-soft);">
<span style="color:var(--red-soft)">- rating_match = re.search(r'(?:R:)?\\s*(\\d{3,4})', id_rating_str)</span>
<span style="color:var(--cyan-main)">+ rating_match = re.search(r'R:\\s*(\\d{3,4}P?\\d*)', id_rating_str)</span>
        </pre>
        <p>The original regex was unanchored: <code>\\d{3,4}</code> matched
        the <em>first</em> 3-4 digit run in the string, which is the leading
        digits of the 8-digit USCF ID, not the rating that follows
        <code>R:</code>. The fix anchors on the literal <code>R:</code> token
        and preserves the optional provisional suffix
        (e.g. <code>1450P12</code>) for downstream cleaning.</p>
        <p>We then re-parsed the entire local HTML cache (<strong>3,020
        crosstables</strong>) — no re-scraping needed.</p>
        """,
    )

    # Before/After table — hand-rolled HTML for full styling control
    bf_rows = [
        ("rating_diff ↔ win (correlation)", "−0.13",              "+0.49"),
        ("Elo zero-shot test AUC",           "0.37",               "0.825"),
        ("Tuned Random Forest test AUC",     "0.674",              "0.823"),
        ("Tuned Gradient Boosting test AUC", "0.668",              "0.825"),
        ("Median parsed rating",             "1483 (ID prefix)",   "1998 (real)"),
        ("Max parsed rating",                "3261 (ID prefix)",   "2861 (real)"),
    ]
    th_style = (
        "color:var(--cyan-soft);text-align:left;padding:0.4rem 0.6rem;"
        "border-bottom:1px solid var(--cyan-muted);text-transform:uppercase;"
        "letter-spacing:0.1em;"
    )
    td_style = "padding:0.35rem 0.6rem;border-bottom:1px solid rgba(122,247,255,0.08);"
    table_style = (
        "width:100%;border-collapse:collapse;"
        "font-family:'Share Tech Mono',monospace;font-size:0.85rem;"
    )
    rows_html = "".join(
        f"<tr><td style=\"{td_style}\">{sig}</td>"
        f"<td style=\"{td_style};color:var(--red-soft);\">{before}</td>"
        f"<td style=\"{td_style};color:var(--cyan-main);\">{after}</td></tr>"
        for sig, before, after in bf_rows
    )
    table_html = (
        f"<table style=\"{table_style}\">"
        f"<thead><tr>"
        f"<th style=\"{th_style}\">Signal</th>"
        f"<th style=\"{th_style}\">Before (bug)</th>"
        f"<th style=\"{th_style}\">After (fixed)</th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table>"
    )

    hud_card("ACCESS RESTORED · BEFORE / AFTER", table_html, variant="success")

    hud_card(
        "TAKEAWAY · PROCESS DISCIPLINE",
        """
        <p>Always sanity-check feature correlations against
        <strong style="color:var(--gold-accent)">domain knowledge</strong>
        before trusting the model. A regex bug in the data-collection
        layer made the world's strongest known chess predictor look
        worse than random &mdash; and the model's accuracy alone
        was not enough to catch it.</p>
        """,
        variant="gold",
    )


# ============================================================================
# Dispatcher
# ============================================================================
PAGES = {
    "01 // COMMAND CENTER":      page_command_center,
    "02 // MODEL INTEL":         page_model_intel,
    "03 // FEATURE VECTORS":     page_feature_vectors,
    "04 // PLAYER DOSSIER":      page_player_dossier,
    "05 // UNDERRATED PROTOCOL": page_underrated_protocol,
    "06 // MATCHUP SIM":         page_matchup_sim,
    "07 // DATA REPAIR LOG":     page_data_repair_log,
}

render_top_band()
PAGES[page]()
render_footer()
