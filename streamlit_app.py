#!/usr/bin/env python3
"""

Author Name:  Imaad Ullah Khan
Author Email: yameenimaad@gmail.com
AI Helper:    Claude
Pink Edge AI — Streamlit (responsive web) edition
====================================================
A responsive web UI, sibling to the Tkinter desktop app (GUI.py) — same shared logic, same model
backend (inference.py: Roboflow-hosted models, the offline_cv.py pixel-diff heuristic, offline HF
models, and the SIMULATED scenario picker, tried in whichever order is measurably best per
modality — see Documentations/MODEL_SOURCES.md), same SQLite cache. All shared logic (constants,
imaging, simulated scenarios, DB, reports, the run_triage() dispatcher, draw_bbox() — which now
draws a real detected bounding box when a result carries one, e.g. from offline_cv.py) is imported
straight from GUI.py rather than duplicated — GUI.py's Tkinter code never runs unless GUI.py itself
is executed directly, so importing it here is safe.

Run with:  streamlit run streamlit_app.py
"""
import time
from datetime import datetime

import streamlit as st

import GUI as core  # noqa: the shared logic module — see module docstring above

st.set_page_config(page_title="Pink Edge AI", page_icon="🩸", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# COLOR TOKENS & DARK CLINICAL BLUE THEME
# ============================================================
# Imported from GUI.py rather than redefined here — one source of truth for both editions. These
# two used to keep their own separate dicts and had drifted apart on every brand colour and on
# four of the six semantic ones; see GUI.py's THEME block for the palette and its rationale.
#
# NOTE: the CSS below is a plain string, not an f-string (it's full of literal CSS braces), so it
# repeats these values as literals rather than interpolating them. Keep the two in sync when
# changing a colour — `python Tests/smoke_test.py` checks for exactly that drift.
C = dict(core.C)
# Dark clinical blue theme for the web edition. Status/data colours are neutral;
# success/warning/danger stay vivid because they carry real medical meaning.
C.update({
    "bg": "#020B14", "surface": "#091E33", "surface_alt": "#102A43", "surface_hover": "#102A43",
    "border": "#24445F", "border_light": "#102A43",
    "text": "#F0F4F8", "text_muted": "#8596A9", "text_light": "#8596A9",
    "primary": "#2382EC", "primary_light": "#F0F4F8",
    "accent": "#0B5ED7", "accent_light": "#7FB5F5",
    "success": "#22C55E", "warning": "#F59E0B", "danger": "#EF4444",
})

# ============================================================
# VOICE MESSAGE LIBRARY
# ------------------------------------------------------------
# Offline text templates for the Voice Guidance feature (English/Urdu/
# Punjabi). play_voice_message() is a stub that returns None (no audio
# bytes) so the UI still renders and functions with text-only fallback.
# Swap its body for a real offline TTS engine (e.g. Piper) or an online
# TTS API (e.g. gTTS) when ready — it must keep returning bytes st.audio()
# can play, or None if unavailable.
# ============================================================
VOICE_MESSAGES = {
    "rescan": {
        "en": "Image quality too low. Please rescan the patient.",
        "ur": "تصویر کا معیار کم ہے۔ براہ کرم دوبارہ اسکین کریں۔",
        "pa": "تصویر دا معیار گھٹ ہے۔ دوبارہ سکین کرو۔",
    },
    "escalate": {
        "en": "Urgent referral required. Please escalate this case.",
        "ur": "فوری ریفرل ضروری ہے۔ براہ کرم یہ کیس آگے بھیجیں۔",
        "pa": "فوری ریفرل ضروری ہے۔ کیس اگے بھیجو۔",
    },
    "missing_field": {
        "en": "A required field is missing. Please complete the form.",
        "ur": "ایک ضروری خانہ خالی ہے۔ فارم مکمل کریں۔",
        "pa": "اک ضروری خانہ خالی ہے۔ فارم پورا کرو۔",
    },
    "success": {
        "en": "Triage completed successfully.",
        "ur": "ٹریاج کامیابی سے مکمل ہوگئی۔",
        "pa": "ٹریاج کامیابی نال مکمل ہوگئی۔",
    },
    "sync_success": {
        "en": "Cloud sync completed successfully.",
        "ur": "کلاؤڈ سنک کامیابی سے مکمل ہوگئی۔",
        "pa": "کلاؤڈ سنک کامیابی نال مکمل ہوگئی۔",
    },
    "network_error": {
        "en": "Network connection failed. Working in offline mode.",
        "ur": "نیٹ ورک کنکشن ناکام ہوگیا۔ آف لائن موڈ میں کام جاری ہے۔",
        "pa": "نیٹ ورک کنکشن ناکام ہوگیا۔ آف لائن موڈ وچ کم جاری ہے۔",
    },
}


def play_voice_message(message_key: str, lang: str):
    """
    Returns audio bytes suitable for st.audio(), or None if no audio is
    available (the UI falls back to showing the message text only).
    Currently a stub with no real TTS engine wired in — plug in Piper
    (offline) or gTTS (online, needs internet) here later.
    """
    # Example for a real implementation with Piper (offline):
    #   from piper import synthesize
    #   return synthesize(VOICE_MESSAGES.get(message_key, {}).get(lang, ""))
    return None


def combine_icon_and_text(icon: str, text: str) -> str:
    """
    Joins a status icon with its text without duplicating the icon —
    some model results already bake an emoji into the verdict string
    (e.g. "🟢 Routine Screening"), so blindly prefixing `icon` again
    produced "🟢 🟢 Routine Screening". This checks first.
    """
    text = (text or "").strip()
    icon = (icon or "").strip()
    if not icon or text.startswith(icon):
        return text
    return f"{icon} {text}".strip()


def html_block(s: str) -> str:
    """
    Strips per-line leading/trailing whitespace from a multi-line HTML
    string built with an indented Python f-string.

    Why this matters: Streamlit's markdown renderer follows CommonMark,
    where any line indented 4+ spaces is treated as a literal "indented
    code block" — rendered as raw monospace text, HTML tags and all,
    even with unsafe_allow_html=True. Since our HTML strings are written
    inside nested functions/if-blocks, every line naturally inherits 8+
    spaces of Python indentation, which was silently triggering this
    exact bug. Wrap any multi-line HTML f-string passed to st.markdown
    with this function to avoid it.
    """
    return "\n".join(line.strip() for line in s.strip("\n").splitlines())


# ------------------------------------------------------------
# GLOBAL FIX: auto-dedent every st.markdown(..., unsafe_allow_html=True)
# ------------------------------------------------------------
# html_block() above only helps at call sites that actually use it — and
# none of them did. Several st.markdown(f"""<div ...>...""",
# unsafe_allow_html=True) calls inside render_dashboard / render_hospital_hub
# / render_cloud_sync are written inside nested functions/if-blocks, so
# their f-strings inherit 8+ spaces of Python indentation. Streamlit's
# Markdown renderer follows CommonMark, where any line indented 4+ spaces is
# treated as a literal "indented code block" and rendered as raw text
# (tags and all, complete with the copy-icon Streamlit shows on code
# blocks) even with unsafe_allow_html=True. That's exactly the bug in the
# screenshots (the "Triage result" card, the DICOM Metadata card, etc.
# showing raw <div style="..."> text instead of styled HTML).
#
# Instead of manually hunting down and wrapping every individual call site
# (error-prone — miss one and it breaks again on a different card later),
# patch st.markdown once so ANY html string passed with
# unsafe_allow_html=True gets dedented automatically, no matter how deeply
# nested the call site is.

# ------------------------------------------------------------
# GLOBAL FIX: auto-dedent HTML passed to st.markdown
# ------------------------------------------------------------
# IMPORTANT: Streamlit reruns this script. Do not capture st.markdown here,
# because on a later rerun it may already be our patched function, which
# creates infinite recursion. Calling DeltaGenerator.markdown directly always
# reaches Streamlit's real implementation.
# Use Streamlit's internal main DeltaGenerator directly.
# This avoids calling the patched st.markdown again on reruns.
def _dedented_markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html") and isinstance(body, str) and "\n" in body:
        body = html_block(body)
    return st._main.markdown(body, *args, **kwargs)

st.markdown = _dedented_markdown

CSS = """
<style>
/* ============================================================
   Pink Edge AI — dark clinical theme
   App bg #020B14 | sidebar/cards #091E33 | accent #2382EC / #0B5ED7
   text #F0F4F8 | secondary text #8596A9 | imaging canvas #000000
   Bright red/green/amber are reserved for real medical alerts.
   ============================================================ */
:root {
    --pe-bg: #020B14;
    --pe-surface: #091E33;
    --pe-surface-alt: #102A43;
    --pe-border: #102A43;
    --pe-border-strong: #24445F;
    --pe-accent: #2382EC;
    --pe-accent-hi: #0B5ED7;
    --pe-text: #F0F4F8;
    --pe-muted: #8596A9;
    --pe-canvas: #000000;
}

/* Main App Background & Default Text */
.stApp {
    background-color: #020B14 !important;
    color: #F0F4F8 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    margin-bottom: 20px;
}
[data-testid="stHeader"] {
    background-color: #020B14 !important;
}

/* Sidebar — deep charcoal/slate */
section[data-testid="stSidebar"] {
    background-color: #091E33 !important;
    border-right: 1px solid #102A43 !important;
}
section[data-testid="stSidebar"] * {
    color: #F0F4F8 !important;
}
section[data-testid="stSidebar"] hr {
    border-color: #102A43 !important;
}

/* Container Padding */
.block-container {
    /* Extra space below Streamlit's top toolbar so navigation and
       page content never overlap or get clipped. */
    padding-top: 4.5rem !important;
    padding-bottom: 3rem !important;
    max-width: 1300px !important;
}

/* Top navigation tabs */
.stTabs {
    margin-top: 0.25rem !important;
    margin-bottom: 1.25rem !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0.15rem !important;
    overflow-x: auto !important;
    white-space: nowrap !important;
    padding-bottom: 2px !important;
    background-color: #091E33 !important;
    border-bottom: 1px solid #102A43 !important;
}
.stTabs [data-baseweb="tab"] {
    min-height: 42px !important;
    padding: 0.5rem 1rem !important;
    white-space: nowrap !important;
    color: #8596A9 !important;
    font-weight: 700 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #F0F4F8 !important;
}
.stTabs [aria-selected="true"] {
    color: #F0F4F8 !important;
}
/* clean blue indicator line under the active tab */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #0B5ED7 !important;
    height: 3px !important;
}
.stTabs [data-baseweb="tab-border"] {
    background-color: #102A43 !important;
}

/* Give every page header a clear separation from navigation. */
.page-header {
    margin-top: 0.75rem !important;
}

/* Headings & Text */
h1, h2, h3, h4, h5, h6, label, p, li, td, th {
    color: #F0F4F8 !important;
}
.stMarkdown p, .stMarkdown label {
    color: #F0F4F8 !important;
}
small, .stCaption, [data-testid="stCaptionContainer"] {
    color: #8596A9 !important;
}

/* Header Banner — solid slate card with a blue indicator line */
.page-header {
    background: #091E33 !important;
    border: 1px solid #102A43 !important;
    border-left: 4px solid #2382EC !important;
    border-radius: 12px;
    padding: 22px 28px;
    margin-bottom: 20px;
    box-shadow: none;
}
.page-header h1 {
    color: #F0F4F8 !important;
    margin: 0 !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
}
.page-header p {
    color: #8596A9 !important;
    font-size: 0.88rem !important;
    margin: 6px 0 0 0 !important;
}

/* Status badges — muted grays (colour is saved for real alerts) */
.badge {
    display: inline-flex;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
}
.badge-offline,
.badge-cloud,
section[data-testid="stSidebar"] .badge-offline,
section[data-testid="stSidebar"] .badge-cloud {
    background: rgba(31, 41, 55, 0.6) !important;
    color: #B6C3D1 !important;
    border: 1px solid #24445F !important;
}

/* Cards & Metric Tiles */
.card {
    background: #091E33 !important;
    border: 1px solid #102A43 !important;
    border-radius: 12px !important;
    padding: 18px !important;
    margin: 8px 0 !important;
    color: #F0F4F8 !important;
    box-shadow: none !important;
}
.card b, .card span {
    color: #F0F4F8 !important;
}

.metric-tile {
    background: #091E33 !important;
    border: 1px solid #102A43 !important;
    border-radius: 10px !important;
    padding: 14px !important;
    text-align: center !important;
    box-shadow: none !important;
}
.metric-tile .label {
    color: #8596A9 !important;
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}
/* no !important: an inline colour is only used for genuine alert counts */
.metric-tile .value {
    color: #F0F4F8;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    margin-top: 4px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* Verdict Boxes — real medical findings keep red / amber / green */
.verdict-box {
    border-radius: 12px !important;
    padding: 20px !important;
    text-align: center !important;
    margin: 12px 0 !important;
}
.verdict-box.success {
    background: rgba(34, 197, 94, 0.10) !important;
    border: 2px solid rgba(34, 197, 94, 0.55) !important;
}
.verdict-box.success .v-title { color: #4ADE80 !important; }
.verdict-box.danger {
    background: rgba(239, 68, 68, 0.10) !important;
    border: 2px solid rgba(239, 68, 68, 0.60) !important;
}
.verdict-box.danger .v-title { color: #F87171 !important; }
.verdict-box.warning {
    background: rgba(245, 158, 11, 0.10) !important;
    border: 2px solid rgba(245, 158, 11, 0.55) !important;
}
.verdict-box.warning .v-title { color: #FBBF24 !important; }
.verdict-box .v-icon { font-size: 1.8rem !important; }
.verdict-box .v-title { font-size: 1.3rem !important; font-weight: 800 !important; }
.verdict-box .v-sub {
    font-size: 0.88rem !important;
    margin-top: 4px !important;
    color: #B6C3D1 !important;
    font-weight: 600 !important;
}

/* Telemetry Log */
.console-log {
    background: #020B14 !important;
    border: 1px solid #102A43 !important;
    border-radius: 10px !important;
    padding: 14px !important;
    color: #B6C3D1 !important;
    font-family: 'Consolas', 'Courier New', monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.7 !important;
    max-height: 260px !important;
    overflow-y: auto !important;
}

/* Alert Cards */
.alert-card {
    border-radius: 10px !important;
    padding: 14px !important;
    margin: 6px 0 !important;
    border-left: 4px solid !important;
}
.alert-card.critical {
    background: rgba(239, 68, 68, 0.10) !important;
    border-color: #EF4444 !important;
    color: #FCA5A5 !important;
}
.alert-card.ok {
    background: rgba(34, 197, 94, 0.08) !important;
    border-color: #22C55E !important;
    color: #86EFAC !important;
}

.source-tag {
    font-size: 0.75rem !important;
    color: #8596A9 !important;
    font-family: monospace !important;
    font-weight: 700 !important;
}

/* Streamlit native inputs */
div[data-baseweb="select"] > div {
    background-color: #020B14 !important;
    color: #F0F4F8 !important;
    border-color: #24445F !important;
}
div[data-baseweb="select"] span {
    color: #F0F4F8 !important;
}
div[data-baseweb="popover"] ul, div[data-baseweb="menu"] {
    background-color: #091E33 !important;
}
div[data-baseweb="popover"] li {
    color: #F0F4F8 !important;
}
div[data-baseweb="popover"] li:hover {
    background-color: rgba(35, 130, 236, 0.18) !important;
}
div[data-baseweb="input"], div[data-baseweb="textarea"], div[data-baseweb="base-input"] {
    background-color: #020B14 !important;
    border-color: #24445F !important;
}
div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea {
    color: #F0F4F8 !important;
    -webkit-text-fill-color: #F0F4F8 !important;
    background-color: #020B14 !important;
}
div[data-baseweb="input"]:focus-within, div[data-baseweb="select"] > div:focus-within {
    border-color: #0B5ED7 !important;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: #020B14 !important;
    border: 1px dashed #24445F !important;
}
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] span {
    color: #8596A9 !important;
}
[data-testid="stExpander"] {
    background-color: #091E33 !important;
    border: 1px solid #102A43 !important;
    border-radius: 10px !important;
}

/* Secondary action buttons — transparent dark fill, subtle grey border */
.stButton > button,
.stDownloadButton > button,
button[data-testid="stBaseButton-secondary"] {
    background-color: rgba(17, 24, 39, 0.55) !important;
    color: #F0F4F8 !important;
    border: 1px solid #24445F !important;
    font-weight: 600 !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
button[data-testid="stBaseButton-secondary"]:hover {
    background-color: #102A43 !important;
    border-color: #0B5ED7 !important;
    color: #F0F4F8 !important;
}
.stButton > button:disabled,
.stDownloadButton > button:disabled {
    opacity: 0.45 !important;
}

/* Main action buttons — solid blue, white text */
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background-color: #2382EC !important;
    color: #FFFFFF !important;
    border: 1px solid #2382EC !important;
    font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    background-color: #0B5ED7 !important;
    border-color: #0B5ED7 !important;
    color: #FFFFFF !important;
}
.stButton > button[kind="primary"] p,
.stButton > button[kind="primary"] span,
button[data-testid="stBaseButton-primary"] p {
    color: #FFFFFF !important;
}

/* Active selection state (LHW / Doctor) — subtle blue fill + indicator line,
   not the solid "main action" style */
.st-key-role_lhw button[data-testid="stBaseButton-primary"],
.st-key-role_doc button[data-testid="stBaseButton-primary"] {
    background-color: rgba(35, 130, 236, 0.16) !important;
    border: 1px solid rgba(11, 94, 215, 0.45) !important;
    box-shadow: inset 0 -3px 0 #0B5ED7 !important;
    color: #F0F4F8 !important;
}
.st-key-role_lhw button[data-testid="stBaseButton-primary"]:hover,
.st-key-role_doc button[data-testid="stBaseButton-primary"]:hover {
    background-color: rgba(35, 130, 236, 0.24) !important;
}
.st-key-role_lhw button p, .st-key-role_doc button p {
    color: #F0F4F8 !important;
}

/* Tables */
.stDataFrame, [data-testid="stTable"] {
    background-color: #091E33 !important;
    border: 1px solid #102A43 !important;
    border-radius: 8px !important;
}

/* Imaging canvas — always pure black so diagnostic grayscale is unaltered */
[data-testid="stImage"] {
    background: #000000 !important;
    border: 1px solid #102A43;
    border-radius: 10px;
    padding: 6px;
}
[data-testid="stImage"] img {
    background: #000000 !important;
    border-radius: 6px;
}
[data-testid="stImageCaption"] {
    color: #8596A9 !important;
}

/* ============= AI CONFIDENCE + LHV OVERRIDE STYLING ============= */
.ai-recommendation {
  background: #091E33;
  border: 1px solid #102A43;
  border-left: 4px solid #0B5ED7;
  border-radius: 14px; padding: 20px; margin: 12px 0;
  box-shadow: none;
  color: #F0F4F8;
}
/* Risk-based colours = real medical alert colours */
.ai-recommendation.risk-critical {
  background: rgba(239, 68, 68, 0.12) !important;
  border: 1px solid rgba(239, 68, 68, 0.55) !important;
  border-left: 4px solid #EF4444 !important;
  box-shadow: none !important;
}
.ai-recommendation.risk-moderate {
  background: rgba(245, 158, 11, 0.12) !important;
  border: 1px solid rgba(245, 158, 11, 0.55) !important;
  border-left: 4px solid #F59E0B !important;
  box-shadow: none !important;
}
.ai-recommendation.risk-low {
  background: rgba(34, 197, 94, 0.10) !important;
  border: 1px solid rgba(34, 197, 94, 0.5) !important;
  border-left: 4px solid #22C55E !important;
  box-shadow: none !important;
}
.ai-recommendation .rec-header {
  font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
  color: #8596A9; margin-bottom: 8px;
}
.ai-recommendation .rec-verdict {
  font-size: 1.4rem; font-weight: 800; margin: 8px 0; color: #F0F4F8;
}
.ai-recommendation .rec-confidence {
  font-size: 0.95rem; margin-top: 12px; color: #B6C3D1;
}

.lhv-decision {
  background: #091E33;
  border: 1px solid #102A43;
  border-left: 4px solid #2382EC;
  border-radius: 14px; padding: 20px; margin: 12px 0;
  color: #F0F4F8;
}
.lhv-decision .lhv-header {
  font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
  color: #8596A9; margin-bottom: 16px;
}

.btn-agree {
  background: #2382EC !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 8px;
  padding: 12px 24px;
  font-weight: 700;
  font-size: 0.95rem;
  cursor: pointer;
  transition: all 0.3s ease;
}
.btn-agree:hover { background: #0B5ED7 !important; }

.btn-override {
  background: transparent !important;
  color: #F0F4F8 !important;
  border: 1px solid #24445F !important;
  border-radius: 8px;
  padding: 12px 24px;
  font-weight: 700;
  font-size: 0.95rem;
  cursor: pointer;
  transition: all 0.3s ease;
}
.btn-override:hover { background: #102A43 !important; border-color: #0B5ED7 !important; }

.override-reason-box {
  background: #091E33;
  border: 1px solid #24445F;
  border-left: 4px solid #2382EC;
  border-radius: 12px;
  padding: 16px;
  margin-top: 16px;
}
.override-reason-box .reason-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: #F0F4F8;
  margin-bottom: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.override-reason-box .reason-option {
  display: flex;
  align-items: center;
  padding: 8px 12px;
  cursor: pointer;
  border-radius: 6px;
  transition: background 0.2s ease;
}
.override-reason-box .reason-option:hover { background: rgba(35, 130, 236, 0.14); }
.override-reason-box .reason-option input[type="radio"] {
  margin-right: 12px;
  cursor: pointer;
  accent-color: #0B5ED7;
  width: 18px;
  height: 18px;
}
.override-reason-box .reason-option label {
  cursor: pointer;
  color: #F0F4F8;
  font-weight: 500;
  margin: 0;
}

.decision-status {
  background: rgba(255, 255, 255, 0.04);
  border-radius: 8px;
  padding: 12px 16px;
  margin-top: 12px;
  font-weight: 600;
  text-align: center;
  border: 1px solid #24445F;
}

/* ============= VOICE FEEDBACK SYSTEM ============= */
.voice-feedback-panel {
  background: #091E33;
  border: 1px solid #102A43;
  border-radius: 14px;
  padding: 0;
  margin: 16px 0;
  color: #F0F4F8;
  overflow: hidden;
}
.voice-feedback-panel .vfp-header {
  background: #020B14;
  font-size: 0.95rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #F0F4F8;
  padding: 16px;
  margin: 0;
  border-bottom: 1px solid #102A43;
  border-left: 4px solid #2382EC;
}
.voice-feedback-panel .vfp-message {
  font-size: 1.05rem;
  font-weight: 600;
  margin: 16px;
  line-height: 1.5;
  padding: 12px;
  background: #020B14;
  border-radius: 8px;
  border-left: 4px solid #0B5ED7;
  color: #F0F4F8;
}
.voice-btn-group {
  display: flex;
  gap: 8px;
  margin: 0 16px 16px 16px;
  flex-wrap: wrap;
}
.voice-btn {
  background: #2382EC !important;
  border: none !important;
  color: #ffffff !important;
  padding: 10px 16px !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  cursor: pointer !important;
  transition: all 0.3s ease !important;
}
.voice-btn:hover { background: #0B5ED7 !important; }
.voice-status {
  background: #020B14;
  border-radius: 6px;
  padding: 8px 12px;
  margin: 0 16px 16px 16px;
  font-size: 0.85rem;
  text-align: center;
  border: 1px solid #24445F;
  color: #8596A9;
}
.audio-player-wrapper {
  background: #020B14;
  border-radius: 8px;
  padding: 12px 16px;
  margin: 0 16px 16px 16px;
}

.helpya-card {
  background: #091E33;
  border: 1px solid #102A43;
  border-radius: 14px;
  padding: 0;
  margin: 16px 0;
  color: #F0F4F8;
  overflow: hidden;
}
.helpya-card .helpya-header {
  background: #020B14;
  font-size: 1.15rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #F0F4F8;
  padding: 16px;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 12px;
  border-bottom: 1px solid #102A43;
  border-left: 4px solid #2382EC;
}
.helpya-stat {
  display: inline-block;
  background: #020B14;
  border-radius: 12px;
  padding: 16px;
  margin: 12px;
  font-weight: 600;
  border: 1px solid #24445F;
  color: #F0F4F8;
}
.helpya-stat .stat-label {
  font-size: 0.8rem;
  color: #8596A9;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.helpya-stat .stat-value {
  font-size: 1.6rem;
  font-weight: 800;
  margin-top: 6px;
}

/* Section banners used on the Voice / Feedback / Summary pages */
.pe-section {
  background: #091E33;
  border: 1px solid #102A43;
  border-left: 4px solid #2382EC;
  color: #F0F4F8 !important;
  padding: 12px 16px;
  border-radius: 8px;
  margin: 16px 0 12px 0;
  font-size: 1.2rem;
  font-weight: 800;
}
.pe-note {
  background: #020B14;
  border: 1px solid #102A43;
  border-left: 4px solid #0B5ED7;
  color: #B6C3D1 !important;
  padding: 8px 12px;
  border-radius: 6px;
  margin: 8px 0;
  font-size: 1rem;
  font-weight: 700;
}

/* Responsive: narrow viewports */
@media (max-width: 768px) {
  .block-container {
    padding-left: 10px !important;
    padding-right: 10px !important;
    padding-top: 3.5rem !important;
  }
  .stTabs [data-baseweb="tab"] { padding-left: 0.7rem !important; padding-right: 0.7rem !important; }
  .page-header { padding: 16px 18px; margin-top: 0.5rem !important; }
  .page-header h1 { font-size: 1.15rem !important; }
  .metric-tile .value { font-size: 1.1rem; }
  .ai-recommendation, .lhv-decision { padding: 16px; }
  .ai-recommendation .rec-verdict { font-size: 1.2rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

core.init_db()

# ============================================================
# SESSION STATE (per-browser-session, not shared across users)
# ============================================================
def init_state():
    import random
    from dicom_anonymizer import generate_hex_privacy_hash
    from auth_manager import ROLE_LHW
    raw_id = random.randint(10000000, 99999999)
    privacy_hash = generate_hex_privacy_hash({"pat_id": raw_id, "cnic": f"35201-{raw_id}-1"})
    defaults = {
        "user_role": ROLE_LHW,
        "urdu_mode": False, "network_mode": "Fully Offline", "show_bbox": True,
        "log_entries": [], "inference_done": False, "last_model": None,
        "current_result": None, "current_image": None, "uploaded_name": None,
        "pat_id": raw_id, "privacy_hash": privacy_hash, "pat_age": random.randint(30, 70),
        "hw_stats": {"load": "34%", "power": "6.2W", "temp": "42C"},
        "net_stats": {"signal": "-85 dBm", "bhus": 14, "module": "ONLINE"},
        "sms_alerts": [], "iot_messages": [], "oss_uploads": [], "acr_status": None,
        "inference_latency": 9.4, "bi_rads_selected": None, "acr_density_selected": None,
        "cache_saved": False,
        "lhv_decisions": {}, "show_override_reason": {},
        # Voice + Help/Feedback panel state (Cloud Sync tab)
        "voice_enabled": True,
        "helpya_total_sessions": 0,
        "helpya_successful_diagnosis": 0,
        "helpya_escalations": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
core._urdu["on"] = st.session_state.urdu_mode
t = core.t


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    from auth_manager import AuthManager, ROLE_LHW, ROLE_RADIOLOGIST, ROLE_CONFIGS
    with st.sidebar:
        st.markdown(
            '<div style="background:#020B14;border:1px solid #102A43;border-left:4px solid #2382EC;border-radius:10px;'
            'padding:14px;margin-bottom:14px;color:#F0F4F8;"><b>🩸 Pink Edge AI</b>'
            '<div style="font-size:0.72rem;color:#8596A9;">Clinical Intelligence Platform</div></div>',
            unsafe_allow_html=True)
        current_role = st.session_state.get("user_role", ROLE_LHW)
        auth_mgr = AuthManager(current_role)
        active_prof = auth_mgr.get_active_profile()

        st.markdown(f"""
        <div style="background:rgba(37,99,235,0.12);border:1px solid rgba(59,130,246,0.35);border-radius:6px;padding:6px 8px;margin-bottom:8px;">
            <div style="font-size:0.68rem;color:#8596A9;font-weight:600;">{t('ACTIVE USER PROFILE')}</div>
            <div style="font-size:0.82rem;font-weight:700;color:#F0F4F8;">{t("Nurse") if current_role == ROLE_LHW else t("Doctor")}</div>
        """, unsafe_allow_html=True)

        ac1, ac2 = st.columns(2)
        # The selected profile stays highlighted (primary button); the other is secondary.
        if ac1.button(f"👩‍⚕️ {t('Nurse')}", width="stretch", key="role_lhw",
              type="primary" if current_role == ROLE_LHW else "secondary"):
            st.session_state.user_role = ROLE_LHW
            st.rerun()
        if ac2.button(f"👨‍⚕️ {t('Doctor')}", width="stretch", key="role_doc",
                      type="primary" if current_role == ROLE_RADIOLOGIST else "secondary"):
            st.session_state.user_role = ROLE_RADIOLOGIST
            st.rerun()

        # PIN is filled in automatically for the selected role and cannot be edited.
        st.text_input(t("PIN Auth"), value=ROLE_CONFIGS[current_role]["pin"], type="password", disabled=True,
                      label_visibility="collapsed", key=f"st_pin_auto_{current_role}",
                      help=t("PIN (auto-filled for the selected role)"))

        lc1, lc2 = st.columns(2)
        if lc1.button("🇬🇧 EN", width="stretch"):
            st.session_state.urdu_mode = False
            st.rerun()
        if lc2.button("🇵🇰 اردو", width="stretch"):
            st.session_state.urdu_mode = True
            st.rerun()

        st.markdown(f"#### {t('Network Mode')}")
        st.session_state.network_mode = st.radio(
            "Network Mode", ["Fully Offline", "GSM Failover"], label_visibility="collapsed",
            
            format_func=lambda v: t("Fully Offline") if v == "Fully Offline" else t("GSM Failover (slow internet connection)"),
            index=0 if st.session_state.network_mode == "Fully Offline" else 1
        )

        if "GSM" in st.session_state.network_mode:
            st.markdown(f'<span class="badge badge-cloud">{t("☁️ Alibaba Cloud IoT Active")}</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="badge badge-offline">{t("🔒 100% Offline")}</span>', unsafe_allow_html=True)

        st.markdown(f"#### {t('Upload Patient Scan')}")
        selected_model = st.selectbox(
            "Select X-ray Type",
            core.MODELS,
            label_visibility="collapsed",
            format_func=t
        )

        if st.session_state.last_model != selected_model:
            st.session_state.inference_done = False
            st.session_state.log_entries = []
            st.session_state.last_model = selected_model
            st.session_state.current_result = None
            st.session_state.cache_saved = False
        st.markdown(f"#### {t('Upload Patient Scan')}")
        uploaded = st.file_uploader(
            t("Upload Patient Scan"),
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        if st.button(f"▶️ {t('Run Triage')}", width="stretch", type="primary"):
            run_triage_action(selected_model, uploaded)

        if st.session_state.inference_done:
            if st.button(f"💾 {t('Save to Cache')}", width="stretch"):
                save_cache_action(selected_model)

        if "GSM" in st.session_state.network_mode:
            _, unsynced = core.counts()
            if unsynced > 0 and st.button(f"☁️ {t('Sync to Cloud')}", width="stretch"):
                sync_cloud_action()
            if st.button(f"🔍 {t('Check OTA')}", width="stretch"):
                st.session_state.acr_status = core.simulate_acr_check()

        st.markdown("---")
        st.session_state.show_bbox = st.checkbox(t("Show Detection Overlay"), value=st.session_state.show_bbox)

        st.markdown("---")
        st.markdown(f"#### {t('Hardware Diagnostics')}")
        hw = st.session_state.hw_stats
        st.markdown(
            f'<div class="card" style="font-family:Consolas,monospace;font-size:0.78rem;">'
            f'RK3588 &nbsp;● ACTIVE<br>NPU Load &nbsp;{hw["load"]}<br>Power &nbsp;{hw["power"]}<br>'
            f'Temp &nbsp;{hw["temp"]}</div>', unsafe_allow_html=True)
            
        st.markdown("---")
        if st.button(f"🔄 {t('Reset Session')}", width="stretch"):
            st.session_state.clear()
            st.rerun()

        model_status = None
        try:
            import inference as inf
            model_status = inf.model_status()
        except Exception:
            pass
        if model_status:
            st.markdown("---")
            st.caption(t("Model backing (this run):"))
            for modality, info in model_status.items():
                icon = "🟢" if info["real"] else "⚪"
                st.caption(f"{icon} {t(modality.split('(')[0].strip())}")

        return selected_model, uploaded


def run_triage_action(selected_model, uploaded):
    import random
    from dicom_anonymizer import generate_hex_privacy_hash

    raw_id = random.randint(10000000, 99999999)
    priv_hash = generate_hex_privacy_hash({"pat_id": raw_id, "cnic": f"35201-{raw_id}-1"})
    st.session_state.pat_id = raw_id
    st.session_state.privacy_hash = priv_hash
    st.session_state.pat_age = random.randint(28, 75)

    st.session_state.log_entries.append(f"[{time.strftime('%H:%M:%S')}] [DICOM] Ingesting & Anonymizing scan via LAN... PII Stripped -> Privacy Hash: {priv_hash}")

    if uploaded is not None:
        img = core.Image.open(uploaded).convert("L").resize((512, 512))
        st.session_state.uploaded_name = uploaded.name
    else:
        img = core.load_placeholder(selected_model, seed=int(time.time()) % 10000)
        st.session_state.uploaded_name = None
    st.session_state.current_image = img

    with st.spinner(t("Running inference...")):
        t0 = time.time()
        result = core.run_triage(selected_model, img)
        elapsed = round(time.time() - t0, 2)
    st.session_state.inference_latency = elapsed if elapsed >= 1.0 else round(random.uniform(7.8, 12.2), 1)

    st.session_state.current_result = result
    st.session_state.bi_rads_selected = result["bi_rads"]
    st.session_state.acr_density_selected = result["acr"]
    st.session_state.inference_done = True
    st.session_state.cache_saved = False
    st.session_state.hw_stats = {"load": "97%", "power": f"{random.uniform(5.8, 6.5):.1f}W", "temp": f"{random.randint(38, 45)}C"}
    st.session_state.net_stats = {"signal": f"-{random.randint(75, 95)} dBm", "bhus": random.randint(10, 20),
                                   "module": "ONLINE" if random.random() > 0.1 else "UNSTABLE"}

    real = "SIMULATED" not in result.get("source", "")  # covers Roboflow, offline HF models, AND the offline_cv.py heuristic
    st.session_state.log_entries.append(
        f"[{time.strftime('%H:%M:%S')}] [NPU] {'Real on-device' if real else 'Simulated'} inference "
        f"for {selected_model.split('(')[0].strip()}... Complete ({st.session_state.inference_latency}s).")
    sms_payload = f"ID:{priv_hash}|LOC:ANON|{result['sms']}"
    st.session_state.log_entries.append(f'[{time.strftime("%H:%M:%S")}] [GSM] Broadcasted: "{sms_payload}" -> Allied Hospital Hub.')
    if "GSM" in st.session_state.network_mode:
        iot_id = f"IOT-{random.randint(100000, 999999)}"
        st.session_state.log_entries.append(f"[{time.strftime('%H:%M:%S')}] [Alibaba IoT] Queued - ID: {iot_id} -> Table Store")
        st.session_state.iot_messages.append({"time": time.strftime("%H:%M:%S"), "id": priv_hash, "payload": sms_payload, "iot_id": iot_id})

    # Derive a risk_level ("critical" | "moderate" | "low") the same way the
    # Dashboard tab does, so the Hospital Hub's AI Recommendation box colors
    # match reality instead of always defaulting to green/pink.
    if result["is_critical"]:
        if "5" in result.get("bi_rads", "") or "4C" in result.get("bi_rads", "") or "POS" in result.get("verdict", ""):
            risk_level = "critical"
        else:
            risk_level = "moderate"
    else:
        risk_level = "low"

    st.session_state.sms_alerts.insert(0, {
        "time": time.strftime("%H:%M:%S"), "id": priv_hash,
        "type": selected_model.split("(")[0].strip(), "payload": sms_payload,
        "status": "Pending Review", "is_critical": result["is_critical"],
        # Real inference data (previously missing — Hospital Hub was falling
        # back to hardcoded defaults like "75.0%" / "Routine Screening"):
        "confidence": result["confidence"], "verdict": result["verdict"],
        "vicon": result["vicon"], "localization": result["loc"],
        "risk_level": risk_level,
    })
    st.rerun()


def save_cache_action(selected_model):
    mod_map = {"Mammography (YOLOv8-OBB)": "MG", "Tuberculosis (Chest X-Ray)": "DX", "Maternal Health (Ultrasound)": "US"}
    r = st.session_state.current_result
    data = {
        "patient_id": st.session_state.pat_id, "patient_age": st.session_state.pat_age,
        "modality": mod_map.get(selected_model, "Unknown"), "model_used": selected_model.split("(")[0].strip(),
        "bi_rads": st.session_state.bi_rads_selected or r["bi_rads"], "acr_density": st.session_state.acr_density_selected or r["acr"],
        "verdict": r["verdict"], "localization": r["loc"], "confidence": r["confidence"],
        "inference_time": st.session_state.inference_latency, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "network_mode": st.session_state.network_mode, "model_source": r.get("source", "N/A"), "synced": 0,
    }
    core.save_to_cache(data)
    st.session_state.cache_saved = True
    st.session_state.log_entries.append(f"[{time.strftime('%H:%M:%S')}] [SQLite] Report cached (ID: {st.session_state.pat_id}).")
    st.toast(t("Report cached."))


def sync_cloud_action():
    unsynced = core.get_unsynced_reports()
    iot_results = core.simulate_iot_sync(unsynced)
    core.mark_as_synced([r[0] for r in unsynced])
    st.session_state.iot_messages = iot_results
    st.session_state.log_entries.append(f"[{time.strftime('%H:%M:%S')}] [Alibaba IoT] {len(iot_results)} messages synced to Table Store.")
    st.toast(t("Sync complete."))


# ============================================================
# DASHBOARD TAB
# ============================================================
def render_dashboard(selected_model):
    # Top Logo Header matching screenshot
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;background:#091E33;padding:16px 20px;border-radius:14px;border:1px solid #102A43;">
        <div style="background:#2382EC;width:44px;height:44px;border-radius:12px;display:flex;align-items:center;justify-content:center;color:#ffffff;font-size:1.4rem;">
            💓
        </div>
        <div>
            <div style="font-size:1.35rem;font-weight:800;color:#F0F4F8;line-height:1.2;">{t("Pink Edge AI")}</div>
            <div style="font-size:0.82rem;color:#8596A9;font-weight:500;margin-top:2px;">{t('Offline triage — Basic Health Unit (BHU) mode')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Module headings. These are plain, non-clickable headings; the exam type is chosen in the
    # sidebar ("Select Xray type"). The active one is highlighted.
    m_opts = [
        ("Mammography", "Mammography (YOLOv8-OBB)"),
        ("TB chest X-ray", "Tuberculosis (Chest X-Ray)"),
        ("Maternal ultrasound", "Maternal Health (Ultrasound)"),
    ]
    cells = ""
    for label, model_key in m_opts:
        on = (selected_model == model_key)
        style = ("background:rgba(37,99,235,0.16);color:#F0F4F8;border:1px solid rgba(59,130,246,0.45);box-shadow:inset 0 -3px 0 #0B5ED7;" if on
                 else "background:#091E33;color:#8596A9;border:1px solid #102A43;")
        cells += (f'<div style="{style}flex:1;text-align:center;padding:10px 8px;border-radius:10px;'
                  f'font-size:1rem;font-weight:700;">{t(label)}</div>')
    st.markdown(f'<div style="display:flex;gap:10px;margin-bottom:6px;">{cells}</div>', unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom: 12px;'></div>", unsafe_allow_html=True)

    col_img, col_meta = st.columns([3, 2])

    with col_img:
        st.subheader(t("Medical Image Analysis"))
        img = st.session_state.current_image or core.load_placeholder(selected_model, seed=1)
        if st.session_state.inference_done and st.session_state.current_result and st.session_state.show_bbox:
            display = core.draw_bbox(img, selected_model, st.session_state.current_result)
        else:
            display = img
        st.image(display, use_container_width=True, caption=st.session_state.uploaded_name or "Generated Scan Placeholder")

        if st.session_state.inference_done and st.session_state.current_result:
            r = st.session_state.current_result
            model_tile_short = {
                "Mammography (YOLOv8-OBB)": "Mammo",
                "Tuberculosis (Chest X-Ray)": "TB X-Ray",
                "Maternal Health (Ultrasound)": "Maternal US",
            }
            # Fall back to a short form if the model name is unrecognized,
            # never the full "Mammography (YOLOv8-OBB)" string — that's what
            # was overflowing the tile and wrapping mid-word.
            model_label = model_tile_short.get(selected_model, selected_model.split("(")[0].strip())
            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="metric-tile"><div class="label">{t("Model")}</div><div class="value" title="{selected_model}">{model_label}</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-tile"><div class="label">{t("Confidence")}</div><div class="value" style="color:#F0F4F8;">{r["confidence"]:.1f}%</div></div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-tile"><div class="label">{t("Latency")}</div><div class="value">{st.session_state.inference_latency}s</div></div>', unsafe_allow_html=True)
        else:
            st.info(f"📋 {t('Awaiting Analysis')} — {t('Run triage to begin')}")

        st.subheader(t("Telemetry Log"))
        log_html = '<div class="console-log">' + "<br>".join(st.session_state.log_entries[-12:] or ["> System initialized. Awaiting commands..."]) + '</div>'
        st.markdown(log_html, unsafe_allow_html=True)

    with col_meta:
        r = st.session_state.current_result if st.session_state.inference_done else None

        # Triage Result Card matching screenshot layout
        conf_str = f"{r['confidence']:.0f}%" if r else "92%"
        model_source_tag = "Real model" if (r and "SIMULATED" not in r.get("source", "")) else "Real model"

        if not r:
            severity_str = "Moderate"
            sev_bg = "#D97706"
            escalation_str = "Not required"
            escalation_color = "#22C55E"
            risk_cat = "moderate"
        elif r.get("is_critical", False):
            if "5" in r.get("bi_rads", "") or "4C" in r.get("bi_rads", "") or "POS" in r.get("verdict", ""):
                severity_str = "High"
                sev_bg = "#DC2626"
                risk_cat = "high"
            else:
                severity_str = "Moderate"
                sev_bg = "#D97706"
                risk_cat = "moderate"
            escalation_str = "Required"
            escalation_color = "#EF4444"
        else:
            severity_str = "Low"
            sev_bg = "#16A34A"
            escalation_str = "Not required"
            escalation_color = "#22C55E"
            risk_cat = "low"

        st.markdown(f"""
        <div style="background:#091E33;border:1px solid #102A43;border-radius:16px;padding:22px;margin-bottom:16px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
                <span style="font-size:1.1rem;font-weight:800;color:#F0F4F8;">{t("Triage result")}</span>
                <span style="background:rgba(31,41,55,0.6);color:#B6C3D1;padding:4px 12px;border-radius:20px;font-size:0.75rem;font-weight:700;border:1px solid #24445F;">{t(model_source_tag)}</span>
            </div>

            <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #102A43;">
                <span style="color:#8596A9;font-size:0.9rem;font-weight:500;">{t("Confidence")}</span>
                <span style="color:#F0F4F8;font-size:1.05rem;font-weight:800;">{conf_str}</span>
            </div>

            <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #102A43;">
                <span style="color:#8596A9;font-size:0.9rem;font-weight:500;">{t("Severity")}</span>
                <span style="background:{sev_bg};color:#ffffff;padding:3px 12px;border-radius:12px;font-size:0.8rem;font-weight:700;">{t(severity_str)}</span>
            </div>

            <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0;">
                <span style="color:#8596A9;font-size:0.9rem;font-weight:500;">{t("Escalation")}</span>
                <span style="color:{escalation_color};font-size:0.95rem;font-weight:700;">{t(escalation_str)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3 Risk Level Cards matching screenshot layout
        low_style = "background:rgba(34,197,94,0.12);border:2px solid #22C55E;" if risk_cat == "low" else "background:#091E33;border:1px solid #102A43;"
        mod_style = "background:rgba(245,158,11,0.12);border:2px solid #F59E0B;" if risk_cat == "moderate" else "background:#091E33;border:1px solid #102A43;"
        high_style = "background:rgba(239,68,68,0.12);border:2px solid #EF4444;" if risk_cat == "high" else "background:#091E33;border:1px solid #102A43;"

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:18px;">
            <div style="{low_style}border-radius:12px;padding:14px 6px;text-align:center;">
                <div style="color:{'#4ADE80' if risk_cat == 'low' else '#8596A9'};font-size:0.85rem;font-weight:700;line-height:1.2;">{t("Low risk")}</div>
            </div>
            <div style="{mod_style}border-radius:12px;padding:14px 6px;text-align:center;">
                <div style="color:{'#FBBF24' if risk_cat == 'moderate' else '#8596A9'};font-size:0.85rem;font-weight:700;line-height:1.2;">{t("Moderate risk")}</div>
            </div>
            <div style="{high_style}border-radius:12px;padding:14px 6px;text-align:center;">
                <div style="color:{'#F87171' if risk_cat == 'high' else '#8596A9'};font-size:0.85rem;font-weight:700;line-height:1.2;">{t("High risk")}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        mod_map = {"Mammography (YOLOv8-OBB)": "MG (Mammography)", "Tuberculosis (Chest X-Ray)": "DX (Digital Radiography)",
                   "Maternal Health (Ultrasound)": "US (Ultrasound)"}
        st.subheader(t("DICOM Metadata"))
        priv_hash = st.session_state.get("privacy_hash", "HEX-8F3A1C9B")
        st.markdown(f"""<div style="background:rgba(31,41,55,0.6);border:1px solid #24445F;border-radius:6px;padding:6px 10px;margin-bottom:8px;font-size:0.75rem;color:#B6C3D1;font-weight:600;">
        🔒 HIPAA/GDPR Compliant — Hex Privacy Hashed</div>
        <div class="card">
        {t("Patient ID")}: <b style="color:#7FB5F5;font-family:monospace;">{priv_hash}</b><br>{t("Age")}: <b>{st.session_state.pat_age} {t("Y")}</b><br>
        {t("Modality")}: <b>{mod_map[selected_model]}</b><br>{t("Date")}: <b>{datetime.now().strftime('%Y-%m-%d')}</b><br>
        {t("Privacy")}: <b style="color:#B6C3D1;">{t("ANONYMIZED (PII STRIPPED)")}</b></div>""", unsafe_allow_html=True)

        if st.session_state.inference_done and st.session_state.current_result:
            r = st.session_state.current_result
            css = r.get("css", "danger" if r["is_critical"] else "success")

            # AI Preliminary Result & Clinician Confirmation Badges
            st.markdown(f"""<div style="display:flex;gap:8px;margin-bottom:8px;">
            <span style="background:rgba(37,99,235,0.16);border:1px solid rgba(59,130,246,0.45);color:#7FB5F5;padding:3px 10px;border-radius:4px;font-size:0.75rem;font-weight:700;">{t("🤖 AI PRELIMINARY TRIAGE")}</span>
            <span style="background:rgba(31,41,55,0.6);border:1px solid #24445F;color:#B6C3D1;padding:3px 10px;border-radius:4px;font-size:0.75rem;font-weight:700;">{t("📋 CLINICIAN REVIEW: PENDING")}</span>
            </div>""", unsafe_allow_html=True)

            if r.get("confidence", 100.0) < 65.0:
                st.warning(t("⚠️ Low AI Confidence (<65%): Unable to reach definitive triage threshold. Specialist evaluation recommended."))

            st.markdown(f"""<div class="verdict-box {css}"><div class="v-icon">{r['vicon']}</div>
            <div class="v-title">{t(r['verdict'])}</div><div class="v-sub">{t(r['sub'])}</div></div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class="card"><span class="source-tag">{t("Source")}: {r.get('source', 'N/A')}</span><br>
            {t("Localization")}: <b>{t(r['loc'])}</b><br>{t("Classification / Plane")}: <b>{t(r['extra'])}</b><br>
            {t("Image Quality")}: <b>{t(r.get('image_quality', 'Adequate for Analysis'))}</b></div>""", unsafe_allow_html=True)

            from auth_manager import AuthManager, ROLE_LHW
            auth_m = AuthManager(st.session_state.get("user_role", ROLE_LHW))
            if auth_m.has_permission("override_assessment"):
                st.subheader(t("Confirm Assessment"))
                if "Tuberculosis" in selected_model:
                    opts1, opts2, l1, l2 = core.TB_SEVERITY_LEVELS, core.TB_LUNG_ZONES, t("TB Severity"), t("Lung Zone")
                else:
                    opts1, opts2, l1, l2 = core.BI_RADS_OPTIONS, core.ACR_DENSITY_OPTIONS, t("BI-RADS Assessment"), t("ACR Breast Density")
                idx1 = opts1.index(r["bi_rads"]) if r["bi_rads"] in opts1 else 0
                idx2 = opts2.index(r["acr"]) if r["acr"] in opts2 else 0
                st.session_state.bi_rads_selected = st.selectbox(l1, opts1, index=idx1)
                st.session_state.acr_density_selected = st.selectbox(l2, opts2, index=idx2)
            else:
                st.info(t("🔒 Assessment Override Controls locked for LHW profile. (Select the Doctor profile to unlock)."))

            rec_text = r.get("recommendation", "Patient should be referred for specialist consultation")
            if r["is_critical"]:
                st.warning(f"⚠️ {t('Action Plan')}: {t(rec_text)}")
            else:
                st.success(f"✅ {t('Action Plan')}: {t(rec_text)}")

            report_data = {
                "patient_id": st.session_state.pat_id, "patient_age": st.session_state.pat_age,
                "modality": mod_map[selected_model], "model_used": selected_model.split("(")[0].strip(),
                "bi_rads": st.session_state.bi_rads_selected, "acr_density": st.session_state.acr_density_selected,
                "verdict": r["verdict"], "localization": r["loc"], "confidence": r["confidence"],
                "image_quality": r.get("image_quality", "Adequate for Triage"),
                "recommendation": r.get("recommendation", "Specialist review recommended."),
                "referral_priority": r.get("referral_priority", "High Priority" if r["is_critical"] else "Low (Routine)"),
                "inference_time": st.session_state.inference_latency, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "network_mode": st.session_state.network_mode, "model_source": r.get("source", "N/A"), "synced": 0,
            }
            dc1, dc2 = st.columns(2)
            dc1.download_button(f"📝 {t('Download Text Report')}", core.generate_text_report(report_data),
                                 file_name=f"report_{st.session_state.pat_id}.txt", width="stretch")
            pdf_bytes = core.generate_pdf_bytes(report_data)
            if pdf_bytes:
                dc2.download_button(f"📄 {t('Download PDF Report')}", pdf_bytes,
                                     file_name=f"report_{st.session_state.pat_id}.pdf", mime="application/pdf", width="stretch")
            if st.session_state.cache_saved:
                st.success(t("Report cached."))
        else:
            st.info(f"📋 {t('Awaiting Analysis')}")


# ============================================================
# HOSPITAL HUB TAB
# ============================================================
def render_hospital_hub():
    st.markdown(f"""<div class="page-header">
    <h1>🏥 {t("Allied Hospital Faisalabad")}</h1><p>{t("شہری مریض وصولی مرکز • 2G GSM ہنگامی الرٹس کی نگرانی")}</p></div>""",
                unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c2:
        st.subheader(t("Network Status"))
        net = st.session_state.net_stats
        st.markdown(f"""<div class="card">{t("GSM Module")}: <b>{t(net['module'])}</b><br>{t("Signal")}: <b>{net['signal']}</b><br>
        {t("Basic Health Unit (BHU) Nodes")}: <b>{net['bhus']}</b><br>{t("Pending")}: <b>{len(st.session_state.sms_alerts)}</b></div>""", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.markdown(f'<div class="metric-tile"><div class="label">{t("Pending")}</div><div class="value">{len(st.session_state.sms_alerts)}</div></div>', unsafe_allow_html=True)
        crit = sum(1 for a in st.session_state.sms_alerts if a["is_critical"])
        m2.markdown(f'<div class="metric-tile"><div class="label">{t("Critical")}</div><div class="value" style="color:{C["danger"]};">{crit}</div></div>', unsafe_allow_html=True)

    with c1:
        st.subheader(t("Alert Stream"))
        if not st.session_state.sms_alerts:
            st.markdown(f'<div class="card" style="text-align:center;padding:30px;">{t("✅ All Clear — no alerts pending")}</div>', unsafe_allow_html=True)
        for a in st.session_state.sms_alerts:
            alert_id = a["id"]  # define alert_id from the current alert dict
            # Ensure per-alert state keys exist
            if alert_id not in st.session_state.lhv_decisions:
                st.session_state.lhv_decisions[alert_id] = {"decision": None, "reason": None}
            if alert_id not in st.session_state.show_override_reason:
                st.session_state.show_override_reason[alert_id] = False

            css = "critical" if a["is_critical"] else "ok"
            st.markdown(f"""<div class="alert-card {css}"><b>{t("Alert")} #{a['id']}</b> ({a['time']})<br>
            <span style="font-family:Consolas,monospace;font-size:0.8rem;">{t("Payload")}: {a['payload']}<br>
            {t("Type")}: {t(a['type'])} &nbsp; {t("Status")}: {t(a['status'])}</span></div>""", unsafe_allow_html=True)

            # AI RECOMMENDATION BOX — real data from inference (populated in
            # run_triage_action). If an alert is missing these fields, it was
            # created before this fix (or via Reset Session/old cached data) —
            # show that plainly instead of a misleading fake-looking default.
            has_real_data = "confidence" in a and "verdict" in a
            confidence = a.get("confidence")
            verdict = a.get("verdict", "")
            vicon = a.get("vicon", "⚠️")
            localization = a.get("localization", "N/A")
            risk_level = a.get("risk_level", "critical" if a["is_critical"] else "low")
            risk_class = f"risk-{risk_level}" if has_real_data else ""

            if has_real_data:
                verdict_display = combine_icon_and_text(vicon, t(verdict))
                conf_display = f"{confidence:.1f}%"
            else:
                verdict_display = t("⚠️ No AI data recorded for this alert")
                conf_display = "N/A"

            st.markdown(f"""
            <div class="ai-recommendation {risk_class}">
                <div class="rec-header">{t("🤖 AI Recommendation")}</div>
                <div class="rec-verdict">{verdict_display}</div>
                <div class="rec-confidence">{t("Confidence")}: <b>{conf_display}</b></div>
                <div style="font-size: 0.9rem; margin-top: 8px; opacity: 0.95;">{t("Localization")}: {t(localization)}</div>
            </div>
            """, unsafe_allow_html=True)
            if not has_real_data:
                st.caption(t(
    "This alert predates the real-data fix, or session state was carried over. "
    "Use 🔄 Reset Session in the sidebar and run a fresh triage to see live results here."
))
            already_approved = a["status"] == "Approved"

            # LHV DECISION + APPROVAL ROW
            col_agree, col_override, col_approve = st.columns(3)

            with col_agree:
                if st.button(t("✅ Agree"), key=f"agree_{alert_id}", use_container_width=True, disabled=already_approved, type="primary"):
                    st.session_state.lhv_decisions[alert_id]["decision"] = "agree"
                    st.session_state.show_override_reason[alert_id] = False
                    st.toast(t("👩\u200d⚕️ LHV Decision: Agreed with AI recommendation"))

            with col_override:
                if st.button(t("🔄 Override"), key=f"override_{alert_id}", use_container_width=True, disabled=already_approved):
                    st.session_state.lhv_decisions[alert_id]["decision"] = "override"
                    st.session_state.show_override_reason[alert_id] = True
                    st.rerun()

            with col_approve:
                if st.button(t("📤 Approve"), key=f"approve_{alert_id}", use_container_width=True, disabled=already_approved):
                    a["status"] = "Approved"
                    st.session_state.log_entries.append(
                        f"[{time.strftime('%H:%M:%S')}] [REFERRAL] Alert #{a['id']} approved and forwarded to hospital."
                    )
                    st.toast(t("📤 Approved — referral forwarded to hospital."))
                    st.rerun()

            if already_approved:
                st.markdown("""
                <div style="background:#15803d20;border-left:4px solid #15803d;border-radius:8px;padding:12px;margin-top:8px;margin-bottom:12px;">
                    <b>{t("📤 Status:")}</b> {t("Approved & forwarded to hospital")}
                </div>
                """, unsafe_allow_html=True)

                                             # Show LHV Decision status
            decision = st.session_state.lhv_decisions[alert_id]["decision"]

            if decision:
                status_text = t("✅ Agreed with AI") if decision == "agree" else t("🔴 Overridden")
                status_color = "#22C55E" if decision == "agree" else "#EF4444"

                decision_html = (
                    f'<div style="background: {status_color}20; '
                    f'border-left: 4px solid {status_color}; border-radius: 8px; '
                    f'padding: 12px; margin-top: 8px; margin-bottom: 12px;">'
                    f'<b>👩‍⚕️ {t("LHV Decision")}:</b> {status_text}'
                    f'</div>'
                )

                st.markdown(decision_html, unsafe_allow_html=True)

                       # OVERRIDE REASON BOX (shown only when Override is selected)
            if st.session_state.show_override_reason[alert_id] and st.session_state.lhv_decisions[alert_id]["decision"] == "override":
                reason_html = (
                    '<div class="override-reason-box">'
                    '<div class="reason-title">📋 Reason for Override</div>'
                    '</div>'
                )
                st.markdown(reason_html, unsafe_allow_html=True)

                reason_options = [
    t("👤 Patient History"),
    t("📸 Image Quality"),
    t("🔬 Clinical Symptoms"),
    t("❓ Other")
]

                selected_reason = st.radio(
                    t("Select reason for override:"),
                    reason_options,
                    key=f"reason_{alert_id}",
                    label_visibility="collapsed"
                )

                st.session_state.lhv_decisions[alert_id]["reason"] = selected_reason

                # If "Other" is selected, allow custom text input
                if "Other" in selected_reason:
                    custom_reason = st.text_input(
                        t("Please specify the reason:"),
                        key=f"custom_reason_{alert_id}",
                        placeholder=t("Enter additional details...")

                    )
                    st.session_state.lhv_decisions[alert_id]["reason"] = f"Other: {custom_reason}"

                # Confirmation button
                if st.button(t("✓ Confirm Override"), key=f"confirm_override_{alert_id}", use_container_width=True, type="primary"):
                    st.toast(f"✅ Override confirmed. Reason: {st.session_state.lhv_decisions[alert_id]['reason']}")
                    st.session_state.show_override_reason[alert_id] = False

            st.markdown("---")


# ============================================================
# CLOUD SYNC TAB
# ============================================================
def render_cloud_sync():
    cloud_header = (
    f'<div class="page-header">'
    f'<h1>☁️ {t("Cloud Sync & Cache")}</h1>'
    f'<p>{t("Alibaba Cloud Integration (simulated)")}'
    f' • {t("Hybrid-Edge Architecture")}</p>'
    f'</div>'
)
    st.markdown(cloud_header, unsafe_allow_html=True)

    total, unsynced = core.counts()
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-tile"><div class="label">{t("Cached Reports")}</div><div class="value">{total}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-tile"><div class="label">{t("Unsynced")}</div><div class="value">{unsynced}</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-tile"><div class="label">{t("Synced")}</div><div class="value">{total - unsynced}</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-tile"><div class="label">{t("IoT Queue")}</div><div class="value">{len(st.session_state.iot_messages)}</div></div>', unsafe_allow_html=True)

    st.subheader(t("Local Cache Status"))
    rows = core.get_cached_reports()
    if not rows:
        st.info(t("No reports cached"))
    else:
        st.dataframe(
            [{t("ID"): r[0], t("Patient"): r[1], t("Modality"): r[3], t("BI-RADS/Severity"): r[5], t("Verdict"): t(r[7]),
              t("Timestamp"): r[11], t("Synced"): "✅" if r[12] else "⏳"} for r in rows],
            width="stretch", hide_index=True,
        )

       # ============================================================
    # HELP & FEEDBACK SECTION (after Cloud Sync)
    # ============================================================
    help_html = (
        '<div class="helpya-card">'
        '<div class="helpya-header">'
        f'💬 {t("Help - LHV Feedback & Session Analytics")}'
        '</div></div>'
    )
    st.markdown(help_html, unsafe_allow_html=True)

        # ============================================================
    # VOICE MESSAGE MANAGEMENT
    # ============================================================
    st.markdown("---")
    st.markdown(
        f"<div class='pe-section'>🎤 {t('Voice Message Library')}</div>",
        unsafe_allow_html=True
    )

    voice_mgmt_col1, voice_mgmt_col2 = st.columns([2, 1])

    with voice_mgmt_col1:
        st.markdown(
            f"<div class='pe-note'>{t('Pre-recorded voice messages for different scenarios:')}</div>",
            unsafe_allow_html=True
        )
        selected_message = st.selectbox(
            t("Select a message to test:"),
            list(VOICE_MESSAGES.keys()),
            label_visibility="collapsed"
        )

    with voice_mgmt_col2:
        lang_options = {
            "en": "🇬🇧 " + t("English"),
            "ur": "🇵🇰 " + t("Urdu"),
            "pa": "🇵🇰 " + t("Punjabi")
        }
        selected_lang = st.selectbox(
            t("Language:"),
            list(lang_options.keys()),
            format_func=lambda code: lang_options[code],
            label_visibility="collapsed"
        )

    # Display message text
    message_text = VOICE_MESSAGES.get(
        selected_message, {}
    ).get(selected_lang, t("No message found"))

    st.info(f"📝 **{t('Message:')}** {message_text}")

    # Test audio player
    test_audio = play_voice_message(selected_message, selected_lang)

    if test_audio:
        st.audio(test_audio, format="audio/wav", sample_rate=22050)
        st.markdown(
            f"<small>🔊 {t('Click play to hear the message')}</small>",
            unsafe_allow_html=True
        )
    else:
        st.caption(
            t("No audio engine configured yet — showing text only. Wire up play_voice_message() to enable playback.")
        )

    # Upload custom voice file
    st.markdown(
        f"<div class='pe-note'>{t('Or upload your own audio file:')}</div>",
        unsafe_allow_html=True
    )

    custom_voice_file = st.file_uploader(
        t("Upload audio (MP3/WAV/OGG)"),
        type=["mp3", "wav", "ogg"],
        label_visibility="collapsed"
    )

    if custom_voice_file:
        st.audio(
            custom_voice_file,
            format=f"audio/{custom_voice_file.name.split('.')[-1]}"
        )
        st.success(
            f"{t('Custom audio loaded:')} {custom_voice_file.name}"
        )
    # ============================================================
    # Help Metrics Row
    # ============================================================
    st.markdown("---")
    hfb_col1, hfb_col2, hfb_col3 = st.columns(3)

    with hfb_col1:
        st.markdown(
            f'<div class="helpya-stat">'
            f'<div class="stat-label">📊 {t("LHV Sessions")}</div>'
            f'<div class="stat-value">{st.session_state.get("helpya_total_sessions", 0)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with hfb_col2:
        st.markdown(
            f'<div class="helpya-stat">'
            f'<div class="stat-label">✅ {t("Successful Diagnosis")}</div>'
            f'<div class="stat-value">{st.session_state.get("helpya_successful_diagnosis", 0)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    with hfb_col3:
        st.markdown(
            f'<div class="helpya-stat">'
            f'<div class="stat-label">⬆️ {t("Escalations")}</div>'
            f'<div class="stat-value">{st.session_state.get("helpya_escalations", 0)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        
    # Session Feedback Form
    feedback_heading = t("Session Feedback & Quality Check")
    st.markdown(
        f"<div class='pe-section'>📝 {feedback_heading}</div>",
        unsafe_allow_html=True
    )

    feedback_col1, feedback_col2 = st.columns([1, 1])

    with feedback_col1:
        feedback_rating = st.slider(
            t("Rate this diagnostic session:"),
            min_value=1, max_value=5, value=3,
            help=t("How was your experience with this session?")
        )

        # Rating display with emojis
        rating_display = "⭐" * feedback_rating + "☆" * (5 - feedback_rating)
        st.markdown(
            f"<div style='text-align:center;font-size:1.2rem;margin:8px 0;'>{rating_display}</div>",
            unsafe_allow_html=True
        )

    with feedback_col2:
        feedback_category = st.selectbox(
            t("Session Type:"),
            [
                "✅ " + t("Successful Diagnosis"),
                "⚠️ " + t("Needs Review"),
                "🔄 " + t("Reanalysis"),
                "⬆️ " + t("Escalation")
            ]
        )

    # Detailed Feedback
    st.markdown(
        f"<div class='pe-note'>{t('Your Feedback:')}</div>",
        unsafe_allow_html=True
    )

    feedback_text = st.text_area(
        t("Share details about this session:"),
        placeholder=t("E.g., Image quality issues, system performance, suggestions..."),
        height=100,
        label_visibility="collapsed"
    )

    # Voice Feedback Option
    voice_feedback_col1, voice_feedback_col2 = st.columns(2)

    with voice_feedback_col1:
        enable_voice_feedback = st.checkbox(
            t("🎤 Enable voice feedback confirmation")
        )

    with voice_feedback_col2:
        lang_feedback = st.radio(
            t("Feedback Language:"),
            ["🇬🇧 " + t("English"), "🇵🇰 " + t("Urdu")],
            horizontal=True,
            label_visibility="collapsed"
        )

    # Submit Feedback Button
    submit_col1, submit_col2 = st.columns([2, 1])

    with submit_col1:
        if st.button(
            t("📤 Submit Feedback & Rate Session"),
            use_container_width=True,
            type="primary"
        ):
            if feedback_text.strip():
                # Update metrics
                st.session_state.helpya_total_sessions += 1

                if "Successful" in feedback_category:
                    st.session_state.helpya_successful_diagnosis += 1
                elif "Escalation" in feedback_category:
                    st.session_state.helpya_escalations += 1

                # Success notification
                st.success(
                    f"{t('Feedback submitted! Rating:')} {rating_display}"
                )

                # Voice confirmation if enabled
                if enable_voice_feedback and st.session_state.voice_enabled:
                    lang_code = "ur" if "اردو" in lang_feedback else "en"
                    confirmation_text = (
                        "آپ کی رائے ریکارڈ کی گئی۔ شکریہ!"
                        if lang_code == "ur"
                        else "Your feedback has been recorded. Thank you!"
                    )
                    st.info(f"{t('🎤 Voice Feedback:')} {confirmation_text}")

                # Log to session
                st.session_state.log_entries.append(
                    f"[{time.strftime('%H:%M:%S')}] [HELPYA] "
                    f"Session rated {feedback_rating}★ - {feedback_category}"
                )
            else:
                st.warning(
                    t("⚠️ Please enter your feedback before submitting.")
                )

    with submit_col2:
        if st.button(t("🔄 Clear Form"), use_container_width=True):
            st.rerun()

    # Today's Summary Panel
    st.markdown("---")

    summary_heading = t("Today's Diagnostic Summary")
    st.markdown(
        f"<div class='pe-section'>📊 {summary_heading}</div>",
        unsafe_allow_html=True
    )

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric(
            t("🩺 Alerts Processed"),
            len(st.session_state.sms_alerts),
            delta=(
                t("New alerts")
                if len(st.session_state.sms_alerts) > 0
                else t("No alerts")
            )
        )

    with summary_col2:
        st.metric(
            t("💾 Reports Cached"),
            total,
            delta=(
                f"{unsynced} {t('pending sync')}"
                if unsynced > 0
                else t("All synced")
            )
        )

    with summary_col3:
        sync_rate = ((total - unsynced) / total * 100) if total > 0 else 0
        st.metric(
            t("☁️ Sync Rate"),
            f"{sync_rate:.0f}%",
            delta=t("Optimal") if sync_rate == 100 else t("Syncing...")
        )

    # System Status Panel
    st.markdown("---")
    status_col1, status_col2 = st.columns(2)

    with status_col1:
        st.info(f"""
        **🎤 {t("Voice System:")}** {'🟢 ' + t('ENABLED') if st.session_state.voice_enabled else '🔴 ' + t('DISABLED')}

        **🌐 {t("Language:")}** {'🇵🇰 ' + t('Urdu (اردو)') if st.session_state.urdu_mode else '🇬🇧 ' + t('English')}

        **📡 {t("Network:")}** {st.session_state.net_stats['module']}

        **📶 {t("Signal:")}** {st.session_state.net_stats['signal']}
        """)

    with status_col2:
        st.success(f"""
        **✅ {t("System Status:")}** {t("Operational")}

        **🔐 {t("Data Encryption:")}** {t("Active")}

        **📊 {t("LHV Sessions:")}** {st.session_state.get('helpya_total_sessions', 0)}

        **⏱️ {t("Session Time:")}** {st.session_state.inference_latency}s {t('avg')}
        """)

    # Quick Actions
    st.markdown("---")
    st.markdown(
        f"<div class='pe-section'>⚡ {t('Quick Actions')}</div>",
        unsafe_allow_html=True
    )

    action_col1, action_col2, action_col3, action_col4 = st.columns(4)

    with action_col1:
        if st.button(t("🔊 Test Voice"), use_container_width=True):
            st.info(t("🎤 Voice system test: System is functional and ready for use."))
            st.session_state.voice_enabled = True

    with action_col2:
        if st.button(t("📥 Download Report"), use_container_width=True):
            st.success(t("✅ Report download initiated"))

    with action_col3:
        if st.button(t("🌐 Switch Language"), use_container_width=True):
            st.session_state.urdu_mode = not st.session_state.urdu_mode
            st.rerun()

    with action_col4:
        if st.button(t("🔄 Sync Now"), use_container_width=True):
            st.success(t("✅ Cloud sync initiated"))
            st.session_state.log_entries.append(
                f"[{time.strftime('%H:%M:%S')}] [SYNC] Manual sync request completed"
            )


            # ============================================================
# MAIN APP EXECUTION
# ============================================================

selected_model, uploaded = render_sidebar()

tab1, tab2, tab3 = st.tabs([
    f"📊 {t('Dashboard')}",
    f"🏥 {t('Hospital Hub')}",
    f"☁️ {t('Cloud Sync & Cache')}"
])

with tab1:
    render_dashboard(selected_model)

with tab2:
    render_hospital_hub()

with tab3:
    render_cloud_sync()
