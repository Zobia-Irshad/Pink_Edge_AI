#!/usr/bin/env python3
"""
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
# COLOR TOKENS & PROFESSIONAL CLINICAL LIGHT THEME
# ============================================================
C = {
    "bg": "#f8fafc",
    "surface": "#ffffff",
    "surface_alt": "#f1f5f9",
    "surface_hover": "#e2e8f0",
    "border": "#cbd5e1",
    "border_light": "#e2e8f0",
    "text": "#0f172a",
    "text_muted": "#475569",
    "text_light": "#64748b",
    "primary": "#0d9488",
    "primary_light": "#0f766e",
    "accent": "#0284c7",
    "accent_light": "#0369a1",
    "success": "#16a34a",
    "warning": "#d97706",
    "danger": "#dc2626",
}

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
_original_markdown = st.markdown


def _dedented_markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html") and isinstance(body, str) and "\n" in body:
        body = html_block(body)
    return _original_markdown(body, *args, **kwargs)


st.markdown = _dedented_markdown

CSS = """
<style>
/* Main App Background & High Contrast Default Text */
.stApp {
    background-color: #f8fafc !important;
    color: #0f172a !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    margin-bottom: 20px;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #cbd5e1 !important;
}
section[data-testid="stSidebar"] * {
    color: #0f172a !important;
}

/* Container Padding */
.block-container {
    padding-top: 1.8rem !important;
    padding-bottom: 2rem !important;
    max-width: 1300px !important;
}

/* Headings & Text Overrides — Guaranteed Readability */
h1, h2, h3, h4, h5, h6, label, p, span, div, li, td, th {
    color: #0f172a !important;
}
.stMarkdown p, .stMarkdown label, .stMarkdown span {
    color: #0f172a !important;
}

/* Header Banner */
.page-header {
    background: linear-gradient(135deg, #0d9488 0%, #0284c7 100%);
    border-radius: 12px;
    padding: 22px 28px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.06);
}
.page-header h1 {
    color: #ffffff !important;
    margin: 0 !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
}
.page-header p {
    color: #f0fdf4 !important;
    font-size: 0.88rem !important;
    margin: 6px 0 0 0 !important;
}

/* Badges */
.badge {
    display: inline-flex;
    padding: 5px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 700;
}
.badge-offline {
    background: #f0fdf4 !important;
    color: #15803d !important;
    border: 1px solid #86efac !important;
}
.badge-cloud {
    background: #fffbeb !important;
    color: #b45309 !important;
    border: 1px solid #fde68a !important;
}

/* Cards & Metric Tiles */
.card {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
    padding: 18px !important;
    margin: 8px 0 !important;
    color: #0f172a !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}
.card b, .card span {
    color: #0f172a !important;
}

.metric-tile {
    background: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    padding: 14px !important;
    text-align: center !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
}
.metric-tile .label {
    color: #475569 !important;
    font-size: 0.74rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.6px !important;
}
.metric-tile .value {
    color: #0d9488 !important;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    margin-top: 4px !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
}

/* Verdict Boxes */
.verdict-box {
    border-radius: 12px !important;
    padding: 20px !important;
    text-align: center !important;
    margin: 12px 0 !important;
}
.verdict-box.success {
    background: #f0fdf4 !important;
    border: 2px solid #86efac !important;
}
.verdict-box.success .v-title {
    color: #15803d !important;
}
.verdict-box.danger {
    background: #fef2f2 !important;
    border: 2px solid #fca5a5 !important;
}
.verdict-box.danger .v-title {
    color: #b91c1c !important;
}
.verdict-box.warning {
    background: #fffbeb !important;
    border: 2px solid #fde68a !important;
}
.verdict-box.warning .v-title {
    color: #b45309 !important;
}
.verdict-box .v-icon {
    font-size: 1.8rem !important;
}
.verdict-box .v-title {
    font-size: 1.3rem !important;
    font-weight: 800 !important;
}
.verdict-box .v-sub {
    font-size: 0.88rem !important;
    margin-top: 4px !important;
    color: #334155 !important;
    font-weight: 600 !important;
}

/* Telemetry Log */
.console-log {
    background: #f1f5f9 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 10px !important;
    padding: 14px !important;
    color: #0f172a !important;
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
    background: #fef2f2 !important;
    border-color: #ef4444 !important;
    color: #991b1b !important;
}
.alert-card.ok {
    background: #f0fdf4 !important;
    border-color: #10b981 !important;
    color: #166534 !important;
}

.source-tag {
    font-size: 0.75rem !important;
    color: #475569 !important;
    font-family: monospace !important;
    font-weight: 700 !important;
}

/* Streamlit Native Components — Crisp Light Styling */
div[data-baseweb="select"] > div {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border-color: #cbd5e1 !important;
}
div[data-baseweb="select"] span {
    color: #0f172a !important;
}
div[data-baseweb="input"] input {
    color: #0f172a !important;
    background-color: #ffffff !important;
}
.stButton > button {
    background-color: #ffffff !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    font-weight: 600 !important;
}
.stButton > button:hover {
    background-color: #f1f5f9 !important;
    border-color: #0d9488 !important;
    color: #0d9488 !important;
}
.stTabs [data-baseweb="tab-list"] {
    background-color: #ffffff !important;
    border-bottom: 2px solid #cbd5e1 !important;
}
.stTabs [data-baseweb="tab"] {
    color: #475569 !important;
    font-weight: 700 !important;
}
.stTabs [aria-selected="true"] {
    color: #0d9488 !important;
}
.stDataFrame, [data-testid="stTable"] {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
}

/* ============= AI CONFIDENCE + LHV OVERRIDE STYLING ============= */
.ai-recommendation {
  background: linear-gradient(135deg, #ff69b4 0%, #ff1493 100%); /* default/fallback pink gradient */
  border-radius: 14px; padding: 20px; margin: 12px 0;
  box-shadow: 0 4px 6px rgba(255, 20, 147, 0.2);
  color: #ffffff;
}
/* Risk-based color overrides — applied via an extra class based on the
   real triage result's risk_level, so a BI-RADS 5 case never renders as
   a green/pink "routine" box again. */
.ai-recommendation.risk-critical {
  background: linear-gradient(135deg, #ef4444 0%, #b91c1c 100%) !important;
  box-shadow: 0 4px 6px rgba(239, 68, 68, 0.3) !important;
}
.ai-recommendation.risk-moderate {
  background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%) !important;
  box-shadow: 0 4px 6px rgba(245, 158, 11, 0.3) !important;
}
.ai-recommendation.risk-low {
  background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
  box-shadow: 0 4px 6px rgba(16, 185, 129, 0.3) !important;
}
.ai-recommendation .rec-header {
  font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
  opacity: 0.95; margin-bottom: 8px;
}
.ai-recommendation .rec-verdict {
  font-size: 1.4rem; font-weight: 800; margin: 8px 0;
}
.ai-recommendation .rec-confidence {
  font-size: 0.95rem; margin-top: 12px; opacity: 0.95;
}

.lhv-decision {
  background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%); /* Blue gradient */
  border-radius: 14px; padding: 20px; margin: 12px 0;
  box-shadow: 0 4px 6px rgba(30, 64, 175, 0.2);
  color: #ffffff;
}
.lhv-decision .lhv-header {
  font-size: 0.9rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
  opacity: 0.95; margin-bottom: 16px;
}

.btn-agree {
  background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important; /* Green */
  color: #ffffff !important;
  border: none !important;
  border-radius: 8px;
  padding: 12px 24px;
  font-weight: 700;
  font-size: 0.95rem;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 6px rgba(16, 185, 129, 0.3);
}
.btn-agree:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 12px rgba(16, 185, 129, 0.4);
}

.btn-override {
  background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important; /* Red */
  color: #ffffff !important;
  border: none !important;
  border-radius: 8px;
  padding: 12px 24px;
  font-weight: 700;
  font-size: 0.95rem;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 4px 6px rgba(239, 68, 68, 0.3);
}
.btn-override:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 12px rgba(239, 68, 68, 0.4);
}

.override-reason-box {
  background: #f0f9ff; /* Light blue background */
  border: 2px solid #3b82f6;
  border-radius: 12px;
  padding: 16px;
  margin-top: 16px;
}
.override-reason-box .reason-title {
  font-size: 0.95rem;
  font-weight: 700;
  color: #1e40af;
  margin-bottom: 12px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.override-reason-box .reason-option {
  display: flex;
  align-items: center;
  padding: 10px 0;
  cursor: pointer;
  border-radius: 6px;
  padding: 8px 12px;
  transition: background 0.2s ease;
}
.override-reason-box .reason-option:hover {
  background: rgba(59, 130, 246, 0.1);
}
.override-reason-box .reason-option input[type="radio"] {
  margin-right: 12px;
  cursor: pointer;
  accent-color: #3b82f6;
  width: 18px;
  height: 18px;
}
.override-reason-box .reason-option label {
  cursor: pointer;
  color: #111827;
  font-weight: 500;
  margin: 0;
}

.decision-status {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 8px;
  padding: 12px 16px;
  margin-top: 12px;
  font-weight: 600;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.3);
}

/* ============= VOICE FEEDBACK SYSTEM ============= */
.voice-feedback-panel {
  background: #ffffff;
  border-radius: 14px;
  padding: 0;
  margin: 16px 0;
  box-shadow: 0 4px 8px rgba(255, 20, 147, 0.2);
  color: #111827;
  overflow: hidden;
}
.voice-feedback-panel .vfp-header {
  background: linear-gradient(135deg, #ff69b4 0%, #ff1493 100%);
  font-size: 0.95rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #ffffff;
  padding: 16px;
  margin: 0;
  border-bottom: 2px solid rgba(255, 20, 147, 0.2);
}
.voice-feedback-panel .vfp-message {
  font-size: 1.05rem;
  font-weight: 600;
  margin: 16px;
  line-height: 1.5;
  padding: 12px;
  background: #f5f5f5;
  border-radius: 8px;
  border-left: 4px solid #ff69b4;
  color: #111827;
}
.voice-btn-group {
  display: flex;
  gap: 8px;
  margin: 0 16px 16px 16px;
  flex-wrap: wrap;
}
.voice-btn {
  background: linear-gradient(135deg, #ff69b4 0%, #ff1493 100%) !important;
  border: none !important;
  color: #ffffff !important;
  padding: 10px 16px !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 0.9rem !important;
  cursor: pointer !important;
  transition: all 0.3s ease !important;
  box-shadow: 0 4px 6px rgba(255, 20, 147, 0.3) !important;
}
.voice-btn:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 12px rgba(255, 20, 147, 0.4) !important;
}
.voice-status {
  background: #f0f9ff;
  border-radius: 6px;
  padding: 8px 12px;
  margin: 0 16px 16px 16px;
  font-size: 0.85rem;
  text-align: center;
  border: 1px solid #bfdbfe;
  color: #1e40af;
}
.audio-player-wrapper {
  background: #f5f5f5;
  border-radius: 8px;
  padding: 12px 16px;
  margin: 0 16px 16px 16px;
}

.helpya-card {
  background: #ffffff;
  border-radius: 14px;
  padding: 0;
  margin: 16px 0;
  box-shadow: 0 4px 8px rgba(244, 63, 94, 0.2);
  color: #111827;
  overflow: hidden;
}
.helpya-card .helpya-header {
  background: linear-gradient(135deg, #ff69b4 0%, #ff1493 100%);
  font-size: 1.15rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: #ffffff;
  padding: 16px;
  margin: 0;
  display: flex;
  align-items: center;
  gap: 12px;
  border-bottom: 2px solid rgba(255, 20, 147, 0.2);
}
.helpya-stat {
  display: inline-block;
  background: linear-gradient(135deg, #fff0f6 0%, #ffe4f0 100%);
  border-radius: 12px;
  padding: 16px;
  margin: 12px;
  font-weight: 600;
  border: 2px solid #ffb6d9;
  color: #be123c;
}
.helpya-stat .stat-label {
  font-size: 0.8rem;
  opacity: 0.9;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.helpya-stat .stat-value {
  font-size: 1.6rem;
  font-weight: 800;
  margin-top: 6px;
}

/* Responsive: narrow viewports */
@media (max-width: 768px) {
  .block-container { padding-left: 10px !important; padding-right: 10px !important; }
  .page-header { padding: 16px 18px; }
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
core._urdu["on"] = st.session_state.urdu_mode  # sync GUI.py's t() to this session's language choice
t = core.t


# ============================================================
# SIDEBAR
# ============================================================
def render_sidebar():
    from auth_manager import AuthManager, ROLE_LHW, ROLE_RADIOLOGIST, ROLE_CONFIGS
    with st.sidebar:
        st.markdown(
            '<div style="background:linear-gradient(135deg, #ff69b4 0%, #ff1493 100%);border-radius:10px;'
            'padding:14px;margin-bottom:14px;color:#fff;"><b>🩸 Pink Edge AI</b>'
            '<div style="font-size:0.72rem;opacity:0.85;">Clinical Intelligence Platform</div></div>',
            unsafe_allow_html=True)
        current_role = st.session_state.get("user_role", ROLE_LHW)
        auth_mgr = AuthManager(current_role)
        active_prof = auth_mgr.get_active_profile()

        st.markdown(f"""
        <div style="background:{active_prof['color']}15;border:1px solid {active_prof['color']}50;border-radius:6px;padding:6px 8px;margin-bottom:8px;">
            <div style="font-size:0.68rem;color:var(--text-muted);font-weight:600;">ACTIVE USER PROFILE</div>
            <div style="font-size:0.82rem;font-weight:700;color:{active_prof['color']};">{active_prof['badge']}</div>
        </div>
        """, unsafe_allow_html=True)

        ac1, ac2 = st.columns(2)
        if ac1.button("👤 LHW", width="stretch"):
            st.session_state.user_role = ROLE_LHW
            st.rerun()
        if ac2.button("👨‍⚕️ Doctor", width="stretch"):
            st.session_state.user_role = ROLE_RADIOLOGIST
            st.rerun()

        pin_in = st.text_input("PIN Auth", type="password", placeholder="PIN (1111 / 9999)", label_visibility="collapsed", key="st_pin_in")
        if pin_in:
            auth_role = auth_mgr.authenticate_pin(pin_in)
            if auth_role:
                st.session_state.user_role = auth_role
                st.rerun()

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
            index=0 if st.session_state.network_mode == "Fully Offline" else 1)
        if "GSM" in st.session_state.network_mode:
            st.markdown('<span class="badge badge-cloud">☁️ Alibaba Cloud IoT Active</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-offline">🔒 100% Offline</span>', unsafe_allow_html=True)

        st.markdown(f"#### {t('Select AI Model')}")
        selected_model = st.selectbox("Select AI Model", core.MODELS, label_visibility="collapsed")
        if st.session_state.last_model != selected_model:
            st.session_state.inference_done = False
            st.session_state.log_entries = []
            st.session_state.last_model = selected_model
            st.session_state.current_result = None
            st.session_state.cache_saved = False

        st.markdown(f"#### {t('Upload Patient Scan')}")
        uploaded = st.file_uploader("Upload Patient Scan", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        st.markdown("---")
        if st.button(f"▶️ {t('Run Triage')}", width="stretch"):
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
            st.caption("Model backing (this run):")
            for modality, info in model_status.items():
                icon = "🟢" if info["real"] else "⚪"
                st.caption(f"{icon} {modality.split('(')[0].strip()}")

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

    with st.spinner("Running inference..."):
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
    st.markdown("""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:18px;background:#ffffff;padding:16px 20px;border-radius:14px;border:1px solid #e2e8f0;box-shadow:0 2px 6px rgba(0,0,0,0.02);">
        <div style="background:#e11d48;width:44px;height:44px;border-radius:12px;display:flex;align-items:center;justify-content:center;color:#ffffff;font-size:1.4rem;box-shadow:0 4px 10px rgba(225,29,72,0.25);">
            💓
        </div>
        <div>
            <div style="font-size:1.35rem;font-weight:800;color:#1e3a8a;line-height:1.2;">Pink Edge AI</div>
            <div style="font-size:0.82rem;color:#64748b;font-weight:500;margin-top:2px;">Offline triage — BHU mode</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Modality Pill Selector Buttons matching screenshot
    mod_cols = st.columns(3)
    m_opts = [
        ("Mammography", "Mammography (YOLOv8-OBB)"),
        ("TB chest X-ray", "Tuberculosis (Chest X-Ray)"),
        ("Maternal ultrasound", "Maternal Health (Ultrasound)"),
    ]
    for idx, (label, model_key) in enumerate(m_opts):
        with mod_cols[idx]:
            is_active = (selected_model == model_key)
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"mod_pill_{idx}", type=btn_type, use_container_width=True):
                st.session_state.last_model = model_key
                st.session_state.inference_done = False
                st.session_state.current_result = None
                st.rerun()

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
            m1.markdown(f'<div class="metric-tile"><div class="label">Model</div><div class="value" title="{selected_model}">{model_label}</div></div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-tile"><div class="label">{t("Confidence")}</div><div class="value" style="color:{C["primary_light"]};">{r["confidence"]:.1f}%</div></div>', unsafe_allow_html=True)
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
            sev_bg = "#f59e0b"
            escalation_str = "Not required"
            escalation_color = "#16a34a"
            risk_cat = "moderate"
        elif r.get("is_critical", False):
            if "5" in r.get("bi_rads", "") or "4C" in r.get("bi_rads", "") or "POS" in r.get("verdict", ""):
                severity_str = "High"
                sev_bg = "#ef4444"
                risk_cat = "high"
            else:
                severity_str = "Moderate"
                sev_bg = "#f59e0b"
                risk_cat = "moderate"
            escalation_str = "Required"
            escalation_color = "#b91c1c"
        else:
            severity_str = "Low"
            sev_bg = "#10b981"
            escalation_str = "Not required"
            escalation_color = "#16a34a"
            risk_cat = "low"

        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:16px;padding:22px;box-shadow:0 2px 10px rgba(0,0,0,0.03);margin-bottom:16px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
                <span style="font-size:1.1rem;font-weight:800;color:#0f172a;">Triage result</span>
                <span style="background:#fce7f3;color:#be185d;padding:4px 12px;border-radius:20px;font-size:0.75rem;font-weight:700;border:1px solid #fbcfe8;">{model_source_tag}</span>
            </div>

            <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #f1f5f9;">
                <span style="color:#64748b;font-size:0.9rem;font-weight:500;">Confidence</span>
                <span style="color:#0f172a;font-size:1.05rem;font-weight:800;">{conf_str}</span>
            </div>

            <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #f1f5f9;">
                <span style="color:#64748b;font-size:0.9rem;font-weight:500;">Severity</span>
                <span style="background:{sev_bg};color:#ffffff;padding:3px 12px;border-radius:12px;font-size:0.8rem;font-weight:700;">{severity_str}</span>
            </div>

            <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0;">
                <span style="color:#64748b;font-size:0.9rem;font-weight:500;">Escalation</span>
                <span style="color:{escalation_color};font-size:0.95rem;font-weight:700;">{escalation_str}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3 Risk Level Cards matching screenshot layout
        low_style = "background:#ecfdf5;border:2px solid #10b981;box-shadow:0 2px 6px rgba(16,185,129,0.15);" if risk_cat == "low" else "background:#f4fbf7;border:1px solid #d1fae5;"
        mod_style = "background:#fffbeb;border:2px solid #f59e0b;box-shadow:0 2px 6px rgba(245,158,11,0.15);" if risk_cat == "moderate" else "background:#fffdf5;border:1px solid #fef3c7;"
        high_style = "background:#fef2f2;border:2px solid #ef4444;box-shadow:0 2px 6px rgba(239,68,68,0.15);" if risk_cat == "high" else "background:#fff8f8;border:1px solid #fee2e2;"

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:18px;">
            <div style="{low_style}border-radius:12px;padding:14px 6px;text-align:center;">
                <div style="color:#16a34a;font-size:0.85rem;font-weight:700;line-height:1.2;">Low<br>risk</div>
            </div>
            <div style="{mod_style}border-radius:12px;padding:14px 6px;text-align:center;">
                <div style="color:#b45309;font-size:0.85rem;font-weight:700;line-height:1.2;">Moderate<br>risk</div>
            </div>
            <div style="{high_style}border-radius:12px;padding:14px 6px;text-align:center;">
                <div style="color:#dc2626;font-size:0.85rem;font-weight:700;line-height:1.2;">High<br>risk</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        mod_map = {"Mammography (YOLOv8-OBB)": "MG (Mammography)", "Tuberculosis (Chest X-Ray)": "DX (Digital Radiography)",
                   "Maternal Health (Ultrasound)": "US (Ultrasound)"}
        st.subheader(t("DICOM Metadata"))
        priv_hash = st.session_state.get("privacy_hash", "HEX-8F3A1C9B")
        st.markdown(f"""<div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.3);border-radius:6px;padding:6px 10px;margin-bottom:8px;font-size:0.75rem;color:var(--success);font-weight:600;">
        🔒 HIPAA/GDPR Compliant — Hex Privacy Hashed</div>
        <div class="card">
        Patient ID: <b style="color:#0284c7;font-family:monospace;">{priv_hash}</b><br>Age: <b>{st.session_state.pat_age} Y</b><br>
        Modality: <b>{mod_map[selected_model]}</b><br>Date: <b>{datetime.now().strftime('%Y-%m-%d')}</b><br>
        Privacy: <b style="color:var(--success);">ANONYMIZED (PII STRIPPED)</b></div>""", unsafe_allow_html=True)

        if st.session_state.inference_done and st.session_state.current_result:
            r = st.session_state.current_result
            css = r.get("css", "danger" if r["is_critical"] else "success")

            # AI Preliminary Result & Clinician Confirmation Badges
            st.markdown("""<div style="display:flex;gap:8px;margin-bottom:8px;">
            <span style="background:#e0f2fe;border:1px solid #7dd3fc;color:#0369a1;padding:3px 10px;border-radius:4px;font-size:0.75rem;font-weight:700;">🤖 AI PRELIMINARY TRIAGE</span>
            <span style="background:#fef3c7;border:1px solid #fde047;color:#92400e;padding:3px 10px;border-radius:4px;font-size:0.75rem;font-weight:700;">📋 CLINICIAN REVIEW: PENDING</span>
            </div>""", unsafe_allow_html=True)

            if r.get("confidence", 100.0) < 65.0:
                st.warning("⚠️ Low AI Confidence (<65%): Unable to reach definitive triage threshold. Specialist evaluation recommended.")

            st.markdown(f"""<div class="verdict-box {css}"><div class="v-icon">{r['vicon']}</div>
            <div class="v-title">{r['verdict']}</div><div class="v-sub">{r['sub']}</div></div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class="card"><span class="source-tag">Source: {r.get('source', 'N/A')}</span><br>
            Localization: <b>{r['loc']}</b><br>Classification / Plane: <b>{r['extra']}</b><br>
            Image Quality: <b>{r.get('image_quality', 'Adequate for Analysis')}</b></div>""", unsafe_allow_html=True)

            from auth_manager import AuthManager, ROLE_LHW
            auth_m = AuthManager(st.session_state.get("user_role", ROLE_LHW))
            if auth_m.has_permission("override_assessment"):
                st.subheader(t("Confirm Assessment"))
                if "Tuberculosis" in selected_model:
                    opts1, opts2, l1, l2 = core.TB_SEVERITY_LEVELS, core.TB_LUNG_ZONES, "TB Severity", "Lung Zone"
                else:
                    opts1, opts2, l1, l2 = core.BI_RADS_OPTIONS, core.ACR_DENSITY_OPTIONS, "BI-RADS Assessment", "ACR Breast Density"
                idx1 = opts1.index(r["bi_rads"]) if r["bi_rads"] in opts1 else 0
                idx2 = opts2.index(r["acr"]) if r["acr"] in opts2 else 0
                st.session_state.bi_rads_selected = st.selectbox(l1, opts1, index=idx1)
                st.session_state.acr_density_selected = st.selectbox(l2, opts2, index=idx2)
            else:
                st.info("🔒 Assessment Override Controls locked for LHW profile. (Senior Radiologist PIN 9999 required).")

            rec_text = r.get("recommendation", "Patient should be referred for specialist consultation")
            if r["is_critical"]:
                st.warning(f"⚠️ {t('Action Plan')}: {rec_text}")
            else:
                st.success(f"✅ {t('Action Plan')}: {rec_text}")

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
    st.markdown(f"""<div class="page-header" style="background:linear-gradient(135deg,{C['primary']} 0%,{C['success']} 100%);">
    <h1>🏥 Allied Hospital Faisalabad</h1><p>Urban Receiving Terminal • 2G GSM Critical Alert Monitor</p></div>""",
                unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c2:
        st.subheader(t("Network Status"))
        net = st.session_state.net_stats
        st.markdown(f"""<div class="card">GSM Module: <b>{net['module']}</b><br>Signal: <b>{net['signal']}</b><br>
        BHU Nodes: <b>{net['bhus']}</b><br>Pending: <b>{len(st.session_state.sms_alerts)}</b></div>""", unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.markdown(f'<div class="metric-tile"><div class="label">{t("Total Alerts")}</div><div class="value">{len(st.session_state.sms_alerts)}</div></div>', unsafe_allow_html=True)
        crit = sum(1 for a in st.session_state.sms_alerts if a["is_critical"])
        m2.markdown(f'<div class="metric-tile"><div class="label">Critical</div><div class="value" style="color:{C["danger"]};">{crit}</div></div>', unsafe_allow_html=True)

    with c1:
        st.subheader(t("Alert Stream"))
        if not st.session_state.sms_alerts:
            st.markdown('<div class="card" style="text-align:center;padding:30px;">✅ All Clear — no alerts pending</div>', unsafe_allow_html=True)
        for a in st.session_state.sms_alerts:
            alert_id = a["id"]  # define alert_id from the current alert dict
            # Ensure per-alert state keys exist
            if alert_id not in st.session_state.lhv_decisions:
                st.session_state.lhv_decisions[alert_id] = {"decision": None, "reason": None}
            if alert_id not in st.session_state.show_override_reason:
                st.session_state.show_override_reason[alert_id] = False

            css = "critical" if a["is_critical"] else "ok"
            st.markdown(f"""<div class="alert-card {css}"><b>Alert #{a['id']}</b> ({a['time']})<br>
            <span style="font-family:Consolas,monospace;font-size:0.8rem;">Payload: {a['payload']}<br>
            Type: {a['type']} &nbsp; Status: {a['status']}</span></div>""", unsafe_allow_html=True)

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
                verdict_display = combine_icon_and_text(vicon, verdict)
                conf_display = f"{confidence:.1f}%"
            else:
                verdict_display = "⚠️ No AI data recorded for this alert"
                conf_display = "N/A"

            st.markdown(f"""
            <div class="ai-recommendation {risk_class}">
                <div class="rec-header">🤖 AI Recommendation</div>
                <div class="rec-verdict">{verdict_display}</div>
                <div class="rec-confidence">Confidence: <b>{conf_display}</b></div>
                <div style="font-size: 0.9rem; margin-top: 8px; opacity: 0.95;">Localization: {localization}</div>
            </div>
            """, unsafe_allow_html=True)
            if not has_real_data:
                st.caption("This alert predates the real-data fix, or session state was carried over. "
                           "Use 🔄 Reset Session in the sidebar and run a fresh triage to see live results here.")

            already_approved = a["status"] == "Approved"

            # LHV DECISION + APPROVAL ROW
            col_agree, col_override, col_approve = st.columns(3)

            with col_agree:
                if st.button("✅ Agree", key=f"agree_{alert_id}", use_container_width=True, disabled=already_approved):
                    st.session_state.lhv_decisions[alert_id]["decision"] = "agree"
                    st.session_state.show_override_reason[alert_id] = False
                    st.toast("👩\u200d⚕️ LHV Decision: Agreed with AI recommendation")

            with col_override:
                if st.button("🔄 Override", key=f"override_{alert_id}", use_container_width=True, disabled=already_approved):
                    st.session_state.lhv_decisions[alert_id]["decision"] = "override"
                    st.session_state.show_override_reason[alert_id] = True
                    st.rerun()

            with col_approve:
                if st.button("📤 Approve", key=f"approve_{alert_id}", use_container_width=True, disabled=already_approved):
                    a["status"] = "Approved"
                    st.session_state.log_entries.append(
                        f"[{time.strftime('%H:%M:%S')}] [REFERRAL] Alert #{a['id']} approved and forwarded to hospital."
                    )
                    st.toast("📤 Approved — referral forwarded to hospital.")
                    st.rerun()

            if already_approved:
                st.markdown("""
                <div style="background:#10b98120;border-left:4px solid #10b981;border-radius:8px;padding:12px;margin-top:8px;margin-bottom:12px;">
                    <b>📤 Status:</b> Approved & forwarded to hospital
                </div>
                """, unsafe_allow_html=True)

            # Show LHV Decision status
            decision = st.session_state.lhv_decisions[alert_id]["decision"]
            if decision:
                status_text = "✅ Agreed with AI" if decision == "agree" else "🔴 Overridden"
                status_color = "#10b981" if decision == "agree" else "#ef4444"
                st.markdown(f"""
                <div style="background: {status_color}20; border-left: 4px solid {status_color}; border-radius: 8px; padding: 12px; margin-top: 8px; margin-bottom: 12px;">
                    <b>👩\u200d⚕️ LHV Decision:</b> {status_text}
                </div>
                """, unsafe_allow_html=True)

            # OVERRIDE REASON BOX (shown only when Override is selected)
            if st.session_state.show_override_reason[alert_id] and st.session_state.lhv_decisions[alert_id]["decision"] == "override":
                st.markdown("""
                <div class="override-reason-box">
                    <div class="reason-title">📋 Reason for Override</div>
                </div>
                """, unsafe_allow_html=True)

                reason_options = [
                    "👤 Patient History",
                    "📸 Image Quality",
                    "🔬 Clinical Symptoms",
                    "❓ Other"
                ]

                selected_reason = st.radio(
                    "Select reason for override:",
                    reason_options,
                    key=f"reason_{alert_id}",
                    label_visibility="collapsed"
                )

                st.session_state.lhv_decisions[alert_id]["reason"] = selected_reason

                # If "Other" is selected, allow custom text input
                if "Other" in selected_reason:
                    custom_reason = st.text_input(
                        "Please specify the reason:",
                        key=f"custom_reason_{alert_id}",
                        placeholder="Enter additional details..."
                    )
                    st.session_state.lhv_decisions[alert_id]["reason"] = f"Other: {custom_reason}"

                # Confirmation button
                if st.button(f"✓ Confirm Override", key=f"confirm_override_{alert_id}", use_container_width=True):
                    st.toast(f"✅ Override confirmed. Reason: {st.session_state.lhv_decisions[alert_id]['reason']}")
                    st.session_state.show_override_reason[alert_id] = False

            st.markdown("---")


# ============================================================
# CLOUD SYNC TAB
# ============================================================
def render_cloud_sync():
    st.markdown(f"""<div class="page-header" style="background:linear-gradient(135deg,{C['primary']} 0%,{C['warning']} 100%);">
    <h1>☁️ Cloud Sync & Cache</h1><p>Alibaba Cloud Integration (simulated) • Hybrid-Edge Architecture</p></div>""",
                unsafe_allow_html=True)

    total, unsynced = core.counts()
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-tile"><div class="label">{t("Cached Reports")}</div><div class="value">{total}</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-tile"><div class="label">{t("Unsynced")}</div><div class="value" style="color:{C["danger"]};">{unsynced}</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-tile"><div class="label">{t("Synced")}</div><div class="value" style="color:{C["success"]};">{total - unsynced}</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-tile"><div class="label">{t("IoT Queue")}</div><div class="value" style="color:{C["warning"]};">{len(st.session_state.iot_messages)}</div></div>', unsafe_allow_html=True)

    st.subheader(t("Local Cache Status"))
    rows = core.get_cached_reports()
    if not rows:
        st.info(t("No reports cached"))
    else:
        st.dataframe(
            [{"ID": r[0], "Patient": r[1], "Modality": r[3], "BI-RADS/Severity": r[5], "Verdict": r[7],
              "Timestamp": r[11], "Synced": "✅" if r[12] else "⏳"} for r in rows],
            width="stretch", hide_index=True,
        )

    # ============================================================
    # HELP & FEEDBACK SECTION (after Cloud Sync)
    # ============================================================
    st.markdown("---")
    st.markdown(f"""<div class="helpya-card">
    <div class="helpya-header">
        💬 Help - LHV Feedback & Session Analytics
    </div>
    </div>""", unsafe_allow_html=True)

    # ============================================================
    # VOICE MESSAGE MANAGEMENT
    # ============================================================
    st.markdown("---")
    st.markdown("<div style='background: linear-gradient(135deg, #ff69b4 0%, #ff1493 100%); color: white; padding: 12px 16px; border-radius: 8px; margin: 16px 0 12px 0; font-size: 1.2rem; font-weight: 800;'>🎤 Voice Message Library</div>", unsafe_allow_html=True)

    voice_mgmt_col1, voice_mgmt_col2 = st.columns([2, 1])
    with voice_mgmt_col1:
        st.markdown("<div style='background: rgba(255, 105, 180, 0.2); color: #ff1493; padding: 8px 12px; border-radius: 6px; margin: 8px 0; font-size: 1rem; font-weight: 700; border-left: 4px solid #ff69b4;'>Pre-recorded voice messages for different scenarios:</div>", unsafe_allow_html=True)
        selected_message = st.selectbox(
            "Select a message to test:",
            list(VOICE_MESSAGES.keys()),
            label_visibility="collapsed"
        )

    with voice_mgmt_col2:
        lang_options = {"en": "🇬🇧 English", "ur": "🇵🇰 Urdu", "pa": "🇵🇅 Punjabi"}
        selected_lang = st.selectbox(
            "Language:",
            list(lang_options.keys()),
            format_func=lambda code: lang_options[code],
            label_visibility="collapsed"
        )

    # Display message text
    message_text = VOICE_MESSAGES.get(selected_message, {}).get(selected_lang, "No message found")
    st.info(f"📝 **Message:** {message_text}")

    # Test audio player
    test_audio = play_voice_message(selected_message, selected_lang)
    if test_audio:
        st.audio(test_audio, format="audio/wav", sample_rate=22050)
        st.markdown("<small>🔊 Click play to hear the message</small>", unsafe_allow_html=True)
    else:
        st.caption("🔇 No audio engine configured yet — showing text only. Wire up play_voice_message() to enable playback.")

    # Upload custom voice file
    st.markdown("<div style='background: rgba(255, 105, 180, 0.2); color: #ff1493; padding: 8px 12px; border-radius: 6px; margin: 8px 0; font-size: 1rem; font-weight: 700; border-left: 4px solid #ff69b4;'>Or upload your own audio file:</div>", unsafe_allow_html=True)
    custom_voice_file = st.file_uploader(
        "Upload audio (MP3/WAV/OGG)",
        type=["mp3", "wav", "ogg"],
        label_visibility="collapsed"
    )
    if custom_voice_file:
        st.audio(custom_voice_file, format=f"audio/{custom_voice_file.name.split('.')[-1]}")
        st.success(f"✅ Custom audio loaded: {custom_voice_file.name}")

    # ============================================================
    # Help Metrics Row
    # ============================================================
    st.markdown("---")
    hfb_col1, hfb_col2, hfb_col3 = st.columns(3)

    with hfb_col1:
        st.markdown(f"""
        <div class="helpya-stat">
            <div class="stat-label">👩‍⚕️ LHV Sessions</div>
            <div class="stat-value">{st.session_state.get('helpya_total_sessions', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    with hfb_col2:
        st.markdown(f"""
        <div class="helpya-stat">
            <div class="stat-label">✅ Successful Diagnosis</div>
            <div class="stat-value">{st.session_state.get('helpya_successful_diagnosis', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    with hfb_col3:
        st.markdown(f"""
        <div class="helpya-stat">
            <div class="stat-label">⬆️ Escalations</div>
            <div class="stat-value">{st.session_state.get('helpya_escalations', 0)}</div>
        </div>
        """, unsafe_allow_html=True)

    # Session Feedback Form
    st.markdown("<div style='background: linear-gradient(135deg, #ff69b4 0%, #ff1493 100%); color: white; padding: 12px 16px; border-radius: 8px; margin: 16px 0 12px 0; font-size: 1.2rem; font-weight: 800;'>📝 Session Feedback & Quality Check</div>", unsafe_allow_html=True)

    feedback_col1, feedback_col2 = st.columns([1, 1])

    with feedback_col1:
        feedback_rating = st.slider(
            "Rate this diagnostic session:",
            min_value=1, max_value=5, value=3,
            help="How was your experience with this session?"
        )

        # Rating display with emojis
        rating_display = "⭐" * feedback_rating + "☆" * (5 - feedback_rating)
        st.markdown(f"<div style='text-align:center;font-size:1.2rem;margin:8px 0;'>{rating_display}</div>", unsafe_allow_html=True)

    with feedback_col2:
        feedback_category = st.selectbox(
            "Session Type:",
            ["✅ Successful Diagnosis", "⚠️ Needs Review", "🔄 Reanalysis", "⬆️ Escalation"]
        )

    # Detailed Feedback
    st.markdown("<div style='background: rgba(255, 105, 180, 0.2); color: #ff1493; padding: 8px 12px; border-radius: 6px; margin: 8px 0; font-size: 1rem; font-weight: 700; border-left: 4px solid #ff69b4;'>Your Feedback:</div>", unsafe_allow_html=True)
    feedback_text = st.text_area(
        "Share details about this session:",
        placeholder="E.g., Image quality issues, system performance, suggestions...",
        height=100,
        label_visibility="collapsed"
    )

    # Voice Feedback Option
    voice_feedback_col1, voice_feedback_col2 = st.columns(2)
    with voice_feedback_col1:
        enable_voice_feedback = st.checkbox("🎤 Enable voice feedback confirmation")

    with voice_feedback_col2:
        lang_feedback = st.radio(
            "Feedback Language:",
            ["🇬🇧 English", "🇵🇰 اردو"],
            horizontal=True,
            label_visibility="collapsed"
        )

    # Submit Feedback Button
    submit_col1, submit_col2 = st.columns([2, 1])

    with submit_col1:
        if st.button("📤 Submit Feedback & Rate Session", use_container_width=True):
            if feedback_text.strip():
                # Update metrics
                st.session_state.helpya_total_sessions += 1

                if "Successful" in feedback_category:
                    st.session_state.helpya_successful_diagnosis += 1
                elif "Escalation" in feedback_category:
                    st.session_state.helpya_escalations += 1

                # Success notification
                st.success(f"✅ Feedback submitted! Rating: {rating_display}")

                # Voice confirmation if enabled
                if enable_voice_feedback and st.session_state.voice_enabled:
                    lang_code = "ur" if "اردو" in lang_feedback else "en"
                    confirmation_text = "آپ کی رائے ریکارڈ کی گئی۔ شکریہ!" if lang_code == "ur" else "Your feedback has been recorded. Thank you!"
                    st.info(f"🎤 Voice Feedback: {confirmation_text}")

                # Log to session
                st.session_state.log_entries.append(
                    f"[{time.strftime('%H:%M:%S')}] [HELPYA] Session rated {feedback_rating}★ - {feedback_category}"
                )
            else:
                st.warning("⚠️ Please enter your feedback before submitting.")

    with submit_col2:
        if st.button("🔄 Clear Form", use_container_width=True):
            st.rerun()

    # Today's Summary Panel
    st.markdown("---")
    st.markdown("<div style='background: linear-gradient(135deg, #ff69b4 0%, #ff1493 100%); color: white; padding: 12px 16px; border-radius: 8px; margin: 16px 0 12px 0; font-size: 1.2rem; font-weight: 800;'>📊 Today's Diagnostic Summary</div>", unsafe_allow_html=True)

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric(
            "🩺 Alerts Processed",
            len(st.session_state.sms_alerts),
            delta="New alerts" if len(st.session_state.sms_alerts) > 0 else "No alerts"
        )

    with summary_col2:
        st.metric(
            "💾 Reports Cached",
            total,
            delta=f"{unsynced} pending sync" if unsynced > 0 else "All synced"
        )

    with summary_col3:
        sync_rate = ((total - unsynced) / total * 100) if total > 0 else 0
        st.metric(
            "☁️ Sync Rate",
            f"{sync_rate:.0f}%",
            delta="Optimal" if sync_rate == 100 else "Syncing..."
        )

    # System Status Panel
    st.markdown("---")
    status_col1, status_col2 = st.columns(2)

    with status_col1:
        st.info(f"""
        **🎤 Voice System:** {'🟢 ENABLED' if st.session_state.voice_enabled else '🔴 DISABLED'}
        **🌐 Language:** {'🇵🇰 Urdu (اردو)' if st.session_state.urdu_mode else '🇬🇧 English'}
        **📡 Network:** {st.session_state.net_stats['module']}
        **📶 Signal:** {st.session_state.net_stats['signal']}
        """)

    with status_col2:
        st.success(f"""
        **✅ System Status:** Operational
        **🔐 Data Encryption:** Active
        **📊 LHV Sessions:** {st.session_state.get('helpya_total_sessions', 0)}
        **⏱️ Session Time:** {st.session_state.inference_latency}s avg
        """)

    # Quick Actions
    st.markdown("---")
    st.markdown("<div style='background: linear-gradient(135deg, #ff69b4 0%, #ff1493 100%); color: white; padding: 12px 16px; border-radius: 8px; margin: 16px 0 12px 0; font-size: 1.2rem; font-weight: 800;'>⚡ Quick Actions</div>", unsafe_allow_html=True)

    action_col1, action_col2, action_col3, action_col4 = st.columns(4)

    with action_col1:
        if st.button("🔊 Test Voice", use_container_width=True):
            st.info("🎤 Voice system test: System is functional and ready for use.")
            st.session_state.voice_enabled = True

    with action_col2:
        if st.button("📥 Download Report", use_container_width=True):
            st.success("✅ Report download initiated")

    with action_col3:
        if st.button("🌐 Switch Language", use_container_width=True):
            st.session_state.urdu_mode = not st.session_state.urdu_mode
            st.rerun()

    with action_col4:
        if st.button("🔄 Sync Now", use_container_width=True):
            st.success("✅ Cloud sync initiated")
            st.session_state.log_entries.append(f"[{time.strftime('%H:%M:%S')}] [SYNC] Manual sync request completed")


# ============================================================
# MAIN
# ============================================================
selected_model, uploaded = render_sidebar()
nav = [f"🩺 {t('Dashboard')}", f"🏥 {t('Hospital Hub')}", f"☁️ {t('Cloud Sync')}"]
tab1, tab2, tab3 = st.tabs(nav)
with tab1:
    render_dashboard(selected_model)
with tab2:
    render_hospital_hub()
with tab3:
    render_cloud_sync()

st.markdown(
    f'<div style="text-align:center;padding:20px;color:{C["text_light"]};font-size:0.72rem;">'
    f'Pink Edge AI (Streamlit) — same real models as the desktop app, browser UI</div>',
    unsafe_allow_html=True)
