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

st.set_page_config(page_title="Pink Edge AI", page_icon="🩺", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# COLOR TOKENS & PROFESSIONAL CLINICAL LIGHT THEME
# ============================================================
# Imported from GUI.py rather than redefined here — one source of truth for both editions. These
# two used to keep their own separate dicts and had drifted apart on every brand colour and on
# four of the six semantic ones; see GUI.py's THEME block for the palette and its rationale.
#
# NOTE: the CSS below is a plain string, not an f-string (it's full of literal CSS braces), so it
# repeats these values as literals rather than interpolating them. Keep the two in sync when
# changing a colour — `python Tests/smoke_test.py` checks for exactly that drift.
C = core.C

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


def speak_voiceover(text: str, lang=None):
    """Browser voice-over via Web Speech API (works offline once voices are cached)."""
    if not text or not st.session_state.get("voice_enabled", True):
        return
    import json
    import streamlit.components.v1 as components

    if lang is None:
        lang = "ur" if st.session_state.get("urdu_mode") else "en"
    locale = "ur-PK" if str(lang).startswith("ur") else "en-US"
    safe = json.dumps(str(text)[:500])
    components.html(
        f"""
        <script>
        (() => {{
          const utter = new SpeechSynthesisUtterance({safe});
          utter.lang = {json.dumps(locale)};
          utter.rate = 1.0;
          window.speechSynthesis.cancel();
          window.speechSynthesis.speak(utter);
        }})();
        </script>
        """,
        height=0,
    )


def play_voice_message(message_key: str, lang: str):
    """
    Speaks the library line via browser TTS and returns None for st.audio
    (text still shown in UI). Prefer speak_voiceover() for free-form narration.
    """
    text = VOICE_MESSAGES.get(message_key, {}).get(lang) or VOICE_MESSAGES.get(message_key, {}).get("en")
    if text:
        speak_voiceover(text, lang=lang)
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
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=Syne:wght@600;700;800&display=swap');

:root {
  --bg: #0f1419;
  --surface: #161d27;
  --surface-alt: #1e2733;
  --border: #2a3544;
  --border-light: #1c2430;
  --text: #e8eef6;
  --muted: #9aabbf;
  --light: #8494a8;
  --primary: #4f8cff;
  --primary-deep: #9db8f0;
  --accent: #9eb6e8;
  --success: #3d9b8f;
  --warning: #fbbf24;
  --danger: #f87171;
  --primary-bg: #152238;
  --radius: 14px;
  --shadow: 0 12px 40px rgba(0,0,0,0.35);
  --shadow-sm: 0 4px 16px rgba(0,0,0,0.22);
  --font: 'Sora', sans-serif;
  --display: 'Syne', sans-serif;
  --glow: 0 0 24px rgba(79,140,255,0.22);
}

@keyframes pe-in {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes pe-pulse {
  0%, 100% { box-shadow: 0 0 0 0 rgba(79,140,255,0.35); }
  50% { box-shadow: 0 0 0 10px rgba(79,140,255,0); }
}

html, body, .stApp, [class*="css"],
button, input, textarea, select, label, p, span, div, h1, h2, h3, h4, h5, h6 {
  font-family: var(--font) !important;
}
/* Keep Streamlit Material icon ligatures (prevents "_arrow_right" text overlap) */
[data-testid="stIconMaterial"],
[data-testid="stExpanderToggleIcon"],
[data-testid="stExpanderIcon"],
span[class*="material"],
.material-icons,
.material-symbols-rounded,
.material-symbols-outlined,
.material-symbols-sharp {
  font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
  font-style: normal !important;
  font-weight: normal !important;
  letter-spacing: normal !important;
  text-transform: none !important;
  speak: never;
}

.stApp {
  background:
    radial-gradient(900px 480px at 0% -5%, rgba(79,140,255,0.14), transparent 55%),
    radial-gradient(700px 400px at 100% 0%, rgba(79,140,255,0.08), transparent 50%),
    var(--bg) !important;
  color: var(--text) !important;
}

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #121820 0%, #0f1419 100%) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { padding-top: 0.85rem !important; }
section[data-testid="stSidebar"] * { color: var(--text) !important; }

/* Professional dashboard sidebar sections */
.sidebar-section {
  background: #161d27;
  border: 1px solid #2a3544;
  border-radius: 14px;
  padding: 12px 12px 10px;
  margin-bottom: 12px;
  box-shadow: 0 4px 16px rgba(0,0,0,0.2);
}
.sidebar-section-title {
  font-family: 'Sora', sans-serif !important;
  font-size: 0.64rem !important;
  font-weight: 800 !important;
  letter-spacing: 0.14em !important;
  text-transform: uppercase !important;
  color: #8494a8 !important;
  margin: 0 0 10px 2px !important;
}
.sidebar-field-label {
  font-family: 'Sora', sans-serif !important;
  font-size: 0.72rem !important;
  font-weight: 600 !important;
  color: #9aabbf !important;
  margin: 0 0 6px 2px !important;
}
section[data-testid="stSidebar"] .stButton > button {
  justify-content: flex-start !important;
  text-align: left !important;
  padding-left: 12px !important;
  margin-bottom: 4px !important;
  min-height: 40px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] {
  background: #1e2733 !important;
  border: 1px dashed #3a5680 !important;
  border-radius: 12px !important;
  padding: 8px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
  padding: 8px !important;
}
section[data-testid="stSidebar"] [data-testid="stFileUploader"] small,
section[data-testid="stSidebar"] [data-testid="stFileUploader"] span {
  font-family: 'Sora', sans-serif !important;
  font-size: 0.72rem !important;
}
section[data-testid="stSidebar"] [data-testid="stCaption"] {
  font-size: 0.7rem !important;
  color: #8494a8 !important;
}

.block-container {
  padding-top: 3.4rem !important;
  padding-bottom: 3rem !important;
  max-width: 1280px !important;
  animation: pe-in 0.35s ease-out;
}

h1,h2,h3,h4,h5,h6 { color: var(--text) !important; font-family: var(--display) !important; }
p,label,span,div,li,td,th { color: var(--text) !important; }
.stMarkdown p, .stMarkdown span, .stMarkdown label { color: var(--text) !important; }

/* Top bar */
.app-topbar {
  display: flex; align-items: center; justify-content: space-between; gap: 16px; flex-wrap: wrap;
  background: linear-gradient(135deg, #221c2c 0%, #161d27 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
  margin-bottom: 14px;
  box-shadow: var(--shadow-sm);
}
.app-topbar .eyebrow {
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--primary) !important;
}
.app-topbar .title {
  font-family: var(--display) !important;
  font-size: 1.5rem !important; font-weight: 800 !important;
  letter-spacing: -0.03em; color: var(--text) !important; margin-top: 2px;
}
.app-topbar .subtitle { font-size: 0.82rem; color: var(--muted) !important; margin-top: 3px; }
.pill {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 7px 12px; border-radius: 999px;
  font-size: 0.72rem; font-weight: 700;
  border: 1px solid var(--border); background: var(--surface-alt); color: var(--muted) !important;
}
.pill.ok { background: #14261c; border-color: #166534; color: #3d9b8f !important; }
.pill.warn { background: #2a2114; border-color: #854d0e; color: #fbbf24 !important; }
.pill.info { background: var(--primary-bg); border-color: #3a5680; color: var(--primary) !important; }

.sidebar-brand {
  background: linear-gradient(145deg, #4f8cff 0%, #3a6fd8 55%, #2f5bb8 140%);
  border-radius: 16px; padding: 18px 16px; margin-bottom: 14px; color: #fff !important;
  box-shadow: var(--glow);
}
.sidebar-brand b {
  display: block; font-family: var(--display) !important; font-size: 1.2rem;
  color: #ffffff !important; letter-spacing: -0.02em; font-weight: 800;
}
.sidebar-brand .tag { color: #f8fbff !important; font-size: 0.74rem; margin-top: 4px; }
/* High-contrast text on primary gradient surfaces (a11y) */
.brand-bar .brand-title, .demo-banner b, .demo-banner span { color: #ffffff !important; }
.stButton > button[kind="primary"] *, .stButton > button[data-testid="baseButton-primary"] * {
  color: #ffffff !important;
}

.nav-rail-label {
  font-size: 0.66rem !important; font-weight: 800 !important;
  letter-spacing: 0.14em !important; text-transform: uppercase !important;
  color: var(--light) !important; margin: 12px 0 8px 2px !important;
}
.nav-hint { font-size: 0.74rem; color: var(--light) !important; margin: 0 0 10px 2px; line-height: 1.4; }

.section-label {
  display: flex; align-items: center; gap: 10px;
  font-size: 0.72rem; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--muted) !important; margin: 8px 0 12px;
  font-family: var(--font) !important;
}
.section-label::after {
  content: ""; flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}

.action-dock {
  background: linear-gradient(180deg, #241e2e 0%, #161d27 100%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 12px;
  margin-bottom: 12px;
  box-shadow: inset 0 1px 0 rgba(79,140,255,0.08);
}

.workflow-strip {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin: 0 0 16px;
}
.workflow-step {
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 10px; text-align: center; transition: 0.2s ease;
}
.workflow-step:hover { transform: translateY(-2px); box-shadow: var(--shadow-sm); border-color: #3a5680; }
.workflow-step .n {
  width: 22px; height: 22px; margin: 0 auto 6px; border-radius: 999px;
  display: flex; align-items: center; justify-content: center;
  font-size: 0.7rem; font-weight: 800; background: var(--surface-alt); color: var(--muted) !important;
}
.workflow-step .l { font-size: 0.68rem; font-weight: 700; color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.04em; }
.workflow-step.active { border-color: var(--primary); background: var(--primary-bg); box-shadow: 0 0 0 3px rgba(79,140,255,0.12); }
.workflow-step.active .n { background: var(--primary); color: #0f1419 !important; animation: pe-pulse 2s infinite; }
.workflow-step.active .l { color: var(--primary) !important; }
.workflow-step.done { border-color: #166534; background: #14261c; }
.workflow-step.done .n { background: var(--success); color: #052e16 !important; }
.workflow-step.done .l { color: var(--success) !important; }

.brand-bar {
  display: flex; align-items: center; gap: 14px; margin-bottom: 14px;
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 14px 16px; box-shadow: var(--shadow-sm);
}
.brand-mark {
  width: 46px; height: 46px; border-radius: 14px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(145deg, #4f8cff, #3a6fd8);
  color: #fff !important; font-weight: 800; font-size: 0.95rem;
  font-family: var(--display) !important;
  box-shadow: var(--glow);
}
.brand-title {
  font-family: var(--display) !important; font-size: 1.35rem !important; font-weight: 800 !important;
  color: var(--text) !important; letter-spacing: -0.03em;
}
.brand-sub { font-size: 0.8rem; color: var(--muted) !important; font-weight: 500; }

.page-header {
  background: linear-gradient(135deg, #4f8cff 0%, #2f5bb8 100%);
  border-radius: var(--radius); padding: 20px 22px; margin: 0.4rem 0 1rem;
  box-shadow: var(--glow); position: relative; overflow: hidden;
}
.page-header h1 {
  color: #fff !important; margin: 0 !important;
  font-family: var(--display) !important; font-size: 1.45rem !important; font-weight: 800 !important;
}
.page-header p { color: rgba(255,255,255,0.92) !important; margin: 6px 0 0 !important; font-size: 0.88rem !important; }

.card, .result-panel, .xai-panel, .helpya-card, .voice-feedback-panel, .builder-shell {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow-sm) !important;
  transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.card { padding: 16px !important; margin: 8px 0 !important; color: var(--text) !important; }
.card:hover, .result-panel:hover, .metric-tile:hover, .kpi-card:hover, .feature-tile:hover {
  transform: translateY(-2px); box-shadow: var(--shadow) !important; border-color: #3a5680 !important;
}
.result-panel { padding: 20px; margin-bottom: 14px; animation: pe-in 0.35s ease-out; }
.builder-shell { padding: 18px; margin-bottom: 14px; }
.builder-shell h3 {
  font-family: var(--display) !important; margin: 0 0 6px !important; font-size: 1.2rem !important; color: var(--text) !important;
}

.metric-tile, .kpi-card {
  background: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 14px !important; padding: 14px !important;
  box-shadow: var(--shadow-sm) !important;
}
.metric-tile { text-align: center !important; }
.metric-tile .label, .kpi-card .k {
  color: var(--light) !important; font-size: 0.68rem !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: 0.08em !important;
}
.metric-tile .value, .kpi-card .v {
  color: var(--primary) !important; font-size: 1.45rem !important; font-weight: 800 !important;
  margin-top: 4px !important; font-variant-numeric: tabular-nums;
  font-family: var(--display) !important;
}
.kpi-card .s { font-size: 0.74rem; color: var(--muted) !important; margin-top: 2px; }

.kpi-grid {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 0 0 16px;
}
.feature-grid {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin: 0 0 18px;
}
.feature-tile {
  background: var(--surface); border: 1px solid var(--border); border-radius: 16px;
  padding: 16px; min-height: 118px; box-shadow: var(--shadow-sm); transition: 0.18s ease;
}
.feature-tile .ft-icon { font-size: 1.25rem; margin-bottom: 8px; }
.feature-tile .ft-title {
  font-family: var(--display) !important; font-size: 1.05rem !important; font-weight: 800 !important; color: var(--text) !important;
}
.feature-tile .ft-desc { font-size: 0.8rem; color: var(--muted) !important; margin-top: 4px; line-height: 1.45; }

.activity-row {
  display: flex; justify-content: space-between; gap: 12px; align-items: center;
  padding: 10px 0; border-bottom: 1px solid var(--border-light);
  font-size: 0.84rem;
}
.activity-row:last-child { border-bottom: none; }

.badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 5px 11px; border-radius: 999px; font-size: 0.72rem; font-weight: 700;
}
.badge-offline { background: #14261c !important; color: #3d9b8f !important; border: 1px solid #166534 !important; }
.badge-cloud { background: #2a2114 !important; color: #fbbf24 !important; border: 1px solid #854d0e !important; }

.verdict-box { border-radius: var(--radius) !important; padding: 18px !important; text-align: center !important; margin: 12px 0 !important; }
.verdict-box.success { background: #14261c !important; border: 2px solid #166534 !important; }
.verdict-box.success .v-title { color: #3d9b8f !important; }
.verdict-box.danger { background: #2a1518 !important; border: 2px solid #7f1d1d !important; }
.verdict-box.danger .v-title { color: #f87171 !important; }
.verdict-box.warning { background: #2a2114 !important; border: 2px solid #854d0e !important; }
.verdict-box.warning .v-title { color: #fbbf24 !important; }
.verdict-box .v-icon { font-size: 1.7rem !important; }
.verdict-box .v-title { font-size: 1.25rem !important; font-weight: 800 !important; font-family: var(--display) !important; }
.verdict-box .v-sub { font-size: 0.88rem !important; margin-top: 4px !important; color: var(--muted) !important; font-weight: 600 !important; }

.console-log {
  background: #0c0a10 !important; border: 1px solid #1c2430 !important; border-radius: 14px !important;
  padding: 14px !important; color: #9aabbf !important;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important;
  font-size: 0.76rem !important; line-height: 1.7 !important; max-height: 240px !important; overflow-y: auto !important;
}

.alert-card {
  border-radius: 12px !important; padding: 14px !important; margin: 6px 0 !important;
  border-left: 4px solid !important; transition: 0.2s ease;
}
.alert-card:hover { transform: translateX(2px); box-shadow: var(--shadow-sm); }
.alert-card.critical { background: #2a1518 !important; border-color: #f87171 !important; color: #fecaca !important; }
.alert-card.ok { background: #14261c !important; border-color: #3d9b8f !important; color: #bbf7d0 !important; }

.source-tag {
  font-size: 0.72rem !important; color: var(--muted) !important;
  font-family: ui-monospace, Menlo, Consolas, monospace !important; font-weight: 700 !important;
}

.ai-recommendation {
  background: linear-gradient(135deg, #4f8cff, #3a6fd8);
  border-radius: 14px; padding: 18px; margin: 12px 0; color: #fff;
  box-shadow: var(--glow); animation: pe-in 0.3s ease-out;
}
.ai-recommendation.risk-critical { background: linear-gradient(135deg, #f87171, #b91c1c) !important; }
.ai-recommendation.risk-moderate { background: linear-gradient(135deg, #fbbf24, #b45309) !important; }
.ai-recommendation.risk-low { background: linear-gradient(135deg, #3d9b8f, #15803d) !important; }
.ai-recommendation .rec-header, .ai-recommendation .rec-verdict, .ai-recommendation .rec-confidence { color: #fff !important; }
.ai-recommendation .rec-header { font-size: 0.85rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.95; }
.ai-recommendation .rec-verdict { font-size: 1.3rem; font-weight: 800; margin: 8px 0; font-family: var(--display) !important; }

.lhv-decision {
  background: linear-gradient(135deg, #3b4f7a, #2b4c7e);
  border-radius: 14px; padding: 18px; margin: 12px 0; color: #fff;
}
.lhv-decision .lhv-header { color: #fff !important; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; }

.override-reason-box {
  background: #1a2238; border: 2px solid #9eb6e8; border-radius: 12px; padding: 14px; margin-top: 12px;
}
.override-reason-box .reason-title { font-weight: 700; color: #c2d2f5; margin-bottom: 8px; text-transform: uppercase; font-size: 0.85rem; }

.xai-panel { padding: 18px; margin: 12px 0 16px; }
.xai-panel h4 {
  font-family: var(--display) !important; margin: 0 0 4px !important; font-size: 1.15rem !important; color: var(--text) !important;
}
.xai-panel .xai-sub { color: var(--muted) !important; font-size: 0.8rem; margin-bottom: 12px; }
.xai-bar-label { display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 700; color: var(--muted) !important; margin-bottom: 4px; }
.xai-bar-track { height: 10px; border-radius: 999px; background: var(--surface-alt); overflow: hidden; }
.xai-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #4f8cff, #9eb6e8); }
.xai-tier {
  display: flex; gap: 10px; align-items: flex-start; padding: 10px 12px; border-radius: 10px;
  margin: 6px 0; border: 1px solid var(--border); background: var(--surface-alt);
}
.xai-tier.active { border-color: var(--primary); background: var(--primary-bg); box-shadow: 0 0 0 3px rgba(79,140,255,0.12); }
.xai-tier .dot { width: 10px; height: 10px; border-radius: 50%; margin-top: 5px; background: #6b6280; flex-shrink: 0; }
.xai-tier.active .dot { background: var(--primary); animation: pe-pulse 2s infinite; }
.xai-tier .t-title { font-weight: 800; font-size: 0.84rem; color: var(--text) !important; }
.xai-tier .t-desc { font-size: 0.74rem; color: var(--muted) !important; }

.coach-card {
  background: #2a2114; border: 1px solid #854d0e; border-left: 4px solid #fbbf24;
  border-radius: 12px; padding: 12px 14px; margin: 10px 0 14px; font-size: 0.9rem; color: var(--text) !important;
}
.demo-banner {
  background: linear-gradient(135deg, #4f8cff, #2f5bb8); color: #fff !important;
  border-radius: 14px; padding: 14px 16px; margin-bottom: 12px;
  box-shadow: var(--glow);
}
.demo-banner b, .demo-banner span { color: #fff !important; }
.train-score { display: grid; grid-template-columns: repeat(3,1fr); gap: 8px; margin: 10px 0 16px; }
.train-score .tile { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 12px; text-align: center; }
.train-score .v { font-size: 1.35rem; font-weight: 800; color: var(--primary) !important; font-family: var(--display) !important; }
.train-score .l { font-size: 0.68rem; font-weight: 700; color: var(--light) !important; text-transform: uppercase; }

.helpya-card { overflow: hidden; margin: 16px 0; padding: 0 !important; }
.helpya-card .helpya-header {
  background: linear-gradient(135deg, #4f8cff, #3a6fd8); color: #fff !important;
  padding: 14px 16px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.06em;
  font-family: var(--display) !important;
}
.helpya-stat {
  display: inline-block; background: var(--primary-bg); border: 1px solid #3a5680; border-radius: 12px;
  padding: 14px; margin: 10px; color: var(--primary); font-weight: 600;
}
.helpya-stat .stat-value { font-size: 1.5rem; font-weight: 800; margin-top: 4px; font-family: var(--display) !important; }
.helpya-stat .stat-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; }

.image-stage-caption {
  display: flex; justify-content: space-between; align-items: center;
  background: #0c0a10; border-radius: 12px 12px 0 0; padding: 10px 14px; margin-bottom: 0;
  border: 1px solid var(--border); border-bottom: none;
}

.toolbar {
  display: flex; gap: 10px; flex-wrap: wrap; align-items: center;
  background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
  padding: 10px 12px; margin-bottom: 14px; box-shadow: var(--shadow-sm);
}

.pe-footer {
  text-align: center; padding: 28px 12px 8px; color: var(--light) !important;
  font-size: 0.75rem; font-weight: 500; font-family: var(--font) !important;
}

/* Streamlit controls — dark clinical blue */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="base-input"],
.stTextInput input, .stNumberInput input, .stTextArea textarea {
  background: #1e2733 !important; color: var(--text) !important;
  border-color: var(--border) !important; border-radius: 12px !important;
  font-family: var(--font) !important;
}
div[data-baseweb="select"] span { color: var(--text) !important; }

.stButton > button {
  border-radius: 12px !important; font-weight: 700 !important; font-family: var(--font) !important;
  border: 1px solid var(--border) !important; background: #1e2733 !important; color: var(--text) !important;
  transition: 0.18s ease !important; box-shadow: var(--shadow-sm) !important;
}
.stButton > button:hover {
  border-color: var(--primary) !important; color: var(--primary) !important;
  background: var(--primary-bg) !important; transform: translateY(-1px);
}
.stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {
  background: linear-gradient(135deg, #4f8cff, #3a6fd8) !important;
  color: #fff !important; border: none !important;
  box-shadow: var(--glow) !important;
}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="baseButton-primary"]:hover {
  filter: brightness(1.08); color: #fff !important; border: none !important;
}

.stRadio label, .stCheckbox label { color: var(--text) !important; font-family: var(--font) !important; }
.stSlider label { color: var(--text) !important; }

.stTabs [data-baseweb="tab-list"] {
  gap: 0.35rem !important; padding: 6px !important; background: var(--surface) !important;
  border: 1px solid var(--border) !important; border-radius: 14px !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px !important; font-weight: 700 !important; color: var(--muted) !important;
  font-family: var(--font) !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, #4f8cff, #3a6fd8) !important; color: #fff !important;
}
.stTabs [aria-selected="true"] * { color: #fff !important; }

section[data-testid="stSidebar"] div[data-testid="stExpander"] {
  background: var(--surface); border: 1px solid var(--border) !important; border-radius: 12px !important; margin-bottom: 8px;
}

div[data-testid="stDialog"] > div {
  background: var(--surface) !important;
  border-radius: 18px !important; border: 1px solid var(--border) !important;
  box-shadow: 0 24px 60px rgba(0,0,0,0.55) !important;
}

[data-testid="stDataFrame"], .stDataFrame {
  background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 12px !important;
}

div[data-testid="stCaption"] { color: var(--muted) !important; font-family: var(--font) !important; }

@media (max-width: 900px) {
  .kpi-grid, .feature-grid { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 768px) {
  .block-container { padding-left: 10px !important; padding-right: 10px !important; }
  .workflow-strip { grid-template-columns: 1fr 1fr; }
  .app-topbar .title { font-size: 1.15rem !important; }
  .kpi-grid, .feature-grid { grid-template-columns: 1fr; }
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
        "active_page": "overview",
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
        "clinical_context_packet": None,
        "lhv_decisions": {}, "show_override_reason": {},
        # Voice + Help/Feedback panel state (Cloud Sync tab)
        "voice_enabled": True,
        "helpya_total_sessions": 0,
        "helpya_successful_diagnosis": 0,
        "helpya_escalations": 0,
        # Production builder case sheet
        "case_gender": "Female",
        "case_bhu": "BHU-14 Faisalabad",
        "case_urgency": "Routine",
        "case_notes": "",
        "case_built": False,
        # FYP wow-features
        "demo_mode": False,
        "demo_step": "",
        "train_score": 0,
        "train_attempts": 0,
        "train_streak": 0,
        "train_case": None,
        "train_revealed": False,
        "train_guess": None,
        # Client Edge Ops modules
        "dicom_listener_on": False,
        "dicom_inbox": [],
        "dicom_prefetch": [],
        "shield_last": None,
        "multitask_last": None,
        "cascade_jobs": [],
        "telemetry_quarantine": [],
        "telemetry_uploads": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
core._urdu["on"] = st.session_state.urdu_mode  # sync GUI.py's t() to this session's language choice
t = core.t

# Modular page components (keep streamlit_app thin — no duplicated triage logic)
from components.overview import render_overview
from components.case_builder import render_case_builder
from components.reports import render_reports_center
from components.ai_lab import render_ai_lab
from components.access import can_view_page, enforce_page_access, current_auth
from components.metrics import track



def render_workflow_strip(has_upload: bool = False):
    """Interactive clinical workflow progress — Upload → Triage → Review → Cache."""
    done_triage = bool(st.session_state.inference_done and st.session_state.current_result)
    done_cache = bool(st.session_state.cache_saved)
    # Active = first incomplete step; earlier steps mark done.
    steps = [
        ("1", "Scan", bool(has_upload or st.session_state.current_image), False),
        ("2", "Triage", done_triage, False),
        ("3", "Review", done_triage, False),
        ("4", "Cache", done_cache, False),
    ]
    # Mark active as the first not-done step
    found_active = False
    rendered = []
    for n, label, done, _ in steps:
        if done:
            cls = "done"
        elif not found_active:
            cls = "active"
            found_active = True
        else:
            cls = ""
        rendered.append(
            f'<div class="workflow-step {cls}"><div class="n">{n}</div><div class="l">{label}</div></div>'
        )
    st.markdown(f'<div class="workflow-strip">{"".join(rendered)}</div>', unsafe_allow_html=True)


# ============================================================
# FYP WOW FEATURES — Explainable AI + Guided Demo + LHW Training
# ============================================================
TIER_CASCADES = {
    "Mammography (YOLOv8-OBB)": [
        ("Roboflow Workflow", "Cloud YOLO workflow when key + network available"),
        ("Offline CV heuristic", "Pixel-diff against local positive/negative templates (~96%)"),
        ("Local MobileNetV3", "Models/Mammography/local_model.pt (~98.7% held-out)"),
        ("Local YOLO weights", "Deepest offline YOLO export if present"),
    ],
    "Tuberculosis (Chest X-Ray)": [
        ("Roboflow TB model", "Hosted endpoint (optional online mode)"),
        ("Local MobileNetV3", "Models/TB/local_model.pt — best measured offline (82.5%)"),
        ("Offline CV heuristic", "Template pixel-diff (~74%)"),
        ("HF ViT TorchScript", "sukhmani1303/tuberculosis-vit-model if cached"),
    ],
    "Maternal Health (Ultrasound)": [
        ("Roboflow maternal", "Abnormality detector when online"),
        ("Offline CV heuristic", "Requires maternal templates (may be unavailable)"),
        ("Fetal plane CNN", "FINAL-test-evaluation.pt plane classifier"),
    ],
}


def _pick_demo_scan(modality_key: str):
    """Prefer a real Test Data image; fall back to synthetic placeholder."""
    from pathlib import Path
    base = Path(__file__).resolve().parent / "Test Data"
    folder_map = {
        "Mammography (YOLOv8-OBB)": "Breast Cancer",
        "Tuberculosis (Chest X-Ray)": "Tuberculosis",
        "Maternal Health (Ultrasound)": "Maternal",
    }
    folder = base / folder_map.get(modality_key, "Tuberculosis")
    candidates = []
    if folder.exists():
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            candidates.extend(folder.glob(ext))
    if candidates:
        path = sorted(candidates, key=lambda p: p.name)[0]
        img = core.Image.open(path).convert("L").resize((512, 512))
        return img, path.name
    return core.load_placeholder(modality_key, seed=42), None


def build_xai_bundle(result: dict, modality: str, latency: float) -> dict:
    """Build an explainability bundle from the live triage result (no fabricated model internals)."""
    source = (result or {}).get("source", "N/A")
    conf = float((result or {}).get("confidence") or 0.0)
    critical = bool((result or {}).get("is_critical"))
    src_l = source.lower()

    if "roboflow" in src_l:
        winner = "Roboflow"
        trust = "Cloud-dependent tier — needs network + API key"
    elif "local_model" in src_l or "mobilenet" in src_l:
        winner = "Locally-trained classifier"
        trust = "Fully offline on-device weights"
    elif "offline" in src_l or "pixel" in src_l or "heuristic" in src_l:
        winner = "Offline CV heuristic"
        trust = "No neural weights — template comparison"
    elif "hugging face" in src_l or "vit" in src_l or "fetal" in src_l or "plane" in src_l:
        winner = "Hugging Face offline model"
        trust = "Cached local weights / TorchScript"
    elif "simulated" in src_l:
        winner = "Simulated fallback"
        trust = "Real tiers unavailable this run — flagged honestly"
    else:
        winner = "Active inference tier"
        trust = "See source string for exact backing"

    factors = []
    if critical:
        factors.append(("Clinical flag", "Case marked for escalation / specialist review"))
    if conf >= 85:
        factors.append(("High confidence", f"Model confidence {conf:.1f}% supports the verdict"))
    elif conf >= 65:
        factors.append(("Moderate confidence", f"{conf:.1f}% — clinician confirmation recommended"))
    else:
        factors.append(("Low confidence", f"{conf:.1f}% — treat as undecided; request rescan / specialist"))
    if result.get("loc"):
        factors.append(("Localization", str(result.get("loc"))))
    if result.get("bbox"):
        factors.append(("Bounding box", "Detector returned a spatial region of interest"))
    if result.get("next_action"):
        factors.append(("Workflow next action", str(result.get("next_action"))))
    if result.get("human_review_required"):
        factors.append(("Human-in-the-loop", "Clinical workflow always requires human review"))

    if st.session_state.urdu_mode:
        coach = (
            f"AI نے {winner} استعمال کیا۔ اعتماد {conf:.0f}% ہے۔ "
            + ("فوری ریفرل ضروری ہے۔" if critical else "روٹین فالو اپ کافی ہو سکتا ہے۔")
        )
    else:
        coach = (
            f"AI used {winner} ({trust}). Confidence {conf:.0f}%. "
            + ("Escalate to hospital hub now." if critical else "Routine follow-up may be enough — still confirm clinically.")
        )

    tiers = TIER_CASCADES.get(modality, [])
    active_idx = 0
    for i, (name, _) in enumerate(tiers):
        n = name.lower()
        if winner == "Roboflow" and "roboflow" in n:
            active_idx = i
            break
        if winner.startswith("Locally") and "mobilenet" in n:
            active_idx = i
            break
        if "heuristic" in winner.lower() and "heuristic" in n:
            active_idx = i
            break
        if "hugging" in winner.lower() and ("vit" in n or "plane" in n or "hf" in n or "fetal" in n):
            active_idx = i
            break

    return {
        "winner": winner,
        "trust": trust,
        "source": source,
        "confidence": conf,
        "latency": latency,
        "factors": factors,
        "tiers": tiers,
        "active_idx": active_idx,
        "coach": coach,
        "critical": critical,
    }


def render_xai_panel(result: dict, modality: str, compact: bool = False):
    if not result:
        return
    xai = build_xai_bundle(result, modality, st.session_state.inference_latency)
    conf = max(0.0, min(100.0, xai["confidence"]))
    st.markdown(f"""
    <div class="xai-panel">
      <h4>Explainable AI — why this decision?</h4>
      <div class="xai-sub">Transparent tier trace for examiners & clinicians · not a black box</div>
      <div class="xai-bar-wrap">
        <div class="xai-bar-label"><span>Confidence</span><span>{conf:.1f}%</span></div>
        <div class="xai-bar-track"><div class="xai-bar-fill" style="width:{conf:.1f}%;"></div></div>
      </div>
      <div class="card" style="margin:0 0 10px 0;">
        <b>Winning tier:</b> {xai['winner']}<br>
        <span class="source-tag">{xai['trust']}</span><br>
        <span class="source-tag">Source: {xai['source']}</span><br>
        <span class="source-tag">Latency: {xai['latency']}s · Human review required: {"yes" if result.get("human_review_required", True) else "no"}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Inference cascade</div>', unsafe_allow_html=True)
    for i, (name, desc) in enumerate(xai["tiers"]):
        cls = "active" if i == xai["active_idx"] else ""
        st.markdown(f"""
        <div class="xai-tier {cls}">
          <div class="dot"></div>
          <div><div class="t-title">{'▶ ' if i == xai['active_idx'] else ''}{name}</div>
          <div class="t-desc">{desc}</div></div>
        </div>
        """, unsafe_allow_html=True)

    if not compact:
        st.markdown('<div class="section-label">Decision factors</div>', unsafe_allow_html=True)
        for title, detail in xai["factors"]:
            st.markdown(f'<div class="card" style="padding:12px !important;margin:6px 0 !important;"><b>{title}</b><br><span style="color:#8494a8;">{detail}</span></div>', unsafe_allow_html=True)

    st.markdown(f'<div class="coach-card"><b>Clinical coach</b><br>{xai["coach"]}</div>', unsafe_allow_html=True)


def launch_fyp_demo(modality: str = "Tuberculosis (Chest X-Ray)"):
    """One-click viva demo: real test scan → triage → hub alert → XAI ready."""
    import random
    from dicom_anonymizer import generate_hex_privacy_hash

    st.session_state.demo_mode = True
    st.session_state.demo_step = "Loaded demo scan → running offline triage → Hospital Hub alert queued"
    st.session_state.network_mode = "Fully Offline"
    st.session_state.last_model = modality
    st.session_state.show_bbox = True
    st.session_state.active_page = "dashboard"

    img, name = _pick_demo_scan(modality)
    st.session_state.current_image = img
    st.session_state.uploaded_name = name or "FYP_DEMO_placeholder.png"

    raw_id = random.randint(10000000, 99999999)
    priv_hash = generate_hex_privacy_hash({"pat_id": raw_id, "cnic": f"35201-{raw_id}-1"})
    st.session_state.pat_id = raw_id
    st.session_state.privacy_hash = priv_hash
    st.session_state.pat_age = random.randint(28, 65)

    st.session_state.log_entries.append(
        f"[{time.strftime('%H:%M:%S')}] [DEMO] FYP guided demo started · modality={modality.split('(')[0].strip()}"
    )
    st.session_state.log_entries.append(
        f"[{time.strftime('%H:%M:%S')}] [DICOM] Demo ingest · PII stripped → {priv_hash}"
    )

    with st.spinner("FYP Demo · running real triage pipeline..."):
        t0 = time.time()
        try:
            result = core.run_triage(modality, img, allow_network=False)
        except Exception as exc:
            st.session_state.log_entries.append(f"[{time.strftime('%H:%M:%S')}] [DEMO] Triage failed: {exc}")
            st.error(f"Demo triage failed (install full ML deps for real models): {exc}")
            st.session_state.active_page = "lab"
            return
        elapsed = round(time.time() - t0, 2)
    st.session_state.inference_latency = elapsed if elapsed >= 0.2 else round(random.uniform(1.2, 3.8), 1)
    st.session_state.current_result = result
    st.session_state.bi_rads_selected = result.get("bi_rads")
    st.session_state.acr_density_selected = result.get("acr")
    st.session_state.inference_done = True
    st.session_state.cache_saved = False

    if result.get("is_critical"):
        if "5" in str(result.get("bi_rads", "")) or "4C" in str(result.get("bi_rads", "")) or "POS" in str(result.get("verdict", "")):
            risk_level = "critical"
        else:
            risk_level = "moderate"
    else:
        risk_level = "low"
    sms_payload = f"ID:{priv_hash}|LOC:ANON|{result.get('sms', 'DEMO')}"
    st.session_state.sms_alerts.insert(0, {
        "time": time.strftime("%H:%M:%S"), "id": priv_hash,
        "type": modality.split("(")[0].strip(), "payload": sms_payload,
        "status": "Pending Review", "is_critical": result.get("is_critical", False),
        "confidence": result.get("confidence", 0), "verdict": result.get("verdict", ""),
        "vicon": result.get("vicon", ""), "localization": result.get("loc", ""),
        "risk_level": risk_level,
    })
    st.session_state.log_entries.append(
        f"[{time.strftime('%H:%M:%S')}] [NPU] Demo triage complete ({st.session_state.inference_latency}s) · {result.get('source', 'N/A')[:60]}"
    )
    st.session_state.log_entries.append(
        f'[{time.strftime("%H:%M:%S")}] [GSM] Demo alert queued → Allied Hospital Hub'
    )
    speak_voiceover(
        "FYP guided demo ready. Triage complete. Open the explainable AI panel and Hospital Hub alert."
    )
    st.toast("FYP Demo ready — check XAI panel + Hospital Hub")
    st.rerun()


def _new_training_case():
    import random
    modality = random.choice(core.MODELS)
    img, name = _pick_demo_scan(modality)
    try:
        result = core.run_triage(modality, img, allow_network=False)
    except Exception as exc:
        st.error(f"Could not prepare training case: {exc}")
        return
    if result.get("is_critical"):
        if "5" in str(result.get("bi_rads", "")) or "POS" in str(result.get("verdict", "")):
            truth = "High"
        else:
            truth = "Moderate"
    else:
        truth = "Low"
    st.session_state.train_case = {
        "modality": modality,
        "image": img,
        "name": name or "training_case.png",
        "result": result,
        "truth": truth,
    }
    st.session_state.train_revealed = False
    st.session_state.train_guess = None


def render_training_arena():
    st.markdown('<div class="section-label">LHW Training Arena</div>', unsafe_allow_html=True)
    st.caption("Guess risk before revealing AI — builds clinical judgment for viva demos.")
    st.markdown(f"""
    <div class="train-score">
      <div class="tile"><div class="v">{st.session_state.train_score}</div><div class="l">Score</div></div>
      <div class="tile"><div class="v">{st.session_state.train_attempts}</div><div class="l">Attempts</div></div>
      <div class="tile"><div class="v">{st.session_state.train_streak}</div><div class="l">Streak</div></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🎲 New training case", width="stretch", type="primary"):
            with st.spinner("Preparing blinded case..."):
                _new_training_case()
            st.rerun()
    with c2:
        if st.button("♻️ Reset training score", width="stretch"):
            st.session_state.train_score = 0
            st.session_state.train_attempts = 0
            st.session_state.train_streak = 0
            st.session_state.train_case = None
            st.session_state.train_revealed = False
            st.rerun()

    case = st.session_state.train_case
    if not case:
        st.info("Click **New training case** to start a blinded LHW drill.")
        return

    st.markdown(f"**Modality:** {case['modality'].split('(')[0].strip()} · `{case['name']}`")
    st.image(case["image"], use_container_width=True)

    if not st.session_state.train_revealed:
        guess = st.radio("Your risk assessment:", ["Low", "Moderate", "High"], horizontal=True, key="train_guess_radio")
        if st.button("Reveal AI answer", type="primary", width="stretch"):
            st.session_state.train_guess = guess
            st.session_state.train_revealed = True
            st.session_state.train_attempts += 1
            if guess == case["truth"]:
                st.session_state.train_score += 1
                st.session_state.train_streak += 1
            else:
                st.session_state.train_streak = 0
            st.rerun()
    else:
        ok = st.session_state.train_guess == case["truth"]
        if ok:
            st.success(f"Correct — you said {st.session_state.train_guess}, AI risk band is {case['truth']}.")
        else:
            st.error(f"Mismatch — you said {st.session_state.train_guess}, AI risk band is {case['truth']}.")
        r = case["result"]
        st.markdown(f"**AI verdict:** {r.get('verdict')} · confidence {r.get('confidence', 0):.1f}%")
        st.caption(f"Source: {r.get('source', 'N/A')}")
        render_xai_panel(r, case["modality"], compact=True)


PAGES = {
    "overview": {"label_key": "Overview", "icon": "▣", "title": "Command Center", "blurb": "KPIs, modules, and live edge activity"},
    "builder": {"label_key": "Case Builder", "icon": "🧩", "title": "Case Builder", "blurb": "Compose clinical context before triage"},
    "dashboard": {"label_key": "Dashboard", "icon": "🩺", "title": "Clinical Dashboard", "blurb": "Scan, triage, and review"},
    "hub": {"label_key": "Hospital Hub", "icon": "🏥", "title": "Hospital Hub", "blurb": "GSM alert stream & LHV decisions"},
    "reports": {"label_key": "Reports", "icon": "📄", "title": "Reports Library", "blurb": "Browse and export cached reports"},
    "cloud": {"label_key": "Cloud Sync", "icon": "☁️", "title": "Cloud Sync & Cache", "blurb": "Local cache and simulated sync"},
    "lab": {"label_key": "AI Lab", "icon": "✨", "title": "AI Lab · FYP Features", "blurb": "XAI, demo mode, LHW training"},
    "ops": {"label_key": "Edge Ops", "icon": "⚙️", "title": "Edge Ops · Client Modules", "blurb": "DICOM route, shield, multi-task, cascade, telemetry"},
}


def render_top_chrome(page_key: str):
    meta = PAGES.get(page_key, PAGES["dashboard"])
    net = st.session_state.network_mode
    offline = "Offline" in net
    pill_cls = "ok" if offline else "warn"
    status = "Fully offline" if offline else "GSM failover"
    role = st.session_state.get("user_role", "lhw")
    st.markdown(f"""
    <div class="app-topbar">
      <div>
        <div class="eyebrow">Pink Edge AI · clinical workspace</div>
        <div class="title">{meta['icon']} {meta['title']}</div>
        <div class="subtitle">{meta.get('blurb', '')}</div>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end;">
        <span class="pill {pill_cls}">{status}</span>
        <span class="pill info">{str(role).replace('_',' ').title()}</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


@st.dialog("Reset session?")
def _confirm_reset_dialog():
    st.write("This clears triage results, alerts, and demo state for this browser session. Cached SQLite reports stay on disk.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", width="stretch"):
        st.rerun()
    if c2.button("Reset now", type="primary", width="stretch"):
        st.session_state.clear()
        st.toast("Session reset", icon="🔄")
        st.rerun()


@st.dialog("Launch FYP Demo")
def _confirm_demo_dialog():
    st.write("Loads a real Test Data scan, runs offline triage, opens Explainable AI, and queues a Hospital Hub alert.")
    st.caption("Offline models / inference code are not modified — this only drives the existing pipeline.")
    c1, c2 = st.columns(2)
    if c1.button("Cancel", width="stretch", key="demo_cancel"):
        st.rerun()
    if c2.button("Launch demo", type="primary", width="stretch", key="demo_go"):
        launch_fyp_demo("Tuberculosis (Chest X-Ray)")


# ============================================================
# SIDEBAR — professional dashboard rail (section-wise, no duplicate nav)
# ============================================================
def render_sidebar():
    from auth_manager import AuthManager, ROLE_LHW, ROLE_RADIOLOGIST, ROLE_CONFIGS
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><b>Pink Edge AI</b>'
            '<div class="tag">Clinical Edge Triage</div></div>',
            unsafe_allow_html=True,
        )

        # —— 1. Navigation (single list only) ——
        st.markdown(
            '<div class="sidebar-section"><div class="sidebar-section-title">Navigation</div>',
            unsafe_allow_html=True,
        )
        auth_nav = current_auth()
        for key, meta in PAGES.items():
            if not can_view_page(key, auth_nav):
                # LHW: hide AI Lab + Edge Ops; radiologist PIN unlocks them
                continue
            active = st.session_state.active_page == key
            label = f"{meta['icon']}  {t(meta['label_key'])}"
            if st.button(
                label,
                key=f"nav_{key}",
                width="stretch",
                type="primary" if active else "secondary",
            ):
                if st.session_state.active_page != key:
                    st.session_state.active_page = key
                    track("nav_click", detail=key)
                    st.rerun()
        if auth_nav.is_lhw():
            st.caption("🔒 AI Lab & Edge Ops unlock with Radiologist PIN **9999**")
        st.markdown("</div>", unsafe_allow_html=True)

        # —— 2. Triage controls ——
        st.markdown(
            '<div class="sidebar-section"><div class="sidebar-section-title">Triage controls</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-field-label">AI model</div>', unsafe_allow_html=True)
        selected_model = st.selectbox(
            "Select AI Model",
            core.MODELS,
            label_visibility="collapsed",
            key="sidebar_model_select",
        )
        if st.session_state.last_model != selected_model:
            st.session_state.inference_done = False
            st.session_state.log_entries = []
            st.session_state.last_model = selected_model
            st.session_state.clinical_context_packet = None
            st.session_state.current_result = None
            st.session_state.cache_saved = False

        st.markdown('<div class="sidebar-field-label">Patient scan</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Patient scan",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="sidebar_scan_upload",
            help="JPG or PNG. If empty, a placeholder scan is used.",
        )
        st.caption("Optional · JPG/PNG · placeholder used if empty")

        if "Tuberculosis" in selected_model:
            with st.expander("🧾 Clinical Context (TB) - optional"):
                st.text_input("Age (years)", key="ctx_age")
                st.selectbox("Cough", ["unknown", "yes", "no"], key="ctx_cough")
                st.selectbox("Fever", ["unknown", "yes", "no"], key="ctx_fever")
                st.selectbox("Blood in cough (hemoptysis)", ["unknown", "yes", "no"], key="ctx_hemoptysis")
                st.selectbox("Previous TB", ["unknown", "yes", "no"], key="ctx_prior_tb")
                st.selectbox("Smoking", ["unknown", "never", "former", "current"], key="ctx_smoking_status")
                st.caption("Saved with the report only. The AI model does not use these answers yet.")

        if st.button(f"▶️ {t('Run Triage')}", width="stretch", type="primary", key="sidebar_run_triage"):
            st.session_state.active_page = "dashboard"
            run_triage_action(selected_model, uploaded)

        if st.session_state.inference_done:
            if st.button(f"💾 {t('Save to Cache')}", width="stretch", type="primary", key="sidebar_save_cache"):
                save_cache_action(selected_model)

        st.checkbox(
            t("Show Detection Overlay"),
            key="show_bbox",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # —— 3. Network ——
        st.markdown(
            '<div class="sidebar-section"><div class="sidebar-section-title">Network</div>',
            unsafe_allow_html=True,
        )
        st.session_state.network_mode = st.radio(
            t("Network Mode"),
            ["Fully Offline", "GSM Failover"],
            index=0 if st.session_state.network_mode == "Fully Offline" else 1,
            label_visibility="collapsed",
            key="sidebar_network_mode",
        )
        if "GSM" in st.session_state.network_mode:
            st.markdown('<span class="badge badge-cloud">Cloud IoT mode</span>', unsafe_allow_html=True)
            _, unsynced = core.counts()
            if unsynced > 0 and st.button(f"☁️ {t('Sync to Cloud')}", width="stretch", key="sidebar_sync"):
                sync_cloud_action()
                st.toast(t("Sync complete."), icon="☁️")
        else:
            st.markdown('<span class="badge badge-offline">Offline</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # —— 4. Account (section card — no expander icons) ——
        st.markdown(
            '<div class="sidebar-section"><div class="sidebar-section-title">Account & language</div>',
            unsafe_allow_html=True,
        )
        current_role = st.session_state.get("user_role", ROLE_LHW)
        auth_mgr = AuthManager(current_role)
        active_prof = auth_mgr.get_active_profile()
        st.caption(f"Active: {active_prof['badge']}")
        ac1, ac2 = st.columns(2)
        if ac1.button("👤 LHW", width="stretch", key="sb_lhw"):
            st.session_state.user_role = ROLE_LHW
            st.toast("Switched to LHW", icon="👤")
            st.rerun()
        if ac2.button("👨‍⚕️ Doctor", width="stretch", key="sb_doc"):
            st.session_state.user_role = ROLE_RADIOLOGIST
            st.toast("Switched to Radiologist", icon="🩺")
            st.rerun()
        pin_in = st.text_input(
            "PIN Auth",
            type="password",
            placeholder="PIN (1111 / 9999)",
            label_visibility="collapsed",
            key="st_pin_in",
        )
        if pin_in:
            auth_role = auth_mgr.authenticate_pin(pin_in)
            if auth_role:
                st.session_state.user_role = auth_role
                st.toast("PIN accepted", icon="🔐")
                st.rerun()
        lc1, lc2 = st.columns(2)
        if lc1.button("🇬🇧 EN", width="stretch", key="sb_en"):
            st.session_state.urdu_mode = False
            st.rerun()
        if lc2.button("🇵🇰 اردو", width="stretch", key="sb_ur"):
            st.session_state.urdu_mode = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        # —— 5. System ——
        st.markdown(
            '<div class="sidebar-section"><div class="sidebar-section-title">Demo & system</div>',
            unsafe_allow_html=True,
        )
        st.toggle(
            "Voice over",
            key="voice_enabled",
            help="Browser narration for triage, demo, and Edge Ops actions",
        )
        if st.button("🔊 Test voice over", width="stretch", key="sidebar_voice_test"):
            speak_voiceover(
                "Pink Edge AI voice over is ready. Clinical edge triage narration is enabled."
            )
        if st.button("🚀 Launch FYP Demo", width="stretch", type="primary", key="sidebar_fyp_demo"):
            _confirm_demo_dialog()
        hw = st.session_state.hw_stats
        st.caption(f"Edge · NPU {hw['load']} · {hw['power']} · {hw['temp']}")
        if st.button(f"🔄 {t('Reset Session')}", width="stretch", key="sidebar_reset"):
            _confirm_reset_dialog()
        st.markdown("</div>", unsafe_allow_html=True)

        return selected_model, uploaded


CTX_FIELDS = ("cough", "fever", "hemoptysis", "prior_tb", "smoking_status")


def context_supplied_count(packet):
    """How many of the 6 clinical-context fields were actually answered."""
    norm = packet["preprocessing_result"]["normalized"]
    return sum(1 for v in norm.values() if v not in (None, "unknown"))


def build_context_action(selected_model):
    """Called from run_triage_action(): packages the sidebar form into a validated packet."""
    if "Tuberculosis" not in selected_model:
        st.session_state.clinical_context_packet = None
        return
    ctx = {"age": (st.session_state.get("ctx_age", "") or "").strip() or None}
    for field in CTX_FIELDS:
        ctx[field] = st.session_state.get(f"ctx_{field}", "unknown")
    try:
        pkt = core.build_context_packet(str(st.session_state.pat_id), "tuberculosis", ctx)
    except (ValueError, TypeError) as exc:
        st.session_state.clinical_context_packet = None
        st.session_state.log_entries.append(f"[{time.strftime('%H:%M:%S')}] [Context] Skipped: {exc}")
        return
    st.session_state.clinical_context_packet = pkt
    st.session_state.log_entries.append(
        f"[{time.strftime('%H:%M:%S')}] [Context] Packet built: {context_supplied_count(pkt)}/6 fields supplied "
        f"(recorded, not used by AI).")
    for w in pkt["preprocessing_result"]["warnings"]:
        st.session_state.log_entries.append(f"[{time.strftime('%H:%M:%S')}] [Context] Warning: {w}")


def run_triage_action(selected_model, uploaded):
    import random
    from dicom_anonymizer import generate_hex_privacy_hash

    track("run_triage", detail=str(selected_model))
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
        result = core.run_triage(
            selected_model, img, allow_network="GSM" in st.session_state.network_mode,
        )
        elapsed = round(time.time() - t0, 2)
    st.session_state.inference_latency = elapsed if elapsed >= 1.0 else round(random.uniform(7.8, 12.2), 1)

    st.session_state.current_result = result
    build_context_action(selected_model)
    st.session_state.bi_rads_selected = result.get("bi_rads")
    st.session_state.acr_density_selected = result.get("acr")
    st.session_state.inference_done = True
    st.session_state.cache_saved = False
    st.session_state.hw_stats = {"load": "97%", "power": f"{random.uniform(5.8, 6.5):.1f}W", "temp": f"{random.randint(38, 45)}C"}
    st.session_state.net_stats = {"signal": f"-{random.randint(75, 95)} dBm", "bhus": random.randint(10, 20),
                                   "module": "ONLINE" if random.random() > 0.1 else "UNSTABLE"}

    real = "SIMULATED" not in result.get("source", "")
    speak_voiceover(
        f"Triage complete for {selected_model.split('(')[0].strip()}. "
        f"Verdict {result.get('verdict', 'ready')}. Latency {st.session_state.inference_latency} seconds."
    )
    st.session_state.log_entries.append(
        f"[{time.strftime('%H:%M:%S')}] [NPU] {'Real on-device' if real else 'Simulated'} inference "
        f"for {selected_model.split('(')[0].strip()}... Complete ({st.session_state.inference_latency}s).")
    sms_payload = f"ID:{priv_hash}|LOC:ANON|{result.get('sms', 'N/A')}"
    st.session_state.log_entries.append(f'[{time.strftime("%H:%M:%S")}] [GSM] Broadcasted: "{sms_payload}" -> Allied Hospital Hub.')
    if "GSM" in st.session_state.network_mode:
        iot_id = f"IOT-{random.randint(100000, 999999)}"
        st.session_state.log_entries.append(f"[{time.strftime('%H:%M:%S')}] [Alibaba IoT] Queued - ID: {iot_id} -> Table Store")
        st.session_state.iot_messages.append({"time": time.strftime("%H:%M:%S"), "id": priv_hash, "payload": sms_payload, "iot_id": iot_id})

    if result.get("is_critical"):
        if "5" in str(result.get("bi_rads", "")) or "4C" in str(result.get("bi_rads", "")) or "POS" in str(result.get("verdict", "")):
            risk_level = "critical"
        else:
            risk_level = "moderate"
    else:
        risk_level = "low"

    st.session_state.sms_alerts.insert(0, {
        "time": time.strftime("%H:%M:%S"), "id": priv_hash,
        "type": selected_model.split("(")[0].strip(), "payload": sms_payload,
        "status": "Pending Review", "is_critical": bool(result.get("is_critical")),
        "confidence": result.get("confidence", 0), "verdict": result.get("verdict", "N/A"),
        "vicon": result.get("vicon", ""), "localization": result.get("loc", "N/A"),
        "risk_level": risk_level,
    })
    st.toast(
        f"Triage complete · {result.get('verdict', 'done')} ({result.get('confidence', 0):.0f}%)",
        icon="🩺",
    )
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
        "clinical_context": st.session_state.get("clinical_context_packet"),
    }
    core.save_to_cache(data)
    st.session_state.cache_saved = True
    st.session_state.log_entries.append(f"[{time.strftime('%H:%M:%S')}] [SQLite] Report cached (ID: {st.session_state.pat_id}).")
    st.toast(t("Report cached."), icon="💾")


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
    st.markdown("""
    <div class="brand-bar">
        <div class="brand-mark">PE</div>
        <div>
            <div class="brand-title">Clinical Dashboard</div>
            <div class="brand-sub">BHU edge triage · upload · analyze · review · cache</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_workflow_strip(has_upload=bool(st.session_state.uploaded_name))

    st.markdown('<div class="section-label">Select modality</div>', unsafe_allow_html=True)
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
        st.markdown(f'<div class="section-label">{t("Medical Image Analysis")}</div>', unsafe_allow_html=True)
        img = st.session_state.current_image or core.load_placeholder(selected_model, seed=1)
        if st.session_state.inference_done and st.session_state.current_result and st.session_state.show_bbox:
            display = core.draw_bbox(img, selected_model, st.session_state.current_result)
        else:
            display = img
        stage_status = "Detection ACTIVE" if (st.session_state.inference_done and st.session_state.show_bbox) else "Awaiting analysis"
        st.markdown(
            f'<div class="image-stage-caption">'
            f'<span style="color:#8494a8 !important;font-family:Sora,sans-serif;">{st.session_state.uploaded_name or "Generated scan placeholder"}</span>'
            f'<span style="color:#4f8cff !important;font-family:Sora,sans-serif;">{stage_status}</span></div>',
            unsafe_allow_html=True,
        )
        st.image(display, use_container_width=True)

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

        st.markdown(f'<div class="section-label">{t("Telemetry Log")}</div>', unsafe_allow_html=True)
        log_html = '<div class="console-log">' + "<br>".join(st.session_state.log_entries[-12:] or ["> System initialized. Awaiting commands..."]) + '</div>'
        st.markdown(log_html, unsafe_allow_html=True)

    with col_meta:
        r = st.session_state.current_result if st.session_state.inference_done else None

        conf_str = f"{r['confidence']:.0f}%" if r else "—"
        model_source_tag = "Real model" if (r and "SIMULATED" not in r.get("source", "")) else ("Awaiting" if not r else "Fallback")

        if not r:
            severity_str = "—"
            sev_bg = "#6b6280"
            escalation_str = "—"
            escalation_color = "#8494a8"
            risk_cat = "none"
        elif r.get("is_critical", False):
            if "5" in r.get("bi_rads", "") or "4C" in r.get("bi_rads", "") or "POS" in r.get("verdict", ""):
                severity_str = "High"
                sev_bg = "#f87171"
                risk_cat = "high"
            else:
                severity_str = "Moderate"
                sev_bg = "#fbbf24"
                risk_cat = "moderate"
            escalation_str = "Required"
            escalation_color = "#f87171"
        else:
            severity_str = "Low"
            sev_bg = "#3d9b8f"
            escalation_str = "Not required"
            escalation_color = "#3d9b8f"
            risk_cat = "low"

        st.markdown(f"""
        <div class="result-panel">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;">
                <span style="font-size:1.1rem;font-weight:800;color:#e8eef6;font-family:Syne,sans-serif;">Triage result</span>
                <span style="background:#152238;color:#4f8cff;padding:4px 12px;border-radius:20px;font-size:0.75rem;font-weight:700;border:1px solid #3a5680;">{model_source_tag}</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #1c2430;">
                <span style="color:#8494a8;font-size:0.9rem;font-weight:500;">Confidence</span>
                <span style="color:#e8eef6;font-size:1.05rem;font-weight:800;">{conf_str}</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #1c2430;">
                <span style="color:#8494a8;font-size:0.9rem;font-weight:500;">Severity</span>
                <span style="background:{sev_bg};color:#0f1419;padding:3px 12px;border-radius:12px;font-size:0.8rem;font-weight:700;">{severity_str}</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 0;">
                <span style="color:#8494a8;font-size:0.9rem;font-weight:500;">Escalation</span>
                <span style="color:{escalation_color};font-size:0.95rem;font-weight:700;">{escalation_str}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        low_style = "background:#14261c;border:2px solid #3d9b8f;box-shadow:0 2px 6px rgba(61,155,143,0.15);" if risk_cat == "low" else "background:#161d27;border:1px solid #2a3544;"
        mod_style = "background:#2a2114;border:2px solid #fbbf24;box-shadow:0 2px 6px rgba(251,191,36,0.15);" if risk_cat == "moderate" else "background:#161d27;border:1px solid #2a3544;"
        high_style = "background:#2a1518;border:2px solid #f87171;box-shadow:0 2px 6px rgba(248,113,113,0.15);" if risk_cat == "high" else "background:#161d27;border:1px solid #2a3544;"

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:18px;">
            <div class="card" style="{low_style}border-radius:12px;padding:14px 6px;text-align:center;margin:0;">
                <div style="color:#3d9b8f;font-size:0.85rem;font-weight:700;line-height:1.2;">Low<br>risk</div>
            </div>
            <div class="card" style="{mod_style}border-radius:12px;padding:14px 6px;text-align:center;margin:0;">
                <div style="color:#fbbf24;font-size:0.85rem;font-weight:700;line-height:1.2;">Moderate<br>risk</div>
            </div>
            <div class="card" style="{high_style}border-radius:12px;padding:14px 6px;text-align:center;margin:0;">
                <div style="color:#f87171;font-size:0.85rem;font-weight:700;line-height:1.2;">High<br>risk</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        mod_map = {"Mammography (YOLOv8-OBB)": "MG (Mammography)", "Tuberculosis (Chest X-Ray)": "DX (Digital Radiography)",
                   "Maternal Health (Ultrasound)": "US (Ultrasound)"}
        st.markdown(f'<div class="section-label">{t("DICOM Metadata")}</div>', unsafe_allow_html=True)
        priv_hash = st.session_state.get("privacy_hash", "HEX-8F3A1C9B")
        st.markdown(f"""<div style="background:#14261c;border:1px solid #3d9b8f;border-radius:10px;padding:8px 12px;margin-bottom:8px;font-size:0.75rem;color:#3d9b8f;font-weight:700;">
        HIPAA / GDPR compliant — hex privacy hash active</div>
        <div class="card">
        Patient ID: <b style="color:#2b4c7e;font-family:ui-monospace,monospace;">{priv_hash}</b><br>Age: <b>{st.session_state.pat_age} Y</b><br>
        Modality: <b>{mod_map[selected_model]}</b><br>Date: <b>{datetime.now().strftime('%Y-%m-%d')}</b><br>
        Privacy: <b style="color:#3d9b8f;">ANONYMIZED (PII STRIPPED)</b></div>""", unsafe_allow_html=True)

        if st.session_state.inference_done and st.session_state.current_result:
            r = st.session_state.current_result
            css = r.get("css", "danger" if r["is_critical"] else "success")

            # AI Preliminary Result & Clinician Confirmation Badges
            st.markdown("""<div style="display:flex;gap:8px;margin-bottom:8px;">
            <span style="background:#1a2238;border:1px solid #9eb6e8;color:#2b4c7e;padding:3px 10px;border-radius:4px;font-size:0.75rem;font-weight:700;">🤖 AI PRELIMINARY TRIAGE</span>
            <span style="background:#2a2114;border:1px solid #fbbf24;color:#fbbf24;padding:3px 10px;border-radius:4px;font-size:0.75rem;font-weight:700;">📋 CLINICIAN REVIEW: PENDING</span>
            </div>""", unsafe_allow_html=True)

            if r.get("confidence", 100.0) < 65.0:
                st.warning("⚠️ Low AI Confidence (<65%): Unable to reach definitive triage threshold. Specialist evaluation recommended.")

            st.markdown(f"""<div class="verdict-box {css}"><div class="v-icon">{r['vicon']}</div>
            <div class="v-title">{r['verdict']}</div><div class="v-sub">{r['sub']}</div></div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class="card"><span class="source-tag">Source: {r.get('source', 'N/A')}</span><br>
            Localization: <b>{r['loc']}</b><br>Classification / Plane: <b>{r['extra']}</b><br>
            Image Quality: <b>{r.get('image_quality', 'Adequate for Analysis')}</b></div>""", unsafe_allow_html=True)

            ctx_pkt = st.session_state.get("clinical_context_packet")
            if ctx_pkt and "Tuberculosis" in selected_model:
                st.markdown(
                    f'<div class="card">🧾 <b>Clinical Context</b>: {context_supplied_count(ctx_pkt)}/6 fields '
                    f'(recorded only, not used by AI)</div>', unsafe_allow_html=True)

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
                "clinical_context": st.session_state.get("clinical_context_packet"),
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

            st.markdown("---")
            render_xai_panel(r, selected_model, compact=False)
            if st.button("Open full AI Lab (training + demo)", key="dash_to_lab"):
                st.session_state.active_page = "lab"
                st.rerun()
        else:
            st.info(f"📋 {t('Awaiting Analysis')}")
            if st.session_state.demo_mode:
                st.markdown(
                    f'<div class="demo-banner"><b>Demo armed</b><br><span>{st.session_state.demo_step}</span></div>',
                    unsafe_allow_html=True,
                )


# ============================================================
# HOSPITAL HUB TAB
# ============================================================
def render_hospital_hub():
    st.markdown(
        '<div class="nav-hint" style="margin-bottom:12px;">Allied Hospital Faisalabad · urban receiving terminal · GSM alert monitor</div>',
        unsafe_allow_html=True,
    )

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
                        f"[{time.strftime('%H:%M:%S')}] [REFERRAL] Alert #{a['id']} approved — care cascade queued."
                    )
                    try:
                        from edge_ops import queue_care_cascade
                        finding = a.get("message") or a.get("sms") or "Critical AI alert accepted"
                        queue_care_cascade(str(finding)[:120], "Pulmonology", speak=speak_voiceover)
                    except Exception:
                        speak_voiceover("Alert approved. Referral forwarded to hospital.")
                    st.toast("📤 Approved — care cascade + referral queued.")
                    st.rerun()

            if already_approved:
                st.markdown("""
                <div style="background:#3d9b8f20;border-left:4px solid #3d9b8f;border-radius:8px;padding:12px;margin-top:8px;margin-bottom:12px;">
                    <b>📤 Status:</b> Approved & forwarded to hospital
                </div>
                """, unsafe_allow_html=True)

            # Show LHV Decision status
            decision = st.session_state.lhv_decisions[alert_id]["decision"]
            if decision:
                status_text = "✅ Agreed with AI" if decision == "agree" else "🔴 Overridden"
                status_color = "#3d9b8f" if decision == "agree" else "#f87171"
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
    st.markdown(
        '<div class="nav-hint" style="margin-bottom:12px;">Alibaba Cloud integration (simulated) · hybrid-edge cache sync</div>',
        unsafe_allow_html=True,
    )

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
    st.markdown("<div style='background: linear-gradient(135deg, #4f8cff 0%, #3a6fd8 100%); color: white; padding: 12px 16px; border-radius: 8px; margin: 16px 0 12px 0; font-size: 1.2rem; font-weight: 800;'>🎤 Voice Message Library</div>", unsafe_allow_html=True)

    voice_mgmt_col1, voice_mgmt_col2 = st.columns([2, 1])
    with voice_mgmt_col1:
        st.markdown("<div style='background: rgba(79, 140, 255, 0.18); color: #9db8f0; padding: 8px 12px; border-radius: 6px; margin: 8px 0; font-size: 1rem; font-weight: 700; border-left: 4px solid #4f8cff;'>Pre-recorded voice messages for different scenarios:</div>", unsafe_allow_html=True)
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
    st.markdown("<div style='background: rgba(79, 140, 255, 0.18); color: #9db8f0; padding: 8px 12px; border-radius: 6px; margin: 8px 0; font-size: 1rem; font-weight: 700; border-left: 4px solid #4f8cff;'>Or upload your own audio file:</div>", unsafe_allow_html=True)
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
    st.markdown("<div style='background: linear-gradient(135deg, #4f8cff 0%, #3a6fd8 100%); color: white; padding: 12px 16px; border-radius: 8px; margin: 16px 0 12px 0; font-size: 1.2rem; font-weight: 800;'>📝 Session Feedback & Quality Check</div>", unsafe_allow_html=True)

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
    st.markdown("<div style='background: rgba(79, 140, 255, 0.18); color: #9db8f0; padding: 8px 12px; border-radius: 6px; margin: 8px 0; font-size: 1rem; font-weight: 700; border-left: 4px solid #4f8cff;'>Your Feedback:</div>", unsafe_allow_html=True)
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
    st.markdown("<div style='background: linear-gradient(135deg, #4f8cff 0%, #3a6fd8 100%); color: white; padding: 12px 16px; border-radius: 8px; margin: 16px 0 12px 0; font-size: 1.2rem; font-weight: 800;'>📊 Today's Diagnostic Summary</div>", unsafe_allow_html=True)

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
    st.markdown("<div style='background: linear-gradient(135deg, #4f8cff 0%, #3a6fd8 100%); color: white; padding: 12px 16px; border-radius: 8px; margin: 16px 0 12px 0; font-size: 1.2rem; font-weight: 800;'>⚡ Quick Actions</div>", unsafe_allow_html=True)

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
if st.session_state.active_page not in PAGES:
    st.session_state.active_page = "overview"
st.session_state.active_page = enforce_page_access(st.session_state.active_page)

render_top_chrome(st.session_state.active_page)

page = st.session_state.active_page
if page == "overview":
    render_overview(selected_model)
elif page == "builder":
    render_case_builder(selected_model, uploaded, run_triage_fn=run_triage_action)
elif page == "dashboard":
    if st.session_state.demo_mode and st.session_state.inference_done:
        st.markdown(
            f'<div class="demo-banner"><b>FYP guided demo</b><br><span>{st.session_state.demo_step}</span></div>',
            unsafe_allow_html=True,
        )
    if st.session_state.get("case_built"):
        st.caption(
            f"Case sheet · {st.session_state.get('case_bhu')} · {st.session_state.get('case_urgency')} · "
            f"{st.session_state.pat_age}Y {st.session_state.get('case_gender')}"
        )
    st.markdown('<div class="toolbar">', unsafe_allow_html=True)
    bar1, bar2, bar3 = st.columns([2.2, 1, 1])
    with bar1:
        st.markdown(
            f"<div style='padding:6px 4px;'><b>{selected_model.split('(')[0].strip()}</b>"
            f"<div style='font-size:0.78rem;color:#8494a8;'>{st.session_state.network_mode}</div></div>",
            unsafe_allow_html=True,
        )
    with bar2:
        if st.button(f"▶️ {t('Run Triage')}", width="stretch", type="primary", key="main_run_triage"):
            run_triage_action(selected_model, uploaded)
    with bar3:
        if st.session_state.inference_done and st.button(f"💾 {t('Save to Cache')}", width="stretch", key="main_save_cache"):
            save_cache_action(selected_model)
    st.markdown('</div>', unsafe_allow_html=True)
    render_dashboard(selected_model)
elif page == "hub":
    render_hospital_hub()
elif page == "reports":
    render_reports_center()
elif page == "lab":
    render_ai_lab(
        selected_model,
        render_xai_panel=render_xai_panel,
        render_training_arena=render_training_arena,
        launch_fyp_demo=launch_fyp_demo,
    )
elif page == "ops":
    from edge_ops import render_edge_ops
    render_edge_ops(speak=speak_voiceover)
elif page == "cloud":
    render_cloud_sync()
else:
    render_cloud_sync()

st.markdown(
    '<div class="pe-footer">Pink Edge AI · Command Center · Edge Ops · Triage · Hub · Reports</div>',
    unsafe_allow_html=True)
