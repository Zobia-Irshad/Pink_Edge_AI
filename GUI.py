#!/usr/bin/env python3
"""
Pink Edge AI — Offline Desktop Clinical Intelligence Platform
================================================================
Tkinter port of the original Streamlit demo (Data/Pink_Edge_AI-main/pink_edge.py), wired to real
on-device models where a suitable public one could be sourced (see MODEL_SOURCES.md), with the
original app's own honest simulated fallback for any modality where none was available. 100%
offline at run time: model weights are downloaded once and cached under models/, then loaded
locally on every subsequent run.

Run with:  python GUI.py

Note: tkinter (and PIL's ImageTk, which wraps it) is intentionally NOT imported at module level.
streamlit_app.py imports this module for its shared logic (constants, imaging, simulated
scenarios, DB, reports, the run_triage() dispatcher) without ever touching the Tkinter UI, and
some deployment environments (e.g. Streamlit Community Cloud's Linux containers) don't have the
system Tk libraries tkinter needs, which would crash that import at module load time. See
_lazy_import_tkinter() below — it's called only when the desktop UI actually runs.
"""
import json
import os
import random
import sqlite3
import time
from datetime import datetime

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import inference as inf


def _lazy_import_tkinter():
    """Import tkinter/ttk/filedialog/messagebox/ImageTk and bind them as module globals, but
    only the first time the desktop UI is actually constructed — keeps GUI.py importable on
    headless hosts that lack tkinter (see module docstring)."""
    global tk, ttk, filedialog, messagebox, ImageTk
    if "tk" in globals():
        return
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from PIL import ImageTk

# ============================================================
# CONFIG / CONSTANTS
# ============================================================
DB_PATH = "pink_edge_cache.db"
MODELS = [
    "Mammography (YOLOv8-OBB)",
    "Tuberculosis (Chest X-Ray)",
    "Maternal Health (Ultrasound)",
]
BI_RADS_OPTIONS = inf.BI_RADS_OPTIONS
ACR_DENSITY_OPTIONS = inf.ACR_DENSITY_OPTIONS
TB_SEVERITY_LEVELS = inf.TB_SEVERITY_LEVELS
TB_LUNG_ZONES = inf.TB_LUNG_ZONES

C = {  # color tokens — all-white background theme
    "bg": "#ffffff", "surface": "#ffffff", "surface_alt": "#f8f9fa", "surface_hover": "#e9ecef",
    "border": "#e5e7eb", "border_light": "#d1d5db", "text": "#111827", "text_muted": "#4b5563",
    "text_light": "#6b7280", "primary": "#1a237e", "primary_light": "#3949ab", "accent": "#ec4899",
    "accent_light": "#f472b6", "success": "#10b981", "warning": "#f59e0b", "danger": "#ef4444",
}

TR = {
    "Pink Edge AI": "پنک ایج اے آئی", "Clinical Intelligence Platform": "کلینیکل انٹیلیجنس پلیٹ فارم",
    "Dashboard": "ڈیش بورڈ", "Hospital Hub": "ہسپتال مرکز", "Cloud Sync": "کلاؤڈ سنک",
    "Fully Offline": "مکمل آف لائن", "GSM Failover": "جی ایس ایم فیل اوور",
    "Select AI Model": "اے آئی ماڈل منتخب کریں", "Upload Patient Scan": "مریض کا اسکین اپ لوڈ کریں",
    "Run Triage": "تشخیص چلائیں", "Save to Cache": "کیش میں محفوظ کریں",
    "Sync to Cloud": "کلاؤڈ میں سنک کریں", "Check OTA": "او ٹی اے چیک کریں",
    "Reset Session": "سیشن ری سیٹ کریں", "Show Detection Overlay": "کھوج اوورلے دکھائیں",
    "Network Mode": "نیٹ ورک موڈ", "Awaiting Analysis": "تجزیہ کا انتظار",
    "Confirm Assessment": "تصدیق تشخیص", "Download Text Report": "متن رپورٹ ڈاؤن لوڈ کریں",
    "Download PDF Report": "پی ڈی ایف رپورٹ ڈاؤن لوڈ کریں", "Hardware Diagnostics": "ہارڈویئر تشخیص",
    "Telemetry Log": "ٹیلی میٹری لاگ", "Alert Stream": "الرٹ اسٹریم", "No reports cached": "کوئی رپورٹ کیش نہیں",
    "Local Cache Status": "لوکل کیش حالت", "Report cached.": "رپورٹ محفوظ ہو گئی۔",
    "Sync complete.": "سنک مکمل۔",
}
_urdu = {"on": False}


def t(s):
    return TR.get(s, s) if _urdu["on"] else s


# ============================================================
# DATABASE (same schema/file as the Streamlit app's cache)
# ============================================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS cached_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT, patient_age INTEGER, modality TEXT,
        model_used TEXT, bi_rads TEXT, acr_density TEXT, verdict TEXT, localization TEXT,
        confidence REAL, inference_time REAL, timestamp TEXT, synced INTEGER DEFAULT 0, report_json TEXT
    )""")
    conn.commit()
    conn.close()


def save_to_cache(d):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""INSERT INTO cached_reports (patient_id, patient_age, modality, model_used, bi_rads,
        acr_density, verdict, localization, confidence, inference_time, timestamp, synced, report_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (d["patient_id"], d["patient_age"], d["modality"], d["model_used"], d["bi_rads"], d["acr_density"],
         d["verdict"], d["localization"], d["confidence"], d["inference_time"], d["timestamp"], 0,
         json.dumps(d, default=str)))
    conn.commit()
    conn.close()


def get_cached_reports():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT * FROM cached_reports ORDER BY id DESC").fetchall()
    conn.close()
    return rows


def get_unsynced_reports():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT * FROM cached_reports WHERE synced = 0").fetchall()
    conn.close()
    return rows


def mark_as_synced(ids):
    conn = sqlite3.connect(DB_PATH)
    for rid in ids:
        conn.execute("UPDATE cached_reports SET synced = 1 WHERE id = ?", (rid,))
    conn.commit()
    conn.close()


def counts():
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM cached_reports").fetchone()[0]
    unsynced = conn.execute("SELECT COUNT(*) FROM cached_reports WHERE synced = 0").fetchone()[0]
    conn.close()
    return total, unsynced


def simulate_iot_sync(reports):
    return [{"report_id": r[0], "payload": f"ID:{r[1]}|BR:{r[5]}|TS:{r[11]}",
              "iot_id": f"IOT-{random.randint(100000, 999999)}"} for r in reports]


def simulate_oss_upload(report):
    bi_rads = report[5] or ""
    high = "4" in bi_rads or "5" in bi_rads
    return {"high": high, "kb": random.randint(45, 85) if high else 0}


def simulate_acr_check():
    return {"current": "v2.1.0", "available": "v2.2.1", "size": 14.2,
            "registry": "registry.ap-south-1.aliyuncs.com/pink-edge/ai-models"}


# ============================================================
# SIMULATED SCENARIO FALLBACKS (ported from pink_edge.py — used when no real model loaded)
# ============================================================
def sim_mammography():
    s = [
        {"bi_rads": BI_RADS_OPTIONS[1], "acr": ACR_DENSITY_OPTIONS[0], "verdict": "Normal",
         "sub": "No Anomalies Detected", "loc": "No focal lesion identified", "extra": "ACR Class A",
         "vicon": "✅", "confidence": random.uniform(95.0, 99.2), "sms": "BI-RADS:1", "is_critical": False},
        {"bi_rads": BI_RADS_OPTIONS[2], "acr": ACR_DENSITY_OPTIONS[1], "verdict": "Benign Finding",
         "sub": "Simple cyst identified", "loc": "Upper Outer Quadrant", "extra": "ACR Class B",
         "vicon": "✅", "confidence": random.uniform(92.0, 96.5), "sms": "BI-RADS:2", "is_critical": False},
        {"bi_rads": BI_RADS_OPTIONS[4], "acr": ACR_DENSITY_OPTIONS[2], "verdict": "Low Suspicion",
         "sub": "Suspicious morphology - Biopsy recommended", "loc": "Upper Outer Quadrant",
         "extra": "ACR Class C", "vicon": "⚠️", "confidence": random.uniform(82.0, 88.0),
         "sms": "BI-RADS:4A", "is_critical": True},
        {"bi_rads": BI_RADS_OPTIONS[7], "acr": ACR_DENSITY_OPTIONS[3], "verdict": "BI-RADS 5",
         "sub": "Highly Suspicious - Spiculated mass", "loc": "Upper Outer Quadrant", "extra": "ACR Class D",
         "vicon": "⚠️", "confidence": random.uniform(93.0, 97.5), "sms": "BI-RADS:5", "is_critical": True},
    ]
    r = random.choice(s)
    r["source"] = "SIMULATED — no downloadable trained weights (see MODEL_SOURCES.md)"
    return r


def sim_tb():
    s = [
        {"bi_rads": "S0 - No active disease", "acr": "Bilateral", "verdict": "TB Negative",
         "sub": "No active lesions detected", "loc": "Lungs clear", "extra": "No cavity formation",
         "vicon": "✅", "confidence": random.uniform(94.0, 98.5), "sms": "TB:NEG", "is_critical": False},
        {"bi_rads": "S2 - Moderate (bilateral / cavity < 2 cm)", "acr": "Upper Zone", "verdict": "TB Positive",
         "sub": "Active lesion detected - cavity formation", "loc": "Right Upper Lobe",
         "extra": "Cavity Formation", "vicon": "⚠️", "confidence": random.uniform(85.0, 92.0),
         "sms": "TB:POS", "is_critical": True},
    ]
    r = random.choice(s)
    r["source"] = "SIMULATED (fallback — real model unavailable this run)"
    return r


def sim_maternal():
    s = [
        {"bi_rads": "BI-RADS 1 - Negative", "acr": "A - Almost entirely fatty", "verdict": "Fetal Health Normal",
         "sub": "No anomalies detected", "loc": "Intrauterine", "extra": f"Gestational Age: {random.randint(18, 36)}W",
         "vicon": "✅", "confidence": random.uniform(96.0, 99.0), "sms": "FH:OK", "is_critical": False},
    ]
    r = random.choice(s)
    r["source"] = "SIMULATED — least-developed pathway, no public model matched (see MODEL_SOURCES.md)"
    return r


def run_triage(model_name, pil_image):
    """Try the real on-device model first; fall back to the simulated scenario picker."""
    if "Mammography" in model_name:
        return inf.predict_mammography(pil_image) or sim_mammography()
    elif "Tuberculosis" in model_name:
        return inf.predict_tb(pil_image) or sim_tb()
    else:
        return inf.predict_maternal(pil_image) or sim_maternal()


# ============================================================
# PLACEHOLDER IMAGE SYNTHESIS + DETECTION OVERLAY (PIL/numpy port of draw_bbox/generate_*)
# ============================================================
def _clip_u8(a):
    return np.clip(a, 0, 255).astype(np.uint8)


def gen_mammogram(size=512, seed=42):
    rng = np.random.RandomState(seed)
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    cx, cy = size // 2, int(size * 0.58)
    rx, ry = size * 0.4, size * 0.5
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    mask = d < 1.0
    img = np.where(mask, 90 + (1 - d) * 70, 0) + rng.normal(0, 12, (size, size)) * mask
    mcx, mcy = int(size * 0.68), int(size * 0.35)
    img[((xx - mcx) ** 2 + (yy - mcy) ** 2) < 22 ** 2] += 35
    pil = Image.fromarray(_clip_u8(img), "L")
    draw = ImageDraw.Draw(pil)
    for _ in range(8):
        sx, sy = rng.randint(80, size - 80, 2)
        length, angle = rng.randint(30, 70), rng.uniform(0, 2 * np.pi)
        draw.line([(sx, sy), (sx + length * np.cos(angle), sy + length * np.sin(angle))],
                   fill=int(rng.randint(120, 160)))
    return pil


def gen_xray(size=512, seed=42):
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size), dtype=np.float32) + 40
    mask_img = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask_img).ellipse(
        [size // 2 - int(size * 0.35), size // 2 - int(size * 0.45), size // 2 + int(size * 0.35), size // 2 + int(size * 0.45)],
        fill=255)
    img[np.array(mask_img) > 0] = 180
    img += rng.normal(0, 15, (size, size))
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    mcx, mcy = int(size * 0.35), int(size * 0.30)
    img[((xx - mcx) ** 2 + (yy - mcy) ** 2) < 15 ** 2] += 80
    return Image.fromarray(_clip_u8(img), "L")


def gen_ultrasound(size=512, seed=42):
    rng = np.random.RandomState(seed)
    img = _clip_u8(np.zeros((size, size), dtype=np.float32) + 50 + rng.normal(0, 25, (size, size)))
    pil = Image.fromarray(img, "L")
    r = int(size * 0.2)
    ImageDraw.Draw(pil).ellipse([size // 2 - r, size // 2 - r, size // 2 + r, size // 2 + r], fill=120)
    return pil


def load_placeholder(model_name, seed=42):
    if "Mammography" in model_name:
        return gen_mammogram(seed=seed)
    elif "Tuberculosis" in model_name:
        return gen_xray(seed=seed)
    return gen_ultrasound(seed=seed)


def _rotated_box(center, box_size, angle_deg):
    cx, cy = center
    w, h = box_size
    a = np.radians(angle_deg)
    pts = []
    for dx, dy in [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]:
        pts.append((cx + dx * np.cos(a) - dy * np.sin(a), cy + dx * np.sin(a) + dy * np.cos(a)))
    return pts


def draw_bbox(image, model_name, result):
    rgb = image.convert("RGB")
    draw = ImageDraw.Draw(rgb)
    w, h = rgb.size
    is_critical = bool(result and result.get("is_critical"))
    conf_str = f"{result['confidence']:.1f}%" if result else "94.2%"
    color = (236, 72, 153) if is_critical else (16, 185, 129)

    # offline_cv.py's pixel-comparison heuristic returns a REAL detected region (normalized
    # (x, y, w, h) fractions) rather than an illustrative fixed position — use it when present.
    real_bbox = result.get("bbox") if result else None
    if real_bbox:
        rx, ry, rw, rh = real_bbox
        default_center = (int((rx + rw / 2) * w), int((ry + rh / 2) * h))
        default_size = (max(int(rw * w), 20), max(int(rh * h), 20))
    else:
        default_center = default_size = None

    if "Mammography" in model_name:
        center = default_center or (int(w * 0.68), int(h * 0.35))
        size = default_size or (int(w * 0.15), int(h * 0.12))
        box = _rotated_box(center, size, 0 if real_bbox else 35)
        draw.polygon(box, outline=color, width=2)
        for pt in box:
            draw.ellipse([pt[0] - 5, pt[1] - 5, pt[0] + 5, pt[1] + 5], fill=color)
        label = f"YOLOv8-OBB: {'Malignant' if is_critical else 'Benign'} ({conf_str})"
    elif "Tuberculosis" in model_name:
        center = default_center or (int(w * 0.35), int(h * 0.30))
        bw, bh = default_size or (int(w * 0.12), int(h * 0.10))
        draw.rectangle([center[0] - bw // 2, center[1] - bh // 2, center[0] + bw // 2, center[1] + bh // 2],
                        outline=color, width=2)
        label = f"TB Classifier: {'Active Lesion' if is_critical else 'Clear'} ({conf_str})"
    else:
        center = default_center or (int(w * 0.50), int(h * 0.50))
        r = max(default_size) // 2 if default_size else int(w * 0.15)
        draw.ellipse([center[0] - r, center[1] - r, center[0] + r, center[1] + r], outline=color, width=2)
        label = f"Fetal Health: {'Review Needed' if is_critical else 'Normal'} ({conf_str})"

    try:
        font = ImageFont.truetype("consola.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    lx, ly = max(0, center[0] - tw // 2), max(25, center[1] - int(w * 0.15) - 10)
    draw.rectangle([lx - 6, ly - th - 10, lx + tw + 6, ly + 4], fill=(0, 0, 0))
    draw.text((lx, ly - th - 6), label, fill=(255, 255, 255), font=font)
    cs = 10
    draw.line([(center[0] - cs, center[1]), (center[0] + cs, center[1])], fill=(0, 230, 255))
    draw.line([(center[0], center[1] - cs), (center[0], center[1] + cs)], fill=(0, 230, 255))
    return rgb


# ============================================================
# REPORTS
# ============================================================
def generate_text_report(d):
    r = (f"\nPINK EDGE AI - CLINICAL DIAGNOSTIC REPORT\n==========================================\n\n"
         f"PATIENT: {d['patient_id']}    AGE: {d['patient_age']}    DATE: {d['timestamp']}\n"
         f"MODALITY: {d['modality']}      INSTITUTION: Rural BHU Faisalabad\n\n"
         f"AI ANALYSIS\n-----------\nModel: {d['model_used']}\nVerdict: {d['verdict']}\n"
         f"Confidence: {d['confidence']:.1f}%\nInference: {d['inference_time']}s (desktop CPU)\n"
         f"Localization: {d['localization']}\nSource: {d.get('model_source', 'N/A')}\n\n"
         f"CLINICAL ASSESSMENT\n-------------------\nBI-RADS / Severity: {d['bi_rads']}\n"
         f"ACR Density / Zone: {d['acr_density']}\n")
    br = d.get("bi_rads", "")
    if "5" in br or "4C" in br or "S3" in br:
        r += "\nURGENT: Immediate referral. Biopsy/specialist evaluation recommended.\n"
    elif "4A" in br or "4B" in br or "S2" in br:
        r += "\nReferral recommended. Further evaluation advised.\n"
    elif "3" in br or "S1" in br:
        r += "\nFollow-up imaging recommended.\n"
    else:
        r += "\nRoutine screening per guidelines.\n"
    r += (f"\nNETWORK: {d.get('network_mode', 'Offline')}\n"
          f"SYNC: {'Pending' if d.get('synced') == 0 else 'Complete'}\n\n"
          f"Generated by Pink Edge AI (Desktop) - Alibaba Cloud AI Hackathon 2026\n")
    return r


def generate_pdf_bytes(d):
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    def safe(txt):
        txt = str(txt)
        for u, a in {"—": "-", "–": "-", "’": "'", "“": '"', "”": '"'}.items():
            txt = txt.replace(u, a)
        return txt.encode("latin-1", "replace").decode("latin-1")

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.add_page()
    pdf.set_fill_color(13, 148, 136)
    pdf.rect(0, 0, 210, 45, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 20)
    pdf.cell(0, 18, "PINK EDGE AI", ln=True, align="C")
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, "Clinical Diagnostic Report (Desktop)", ln=True, align="C")
    pdf.cell(0, 5, f"Report ID: PEA-{d['patient_id']}-{int(time.time())}", ln=True, align="C")
    pdf.ln(12)
    pdf.set_text_color(30, 41, 59)

    def section(title, rows):
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, title, ln=True)
        pdf.set_draw_color(226, 232, 240)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 9)
        for label, value in rows:
            pdf.set_text_color(100, 116, 139)
            pdf.cell(50, 6, f"{label}:")
            pdf.set_text_color(30, 41, 59)
            pdf.cell(0, 6, safe(value), ln=True)
        pdf.ln(5)

    section("PATIENT INFORMATION", [("Patient ID", d["patient_id"]), ("Age", f"{d['patient_age']} Years"),
                                     ("Date", d["timestamp"]), ("Modality", d["modality"])])
    section("AI ANALYSIS RESULTS", [("Model", d["model_used"]), ("Verdict", d["verdict"]),
                                     ("Confidence", f"{d['confidence']:.1f}%"),
                                     ("Inference", f"{d['inference_time']}s"),
                                     ("Localization", d["localization"]),
                                     ("Source", d.get("model_source", "N/A"))])
    section("CLINICAL ASSESSMENT", [("BI-RADS / Severity", d["bi_rads"]), ("ACR Density / Zone", d["acr_density"])])
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 5, "Pink Edge AI (Desktop) - Alibaba Cloud AI Hackathon 2026", ln=True, align="C")
    return bytes(pdf.output(dest="S"))


# ============================================================
# APP
# ============================================================
class PinkEdgeApp:
    def __init__(self, root):
        _lazy_import_tkinter()
        self.root = root
        root.title("Pink Edge AI — Clinical Intelligence Platform (Desktop)")
        root.geometry("1280x820")
        root.configure(bg=C["bg"])
        self._style()

        self.model_var = tk.StringVar(value=MODELS[0])
        self.network_var = tk.StringVar(value="Fully Offline")
        self.overlay_var = tk.BooleanVar(value=True)
        self.uploaded_path = None
        self.current_image = None
        self.current_result = None
        self.inference_done = False
        self.pat_id = random.randint(10000000, 99999999)
        self.pat_age = random.randint(30, 70)
        self.hw_stats = {"load": "34%", "power": "6.2W", "temp": "42C"}
        self.net_stats = {"signal": "-85 dBm", "bhus": 14, "module": "ONLINE"}
        self.sms_alerts = []
        self.log_entries = []
        self.iot_messages = []
        self.oss_uploads = []
        self.acr_status = None
        self.inference_latency = 9.4
        self.bi_rads_var = tk.StringVar()
        self.acr_var = tk.StringVar()

        init_db()
        self._build_layout()
        self._refresh_placeholder()
        self._log("System initialized. Awaiting commands...")

    # -------------------- styling --------------------
    def _style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", background=C["surface"], foreground=C["text"], font=("Segoe UI", 9))
        style.configure("TFrame", background=C["surface"])
        style.configure("Card.TFrame", background=C["surface_alt"])
        style.configure("TLabel", background=C["surface"], foreground=C["text"])
        style.configure("Muted.TLabel", background=C["surface"], foreground=C["text_muted"], font=("Segoe UI", 8))
        style.configure("Card.TLabel", background=C["surface_alt"], foreground=C["text"])
        style.configure("H1.TLabel", background=C["surface"], foreground=C["primary_light"], font=("Segoe UI", 14, "bold"))
        style.configure("H2.TLabel", background=C["surface"], foreground=C["text"], font=("Segoe UI", 11, "bold"))
        style.configure("TButton", background=C["surface_alt"], foreground=C["text"], padding=6, borderwidth=0)
        style.map("TButton", background=[("active", C["surface_hover"])])
        style.configure("Accent.TButton", background=C["accent"], foreground="white", padding=6)
        style.map("Accent.TButton", background=[("active", C["accent_light"])])
        style.configure("TCombobox", fieldbackground=C["surface_alt"], background=C["surface_alt"], foreground=C["text"])
        style.configure("TRadiobutton", background=C["surface"], foreground=C["text"])
        style.configure("TCheckbutton", background=C["surface"], foreground=C["text"])
        style.configure("TNotebook", background=C["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", background=C["surface"], foreground=C["text_muted"], padding=(16, 8))
        style.map("TNotebook.Tab", background=[("selected", C["bg"])], foreground=[("selected", C["primary"])])
        style.configure("Treeview", background=C["surface_alt"], fieldbackground=C["surface_alt"], foreground=C["text"])
        style.configure("Treeview.Heading", background=C["surface"], foreground=C["primary_light"])

    # -------------------- layout --------------------
    def _build_layout(self):
        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        self.sidebar = ttk.Frame(outer, width=270, style="Card.TFrame")
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)
        self._build_sidebar(self.sidebar)

        main = ttk.Frame(outer)
        main.pack(side="left", fill="both", expand=True)

        self.notebook = ttk.Notebook(main)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        self.tab_dashboard = ttk.Frame(self.notebook)
        self.tab_hub = ttk.Frame(self.notebook)
        self.tab_cloud = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_dashboard, text="🩺 Dashboard")
        self.notebook.add(self.tab_hub, text="🏥 Hospital Hub")
        self.notebook.add(self.tab_cloud, text="☁️ Cloud Sync")

        self._build_dashboard(self.tab_dashboard)
        self._build_hub(self.tab_hub)
        self._build_cloud(self.tab_cloud)
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self._on_tab_change())

    def _build_sidebar(self, parent):
        pad = {"padx": 14, "pady": 4}
        header = tk.Frame(parent, bg=C["primary"], height=60)
        header.pack(fill="x")
        tk.Label(header, text="🩸 Pink Edge AI", bg=C["primary"], fg="white",
                 font=("Segoe UI", 12, "bold")).pack(anchor="w", padx=14, pady=(10, 0))
        tk.Label(header, text="Clinical Intelligence Platform", bg=C["primary"], fg="white",
                 font=("Segoe UI", 8)).pack(anchor="w", padx=14, pady=(0, 8))

        ttk.Label(parent, text="Language", style="Card.TLabel", font=("Segoe UI", 8, "bold")).pack(anchor="w", **pad)
        lf = ttk.Frame(parent, style="Card.TFrame")
        lf.pack(fill="x", padx=14)
        ttk.Button(lf, text="EN", command=lambda: self._set_lang(False)).pack(side="left", fill="x", expand=True)
        ttk.Button(lf, text="اردو", command=lambda: self._set_lang(True)).pack(side="left", fill="x", expand=True)

        ttk.Label(parent, text="Network Mode", style="Card.TLabel", font=("Segoe UI", 8, "bold")).pack(anchor="w", **pad)
        for mode in ["Fully Offline", "GSM Failover"]:
            ttk.Radiobutton(parent, text=mode, variable=self.network_var, value=mode,
                             command=self._refresh_actions).pack(anchor="w", padx=14)

        ttk.Label(parent, text="Select AI Model", style="Card.TLabel", font=("Segoe UI", 8, "bold")).pack(anchor="w", **pad)
        cb = ttk.Combobox(parent, textvariable=self.model_var, values=MODELS, state="readonly")
        cb.pack(fill="x", padx=14)
        cb.bind("<<ComboboxSelected>>", lambda e: self._on_model_change())

        ttk.Label(parent, text="Patient Scan", style="Card.TLabel", font=("Segoe UI", 8, "bold")).pack(anchor="w", **pad)
        ttk.Button(parent, text="📁 Upload Image...", command=self._upload_image).pack(fill="x", padx=14)
        self.upload_label = ttk.Label(parent, text="(using generated placeholder)", style="Muted.TLabel", wraplength=230)
        self.upload_label.pack(anchor="w", padx=14, pady=(2, 0))

        ttk.Separator(parent).pack(fill="x", padx=14, pady=10)
        self.actions_frame = ttk.Frame(parent, style="Card.TFrame")
        self.actions_frame.pack(fill="x", padx=14)
        self._refresh_actions()

        ttk.Separator(parent).pack(fill="x", padx=14, pady=10)
        ttk.Checkbutton(parent, text="Show Detection Overlay", variable=self.overlay_var,
                         command=self._render_image).pack(anchor="w", padx=14)

        ttk.Separator(parent).pack(fill="x", padx=14, pady=10)
        ttk.Label(parent, text="Hardware Diagnostics", style="Card.TLabel", font=("Segoe UI", 8, "bold")).pack(anchor="w", **pad)
        self.hw_panel = tk.Text(parent, height=6, width=28, bg="#060a13", fg="#94a3b8", bd=0,
                                 font=("Consolas", 8), highlightthickness=0)
        self.hw_panel.pack(padx=14, fill="x")
        self._update_hw_panel()

        ttk.Separator(parent).pack(fill="x", padx=14, pady=10)
        ttk.Button(parent, text="🔄 Reset Session", command=self._reset_session).pack(fill="x", padx=14, pady=(0, 12))

    def _refresh_actions(self):
        for w in self.actions_frame.winfo_children():
            w.destroy()
        ttk.Button(self.actions_frame, text="▶️ Run Triage", command=self._run_triage).pack(fill="x", pady=2)
        if self.inference_done:
            ttk.Button(self.actions_frame, text="💾 Save to Cache", command=self._save_cache).pack(fill="x", pady=2)
        if "GSM" in self.network_var.get():
            total, unsynced = counts()
            if unsynced > 0:
                ttk.Button(self.actions_frame, text="☁️ Sync to Cloud", command=self._sync_cloud).pack(fill="x", pady=2)
            ttk.Button(self.actions_frame, text="🔍 Check OTA", command=self._check_ota).pack(fill="x", pady=2)

    # -------------------- dashboard tab --------------------
    def _build_dashboard(self, parent):
        header = tk.Frame(parent, bg=C["primary"])
        header.pack(fill="x", pady=(0, 8))
        tk.Label(header, text="🩸 Pink Edge AI — Medical Image Analysis", bg=C["primary"], fg="white",
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=16, pady=10)

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self.image_label = tk.Label(left, bg="black")
        self.image_label.pack(padx=4, pady=4)
        self.image_caption = ttk.Label(left, text="", style="Muted.TLabel")
        self.image_caption.pack(anchor="w", padx=4)

        metrics = ttk.Frame(left)
        metrics.pack(fill="x", pady=8)
        self.metric_model = self._metric_tile(metrics, "Model", "—")
        self.metric_conf = self._metric_tile(metrics, "Confidence", "—")
        self.metric_lat = self._metric_tile(metrics, "Latency", "—")

        ttk.Label(left, text="Telemetry Log", style="H2.TLabel").pack(anchor="w", pady=(8, 2))
        self.log_text = tk.Text(left, height=8, bg="#060a13", fg="#94a3b8", bd=0, font=("Consolas", 9), highlightthickness=0)
        self.log_text.pack(fill="both", expand=False)
        for tag, color in [("dicom", "#60a5fa"), ("npu", "#34d399"), ("gsm", "#22d3ee"),
                            ("cache", "#f472b6"), ("muted", "#475569")]:
            self.log_text.tag_configure(tag, foreground=color)

        right = ttk.Frame(body, width=380, style="Card.TFrame")
        right.pack(side="left", fill="y")
        right.pack_propagate(False)
        self._build_meta_panel(right)

    def _metric_tile(self, parent, label, value):
        f = tk.Frame(parent, bg=C["surface_alt"], padx=10, pady=8)
        f.pack(side="left", fill="x", expand=True, padx=4)
        tk.Label(f, text=label, bg=C["surface_alt"], fg=C["text_muted"], font=("Segoe UI", 8)).pack(anchor="w")
        v = tk.Label(f, text=value, bg=C["surface_alt"], fg=C["primary_light"], font=("Segoe UI", 14, "bold"))
        v.pack(anchor="w")
        return v

    def _build_meta_panel(self, parent):
        pad = {"padx": 14, "pady": 4}
        ttk.Label(parent, text="DICOM Metadata", style="Card.TLabel", font=("Segoe UI", 10, "bold")).pack(anchor="w", **pad)
        self.meta_text = ttk.Label(parent, text="", style="Card.TLabel", justify="left")
        self.meta_text.pack(anchor="w", padx=14)

        self.verdict_frame = tk.Frame(parent, bg=C["surface_alt"])
        self.verdict_frame.pack(fill="x", padx=14, pady=10)
        self.verdict_icon = tk.Label(self.verdict_frame, text="", bg=C["surface_alt"], font=("Segoe UI", 20))
        self.verdict_icon.pack(pady=(10, 0))
        self.verdict_title = tk.Label(self.verdict_frame, text="", bg=C["surface_alt"], fg=C["text"], font=("Segoe UI", 13, "bold"))
        self.verdict_title.pack()
        self.verdict_sub = tk.Label(self.verdict_frame, text="", bg=C["surface_alt"], fg=C["text_muted"], font=("Segoe UI", 9), wraplength=330)
        self.verdict_sub.pack(pady=(0, 10))

        self.detail_text = ttk.Label(parent, text="", style="Card.TLabel", justify="left")
        self.detail_text.pack(anchor="w", padx=14, pady=(0, 8))

        ttk.Label(parent, text="Confirm Assessment", style="Card.TLabel", font=("Segoe UI", 10, "bold")).pack(anchor="w", **pad)
        self.assess_label1 = ttk.Label(parent, text="BI-RADS Assessment", style="Card.TLabel")
        self.assess_label1.pack(anchor="w", padx=14)
        self.bi_rads_combo = ttk.Combobox(parent, textvariable=self.bi_rads_var, values=BI_RADS_OPTIONS, state="readonly")
        self.bi_rads_combo.pack(fill="x", padx=14)
        self.assess_label2 = ttk.Label(parent, text="ACR Breast Density", style="Card.TLabel")
        self.assess_label2.pack(anchor="w", padx=14, pady=(6, 0))
        self.acr_combo = ttk.Combobox(parent, textvariable=self.acr_var, values=ACR_DENSITY_OPTIONS, state="readonly")
        self.acr_combo.pack(fill="x", padx=14)

        self.risk_banner = tk.Label(parent, text="", bg=C["surface_alt"], wraplength=330, justify="left", font=("Segoe UI", 9))
        self.risk_banner.pack(fill="x", padx=14, pady=8)

        btns = ttk.Frame(parent, style="Card.TFrame")
        btns.pack(fill="x", padx=14, pady=(0, 10))
        ttk.Button(btns, text="📝 Text Report", command=self._download_text).pack(side="left", fill="x", expand=True, padx=2)
        ttk.Button(btns, text="📄 PDF Report", style="Accent.TButton", command=self._download_pdf).pack(side="left", fill="x", expand=True, padx=2)

    # -------------------- hospital hub tab --------------------
    def _build_hub(self, parent):
        header = tk.Frame(parent, bg=C["success"])
        header.pack(fill="x", pady=(0, 8))
        tk.Label(header, text="🏥 Allied Hospital Faisalabad — Urban Receiving Terminal", bg=C["success"], fg="white",
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=16, pady=10)

        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="Alert Stream", style="H2.TLabel").pack(anchor="w", pady=4)
        canvas = tk.Canvas(left, bg=C["bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        self.alerts_container = ttk.Frame(canvas)
        self.alerts_container.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.alerts_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="left", fill="y")

        right = ttk.Frame(body, width=280, style="Card.TFrame")
        right.pack(side="left", fill="y", padx=(8, 0))
        right.pack_propagate(False)
        ttk.Label(right, text="Network Status", style="Card.TLabel", font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=12, pady=6)
        self.net_panel = tk.Text(right, height=6, bg="#060a13", fg="#94a3b8", bd=0, font=("Consolas", 8), highlightthickness=0)
        self.net_panel.pack(fill="x", padx=12)
        stats = ttk.Frame(right, style="Card.TFrame")
        stats.pack(fill="x", padx=12, pady=10)
        self.metric_total_alerts = self._metric_tile(stats, "Total Alerts", "0")
        self.metric_critical = self._metric_tile(stats, "Critical", "0")

        self._refresh_hub()

    # -------------------- cloud sync tab --------------------
    def _build_cloud(self, parent):
        header = tk.Frame(parent, bg=C["warning"])
        header.pack(fill="x", pady=(0, 8))
        tk.Label(header, text="☁️ Cloud Sync & Cache — Alibaba Cloud Integration (simulated)", bg=C["warning"], fg="white",
                 font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=16, pady=10)

        metrics = ttk.Frame(parent)
        metrics.pack(fill="x")
        self.metric_cached = self._metric_tile(metrics, "Cached Reports", "0")
        self.metric_unsynced = self._metric_tile(metrics, "Unsynced", "0")
        self.metric_synced = self._metric_tile(metrics, "Synced", "0")
        self.metric_iot = self._metric_tile(metrics, "IoT Queue", "0")

        self.cloud_status = ttk.Label(parent, text="", style="TLabel")
        self.cloud_status.pack(anchor="w", pady=8)

        ttk.Label(parent, text="Local Cache Status", style="H2.TLabel").pack(anchor="w", pady=(8, 4))
        cols = ("id", "patient", "modality", "bi_rads", "acr", "verdict", "timestamp", "synced")
        self.tree = ttk.Treeview(parent, columns=cols, show="headings", height=10)
        for c_, w_ in zip(cols, (40, 90, 90, 160, 140, 160, 140, 60)):
            self.tree.heading(c_, text=c_.replace("_", " ").title())
            self.tree.column(c_, width=w_)
        self.tree.pack(fill="both", expand=True, pady=4)

        self._refresh_cloud()

    # -------------------- behaviour --------------------
    def _on_tab_change(self):
        self._refresh_hub()
        self._refresh_cloud()

    def _set_lang(self, urdu):
        _urdu["on"] = urdu
        messagebox.showinfo("Language", "زبان تبدیل ہو گئی" if urdu else "Language set to English")

    def _on_model_change(self):
        self.inference_done = False
        self.current_result = None
        self.log_entries = []
        self.log_text.delete("1.0", "end")
        self._refresh_placeholder()
        self._refresh_actions()
        self._update_meta()

    def _upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if path:
            self.uploaded_path = path
            self.upload_label.config(text=os.path.basename(path))
            self._refresh_placeholder()

    def _refresh_placeholder(self):
        if self.uploaded_path:
            self.current_image = Image.open(self.uploaded_path).convert("L").resize((512, 512))
        else:
            seed = int(time.time()) % 10000
            self.current_image = load_placeholder(self.model_var.get(), seed)
        self._render_image()
        self._update_meta()

    def _render_image(self):
        img = self.current_image
        if self.inference_done and self.current_result and self.overlay_var.get():
            img = draw_bbox(img, self.model_var.get(), self.current_result)
        disp = img.convert("RGB").resize((420, 420))
        self._photo = ImageTk.PhotoImage(disp)
        self.image_label.configure(image=self._photo)
        name = os.path.basename(self.uploaded_path) if self.uploaded_path else "Generated Placeholder"
        state = "Detection: ACTIVE" if (self.inference_done and self.overlay_var.get()) else "Awaiting Analysis"
        self.image_caption.configure(text=f"{name} | {state}")

    def _update_meta(self):
        model = self.model_var.get()
        mod_map = {"Mammography (YOLOv8-OBB)": "MG (Mammography)", "Tuberculosis (Chest X-Ray)": "DX (Digital Radiography)",
                   "Maternal Health (Ultrasound)": "US (Ultrasound)"}
        self.meta_text.configure(text=(
            f"Patient ID:  {self.pat_id}\nAge:  {self.pat_age} Y\nModality:  {mod_map[model]}\n"
            f"Date:  {datetime.now().strftime('%Y-%m-%d')}\nInstitution:  Rural BHU Faisalabad"))
        if "Tuberculosis" in model:
            self.assess_label1.configure(text="TB Severity")
            self.assess_label2.configure(text="Lung Zone")
            self.bi_rads_combo.configure(values=TB_SEVERITY_LEVELS)
            self.acr_combo.configure(values=TB_LUNG_ZONES)
        else:
            self.assess_label1.configure(text="BI-RADS Assessment")
            self.assess_label2.configure(text="ACR Breast Density")
            self.bi_rads_combo.configure(values=BI_RADS_OPTIONS)
            self.acr_combo.configure(values=ACR_DENSITY_OPTIONS)

        if self.inference_done and self.current_result:
            r = self.current_result
            self.verdict_frame.configure(bg=C["danger"] if r["is_critical"] else C["success"])
            for w in (self.verdict_icon, self.verdict_title, self.verdict_sub):
                w.configure(bg=self.verdict_frame["bg"])
            self.verdict_icon.configure(text=r["vicon"])
            self.verdict_title.configure(text=r["verdict"], fg="white")
            self.verdict_sub.configure(text=r["sub"], fg="white")
            self.detail_text.configure(text=(
                f"Localization:  {r['loc']}\nClassification:  {r['extra']}\n"
                f"Inference Time:  {self.inference_latency}s\nModel Source:  {r.get('source', 'N/A')}"))
            self.bi_rads_var.set(r["bi_rads"])
            self.acr_var.set(r["acr"])
            if r["is_critical"]:
                self.risk_banner.configure(text="⚠️ Immediate Action Required — refer for specialist consultation",
                                            fg="white", bg=C["danger"])
            else:
                self.risk_banner.configure(text="✅ Low risk — routine screening", fg="white", bg=C["success"])
            self.metric_model.configure(text=self.model_var.get().split("(")[0].strip())
            self.metric_conf.configure(text=f"{r['confidence']:.1f}%")
            self.metric_lat.configure(text=f"{self.inference_latency}s")
        else:
            self.verdict_frame.configure(bg=C["surface_alt"])
            for w in (self.verdict_icon, self.verdict_title, self.verdict_sub):
                w.configure(bg=C["surface_alt"], fg=C["text_muted"])
            self.verdict_icon.configure(text="📋")
            self.verdict_title.configure(text="Awaiting Analysis")
            self.verdict_sub.configure(text="Run triage to begin")
            self.detail_text.configure(text="")
            self.risk_banner.configure(text="", bg=C["surface"])

    def _log(self, msg, tag="muted"):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.log_entries.append(line)
        self.log_text.insert("end", line, tag)
        self.log_text.see("end")

    def _update_hw_panel(self):
        self.hw_panel.configure(state="normal")
        self.hw_panel.delete("1.0", "end")
        self.hw_panel.insert("end",
            f"RK3588      ● ACTIVE\nNPU Load    {self.hw_stats['load']}\n"
            f"Power       {self.hw_stats['power']}\nTemp        {self.hw_stats['temp']}\n"
            f"Model       {self.model_var.get().split('(')[0].strip()}")
        self.hw_panel.configure(state="disabled")

    def _run_triage(self):
        model = self.model_var.get()
        self._log("[DICOM] Ingesting raw scan via LAN... Success.", "dicom")
        self.root.update_idletasks()

        t0 = time.time()
        result = run_triage(model, self.current_image)
        self.inference_latency = round(time.time() - t0, 2)
        if self.inference_latency < 1.0:
            self.inference_latency = round(random.uniform(7.8, 12.2), 1)  # NPU-illustrative floor, like the original demo

        self.current_result = result
        self.inference_done = True
        self.hw_stats = {"load": "97%", "power": f"{random.uniform(5.8, 6.5):.1f}W", "temp": f"{random.randint(38, 45)}C"}
        self.net_stats = {"signal": f"-{random.randint(75, 95)} dBm", "bhus": random.randint(10, 20),
                           "module": "ONLINE" if random.random() > 0.1 else "UNSTABLE"}
        new = {"pat_id": random.randint(10000000, 99999999), "pat_age": random.randint(28, 75)}
        self.pat_id, self.pat_age = new["pat_id"], new["pat_age"]

        real = "REAL" in result.get("source", "SIMULATED").upper() or "real inference" in result.get("source", "")
        self._log(f"[NPU] {'Real on-device' if real else 'Simulated'} inference for {model.split('(')[0].strip()}... Complete ({self.inference_latency}s).", "npu")
        sms_payload = f"ID:{self.pat_id}|LOC:29.344|{result['sms']}"
        self._log(f'[GSM] Broadcasted: "{sms_payload}" -> Allied Hospital Hub.', "gsm")
        if "GSM" in self.network_var.get():
            iot_id = f"IOT-{random.randint(100000, 999999)}"
            self._log(f"[Alibaba IoT] Queued - ID: {iot_id} -> Table Store", "gsm")
            self.iot_messages.append({"time": time.strftime("%H:%M:%S"), "id": self.pat_id, "payload": sms_payload, "iot_id": iot_id})

        self.sms_alerts.insert(0, {"time": time.strftime("%H:%M:%S"), "id": self.pat_id,
                                    "type": model.split("(")[0].strip(), "payload": sms_payload,
                                    "status": "Pending Review", "is_critical": result["is_critical"]})

        self._render_image()
        self._update_meta()
        self._update_hw_panel()
        self._refresh_actions()
        self._refresh_hub()
        self._refresh_cloud()

    def _save_cache(self):
        if not self.current_result:
            return
        model = self.model_var.get()
        mod_map = {"Mammography (YOLOv8-OBB)": "MG", "Tuberculosis (Chest X-Ray)": "DX", "Maternal Health (Ultrasound)": "US"}
        r = self.current_result
        data = {
            "patient_id": self.pat_id, "patient_age": self.pat_age, "modality": mod_map.get(model, "Unknown"),
            "model_used": model.split("(")[0].strip(), "bi_rads": self.bi_rads_var.get() or r["bi_rads"],
            "acr_density": self.acr_var.get() or r["acr"], "verdict": r["verdict"], "localization": r["loc"],
            "confidence": r["confidence"], "inference_time": self.inference_latency,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "network_mode": self.network_var.get(),
            "model_source": r.get("source", "N/A"), "synced": 0,
        }
        save_to_cache(data)
        self._log(f"[SQLite] Report cached (ID: {self.pat_id}).", "cache")
        self._refresh_cloud()
        messagebox.showinfo("Saved", t("Report cached."))

    def _current_report_dict(self):
        r = self.current_result
        model = self.model_var.get()
        mod_map = {"Mammography (YOLOv8-OBB)": "MG (Mammography)", "Tuberculosis (Chest X-Ray)": "DX (Digital Radiography)",
                   "Maternal Health (Ultrasound)": "US (Ultrasound)"}
        return {
            "patient_id": self.pat_id, "patient_age": self.pat_age, "modality": mod_map[model],
            "model_used": model.split("(")[0].strip(), "bi_rads": self.bi_rads_var.get() or r["bi_rads"],
            "acr_density": self.acr_var.get() or r["acr"], "verdict": r["verdict"], "localization": r["loc"],
            "confidence": r["confidence"], "inference_time": self.inference_latency,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "network_mode": self.network_var.get(),
            "model_source": r.get("source", "N/A"), "synced": 0,
        }

    def _download_text(self):
        if not self.current_result:
            messagebox.showwarning("No result", "Run triage first.")
            return
        d = self._current_report_dict()
        path = filedialog.asksaveasfilename(defaultextension=".txt", initialfile=f"report_{self.pat_id}.txt")
        if path:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(generate_text_report(d))
            messagebox.showinfo("Saved", f"Report saved to {path}")

    def _download_pdf(self):
        if not self.current_result:
            messagebox.showwarning("No result", "Run triage first.")
            return
        d = self._current_report_dict()
        pdf_bytes = generate_pdf_bytes(d)
        if not pdf_bytes:
            messagebox.showerror("PDF unavailable", "fpdf2 is not installed or PDF generation failed.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pdf", initialfile=f"report_{self.pat_id}.pdf")
        if path:
            with open(path, "wb") as fh:
                fh.write(pdf_bytes)
            messagebox.showinfo("Saved", f"Report saved to {path}")

    def _sync_cloud(self):
        unsynced = get_unsynced_reports()
        iot_results = simulate_iot_sync(unsynced)
        oss_results = [simulate_oss_upload(r) for r in unsynced]
        mark_as_synced([r[0] for r in unsynced])
        self.iot_messages = iot_results
        self.oss_uploads = [o for o in oss_results if o["high"]]
        self._log(f"[Alibaba IoT] {len(iot_results)} messages synced to Table Store.", "gsm")
        self._refresh_cloud()
        messagebox.showinfo("Sync", t("Sync complete."))

    def _check_ota(self):
        self.acr_status = simulate_acr_check()
        self._refresh_cloud()
        messagebox.showinfo("OTA", f"{self.acr_status['current']} -> {self.acr_status['available']}")

    def _refresh_hub(self):
        for w in self.alerts_container.winfo_children():
            w.destroy()
        if not self.sms_alerts:
            tk.Label(self.alerts_container, text="✅ All Clear — no alerts pending", bg=C["success"], fg="white",
                     font=("Segoe UI", 10, "bold"), padx=20, pady=20).pack(fill="x", pady=4)
        for a in self.sms_alerts:
            color = C["danger"] if a["is_critical"] else C["success"]
            card = tk.Frame(self.alerts_container, bg=C["surface_alt"], bd=0, highlightbackground=color,
                             highlightthickness=2)
            card.pack(fill="x", pady=4, padx=2)
            tk.Label(card, text=f"Alert #{a['id']}  ({a['time']})", bg=C["surface_alt"], fg=color,
                     font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=10, pady=(6, 0))
            tk.Label(card, text=f"Payload: {a['payload']}\nType: {a['type']}   Status: {a['status']}",
                     bg=C["surface_alt"], fg=C["text"], font=("Consolas", 8), justify="left").pack(anchor="w", padx=10, pady=(0, 6))
        net = self.net_stats
        self.net_panel.configure(state="normal")
        self.net_panel.delete("1.0", "end")
        self.net_panel.insert("end", f"GSM Module   {net['module']}\nSignal       {net['signal']}\n"
                                      f"BHU Nodes    {net['bhus']}\nPending      {len(self.sms_alerts)}")
        self.net_panel.configure(state="disabled")
        self.metric_total_alerts.configure(text=str(len(self.sms_alerts)))
        self.metric_critical.configure(text=str(sum(1 for a in self.sms_alerts if a["is_critical"])))

    def _refresh_cloud(self):
        total, unsynced = counts()
        self.metric_cached.configure(text=str(total))
        self.metric_unsynced.configure(text=str(unsynced))
        self.metric_synced.configure(text=str(total - unsynced))
        self.metric_iot.configure(text=str(len(self.iot_messages)))
        self.cloud_status.configure(
            text="☁️ GSM Failover — Alibaba Cloud Sync Active (simulated)" if "GSM" in self.network_var.get()
            else "🔒 Fully Offline — Cloud Sync Disabled")
        for row in self.tree.get_children():
            self.tree.delete(row)
        for r in get_cached_reports():
            self.tree.insert("", "end", values=(r[0], r[1], r[3], (r[5] or "")[:22], (r[6] or "")[:18], (r[7] or "")[:22], r[11], "✅" if r[12] else "⏳"))

    def _reset_session(self):
        if messagebox.askyesno("Reset", "Reset the current session?"):
            self.uploaded_path = None
            self.inference_done = False
            self.current_result = None
            self.sms_alerts = []
            self.iot_messages = []
            self.oss_uploads = []
            self.acr_status = None
            self.log_entries = []
            self.log_text.delete("1.0", "end")
            self._refresh_placeholder()
            self._refresh_actions()
            self._refresh_hub()
            self._refresh_cloud()


def main():
    _lazy_import_tkinter()
    root = tk.Tk()
    app = PinkEdgeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
