# Pink Edge AI — Technical Reference

Consolidated from 5 separate files (`API_DOCUMENTATION.md`, `BACKEND_DOCUMENTATION.md`,
`FRONTEND_DOCUMENTATION.md`, `MODEL_DOCUMENTATION.md`, `PROJECT_ARCHITECTURE.md`) — content
unchanged, just combined into one file. **Scope note:** these describe the original
Streamlit/notebook submission (`App/Archive/pink_edge.py` and
`App/Archive/Misc/Pink_Edge_AI-main/pink_edge.py`), not the current `GUI.py`/`streamlit_app.py`
rebuild — kept as historical/reference material on the original app's internals. For the current
app, see the root `README.md`, `MODEL_SOURCES.md` (which model really backs which modality today),
and `USER_GUIDE.md`.

---

## Project Architecture

# Pink Edge AI — Project Architecture

**Version:** 5.3
**Context:** Alibaba Cloud AI Hackathon 2026
**Tagline:** Offline Edge AI Triage — Clinical Intelligence Platform

## 1. What This Project Is

Pink Edge AI is a single-page Streamlit application that demonstrates an **edge-AI clinical triage workflow** for rural Basic Health Units (BHUs) in Pakistan. It simulates a low-cost NPU-powered device (Rockchip RK3588) that can screen medical images — mammography, chest X-ray (TB), and obstetric ultrasound — **entirely offline**, then opportunistically sync results to Alibaba Cloud over a 2G GSM link when connectivity is available.

The app is built to be demoed end-to-end without any real cloud account or physical hardware: cloud services, GSM telemetry, and (for two of the three models) the AI inference itself are simulated, while a real on-device model path exists for the TB classifier if a trained weights file is supplied.

## 2. Deployment Target (Conceptual)

```
┌─────────────────────────────────┐
│    Rural BHU (Edge Node)        │
│   ┌─────────────────────────┐   │
│   │ RK3588 NPU (AI Inference)│   │
│   │ SQLite3 (Local Cache)    │   │
│   │ YOLOv8-OBB (INT8)        │   │
│   └─────────────────────────┘   │
│           │ 2G GSM │             │
│           ▼        ▼            │
│   ┌─────────────────────────┐   │
│   │ Alibaba Cloud IoT Hub    │   │
│   │ + OSS + ACR (OTA)        │   │
│   └─────────────────────────┘   │
└─────────────────────────────────┘
```

- **Edge node** — a Rockchip RK3588 single-board device physically located at a rural BHU. Runs the Streamlit app, an on-device NPU model, and a local SQLite cache. Designed to function with zero connectivity.
- **Uplink** — a 2G GSM module (e.g. SIM800L) used for low-bandwidth "GSM Failover" mode, since rural sites may lack broadband.
- **Cloud tier** — Alibaba Cloud, used only when GSM Failover is enabled:
  - **IoT Platform (Link SDK)** — receives compact (~140-char) telemetry strings.
  - **Object Storage Service (OSS)** — receives compressed, anonymized image patches for high-risk cases only (BI-RADS 4/5).
  - **Container Registry (ACR)** — source of over-the-air (OTA) model updates.
- **Receiving terminal** — "Allied Hospital Faisalabad" (Hospital Hub view), an urban monitoring dashboard for alerts broadcast from rural edge nodes.

## 3. Application Structure

The app is a **single Jupyter notebook / Python script** (`Pink_Edge_AI.ipynb`) organized into logical cells that map to functional layers:

| Layer | Responsibility | Key Functions |
|---|---|---|
| Model Loader | Loads the real TB YOLO model if present | `load_tb_model()` |
| Config | App-wide constants (BI-RADS scale, ACR density scale, model list) | `DB_PATH`, `BI_RADS_OPTIONS`, `ACR_DENSITY_OPTIONS`, `MODELS` |
| i18n | English/Urdu translation dictionary | `TR`, `t()` |
| Clinical Data Generators | Produces varied, realistic mock results per modality | `generate_mammography_result()`, `generate_tb_result()`, `generate_fetal_result()` |
| TB Scoring | WHO-style severity index for chest X-ray (replaces BI-RADS for that modality) | `TB_SEVERITY_LEVELS`, `TB_LUNG_ZONES`, `_run_tb_inference()` |
| Simulation Fallback | Deterministic-random mock TB scenarios when no real model is loaded | `generate_tb_simulation_result()` |
| Database | Local SQLite cache of triage reports | `init_db()`, `save_to_cache()`, `get_cached_reports()`, `mark_as_synced()` |
| Cloud Simulation | Mocks Alibaba IoT / OSS / ACR calls | `simulate_iot_sync()`, `simulate_oss_upload()`, `simulate_acr_check()` |
| Report Generation | Builds downloadable text and PDF clinical reports | `generate_text_report()`, `generate_pdf_report()` |
| Image Generation | Procedurally generates placeholder mammogram/X-ray/ultrasound images and bounding-box overlays | `generate_mammogram()`, `load_image()`, `draw_bbox()` |
| Styling | Global dark-theme CSS injected into Streamlit | `CSS` |
| Session State | Streamlit `st.session_state` defaults and lifecycle | `init_state()` |
| UI Views | The three navigable tabs | `render_sidebar()`, `render_dashboard()`, `render_hospital_hub()`, `render_cloud_sync()` |
| Entry Point | Wires everything together | `main()` |

See the Backend Documentation section, the Frontend Documentation section, the Model Documentation section, and the API Documentation section for details on each layer.

## 4. Navigation

The app renders three tabs inside a single page (`st.tabs`):

1. **🩺 Dashboard** — the edge node's own console: select a model, upload/generate a scan, run triage, review the AI verdict, confirm/override the clinical assessment, save to local cache, and download a report.
2. **🏥 Hospital Hub** — the urban receiving terminal: shows the live GSM alert stream, network/signal stats, and total/critical case counts.
3. **☁️ Cloud Sync** — the hybrid-edge control panel: cache status (synced/unsynced counts), Alibaba IoT/OSS/ACR panel, and a table of all cached reports.

## 5. Two Operating Modes

| Mode | Behavior |
|---|---|
| **Fully Offline** | Default mode. No cloud calls are made. Reports are only ever saved to the local SQLite cache. |
| **GSM Failover** | Unlocks "Sync to Cloud" and "Check OTA" actions. Triggers simulated Alibaba IoT telemetry, OSS uploads for high-risk cases, and ACR version checks. |

## 6. Data Flow (Single Triage Cycle)

1. User selects a model and (optionally) uploads an image in the sidebar.
2. **Run Triage** → generates a new mock patient, hardware stats, and network stats, then produces a clinical result (real inference for TB if a model is loaded, otherwise a scenario picked from a curated mock list).
3. A GSM-style alert payload is composed and appended to the in-memory alert stream (and, in GSM mode, to the IoT queue).
4. The Dashboard renders the image (with optional bounding-box overlay), the verdict, and an editable clinical assessment (BI-RADS/ACR or TB severity/lung zone).
5. **Save to Cache** persists the confirmed report to SQLite (`cached_reports` table, `synced = 0`).
6. In GSM Failover mode, **Sync to Cloud** pushes all unsynced rows through the simulated IoT/OSS pipeline and flips `synced = 1`.
7. **Hospital Hub** and **Cloud Sync** views reflect the updated alert stream and cache state.

## 7. Notable Design Notes / Caveats

- This is a **hackathon demo / prototype**, not a production medical device. Two of the three models (Mammography, Maternal Health) are entirely simulated — there is no real image classifier behind them. Only the TB pathway has a real-model hook (`models/tb_classifier.pt`).
- Cloud integration (Alibaba IoT/OSS/ACR) is mocked in-process; no real network calls are made.
- The local database (`pink_edge_cache.db`) is a plain SQLite file with no encryption; patient IDs are randomly generated integers, not real identifiers.
- PDF report generation depends on `fpdf2`; if unavailable, the app degrades gracefully to text-only reports.

---

## API Documentation

# Pink Edge AI — API Documentation

> **Note:** Pink Edge AI does not currently expose an external HTTP/REST API. It is a single-process Streamlit application — the UI calls Python functions directly, in-process. This document catalogs those internal function "APIs" module-by-module, since they are the actual integration surface today. Section 9 sketches what a real REST API would look like if the backend logic were extracted into a standalone service.

## 1. Model API

### `load_tb_model() -> ultralytics.YOLO | None`
Loads and caches `models/tb_classifier.pt`. Returns `None` if the file is a placeholder/dummy or cannot be loaded. Thread-safe.

### `_run_tb_inference(image_np: np.ndarray) -> dict | None`
Runs YOLO inference on an image array.
**Returns:** `{"class_id": int, "confidence": float, "box": [x1,y1,x2,y2] | None}` or `None` if no model is loaded / inference fails.

### `generate_tb_result(image_np: np.ndarray | None = None) -> dict`
Public entry point for TB triage. Runs real inference if possible; caller is expected to fall back to `generate_tb_simulation_result()` when this can't produce a result.
**Returns:** result dict — see §4 "Result Schema" below.

## 2. Clinical Data Generation API

| Function | Signature | Returns |
|---|---|---|
| `generate_new_patient()` | `() -> dict` | `{"pat_id": int, "pat_age": int}` |
| `generate_mammography_result()` | `() -> dict` | Result dict (see §4) |
| `generate_tb_simulation_result()` | `() -> dict` | Result dict (see §4) |
| `generate_fetal_result()` | `() -> dict` | Result dict (see §4) |
| `generate_hw_stats()` | `() -> dict` | `{"load": str, "power": str, "temp": str}` |
| `generate_net_stats()` | `() -> dict` | `{"signal": str, "bhus": int, "module": str}` |

## 3. Database API

All functions connect to the SQLite file at `DB_PATH` ("pink_edge_cache.db") per call.

| Function | Signature | Notes |
|---|---|---|
| `init_db()` | `() -> None` | Idempotent `CREATE TABLE IF NOT EXISTS` |
| `save_to_cache(data: dict)` | `(dict) -> None` | Inserts one row into `cached_reports`; `synced` defaults to 0 |
| `get_cached_reports()` | `() -> list[tuple]` | All rows, `ORDER BY id DESC` |
| `get_unsynced_reports()` | `() -> list[tuple]` | Rows where `synced = 0` |
| `mark_as_synced(report_ids: list[int])` | `(list[int]) -> None` | Sets `synced = 1` for the given row IDs |
| `get_cache_count()` | `() -> int` | Total row count |
| `get_unsynced_count()` | `() -> int` | Count where `synced = 0` |

`data` dict expected by `save_to_cache`: `patient_id, patient_age, modality, model_used, bi_rads, acr_density, verdict, localization, confidence, inference_time, timestamp`.

## 4. Result Schema

Every `generate_*_result()` function (mammography, TB, fetal) returns a dict with this shape (some keys are modality-specific):

```json
{
  "bi_rads": "string (or 'N/A (TB Triage)' for TB)",
  "acr": "string (or 'N/A' for TB)",
  "verdict": "string",
  "sub": "string — sub-headline shown under the verdict",
  "css": "success | danger",
  "loc": "string — anatomical localization",
  "extra": "string — classification detail",
  "vicon": "✅ | ⚠️ — verdict icon",
  "confidence": "float 0-100",
  "sms": "string — compact telemetry payload fragment",
  "is_critical": "boolean",
  "tb_severity": "string, TB only",
  "real_inference": "boolean, TB only",
  "detection_box": "[x1,y1,x2,y2] | null, TB only"
}
```

## 5. Cloud Simulation API

| Function | Signature | Returns |
|---|---|---|
| `simulate_iot_sync(reports: list[tuple])` | `(list) -> list[dict]` | One `{"report_id", "patient_id", "payload", "iot_id"}` per report |
| `simulate_oss_upload(report: tuple)` | `(tuple) -> dict` | `{"status": "UPLOADED"\|"SKIPPED", "key", "kb", "high"}` |
| `simulate_acr_check()` | `() -> dict` | `{"current", "available", "size", "registry"}` |

None of these make real network calls — they are pure in-process mocks used to drive the Cloud Sync UI.

## 6. Reporting API

| Function | Signature | Returns |
|---|---|---|
| `generate_text_report(d: dict)` | `(dict) -> str` | Formatted plain-text clinical report |
| `generate_pdf_report(d: dict)` | `(dict) -> bytes \| None` | PDF bytes, or `None` if `fpdf2` isn't installed or generation fails |
| `safe_pdf_text(text)` | `(Any) -> str` | Latin-1-safe string for use inside the PDF |

`d` is the same "report_data" shape assembled in the Dashboard view: `patient_id, patient_age, modality, model_used, bi_rads, acr_density, verdict, localization, confidence, inference_time, timestamp, network_mode, synced`.

## 7. i18n API

| Function | Signature | Notes |
|---|---|---|
| `t(text: str)` | `(str) -> str` | Looks up `text` in the `TR` dict when `st.session_state.urdu_mode` is `True`; otherwise returns `text` unchanged. Unmapped strings pass through as English. |

## 8. UI Render API

These are the top-level functions `main()` calls; they read/write `st.session_state` directly rather than taking/returning conventional arguments.

| Function | Signature | Renders |
|---|---|---|
| `render_sidebar()` | `() -> (str, UploadedFile\|None)` | Sidebar; returns `(selected_model, uploaded_file)` |
| `render_dashboard(selected_model, uploaded_file)` | `(str, UploadedFile\|None) -> None` | Dashboard tab |
| `render_hospital_hub()` | `() -> None` | Hospital Hub tab |
| `render_cloud_sync()` | `() -> None` | Cloud Sync tab |
| `main()` | `() -> None` | Entry point; wires page config, DB/state init, and the three tabs |

## 9. If This Became a Real Service

The cleanest extraction path would separate the "backend" functions above (§1–§7, none of which touch Streamlit) into a FastAPI service, and have the Streamlit app (or any other client) call it over HTTP. A natural REST surface, mirroring the internal functions:

| Method & Path | Maps to |
|---|---|
| `POST /triage` (body: `{model, image}`) | `generate_tb_result` / `generate_mammography_result` / `generate_fetal_result` |
| `POST /reports` | `save_to_cache` |
| `GET /reports?synced=false` | `get_unsynced_reports` / `get_cached_reports` |
| `POST /reports/sync` | `mark_as_synced` + `simulate_iot_sync` + `simulate_oss_upload` |
| `GET /reports/{id}/report.pdf` | `generate_pdf_report` |
| `GET /reports/{id}/report.txt` | `generate_text_report` |
| `GET /system/ota` | `simulate_acr_check` |

This is a proposed refactor, not something implemented in the current codebase.

---

## Backend Documentation

# Pink Edge AI — Backend Documentation

The "backend" of Pink Edge AI is the Python logic embedded in the notebook/script that runs underneath the Streamlit UI: model loading, data generation, persistence, cloud simulation, and report generation. There is no separate server process — everything executes in-process inside the Streamlit run.

## 1. Configuration

```python
DB_PATH = "pink_edge_cache.db"

BI_RADS_OPTIONS = [ ... 9 BI-RADS categories, 0 through 6 with A/B/C sub-levels ... ]
ACR_DENSITY_OPTIONS = ["A - Almost entirely fatty", "B - Scattered fibroglandular density",
                        "C - Heterogeneously dense", "D - Extremely dense"]
MODELS = ["Mammography (YOLOv8-OBB)", "Tuberculosis (Chest X-Ray)", "Maternal Health (Ultrasound)"]
```

These constants back every dropdown and classification scheme in the UI, and are also the values persisted to the local database.

## 2. Model Loading

### `load_tb_model()`
Loads `models/tb_classifier.pt` through Ultralytics YOLO.

- Thread-safe: guarded by `_model_lock`, cached in `_model_cache` so the weights are loaded once per session.
- **Dummy-file detection**: reads the first 16 bytes of the weights file and checks for the literal string `"Dummy"`/`"dummy"`. If found (i.e. a placeholder text file was shipped instead of real weights), returns `None` without attempting to load — this is what causes the app to fall back to simulation.
- Any import or load exception (missing `ultralytics`/`torch`, corrupt file, etc.) is caught and also resolves to `None`.
- This is the **only** model-loading path in the app; Mammography and Maternal Health have no equivalent and are always simulated.

## 3. Internationalization

```python
TR = { "English string": "اردو ترجمہ", ... }   # ~60 key/value pairs

def t(text):
    return TR.get(text, text) if st.session_state.get("urdu_mode", False) else text
```

Every user-facing string in the UI is wrapped in `t(...)`. When `urdu_mode` is `True`, `t()` looks the string up in `TR`; unmapped strings pass through untranslated (silent fallback to English).

`safe_pdf_text(text)` sanitizes strings for the FPDF Helvetica font (Latin-1 only) by transliterating common Unicode punctuation (em/en dashes, curly quotes, bullets, arrows) and replacing any remaining non-Latin-1 characters.

## 4. Clinical Data Generators

Because there is no real model behind Mammography or Maternal Health, results are drawn from **hand-authored scenario lists** using `random.choice()` / `random.uniform()`:

- `generate_mammography_result()` — 7 scenarios spanning BI-RADS 1 through 5, with matched ACR density, verdict text, localization, and a confidence range appropriate to the severity (higher confidence for clear negatives and unambiguous BI-RADS 5 findings; lower for borderline categories).
- `generate_fetal_result()` — 2 "normal" scenarios with randomized gestational age.
- `generate_tb_simulation_result()` — 4 scenarios (2 negative, 2 positive) with WHO-style severity, used whenever no real TB model is available.
- `generate_new_patient()` — random 8-digit patient ID (10,000,000–99,999,999) and age (28–75).
- `generate_hw_stats()` — random NPU load %, power draw (W), temperature (°C) shown in the sidebar's "Hardware Diagnostics" panel.
- `generate_net_stats()` — random GSM signal strength (dBm), connected-BHU count, and module status (90% chance "ONLINE").

## 5. TB Inference Pipeline

`_run_tb_inference(image_np)`:
1. Calls `load_tb_model()`; if `None`, returns `None` immediately (caller falls back to simulation).
2. Runs `model.predict(source=image_np, imgsz=512, conf=0.25, device="cpu")`.
3. If no boxes are returned, treats the image as TB-negative with a random high confidence (94–98.5%).
4. Otherwise selects the highest-confidence detection box and returns `{class_id, confidence, box}` (`class_id` 0 = negative, 1 = positive).

`generate_tb_result(image_np=None)` wraps this:
- If real inference succeeds, maps confidence to a severity tier (≥90% → S3/Advanced, ≥75% → S2/Moderate, else → S1/Minimal) and builds the full result dict (verdict, sub-text, localization, severity, SMS payload, `is_critical`, `real_inference: True`, detection box).
- If real inference is unavailable, the caller (sidebar action handler) falls back to `generate_tb_simulation_result()`.

## 6. Local Database (SQLite)

Schema (`cached_reports` table, created by `init_db()`):

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK AUTOINCREMENT | |
| `patient_id` | TEXT | |
| `patient_age` | INTEGER | |
| `modality` | TEXT | `MG` / `DX` / `US` |
| `model_used` | TEXT | e.g. "Mammography" |
| `bi_rads` | TEXT | BI-RADS string, or TB severity string when modality is `DX` |
| `acr_density` | TEXT | ACR density string, or `Zone: <lung zone>` for TB |
| `verdict` | TEXT | |
| `localization` | TEXT | |
| `confidence` | REAL | |
| `inference_time` | REAL | seconds |
| `timestamp` | TEXT | `YYYY-MM-DD HH:MM:SS` |
| `synced` | INTEGER | 0 = not yet synced, 1 = synced |
| `report_json` | TEXT | full result dict, JSON-serialized |

Functions: `init_db()`, `save_to_cache(data)`, `get_cached_reports()` (all rows, newest first), `get_unsynced_reports()`, `mark_as_synced(report_ids)`, `get_cache_count()`, `get_unsynced_count()`. All open/close a fresh `sqlite3.connect(DB_PATH)` per call (no persistent connection pool).

## 7. Cloud Simulation (Alibaba Cloud)

No real Alibaba SDK calls are made anywhere in the app — these functions synthesize plausible responses:

- `simulate_iot_sync(reports)` — for each unsynced row, builds a compact telemetry payload (`ID:<patient>|BR:<bi_rads>|TS:<timestamp>`) and a fake `IOT-xxxxxx` message ID, with a small `time.sleep(0.03)` per message to simulate latency.
- `simulate_oss_upload(report)` — uploads (simulated) only if the BI-RADS string contains `"4"` or `"5"` (i.e. high-risk cases), returning a fake object key (`pink-edge/hr/<patient_id>.jpg`) and a random file size in KB; otherwise returns `SKIPPED`.
- `simulate_acr_check()` — returns a hard-coded "current vs. available" OTA version pair (`v2.1.0` → `v2.2.1`, 14.2 MB) from a fake registry path.

## 8. Report Generation

- `generate_text_report(d)` — builds a plain-text clinical report. Branches on modality: TB reports show WHO severity + lung zone with an urgent-referral note for positives; all other modalities show BI-RADS + ACR density with a referral note tiered by BI-RADS category (5/4C → urgent oncology referral, 4A/4B → referral recommended, 3 → 6-month follow-up, else → routine).
- `generate_pdf_report(d)` — builds an equivalent branded PDF via `fpdf2` (teal header banner, patient info block, AI analysis block, clinical assessment block, urgent-referral banner for TB positives). Returns `None` (with a Streamlit error) if PDF generation fails; the UI checks a `PDF_AVAILABLE` flag (set at import time based on whether `fpdf` is installed) before offering the PDF download button at all.

## 9. Session State

`init_state()` seeds `st.session_state` with ~20 default keys (patient info, hardware/network stats, `urdu_mode`, `network_mode`, cache/sync flags, alert/message queues, `current_result`, `inference_latency`) — idempotently, only setting keys that don't already exist, so state survives Streamlit re-runs within a session.

Selecting a different model in the sidebar resets `inference_done`, `log_entries`, `cache_saved`, and `current_result` for that model.

---

## Frontend Documentation

# Pink Edge AI — Frontend Documentation

The entire UI is built with **Streamlit**, styled with a large block of injected custom CSS (dark clinical theme), and rendered through a small set of `render_*()` functions. There is no separate JS/React frontend — Streamlit generates the DOM from Python calls each re-run.

## 1. Page Setup

```python
st.set_page_config(page_title="Pink Edge AI", page_icon="🩸", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)
```

Wide layout, sidebar expanded by default, custom CSS injected once at the top of `main()`.

## 2. Design System (CSS tokens)

Defined as CSS custom properties on `:root`:

| Token | Value | Purpose |
|---|---|---|
| `--bg` | `#0a0e1a` | Page background |
| `--surface` / `--surface-alt` / `--surface-hover` | `#111827` / `#1e293b` / `#334155` | Card/panel backgrounds |
| `--border` / `--border-light` | `#1e293b` / `#334155` | Dividers |
| `--text` / `--text-muted` / `--text-light` | `#f1f5f9` / `#94a3b8` / `#64748b` | Typography hierarchy |
| `--primary` / `--primary-light` | `#14b8a6` / `#2dd4bf` | Brand teal (buttons, active states, key metrics) |
| Status colors (used inline) | success `#10b981`, danger `#ef4444`, warning `#f59e0b` | Verdict boxes, alert severity |

**Fonts**: Inter (UI text), JetBrains Mono (telemetry/console log and data values), Noto Nastaliq Urdu (loaded for the Urdu translation mode).

## 3. Layout: Sidebar (`render_sidebar()`)

Persistent left rail, present on every tab:

1. **Brand header** — logo emoji + "Pink Edge AI / Clinical Intelligence Platform".
2. **Language switch** — two buttons, 🇬🇧 EN / 🇵🇰 اردو, toggling `st.session_state.urdu_mode`.
3. **Network Mode** — radio: `Fully Offline` / `GSM Failover`. Shows a colored badge reflecting the current mode.
4. **Select AI Model** — dropdown of the 3 `MODELS`.
5. **Upload Patient Scan** — `st.file_uploader` (jpg/jpeg/png).
6. **Actions**:
   - **Ingest DICOM** — cosmetic log entry only.
   - **Run Triage** — the core action; see the Backend Documentation section §5 for what it triggers. Shows a 2-second spinner ("Running INT8 on NPU…") to sell the on-device-inference illusion.
   - **Save to Cache** — appears only after a triage has run; persists the confirmed report.
   - **Sync to Cloud** — appears only in GSM Failover mode with unsynced reports pending.
   - **Check OTA** — appears only in GSM Failover mode; simulated ACR version check.
7. **Show Detection Overlay** — checkbox toggling the bounding-box overlay on the analyzed image.
8. **Hardware Diagnostics** — live-updating panel (RK3588 status, NPU load %, power, temperature, active model).
9. **Reset Session** — clears all `st.session_state` and reruns.

## 4. Tab 1 — Dashboard (`render_dashboard()`)

Two-column layout (`3:2` split):

**Left column — Medical Image Analysis**
- Procedurally generated or uploaded image, rendered inside a styled `.image-viewer` frame with a caption overlay.
- After a triage run, shows the image with an optional AI bounding-box overlay (`draw_bbox()`), plus three metric tiles: Model, Confidence, Latency.
- Before any triage, shows the plain image with an "Awaiting Analysis" info banner.

**Right column — DICOM Metadata + Verdict**
- A metadata grid (Patient ID, Age, Modality, Date, Body Part, Institution) styled to look like a DICOM header.
- After triage: a color-coded **verdict box** (green/red) with icon, title, and sub-text; a detail card (Localization, Classification, Inference Time, GSM Alert status); an editable **Confirm Assessment** section — BI-RADS + ACR dropdowns for Mammography/Maternal Health, or TB Severity + Lung Zone dropdowns for the TB pathway (the UI deliberately swaps the field set per modality, since BI-RADS doesn't apply to chest X-ray); a warning/success banner based on `is_critical`; and **Download Text Report** / **Download PDF Report** buttons.

**Bottom — Telemetry Log**
A monospace "console" (`.console-log`) that appends color-coded lines per pipeline stage (`[DICOM]`, `[NPU]`, `[GSM]`, `[Alibaba IoT]`, `[SQLite]`) as the user progresses through Ingest → Triage → Cache → Sync.

## 5. Tab 2 — Hospital Hub (`render_hospital_hub()`)

Represents the urban receiving terminal ("Allied Hospital Faisalabad"). Two-column layout (`2:1`):

- **Alert Stream** (main column) — a card per incoming SMS/GSM alert, color-coded critical (red) vs. routine (green), showing the raw payload string, alert type, and status, with an "Acknowledge" button. Shows an "All Clear" empty state when there are no alerts.
- **Network Status + Stats** (side column) — GSM module status, signal strength, connected-BHU node count, pending-alert count, and two metric tiles (Total Alerts, Critical Cases).

## 6. Tab 3 — Cloud Sync (`render_cloud_sync()`)

- Cloud-sync status badge (GSM active vs. offline/disabled).
- Four metric tiles: Cached Reports, Unsynced, Synced, IoT Queue.
- **Alibaba Cloud IoT Platform** card — endpoint, message count, target table.
- **Alibaba Cloud OSS** card — bucket, region, upload count (only high-risk BI-RADS 4/5 patches are "uploaded").
- **Alibaba Cloud ACR — OTA** card — current vs. available model version once a check has been run.
- **Hybrid-Edge Architecture** card — the ASCII architecture diagram reproduced inline.
- **Local Cache Status** table — every cached report (ID, Patient ID, Modality, BI-RADS, ACR, Verdict, Timestamp, Synced ✅/⏳).

## 7. Shared UI Components (CSS classes)

| Class | Used for |
|---|---|
| `.page-header` | Gradient banner heading each tab |
| `.badge` (`.badge-cloud` / `.badge-offline`) | Small pill indicating network mode |
| `.section-header` | Icon + title row above each content block |
| `.metric-tile` | Small KPI card (label + big value, color variants: `primary`/`success`/`danger`/`warning`/`accent`) |
| `.verdict-box` (`.success` / `.danger`) | Large AI verdict callout |
| `.dicom-grid` / `.dicom-item` | Metadata grid |
| `.card` / `.card-primary` / `.card-success` | Generic bordered content panels |
| `.cloud-card` | Alibaba service status cards |
| `.hw-panel` / `.hw-row` | Sidebar hardware/network diagnostics rows |
| `.console-log` | Monospace telemetry log |
| `.alert-card` (`.critical` / `.ok`) | Hospital Hub alert entries |
| `.data-table` | Cached reports table |
| `.app-footer` | Version/footer line at the bottom of the page |

## 8. Internationalization in the UI

Every visible label calls `t("...")`. Toggling the language buttons flips `st.session_state.urdu_mode` and triggers `st.rerun()`, so the whole page re-renders in the selected language on the next pass — there is no partial/localized re-render.

## 9. Image Handling

- `generate_mammogram(size=512, seed=42)` (and sibling generators for X-ray/ultrasound, referenced via `load_image()`) procedurally synthesize a plausible grayscale medical image using NumPy when no file is uploaded, cached via `@st.cache_data`.
- `draw_bbox()` overlays a detection rectangle (OpenCV) on the image when "Show Detection Overlay" is enabled and a triage result exists.

---

## Model Documentation

# Pink Edge AI — Model Documentation

Pink Edge AI exposes three selectable "AI models" in the sidebar. Only one of them currently has a real inference path; the other two are simulated for demo purposes. This document describes each, and what would be needed to make all three real.

## 1. Model Inventory

| Model (UI label) | Modality | Target Architecture | Current Status |
|---|---|---|---|
| Mammography (YOLOv8-OBB) | Digital mammography (MG) | YOLOv8 Oriented Bounding Box, INT8-quantized for RK3588 NPU | **Simulated** — scenario picker only |
| Tuberculosis (Chest X-Ray) | Digital radiography (DX) | YOLOv8 (Ultralytics) classifier/detector, `models/tb_classifier.pt` | **Real inference supported**, with automatic fallback to simulation |
| Maternal Health (Ultrasound) | Ultrasound (US) | Not yet specified | **Simulated** — scenario picker only |

## 2. Tuberculosis Model (the one real pathway)

### Loading
`load_tb_model()` loads `models/tb_classifier.pt` via `ultralytics.YOLO`. A guard checks the first 16 bytes of the weights file for the string `"Dummy"`; if present, the app treats the file as a placeholder and returns `None`, which routes all TB requests to the simulation path (`generate_tb_simulation_result()`). The same fallback occurs on any exception (missing `ultralytics`/`torch`, corrupted weights, unsupported format, etc.).

### Inference
`_run_tb_inference(image_np)` runs `model.predict(imgsz=512, conf=0.25, device="cpu")` on the uploaded/generated image array.
- **class_id 0** → TB Negative
- **class_id 1** → TB Positive
- If the model returns zero boxes, the image is treated as negative with a random confidence in 94–98.5% (i.e. an implicit "clear" class rather than a true negative-class score).
- When a box is returned, the single highest-confidence detection is used; its `xyxy` box is retained for the on-image overlay.

### Confidence → Severity Mapping (positive cases only)
A simple confidence heuristic stands in for a proper severity classifier:

| Confidence | Severity (WHO-style index) | Description |
|---|---|---|
| ≥ 90% | S3 — Advanced | Large cavity / miliary pattern, bilateral upper lobes |
| 75–90% | S2 — Moderate | Cavity < 2 cm, right upper lobe |
| < 75% | S1 — Minimal | Unilateral, no cavitation |

Negative cases are always reported as **S0 — No active disease**.

### Severity / Zone Vocabulary
```
TB_SEVERITY_LEVELS = ["S0 - No active disease", "S1 - Minimal (unilateral, no cavity)",
                       "S2 - Moderate (bilateral / cavity < 2 cm)", "S3 - Advanced (large cavity / miliary pattern)"]
TB_LUNG_ZONES = ["Upper Zone", "Middle Zone", "Lower Zone", "Bilateral"]
```
These replace the BI-RADS/ACR fields in the UI whenever the TB model is selected, since BI-RADS is a mammography-specific standard.

### Clinician Override
The Dashboard's "Confirm Assessment" section always lets the health worker override the AI-suggested severity and lung zone via dropdowns before the report is saved/cached — the AI output is a suggestion, not a final, uneditable diagnosis.

## 3. Mammography (Simulated)

No image classifier is invoked. `generate_mammography_result()` selects uniformly at random from 7 hand-written clinical scenarios spanning BI-RADS 1 (Negative) through BI-RADS 5 (Highly Suggestive of Malignancy), each with a plausible confidence range, localization (quadrant), and ACR density pairing. This is intended to showcase the **UI/reporting workflow** for a mammography triage device, not real image analysis.

BI-RADS scale used throughout:
```
0 Incomplete · 1 Negative · 2 Benign · 3 Probably Benign · 4A/4B/4C Suspicious (increasing) ·
5 Highly Suggestive of Malignancy · 6 Known Biopsy-Proven Malignancy
```
ACR breast-density scale: `A` (almost entirely fatty) → `D` (extremely dense).

## 4. Maternal Health / Fetal Ultrasound (Simulated)

`generate_fetal_result()` currently only implements 2 "normal" scenarios (no anomalies / normal cardiac activity) with a randomized gestational age (18–38 weeks). There is no positive/abnormal-finding scenario implemented yet, and no real model hook — this pathway is the least developed of the three.

## 5. Hardware Target

All three models are framed as running via **INT8-quantized YOLOv8** inference on a **Rockchip RK3588 NPU**, chosen for its availability in low-cost edge boards suitable for deployment at rural BHUs without reliable power or connectivity. The app's "Hardware Diagnostics" panel and inference-latency figures (7.8–12.2s, randomly generated) are illustrative of expected on-device performance, not measurements from real hardware.

## 6. Path to Production

To move from this demo to a real triage tool, the two simulated pathways would need:
1. A trained detection/classification model per modality (see `ML_Dataset_Planning_Document` for candidate datasets and training plan).
2. INT8 quantization and conversion to a Rockchip-compatible runtime (e.g. RKNN) for on-NPU deployment.
3. Replacing each `generate_*_result()` scenario picker with a call into the corresponding real model, mirroring the pattern already established for `generate_tb_result()`.
4. Clinical validation (sensitivity/specificity against radiologist ground truth) before any real deployment — the current confidence numbers are randomly generated and carry no statistical meaning.
