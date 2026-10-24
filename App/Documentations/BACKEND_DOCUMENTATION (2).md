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
