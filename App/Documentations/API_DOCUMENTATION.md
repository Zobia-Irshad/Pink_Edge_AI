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
