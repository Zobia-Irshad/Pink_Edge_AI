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

See `BACKEND_DOCUMENTATION.md`, `FRONTEND_DOCUMENTATION.md`, `MODEL_DOCUMENTATION.md`, and `API_DOCUMENTATION.md` for details on each layer.

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
- The local database (`pink_edge_cache.db`) is a plain SQLite file; patient IDs are deterministically hashed via `dicom_anonymizer.py`.
- PDF report generation depends on `fpdf2`; if unavailable, the app degrades gracefully to text-only reports.

## 8. Offline DICOM Anonymization & Hexadecimal Privacy Hashing (HIPAA/GDPR Compliance)

Pink Edge AI includes an offline privacy engine (`dicom_anonymizer.py`) that guarantees patient privacy compliance without internet connectivity:

- **PII Stripping Engine**: Automatically strips all identifiable personal metrics (Patient Name, CNIC / National ID, Exact Location Coordinates / Address, Date of Birth, Contact numbers) from DICOM headers and metadata.
- **Hexadecimal Privacy Hashing**: Generates an 8-character HMAC-SHA256 privacy hash (e.g. `HEX-8F3A1C9B`) from the patient PII and local salt key.
- **Over-The-Air Airwave Protection**: Transmitted 2G SMS payloads use the anonymized Hex Privacy Hash (`ID:HEX-8F3A1C9B|LOC:ANON|...`), rendering intercepted airwave packets completely anonymous.
- **Clinical Parameter Preservation**: Retains vital non-PII clinical parameters (`PatientAge`, `PatientSex`, `Modality`, `BodyPartExamined`, `PixelData`) required for Edge NPU inference and specialist triage.

## 9. Multi-Tenant Biometric Access Control & RBAC (LHW vs. Senior Radiologist Profiles)

The platform features a gatekeeper access control engine (`auth_manager.py`) providing data security and liability protection:

- **Lady Health Worker (LHW) Profile (PIN `1111`)**: Tailored for village health workers. Provides simplified patient intake and clear binary triage guidance (`✅ Normal` vs `⚠️ Referral Required`), while locking diagnostic overrides, raw NPU latency metrics, telemetry console, and cloud sync.
- **Senior Radiologist Profile (PIN `9999`)**: Unlocks full clinical intelligence suite, BI-RADS 0–5 & ACR density diagnostic override selectors, raw INT8 NPU performance metrics, telemetry console, Hospital Hub receiving terminal, and Alibaba Cloud Sync / OTA update controls.
- **Biometric / RFID Card Auth Simulation**: Supports 1-touch profile switching for fast workflow testing.


