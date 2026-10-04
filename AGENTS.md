# Pink Edge AI — AI Agent Context & Engineering Guidelines

This document provides definitive guidance for AI coding agents (Antigravity, Cursor, Claude, Copilot, etc.) working on the **Pink Edge AI** repository. Always consult this document before proposing or applying code changes, debugging, or introducing new features.

---

## 1. Project Overview & Mission

**Pink Edge AI** is an offline-first clinical triage and intelligence platform originally developed for the **Alibaba Cloud AI Hackathon 2026**.

- **Target Deployment**: Rural Basic Health Units (BHUs) in underserved regions (specifically modeled after Pakistan's rural healthcare network) operating on low-power single-board edge computers (e.g., Rockchip RK3588 NPU).
- **Core Clinical Modalities**:
  1. **Mammography** (Breast Cancer screening / BI-RADS classification).
  2. **Tuberculosis** (Chest X-Ray / WHO severity scoring).
  3. **Maternal Health** (Obstetric Ultrasound / fetal anomaly screening).
- **Operating Philosophy**:
  - **100% Offline-First**: Zero reliance on persistent internet connectivity. All triage, localization, reporting, and caching run locally on device.
  - **Opportunistic Cloud Sync**: When 2G GSM or intermittent network is available ("GSM Failover" mode), lightweight telemetry and high-risk case summaries sync to Alibaba Cloud (IoT Hub, OSS, ACR).
  - **Strict Patient Privacy**: Enforces deterministic HMAC/SHA-256 hexadecimal privacy hashing and PII stripping to comply with HIPAA/GDPR standards.
  - **Role-Based Access Control (RBAC)**: Enforces clinical separation of duty between Lady Health Workers (LHWs) and Senior Radiologists.

---

## 2. Repository Structure & Layout

The project maintains a dual-structure layout: a self-contained runtime directory (`App/`) and a mirrored root directory (for Streamlit Community Cloud and root launchers).

```
Pink_Edge_AI/
├── AGENTS.md                  # <-- This file: AI agent instructions & system guide
├── README.md                  # Comprehensive user & developer project guide
├── requirements.txt           # Unified Python dependencies (CPU PyTorch, OpenCV headless, etc.)
├── Start.bat                  # Desktop GUI launcher (installs deps, runs App/GUI.py)
├── Start_Web.bat              # Streamlit Web GUI launcher (runs App/streamlit_app.py)
├── Start_Web_GUI.bat          # Plain HTML/JS launcher (runs App/web_api.py on port 5000)
│
├── App/                       # Primary self-contained application root
│   ├── GUI.py                 # Tkinter Desktop App + shared constants, DB, and triage dispatcher
│   ├── streamlit_app.py       # Streamlit Web App (bilingual, 3 tabs, interactive triage)
│   ├── inference.py           # Model loading, Roboflow integrations, and triage execution
│   ├── offline_cv.py          # Classical pixel-diff heuristic (zero weights, zero network)
│   ├── auth_manager.py        # RBAC and PIN authentication (LHW vs Radiologist)
│   ├── dicom_anonymizer.py    # PII stripper & deterministic SHA-256 privacy hash generator
│   ├── train_local_model.py   # MobileNetV3-Small transfer learning training script
│   ├── train_tb_classifier.py # TB classifier training script (YOLO/ultralytics)
│   ├── Models/                # Weights, COCO datasets, and cached model checkpoints
│   ├── Tests/                 # Automated unit and integration test suites
│   ├── Documentations/        # In-depth architectural, model, backend, and API docs
│   └── Assets/ / Hardware/    # Static assets, branding, and RK3588 edge SBC specs
│
├── Web GUI/                   # Vanilla HTML/CSS/JS frontend (no build step, no Streamlit)
│   ├── index.html             # UI layout for lightweight edge browsers
│   ├── style.css              # Styled using color tokens from GUI.py's C dictionary
│   └── app.js                 # Vanilla JS communicating via REST to App/web_api.py
│
├── RK3588 SBC/                # Rockchip RK3588 edge node deployment & optimization files
└── Apk/                       # Optional Android companion application build files
```

> **Important Synchronization Note:**
> Key modules (`GUI.py`, `streamlit_app.py`, `inference.py`, `auth_manager.py`, `dicom_anonymizer.py`, `offline_cv.py`) are present at both the repository root and inside `App/`. Batch launchers execute within `App/`. When modifying core application logic, ensure both copies remain synchronized or verify which location is being tested.

---

## 3. The Three Frontend Editions

All three editions share identical business logic, database persistence, and model dispatchers:

| Edition | Entrypoint | Technologies | Purpose / Notes |
|---|---|---|---|
| **Desktop GUI** | `App/GUI.py` | Tkinter + PIL | Primary offline clinical workstation for rural clinics. Note: Tkinter is lazily imported so `GUI.py` can be safely imported on headless environments. |
| **Streamlit Web** | `App/streamlit_app.py` | Streamlit + Markdown/CSS | Responsive web UI with 3 navigable tabs (Dashboard, Hospital Hub, Cloud Sync), voice templates, and hardware diagnostics. |
| **Vanilla HTML/JS** | `Web GUI/index.html` | Plain HTML5, CSS3, Vanilla JS + `App/web_api.py` (Flask) | Lightweight client for low-power edge browsers with zero framework overhead. Communicates via REST with backend. |

---

## 4. Multi-Tiered Inference Hierarchy & Accuracy Grounding

Inference order is strictly governed by **empirically measured accuracy on held-out test data**, not an assumption of "online first":

```
[Incoming Scan (DICOM / JPEG / PNG)]
           │
           ▼
[Privacy Hashing & Anonymization]
           │
           ▼
   Select Modality
 ┌─────────┴───────────────────────┬────────────────────────┐
 ▼                                 ▼                        ▼
Mammography                   Tuberculosis             Maternal Health
1. Roboflow (90.9%)           1. Local Model (82.5%)   1. Roboflow (88.3%)
2. Offline CV (~96%)          2. Offline CV (74.0%)    2. Offline HF CNN
3. Local Model (98.7%)        3. Roboflow              3. Simulated Fallback
4. Local YOLO Weights         4. Offline HF ViT
5. Simulated Fallback         5. Simulated Fallback
```

### Detailed Tier Breakdown:

1. **Locally-Trained Classifiers (`train_local_model.py` / `Models/*/local_model.pt`)**:
   - Architecture: MobileNetV3-Small with frozen ImageNet backbone + trained linear head.
   - Runs fast on CPU-only hardware; completely offline once trained.
   - Accuracy: **98.7%** on Mammography; **82.5%** on TB (best among all TB options).
2. **Offline Pixel-Difference Heuristic (`offline_cv.py`)**:
   - Zero model weights, zero internet, zero external ML libraries required.
   - Computes canonical 256x256 histogram-equalized template averages from local labeled datasets (`Models/<Modality>/Data Set/*.coco`).
   - Evaluates pixel-wise difference against positive vs. negative templates. Identifies the largest connected region above threshold as the bounding box.
   - Measured Accuracy: **~96%** on Mammography; **74%** on TB.
3. **Roboflow Hosted Models (`inference.py`)**:
   - Called via `inference_sdk.InferenceHTTPClient` when an API key is available (via `roboflow_key.txt`, `ROBOFLOW_API_KEY`, or `st.secrets`).
   - Workspace: `imaad-ullah-khan-yameen`.
   - Modalities: `breastcancer-yolov8-78tni` (Mammography), `tuberculosis-tp2pv/1` (TB), `hash-maternal-health/1` (Maternal).
   - If network fails or key is absent, raises `RoboflowError` and seamlessly drops to offline tiers.
4. **Hugging Face / Local Weights**:
   - Offline cached weights downloaded on first setup.
5. **Deterministic Simulation Fallback**:
   - Curated clinical scenarios clearly labeled as `SIMULATED` in the UI to ensure the application never crashes during clinical demonstration when no weights or internet exist.

---

## 5. Security, Privacy & RBAC Architecture

### 5.1 DICOM Anonymization & Privacy Hashing (`dicom_anonymizer.py`)
- **PII Stripping**: All identifiable personal metrics (`PatientName`, `PatientID`, `PatientBirthDate`, `PatientAddress`, `PatientTelephoneNumbers`, `ReferringPhysicianName`, `InstitutionName`) are stripped or replaced with hashes.
- **Hexadecimal Privacy Hash**: Generates a 64-character uppercase SHA-256 hexadecimal hash using canonical JSON formatting (`sort_keys=True`, no extra whitespace).
- **Rule**: Never expose raw patient names, national IDs (CNIC), or precise GPS coordinates in UI tables, logs, GSM alerts, or cloud payloads. Always use `generate_hex_privacy_hash()`.

### 5.2 Role-Based Access Control (`auth_manager.py`)
- **Lady Health Worker (LHW)**:
  - Role ID: `lhw` | Default PIN: `1111`
  - Permissions: `view_results`, `download_report`, `run_triage`.
  - Restriction: Cannot override AI assessments or edit clinical classifications.
- **Senior Radiologist**:
  - Role ID: `radiologist` | Elevating PIN: `9999`
  - Permissions: Full access including `override_assessment`, `view_audit_log`, `export_dicom`, `manage_users`.
  - Authority: Can override BI-RADS categories, ACR density, or TB severity ratings with mandatory clinical documentation.

---

## 6. Data Persistence, Reporting & Edge Simulation

### 6.1 Local SQLite Cache (`pink_edge_cache.db`)
- Managed by `GUI.py` (`init_db()`, `save_to_cache()`, `get_cached_reports()`, `mark_as_synced()`).
- Table: `cached_reports`
  - Columns: `id`, `timestamp`, `patient_id` (hashed), `modality`, `verdict`, `bi_rads`, `acr_density`, `tb_severity`, `confidence`, `source`, `synced`, `synced_at`.
- Both desktop and web editions connect to this identical database.

### 6.2 Clinical Report Generation
- **PDF Reports**: Generated via `fpdf2`. Features clinical headers, patient hash, modality details, bounding box localization, and radiologist sign-off.
- **Graceful Fallback**: If `fpdf2` is not installed or encounters an error, automatically falls back to plain `.txt` clinical report generation.

### 6.3 Hardware & Telemetry Simulation
- **Edge SBC Telemetry**: Simulates RK3588 NPU load (0-100%), power draw (~4.5W - 12W), and temperature (38°C - 65°C).
- **2G GSM Failover**: Simulates 140-character GSM SMS broadcast to urban hub ("Allied Hospital Faisalabad").
- **Alibaba Cloud Integration**: Simulates IoT Link SDK telemetry, Object Storage Service (OSS) compressed patch uploads for high-risk findings (BI-RADS 4/5 or TB S2/S3), and Container Registry (ACR) over-the-air (OTA) model checks.

---

## 7. Developer Workflows & Commands

### 7.1 Environment Setup
```powershell
# Python 3.10+ recommended
pip install -r requirements.txt
```
*Key packages*: `streamlit`, `torch` (CPU), `torchvision`, `ultralytics`, `opencv-python-headless`, `fpdf2`, `inference-sdk`, `huggingface_hub`.

### 7.2 Launching Applications
- **Desktop Tkinter GUI**:
  ```powershell
  python App\GUI.py
  # or execute Start.bat
  ```
- **Streamlit Web GUI**:
  ```powershell
  streamlit run App\streamlit_app.py
  # or execute Start_Web.bat
  ```
- **Vanilla HTML/JS Edition**:
  ```powershell
  python App\web_api.py
  # or execute Start_Web_GUI.bat
  ```

### 7.3 Running Tests
Run unit and regression test suites from the `App/` directory:
```powershell
cd App
python -m unittest discover -s Tests
# Or run specific test files:
python Tests\test_anonymizer.py
python Tests\test_auth.py
python Tests\test_dicom_adapter.py
```

---

## 8. Critical Implementation Rules & Gotchas for AI Agents

1. **DO NOT CONTROL GIT**:
   The user explicitly manages git independently. **Do not run `git commit`, `git checkout`, `git stash`, `git push`, `git reset`, or any mutating git commands**.

2. **NEVER BREAK OFFLINE-FIRST OPERATION**:
   Never introduce mandatory network requests or external API calls into the primary triage flow. Any cloud call (Roboflow, Hugging Face, Alibaba Cloud) must be wrapped in a `try...except` block with a seamless fallback to the local classifier, pixel-diff heuristic, or simulation.

3. **STREAMLIT HTML INDENTATION BUG (CRITICAL)**:
   Streamlit's Markdown parser uses CommonMark. **Any line indented with 4 or more spaces inside an `st.markdown(..., unsafe_allow_html=True)` block is rendered as raw code block text** instead of styled HTML.
   - Use the `html_block()` helper function in `streamlit_app.py`.
   - Ensure multiline strings passed to `st.markdown` are dedented.

4. **USE HEADLESS OPENCV ONLY**:
   Never install `opencv-python`. Always maintain `opencv-python-headless` in `requirements.txt`. Never call `cv2.imshow()`, `cv2.waitKey()`, or any GUI-dependent OpenCV functions, as they crash headless cloud environments (e.g. Streamlit Community Cloud).

5. **PRESERVE LAZY TKINTER IMPORT IN `GUI.py`**:
   `GUI.py` contains shared clinical constants, DB logic, and the triage dispatcher imported by `streamlit_app.py`. Tkinter and PIL's `ImageTk` **must remain lazily imported** inside `_lazy_import_tkinter()`. Importing Tkinter at module level will cause headless Linux servers to crash upon importing `GUI.py`.

6. **COLOR SYSTEM & WCAG CONTRAST**:
   The single source of truth for the application's color palette is `GUI.py`'s `C` dictionary (Brand Rose `#be185d`, Accent Sky `#0369a1`, Success `#15803d`, Warning `#b45309`, Danger `#b91c1c`). All text and icons must satisfy WCAG AA contrast (minimum 4.5:1 ratio against background).

7. **BILINGUAL LOCALIZATION (ENGLISH / URDU)**:
   The UI supports bilingual English/Urdu toggling. Maintain the translation mapping in `TR` and wrap display text in `t(key)` where appropriate.

8. **CONFIDENCE SCORE CALIBRATION**:
   Confidence metrics returned by offline heuristics or local classifiers must be explicitly calibrated (e.g. using `CONFIDENCE_SCALE` in `offline_cv.py`) to prevent overconfident synthetic scores on borderline cases.

---

## 9. Summary Checklist for Code Modifications

Before completing any task, verify:
- [ ] No git commands were executed.
- [ ] Triage works completely without internet access.
- [ ] Patient PII is scrubbed and hashed via `dicom_anonymizer.py`.
- [ ] Role permissions are checked via `auth_manager.py` before allowing clinical overrides.
- [ ] Headless environments can import `GUI.py` without Tkinter dependency errors.
- [ ] HTML cards in Streamlit are dedented using `html_block()`.
- [ ] Code changes in `App/` are mirrored at the root if touching shared deployment files.
- [ ] Unit tests in `App/Tests/` pass cleanly.
