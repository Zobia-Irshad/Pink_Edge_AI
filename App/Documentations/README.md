## Pink Edge AI
Offline Edge AI Radiology Triage on Rockchip RK3588 NPU.
Desktop (`GUI.py`) + responsive web (`streamlit_app.py`) editions over a shared model backend
(`inference.py`) — see the root `README.md` for the full project layout and how to run it.

## Overview
Pink Edge AI is a hybrid-edge clinical intelligence platform designed for rural healthcare centers
in Punjab, Pakistan. It runs AI-powered medical imaging triage 100% offline on low-cost edge
hardware (Rockchip RK3588 NPU) and uses Alibaba Cloud as an optional background sync layer when
internet connectivity becomes available.

**The platform currently supports three diagnostic modalities:**

1. Mammography (Breast Cancer Screening)
2. Tuberculosis (Chest X-Ray Analysis)
3. Maternal Health (Ultrasound)

The system is specifically built for Lady Health Visitors (LHVs) working at Basic Health Units (BHUs) in rural areas where no radiologist is available and internet connectivity is unreliable.

## Problem
Rural Punjab lacks breast cancer screening. No radiologists at village clinics. No internet for cloud AI. Late detection costs lives.

## Solution
Pink Edge AI runs 100% offline on cheap edge hardware, each modality trying the methods measured
most accurate against real ground truth first (see `MODEL_SOURCES.md`):
* Mammography (Breast Cancer Screening): real-time triage via a hosted YOLOv8-OBB workflow, with an
  offline pixel-diff heuristic fallback. When internet becomes available, it syncs critical data to
  Alibaba Cloud as a backup layer.
* Tuberculosis (Chest X-Ray Engine): processes local digital X-ray scans offline (a locally-trained
  classifier, 82.5% held-out accuracy) to flag active disease, routing critical findings to the
  Allied Hospital hub via 2G GSM.
* Maternal Health (Ultrasound Engine): classifies fetal ultrasound findings via a hosted abnormality
  detector, with an offline fetal-brain-plane CNN fallback — zero cloud reliance when offline.

## Architecture
```text
Rural BHU (Edge Node)
├── RK3588 NPU (AI Inference)
├── SQLite3 (Local Cache)
├── YOLOv8-OBB (INT8)
├── 2G GSM → Alibaba Cloud IoT
├── OSS (High-Risk Image Backup)
└── ACR (OTA Model Updates)
│
▼
Allied Hospital (Urban Hub)
└── 2G GSM Alert Receiver
```


---

## Features

1. Offline AI Inference — real model backends per modality, offline-first with cloud-hosted primaries where available.
2. Multi-Modal — Mammography, Tuberculosis, Maternal Health.
3. Clinical Output — BI-RADS 0-6, ACR Density A-D, TB severity/zone, confidence scores.
4. Local Cache — SQLite3 database for offline storage.
5. PDF and Text Reports — Downloadable without internet.
6. 2G GSM Alerts — 140-char telemetry to urban hospitals.
7. Alibaba Cloud — IoT Platform, OSS, ACR (GSM Failover mode).
8. Bilingual — English and Urdu support.
9. Three Views — Edge Node, Hospital Hub, Cloud Sync.

---

## Tech Stack

- AI Models: YOLOv8-OBB, a locally-trained MobileNetV3-Small classifier, offline pixel-diff heuristic, Hugging Face ViT/CNN fallbacks
- Hardware: Rockchip RK3588 NPU (see `../Hardware/`)
- Backend: Python 3, Streamlit / Tkinter
- Database: SQLite3
- Cloud: Alibaba Cloud (IoT, OSS, ACR) — simulated
- Comms: 2G GSM (SIM800L)

---

## Installation

Step 1: Clone the repository.

```bash
git clone <this-repo-url>
cd <repo>/App
```

Step 2: Install dependencies.
```bash
pip install -r requirements.txt
```

Step 3: Run the application.
```bash
streamlit run streamlit_app.py
```
(or `python GUI.py` for the desktop edition — see the root `README.md`, or just run `Start.bat` /
`Start_Web.bat` from the repo root.)

## Usage

1. Select AI Model in sidebar.
2. Upload patient scan or use placeholder.
3. Click Run Triage.
4. Review BI-RADS and ACR assessment.
5. Click Save to Cache for offline storage.
6. Download Text or PDF report.
7. Switch to GSM Failover mode for cloud sync.
