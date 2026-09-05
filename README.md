## Pink Edge AI
Offline Edge AI Mammography Triage on Rockchip RK3588 NPU.
Alibaba Cloud AI Hackathon 2026 Submission.

## Overview
Pink Edge AI is a hybrid-edge clinical intelligence platform designed for rural healthcare centers in Punjab, Pakistan. It runs AI-powered medical imaging triage 100% offline on low-cost edge hardware (Rockchip RK3588 NPU) and uses Alibaba Cloud as an optional background sync layer when internet connectivity becomes available.

**The platform currently supports three diagnostic models:**

1. Mammography (Breast Cancer Screening)
2. Tuberculosis (Chest X-Ray Analysis)
3. Maternal Health (Ultrasound Triage)

The system is specifically built for Lady Health Visitors (LHVs) working at Basic Health Units (BHUs) in rural areas where no radiologist is available and internet connectivity is unreliable.



## Problem
Rural Punjab lacks breast cancer screening. No radiologists at village clinics. No internet for cloud AI. Late detection costs lives.

## Solution 
Pink Edge AI runs 100% offline on cheap edge hardware.
* Mammography (Breast Cancer Screening): It performs real-time mammography triage using YOLOv8-OBB. When internet becomes available, it syncs critical data to Alibaba Cloud as a backup layer.
* Tuberculosis (Chest X-Ray Engine): Processes local digital X-ray scans offline to detect pulmonary opacities and cavitary lesions, routing critical findings to the Allied Hospital hub via 2G GSM.
* Maternal Health (Ultrasound Engine): Analyzes off-grid ultrasound video streams natively to identify standard anatomical planes and fetal growth parameters with zero cloud reliance.

## Architecture
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


---

## Features

1. Offline AI Inference — YOLOv8-OBB on RK3588 NPU with INT8 quantization.
2. Multi-Modal — Mammography, Tuberculosis, Maternal Health.
3. Clinical Output — BI-RADS 0-6, ACR Density A-D, confidence scores.
4. Local Cache — SQLite3 database for offline storage.
5. PDF and Text Reports — Downloadable without internet.
6. 2G GSM Alerts — 140-char telemetry to urban hospitals.
7. Alibaba Cloud — IoT Platform, OSS, ACR (GSM Failover mode).
8. Bilingual — English and Urdu support.
9. Three Views — Edge Node, Hospital Hub, Cloud Sync.

---

## Tech Stack

- AI Model: YOLOv8-OBB (INT8)
- Hardware: Rockchip RK3588 NPU
- Backend: Python 3, Streamlit
- Database: SQLite3
- Cloud: Alibaba Cloud (IoT, OSS, ACR)
- Comms: 2G GSM (SIM800L)

---

## Installation

Step 1: Clone the repository.

```bash
git clone https://github.com/Zobia-Irshad/Pink_Edge_AI.git
cd pink-edge-ai
```

Step 2: Install dependencies.
```bash
pip install streamlit numpy opencv-python Pillow fpdf2
```

Step 3: Run the application.
```bash
streamlit run pink_edge.py
```

## Usage

1. Select AI Model in sidebar.
2. Upload patient scan or use placeholder.
3. Click Run Triage.
4. Review BI-RADS and ACR assessment.
5. Click Save to Cache for offline storage.
6. Download Text or PDF report.
7. Switch to GSM Failover mode for cloud sync.

