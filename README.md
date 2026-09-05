## Pink Edge AI
Offline Edge AI Mammography Triage on Rockchip RK3588 NPU.

Alibaba Cloud AI Hackathon 2026 Submission.

## Problem
Rural Punjab lacks breast cancer screening. No radiologists at village clinics. No internet for cloud AI. Late detection costs lives.

## Solution 
Pink Edge AI runs 100% offline on cheap edge hardware. It performs real-time mammography triage using YOLOv8-OBB. When internet becomes available, it syncs critical data to Alibaba Cloud as a backup layer.

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
git clone https://github.com/yourusername/pink-edge-ai.git
cd pink-edge-ai