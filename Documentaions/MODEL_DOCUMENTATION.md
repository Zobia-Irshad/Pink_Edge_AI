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
