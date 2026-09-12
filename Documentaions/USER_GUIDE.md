# Pink Edge AI — User Guide

This guide walks through using the Pink Edge AI demo app as a rural BHU health worker (Dashboard) and as urban hospital staff (Hospital Hub / Cloud Sync).

## Getting Started

1. Launch the app (`streamlit run` on the exported script, or run the notebook cells in order).
2. The app opens on the **Dashboard** tab with the sidebar expanded on the left.

## 1. Choose Your Language

At the top of the sidebar, tap **🇬🇧 EN** or **🇵🇰 اردو** to switch the entire interface between English and Urdu. You can switch at any time; your current screen and data are preserved.

## 2. Choose Your Network Mode

Under **Network Mode**, pick:
- **Fully Offline** — the default. Everything works locally; no cloud actions are available. Use this to reflect a BHU with no connectivity.
- **GSM Failover** — unlocks the **Sync to Cloud** and **Check OTA** buttons, and shows cloud activity on the Cloud Sync tab. Use this to demonstrate what happens once a 2G signal becomes available.

## 3. Run a Triage

1. Under **Select AI Model**, choose one of:
   - Mammography (YOLOv8-OBB)
   - Tuberculosis (Chest X-Ray)
   - Maternal Health (Ultrasound)
2. Optionally, under **Upload Patient Scan**, upload a JPG/PNG image. If you skip this, the app generates a representative placeholder scan automatically.
3. Click **▶️ Run Triage**. You'll see a short "Running INT8 on NPU…" spinner while a new patient record, hardware readings, and an AI result are generated.
4. The main panel updates with:
   - The scan image (with a detection box overlay if enabled)
   - Model / Confidence / Latency tiles
   - DICOM-style metadata (Patient ID, Age, Modality, Date, Body Part, Institution)
   - A color-coded **verdict box** (green = reassuring, red = needs attention)

## 4. Review and Confirm the Assessment

Under **Confirm Assessment**, review the AI-suggested classification and adjust it if your own clinical judgement differs:
- **Mammography / Maternal Health** — set the **BI-RADS Assessment** and **ACR Breast Density** dropdowns.
- **Tuberculosis** — set the **TB Severity (WHO Index)** and **Affected Lung Zone** dropdowns instead.

A banner below will confirm whether the case requires **immediate referral** or is **low risk / routine**.

## 5. Save and Download

- **💾 Save to Cache** — stores the confirmed report locally (works in both network modes; this is the offline-safe path).
- **📝 Download Text Report** / **📄 Download PDF Report** — generates a patient-ready report file. (PDF requires the optional `fpdf2` package; if it isn't installed, only the text report is offered.)

## 6. Toggle the Detection Overlay

Use the **Show Detection Overlay** checkbox in the sidebar to turn the AI bounding box on the image on or off.

## 7. Watch the Telemetry Log

At the bottom of the Dashboard, the **Telemetry Log** shows a running, color-coded console of every step taken so far — DICOM ingest, NPU inference, GSM broadcast, cloud sync, and cache writes — useful for narrating a live demo.

## 8. Monitor Incoming Alerts (Hospital Hub tab)

Switch to **🏥 Hospital Hub** to see the receiving side:
- **Alert Stream** — every triage run appears here as a card, color-coded critical (red) vs. routine (green), with its raw GSM payload string. Click **Acknowledge** to mark an alert as reviewed.
- **Network Status** — simulated GSM signal, connected BHU count, and pending-alert count.
- **Stats** — total alerts and critical-case counts.

## 9. Sync and Inspect the Cache (Cloud Sync tab)

Switch to **☁️ Cloud Sync**:
- View how many reports are cached, synced, and unsynced, and the current IoT queue size.
- If you're in **GSM Failover** mode, use the sidebar's **☁️ Sync to Cloud** button to push all unsynced reports through the simulated Alibaba IoT/OSS pipeline. Only high-risk (BI-RADS 4/5) cases upload an image patch to OSS.
- Use **🔍 Check OTA** in the sidebar to simulate checking Alibaba Container Registry for a newer on-device model version.
- Scroll down to the **Local Cache Status** table to see every report stored on the device, with its sync status.

## 10. Reset

Click **🔄 Reset Session** in the sidebar at any time to clear all in-memory state (patient info, logs, alerts, cache flags) and start over. Note: this does **not** delete the SQLite database file — previously cached reports remain in `pink_edge_cache.db`.

## Tips for Demoing

- Run a few triages in **Fully Offline** mode first to build up unsynced reports, then switch to **GSM Failover** and hit **Sync to Cloud** to show the hybrid-edge story end to end.
- Trigger at least one high-severity result (e.g. re-run Mammography a few times until a BI-RADS 4/5 scenario appears) to show the OSS high-risk-upload behavior and the Hospital Hub's critical-alert styling.
- The Tuberculosis pathway is the only one with a real-model hook — if you have a trained `models/tb_classifier.pt`, drop it in to show real on-device inference instead of the scenario-based simulation.

## Troubleshooting

| Symptom | Likely Cause |
|---|---|
| "pip install fpdf2 for PDF" instead of a PDF button | `fpdf2` is not installed in the environment |
| TB results always look the same few scenarios | No real model at `models/tb_classifier.pt` — app is using simulation fallback |
| Sync/OTA buttons missing | You're in **Fully Offline** mode — switch to **GSM Failover** |
| Cached reports table stays empty | You haven't clicked **Save to Cache** after a triage run |
