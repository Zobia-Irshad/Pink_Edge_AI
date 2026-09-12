# Changes from the original Streamlit demo

Base: `Misc/Pink_Edge_AI-main/Pink_Edge_AI-main/pink_edge.py` (v5.3, Streamlit).
This: `GUI.py` + `inference.py` (Tkinter, desktop, offline) + `Validation/validate.py`.

## Platform
- **Streamlit → Tkinter.** No browser, no local web server, no Gradio — a native desktop window.
  Sidebar became a persistent left control panel; the 3 tabs (Dashboard / Hospital Hub / Cloud Sync)
  became a `ttk.Notebook`.
- **Android was the original ask, deferred.** This machine had no Java/Android SDK/Gradle installed;
  desktop was the fast, low-risk path using the Python already available. Nothing here forecloses an
  Android build later.

## Models — the actual point of this pass
The original app's three `.pt` weight files were fake placeholders (`"Dummy Model"` text stubs); only
its TB code path was ever wired for real inference, with nothing real to load. This build:

- **Tuberculosis**: now runs a real model — [sukhmani1303/tuberculosis-vit-model](https://huggingface.co/sukhmani1303/tuberculosis-vit-model)
  (TorchScript Vision Transformer, loaded via `torch.jit.load` — no custom architecture code needed),
  with its exact published preprocessing (grayscale → CLAHE → Gaussian blur → resize 224×224 → back
  to RGB → per-image z-score, ported from the model repo's `handler.py`). Verified end-to-end with a
  live forward pass, including on 3 real sample chest X-rays.
  - First swapped in `Owos/tb-classifier` (Keras/TensorFlow InceptionV3), which worked but pulled in a
    full TensorFlow install for one model. Replaced with the ViT above once the user pointed to it —
    pure PyTorch, no TensorFlow dependency at all anymore.
- **Maternal Health**: now runs a real model — [shr3m/fetal-brain-plane-cnn](https://huggingface.co/shr3m/fetal-brain-plane-cnn)
  (fetal-brain standard-plane CNN, custom 4-block architecture rebuilt in `inference.py` to match the
  published checkpoint). Also verified end-to-end.
- **Mammography**: still simulated by default (same scenario picker as the original app) — no
  Roboflow API key was supplied, and the `b-davmu/breastcancer-yolov8` Universe project's page can't
  be scraped without one. `inference.py` picks up a key (`roboflow_key.txt` or `ROBOFLOW_API_KEY`)
  and attempts a real weight download into `Models/Mammography/` automatically if/when one is provided.
- **Considered and rejected**: `Astaxanthin/KEEP` (suggested as a "cancer model") is a histopathology
  foundation model, not a mammography one — wrong imaging modality (biopsy slides vs. radiographic
  X-ray), so it was left out rather than force-fit. See `Documentations/MODEL_SOURCES.md`.

Every result-dict a model (real or simulated) produces carries a `source` field so the UI and reports
show plainly which of the three cases produced it.

## Validation
Added `Validation/validate.py` — a 13-check validation suite covering module imports, placeholder
image synthesis, the detection-overlay drawing, the simulated scenario generators, a full SQLite
cache round-trip (throwaway DB, never the real `pink_edge_cache.db`), text/PDF report generation,
real-model inference for TB and Maternal Health on both synthetic images and the real sample images
in `Test Data/`, the mammography SIMULATED-fallback path, the `run_triage()` dispatcher, and that the
Tkinter UI itself builds and can run one full triage cycle with no visible window. One real issue was
found and fixed while writing it: the `patient_id` column (schema ported as-is from the original app)
has SQLite TEXT affinity, so an inserted `int` silently comes back as a `str` on read — not a bug in
the app's own behavior, just a trap for any test doing strict type equality on that column.

## File organization
Everything was reorganized out of a flat root into destined folders:
- `Models/TB/`, `Models/Maternal/`, `Models/Mammography/` — one folder per modality's downloaded
  weights (previously ad-hoc `tb_hf`/`maternal_hf` folder names; `Models/Memograhpy` typo fixed to
  `Mammography`); `inference.py` updated to match.
- `Documentations/` — `MODEL_SOURCES.md` plus the original project's own docs.
- `Assets/Changes/Changes.md` — this file (previously `Documentations/CHANGES.md`).
- `Validation/validate.py` — the validation suite (previously root `validate.py`).
- `Test Data/` — real sample images (Tuberculosis X-rays, a breast-cancer mammogram) used by the
  validation suite for real (not just synthetic) inference checks.
- `Misc/` — the original hackathon submission (left untouched, per instruction).
- Root kept to just the runnable app: `GUI.py`, `inference.py`, `requirements.txt`, `Start.bat`,
  `README.md`, `pink_edge_cache.db`.

## Functional parity kept
Same BI-RADS/ACR vocabulary, same TB severity/lung-zone vocabulary, same SQLite `cached_reports`
schema and filename (`pink_edge_cache.db`), same text/PDF report structure, same simulated
Alibaba Cloud IoT/OSS/ACR panel (no real cloud credentials in either version), same EN/UR toggle
(trimmed dictionary), same placeholder mammogram/X-ray/ultrasound image synthesis (re-implemented
with PIL/numpy instead of OpenCV+Streamlit's `st.cache_data`), same bounding-box/crosshair overlay
logic per modality.

## Known gaps vs. the original
- PDF/text report download is a native "Save As" file dialog instead of a browser download button.
- No `st.cache_data`-style caching of generated placeholder images (regenerated per view — cheap
  enough at 512×512 that it doesn't matter in practice).
- Hardware/network diagnostics stay illustrative (randomized), exactly as the original app's own
  admittedly-fake RK3588 stats were — not a regression, just carried forward.
