# Changes from the original Streamlit demo

Base: `Misc/Pink_Edge_AI-main/Pink_Edge_AI-main/pink_edge.py` (v5.3, Streamlit).
This: `GUI.py` (Tkinter desktop) + `streamlit_app.py` (responsive web) + `inference.py` (shared real
model backend) + `Validation/validate.py`.

## Platform
- **Streamlit → Tkinter, first.** No browser, no local web server, no Gradio — a native desktop
  window. Sidebar became a persistent left control panel; the 3 tabs (Dashboard / Hospital Hub /
  Cloud Sync) became a `ttk.Notebook`.
- **Streamlit added back, as a second sibling UI.** `streamlit_app.py` is a fresh, responsive
  (mobile/tablet-width-aware CSS) rebuild — not the original `pink_edge.py` — that imports `GUI.py`
  directly for every piece of shared logic (constants, imaging, simulated scenarios, DB, reports, the
  `run_triage()` dispatcher) instead of duplicating it, so it gets the real TB/Maternal models and the
  honest Mammography SIMULATED fallback for free.
- **Streamlit Community Cloud deploy fix.** First cloud deploy crashed with `ImportError: import
  _tkinter` — `GUI.py` originally imported `tkinter`/`PIL.ImageTk` unconditionally at module level,
  and Streamlit Cloud's Linux container has no system Tk libraries. Fixed by lazy-importing tkinter
  only inside `PinkEdgeApp.__init__()`/`main()` (via `_lazy_import_tkinter()`, binding as module
  globals) — `GUI.py` is now safely importable on a headless host, and the actual Tk import only
  happens if the desktop app is really launched. Also swapped `opencv-python` → `opencv-python-headless`
  in `requirements.txt` for the same reason (the GUI build needs `libGL.so.1`, which headless
  containers don't have; nothing in this codebase calls `cv2.imshow` or other GUI functions).
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

## Roboflow integration (`imaad-ullah-khan-yameen` workspace)
All three modalities now try the user's own trained Roboflow model/workflow first (real,
purpose-trained on this project's data), falling back to the offline Hugging Face models above
(TB, Maternal) or the simulated picker (Mammography) when Roboflow is unreachable. Grounded against
real API calls (not guessed field names) via `inference-sdk`'s `InferenceHTTPClient`:
- **Mammography** — Workflow `breastcancer-yolov8-78tni` (`client.run_workflow`).
- **Tuberculosis** — Model `tuberculosis-tp2pv/1` (`client.infer`); real class taxonomy grounded
  from the project's COCO export (Turkish labels, ASCII-folded in the API response — handled with
  an accent-stripping normalizer, not hardcoded spellings).
- **Maternal Health** — Model `hash-maternal-health/1` (`client.infer`); single-class detector
  (`"abnormal"`), grounded the same way.

**Key finding from validating against ground truth, not just "does it run"**: the TB model
(`tuberculosis-tp2pv/1`) correctly caught a real positive case but called all 3 tested
ground-truth-healthy samples TB Positive at 98-99% confidence — a real accuracy problem in that
specific trained model (see `Documentations/MODEL_SOURCES.md` for full detail and next steps), not
an integration bug. `Validation/validate.py`'s TB ground-truth check is left failing on purpose to
keep surfacing this.

Also added: a shared Roboflow client layer in `inference.py` (`RoboflowError`, retry-with-backoff,
`_roboflow_infer`/`_roboflow_run_workflow`), and 4 new validation checks — ground-truth
cross-checks for all three modalities against their real COCO-annotated datasets (which the user
added under `Models/*/Data Set/`), not just "did it return something."

## Offline pixel-diff heuristic (`offline_cv.py`) — no model, no internet, ever
Added per a direct request for a fully-offline method: image → grayscale → resize/orient (try
identity vs. horizontal-flip, keep whichever best correlates with a generic reference — handles
left/right laterality without full registration) → average local labeled images into "typical
positive"/"typical negative" reference templates → pixel-difference the uploaded image against
both → the region that's furthest from negative and closest to positive becomes a **real** bounding
box (not the old illustrative fixed position — `draw_bbox()` in `GUI.py` now draws it when
present) → the mean of that difference map becomes the confidence score.

**Measured, not assumed**, on a proper held-out test split (`python offline_cv.py`): **74% for TB,
96% for Mammography**. Maternal Health has no negative/healthy images anywhere in its local
dataset (every image is an annotated abnormal case), so it correctly returns `None` there rather
than guessing.

This directly fixed the TB accuracy problem documented above: TB's try-order now puts this
heuristic first (74%, beats both the Roboflow model's 0%-on-healthy and the offline HF ViT's 62%
on the same 50 held-out samples), Mammography gets it as a second offline option after the
(well-performing) Roboflow Workflow. `Validation/validate.py`'s TB ground-truth check — left
failing on purpose in the previous pass — now **passes for real**, because the underlying problem
got fixed rather than tolerated.

## Hardware planning docs (`Hardware/`)
New folder, purely documentation/diagrams — no app code touched. Compares the four hardware ideas
given (RK3588 SBC, Raspberry Pi, mobile APK, ESP32) as what they actually are: three candidate main
compute boards plus one companion MCU that can't run these models at all but is useful for the
GSM/telemetry link regardless of which board is chosen (`ALTERNATIVES.md`). Also: `HARDWARE_
REQUIREMENTS.md` + `BOM.txt` (parts + costs), `ARCHITECTURE.md` (data flow, maps onto the existing
`inference.py`/`offline_cv.py`/`GUI.py` stack), `WIRING.md` (pin-level, including the SIM800L
power-supply gotcha that's a common real-world failure mode for that module), `DESIGN.md`
(enclosure/power-resilience/thermal for an actual rural clinic), `STRUCTURE.md` (this folder's
layout + multi-BHU fleet structure), and 5 generated PNG diagrams (`diagrams/`, produced from
`_generate_diagrams.py` — reproducible from code, not a binary source-of-truth, same dark
teal/pink palette as the app itself). Explicitly flagged as an unbuilt, unbench-tested plan, same
honesty posture as `Documentations/MODEL_SOURCES.md` takes for the AI models.

## streamlit_app.py follow-up fixes
`core.draw_bbox()` is shared with `GUI.py`, so the offline heuristic's real bounding boxes were
already live in the Streamlit edition automatically — no change needed there. Two things weren't
automatic and needed fixing directly in `streamlit_app.py`: its own `"real inference" in source`
check (same bug pattern as `Validation/validate.py` had) was tagging offline_cv.py results as
"Simulated" in the telemetry log since that source string doesn't contain that exact phrase —
switched to the same `"SIMULATED" not in source` check; and the dashboard header / module
docstring still described the old "TB+Maternal real, Mammography simulated" state, updated to
reflect the current accuracy-ranked multi-method precedence.

## Validation
Added `Validation/validate.py` — a 14-check validation suite covering module imports, placeholder
image synthesis, the detection-overlay drawing, the simulated scenario generators, a full SQLite
cache round-trip (throwaway DB, never the real `pink_edge_cache.db`), text/PDF report generation,
real-model inference for TB and Maternal Health on both synthetic images and the real sample images
in `Test Data/`, the mammography SIMULATED-fallback path, the `run_triage()` dispatcher, that the
Tkinter UI itself builds and can run one full triage cycle with no visible window, and (added with
the Streamlit edition) that the Streamlit UI builds and runs one triage cycle headlessly via
`streamlit.testing.v1.AppTest`. One real issue was found and fixed while writing it: the `patient_id`
column (schema ported as-is from the original app) has SQLite TEXT affinity, so an inserted `int`
silently comes back as a `str` on read — not a bug in the app's own behavior, just a trap for any
test doing strict type equality on that column.

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
