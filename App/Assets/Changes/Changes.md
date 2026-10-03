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

## Repo reorganization — everything moved into `App/`
Root cleaned up to exactly 3 visible things: `README.md`, `Start.bat`, `Start_Web.bat`. Everything
else (`GUI.py`, `inference.py`, `offline_cv.py`, `streamlit_app.py`, `requirements.txt`,
`roboflow_key.txt`, `pink_edge_cache.db`, `.streamlit/`, `.python-version`, `Models/`,
`Documentations/`, `Hardware/`, `Assets/`, `Test Data/`, `Validation/`, `Misc/`) moved into a new
`App/` folder as one unit, preserving relative structure — no internal `../`-style references broke
since everything that referenced everything else moved together. What *did* need updating: both
launchers now `cd` into `App/` before running anything; `.gitignore` patterns that had a `/` in them
(anchored, not "any depth") got an `App/` prefix; root `README.md` rewritten with the new paths,
including an honest note that the Streamlit Community Cloud deploy path (`Main file path:
App/streamlit_app.py` now, not `streamlit_app.py`) hasn't been re-verified against this layout yet.
Verified by actually running `Start.bat`/`Start_Web.bat` (not just the underlying Python) and the
full `Validation/validate.py` suite from the new location — 19/19 still pass.

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

## Local Android APK scaffold (`Apk/`, repo root, sibling to `App/`)
Added a from-scratch Kivy + Buildozer build, kept deliberately separate from `App/` rather than
nested inside it — Android build output and `App/`'s own desktop/web code don't belong in the same
tree. It is **not** a port of `GUI.py`/`streamlit_app.py`: `torch`/`transformers`/`ultralytics`/
`opencv-python` have no dependable `python-for-android` recipes, so the mobile app is a separate
two-tier dispatch (`mobile_inference.py`) using only `requests`/`numpy`/`Pillow`/`kivy`/`plyer`:
- Tier 1: Roboflow hosted REST API, called directly over HTTPS (no `inference-sdk` — too heavy),
  same model/workflow IDs as `App/inference.py`.
- Tier 2: the offline pixel-diff heuristic, reimplemented with pure numpy/Pillow (no OpenCV) against
  small template PNGs bundled into the APK by `generate_assets.py` (which reuses `App/offline_cv.py`'s
  own `_get_templates()` on a desktop, so the two editions' offline heuristic stays derived from the
  same reference images) instead of shipping the full multi-megabyte datasets.
- Same measured-accuracy precedence as desktop: TB tries the offline heuristic first, Mammography
  and Maternal try Roboflow first.
- Not built into a real APK in this environment — no Android SDK/NDK here, and Buildozer requires
  Linux (WSL2/VM). Written to Kivy + python-for-android's actual constraints, but unverified building
  until run for real; see `Apk/README.md`'s honesty note.

## RK3588 SBC deployment code (`RK3588 SBC/`, repo root)
Turns the hardware plan in `App/Hardware/` into runnable code for the chosen edge-node board:
- `install_rk3588.sh` + `pinkedge.service` — copies `App/` onto the board and autostarts `GUI.py` as
  a kiosk app on boot via systemd, with restart-on-crash.
- `uart_bridge.py` — the compute-subsystem side of the UART link to the ESP32/SIM800L companion from
  `App/Hardware/WIRING.md` (Option A): watches a local queue file for alert lines and forwards them
  over serial, with a `--dry-run` mode that needs no hardware. A one-time two-line addition to
  `GUI.py` (documented in `RK3588 SBC/README.md`, not made automatically) is what actually queues
  `sms_payload` for it to pick up.
- `convert_to_rknn.py` (run on an x86 dev machine) + `rknn_infer.py` (run on the board) — converts an
  ONNX-exported YOLOv8 mammography checkpoint to `.rknn` and runs it through the RK3588's NPU via
  `rknn-toolkit-lite2`, the NPU follow-up `App/Hardware/ARCHITECTURE.md` flagged as not blocking a
  first pilot. Neither has been run against a real converted model here — no RK3588 hardware and no
  local trained mammography checkpoint are available in this environment to test with.

## Manual positive/negative/validate folders per modality (`App/Models/*/{positive,negative,validate}/`)
Each modality's `offline_cv.py` template no longer depends solely on the COCO-annotated dataset: new
`positive/` and `negative/` folders let images be dropped in directly (no annotation step) and are
folded into template-building automatically, with the on-disk template cache now invalidating itself
whenever those folders' contents change (tracked via a small mtime signature in the cache's own
`metadata.json`) instead of needing a manual cache-clear. A third `validate/` folder is a
no-ground-truth spot-check: `python offline_cv.py` now also prints the heuristic's prediction for
every image dropped there, after the main calibration numbers. `_get_templates()` also no longer
requires the COCO dataset directory to exist at all — a modality with only manually-added images
still gets a working template. Added `python offline_cv.py --gather [--count N]`: seeds any
still-empty `positive`/`negative`/`validate` folder with real sample images pulled straight from
that modality's own `Data Set` (never touching folders a user has already added to). `positive`/
`negative` gathering is careful to pull only from the `train`/`valid` splits (never `test`), so it
can never quietly leak held-out data into the template and inflate `calibrate()`'s accuracy numbers
— `validate` gathering does the opposite on purpose, pulling only from the held-out `test` split,
since genuinely-unseen images are exactly what a spot-check folder wants.

## Locally-trained classifier per modality (`App/train_local_model.py`, `Models/*/local_model.pt`)
In addition to Roboflow, the offline pixel-diff heuristic, and the downloaded Hugging Face models,
each modality can now have a real classifier trained directly on this project's own data: a
torchvision MobileNetV3-Small (ImageNet-pretrained backbone, frozen) with a linear head trained from
scratch on `Models/<Modality>/Data Set/`'s `train`+`valid` splits (capped at 150 images/class, same
budget as `offline_cv.py`'s templates) plus anything in `positive`/`negative/`, evaluated on that
dataset's own held-out `test` split — never trained on, exactly like every other accuracy number in
this project. Run `python train_local_model.py [modality ...]` to (re)train.

**Measured, not assumed, before touching any precedence** (same convention as everywhere else in
this project): trained and evaluated all three modalities, then only moved the new tier ahead of an
existing one where it measurably won on the same held-out methodology:
```
tb:           82.5% (80 held-out samples)  -> beats offline heuristic's 74% -> now TB's tier #1
mammography:  82.1% (78 held-out samples)  -> below offline heuristic's 96% -> stays after it
maternal:     not trained -- dataset has zero negative images (same root cause as the offline
              heuristic's Maternal gap); the script refuses to train on one-class-only data rather
              than silently producing a model that always predicts positive
```
`inference.py`'s `predict_tb()`/`predict_mammography()`/`predict_maternal()` and `model_status()`
updated accordingly; `local_model.pt` (already covered by the existing `Models/*/*.pt` gitignore
rule) plus its paired `local_model_metadata.json` (architecture, image size, sample counts, measured
accuracy, training date) are both gitignored — reproducible by re-running the script, not hand-edited
state. Added a 20th validation check (`Locally-trained classifier: accuracy floor on held-out ground
truth`) mirroring the existing offline-heuristic one.

## ⚠️ Bug found and fixed: Mammography positive/negative folders were inverted
While training the locally-trained classifier above, discovered that `Models/Mammography/positive/`
and `negative/` (pre-populated by the user, independently of `offline_cv.py --gather`) used the
**opposite** convention from every other piece of this project: 81 healthy ("normal (N)"-named)
mammograms sitting in `positive/`, 156 cancer ("mdb###"-named) mammograms sitting in `negative/`.
Every template/model trained from these folders before the fix — the offline heuristic's 96% and the
locally-trained classifier's first 82.1% — was quietly built from mislabeled data. Fixed by swapping
the two folders' contents (verified 100% homogeneous by filename pattern first — only each folder's
own `README.md` wasn't part of the swap) to match the standard convention, then rebuilding
everything that reads from them:
```
offline heuristic (python offline_cv.py):    96% (48/50) -> 98% (49/50)
locally-trained classifier (retrained):       82.1% (64/78) -> 98.7% (77/78)
```
Both numbers *improved* once the mislabeling was fixed — exactly what should happen when training
data stops being contaminated. Mammography's locally-trained classifier is now statistically tied
with the offline heuristic (98.7% vs. 98%, not a real difference given the sample sizes) rather than
measurably worse; `inference.py`'s comments and every doc citing the old 82.1%/96% figures updated.
Saved as a standing memory (`mammography-positive-negative-convention`) so a future session
double-checks this user's manually-sorted image folders rather than assuming standard convention.

## Out-of-domain detection: reject the wrong kind of scan before triage
Per request: "if the image is out of source ... try outputting like wrong image uploaded" for all
three modalities. Fully offline, reuses `offline_cv.py`'s existing template infrastructure rather
than adding a new method: `domain_score()` is the best-orientation normalized cross-correlation
between the uploaded image and the modality's own generic reference image (average of whichever of
the positive/negative templates exist — doesn't require both, unlike full triage); below a
per-modality `DOMAIN_THRESHOLDS` cutoff, `is_out_of_domain()` flags it. `inference.py`'s new
`check_image_domain()` runs this as the very first step of `predict_tb()`/`predict_mammography()`/
`predict_maternal()` — before any tier, including Roboflow (so an obviously-wrong upload doesn't
cost an API call) — and returns a distinct "Wrong Image Type" result (`invalid_image: True` in the
result dict) instead of a triage verdict.

**Measured, not assumed** (`calibrate_domain()` in `offline_cv.py`, each modality's own held-out
test-split images vs. the other two modalities' as a "wrong kind of scan" stand-in):
```
tb:           threshold 0.47 -> 95% real scans pass, 97.5% wrong-modality images caught
maternal:     threshold 0.37 -> 80% real scans pass, 82.5% wrong-modality images caught
mammography:  threshold 0.03 -> 90% real scans pass, only 37.5% wrong-modality images caught
```
Mammography's threshold is deliberately lenient (biased toward never blocking a real mammogram)
because its scans vary too much in crop/zoom for this method to separate as cleanly as TB's more
standardized X-rays — documented honestly rather than hidden, same as the TB Roboflow issue above.
Verified separately that a genuinely unrelated image (random noise, a solid color) scores ~0 against
every modality's reference, reliably below every threshold — the hard cross-modality numbers above
likely understate real-world performance against an actually-wrong upload (a photo, a document).

UI: `GUI.py`'s `draw_bbox()` now special-cases `invalid_image` results with a plain "Wrong image
type — not analyzed" banner instead of drawing a modality-specific overlay that would falsely imply
a real region was localized; both `GUI.py` and `streamlit_app.py`'s verdict panels gained a third
(amber, `C["warning"]`) visual state alongside the existing danger/success ones, with matching
risk-banner text. Added a 21st validation check (`Out-of-domain gate: wrong image type rejected,
real scans pass through`) using random noise as the "wrong image" case (reliable across all three
modalities, unlike the imperfect cross-modality separation) and checking several real samples with a
majority-pass threshold rather than one sample with a must-always-pass assertion, since the gate is
a measured, imperfect heuristic (mammography especially) — not something a single unlucky sample
should be able to fail the whole suite over.

---

# Changes: Pink Edge AI → Medical Radiology AI

This file picks up where `Changes.md` (the "Pink Edge AI" era) leaves off. That file documents
everything that changed building Pink Edge AI from the original hackathon submission; this one
documents the rebuild into **Medical Radiology AI** — a new identity, theme, and layout on the
same proven triage engine.

## Why this rebuild happened

Requested directly: rebrand the app from Pink Edge AI to Medical Radiology AI, with a genuinely
distinct identity (not a find-and-replace job), a light blue / light gray visual theme in place of
the earlier dark navy/teal/pink one, and — done first, ahead of the rest — a reworked UI layout
(control menu moved to the right, more breathing room throughout, a couple of new options). A
fourth "Bone" modality (fracture/dislocation detection) was also requested but is **not** part of
this pass — deferred per "first of all work on ui," to be picked up next.

## Archive: the complete Pink Edge AI build preserved in Misc/

Per explicit instruction, everything that made up the Pink Edge AI build — not just its code, but
its datasets, trained-model cache config, validation suite, docs, and hardware plan — was moved
into `App/Misc/Pink_Edge_AI/` as a full, untouched snapshot, sibling to the already-archived
`App/Misc/Pink_Edge_AI-main/` (the original hackathon submission Pink Edge AI itself was built
from). Moved: `GUI.py`, `streamlit_app.py`, `inference.py`, `offline_cv.py`,
`train_local_model.py`, `requirements.txt`, `pink_edge_cache.db`, `.python-version`,
`roboflow_key.txt`, `Models/`, `Test Data/`, `Validation/`, `Documentations/`, `Hardware/`,
`Assets/`. Nothing in the archive was altered — it's the exact working state Pink Edge AI was in
before this rebuild started.

The new build then got **working copies** of the reusable infrastructure (`Models/`, `Test Data/`,
`Validation/`, `Hardware/`, `requirements.txt`, `.python-version`, `roboflow_key.txt`) copied back
out of the archive into fresh `App/` locations — datasets and trained-model configuration aren't
"Pink Edge" branding, they're just data, so re-downloading or re-sourcing them from scratch would
have been pure waste. This build has no runtime dependency on `Misc/` staying in place.

## New identity throughout

| Old (Pink Edge AI, archived) | New (Medical Radiology AI, active) |
|---|---|
| `GUI.py` | `radiology_console.py` |
| `streamlit_app.py` | `radiology_web.py` |
| `PinkEdgeApp` (class) | `RadiologyConsoleApp` |
| `pink_edge_cache.db` | `radiology_cache.db` |
| "Pink Edge AI" (title, headers, PDF/text report headers) | "Medical Radiology AI" |
| 🩸 (blood-drop icon) | 🩻 (X-ray icon) |

`inference.py`, `offline_cv.py`, and `train_local_model.py` kept their file names — they're
generic/functional names, not brand-specific — but had their "Pink Edge AI" docstring headers
updated. `Validation/validate.py` was updated throughout: its module docstring, print banner, and
content assertions now reference the new names; it still imports the desktop module under the
`GUI` alias internally (`import radiology_console as GUI`) so the ~50 `GUI.xxx` call sites
elsewhere in the file needed no individual changes — only the import line, the `PinkEdgeApp` class
reference, and the `streamlit_app.py` → `radiology_web.py` `AppTest.from_file()` path needed
touching. `Start.bat` / `Start_Web.bat` and the root `README.md` were rewritten for the new file
names and layout. `.gitignore` gained `radiology_cache.db` alongside the still-present
`pink_edge_cache.db` pattern (the archived copy still needs it ignored).

## New color theme: light blue / light gray

Replaced the earlier dark navy/teal/pink palette (`radiology_console.py`'s `C` dict, reused
verbatim by `radiology_web.py`'s injected CSS) with a bright clinical-console look:

```
bg:            #eef2f7   (page background)      was #0a0e1a
surface:       #ffffff   (cards)                 was #111827
surface_alt:   #f1f5f9   (nested panels)         was #1e293b
text:          #1e293b   (main text)             was #f1f5f9
primary:       #2f6fed   (blue, was teal)        was #14b8a6
accent:        #0ea5e9   (sky blue, was pink)    was #ec4899
console_bg/fg: #e7ecf3 / #334155  (new tokens — a light "terminal" panel for the telemetry/
               hardware-diagnostics readouts, replacing a hardcoded near-black #060a13)
```

`success` / `warning` / `danger` were left as their original semantic traffic-light values —
they're universal, not brand colors. Every place that previously hardcoded the old dark console
color (`#060a13`) now uses the new `console_bg`/`console_fg` tokens instead, so there's no
leftover dark panel anywhere in the light theme.

## Layout: menu moved to the right, more relaxed spacing, new options

Requested changes, done as a deliberately visible restructuring rather than a cosmetic tweak:

- **Desktop (`radiology_console.py`)**: added a full-width top header band (branding + tagline)
  that didn't exist before; the control sidebar now packs on the **right** (`side="right"`) instead
  of the left, with the main notebook content taking the left. Padding was widened throughout
  (14px → 18px in the sidebar, 14px → 18px in the metadata/verdict panel, more `pady` between
  sections) for a less cramped, more "relaxed" feel. The dashboard's image card grew a visible
  border/card treatment, metric tiles got more internal padding, and the redundant in-tab header
  (which repeated the app name already shown in the new top band) was replaced with a plainer
  "Medical Image Analysis" section heading.
- **Web (`radiology_web.py`)**: Streamlit has no official right-sidebar API, so the menu move is a
  CSS hack — `div[data-testid="stAppViewContainer"] { flex-direction: row-reverse; }` — flagged in
  a code comment as depending on Streamlit's current internal DOM structure, since it isn't a
  documented API and could need revisiting on a future Streamlit upgrade. Card/verdict-box/
  metric-tile border-radius increased (10–14px → 14–20px) and padding widened to match the
  desktop's relaxed spacing.
- **New sidebar options** (both editions): a **Clinician Notes** free-text field — its content, if
  non-empty, now flows into both the text and PDF reports as a "CLINICIAN NOTES" section (threaded
  through `_current_report_dict()`/`_save_cache()`'s data dict on desktop and the equivalent
  `report_data`/`save_cache_action()` dict on web, then read by `generate_text_report()`/
  `generate_pdf_bytes()`) — and an **About** panel with a short app description and a pointer to
  `Documentations/MODEL_SOURCES.md`.

Verified after every change: `python Validation/validate.py` — **21/21** — and both `Start.bat`
and `Start_Web.bat` actually launched (window title confirmed as "Medical Radiology AI — Clinical
Intelligence Platform (Desktop)"; the Streamlit server confirmed serving HTTP 200).

## Not done in this pass

- The original hackathon-era reference docs carried into `Documentations/` (`API_DOCUMENTATION.md`,
  `BACKEND_DOCUMENTATION (2).md`, `FRONTEND_DOCUMENTATION (1).md`, `MODEL_DOCUMENTATION.md`,
  `PROJECT_ARCHITECTURE.md`, `USER_GUIDE.md`) were left as-is except where they actively described
  the current build (`MODEL_SOURCES.md`, `Documentations/README.md`'s title/overview) — they're
  historical/reference material describing the original hackathon pitch, not "UI."
- `Apk/` and `RK3588 SBC/` (repo root) still reference "Pink Edge AI" in places — not touched this
  pass, since neither is part of the desktop/web UI this request was about.

---

## Batch 2: Bone X-Ray modality (replaces Maternal Health) + a stray doc fix

Requested directly: swap the third modality from Maternal Health to **Bone X-Ray** (fracture /
dislocation triage), and fix the one remaining stale "main file" reference left over from the
previous pass (`Documentations/README.md`'s install steps still named the *original* hackathon's
single-file app and a dead clone URL — not `radiology_console.py`/`radiology_web.py`).

### Doc fix: `Documentations/README.md`
Its "Installation" section still said `git clone .../Zobia-Irshad/Pink_Edge_AI.git`, `cd
pink-edge-ai`, and `streamlit run pink_edge.py` — all three predate even the "Pink Edge AI"
desktop/web build (they're artifacts of the *original* single-file hackathon submission). Fixed to
point at this repo (`Danger-Khan/Medical-Radiology-AI`) and `streamlit run radiology_web.py` /
`python radiology_console.py`. Its modality list, "Multi-Modal" feature bullet, and "Solution"
section bullet were updated from Maternal Health to Bone X-Ray alongside the modality swap below.

### Bone X-Ray: what backs it, honestly

Bone is structured differently from the other three modalities because no single public model
answers "is this OK, a Crack, or a Shift" in one shot — the design and every number below came from
actually probing real Roboflow projects and Hugging Face models (via the same API key already
configured), not from assuming a suitable dataset existed:

- **Presence of a fracture at all** (OK vs. not) — real-time: Roboflow's public
  `yakin/bone-fracture-tn84w` project (workspace `yakin`, not this project's own workspace — same
  pattern as Mammography's `b-davmu/breastcancer-yolov8` placeholder), specifically **version 1**,
  the only one of its 3 versions with an actually trained/deployed model (grounded via a real API
  call to its training summary: precision 86.5%, recall 71.1%, mAP@50 77.3%). Offline fallback:
  `prithivMLmods/Bone-Fracture-Detection` (Hugging Face, SigLIP2, Apache-2.0, ~83% accuracy per its
  model card) — added a `transformers` dependency to `requirements.txt` specifically for this, since
  TB/Mammography's HF integrations were deliberately hand-built to avoid it.
- **Sub-type, Crack vs. Shift** (only once "fracture" is confirmed) — the locally-trained classifier,
  then the offline pixel-diff heuristic, both trained on a **different, richer** export of the same
  dataset (version 3 — 2147 images, real per-type category annotations including `Dislocation`,
  CC BY 4.0), downloaded and placed at `Models/Bone/Data Set/Bone-Fracture.coco/` — this version has
  no Roboflow-hosted model of its own, but `offline_cv.py`/`train_local_model.py` don't need one,
  they train directly on the raw annotated images. `offline_cv.py`'s positive/negative template
  slots are reused for Shift/Crack here instead of disease/healthy — documented explicitly in
  `Models/Bone/positive|negative/README.md` and `Documentations/MODEL_SOURCES.md` so it isn't a
  silent semantic swap.
- **Measured, and honestly weak**: the sub-typer is barely better than chance (offline heuristic
  56%, local classifier 58.9%, both on real held-out data) — visually telling a dislocation from
  another fracture type by pixel pattern alone turned out to be genuinely hard with this method.
  The presence detector fares worse than its official numbers suggest when checked against this
  project's own re-annotated ground truth (2/16 known-Dislocation samples actually detected) — a
  real train/test mismatch between the deployed v1 model and the v3 annotations used to grade it,
  not a bug. Both are written up under new "⚠️ Known issue" / accuracy-table entries in
  `MODEL_SOURCES.md`, the same honesty convention already used for TB's Roboflow bias — **Bone is
  currently the weakest-measured modality in this app**, and that's stated plainly rather than
  glossed over.
- Out-of-domain gating for Bone is similarly weak and documented as such: real bone X-rays vary too
  much in framing (wrist vs. skull vs. shoulder) for the generic pixel-correlation reference to
  separate them from other radiograph types — threshold set low (like Mammography) to protect real
  scans (100% pass) rather than overstate its ~7% wrong-upload catch rate.

### Everywhere the swap touched
`inference.py` (new `predict_bone()`/`load_bone_model()`/`BONE_STATUS_OPTIONS`/
`BONE_TYPE_OPTIONS`, `predict_maternal()` and its Hugging Face CNN loader removed entirely),
`offline_cv.py` (`_DATASETS["bone"]`, a bone-specific `_to_bone_result()`, a new domain threshold),
`radiology_console.py` and `radiology_web.py` (model list, simulated scenarios, placeholder image
generator, bounding-box overlay, the "Confirm Assessment" combo-box pair, every `mod_map`/About-text
mention), `Validation/validate.py` (every Maternal-specific check rewritten for Bone, including a
new ground-truth cross-check calibrated against the measured 2/16 real hit rate so it's a genuine
regression guard rather than a flaky or rubber-stamped assertion), `train_local_model.py` needed
**no changes at all** — it's already generic over `offline_cv._DATASETS`. `Models/Maternal/` (its
dataset, positive/negative/validate folders) is left in place, now unreferenced by any code —
already fully preserved, unedited, at `App/Misc/Pink_Edge_AI/Models/Maternal/` from the earlier
archive move, so nothing here risks losing it; removing the now-dead active copy wasn't requested,
so it was left rather than guessed at.

Verified: `python Validation/validate.py` — **21/21** (this actually caught a real bug the first time
through: the Bone domain-gate threshold was initially set to protect 100% of real scans, which
silently let random noise pass too — see `offline_cv.py`'s `DOMAIN_THRESHOLDS["bone"]` comment for
the fix and why no threshold can fully solve both at once for this modality) — and both `Start.bat`
and `Start_Web.bat` confirmed launching live with "Bone X-Ray (Fracture/Dislocation)" selectable in
place of Maternal Health (window title and HTTP 200 both confirmed, same as every prior pass).

---

## Compliance & evidence pass — offline validation, fabrication audit, privacy, theme

Author Name:  Imaad Ullah Khan
Author Email: yameenimaad@gmail.com
AI Helper:    Claude

Picks up after the merge/rebrand and file-consolidation passes above. Four instruments were added
so the project's central claims are *checkable* rather than asserted, and two real defects they
found were fixed.

### Theme unified across all four places it was defined

The palette lived in four places that had drifted apart: `GUI.py`'s `C` (indigo `#1a237e` + pink
accent), `streamlit_app.py`'s own separate `C` (teal `#0d9488` + sky), ~100 hardcoded hex values in
that file's CSS (a hotpink `#ff69b4`/`#ff1493` brand gradient, plus two different greens, three reds
and three ambers for the same semantic roles), and `.streamlit/config.toml` (teal, with a
`backgroundColor` that disagreed with the CSS). An app named **Pink** Edge AI, whose flagship
modality is breast screening, had no pink at all in its web palette.

Now one palette, defined once in `GUI.py` and imported by `streamlit_app.py` as `C = core.C`, so the
two editions cannot silently diverge again. Brand is rose (`#be185d`/`#9d174d`), secondary is sky,
neutrals are one slate scale, and the semantic colours are shared exactly — they carry clinical
meaning, so a danger verdict must look the same in both editions. Tint tokens
(`primary_bg`/`success_bg`/`warning_bg`/`danger_bg`) replaced hardcoded card backgrounds, and the
dark console panels became named tokens (`console_bg`/`console_text`) rather than looking like
leftovers of the old dark theme — they are a deliberate terminal-style readout, so they stayed dark.

This also fixed a real accessibility defect, not just an inconsistency: the old success/warning/
danger colours **failed WCAG AA as text on white** (`#10b981` was 2.3:1, `#f59e0b` 1.9:1, `#ef4444`
3.3:1), as did the Radiologist role badge (`#a855f7`, 3.5:1). All are now darker shades of the same
hues, and the smoke test asserts a 4.5:1 floor so it cannot regress.

### Tests/smoke_test.py — the fast check (9 checks, ~30s, no model downloads)

Starts the real Streamlit app through `AppTest` and drives role switching, the language toggle, all
three modalities, Run Triage, and Save to Cache (asserting the cache actually gained a row); then
checks theme consistency across the three definition sites, that the retired teal/hotpink values
have not returned, that branding is Pink Edge AI with no `Medical Radiology`/`radiology_console`
references, and the contrast floor. Sidebar controls are looked up **by label, never by index** —
positional lookup silently broke once already in this project when a role switcher was inserted
ahead of the language buttons. Verified the checks can fail: reverting `config.toml` to the old teal
trips two of them with precise messages.

### Validation/validate_offline.py — the offline claim, actually tested (13 checks)

The project's headline claim is that triage works with zero internet. This proves it by *removing*
the network rather than inspecting code: every socket entry point is replaced before `inference.py`
is imported (subclassing `socket.socket` so libraries that inherit from it still work, and allowing
loopback so local tooling is not collateral damage), and Hugging Face is pinned to offline mode.

A **control check runs first** and attempts three escape routes — `create_connection`, `getaddrinfo`,
and a real HTTPS GET — failing the suite if any succeeds, so a pass can never be an artefact of
accidentally having had connectivity. It then establishes that all three modalities still produce
well-formed results, that **none of them claims a hosted/Roboflow source** while nothing is
reachable, that each offline tier works independently, and that the cache, reports and both UIs all
function. 13/13.

### Tests/fabrication_audit.py — which displayed numbers are measured, and which are invented

A clinician reading a triage screen cannot tell which fields a model produced and which the program
generated. This answers that empirically: the **same image is submitted three times** and the result
dicts are diffed. A field derived from the image is identical every time; a field that changes was
not a function of the input. No source-code knowledge needed, and it cannot be fooled by a comment
claiming something is real. It then classifies the finding by what the UI *told the reader* —
fabrication inside a result labelled SIMULATED is honest; fabrication inside one claiming real
inference is flagged.

What it found, on real Test Data samples:

| Modality | Source shown to the reader | Result |
|---|---|---|
| Tuberculosis | `local_model.pt` (locally-trained MobileNetV3) | **clean** — every clinical field identical across runs |
| Mammography | Roboflow workflow, "real inference" | `confidence` varies **94.3 / 96.6 / 98.1** on one image |
| Maternal Health | Roboflow `hash-maternal-health/1`, "real inference" | `extra` varies **Gestational Age: 20W / 25W / 24W**, `confidence` varies |

The gestational age is the notable one: **no model in the Maternal chain estimates gestational age
at all** (`inference.py:563,656` — `ga = random.randint(18, 38)`), yet it is presented in a result
whose source string says real inference. Same pattern for the no-detection confidence branches
(`inference.py:346,565,772,846`). These were **reported, not silently changed** — what a demo should
display when a model cannot produce a field is a product decision, not a cleanup.

### Tests/privacy_leak_test.py — PII handling at every sink (10 checks)

Checks the README's specific promise ("stripped of PII and assigned a hex privacy hash before
anything is displayed, cached, or transmitted") against what the code does, at each place data can
leave a record: the GSM/IoT broadcast, the SQLite cache, generated reports, and credential handling.

**It found a real privacy defect and it was fixed.** The web edition correctly broadcast
`ID:<hash>|LOC:ANON|…`, but the **desktop edition had never used the anonymizer at all** — zero
references to `dicom_anonymizer` in `GUI.py` — and broadcast `ID:{self.pat_id}|LOC:29.344|…`: the raw
identifier plus a literal coordinate, directly contradicting the documented claim. `GUI.py` now
derives a privacy hash via `_privacy_hash_for()` (mirroring the web edition's input shape, so the
same patient hashes identically in both) and uses it for the GSM payload, the IoT queue and the
hospital-hub alert feed. 10/10 after the fix.

### Evidence/generate_evidence.py → Evidence/EVIDENCE_REPORT.md

Runs all seven suites, records their actual exit codes and output, and writes a dated evidence
document with the **SHA-256 of every attested source file**, so the report cannot silently outlive
the code it describes. It distinguishes `FINDINGS` from `FAIL` for the fabrication audit — a
diagnostic that successfully finds something is not a broken suite, and recording it as one would
make the evidence less truthful. The report also states scope limits plainly: synthetic identifiers,
no clinical validation, simulated cloud/GSM/telemetry.

### Documentations/PRIVACY_POLICY.md

Written to be checkable rather than reassuring — every claim maps to something the privacy suite
verifies. It is explicit about what is *not* protected: the cache is unencrypted at rest, the
privacy hash is **unsalted** (an 8-digit ID has only ~90M possibilities, so it is brute-forceable
and inadequate for real identifiers), the RBAC PINs are hardcoded and are a UI demonstration rather
than a security control, and configuring a Roboflow key means images do leave the device.

### Attribution

Author headers (`Author Name` / `Author Email` / `AI Helper: Claude`) added to all 24 Python files
across `App/`, `Apk/` and `RK3588 SBC/`, inserted into each module docstring — or as a comment block
for the two files that had none (`Tools/receiver.py`, which also gained a real docstring, and
`train_tb_classifier.py`, where `from __future__` must stay first).

Verified: `validate.py` 19/19 · `validate_offline.py` 13/13 · `smoke_test.py` 9/9 ·
`privacy_leak_test.py` 10/10 · `test_auth.py` 16/16 · `test_anonymizer.py` 21/21 ·
`fabrication_audit.py` reports its findings as designed.
