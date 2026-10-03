# Model Sources — Pink Edge AI (Desktop)

Honesty convention carried over from the original project's own `MODEL_DOCUMENTATION.md`: every
modality below states exactly what is running behind it. Nothing here should be read as a clinically
validated product — these are public research/demo models and locally-trained classifiers plugged
into a hackathon prototype's existing UI, not FDA/CE-cleared diagnostic devices.

Each modality tries multiple methods in the order set by `predict_tb()` / `predict_maternal()` /
`predict_mammography()` in `inference.py` — the order is **not** "online first"; it's set by
measured accuracy against real held-out ground truth, same convention throughout this project.

| Modality | Try order | Measured accuracy |
|---|---|---|
| Mammography | Roboflow Workflow `breastcancer-yolov8-78tni` → offline pixel-diff heuristic → locally-trained classifier → local YOLO weights (rare) → Simulated | Roboflow: correctly flagged ground truth (90.9% confidence on one sample — not a % accuracy over many). Offline heuristic: **~96%**. Locally-trained classifier: **98.7%** (held out) — statistically tied with the heuristic given sample size, kept second since the heuristic needs no torch/torchvision at all. |
| Tuberculosis | **Locally-trained classifier** → offline pixel-diff heuristic → Roboflow model `tuberculosis-tp2pv/1` → offline HF ViT | Locally-trained classifier: **82.5%** (held out) — best measured of the 4. Offline heuristic: **74%**. Roboflow: known high-confidence false-positive issue, see below. |
| Maternal Health | Roboflow model `hash-maternal-health/1` → offline HF CNN | Roboflow: correctly flagged ground truth (88.3%). No locally-trained classifier exists for this modality (no `Models/Maternal/local_model.pt`) — `Models/Maternal/negative/` is also currently empty, so there is no manually-added negative data for one to train against yet. |

The Roboflow calls need a key (`roboflow_key.txt` / `ROBOFLOW_API_KEY` / Streamlit `st.secrets`) and
internet; the offline pixel-diff heuristic, the locally-trained classifiers, and the Hugging Face
models need neither, ever, once their weights/datasets are on disk.

## The locally-trained classifier (`train_tb_classifier.py`) — a real model, trained on this project's own data

Despite its name, `train_tb_classifier.py` is the training script behind both `Models/TB/local_model.pt`
and `Models/Mammography/local_model.pt` (Maternal has not been trained yet — see above). Architecture:
torchvision's MobileNetV3-Small with an ImageNet-pretrained backbone (frozen) and a linear head
trained from scratch — practical on CPU-only hardware. Trained on the dataset's own `train`+`valid`
splits plus anything manually dropped into `Models/<Modality>/positive|negative/`, evaluated on the
dataset's own held-out `test` split — never seen during training. Saves `Models/<Modality>/local_model.pt`
+ `local_model_metadata.json` (architecture, image size, sample counts, measured accuracy, training
date — so the file documents its own provenance).

This is a genuine **addition** to the lineup, not an assumed upgrade: `inference.py`'s
`load_local_model()` / `_predict_local_trained()` only move it ahead of an existing tier when it
measurably beats that tier on the same held-out methodology — true for TB (82.5% > offline
heuristic's 74%); for Mammography it's statistically tied with the offline heuristic (98.7% vs.
~96%), so it stays second, right after the heuristic.

**Previously unwired (fixed here):** `inference.py` already shipped `Models/TB/local_model.pt`,
`Models/Mammography/local_model.pt`, and their metadata files, but `predict_tb()` never attempted
to load them, and `predict_mammography()` called `load_mammography_model()` / `mammography_available()`
— names that did not exist anywhere in the file — so the Mammography tier would raise an uncaught
`NameError` in production the moment both the Roboflow workflow and the offline heuristic failed to
return a result, and `model_status()` would raise the same way whenever Roboflow was unconfigured
and the offline heuristic reported unavailable. Both are now defined and both local classifiers are
wired into their modality's try order (see `inference.py`'s "LOCALLY-TRAINED CLASSIFIER" section).

## The offline pixel-diff heuristic (`offline_cv.py`) — no model, no internet, ever

No trained weights, no API, nothing but files already in this repo. For each modality it averages
images from the project's own local labeled dataset (`Models/<Modality>/Data Set/*.coco`) into a
"typical positive" and "typical negative" reference template (grayscale, resized,
histogram-equalized), then for a new image: grayscale → resize → histogram-equalize → try identity
vs. horizontal-flip and keep whichever correlates better with a generic reference (handles
left/right laterality without full image registration) → pixel-wise absolute difference against
**both** templates → the region that's furthest from the negative template and closest to the
positive one becomes the highlighted bounding box (real detected coordinates, not an illustrative
fixed position) → the mean of that "leans positive" difference map becomes the confidence score.

Run `python offline_cv.py` to see its own measured accuracy against held-out data for yourself
(builds templates from train+valid only, tests against the held-out `test` split — a proper
train/test separation, not testing on training data).

## ⚠️ Known issue: `tuberculosis-tp2pv/1` (Roboflow) has a high false-positive rate

Validated against the project's own ground-truth COCO annotations
(`Models/TB/Data Set/tuberculosis.coco/test/_annotations.coco.json`), not just "does it run":
correctly flagged a `Tüberküloz` (active TB)-annotated sample as positive, but also called healthy
(`Sağlıklı`)-annotated samples TB Positive at high confidence — high-confidence wrong, not a
borderline call a threshold tweak would fix. Likely a training-quality issue aggravated by class
imbalance in the source dataset, not an integration bug. **This is why TB's try-order puts the
locally-trained classifier and the offline heuristic ahead of Roboflow** — not "prefer offline for
its own sake," but because both offline tiers are measurably more trustworthy for this specific
modality.

## Maternal Health — `shr3m/fetal-brain-plane-cnn`

Classifies fetal-**brain** ultrasound images into one of 4 standard planes (Trans-thalamic,
Trans-cerebellum, Trans-ventricular, Other) — narrower than full obstetric "maternal health" triage,
but it's the closest public match found; the original project's own docs call this pathway "the
least developed" of the three. Preprocessing: grayscale, resize to 224×224, normalize with the
checkpoint's published mean/std (0.17076 / 0.17093). Explicitly licensed CC BY 4.0 and the model
card states it is for research/education, not clinical decision-making — same posture the rest of
this app already takes.

## Mammography — Roboflow `b-davmu/breastcancer-yolov8` (deepest fallback tier only)

Not the primary Mammography path (that's the hosted Workflow `breastcancer-yolov8-78tni` in the
`imaad-ullah-khan-yameen` workspace — see the try-order table above); this is the last-resort tier,
reached only if the Workflow, the offline heuristic, *and* the locally-trained classifier all fail.
To populate it: put your key in `roboflow_key.txt` next to `GUI.py` (or set `ROBOFLOW_API_KEY`) and
restart the app — `inference.py`'s `fetch_roboflow_mammography_weights()` will then try to pull a
trained YOLO export for this separate Universe project automatically. Whether that succeeds depends
on whether the project actually has an exportable trained model version (vs. only an annotated
dataset, which Roboflow Universe projects sometimes only offer); if not, this tier stays unavailable
and the app falls through to the Simulated scenario picker instead of guessing.

## Considered and rejected: `Astaxanthin/KEEP` (suggested cancer model)

Investigated per a link shared mid-project. **Not used** — KEEP is a vision-language foundation model
for zero-shot cancer diagnosis on **histopathology** (H&E-stained biopsy slide) images, a different
imaging modality from digital **mammography** (radiographic breast X-ray). Plugging a histopathology
model into the mammography pathway would silently produce meaningless output on the wrong image
type, which conflicts with this project's own transparency principle — so it was left out rather than
force-fit.
