# Model Sources — Medical Radiology AI

Honesty convention carried over from the original project's own `MODEL_DOCUMENTATION.md`: every
modality below states exactly what is running behind it, including where it gets things wrong.
Nothing here should be read as a clinically validated product.

Each modality tries multiple methods in order, per `predict_tb()` / `predict_bone()` /
`predict_mammography()` in `inference.py` — the order isn't fixed to "online first": it's set by
**measured accuracy against real held-out ground truth**, not by assumption.

| Modality | Try order | Measured accuracy |
|---|---|---|
| Mammography | Roboflow Workflow `breastcancer-yolov8-78tni` → offline pixel-diff heuristic → **locally-trained classifier** → local weights (rare) → Simulated | Roboflow: one ground-truth sample correctly flagged (90.9% confidence — not a % accuracy over many samples, see below). Offline heuristic: **98%** (49/50 held-out). Locally-trained classifier: **98.7%** (77/78 held-out) — tied with the heuristic within sample-size noise. |
| Tuberculosis | **Locally-trained classifier** → offline pixel-diff heuristic → Roboflow model `tuberculosis-tp2pv/1` → offline HF ViT | Locally-trained classifier: **82.5%** (80 held-out samples) — best measured of the 4. Offline heuristic: 74%. Offline HF ViT: 62%. Roboflow: 0% on healthy samples — **known issue, see below**. |
| Bone X-Ray | **Presence** (OK vs. fracture): Roboflow model `bone-fracture-tn84w/1` → offline HF SigLIP2. **Sub-type** (Crack vs. Shift, only once presence says "fracture"): locally-trained classifier → offline pixel-diff heuristic → generic "not determined" | Presence (Roboflow): only **2/16** known-Dislocation ground-truth samples actually detected — see the "known issue" writeup below, this is the weakest-measured tier in the app. Sub-type: locally-trained classifier **58.9%** (33/56 held-out), offline heuristic **56%** (23/41 held-out) — both barely above chance; see below. |

Before any real triage tier runs, every modality is also checked for **out-of-domain input** — "does
this even look like the right kind of scan at all" — see the section near the bottom of this file.

The Roboflow calls need a key (`roboflow_key.txt` / `ROBOFLOW_API_KEY` / Streamlit `st.secrets`) and
internet; the offline pixel-diff heuristic (`offline_cv.py`), the locally-trained classifier
(`Models/<Modality>/local_model.pt`), and the Hugging Face models need neither, ever, once their
weights/datasets are on disk.

## The locally-trained classifier (`train_local_model.py`) — a real model, trained on this project's own data

Run `python train_local_model.py` to (re)train it. Architecture: torchvision's MobileNetV3-Small
with an ImageNet-pretrained backbone (frozen) and a linear head trained from scratch — practical on
CPU-only hardware. Trained on the dataset's own `train`+`valid` splits (capped at 150 images/class,
same sample budget as `offline_cv.py`'s templates) plus anything manually dropped into
`Models/<Modality>/positive|negative/`, and evaluated on the dataset's own held-out `test` split —
**never seen during training**, exactly like every other accuracy figure in this file. Saves
`Models/<Modality>/local_model.pt` + `local_model_metadata.json` (architecture, image size, sample
counts, measured accuracy, training date — so the file documents its own provenance).

This is a genuine **addition** to the lineup, not an assumed upgrade: `inference.py` only moves it
ahead of an existing tier when it measurably beats that tier on the same held-out methodology — true
for TB (82.5% > offline heuristic's 74%); for Mammography it's statistically tied with the offline
heuristic (98.7% vs. 98%, on 78 vs. 50 samples — not a real difference), so it stays second, right
after the heuristic, since the heuristic needs no torch/torchvision at all. Bone is the one modality
where this classifier answers a genuinely different question than "healthy vs. diseased" — see the
Bone section below for why, and for its honestly-modest 58.9% held-out accuracy.

⚠️ **A labeling bug was found and fixed here** (2026-09-13): `Models/Mammography/positive/` and
`negative/` had been manually pre-populated with real images using the **opposite** convention this
project uses everywhere else (COCO categories, `offline_cv.py`, this script) — 81 healthy scans
sitting in `positive/` and 156 cancer scans sitting in `negative/`. Every template/model trained
before the fix was quietly contaminated with mislabeled data pulled from those folders. Caught
before shipping, fixed by swapping the two folders' contents to match the standard convention
(`positive` = disease/abnormal, `negative` = healthy — see each folder's own `README.md`), then
retraining/recalibrating everything that reads from them — the numbers in this file are all
POST-fix. Mammography's accuracy actually improved once fixed (offline heuristic 96%→98%, local
classifier 82.1%→98.7%), consistent with "mislabeled training data hurts, not helps."

## The offline pixel-diff heuristic (`offline_cv.py`) — no model, no internet, ever

Built per your request for a genuinely offline method: no trained weights, no API, nothing but
files already in this repo. For each modality it averages images from the project's own local
labeled dataset (`Models/<Modality>/Data Set/*.coco`) into a "typical positive" and "typical
negative" reference template (grayscale, resized, histogram-equalized), then for a new image:
grayscale → resize → histogram-equalize → try identity vs. horizontal-flip and keep whichever
correlates better with a generic reference (handles left/right laterality without full image
registration) → pixel-wise absolute difference against **both** templates → the region that's
furthest from the negative template and closest to the positive one becomes the highlighted
bounding box (real detected coordinates — `draw_bbox()` in `radiology_console.py` now draws these when present,
not just an illustrative fixed position) → the mean of that "leans positive" difference map becomes
the confidence score.

**Measured, not assumed** — `python offline_cv.py` builds templates from train+valid only and
tests against the held-out `test` split (proper train/test split, not testing on training data):

```
[tb]           positive 16/25 | negative 21/25 | overall 37/50 (74%)
[mammography]  positive 25/25 | negative 24/25 | overall 49/50 (98%)
[bone]         positive 12/16 | negative 11/25 | overall 23/41 (56%) -- see below, this is NOT
               healthy-vs-diseased like the other two rows
```

**Bone reuses the positive/negative template slots for a different question than every other
modality** — Shift (Dislocation) vs. Crack (every other fracture type), not disease-vs-healthy. The
source dataset (`yakin/bone-fracture-tn84w`, CC BY 4.0) has **zero normal/healthy bone X-rays
annotated at all** — all 2147 images show some kind of fracture — so there's no "OK" reference to
build here; "no fracture" is decided separately, by the real-time Roboflow/Hugging Face presence
detectors (see the Bone section below). What this heuristic (and the locally-trained classifier)
measurably CAN'T do well is tell Shift from Crack by pixel pattern alone: 56% (offline heuristic) and
58.9% (locally-trained classifier) are both barely better than a coin flip, and both are worse than
the trivial "always guess Crack" baseline for this held-out split (Crack outnumbers Shift roughly
6:1). Kept wired rather than removed — it's still real, measured, and occasionally right — but
treat any Shift/Crack sub-type call with real skepticism until a better-suited approach is found.

## ⚠️ Known issue: `tuberculosis-tp2pv/1` (Roboflow) has a high false-positive rate

Validated against the project's own ground-truth COCO annotations
(`Models/TB/Data Set/tuberculosis.coco/test/_annotations.coco.json`), not just "does it run":
correctly flagged a `Tüberküloz` (active TB)-annotated sample as positive (91.2%), but called
**all 3** `Sağlıklı` (healthy)-annotated samples tested TB Positive too, at **98–99% confidence** —
high-confidence wrong, not a borderline call a threshold tweak would fix. Only one version of this
project exists on Roboflow (checked via the API). Likely a training-quality issue aggravated by
class imbalance (432 `Tüberküloz` vs. 159 `Sağlıklı` annotations in just the test split), not an
integration bug. **This is why TB's try-order puts offline methods first** — it's not "prefer
offline on principle," it's that both the locally-trained classifier (82.5%) and the offline
heuristic (74%) measurably beat the Roboflow model (0% on healthy) and the offline HF ViT (62%) on
the same held-out data. Revisit this ordering if the Roboflow model is retrained with better class
balance.

## ⚠️ Known issue: `bone-fracture-tn84w/1` (Roboflow) barely detects this project's own re-annotated images

Validated against the project's own ground-truth COCO annotations
(`Models/Bone/Data Set/Bone-Fracture.coco/test/_annotations.coco.json`): of 16 held-out test images
annotated `Dislocation`, the deployed model detected a fracture region on only **2**. That deployed
model (version 1 of the public `yakin/bone-fracture-tn84w` project) reports a real, officially
measured 71.1% recall — but that number comes from ITS OWN 2023 single-class training/test split,
not from this project's richer 2025 re-annotation (version 3, used for the Shift/Crack sub-typer —
see below) — the two versions don't share exact train/test boundaries, so this is a genuine
train/test mismatch between the deployed model and the ground truth used to check it here, not a
fabricated or copy-pasted number. **Bone is the weakest-measured modality in this app** as a direct
consequence — treat every "OK" verdict from it with more caution than the other three modalities'
"no detection = normal" results, since a real fracture is more likely to be missed here than caught.

## Details

### Mammography — Workflow `breastcancer-yolov8-78tni`
Grounded via a real call: detections come back as
`{"x", "y", "width", "height", "confidence", "class", "class_id"}` inside
`result[0]["predictions"]["predictions"]`; class `"cancer"` seen on a real positive sample, empty
list on real negatives. No local trained-weight export was found for the deepest fallback
(`b-davmu/breastcancer-yolov8`, a *different*, public Roboflow Universe project used only as a
placeholder before the user's own workspace/model was available) — falls to SIMULATED only if
Roboflow is unreachable *and* the offline heuristic has no local dataset either.

### Tuberculosis — offline heuristic (primary) / Model `tuberculosis-tp2pv/1` / `sukhmani1303/tuberculosis-vit-model`
Real class taxonomy for the Roboflow model, grounded from the project's own COCO export (Turkish
labels): `Sağlıklı`=Healthy, `Sekelli`=old healed sequelae, `Gizli`=latent infection,
`Hastalıklı`=diseased (non-specific), `Tüberküloz`=active TB. Its API response returns these
**ASCII-folded** (`"Tuberkuloz"`, not `"Tüberküloz"`) — `inference.py`'s `_normalize_class_name()`
accent-strips both sides before matching rather than hardcoding one spelling.
Offline HF fallback: binary ViT classifier (`Normal`/`Tuberculosis`), TorchScript-exported, loaded
via `torch.jit.load`. Preprocessing (ported exactly from the repo's `handler.py`): grayscale →
CLAHE → Gaussian blur → resize to 224×224 → back to RGB → per-image z-score normalization.
Apache-2.0.

### Bone X-Ray — Model `bone-fracture-tn84w/1` (presence, primary) / `prithivMLmods/Bone-Fracture-Detection` (presence, fallback) / local Shift-vs-Crack sub-typer
Two-stage design, unlike the other three modalities' single "is this abnormal" call:

1. **Presence** (OK vs. some kind of fracture) — `bone-fracture-tn84w/1`, a **public** Roboflow
   Universe project (workspace `yakin`, not this project's own workspace), callable with any valid
   Roboflow API key the same way `b-davmu/breastcancer-yolov8` was used as a Mammography placeholder
   before the user's own model existed. Grounded via a real call to its training summary: real
   measured precision 86.5%, recall 71.1%, mAP@50 77.3% — but see the "known issue" above for how
   that recall doesn't carry over cleanly to this project's own re-annotated ground truth. Its own
   trained taxonomy is a single merged "fracture present" class (checked via the Roboflow REST API,
   not assumed) — it cannot itself tell a crack from a dislocation. Fallback:
   `prithivMLmods/Bone-Fracture-Detection`, a SigLIP2 (`google/siglip2-base-patch16-224`) binary
   classifier (`Fractured`/`Not Fractured`, ~83% accuracy per its own model card), Apache-2.0, run via
   a `transformers` `pipeline("image-classification", ...)`.
2. **Sub-type** (Crack vs. Shift), only once presence says "fracture" — the locally-trained
   classifier, then the offline pixel-diff heuristic (see above for their modest 56–59% measured
   accuracy), both trained on the real Dislocation-vs-other-fracture-type category split from this
   project's own copy of the dataset's richer v3 export (`Models/Bone/Data Set/Bone-Fracture.coco`,
   2147 images, CC BY 4.0 — downloaded directly from the Roboflow API's COCO export, independent of
   whether that specific version has a Roboflow-hosted trained model, which v3 does not). If neither
   sub-typer is available, `inference.py` reports a generic "Crack — sub-type not determined" rather
   than guessing.

### Locally-trained classifier — `Models/<Modality>/local_model.pt`
Produced by `train_local_model.py`, not downloaded from anywhere — see the section above. TB:
82.5% (80 held-out samples: 40 positive + 40 negative). Mammography: 98.7% (78 held-out samples: 40
positive + 38 negative). Bone: 58.9% (56 held-out samples: 16 Shift + 40 Crack) — see the Bone
section above for why this number is honestly weak, not a bug.

### Considered and rejected: `Astaxanthin/KEEP` (suggested cancer model)
KEEP is a vision-language foundation model for zero-shot cancer diagnosis on **histopathology**
(H&E-stained biopsy slide) images — a different imaging modality from digital **mammography**
(radiographic breast X-ray). Left out rather than force-fit onto the wrong image type.

## Out-of-domain detection — "does this even look like the right kind of scan at all"

Checked by `inference.py`'s `check_image_domain()` before any real triage tier runs (and before any
Roboflow API call, so an obviously-wrong upload doesn't cost a network round-trip) — if flagged, the
UI shows "Wrong Image Type" instead of a triage verdict, with no bounding box/localization claim
(see `draw_bbox()` in `radiology_console.py`). Fully offline: `offline_cv.py`'s `domain_score()` reuses the same
templates as the triage heuristic — normalized cross-correlation between the uploaded image and the
modality's own generic reference image (the average of whichever of the positive/negative templates
exist), independent of best-orientation search.

**Measured, not assumed** (`calibrate_domain()` in `offline_cv.py`, run via `python offline_cv.py`):
each modality's own held-out test-split images vs. the OTHER two modalities' held-out test-split
images, used as a **deliberately hard** "wrong kind of scan" stand-in (a real medical grayscale
image, not an easy case like a random color photo):

| Modality | Threshold | Real scans correctly pass | Wrong-modality images correctly caught |
|---|---|---|---|
| Tuberculosis | 0.47 | 95% | 97.5% |
| Mammography | 0.03 | 90% | 37.5% |
| Bone X-Ray | 0.0 | 93% (14/15) | ~13% |

**Mammography's catch rate is deliberately weak, and Bone's is weaker still** — Bone X-rays vary even
more than mammograms in framing (a wrist vs. a skull vs. a shoulder are barely comparable images), so
a single generic reference template can't separate them from other radiograph types this way at all:
measured in-domain (mean score 0.28) and out-of-domain (mean score 0.30) distributions almost
entirely overlap, and the single worst real bone sample measured (-0.11) scores BELOW random noise
(~0.00) — meaning no threshold can both pass every real bone X-ray and reject noise for this
modality; unlike the other two rows, this is a hard limit of the method here, not a tuning choice.
0.0 is the threshold actually used: it reliably catches random noise / blank images (confirmed by
`Validation/validate.py`) while still passing 93% of real bone X-rays, rather than the more lenient
negative threshold that was tried first and passed noise straight through. Tested against a
genuinely unrelated image (random noise or a solid color, not another medical scan) TB and
Mammography score near 0 — well below their own thresholds — so a real-world "wrong image" upload
(a photo, a document, a screenshot) should be caught far more reliably than this hard cross-modality
proxy suggests for those two; that easier case just isn't independently measured here, and for Bone
specifically, noise sits right at the edge of its own threshold rather than clearly below it.
