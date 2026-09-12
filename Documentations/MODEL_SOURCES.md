# Model Sources — Pink Edge AI (Desktop)

Honesty convention carried over from the original project's own `MODEL_DOCUMENTATION.md`: every
modality below states exactly what is running behind it. Nothing here should be read as a clinically
validated product — these are public research/demo models plugged into a hackathon prototype's
existing UI, not FDA/CE-cleared diagnostic devices.

| Modality | Status | Model | Source | License |
|---|---|---|---|---|
| Tuberculosis (Chest X-Ray) | **Real** | Vision Transformer, TorchScript | [sukhmani1303/tuberculosis-vit-model](https://huggingface.co/sukhmani1303/tuberculosis-vit-model) | Apache-2.0 |
| Maternal Health (Ultrasound) | **Real** | Custom 4-block CNN (fetal-brain plane classifier) | [shr3m/fetal-brain-plane-cnn](https://huggingface.co/shr3m/fetal-brain-plane-cnn) | CC BY 4.0 |
| Mammography (YOLOv8-OBB) | **Simulated** unless a Roboflow API key with a trained export is supplied | — | [b-davmu/breastcancer-yolov8](https://universe.roboflow.com/b-davmu/breastcancer-yolov8) (Roboflow Universe) | project-defined |

## Details

### Tuberculosis — `sukhmani1303/tuberculosis-vit-model`
Binary chest X-ray classifier (`Normal` vs `Tuberculosis`), TorchScript-exported ViT, loaded with
`torch.jit.load` — no custom architecture code needed. Preprocessing (ported exactly from the repo's
`handler.py`): grayscale → CLAHE (clipLimit 2.0, tile 8×8) → Gaussian blur (5×5) → resize to 224×224
→ back to RGB → per-image z-score normalization. Output is a single sigmoid logit; ≥0.5 = TB
Positive. Severity (S1/S2/S3) is then derived from confidence using the same thresholding the
original Streamlit app used for its own (never-loaded) TB model.

### Maternal Health — `shr3m/fetal-brain-plane-cnn`
Classifies fetal-**brain** ultrasound images into one of 4 standard planes (Trans-thalamic,
Trans-cerebellum, Trans-ventricular, Other) — narrower than full obstetric "maternal health" triage,
but it's the closest public match found; the original project's own docs call this pathway "the
least developed" of the three. Preprocessing: grayscale, resize to 224×224, normalize with the
checkpoint's published mean/std (0.17076 / 0.17093). Explicitly licensed CC BY 4.0 and the model
card states it is for research/education, not clinical decision-making — same posture the rest of
this app already takes.

### Mammography — Roboflow `b-davmu/breastcancer-yolov8`
Not wired to real weights in this build: no Roboflow API key was supplied during setup, so this
modality currently falls back to the original app's own simulated BI-RADS scenario picker (clearly
labeled "SIMULATED" in the UI and reports). To enable it: put your key in a file named
`roboflow_key.txt` next to `GUI.py` (or set the `ROBOFLOW_API_KEY` environment variable) and restart
the app — `inference.py`'s `fetch_roboflow_mammography_weights()` will then try to pull a trained
export for that project automatically. Whether that succeeds depends on whether the project actually
has an exportable trained model version (vs. only an annotated dataset, which Roboflow Universe
projects sometimes only offer) — if not, the app keeps using the simulated fallback rather than
guessing.

### Considered and rejected: `Astaxanthin/KEEP` (suggested cancer model)
Investigated per a link shared mid-project. **Not used** — KEEP is a vision-language foundation model
for zero-shot cancer diagnosis on **histopathology** (H&E-stained biopsy slide) images, a different
imaging modality from digital **mammography** (radiographic breast X-ray). Plugging a histopathology
model into the mammography pathway would silently produce meaningless output on the wrong image
type, which conflicts with this project's own transparency principle — so it was left out rather than
force-fit.
