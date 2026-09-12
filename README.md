# Pink Edge AI — Offline Desktop + Responsive Web Editions

Two sibling UIs over the same shared logic, ported from the `Pink_Edge_AI-main` Streamlit hackathon
demo (see `Misc/`): a Tkinter **desktop** app (`GUI.py`) and a responsive **Streamlit web** app
(`streamlit_app.py`). Both are wired to real, publicly-sourced on-device models wherever one was
available, share the same SQLite report cache, and both run fully offline at inference time once
their model weights are downloaded (first run only) — the web edition is a local browser UI, not a
hosted/cloud service.

## What this is

Clinical-triage UI for three modalities — Mammography, Tuberculosis (chest X-ray), Maternal Health
(ultrasound) — matching the original app's dashboard, hospital-hub alert feed, and simulated
Alibaba-Cloud-sync panel, backed by:

| Modality | Backing |
|---|---|
| Tuberculosis | **Real model** — [sukhmani1303/tuberculosis-vit-model](https://huggingface.co/sukhmani1303/tuberculosis-vit-model) |
| Maternal Health | **Real model** — [shr3m/fetal-brain-plane-cnn](https://huggingface.co/shr3m/fetal-brain-plane-cnn) |
| Mammography | Simulated by default — real Roboflow model needs your API key, see below |

Full detail on every model (why it was picked, exact preprocessing, license) is in
[Documentations/MODEL_SOURCES.md](Documentations/MODEL_SOURCES.md). This is a hackathon-grade demo,
not a validated medical device — confidence numbers and severity mappings are illustrative, same
caveat the original project stated for itself.

## Run it

**Desktop (Tkinter):**
```
Start.bat
```
or manually: `pip install -r requirements.txt` then `python GUI.py`

**Web (Streamlit, responsive — resizes down to phone/tablet widths):**
```
Start_Web.bat
```
or manually: `pip install -r requirements.txt` then `streamlit run streamlit_app.py`
(opens `http://localhost:8501` in your browser; `--server.address 0.0.0.0` if you want it reachable
from another device on your LAN)

Both share `requirements.txt`. First run downloads ~1-2 GB of model weights + PyTorch/Ultralytics
(needs internet once). Every run after that is 100% offline for inference — weights are cached under
`Models/`, and the report cache (`pink_edge_cache.db`, SQLite) is local and shared by both editions.

## Enabling the real mammography model (optional)

The Mammography pathway falls back to the original app's own simulated BI-RADS scenario picker
unless you supply a Roboflow API key for the `b-davmu/breastcancer-yolov8` project:

1. Get a key from [roboflow.com](https://roboflow.com) → your workspace → Settings → API.
2. Save it to a file named `roboflow_key.txt` next to `GUI.py` (just the key, nothing else), or set
   the `ROBOFLOW_API_KEY` environment variable.
3. Restart the app (either edition). `inference.py` will try to pull a trained weight export for that
   project into `Models/Mammography/` automatically; if the project only has an annotated dataset
   with no trained export, it stays on the simulated fallback rather than guessing.

## Project layout

```
GUI.py                    — Tkinter desktop app: UI + local SQLite cache + reports + fallbacks
streamlit_app.py           — Streamlit web app (responsive) — same logic, imported from GUI.py
inference.py                — real model loading + prediction for all three modalities
requirements.txt            — Python dependencies (shared by both editions)
Start.bat / Start_Web.bat    — one-click installer + launcher, desktop / web
pink_edge_cache.db           — local report cache (SQLite; created on first "Save to Cache")

Models/                  — downloaded weight cache, one folder per modality
  TB/model.pt                          — sukhmani1303/tuberculosis-vit-model (TorchScript)
  Maternal/FINAL-test-evaluation.pt    — shr3m/fetal-brain-plane-cnn
  Mammography/                         — populated only if you add a Roboflow key (see above)

Validation/
  validate.py             — validation suite, run with `python Validation/validate.py`

Test Data/                — real sample images validate.py runs through the real models
  Tuberculosis/            — sample chest X-rays
  Breast Cancer/           — sample mammogram

Documentations/           — reference docs
  MODEL_SOURCES.md         — exactly which model backs which modality, and why
  (+ the original project's own docs: API/BACKEND/FRONTEND/MODEL/PROJECT_ARCHITECTURE, USER_GUIDE)

Assets/
  Changes/Changes.md       — what changed from the original Streamlit demo

Misc/                     — the original hackathon submission this was built from
  Pink_Edge_AI-main/       — original Streamlit app (pink_edge.py), notebook, requirements.txt
```

## Validating a change

```
python Validation/validate.py
```
14 checks covering: module imports, placeholder image synthesis, detection overlay drawing, the
simulated scenario generators, a full SQLite cache round-trip (on a throwaway DB under `Validation/`
— never touches the real `pink_edge_cache.db`), text/PDF report generation, real-model inference for
TB and Maternal Health (both on synthetic images *and* the real samples in `Test Data/`), the
mammography SIMULATED-fallback path, the `run_triage()` dispatcher for all 3 modalities, that the
Tkinter UI builds and can run one triage cycle end-to-end with no visible window, and that the
**Streamlit UI** builds and runs one triage cycle headlessly via `streamlit.testing.v1.AppTest` (no
browser needed). Exits non-zero (and prints `inference.py`'s per-modality model status) if anything
fails. Works from any working directory — paths are anchored to the repo root, not the caller's CWD.

## Relationship to the original project

`Misc/Pink_Edge_AI-main` is the original Streamlit/Alibaba-Cloud-Hackathon submission both editions
here are based on — same clinical vocabulary (BI-RADS, ACR density, TB severity/zone), same SQLite
schema, same report layout, same simulated cloud-sync panel (no real Alibaba credentials are used
here either). `streamlit_app.py` is a fresh, responsive rebuild (not the original `pink_edge.py`) that
reuses `GUI.py`'s shared logic and the real model backends instead of the original's all-simulated
scenario pickers. An Android build was discussed but deferred in favor of these desktop/web builds.
