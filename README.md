# Pink Edge AI — Offline Desktop + Responsive Web Editions

Two sibling UIs over the same shared logic, ported from the `Pink_Edge_AI-main` Streamlit hackathon
demo (see `Misc/`): a Tkinter **desktop** app (`GUI.py`) and a responsive **Streamlit web** app
(`streamlit_app.py`). Both share the same SQLite report cache and the same model backend
(`inference.py`).

## What this is

Clinical-triage UI for three modalities — Mammography, Tuberculosis (chest X-ray), Maternal Health
(ultrasound) — matching the original app's dashboard, hospital-hub alert feed, and simulated
Alibaba-Cloud-sync panel. Each modality tries multiple methods in an order set by **measured
accuracy against real ground truth**, not by "online first":

| Modality | Try order | Measured accuracy |
|---|---|---|
| Mammography | Roboflow Workflow `breastcancer-yolov8-78tni` → offline pixel-diff heuristic → Simulated | Roboflow: correctly flagged ground truth (90.9%). Offline heuristic: **96%** |
| Tuberculosis | **Offline pixel-diff heuristic** → Roboflow model → offline HF ViT | Offline heuristic: **74%** (best of the three — Roboflow model has a ⚠️ known issue, see MODEL_SOURCES.md) |
| Maternal Health | Roboflow model `hash-maternal-health/1` → offline HF CNN | Roboflow: correctly flagged ground truth (88.3%) |

The **offline pixel-diff heuristic** (`offline_cv.py`) needs no model weights and no internet,
ever — it builds "typical positive" / "typical negative" reference images by averaging your own
local labeled datasets (`Models/*/Data Set/`) and compares new images against both, drawing a real
bounding box around whatever region actually differs. Run `python offline_cv.py` to see its
measured accuracy against held-out data for yourself. The Roboflow calls need a key + internet
(cloud-dependent at inference time — a deliberate tradeoff for real predictions when available).
Full detail (grounding, exact preprocessing, license, the TB accuracy finding) is in
[Documentations/MODEL_SOURCES.md](Documentations/MODEL_SOURCES.md). This is a hackathon-grade demo,
not a validated medical device — confidence numbers and severity mappings are illustrative.

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

Both share `requirements.txt`. First run downloads ~1-2 GB of Python deps (PyTorch/Ultralytics) plus
the offline fallback model weights (needs internet once). The offline fallbacks then run without
internet on every later run; the Roboflow-hosted primaries need internet + a key every time (see
above) — weights/datasets are cached under `Models/`, and the report cache (`pink_edge_cache.db`,
SQLite) is local and shared by both editions.

## Enabling the Roboflow-hosted models (real, purpose-trained)

All three modalities check for a Roboflow API key and use it if present:

1. Get a key from [roboflow.com](https://roboflow.com) → your workspace → Settings → API.
2. Save it to a file named `roboflow_key.txt` next to `GUI.py` (just the key, nothing else), set the
   `ROBOFLOW_API_KEY` environment variable, or (on Streamlit Community Cloud) add it under
   *App settings → Secrets*.
3. Restart the app (either edition). No key = each modality falls back to its offline model
   (Mammography: simulated, since no local weights are bundled).

## Project layout

```
GUI.py                    — Tkinter desktop app: UI + local SQLite cache + reports + fallbacks
streamlit_app.py           — Streamlit web app (responsive) — same logic, imported from GUI.py
inference.py                — model loading + prediction dispatch for all three modalities
offline_cv.py                — the offline pixel-diff heuristic (no model, no internet, ever);
                                  run directly (`python offline_cv.py`) to see its measured accuracy
requirements.txt            — Python dependencies (shared by both editions)
Start.bat / Start_Web.bat    — one-click installer + launcher, desktop / web
pink_edge_cache.db           — local report cache (SQLite; created on first "Save to Cache")

Models/                  — downloaded weight cache + local datasets, one folder per modality
  TB/model.pt                          — sukhmani1303/tuberculosis-vit-model (TorchScript)
  Maternal/FINAL-test-evaluation.pt    — shr3m/fetal-brain-plane-cnn
  Mammography/                         — populated only if you add a Roboflow key (see above)
  */Data Set/                          — local labeled datasets offline_cv.py builds templates from
  */templates/                         — offline_cv.py's generated reference images (gitignored, auto-rebuilt)

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
19 checks covering: module imports, placeholder image synthesis, detection overlay drawing, the
simulated scenario generators, a full SQLite cache round-trip (on a throwaway DB under `Validation/`
— never touches the real `pink_edge_cache.db`), text/PDF report generation, real-model inference for
all three modalities on both synthetic images *and* the real samples in `Test Data/`, ground-truth
cross-checks against each modality's real COCO-annotated dataset (including an accuracy floor for
the offline pixel-diff heuristic), the `run_triage()` dispatcher, and two full feature sweeps — every
modality, save-to-cache, report downloads, language toggle, network mode, cloud sync — for both the
**Tkinter UI** (hidden window, no mainloop) and the **Streamlit UI** (`streamlit.testing.v1.AppTest`,
no browser). Exits non-zero (and prints `inference.py`'s per-modality model status) if anything
fails. Works from any working directory — paths are anchored to the repo root, not the caller's CWD.

## Deploy the web edition to Streamlit Community Cloud

The repo is already laid out the way [share.streamlit.io](https://share.streamlit.io) expects:
`streamlit_app.py` at the root, a top-level `requirements.txt`, a `.streamlit/config.toml` theme, and
a `.gitignore` that keeps downloaded model weights out of git (they're re-fetched from Hugging Face
automatically on first run instead — see `inference.py`).

1. **Push to GitHub** (once git is installed — see below):
   ```
   git init
   git add .
   git commit -m "Pink Edge AI desktop + Streamlit editions"
   ```
   Create an empty repo at github.com/new (no README/.gitignore/license — this repo already has
   them), then:
   ```
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git branch -M main
   git push -u origin main
   ```
2. **Deploy**: go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub →
   *New app* → pick your repo/branch, set **Main file path** to `streamlit_app.py` → *Deploy*.
3. **Optional — real mammography model**: in the deploy dialog's *Advanced settings* (or later via
   *App settings → Secrets*), add:
   ```toml
   ROBOFLOW_API_KEY = "your-key-here"
   ```
   `inference.py` checks `st.secrets` for this automatically — no code changes needed.

**Resource caveat, honestly stated:** Streamlit Community Cloud's free tier gives each app ~1 CPU
core and ~1 GB RAM. This app's dependency stack (PyTorch, Ultralytics/OpenCV, two real model
checkpoints downloaded at first run) is heavier than a typical Streamlit demo — expect a slow first
boot (installing torch + downloading ~80 MB of weights) and keep an eye out for memory-related
crashes on that tier. If it struggles, the fixes in order of effort are: pin lighter dependency
versions, or deploy on a paid tier / your own server (`streamlit run streamlit_app.py --server.port
80 --server.address 0.0.0.0`) instead.

## Relationship to the original project

`Misc/Pink_Edge_AI-main` is the original Streamlit/Alibaba-Cloud-Hackathon submission both editions
here are based on — same clinical vocabulary (BI-RADS, ACR density, TB severity/zone), same SQLite
schema, same report layout, same simulated cloud-sync panel (no real Alibaba credentials are used
here either). `streamlit_app.py` is a fresh, responsive rebuild (not the original `pink_edge.py`) that
reuses `GUI.py`'s shared logic and the real model backends instead of the original's all-simulated
scenario pickers. An Android build was discussed but deferred in favor of these desktop/web builds.
