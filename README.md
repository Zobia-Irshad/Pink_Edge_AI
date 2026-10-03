# Pink Edge AI — Offline Desktop + Responsive Web Editions

Two sibling UIs over the same shared logic, ported from the original hackathon Streamlit demo (see
`App/Archive/Misc/`): a Tkinter **desktop** app (`GUI.py`) and a responsive **Streamlit web** app
(`streamlit_app.py`). Both share the same SQLite report cache and the same model backend
(`inference.py`). Everything except this README and the two launchers below lives in **`App/`** —
see `## Project layout`.

## What this is

Clinical-triage UI for three modalities — Mammography, Tuberculosis (chest X-ray), Maternal Health
(ultrasound) — matching the original app's dashboard, hospital-hub alert feed, and simulated
Alibaba-Cloud-sync panel. Each modality tries multiple methods in an order set by **measured
accuracy against real ground truth**, not by "online first":

| Modality | Try order | Measured accuracy |
|---|---|---|
| Mammography | Roboflow Workflow `breastcancer-yolov8-78tni` → offline pixel-diff heuristic → locally-trained classifier → local YOLO weights (rare) → Simulated | Roboflow: correctly flagged ground truth (90.9%). Offline heuristic: **~96%**. Locally-trained classifier: **98.7%** (held out, statistically tied with the heuristic) |
| Tuberculosis | **Locally-trained classifier** → offline pixel-diff heuristic → Roboflow model → offline HF ViT | Locally-trained classifier: **82.5%** (held out, best of the four). Offline heuristic: **74%** (Roboflow model has a ⚠️ known issue, see MODEL_SOURCES.md) |
| Maternal Health | Roboflow model `hash-maternal-health/1` → offline HF CNN | Roboflow: correctly flagged ground truth (88.3%). No locally-trained classifier yet — see MODEL_SOURCES.md |

The **offline pixel-diff heuristic** (`App/offline_cv.py`) needs no model weights and no internet,
ever — it builds "typical positive" / "typical negative" reference images by averaging your own
local labeled datasets (`App/Models/*/Data Set/`) and compares new images against both, drawing a
real bounding box around whatever region actually differs. Run `python offline_cv.py` (from `App/`)
to see its measured accuracy against held-out data for yourself. The **locally-trained classifier**
(`App/train_tb_classifier.py` → `App/Models/*/local_model.pt`) is a MobileNetV3-Small transfer-
learning model trained on this project's own datasets — also fully offline once trained. The
Roboflow calls need a key + internet (cloud-dependent at inference time — a deliberate tradeoff for
real predictions when available). Full detail (grounding, exact preprocessing, license, the TB
accuracy finding) is in [App/Documentations/MODEL_SOURCES.md](App/Documentations/MODEL_SOURCES.md).
This is a hackathon-grade demo, not a validated medical device — confidence numbers and severity
mappings are illustrative.

## Feature overview

Both editions (desktop + web) expose the same feature set, since both import their shared logic
straight from `GUI.py`:

- **Three-modality triage** — Mammography, Tuberculosis (chest X-ray), Maternal Health
  (ultrasound), each with a real bounding-box overlay drawn directly on the analyzed image when a
  model returns one.
- **Offline-first inference** — every modality tries at least one genuinely offline path (pixel-diff
  heuristic, locally-trained classifier, or Hugging Face model weights) as a real fallback, not just
  a placeholder, so triage keeps working with zero internet; the Roboflow-hosted primaries are used
  opportunistically when a key + connection are available.
- **Role-based access control** (`auth_manager.py`) — Lady Health Worker (LHW) vs. Senior
  Radiologist profiles, switchable in the sidebar or via PIN, gating who can override an AI
  assessment (BI-RADS / ACR density / TB severity).
- **DICOM anonymization & privacy hashing** (`dicom_anonymizer.py`) — every ingested scan is
  stripped of PII and assigned a hex privacy hash before anything is displayed, cached, or
  transmitted — aimed at HIPAA/GDPR-style compliance even on offline edge hardware.
- **Bilingual UI** — full English/Urdu toggle across both editions, including a voice-message
  library (English/Urdu/Punjabi text templates) with a pluggable offline-TTS hook
  (`play_voice_message()` — currently a stub; wire in Piper or gTTS to enable real audio).
- **SQLite report cache** — every "Save to Cache" writes to the shared local `pink_edge_cache.db`,
  from either edition, with text and PDF report generation.
- **Hospital Hub** — a live, color-coded (by real risk level) alert stream simulating GSM/SMS
  broadcast to a receiving hospital, with LHV agree/override/approve actions and a documented
  override-reason flow.
- **Cloud Sync tab** — simulated Alibaba Cloud IoT sync of unsynced cached reports, plus session
  feedback/analytics widgets (ratings, escalation counts, sync-rate metrics).
- **Hardware diagnostics panel** — simulated RK3588 NPU load/power/temperature readout, standing in
  for telemetry from a real edge-node deployment (see `App/Hardware/README.md`).

## Recent fix — Streamlit HTML rendering (multi-line `st.markdown` cards)

**Symptom:** cards like "Triage result", the AI Recommendation box, and the DICOM Metadata panel
rendered as raw `<div style="...">` text (with Streamlit's code-block copy icon) instead of styled
HTML, even though every call used `unsafe_allow_html=True`.

**Root cause:** Streamlit's Markdown renderer follows CommonMark, where any line indented 4+ spaces
is treated as a literal *indented code block* and shown as raw text — tags included — regardless of
`unsafe_allow_html`. Because these HTML strings are built as f-strings inside nested functions and
`if`/`with` blocks, every line naturally inherited 8+ spaces of Python indentation, silently
triggering this on ~20 different cards across the Dashboard, Hospital Hub, and Cloud Sync tabs.

**Fix (`streamlit_app.py`):** a helper, `html_block()`, strips per-line leading/trailing whitespace
from a multi-line HTML string. Rather than requiring every individual `st.markdown(...,
unsafe_allow_html=True)` call site to remember to wrap its string in `html_block()` — easy to miss,
and exactly how this bug happened in the first place — `st.markdown` itself is patched once, right
after `html_block()` is defined, so *any* HTML string passed with `unsafe_allow_html=True` is
auto-dedented before Streamlit ever sees it:

```python
_original_markdown = st.markdown

def _dedented_markdown(body, *args, **kwargs):
    if kwargs.get("unsafe_allow_html") and isinstance(body, str) and "\n" in body:
        body = html_block(body)
    return _original_markdown(body, *args, **kwargs)

st.markdown = _dedented_markdown
```

This covers every current card and any future one added the same way, with no per-call-site
changes needed elsewhere in `streamlit_app.py`.

## Run it

**Desktop (Tkinter):**
```
Start.bat
```
or manually: `cd App`, `pip install -r requirements.txt`, `python GUI.py`

**Web (Streamlit, responsive — resizes down to phone/tablet widths):**
```
Start_Web.bat
```
or manually: `cd App`, `pip install -r requirements.txt`, `streamlit run streamlit_app.py`
(opens `http://localhost:8501` in your browser; `--server.address 0.0.0.0` if you want it reachable
from another device on your LAN)

Both launchers `cd` into `App/` for you, then share `App/requirements.txt`. First run downloads
~1-2 GB of Python deps (PyTorch/Ultralytics) plus the offline fallback model weights (needs
internet once). The offline fallbacks then run without internet on every later run; the
Roboflow-hosted primaries need internet + a key every time (see above) — weights/datasets are
cached under `App/Models/`, and the report cache (`App/pink_edge_cache.db`, SQLite) is local and
shared by both editions.

Training or the standalone DICOM receiver need a few extra packages not required to just run the
app — see `App/requirements-dev.txt`.

## Enabling the Roboflow-hosted models (real, purpose-trained)

All three modalities check for a Roboflow API key and use it if present:

1. Get a key from [roboflow.com](https://roboflow.com) → your workspace → Settings → API.
2. Save it to a file named `roboflow_key.txt` inside `App/`, next to `GUI.py` (just the key,
   nothing else), set the `ROBOFLOW_API_KEY` environment variable, or (on Streamlit Community
   Cloud) add it under *App settings → Secrets*.
3. Restart the app (either edition). No key = each modality falls back down its own try-order (see
   the table above) — Mammography and Tuberculosis both still reach a real, non-simulated result
   with no key at all, via the offline heuristic / locally-trained classifier.

## Project layout

```
README.md, Start.bat, Start_Web.bat    — this file + the two one-click launchers (stay at repo root)

App/
  GUI.py                    — Tkinter desktop app: UI + local SQLite cache + reports + fallbacks
  streamlit_app.py           — Streamlit web app (responsive) — same logic, imported from GUI.py;
                                  includes the global st.markdown auto-dedent patch (see fix above)
  auth_manager.py             — RBAC module (LHW vs. Senior Radiologist), stdlib only
  dicom_anonymizer.py         — Offline DICOM anonymization & hex privacy hashing, stdlib only
  inference.py                — model loading + prediction dispatch for all three modalities
  offline_cv.py                — the offline pixel-diff heuristic (no model, no internet, ever);
                                    run directly (`python offline_cv.py`) to see its measured accuracy
  train_tb_classifier.py       — trains the locally-trained MobileNetV3 classifier (TB and
                                    Mammography both use it) against Models/<Modality>/Data Set/
  requirements.txt             — Python dependencies to run the app (shared by both editions)
  requirements-dev.txt         — extra dependencies for train_tb_classifier.py and Tools/receiver.py
  packages.txt                 — apt packages needed by .devcontainer (headless OpenCV libs)
  pink_edge_cache.db           — local report cache (SQLite; created on first "Save to Cache")
  roboflow_key.txt             — your Roboflow key, if you added one (gitignored)
  .streamlit/config.toml       — theme config (Streamlit Cloud reads this relative to the app entrypoint)

  Tests/
    test_auth.py               — unit tests for auth_manager.py (RBAC, PIN auth, permissions)
    test_anonymizer.py         — unit tests for dicom_anonymizer.py (PII hashing)

  Tools/
    receiver.py                 — standalone DICOM C-STORE listener (local PACS bridge); not
                                     imported by the app, needs requirements-dev.txt

  Archive/                     — superseded code, kept for reference only — see Archive/README.md
    pink_edge.py                 — a third, unused Streamlit UI fork; has a known latent RBAC bug
    fix_theme.py                 — spent one-shot codemod that already re-themed pink_edge.py
    Misc/
      Pink_Edge_AI-main/          — the original Alibaba-Cloud-Hackathon submission this whole
                                       project is built from (pink_edge.py, notebook, its own
                                       requirements.txt)
      Links.txt, *.pdf, WhatsApp Image *.jpeg  — development scratch

  Models/                  — downloaded weight cache + local datasets, one folder per modality
    TB/model.pt                          — sukhmani1303/tuberculosis-vit-model (TorchScript)
    TB/local_model.pt                    — locally-trained MobileNetV3 classifier (82.5% held-out)
    Maternal/FINAL-test-evaluation.pt    — shr3m/fetal-brain-plane-cnn
    Mammography/local_model.pt           — locally-trained MobileNetV3 classifier (98.7% held-out)
    */Data Set/                          — local labeled datasets offline_cv.py/train_tb_classifier.py use
    */positive/, */negative/, */validate/  — manually-added images, same convention as Data Set/
    */templates/                         — offline_cv.py's generated reference images (gitignored, auto-rebuilt)

  Validation/
    validate.py             — validation suite, run with `python Validation/validate.py` (from App/)

  Test Data/                — real sample images validate.py runs through the real models
    Tuberculosis/            — sample chest X-rays
    Breast Cancer/           — sample mammogram
    Maternal/                — sample ultrasound

  Documentations/           — reference docs
    MODEL_SOURCES.md         — exactly which model backs which modality, and why
    (+ the original project's own docs: API/BACKEND/FRONTEND/MODEL/PROJECT_ARCHITECTURE, USER_GUIDE
       — these describe the original hackathon submission, not this desktop/web rebuild; kept for
       historical reference)

  Hardware/                 — hardware plan for a real edge-node build (RK3588/Pi/ESP32/Mobile) —
                                see Hardware/README.md

  Assets/
    Changes/Changes.md       — what changed from the original Streamlit demo
    Icons/                    — icon set used by the UI
```

## Validating a change

```
python Validation/validate.py
```
(run from inside `App/` — or `python App/Validation/validate.py` from the repo root; paths inside
the script are anchored to `App/`, not the caller's CWD, so both work)

19 checks covering: module imports, placeholder image synthesis, detection overlay drawing, the
simulated scenario generators, a full SQLite cache round-trip (on a throwaway DB under `Validation/`
— never touches the real `pink_edge_cache.db`), text/PDF report generation, real-model inference for
all three modalities on both synthetic images *and* the real samples in `Test Data/`, ground-truth
cross-checks against each modality's real COCO-annotated dataset (including an accuracy floor for
the offline pixel-diff heuristic), the `run_triage()` dispatcher, and two full feature sweeps — every
modality, save-to-cache, report downloads, language toggle, network mode, cloud sync — for both the
**Tkinter UI** (hidden window, no mainloop) and the **Streamlit UI** (`streamlit.testing.v1.AppTest`,
no browser). Exits non-zero (and prints `inference.py`'s per-modality model status) if anything fails.

**Unit tests** (`App/Tests/`) cover `auth_manager.py` and `dicom_anonymizer.py` directly, without
downloading any model weights:
```
python Tests/test_auth.py
python Tests/test_anonymizer.py
```
(also run from inside `App/`)

## Deploy the web edition to Streamlit Community Cloud

1. **Push to GitHub** (once git is installed):
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
   *New app* → pick your repo/branch, set **Main file path** to **`App/streamlit_app.py`** (not
   `streamlit_app.py` — it's inside `App/`) → *Deploy*. Streamlit Cloud looks for `requirements.txt`
   next to the main file, so `App/requirements.txt` should be picked up automatically; if the
   deploy can't find requirements or the theme, check *Advanced settings* for a way to point at
   `App/requirements.txt` explicitly.
3. **Optional — real mammography model**: in the deploy dialog's *Advanced settings* (or later via
   *App settings → Secrets*), add:
   ```toml
   ROBOFLOW_API_KEY = "your-key-here"
   ```
   `inference.py` checks `st.secrets` for this automatically — no code changes needed.

**Resource caveat, honestly stated:** Streamlit Community Cloud's free tier gives each app ~1 CPU
core and ~1 GB RAM. This app's dependency stack (PyTorch, Ultralytics/OpenCV, several real model
checkpoints downloaded at first run) is heavier than a typical Streamlit demo — expect a slow first
boot (installing torch + downloading model weights) and keep an eye out for memory-related crashes
on that tier. If it struggles, the fixes in order of effort are: pin lighter dependency versions, or
deploy on a paid tier / your own server (`streamlit run streamlit_app.py --server.port 80
--server.address 0.0.0.0`, run from `App/`) instead.

## Relationship to the original project

`App/Archive/Misc/Pink_Edge_AI-main` is the original Streamlit/Alibaba-Cloud-Hackathon submission
both editions here are based on — same clinical vocabulary (BI-RADS, ACR density, TB severity/zone),
same SQLite schema, same report layout, same simulated cloud-sync panel (no real Alibaba
credentials are used here either). `streamlit_app.py` is a fresh, responsive rebuild (not the
original `pink_edge.py`) that reuses `GUI.py`'s shared logic and the real model backends instead of
the original's all-simulated scenario pickers. A separate, partially-rethemed fork of the original
(`App/Archive/pink_edge.py`) also exists but is not wired to any launcher — see `App/Archive/README.md`
for why it's archived rather than live. An Android build was discussed but deferred in favor of
these desktop/web builds.
