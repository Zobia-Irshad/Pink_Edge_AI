---
title: Pink Edge AI
emoji: 🩺
colorFrom: pink
colorTo: red
sdk: streamlit
sdk_version: 1.30.0
app_file: streamlit_app.py
pinned: false
---

# Pink Edge AI — Offline Desktop + Responsive Web Editions

Three sibling UIs over the same shared logic, ported from the original hackathon Streamlit demo
(see `App/Archive/Misc/`): a Tkinter **desktop** app (`GUI.py`), a responsive **Streamlit web** app
(`streamlit_app.py`), and a plain **HTML/CSS/JS web** app (`Web GUI/`, backed by `App/web_api.py`).
All three share the same SQLite report cache and the same model backend (`inference.py`); the
HTML/JS edition is its own set of files (no Streamlit, no shared markup) talking to that backend
over a small REST API instead of importing `GUI.py`'s UI code directly. Everything the desktop/
Streamlit editions need lives in **`App/`**; two standalone, optional pieces live at the repo root
next to it instead — a local Android build (`Apk/`) and RK3588 edge-node deployment code
(`RK3588 SBC/`) — since neither is part of running the app itself. See `## Project layout`.

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

**Web (plain HTML/CSS/JS, a separate third edition):**
```
Start_Web_GUI.bat
```
or manually: `cd App`, `pip install -r requirements.txt`, `python web_api.py`, then open
`http://127.0.0.1:5000` (that one process serves both the API and the static files in `Web GUI/`).
See [Web GUI/README.md](Web%20GUI/README.md) for what this edition is and isn't yet.

All three launchers `cd` into `App/` for you, then share `App/requirements.txt`. First run downloads
~1-2 GB of Python deps (PyTorch/Ultralytics) plus the offline fallback model weights (needs
internet once). The offline fallbacks then run without internet on every later run; the
Roboflow-hosted primaries need internet + a key every time (see above) — weights/datasets are
cached under `App/Models/`, and the report cache (`App/pink_edge_cache.db`, SQLite) is local and
shared by both editions.

Training or the standalone DICOM receiver need a few extra packages not required to just run the
app — see the "Optional" section near the bottom of `App/requirements.txt`.

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
README.md, Start.bat, Start_Web.bat,    — this file + the three one-click launchers (stay at
  Start_Web_GUI.bat                        repo root)
index.html                 — GitHub Pages landing page: a static, dependency-free description of
                                the project (what it is, the three editions, the measured
                                accuracy table with its caveats, the data defects found and
                                fixed, the validation suites). Deliberately NOT a live demo —
                                the model backend can't run in a browser, so it claims no
                                inference. Its numbers mirror Documentations/MODEL_SOURCES.md;
                                update that file first, then this page.
Apk/                       — optional: local Android build (Kivy + Buildozer), see Apk/README.md
RK3588 SBC/                — optional: RK3588 edge-node deployment code, see its own README.md
Web GUI/                   — the third edition's front end: static index.html/style.css/app.js,
                                talking to App/web_api.py over REST; see Web GUI/README.md

App/
  GUI.py                    — Tkinter desktop app: UI + local SQLite cache + reports + fallbacks
  streamlit_app.py           — Streamlit web app (responsive) — same logic, imported from GUI.py;
                                  includes the global st.markdown auto-dedent patch (see fix above)
  web_api.py                  — Flask backend for the HTML/JS edition (Web GUI/ at repo root) — a
                                  thin REST wrapper around GUI.py/inference.py, no UI code of its
                                  own; also serves Web GUI/'s static files at the same address
  auth_manager.py             — RBAC module (LHW vs. Senior Radiologist), stdlib only
  dicom_anonymizer.py         — Offline DICOM anonymization & hex privacy hashing, stdlib only
  inference.py                — model loading + prediction dispatch for all three modalities
  offline_cv.py                — the offline pixel-diff heuristic (no model, no internet, ever);
                                    run directly (`python offline_cv.py`) to see its measured accuracy
  train_tb_classifier.py       — trains the locally-trained MobileNetV3 classifier for TB
  train_local_model.py         — the general, modality-parameterized version of the same training
                                    approach (TB + Mammography both ship local_model.pt trained this
                                    way); run `python train_local_model.py [modality ...]`
  requirements.txt             — Python dependencies (core app + an "Optional" section at the
                                    bottom for train_tb_classifier.py and Tools/receiver.py)
  packages.txt                 — apt packages needed by .devcontainer (headless OpenCV libs)
  pink_edge_cache.db           — local report cache (SQLite; created on first "Save to Cache")
  roboflow_key.txt             — your Roboflow key, if you added one (gitignored)
  .streamlit/config.toml       — theme config (Streamlit Cloud reads this relative to the app entrypoint)

  Tests/
    smoke_test.py              — fast end-to-end check (~30s, no model downloads): starts the real
                                    Streamlit app, drives role/language/modality switching, Run
                                    Triage and Save to Cache, then verifies theme consistency
                                    across GUI.py / streamlit_app.py / .streamlit/config.toml
    fabrication_audit.py       — which displayed fields are measured vs. generated (same image in
                                    3x, diff the results; flags generated fields in a result that
                                    claims real inference)
    privacy_leak_test.py       — PII handling at every sink: GSM/IoT broadcast, SQLite cache,
                                    generated reports, API-key hygiene
    test_auth.py               — unit tests for auth_manager.py (RBAC, PIN auth, permissions)
    test_anonymizer.py         — unit tests for dicom_anonymizer.py (PII hashing)

  Evidence/
    generate_evidence.py       — runs all seven suites, writes a dated evidence document with the
                                    SHA-256 of every attested file
    EVIDENCE_REPORT.md         — the generated report (regenerate rather than hand-edit)

  Tools/
    receiver.py                 — standalone DICOM C-STORE listener (local PACS bridge); not
                                     imported by the app, needs the "Optional" deps in requirements.txt

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
    */positive/, */negative/, */validate/  — manually-added images; see Models/README.md for the convention
    */templates/                         — offline_cv.py's generated reference images (gitignored, auto-rebuilt)

  Validation/
    validate.py             — validation suite, run with `python Validation/validate.py` (from App/)
    validate_offline.py     — the same pipeline with the network forcibly removed (proves the
                                 offline claim; a control check first proves the block is real)

  Test Data/                — real sample images validate.py runs through the real models
    Tuberculosis/            — sample chest X-rays
    Breast Cancer/           — sample mammogram
    Maternal/                — sample ultrasound

  Documentations/           — reference docs
    MODEL_SOURCES.md         — exactly which model backs which modality, and why
    PRIVACY_POLICY.md        — what data is handled, where it goes, and what is NOT protected
    USER_GUIDE.md            — end-user walkthrough of the app
    TECHNICAL_REFERENCE.md   — the original hackathon submission's own API/Backend/Frontend/Model/
                                 Architecture docs, combined into one file — describes that original
                                 submission, not this desktop/web rebuild; kept for historical reference

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

**Smoke test** — the fast one to reach for first (~30s, no model downloads). Starts the real
Streamlit app via `AppTest`, drives role switching / language toggle / all three modalities / Run
Triage / Save to Cache, then checks the theme hasn't drifted apart across the places it's defined
(`GUI.py`'s `C`, `streamlit_app.py`, `.streamlit/config.toml`) and that every palette colour used
as text still clears WCAG AA contrast:
```
python Tests/smoke_test.py        # -v to watch each check run
```

**Offline validation** — proves the "works with no internet" claim by actually removing the
network (every outbound socket blocked, `HF_HUB_OFFLINE=1`) rather than by inspecting code. A
control check runs first and tries three escape routes, failing the suite if any succeeds, so a
pass can't be an artefact of accidental connectivity. 13 checks: all three modalities still
produce results, none claims a hosted source, each offline tier works, cache/reports/both UIs
function:
```
python Validation/validate_offline.py
```

**Data fabrication audit** — submits the identical image three times per modality and diffs the
results. A field derived from the image is identical every time; one that changes was generated,
not measured. Flags any such field inside a result whose source claims real inference:
```
python Tests/fabrication_audit.py
```

**Privacy / PII leak validation** — checks the README's own promise ("stripped of PII and assigned
a hex privacy hash before anything is displayed, cached, or transmitted") at every sink: the
GSM/IoT broadcast, the SQLite cache, generated reports, and credential handling. See
[App/Documentations/PRIVACY_POLICY.md](App/Documentations/PRIVACY_POLICY.md):
```
python Tests/privacy_leak_test.py
```

**Unit tests** (`App/Tests/`) cover `auth_manager.py` and `dicom_anonymizer.py` directly, without
downloading any model weights:
```
python Tests/test_auth.py
python Tests/test_anonymizer.py
```

**Evidence report** — runs all seven suites, records their actual exit codes and output, and
writes a dated document with the SHA-256 of every attested source file, so it can't silently
outlive the code it describes:
```
python Evidence/generate_evidence.py        # --quick skips the two slow model suites
```
Output: [App/Evidence/EVIDENCE_REPORT.md](App/Evidence/EVIDENCE_REPORT.md)

(all of the above run from inside `App/`)

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

## Mobile & edge deployment (optional, standalone)

Two pieces at the repo root, neither needed to run the desktop/web app above:

- **`Apk/`** — a local Android build (Kivy + Buildozer, not a port of `GUI.py`/`streamlit_app.py` —
  see `Apk/README.md` for why torch/opencv don't cross-compile for Android and what runs instead).
- **`RK3588 SBC/`** — deployment code for running this app as a real edge node on an RK3588 board
  (kiosk autostart, the UART bridge to the ESP32/SIM800L GSM companion from `App/Hardware/HARDWARE_PLAN.md`'s Wiring section,
  and NPU-accelerated inference via `rknn-toolkit-lite2`) — see `RK3588 SBC/README.md`.

Neither has been built/run against real hardware in this environment (no Android SDK/NDK, no RK3588
board) — both READMEs say so plainly; treat them as a verified-on-real-hardware starting point.

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

## AI Coding Agent Guidance

For AI coding assistants (Antigravity, Cursor, Claude, Copilot) working on this repository, comprehensive architectural context, multi-tier inference fallback hierarchies, RBAC constraints, and engineering guidelines are documented in [AGENTS.md](AGENTS.md).
