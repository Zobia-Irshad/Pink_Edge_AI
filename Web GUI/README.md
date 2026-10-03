# Pink Edge AI — HTML/JS web edition

A **third** UI for this project, alongside the Tkinter desktop app (`App/GUI.py`) and the
Streamlit responsive app (`App/streamlit_app.py`). This one is plain HTML/CSS/vanilla JavaScript
— no framework, no build step, no Streamlit — talking to a small Flask backend
(`App/web_api.py`) over a REST API.

It is deliberately **separate** from the other two editions' UI code (its own files, its own
styling, no shared markup/CSS) but **connected to the same framework underneath**: every API
endpoint in `web_api.py` calls straight into `GUI.py` / `inference.py`, the exact functions the
Tkinter and Streamlit editions call. That means all three editions share:

- the same real/offline model fallbacks per modality (Roboflow → locally-trained classifier →
  offline pixel-diff heuristic → Hugging Face model → simulated, per `MODEL_SOURCES.md`),
- the same SQLite report cache (`App/pink_edge_cache.db`),
- the same privacy-hashing path (`dicom_anonymizer.py`) and RBAC rules (`auth_manager.py`).

## Run it

From the repo root:
```
Start_Web_GUI.bat
```
or manually:
```
cd App
pip install -r requirements.txt
python web_api.py
```
then open **http://127.0.0.1:5000** — `web_api.py` serves both the API (`/api/...`) and this
folder's static files (`index.html`, `style.css`, `app.js`) from that same address, so there's
nothing else to start and no CORS to fight with.

## Files

```
index.html   — page structure (sidebar: model/role/PIN/upload controls; Dashboard + Cloud Sync
                tabs, mirroring the layout of the other two editions at a smaller scope)
style.css    — styling. Color tokens are copied by hand from GUI.py's `C` dict (the single
                source of truth the desktop/Streamlit editions already share) — keep them in
                sync if that palette changes.
app.js       — all behavior: fetches config/model-status on load, drives Run Triage / Save to
                Cache / report downloads / the cache table, all via fetch() calls into
                App/web_api.py. No client-side inference or fabricated fields — every number
                shown (confidence, verdict, source, localization) comes straight back from the
                same run_triage() dispatcher the other editions use.
```

## What's intentionally smaller than the other two editions

This edition ships the **Dashboard** and **Cloud Sync/Cache** views. It does not (yet) port the
Streamlit edition's Hospital Hub (LHV agree/override workflow, GSM alert stream) or its Voice
Message Library / session-feedback panel — those are cosmetic/simulated panels in the Streamlit
edition already (see its own module docstring), not model logic, so porting them is pure UI work
whenever it's wanted next; `web_api.py` already exposes everything the real pipeline needs.
