# Pink Edge AI — Local Android APK (Kivy + Buildozer)

A separate, lean Android build of the triage UI, living at the repo root next to `App/` (its own
build tree — `buildozer`/Android SDK output don't belong nested inside the desktop/web app's
folder). It is **not** a port of `App/GUI.py`/`App/streamlit_app.py` (those depend on `torch`,
`transformers`, and `ultralytics`, none of which realistically cross-compile for Android through
`python-for-android` — no reliable ARM recipes, multi-GB wheel sizes, and long build times that
don't belong on a phone). Instead this is a from-scratch **Kivy** app that keeps the same three
modalities and the same "measured accuracy, not assumption" precedence, using only dependencies
that actually build for Android:

| Tier | What it does | Needs internet? |
|---|---|---|
| 1. Roboflow REST API | Same hosted models as `../App/inference.py` (`breastcancer-yolov8-78tni`, `tuberculosis-tp2pv/1`, `hash-maternal-health/1`), called directly over HTTPS with `requests` (no `inference-sdk` — too heavy for mobile) | Yes + API key |
| 2. Offline pixel-diff heuristic | Same algorithm as `../App/offline_cv.py` (grayscale, histogram-equalize, orientation search, template diff, bounding box), reimplemented with pure `numpy`/`Pillow` (no OpenCV — no reliable Android recipe either) against **pre-built template images bundled into the APK** (see `generate_assets.py`) instead of the full dataset | No, ever |

TB keeps the same precedence as the desktop app (offline heuristic first — the Roboflow TB model has
a documented false-positive issue, see `../App/Documentations/MODEL_SOURCES.md`); Mammography and
Maternal try Roboflow first, same as desktop.

## Why this can't just reuse `App/inference.py`

- `torch`/`transformers` (the offline HF fallback tier) are dropped entirely — no ViT/CNN weights ship
  in the APK. If Roboflow and the offline heuristic both come back empty (no template dataset bundled
  for that modality), the app says so plainly instead of pretending.
- `opencv-python` has no dependable `python-for-android` recipe as of this writing; `_analyze()`'s
  `cv2.equalizeHist`/`cv2.findContours` calls are reimplemented below with `PIL.ImageOps.equalize` and
  a plain numpy min/max-over-threshold bounding box (slightly less precise for multi-blob images, fine
  for a single lesion/finding region).
- The full COCO-annotated datasets under `../App/Models/*/Data Set/` (used to *build* the templates)
  are megabytes of images — too much to ship in an app. `generate_assets.py` builds the templates
  once on a desktop (reusing `../App/offline_cv.py`'s own `_get_templates()`) and exports just the
  two small 256x256 PNGs per modality into `assets/`; those are what actually ship in the APK.

## Prerequisites (Buildozer needs Linux)

Buildozer does not run on native Windows. Use **WSL2** (Ubuntu) or a Linux VM/container:

```bash
# inside WSL2/Ubuntu
sudo apt update && sudo apt install -y python3-pip build-essential git zip unzip openjdk-17-jdk \
    autoconf libtool pkg-config zlib1g-dev libncurses5-dev cmake libffi-dev libssl-dev
pip3 install --user buildozer cython
```

Buildozer downloads the Android SDK/NDK itself on first build (several GB, needs internet once).

## Build steps

```bash
cd Apk
python3 generate_assets.py          # run once, from a machine with ../App/Models/*/Data Set/ present
buildozer android debug             # first run: ~20-40 min, downloads SDK/NDK + all Android wheels
```

Output APK: `bin/pinkedgeai-*-debug.apk`. Install with `adb install -r bin/pinkedgeai-*-debug.apk`, or
copy it to the phone and tap it (enable "Install unknown apps" for the file manager/browser you use).

## Setting the Roboflow key on-device

The app looks for a key in this order: an `ROBOFLOW_API_KEY` environment variable (not normally
settable on Android without a launcher trick — mostly useful for `buildozer android debug deploy run`
testing), then a `roboflow_key.txt` file next to the APK's private storage (the in-app "Settings"
screen writes this — enter the key once, it persists across runs), matching the same "no key = offline
only" fallback the desktop app uses.

## Files here

```
buildozer.spec        — Kivy/Android build config: permissions (INTERNET, CAMERA, storage), the
                         requirements list (kivy, requests, pillow, numpy, plyer — all have real
                         python-for-android recipes), package name/version
main.py                — Kivy UI: modality picker, image picker (camera or gallery via plyer),
                         run-triage button, verdict + bbox overlay, save report, EN/UR toggle
mobile_inference.py     — the two-tier dispatch (Roboflow REST -> offline heuristic), no torch/cv2
generate_assets.py      — desktop-only helper: exports ../App/offline_cv.py's templates as small
                         PNGs into assets/<modality>/{positive,negative}.png for the APK to bundle
assets/                 — template PNGs (generated, not hand-authored) + app icon/presplash
```

## Honesty note

This has not been built into a real APK in this environment (no Android SDK/NDK installed here, and
Buildozer requires Linux). The code is written to run correctly under Kivy + python-for-android's
actual constraints (dependencies chosen specifically because they have working recipes), but "should
build" is not the same as "verified building" — budget time for the first `buildozer android debug`
run to surface a missing recipe or version pin, same as any first Android build of a new Kivy app.
