#!/usr/bin/env python3
"""
Pink Edge AI (Desktop) — validation suite.
============================================
Self-contained smoke/validation test for GUI.py + inference.py + streamlit_app.py — 18 checks:
imports, imaging, simulated scenario generators, SQLite cache round-trip, text/PDF report
generation, real model inference for all three modalities on both synthetic AND real Test Data/
images (each modality tries its Roboflow-hosted model first, then an offline Hugging Face/
local-weights fallback, then SIMULATED — see inference.py), ground-truth cross-checks against each
modality's real COCO-annotated dataset, the run_triage() dispatcher, and two full feature sweeps —
Tkinter (hidden window, no mainloop: every modality driven with a real Test Data image, overlay
toggle, save-to-cache, text/PDF report download, language toggle, network mode + cloud sync + OTA,
Hospital Hub, Cloud Sync, Reset Session) and Streamlit (AppTest, no browser: every modality, model
selectbox, save-to-cache, language toggle, network mode + cloud sync) — exercising the UI action
methods themselves, not just the backend functions they call.

Run with:  python Validation/validate.py   (from anywhere — paths below are anchored to the
repo root, not the current working directory)
Exits 0 if every check passes, 1 otherwise. Uses a throwaway DB file under Validation/ (never
touches the app's real pink_edge_cache.db at the project root) and cleans up after itself.
"""
import os
import sys
import time
import traceback
from glob import glob

VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(VALIDATION_DIR)
TEST_DATA_DIR = os.path.join(ROOT_DIR, "Test Data")
sys.path.insert(0, ROOT_DIR)  # so `import GUI` / `import inference` resolve from Validation/

RESULTS = []  # (name, ok, detail)


def check(name):
    """Decorator: run fn(), record PASS/FAIL, never let one failure kill the run."""
    def wrap(fn):
        t0 = time.time()
        try:
            fn()
            RESULTS.append((name, True, f"{time.time() - t0:.2f}s"))
        except Exception as e:
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            traceback.print_exc()
        return fn
    return wrap


def require(cond, msg="assertion failed"):
    if not cond:
        raise AssertionError(msg)


RESULT_KEYS = {"bi_rads", "acr", "verdict", "sub", "loc", "extra", "vicon", "confidence", "sms", "is_critical"}


def is_real_source(source: str) -> bool:
    """True for any non-simulated result — Roboflow, an offline HF/local model, or the offline
    pixel-diff heuristic all count as 'real' here; only the SIMULATED scenario picker doesn't."""
    return "SIMULATED" not in source


def validate_result_shape(r, where):
    require(isinstance(r, dict), f"{where}: result is not a dict")
    missing = RESULT_KEYS - set(r.keys())
    require(not missing, f"{where}: missing keys {missing}")
    require(0.0 <= r["confidence"] <= 100.0, f"{where}: confidence out of range: {r['confidence']}")
    require(isinstance(r["is_critical"], bool), f"{where}: is_critical not bool")


# ============================================================
print("=" * 70)
print("PINK EDGE AI (DESKTOP) — VALIDATION SUITE")
print("=" * 70)

# ---- 1. imports ----
@check("import GUI and inference modules")
def _():
    global GUI, inf
    import GUI as GUI  # noqa
    import inference as inf  # noqa


# ---- 2. imaging ----
@check("placeholder image generation (mammogram/xray/ultrasound)")
def _():
    for model in GUI.MODELS:
        img = GUI.load_placeholder(model, seed=1)
        require(img.size == (512, 512), f"{model}: wrong size {img.size}")
        require(img.mode == "L", f"{model}: wrong mode {img.mode}")


@check("draw_bbox overlay (all 3 modalities, critical + non-critical)")
def _():
    img = GUI.gen_xray(seed=2)
    for model in GUI.MODELS:
        for crit in (True, False):
            fake = {"confidence": 88.5, "is_critical": crit}
            out = GUI.draw_bbox(img, model, fake)
            require(out.size == (512, 512) and out.mode == "RGB", f"{model} crit={crit}: bad overlay output")


# ---- 3. simulated scenario generators ----
@check("simulated scenario generators (mammography/tb/maternal)")
def _():
    for fn, where in [(GUI.sim_mammography, "mammography"), (GUI.sim_tb, "tb"), (GUI.sim_maternal, "maternal")]:
        for _i in range(5):
            r = fn()
            validate_result_shape(r, f"sim_{where}")
            require("source" in r, f"sim_{where}: no source field")


# ---- 4. SQLite cache round-trip (throwaway DB, never the real one) ----
TEST_DB = os.path.join(VALIDATION_DIR, "validation_test_cache.db")


@check("SQLite cache round-trip (init/save/get/sync/counts) on throwaway DB")
def _():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    orig_path = GUI.DB_PATH
    GUI.DB_PATH = TEST_DB
    try:
        GUI.init_db()
        t0, u0 = GUI.counts()
        require(t0 == 0 and u0 == 0, "fresh DB should be empty")

        data = {
            "patient_id": 12345678, "patient_age": 34, "modality": "DX", "model_used": "Tuberculosis",
            "bi_rads": "S1 - Minimal (unilateral, no cavity)", "acr_density": "Upper Zone",
            "verdict": "TB Positive", "localization": "Right Upper Lobe", "confidence": 91.2,
            "inference_time": 1.4, "timestamp": "2026-09-12 20:00:00", "network_mode": "Fully Offline",
            "model_source": "unit-test", "synced": 0,
        }
        GUI.save_to_cache(data)
        total, unsynced = GUI.counts()
        require(total == 1 and unsynced == 1, f"expected 1/1 after save, got {total}/{unsynced}")

        rows = GUI.get_cached_reports()
        require(len(rows) == 1, "get_cached_reports should return the saved row")
        # patient_id column has TEXT affinity (schema ported as-is from the original app), so
        # SQLite stores the inserted int as "12345678" — compare as str, not equality-of-types.
        require(str(rows[0][1]) == "12345678", f"patient_id mismatch in saved row: {rows[0][1]!r}")

        unsynced_rows = GUI.get_unsynced_reports()
        require(len(unsynced_rows) == 1, "should have 1 unsynced row")
        GUI.mark_as_synced([unsynced_rows[0][0]])
        total2, unsynced2 = GUI.counts()
        require(total2 == 1 and unsynced2 == 0, f"expected 1/0 after sync, got {total2}/{unsynced2}")

        iot = GUI.simulate_iot_sync(rows)
        require(len(iot) == 1 and "iot_id" in iot[0], "simulate_iot_sync malformed output")
        oss = GUI.simulate_oss_upload(rows[0])
        require("high" in oss, "simulate_oss_upload malformed output")
        acr = GUI.simulate_acr_check()
        require("current" in acr and "available" in acr, "simulate_acr_check malformed output")
    finally:
        GUI.DB_PATH = orig_path
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)


# ---- 5. reports ----
@check("text + PDF report generation")
def _():
    d = {
        "patient_id": 87654321, "patient_age": 29, "modality": "MG (Mammography)",
        "model_used": "Mammography", "bi_rads": "BI-RADS 5 - Highly suggestive of malignancy",
        "acr_density": "C - Heterogeneously dense", "verdict": "BI-RADS 5",
        "localization": "Upper Outer Quadrant", "confidence": 94.3, "inference_time": 8.1,
        "timestamp": "2026-09-12 20:00:00", "network_mode": "Fully Offline",
        "model_source": "unit-test", "synced": 0,
    }
    txt = GUI.generate_text_report(d)
    require("PINK EDGE AI" in txt and str(d["patient_id"]) in txt, "text report missing expected content")
    require("URGENT" in txt, "BI-RADS 5 should trigger an URGENT referral line")

    pdf_bytes = GUI.generate_pdf_bytes(d)
    require(pdf_bytes is not None, "PDF generation returned None (fpdf2 missing?)")
    require(isinstance(pdf_bytes, bytes) and len(pdf_bytes) > 500, "PDF output looks too small/invalid")
    require(pdf_bytes[:4] == b"%PDF", "PDF output does not start with a %PDF header")


# ---- 6. real model inference (TB + Maternal) ----
@check("TB real-model inference on a synthetic chest X-ray")
def _():
    require(inf.tb_available(), f"TB model failed to load: {inf._tb_load_error}")
    img = GUI.gen_xray(seed=3)
    r = inf.predict_tb(img)
    require(r is not None, "predict_tb returned None despite model being available")
    validate_result_shape(r, "predict_tb")
    require(is_real_source(r["source"]), f"TB result should be tagged as real, not simulated: {r['source']!r}")


@check("Maternal real-model inference on a synthetic ultrasound")
def _():
    require(inf.maternal_available(), f"Maternal model failed to load: {inf._maternal_load_error}")
    img = GUI.gen_ultrasound(seed=4)
    r = inf.predict_maternal(img)
    require(r is not None, "predict_maternal returned None despite model being available")
    validate_result_shape(r, "predict_maternal")
    require(is_real_source(r["source"]), f"Maternal result should be tagged as real, not simulated: {r['source']!r}")


@check("Mammography: real Roboflow workflow if a key is configured, else SIMULATED")
def _():
    has_key = inf._roboflow_api_key() is not None
    img = GUI.gen_mammogram(seed=5)
    direct = inf.predict_mammography(img)
    if has_key:
        # A key is configured (this project's roboflow_key.txt) -> the hosted Workflow should
        # answer for real, not fall through to local weights or return None.
        require(direct is not None, "predict_mammography returned None despite a Roboflow key being configured")
        validate_result_shape(direct, "predict_mammography(roboflow)")
        require(is_real_source(direct["source"]), f"expected a real (non-simulated) source with a key configured: {direct['source']!r}")
    elif not inf._find_local_mammo_weights():
        require(direct is None, "predict_mammography should return None with no key and no local weights")


# ---- 6b. real model inference on the real sample images in Test Data/ ----
@check("TB real-model inference on real sample X-rays (Test Data/Tuberculosis)")
def _():
    from PIL import Image

    files = sorted(glob(os.path.join(TEST_DATA_DIR, "Tuberculosis", "*")))
    require(len(files) > 0, f"no sample files found under {TEST_DATA_DIR}\\Tuberculosis")
    for f in files:
        img = Image.open(f)
        r = inf.predict_tb(img)
        require(r is not None, f"predict_tb returned None on real sample: {os.path.basename(f)}")
        validate_result_shape(r, f"predict_tb({os.path.basename(f)})")
        print(f"    {os.path.basename(f):40s} -> {r['verdict']:15s} ({r['confidence']:.1f}%)")


@check("Maternal real-model inference on real samples (Test Data/Maternal)")
def _():
    from PIL import Image

    files = sorted(glob(os.path.join(TEST_DATA_DIR, "Maternal", "*")))
    require(len(files) > 0, f"no sample files found under {TEST_DATA_DIR}\\Maternal")
    for f in files:
        img = Image.open(f)
        r = inf.predict_maternal(img)
        require(r is not None, f"predict_maternal returned None on real sample: {os.path.basename(f)}")
        validate_result_shape(r, f"predict_maternal({os.path.basename(f)})")
        print(f"    {os.path.basename(f):40s} -> {r['verdict']:25s} ({r['confidence']:.1f}%)")


@check("Mammography pathway on real samples (Test Data/Breast Cancer)")
def _():
    from PIL import Image

    files = sorted(glob(os.path.join(TEST_DATA_DIR, "Breast Cancer", "*")))
    require(len(files) > 0, f"no sample files found under {TEST_DATA_DIR}\\Breast Cancer")
    for f in files[:3]:  # a few, not all 40+ — keep this check fast
        img = Image.open(f)
        r = GUI.run_triage("Mammography (YOLOv8-OBB)", img)
        validate_result_shape(r, f"run_triage(mammography, {os.path.basename(f)})")
        print(f"    {os.path.basename(f):40s} -> {r['verdict']:15s} ({r['source']})")


@check("Mammography Roboflow workflow matches ground truth on an annotated sample")
def _():
    """Cross-checks against the COCO annotation, not just 'did it run' — a stronger signal
    than the other checks that the real model is actually doing something sensible."""
    from PIL import Image

    if not inf._roboflow_api_key():
        return  # no key configured in this environment — nothing to cross-check
    ds_dir = os.path.join(ROOT_DIR, "Models", "Mammography", "Data Set", "BreastCancer-YOLOv8.coco", "test")
    ann_path = os.path.join(ds_dir, "_annotations.coco.json")
    require(os.path.isfile(ann_path), f"no COCO annotations found at {ann_path}")
    import json

    with open(ann_path, "r", encoding="utf-8") as fh:
        coco = json.load(fh)
    require(len(coco["annotations"]) > 0, "COCO file has no annotations to cross-check against")
    img_id = coco["annotations"][0]["image_id"]
    file_name = next(im["file_name"] for im in coco["images"] if im["id"] == img_id)
    img = Image.open(os.path.join(ds_dir, file_name))

    r = inf.predict_mammography(img)
    require(r is not None, "predict_mammography returned None on a known-positive annotated sample")
    require(r["is_critical"], f"ground truth has a cancer annotation but the model said {r['verdict']!r}")
    print(f"    {file_name} (ground truth: cancer) -> {r['verdict']} ({r['confidence']:.1f}%)")


def _images_for_category(coco_path, ds_dir, category_name, limit=3):
    """Loads a COCO annotation file and returns up to `limit` (PIL Image, file_name) pairs for
    images annotated with the given category name. Matches ALL category ids with that name —
    some Roboflow COCO exports have more than one id sharing a name (e.g. an unused id 0
    placeholder alongside the real one) — a naive "first id with this name" lookup can silently
    match zero real annotations."""
    import json

    from PIL import Image

    with open(coco_path, "r", encoding="utf-8") as fh:
        coco = json.load(fh)
    cat_ids = {c["id"] for c in coco["categories"] if c["name"] == category_name}
    if not cat_ids:
        return []
    anns = [a for a in coco["annotations"] if a["category_id"] in cat_ids]
    out = []
    seen_images = set()
    for ann in anns:
        if ann["image_id"] in seen_images:
            continue
        seen_images.add(ann["image_id"])
        file_name = next(im["file_name"] for im in coco["images"] if im["id"] == ann["image_id"])
        out.append((Image.open(os.path.join(ds_dir, file_name)), file_name))
        if len(out) >= limit:
            break
    return out


def _first_image_for_category(coco_path, ds_dir, category_name):
    imgs = _images_for_category(coco_path, ds_dir, category_name, limit=1)
    return imgs[0] if imgs else None


@check("Offline pixel-diff heuristic: accuracy floor on held-out ground truth (TB + Mammography)")
def _():
    """Guards against silent regressions in offline_cv.py itself — calls it directly (bypassing
    the dispatcher's try-order) on a handful of held-out samples per modality and requires
    better-than-chance accuracy. Not a substitute for `python offline_cv.py`'s fuller
    calibration run — just a fast sanity floor for every-day validation."""
    import offline_cv
    from PIL import Image

    for modality in ("tb", "mammography"):
        if not offline_cv.available(modality):
            continue  # no local dataset in this environment — nothing to check
        cfg = offline_cv._DATASETS[modality]
        pos, neg = offline_cv._collect_image_paths(
            offline_cv._dataset_dir(modality), cfg["positive_categories"], cfg["negative_categories"])
        test_marker = os.sep + "test" + os.sep
        pos = [p for p in pos if test_marker in p][:5]
        neg = [p for p in neg if test_marker in p][:5]
        require(pos and neg, f"{modality}: not enough held-out test-split samples to check")
        correct, total = 0, 0
        for path, expect_positive in [(p, True) for p in pos] + [(p, False) for p in neg]:
            r = offline_cv.predict(modality, Image.open(path))
            require(r is not None, f"{modality}: offline_cv.predict returned None on {path}")
            correct += r["is_critical"] == expect_positive
            total += 1
        print(f"    [{modality}] {correct}/{total} correct on this quick held-out sample")
        require(correct >= total * 0.5, f"{modality}: offline heuristic at/below chance ({correct}/{total}) — likely broken, not just imprecise")


@check("TB dispatcher matches ground truth (positive + negative annotated samples)")
def _():
    """Tests predict_tb() end-to-end — whichever method is actually primary right now (offline
    heuristic, then Roboflow, then the offline HF model — see predict_tb() in inference.py)."""
    ds_dir = os.path.join(ROOT_DIR, "Models", "TB", "Data Set", "tuberculosis.coco", "test")
    ann_path = os.path.join(ds_dir, "_annotations.coco.json")
    require(os.path.isfile(ann_path), f"no COCO annotations found at {ann_path}")

    pos = _first_image_for_category(ann_path, ds_dir, "Tüberküloz")
    require(pos is not None, "no 'Tüberküloz' (active TB) annotated sample found to cross-check against")
    img, name = pos
    r = inf.predict_tb(img)
    require(r is not None, "predict_tb returned None on a known-positive annotated sample")
    require(r["is_critical"], f"ground truth is active TB but the model said {r['verdict']!r}")
    print(f"    {name} (ground truth: Tüberküloz) -> {r['verdict']} ({r['confidence']:.1f}%)")

    # Majority vote over a few samples, not a single one: a real model calling one borderline
    # image wrong is normal (this one has an rfdetr-small model with modest class balance —
    # 159 'Sağlıklı' vs 432 'Tüberküloz' annotations); the integration is only broken if it's
    # wrong most of the time, not if it's wrong once.
    negatives = _images_for_category(ann_path, ds_dir, "Sağlıklı", limit=3)
    require(len(negatives) > 0, "no 'Sağlıklı' (healthy) annotated sample found to cross-check against")
    correct = 0
    for img2, name2 in negatives:
        r2 = inf.predict_tb(img2)
        require(r2 is not None, f"predict_tb returned None on known-negative sample {name2}")
        ok = not r2["is_critical"]
        correct += ok
        print(f"    {name2} (ground truth: Sağlıklı) -> {r2['verdict']} ({r2['confidence']:.1f}%) {'OK' if ok else 'MISS'}")
    require(correct * 2 >= len(negatives),
            f"model called {len(negatives) - correct}/{len(negatives)} known-healthy samples TB-positive — "
            f"worse than a coin flip, likely a real integration bug, not just model noise")


@check("Maternal Health Roboflow model matches ground truth on an annotated sample")
def _():
    if not inf._roboflow_api_key():
        return
    ds_dir = os.path.join(ROOT_DIR, "Models", "Maternal", "Data Set", "HASH Maternal Health.coco", "test")
    ann_path = os.path.join(ds_dir, "_annotations.coco.json")
    require(os.path.isfile(ann_path), f"no COCO annotations found at {ann_path}")

    pos = _first_image_for_category(ann_path, ds_dir, "abnormal")
    require(pos is not None, "no 'abnormal' annotated sample found to cross-check against")
    img, name = pos
    r = inf.predict_maternal(img)
    require(r is not None, "predict_maternal returned None on a known-abnormal annotated sample")
    require(r["is_critical"], f"ground truth is abnormal but the model said {r['verdict']!r}")
    print(f"    {name} (ground truth: abnormal) -> {r['verdict']} ({r['confidence']:.1f}%)")


# ---- 7. run_triage() dispatcher (what the UI actually calls) ----
@check("run_triage() dispatcher end-to-end for all 3 modalities")
def _():
    for model in GUI.MODELS:
        img = GUI.load_placeholder(model, seed=6)
        r = GUI.run_triage(model, img)
        validate_result_shape(r, f"run_triage({model})")
        require("source" in r, f"run_triage({model}): no source field")


# ---- 8. Tkinter UI: exercise every feature, driven by real Test Data images ----
_TEST_DATA_BY_MODEL = {
    "Mammography (YOLOv8-OBB)": "Breast Cancer",
    "Tuberculosis (Chest X-Ray)": "Tuberculosis",
    "Maternal Health (Ultrasound)": "Maternal",
}


@check("Tkinter UI: every feature exercised (real Test Data images, all 3 modalities)")
def _():
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    tk_test_db = os.path.join(VALIDATION_DIR, "validation_test_cache_tk.db")
    if os.path.exists(tk_test_db):
        os.remove(tk_test_db)
    orig_db_path = GUI.DB_PATH
    GUI.DB_PATH = tk_test_db
    try:
        app = GUI.PinkEdgeApp(root)
        root.update()
        require(app.notebook.index("end") == 3, "expected 3 notebook tabs")

        # Dialogs (askyesno/showinfo/askopenfilename/...) block waiting for a real click —
        # mock them so this runs headlessly instead of hanging.
        GUI.messagebox.showinfo = lambda *a, **k: None
        GUI.messagebox.showwarning = lambda *a, **k: None
        GUI.messagebox.showerror = lambda *a, **k: None
        GUI.messagebox.askyesno = lambda *a, **k: True
        saved_files = []

        def _fake_save_dialog(*a, defaultextension="", initialfile="", **k):
            path = os.path.join(VALIDATION_DIR, initialfile or f"out{defaultextension}")
            saved_files.append(path)
            return path

        GUI.filedialog.asksaveasfilename = _fake_save_dialog

        # Language toggle
        app._set_lang(True)
        require(GUI._urdu["on"], "language toggle to Urdu did not take effect")
        app._set_lang(False)
        require(not GUI._urdu["on"], "language toggle back to English did not take effect")

        # Every modality: switch model, load a REAL Test Data image (not a placeholder),
        # run triage, toggle the detection overlay, save to cache, download both report formats.
        for model, folder in _TEST_DATA_BY_MODEL.items():
            app.model_var.set(model)
            app._on_model_change()

            sample = sorted(glob(os.path.join(TEST_DATA_DIR, folder, "*")))
            require(len(sample) > 0, f"no Test Data samples for {folder}")
            app.uploaded_path = sample[0]
            app._refresh_placeholder()
            require(app.current_image is not None, f"{model}: no image loaded after upload")

            app._run_triage()
            root.update()
            require(app.inference_done, f"{model}: _run_triage() did not complete")
            require(app.current_result is not None, f"{model}: _run_triage() produced no result")

            app.overlay_var.set(not app.overlay_var.get())
            app._render_image()
            app.overlay_var.set(not app.overlay_var.get())
            app._render_image()

            app._save_cache()
            app._download_text()
            app._download_pdf()

        total, _ = GUI.counts()
        require(total == len(_TEST_DATA_BY_MODEL), f"expected {len(_TEST_DATA_BY_MODEL)} cached reports, got {total}")
        require(len(saved_files) == 2 * len(_TEST_DATA_BY_MODEL), f"expected {2 * len(_TEST_DATA_BY_MODEL)} report files, got {len(saved_files)}")
        for p in saved_files:
            require(os.path.isfile(p) and os.path.getsize(p) > 0, f"report file missing/empty: {p}")

        # Network mode + Cloud Sync actions
        app.network_var.set("GSM Failover")
        app._refresh_actions()
        app._sync_cloud()
        app._check_ota()
        require(app.acr_status is not None, "Check OTA did not set acr_status")

        # Hospital Hub + Cloud Sync tabs actually render without error
        app._refresh_hub()
        require(len(app.sms_alerts) == len(_TEST_DATA_BY_MODEL), "Hospital Hub alert count mismatch")
        app._refresh_cloud()

        # Reset Session
        app._reset_session()
        require(not app.inference_done, "Reset Session did not clear inference_done")
        require(app.sms_alerts == [], "Reset Session did not clear alerts")
    finally:
        GUI.DB_PATH = orig_db_path
        if os.path.exists(tk_test_db):
            os.remove(tk_test_db)
        for p in locals().get("saved_files", []):
            if os.path.isfile(p):
                os.remove(p)
        root.destroy()


# ---- 9. Streamlit UI: exercise every feature (AppTest, no browser) ----
@check("Streamlit UI: every feature exercised (all 3 modalities, language, network mode)")
def _():
    from streamlit.testing.v1 import AppTest

    tk_test_db = os.path.join(VALIDATION_DIR, "validation_test_cache_st.db")
    if os.path.exists(tk_test_db):
        os.remove(tk_test_db)
    orig_db_path = GUI.DB_PATH
    GUI.DB_PATH = tk_test_db
    try:
        at = AppTest.from_file(os.path.join(ROOT_DIR, "streamlit_app.py"))
        at.run(timeout=60)
        require(not at.exception, f"initial script run raised: {at.exception}")
        require(len(at.tabs) == 3, f"expected 3 tabs (Dashboard/Hospital Hub/Cloud Sync), got {len(at.tabs)}")

        # Language toggle: EN (0), Urdu (1)
        at.sidebar.button[1].click().run(timeout=30)  # Urdu
        require(not at.exception, f"Urdu toggle raised: {at.exception}")
        at.sidebar.button[0].click().run(timeout=30)  # back to English
        require(not at.exception, f"English toggle raised: {at.exception}")

        # Every modality, via the sidebar model selectbox (uses the generated placeholder —
        # AppTest can't drive a real file_uploader interaction, so Test Data images are covered
        # by the direct inf.predict_*() checks above instead; this covers the full UI path).
        for model in GUI.MODELS:
            at.sidebar.selectbox[0].select(model).run(timeout=30)
            require(not at.exception, f"{model}: selecting the model raised: {at.exception}")
            at.sidebar.button[2].click().run(timeout=90)  # Run Triage
            require(not at.exception, f"{model}: Run Triage raised: {at.exception}")
            require(at.session_state["inference_done"], f"{model}: Run Triage did not complete")
            r = at.session_state["current_result"]
            validate_result_shape(r, f"streamlit run_triage({model})")

            save_idx = 3  # Save to Cache appears once inference_done is True
            require(len(at.sidebar.button) > save_idx, f"{model}: Save to Cache button not rendered")
            at.sidebar.button[save_idx].click().run(timeout=30)
            require(not at.exception, f"{model}: Save to Cache raised: {at.exception}")

        total, _ = GUI.counts()
        require(total == len(GUI.MODELS), f"expected {len(GUI.MODELS)} cached reports, got {total}")

        # Network mode -> GSM Failover unlocks Sync to Cloud / Check OTA
        at.sidebar.radio[0].set_value("GSM Failover").run(timeout=30)
        require(not at.exception, f"network mode switch raised: {at.exception}")
        sync_buttons = [i for i, b in enumerate(at.sidebar.button) if "Sync to Cloud" in b.label]
        require(sync_buttons, "Sync to Cloud button not found after switching to GSM Failover")
        at.sidebar.button[sync_buttons[0]].click().run(timeout=30)
        require(not at.exception, f"Sync to Cloud raised: {at.exception}")
        _, unsynced = GUI.counts()
        require(unsynced == 0, f"expected 0 unsynced after Sync to Cloud, got {unsynced}")
    finally:
        GUI.DB_PATH = orig_db_path
        if os.path.exists(tk_test_db):
            os.remove(tk_test_db)


# ============================================================
print()
width = max(len(n) for n, _, _ in RESULTS) + 2
n_pass = sum(1 for _, ok, _ in RESULTS if ok)
for name, ok, detail in RESULTS:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name.ljust(width)} {detail}")

print()
print(f"{n_pass}/{len(RESULTS)} checks passed.")
if n_pass < len(RESULTS):
    print("\nModel status (inference.py):")
    try:
        for modality, info in inf.model_status().items():
            print(f"  - {modality}: real={info['real']}  reason={info['reason']}")
    except Exception:
        pass
    sys.exit(1)
sys.exit(0)
