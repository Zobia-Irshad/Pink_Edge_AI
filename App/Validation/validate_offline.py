#!/usr/bin/env python3
"""
Pink Edge AI — offline validation suite.

Author Name:  Imaad Ullah Khan
Author Email: yameenimaad@gmail.com
AI Helper:    Claude

Proves the project's central claim — that triage keeps working with **zero internet** — by
actually removing the network and re-running the pipeline, rather than by inspecting code or
trusting a comment.

How the network is removed: every socket entry point in the stdlib (`socket.socket`,
`create_connection`, `getaddrinfo`, …) is replaced with a function that raises, *before*
`inference.py` or any HTTP library is imported. Hugging Face's hub is additionally pinned to its
offline mode so it serves cached weights instead of revalidating them over the wire. A control
check at the top confirms the block is really in force, so this suite can never pass by
accidentally having had connectivity.

What that leaves available, and therefore what this validates:
  - the offline pixel-diff heuristic  (offline_cv.py — no weights, no network, ever)
  - the locally-trained MobileNetV3 classifiers  (Models/<Modality>/local_model.pt)
  - the Hugging Face models already cached on disk  (Models/TB/model.pt, Maternal/*.pt)
  - the simulated scenario pickers  (GUI.py — the honest last resort)
and what it must prove is *unreachable*: every Roboflow-hosted path.

Run:  python Validation/validate_offline.py          (from App/)
      python Validation/validate_offline.py -v       (show each check as it runs)

Exits non-zero if any check fails. Companion to validate.py (which allows the network and
exercises the hosted models); this one is the half you can run on a plane.
"""

import os
import sys

# ------------------------------------------------------------------
# Cut the network BEFORE anything that might open a connection is imported.
# ------------------------------------------------------------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
# Make sure a stray Roboflow key can't send us down the hosted path at all.
os.environ.pop("ROBOFLOW_API_KEY", None)

import socket  # noqa: E402


class NetworkBlocked(OSError):
    """Raised on any attempt to reach a non-local host while this suite runs."""


# Loopback stays reachable: "no internet" is about outbound traffic, and blanket-blocking the
# socket module breaks local machinery (Streamlit's test harness, torch's loaders) in ways that
# would make this suite fail for reasons that have nothing to do with connectivity.
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "", None}

_REAL_SOCKET = socket.socket
_REAL_GETADDRINFO = socket.getaddrinfo
_REAL_CREATE_CONNECTION = socket.create_connection


def _host_of(address):
    return address[0] if isinstance(address, (tuple, list)) and address else address


def _guard(address, what):
    host = _host_of(address)
    if str(host) not in {str(h) for h in _LOCAL_HOSTS}:
        raise NetworkBlocked(f"{what} to {host!r} blocked by validate_offline.py")


class _OfflineSocket(_REAL_SOCKET):
    """A real socket that refuses to connect anywhere but loopback. Subclassed rather than
    replaced so libraries that introspect or inherit from socket.socket still work."""

    def connect(self, address, *a, **k):
        _guard(address, "outbound connection")
        return super().connect(address, *a, **k)

    def connect_ex(self, address, *a, **k):
        _guard(address, "outbound connection")
        return super().connect_ex(address, *a, **k)


def _offline_getaddrinfo(host, *a, **k):
    _guard(host, "DNS lookup")
    return _REAL_GETADDRINFO(host, *a, **k)


def _offline_create_connection(address, *a, **k):
    _guard(address, "outbound connection")
    return _REAL_CREATE_CONNECTION(address, *a, **k)


socket.socket = _OfflineSocket
socket.getaddrinfo = _offline_getaddrinfo
socket.create_connection = _offline_create_connection

VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(VALIDATION_DIR)
TEST_DATA_DIR = os.path.join(ROOT_DIR, "Test Data")
sys.path.insert(0, ROOT_DIR)

VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv
RESULTS = []
HOSTED_MARKERS = ("roboflow", "serverless", "hosted")


def check(name):
    def wrap(fn):
        try:
            detail = fn() or ""
            RESULTS.append((name, True, detail))
            if VERBOSE:
                print(f"  [pass] {name} {detail}")
        except Exception as e:
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            if VERBOSE:
                print(f"  [FAIL] {name} — {type(e).__name__}: {e}")
        return fn
    return wrap


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _sample(folder):
    from glob import glob
    files = sorted(glob(os.path.join(TEST_DATA_DIR, folder, "*")))
    require(files, f"no sample images under Test Data/{folder}")
    return files[0]


# ============================================================
# 0. Control — the block itself must be real.
# ============================================================
@check("CONTROL: the network really is unreachable")
def _():
    import urllib.request

    for attempt, fn in (
        ("socket.create_connection", lambda: socket.create_connection(("1.1.1.1", 443), timeout=2)),
        ("socket.getaddrinfo", lambda: socket.getaddrinfo("huggingface.co", 443)),
        ("urllib HTTPS GET", lambda: urllib.request.urlopen("https://huggingface.co", timeout=2)),
    ):
        try:
            fn()
        except Exception:
            continue  # expected
        raise AssertionError(f"{attempt} SUCCEEDED — the network is NOT blocked, so a pass here "
                             f"would prove nothing")
    return "(3 escape routes blocked)"


# ============================================================
# 1. The modules must import with no network at all.
# ============================================================
@check("inference.py and GUI.py import with the network down")
def _():
    import GUI  # noqa: F401
    import inference  # noqa: F401
    import offline_cv  # noqa: F401
    return "(3 modules)"


# ============================================================
# 2. Every modality still produces a usable result.
# ============================================================
def _assert_result_shape(r, who):
    require(r is not None, f"{who}: returned None — no offline tier was able to answer")
    for field in ("verdict", "confidence", "source", "is_critical", "css"):
        require(field in r, f"{who}: result missing {field!r}")
    require(0.0 <= float(r["confidence"]) <= 100.0, f"{who}: confidence out of range ({r['confidence']})")
    require(r["css"] in ("success", "warning", "danger"), f"{who}: bad css token {r['css']!r}")


@check("Mammography triage works offline on a real sample")
def _():
    import inference as inf
    from PIL import Image

    r = inf.predict_mammography(Image.open(_sample("Breast Cancer")).convert("RGB"))
    _assert_result_shape(r, "predict_mammography")
    return f"({r['source'][:52]})"


@check("Tuberculosis triage works offline on a real sample")
def _():
    import inference as inf
    from PIL import Image

    r = inf.predict_tb(Image.open(_sample("Tuberculosis")).convert("RGB"))
    _assert_result_shape(r, "predict_tb")
    return f"({r['source'][:52]})"


@check("Maternal Health triage works offline on a real sample")
def _():
    import inference as inf
    from PIL import Image

    r = inf.predict_maternal(Image.open(_sample("Maternal")).convert("RGB"))
    _assert_result_shape(r, "predict_maternal")
    return f"({r['source'][:52]})"


# ============================================================
# 3. Nothing may claim a hosted source while offline.
# ============================================================
@check("no modality reports a hosted/Roboflow source while offline")
def _():
    import GUI as core
    import inference as inf
    from PIL import Image

    offenders = []
    for model in core.MODELS:
        img = core.load_placeholder(model, seed=11)
        r = core.run_triage(model, img)
        src = (r or {}).get("source", "").lower()
        if any(m in src for m in HOSTED_MARKERS):
            offenders.append(f"{model} -> {r['source']}")
    require(not offenders,
            "a hosted source was reported with no network available, which means the source "
            "label is not trustworthy: " + "; ".join(offenders))
    _ = inf  # imported to prove it loads offline
    return "(3 modalities)"


# ============================================================
# 4. The individual offline tiers each work on their own.
# ============================================================
@check("offline pixel-diff heuristic (no weights, no network) answers for every modality")
def _():
    import offline_cv
    from PIL import Image

    answered = []
    for modality, folder in (("mammography", "Breast Cancer"), ("tb", "Tuberculosis"),
                             ("maternal", "Maternal")):
        if not offline_cv.available(modality):
            continue  # dataset not present in this checkout — not a failure of the offline path
        r = offline_cv.predict(modality, Image.open(_sample(folder)).convert("RGB"))
        require(r is not None, f"offline_cv.predict({modality!r}) returned None")
        answered.append(modality)
    require(answered, "offline_cv could not answer for any modality — no local datasets present")
    return f"({', '.join(answered)})"


@check("locally-trained classifiers load from disk with no network")
def _():
    import inference as inf

    loaded = [m for m in ("tb", "mammography") if inf.local_model_available(m)]
    require(loaded, "no local_model.pt could be loaded offline; errors: " +
                    str({m: inf._local_model_errors.get(m) for m in ("tb", "mammography")}))
    return f"({', '.join(loaded)})"


@check("cached Hugging Face weights load with HF_HUB_OFFLINE=1")
def _():
    import inference as inf

    available = []
    if inf.tb_available():
        available.append("TB ViT")
    if inf.maternal_available():
        available.append("Maternal CNN")
    require(available, "neither cached HF model could be loaded offline — if this is a fresh "
                       "checkout, run validate.py once with internet to populate the cache")
    return f"({', '.join(available)})"


# ============================================================
# 5. Everything downstream of inference is local by nature — prove it.
# ============================================================
@check("SQLite cache round-trip works offline (throwaway DB)")
def _():
    import GUI as core

    test_db = os.path.join(VALIDATION_DIR, "offline_test_cache.db")
    if os.path.exists(test_db):
        os.remove(test_db)
    original = core.DB_PATH
    core.DB_PATH = test_db
    try:
        core.init_db()
        core.save_to_cache({
            "patient_id": 12345678, "patient_age": 41, "modality": "Mammography",
            "model_used": "offline", "bi_rads": "BI-RADS 2 - Benign finding", "acr_density": "B",
            "verdict": "No Focal Suspicious Lesion Detected", "localization": "n/a",
            "confidence": 91.2, "inference_time": 1.0, "timestamp": "2026-01-01 00:00:00",
        })
        total, unsynced = core.counts()
        require(total == 1, f"expected 1 cached report, got {total}")
        require(unsynced == 1, f"expected 1 unsynced report, got {unsynced}")
        core.mark_as_synced([r[0] for r in core.get_unsynced_reports()])
        _, unsynced_after = core.counts()
        require(unsynced_after == 0, f"expected 0 unsynced after sync, got {unsynced_after}")
    finally:
        core.DB_PATH = original
        if os.path.exists(test_db):
            os.remove(test_db)
    return "(save/read/sync)"


@check("text + PDF report generation works offline")
def _():
    import GUI as core

    d = {
        "patient_id": 12345678, "patient_age": 41, "modality": "Mammography (YOLOv8-OBB)",
        "model_used": "Offline heuristic", "bi_rads": "BI-RADS 2 - Benign finding",
        "acr_density": "B - Scattered fibroglandular density", "verdict": "No Focal Suspicious Lesion",
        "localization": "n/a", "confidence": 91.2, "inference_time": 1.0,
        "timestamp": "2026-01-01 00:00:00",
    }
    text = core.generate_text_report(d)
    require("BI-RADS" in text, "text report missing its clinical fields")
    pdf = core.generate_pdf_bytes(d)
    require(pdf[:4] == b"%PDF", "generate_pdf_bytes did not return a PDF")
    return f"(text {len(text)}B, pdf {len(pdf)}B)"


# ============================================================
# 6. Both UIs must stand up with no network.
# ============================================================
@check("Streamlit UI boots and runs a full triage offline")
def _():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(ROOT_DIR, "streamlit_app.py"))
    at.run(timeout=90)
    require(not at.exception, f"startup raised offline: {list(at.exception)}")
    run = [b for b in at.sidebar.button if "Run Triage" in b.label]
    require(run, "no Run Triage control found")
    run[0].click().run(timeout=120)
    require(not at.exception, f"Run Triage raised offline: {list(at.exception)}")
    require(at.session_state["inference_done"], "Run Triage did not complete offline")
    src = at.session_state["current_result"].get("source", "").lower()
    require(not any(m in src for m in HOSTED_MARKERS),
            f"the offline UI reported a hosted source: {src}")
    return "(boot + triage)"


@check("Tkinter desktop UI constructs offline (hidden window, no mainloop)")
def _():
    try:
        import tkinter as tk
    except Exception as e:  # headless CI without Tk — not an offline failure
        return f"(skipped: tkinter unavailable — {type(e).__name__})"

    import GUI as core

    root = tk.Tk()
    root.withdraw()
    test_db = os.path.join(VALIDATION_DIR, "offline_test_cache_tk.db")
    original = core.DB_PATH
    core.DB_PATH = test_db
    try:
        app = core.PinkEdgeApp(root)
        require(app is not None, "PinkEdgeApp failed to construct")
    finally:
        core.DB_PATH = original
        root.destroy()
        if os.path.exists(test_db):
            os.remove(test_db)
    return "(constructed)"


# ============================================================
if __name__ == "__main__":
    print("=" * 78)
    print("PINK EDGE AI — OFFLINE VALIDATION SUITE  (network forcibly disabled)")
    print("=" * 78)
    if not VERBOSE:
        print("(run with -v to see checks as they execute)\n")

    width = max(len(n) for n, _, _ in RESULTS) + 2
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name.ljust(width)} {detail}")

    print(f"\n{n_pass}/{len(RESULTS)} checks passed.")
    if n_pass == len(RESULTS):
        print("\nEvery modality produced a usable triage result with the network physically "
              "unavailable,\nand none of them claimed a hosted model while doing so.")
    sys.exit(0 if n_pass == len(RESULTS) else 1)
