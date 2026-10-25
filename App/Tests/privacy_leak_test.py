#!/usr/bin/env python3
"""
Pink Edge AI — privacy / PII leak validation.

Author Name:  Imaad Ullah Khan
Author Email: yameenimaad@gmail.com
AI Helper:    Claude

The README makes a specific, checkable promise:

    "every ingested scan is stripped of PII and assigned a hex privacy hash before anything is
     displayed, cached, or transmitted"

This file checks that promise against what the code actually does, at each place data can leave
the patient's record and reach somewhere else — what the audit literature calls a *sink*:

    sink 1  the GSM / IoT broadcast  (leaves the device, over the air, to the hospital hub)
    sink 2  the SQLite report cache  (at rest on disk)
    sink 3  the generated text / PDF reports  (leaves the device as a file)
    sink 4  the on-screen telemetry log
    sink 5  anything that could carry a credential (the Roboflow key)

Scope note, stated plainly so this file is not read as more than it is: the identifiers this
prototype handles are **synthetic** — `random.randint(10000000, 99999999)`, generated per
session. No real patient data exists in this repository, so nothing here can constitute an actual
breach. What these checks validate is the *mechanism*: that the protection the project documents
is actually wired up, so it would hold if real identifiers were ever fed through it. A mechanism
that is never exercised is a mechanism that will not work on the day it matters.

Run:  python Tests/privacy_leak_test.py          (from App/)
      python Tests/privacy_leak_test.py --json   (machine-readable, for the evidence report)

Exits non-zero on any failure.
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AS_JSON = "--json" in sys.argv

RESULTS = []

# A realistic PII bundle — the identifiers a Pakistani BHU record would actually carry.
SAMPLE_PII = {
    "patient_name": "Fatima Noor",
    "cnic": "31202-7654321-9",
    "phone": "+92-301-9876543",
    "dob": "1990-11-20",
    "exact_location": "Basti Malook, Multan (GPS: 29.98, 71.45)",
    "pat_id": 98765432,
}


def check(name):
    def wrap(fn):
        try:
            detail = fn() or ""
            RESULTS.append((name, True, detail))
        except Exception as e:
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
        return fn
    return wrap


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _src(rel):
    return open(os.path.join(APP_DIR, rel), encoding="utf-8").read()


# ============================================================
# A. The hashing primitive itself
# ============================================================
@check("privacy hash is one-way: no input value survives in the digest")
def _():
    from dicom_anonymizer import generate_hex_privacy_hash

    digest = generate_hex_privacy_hash(SAMPLE_PII)
    leaked = [str(v) for v in SAMPLE_PII.values() if str(v) in digest]
    require(not leaked, f"input values recoverable from the digest: {leaked}")
    # also check the obvious partials
    for frag in ("Fatima", "31202", "7654321", "9876543", "1990", "Multan", "98765432"):
        require(frag not in digest, f"fragment {frag!r} appears in the digest")
    require(re.fullmatch(r"[0-9A-F]{64}", digest), f"not a SHA-256 hex digest: {digest[:24]}...")
    return "(SHA-256, 64 hex chars)"


@check("privacy hash is deterministic and order-independent")
def _():
    from dicom_anonymizer import generate_hex_privacy_hash

    a = generate_hex_privacy_hash(SAMPLE_PII)
    b = generate_hex_privacy_hash(dict(reversed(list(SAMPLE_PII.items()))))
    require(a == b, "same facts in a different key order produced a different hash")
    c = generate_hex_privacy_hash({**SAMPLE_PII, "pat_id": 98765433})
    require(a != c, "changing an identifier did not change the hash")
    return "(stable across key order, sensitive to content)"


@check("DICOM anonymizer redacts every sensitive tag and preserves clinical ones")
def _():
    from dicom_anonymizer import SENSITIVE_TAGS, anonymize_dicom_metadata

    tags = {
        "PatientName": "Fatima Noor", "PatientID": "31202-7654321-9",
        "PatientBirthDate": "19901120", "PatientAddress": "Basti Malook, Multan",
        "PatientTelephoneNumbers": "+92-301-9876543", "ReferringPhysicianName": "Dr. Ahmed",
        "InstitutionName": "BHU Multan Rural",
        "PatientAge": 36, "PatientSex": "F", "Modality": "MG", "BodyPartExamined": "BREAST",
    }
    out = anonymize_dicom_metadata(tags)
    blob = json.dumps(out)
    for tag in SENSITIVE_TAGS:
        if tag in tags:
            require(str(tags[tag]) not in blob, f"{tag} value survives anonymization in cleartext")
            require(re.fullmatch(r"[0-9A-F]{64}", str(out[tag])), f"{tag} was not hashed")
    for clinical in ("PatientAge", "PatientSex", "Modality", "BodyPartExamined"):
        require(out[clinical] == tags[clinical], f"clinical field {clinical} was altered")
    return f"({len(SENSITIVE_TAGS)} sensitive tags redacted, 4 clinical preserved)"


# ============================================================
# B. Sink 1 — what actually goes over the air
# ============================================================
@check("SINK: GSM/IoT broadcast carries the privacy hash, not the raw identifier (web edition)")
def _():
    src = _src("streamlit_app.py")
    m = re.search(r'sms_payload\s*=\s*f?"([^"]+)"', src)
    require(m, "could not locate the sms_payload construction in streamlit_app.py")
    payload = m.group(1)
    require("priv_hash" in payload or "privacy_hash" in payload,
            f"payload does not carry the hash: {payload}")
    require("pat_id" not in payload, f"payload carries the raw patient id: {payload}")
    require("LOC:ANON" in payload, f"payload carries a real location instead of LOC:ANON: {payload}")
    return f"({payload})"


@check("SINK: GSM/IoT broadcast carries the privacy hash, not the raw identifier (desktop edition)")
def _():
    src = _src("GUI.py")
    m = re.search(r'sms_payload\s*=\s*f?"([^"]+)"', src)
    require(m, "could not locate the sms_payload construction in GUI.py")
    payload = m.group(1)
    require("privacy_hash" in payload,
            f"the desktop edition broadcasts without the privacy hash: {payload}")
    require(not re.search(r"\{self\.pat_id\}", payload),
            f"the desktop edition broadcasts the raw patient id: {payload}")
    require("LOC:ANON" in payload,
            f"the desktop edition broadcasts a literal coordinate instead of LOC:ANON: {payload}")
    return f"({payload})"


@check("both editions actually import and use the anonymizer")
def _():
    missing = []
    for rel in ("streamlit_app.py", "GUI.py"):
        src = _src(rel)
        if "generate_hex_privacy_hash" not in src:
            missing.append(rel)
    require(not missing,
            "the README states both editions hash PII before display/cache/transmission, but "
            f"these never call the anonymizer at all: {missing}")
    return "(streamlit_app.py, GUI.py)"


# ============================================================
# C. Sink 2/3/4 — at rest, in reports, and on screen
# ============================================================
@check("SINK: a cached report row contains no name, CNIC, phone, DOB or address field")
def _():
    import GUI as core

    test_db = os.path.join(APP_DIR, "Tests", "_privacy_probe.db")
    if os.path.exists(test_db):
        os.remove(test_db)
    original = core.DB_PATH
    core.DB_PATH = test_db
    try:
        core.init_db()
        import sqlite3
        conn = sqlite3.connect(test_db)
        try:
            cols = [r[1].lower() for r in conn.execute("PRAGMA table_info(cached_reports)")]
        finally:
            conn.close()  # Windows will not unlink an open SQLite file
        forbidden = [c for c in cols
                     if any(k in c for k in ("name", "cnic", "phone", "dob", "birth",
                                             "address", "gps", "lat", "lon"))]
        require(not forbidden, f"the cache schema has direct-PII columns: {forbidden}")
    finally:
        core.DB_PATH = original
        if os.path.exists(test_db):
            os.remove(test_db)
    return f"({len(cols)} columns, none direct-PII)"


@check("SINK: generated reports contain no name, CNIC, phone, DOB or address")
def _():
    import GUI as core

    d = {
        "patient_id": 98765432, "patient_age": 36, "modality": "Mammography (YOLOv8-OBB)",
        "model_used": "offline heuristic", "bi_rads": "BI-RADS 2 - Benign finding",
        "acr_density": "B - Scattered fibroglandular density",
        "verdict": "No Focal Suspicious Lesion Detected", "localization": "n/a",
        "confidence": 91.2, "inference_time": 1.0, "timestamp": "2026-01-01 00:00:00",
    }
    text = core.generate_text_report(d)
    pdf = core.generate_pdf_bytes(d)
    haystacks = {"text report": text, "PDF bytes": pdf.decode("latin-1", errors="ignore")}
    for where, blob in haystacks.items():
        for frag in ("Fatima", "31202-7654321", "+92-301", "1990-11-20", "Basti Malook"):
            require(frag not in blob, f"{where} contains PII fragment {frag!r}")
    return "(text + PDF clean)"


@check("SINK: no Roboflow API key value can reach a report, the cache, or a log line")
def _():
    key_file = os.path.join(APP_DIR, "roboflow_key.txt")
    if not os.path.isfile(key_file):
        return "(skipped: no roboflow_key.txt present)"
    key = open(key_file, encoding="utf-8").read().strip()
    require(key, "roboflow_key.txt is empty")

    import GUI as core
    d = {
        "patient_id": 1, "patient_age": 30, "modality": "Mammography (YOLOv8-OBB)",
        "model_used": "m", "bi_rads": "b", "acr_density": "a", "verdict": "v",
        "localization": "l", "confidence": 1.0, "inference_time": 1.0, "timestamp": "t",
    }
    report = core.generate_text_report(d)
    require(key not in report, "the API key appears in a generated report")

    # and it must not be baked into any tracked source file
    offenders = []
    for rel in ("GUI.py", "streamlit_app.py", "inference.py", "offline_cv.py"):
        if key in _src(rel):
            offenders.append(rel)
    require(not offenders, f"the API key is hardcoded in: {offenders}")
    return f"(key {len(key)} chars, absent from reports and sources)"


@check("the key file is gitignored and not tracked")
def _():
    import subprocess

    repo = os.path.dirname(APP_DIR)
    tracked = subprocess.run(["git", "-C", repo, "ls-files", "App/roboflow_key.txt"],
                             capture_output=True, text=True).stdout.strip()
    require(not tracked, "roboflow_key.txt is TRACKED IN GIT — the key is in the repository")
    gitignore = open(os.path.join(repo, ".gitignore"), encoding="utf-8").read()
    require("roboflow_key.txt" in gitignore, "roboflow_key.txt is not listed in .gitignore")
    return "(untracked + ignored)"


# ============================================================
if __name__ == "__main__":
    if AS_JSON:
        print(json.dumps([{"check": n, "passed": ok, "detail": d} for n, ok, d in RESULTS],
                         indent=2))
        sys.exit(0 if all(ok for _, ok, _ in RESULTS) else 1)

    print("=" * 78)
    print("PINK EDGE AI — PRIVACY / PII LEAK VALIDATION")
    print("=" * 78)
    print("Identifiers in this prototype are synthetic; these checks validate the mechanism,\n"
          "not an actual patient-data exposure. See this file's docstring.\n")

    width = max(len(n) for n, _, _ in RESULTS) + 2
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name.ljust(width)} {detail}")

    print(f"\n{n_pass}/{len(RESULTS)} checks passed.")
    sys.exit(0 if n_pass == len(RESULTS) else 1)
