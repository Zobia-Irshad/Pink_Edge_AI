#!/usr/bin/env python3
"""
Pink Edge AI — evidence report generator.

Author Name:  Imaad Ullah Khan
Author Email: yameenimaad@gmail.com
AI Helper:    Claude

Runs every validation suite in this repository, captures what each one actually reported, and
writes a single dated evidence document (EVIDENCE_REPORT.md) that a reviewer, judge, or
compliance reader can check without running anything themselves.

Why this exists as a generated file rather than a written one: a claim in a README is an
assertion, and a reader has no way to tell a true one from an aspirational one. This report is
produced *from the actual exit codes and output* of the suites, and records the SHA-256 of every
source file it tested, so any later edit to those files invalidates the hashes and the report can
be regenerated and compared. It is evidence, not advertising.

Suites it runs:
  Validation/validate.py          full suite, network allowed (hosted models exercised)
  Validation/validate_offline.py  same pipeline with the network forcibly removed
  Tests/smoke_test.py             fast end-to-end + theme consistency
  Tests/fabrication_audit.py      which displayed fields are measured vs. generated
  Tests/privacy_leak_test.py      PII handling at every sink
  Tests/test_auth.py              RBAC unit tests
  Tests/test_anonymizer.py        anonymizer unit tests

Run:  python Evidence/generate_evidence.py            (from App/)
      python Evidence/generate_evidence.py --quick    (skip the two slow model suites)

Exit code mirrors the evidence: 0 only if every suite it ran passed.
"""

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone

EVIDENCE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.dirname(EVIDENCE_DIR)
REPO_DIR = os.path.dirname(APP_DIR)
REPORT_PATH = os.path.join(EVIDENCE_DIR, "EVIDENCE_REPORT.md")
QUICK = "--quick" in sys.argv

# Suites, in the order they appear in the report. (label, argv, slow?)
SUITES = [
    ("Full validation (network allowed)", ["Validation/validate.py"], True),
    ("Offline validation (network removed)", ["Validation/validate_offline.py"], True),
    ("Smoke test (end-to-end + theme)", ["Tests/smoke_test.py"], False),
    ("Data fabrication audit", ["Tests/fabrication_audit.py"], False),
    ("Privacy / PII leak validation", ["Tests/privacy_leak_test.py"], False),
    ("RBAC unit tests", ["Tests/test_auth.py"], False),
    ("Anonymizer unit tests", ["Tests/test_anonymizer.py"], False),
]

# Files whose integrity the report attests to.
ATTESTED = [
    "GUI.py", "streamlit_app.py", "inference.py", "offline_cv.py",
    "auth_manager.py", "dicom_anonymizer.py", "requirements.txt",
    "Validation/validate.py", "Validation/validate_offline.py",
    "Tests/smoke_test.py", "Tests/fabrication_audit.py", "Tests/privacy_leak_test.py",
    "Tests/test_auth.py", "Tests/test_anonymizer.py",
]

PASS_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+checks passed")
UNITTEST_RE = re.compile(r"^Ran (\d+) tests? in", re.M)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_suite(argv):
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run([sys.executable] + argv, cwd=APP_DIR, env=env,
                          capture_output=True, text=True, errors="replace", timeout=900)
    out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = PASS_RE.search(out)
    if m:
        score = f"{m.group(1)}/{m.group(2)}"
    elif UNITTEST_RE.search(out):
        n = UNITTEST_RE.search(out).group(1)
        score = f"{n}/{n}" if "\nOK" in out or out.rstrip().endswith("OK") else f"?/{n}"
    else:
        score = "n/a"
    return proc.returncode, score, out


def git(*args):
    try:
        return subprocess.run(["git", "-C", REPO_DIR, *args], capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except Exception:
        return ""


def main():
    started = datetime.now(timezone.utc)
    print(f"Running {len(SUITES)} suites for the evidence report "
          f"({'quick mode' if QUICK else 'full'})...\n")

    rows, logs, all_passed = [], {}, True
    for label, argv, slow in SUITES:
        if QUICK and slow:
            rows.append((label, " ".join(argv), "SKIPPED", "--quick"))
            continue
        print(f"  - {label} ...", end=" ", flush=True)
        try:
            rc, score, out = run_suite(argv)
        except subprocess.TimeoutExpired:
            rc, score, out = 124, "timeout", "suite exceeded its 900s budget"
        # The fabrication audit is a *diagnostic*: a non-zero exit means it found fabricated
        # fields, which is the tool working, not the tool failing. Recording that as "FAIL"
        # alongside a genuinely broken suite would make this report less truthful, not more.
        if label == "Data fabrication audit":
            verdict = "CLEAN" if rc == 0 else "FINDINGS"
        else:
            verdict = "PASS" if rc == 0 else "FAIL"
            all_passed = all_passed and rc == 0
        print(f"{verdict} ({score})")
        rows.append((label, " ".join(argv), verdict, score))
        logs[label] = out.strip()

    # Fabrication findings are the one result worth quoting inline — they're the thing a reader
    # of a medical-AI prototype most needs to see stated plainly.
    fab = logs.get("Data fabrication audit", "")
    fab_block = []
    capture = False
    for line in fab.splitlines():
        if line.startswith("RESULT:"):
            capture = True
        if capture and line.strip() and not line.startswith("="):
            fab_block.append(line.rstrip())

    finished = datetime.now(timezone.utc)
    lines = []
    w = lines.append

    w("# Pink Edge AI — Evidence Report")
    w("")
    w("Author Name:  Imaad Ullah Khan  ")
    w("Author Email: yameenimaad@gmail.com  ")
    w("AI Helper:    Claude")
    w("")
    w("> Generated by `Evidence/generate_evidence.py` from the actual exit codes and output of "
      "the suites listed below. Not hand-written. Regenerate it after any change to the attested "
      "files and the hashes at the bottom will change with them.")
    w("")
    w(f"- **Generated (UTC):** {started.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"- **Duration:** {(finished - started).total_seconds():.0f}s")
    w(f"- **Mode:** {'quick (slow model suites skipped)' if QUICK else 'full'}")
    w(f"- **Commit:** `{git('rev-parse', '--short', 'HEAD') or 'n/a'}` "
      f"({git('log', '-1', '--format=%s') or 'n/a'})")
    w(f"- **Working tree:** {'clean' if not git('status', '--porcelain') else 'MODIFIED since last commit'}")
    w(f"- **Platform:** {platform.platform()} / Python {platform.python_version()}")
    w("")

    w("## 1. Suite results")
    w("")
    w("| Suite | Command | Result | Checks |")
    w("|---|---|---|---|")
    for label, cmd, verdict, score in rows:
        w(f"| {label} | `python {cmd}` | **{verdict}** | {score} |")
    w("")
    w(f"**Overall: {'all suites passed' if all_passed else 'AT LEAST ONE SUITE FAILED'}.**")
    w("")
    w("`FINDINGS` on the fabrication audit is not a failure — that suite is a diagnostic, and a "
      "non-zero exit means it successfully identified generated fields being presented as model "
      "output. Its findings are in §3.")
    w("")

    w("## 2. What each suite establishes")
    w("")
    w("- **Full validation** — all three modalities produce well-formed results against the real "
      "model backends, cross-checked against each dataset's own COCO ground truth, plus full "
      "feature sweeps of both the Tkinter and Streamlit UIs.")
    w("- **Offline validation** — the same pipeline with every outbound socket blocked and "
      "`HF_HUB_OFFLINE=1`. A control check first proves the block is real (three escape routes "
      "attempted and refused), so a pass here cannot be an artefact of accidental connectivity. "
      "Establishes that triage works with no internet and that no modality claims a hosted "
      "source when none is reachable.")
    w("- **Smoke test** — the app starts, the three tabs render, role/language/modality controls "
      "re-run cleanly, Run Triage completes, Save to Cache actually adds a row; plus the theme is "
      "consistent across its definitions and every text colour clears WCAG AA contrast.")
    w("- **Fabrication audit** — submits the identical image three times per modality and diffs "
      "the results. Any field that changes was not derived from the image. See §3.")
    w("- **Privacy validation** — checks the hashing primitive, then what actually reaches each "
      "sink: the GSM/IoT broadcast, the SQLite cache, generated reports, and credential handling.")
    w("- **Unit tests** — RBAC permissions/PIN handling and the DICOM anonymizer, in isolation.")
    w("")

    w("## 3. Data fabrication — measured vs. generated")
    w("")
    w("A reader of a triage screen cannot tell which numbers a model produced and which the "
      "program invented. This is what the audit found by re-running identical inputs:")
    w("")
    if fab_block:
        w("```")
        for line in fab_block[:40]:
            w(line)
        w("```")
    else:
        w("_(audit not run in this report's mode)_")
    w("")

    w("## 4. Scope and limitations, stated plainly")
    w("")
    w("- This is a **hackathon prototype**, not a validated medical device. No output here is "
      "cleared for clinical use by any regulator.")
    w("- Patient identifiers handled by the app are **synthetic** (randomly generated per "
      "session). The privacy suite validates the *mechanism*, not the protection of real patient "
      "data, because no real patient data exists in this repository.")
    w("- Accuracy figures quoted in the project's documentation come from held-out splits of the "
      "project's own datasets, not from an independent clinical evaluation.")
    w("- The Alibaba Cloud sync, the GSM broadcast and the hardware telemetry panel are "
      "**simulated**; no cloud credentials and no radio hardware are involved.")
    w("")

    w("## 5. Integrity — SHA-256 of every attested file")
    w("")
    w("Any edit to these files changes its hash, so this report cannot silently outlive the code "
      "it describes.")
    w("")
    w("| File | SHA-256 |")
    w("|---|---|")
    for rel in ATTESTED:
        path = os.path.join(APP_DIR, rel)
        w(f"| `App/{rel}` | `{sha256(path) if os.path.isfile(path) else 'MISSING'}` |")
    w("")

    w("## 6. Reproducing this report")
    w("")
    w("```")
    w("cd App")
    w("python Evidence/generate_evidence.py")
    w("```")
    w("")
    w("Individual suites can be run on their own; each prints its own per-check results and exits "
      "non-zero on failure.")
    w("")

    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    # Full captured output, for anyone who wants the raw material behind the summary.
    raw_path = os.path.join(EVIDENCE_DIR, "evidence_raw_output.json")
    with open(raw_path, "w", encoding="utf-8") as fh:
        json.dump({"generated_utc": started.isoformat(), "logs": logs}, fh, indent=2)

    print(f"\nWrote {os.path.relpath(REPORT_PATH, APP_DIR)}")
    print(f"Wrote {os.path.relpath(raw_path, APP_DIR)}  (raw suite output)")
    print(f"\nOverall: {'ALL SUITES PASSED' if all_passed else 'AT LEAST ONE SUITE FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
