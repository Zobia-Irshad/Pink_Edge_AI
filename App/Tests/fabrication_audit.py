#!/usr/bin/env python3
"""
Pink Edge AI — data fabrication audit.

Author Name:  Imaad Ullah Khan
Author Email: yameenimaad@gmail.com
AI Helper:    Claude

A clinical triage UI shows a reader a set of fields — verdict, confidence, localization,
gestational age — and the reader cannot tell, by looking, which of those were *measured from the
image* and which were *invented by the program*. This tool answers that question empirically.

Method (the dynamic audit is the important half):

  1. DYNAMIC — run each `predict_*()` twice on the **same image** and diff the two result dicts.
     A field derived from the image is identical both times. A field that changes is, by
     definition, not a function of the input: it was fabricated. This needs no knowledge of the
     source code and cannot be fooled by a comment claiming something is real.

  2. STATIC — locate every `random.*` call site in the inference and UI layers and report it, so
     the dynamic findings can be traced to a line of code.

  3. CLASSIFY — the distinction that actually matters:
       * fabricated inside a result whose `source` says SIMULATED  -> honest; the UI says so.
       * fabricated inside a result whose `source` claims REAL inference -> flagged. The reader
         is told the number came from a model, and for that field it did not.

Exit code: 0 if no fabricated field appears inside a result claiming real inference; 1 if any
does. The full inventory prints either way — this is a diagnostic first and a gate second.

Run:  python Tests/fabrication_audit.py            (from App/)
      python Tests/fabrication_audit.py --json     (machine-readable, for the evidence report)
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AS_JSON = "--json" in sys.argv

# Fields a clinician reads as findings. A non-deterministic value in any of these is the
# serious case; `timestamp`-like bookkeeping fields are excluded as legitimately varying.
CLINICAL_FIELDS = {
    "verdict", "confidence", "bi_rads", "acr", "loc", "sub", "extra",
    "is_critical", "sms", "image_quality", "recommendation", "css", "vicon",
}
# Varies by design, not a finding.
EXEMPT_FIELDS = {"inference_time", "timestamp", "bbox"}

REAL_SOURCE_MARKERS = ("real inference", "roboflow", "hugging face", "local_model.pt",
                       "locally-trained", "heuristic")
SIMULATED_MARKERS = ("simulated",)


def _repeat_predict(fn, image, runs=3):
    """Call fn(image) `runs` times on the identical image; return the list of result dicts."""
    out = []
    for _ in range(runs):
        r = fn(image)
        out.append(dict(r) if r else None)
    return out


def _diff_fields(results):
    """Fields whose value is not identical across every run."""
    varying = {}
    first = results[0]
    if first is None:
        return varying
    for key in first:
        if key in EXEMPT_FIELDS:
            continue
        values = [(r or {}).get(key) for r in results]
        if len({json.dumps(v, default=str, sort_keys=True) for v in values}) > 1:
            varying[key] = values
    return varying


def dynamic_audit():
    """Same image in, N times. Anything that changes was not derived from the image."""
    import inference as inf
    from glob import glob
    from PIL import Image

    findings = []
    targets = [
        ("Mammography", inf.predict_mammography, "Breast Cancer"),
        ("Tuberculosis", inf.predict_tb, "Tuberculosis"),
        ("Maternal Health", inf.predict_maternal, "Maternal"),
    ]
    for modality, fn, folder in targets:
        files = sorted(glob(os.path.join(APP_DIR, "Test Data", folder, "*")))
        if not files:
            findings.append({"modality": modality, "error": f"no Test Data/{folder} sample"})
            continue
        img = Image.open(files[0]).convert("RGB")
        results = _repeat_predict(fn, img)
        if results[0] is None:
            findings.append({"modality": modality, "error": "predict returned None"})
            continue
        source = results[0].get("source", "")
        varying = _diff_fields(results)
        claims_real = (any(m in source.lower() for m in REAL_SOURCE_MARKERS)
                       and not any(m in source.lower() for m in SIMULATED_MARKERS))
        findings.append({
            "modality": modality,
            "source": source,
            "claims_real_inference": claims_real,
            "sample": os.path.basename(files[0]),
            "fabricated_fields": {k: v for k, v in varying.items() if k in CLINICAL_FIELDS},
            "other_varying_fields": {k: v for k, v in varying.items() if k not in CLINICAL_FIELDS},
        })
    return findings


def static_audit():
    """Every random.* call site in the layers that build a result dict."""
    sites = []
    for rel in ("inference.py", "GUI.py", "streamlit_app.py"):
        path = os.path.join(APP_DIR, rel)
        if not os.path.isfile(path):
            continue
        for n, line in enumerate(open(path, encoding="utf-8"), 1):
            if re.search(r"\brandom\.(uniform|randint|choice|random)\b", line):
                sites.append({"file": rel, "line": n, "code": line.strip()})
    return sites


def main():
    dyn = dynamic_audit()
    stat = static_audit()

    violations = [f for f in dyn if f.get("claims_real_inference") and f.get("fabricated_fields")]

    if AS_JSON:
        print(json.dumps({"dynamic": dyn, "static_sites": stat,
                          "violations": len(violations)}, indent=2, default=str))
        return 1 if violations else 0

    print("=" * 78)
    print("PINK EDGE AI — DATA FABRICATION AUDIT")
    print("=" * 78)
    print("\nSame image submitted 3x per modality. Any field that changes between runs was not\n"
          "derived from the image — it was generated.\n")

    for f in dyn:
        print("-" * 78)
        if f.get("error"):
            print(f"{f['modality']}: SKIPPED — {f['error']}")
            continue
        label = "claims REAL inference" if f["claims_real_inference"] else "labelled SIMULATED"
        print(f"{f['modality']}  [{label}]")
        print(f"  source: {f['source']}")
        print(f"  sample: {f['sample']}")
        if not f["fabricated_fields"]:
            print("  -> every clinical field was identical across runs (derived from the image)")
        else:
            for key, values in f["fabricated_fields"].items():
                shown = " | ".join(str(v)[:46] for v in values)
                print(f"  -> FABRICATED  {key:<16} varies: {shown}")
        for key, values in f.get("other_varying_fields", {}).items():
            print(f"     (non-clinical {key} also varies: {str(values[0])[:30]}...)")

    print("-" * 78)
    print(f"\nStatic inventory — {len(stat)} random.* call sites in the result-building layers:")
    by_file = {}
    for s in stat:
        by_file.setdefault(s["file"], []).append(s)
    for fname, entries in by_file.items():
        print(f"\n  {fname}  ({len(entries)} sites)")
        for s in entries[:40]:
            print(f"    L{s['line']:<5} {s['code'][:92]}")

    print("\n" + "=" * 78)
    if violations:
        print(f"RESULT: {len(violations)} modality/ies present fabricated clinical fields inside a")
        print("        result that tells the reader it came from a real model.\n")
        for v in violations:
            print(f"  - {v['modality']}: {', '.join(v['fabricated_fields'])}")
            print(f"      source shown to the reader: {v['source']}")
        print("\nThis is not necessarily a bug to delete — a demo may legitimately fill a field the")
        print("model cannot produce. It IS a labelling problem: the source string should not imply")
        print("those particular fields were measured. See Documentations/MODEL_SOURCES.md.")
    else:
        print("RESULT: no fabricated clinical field appears inside a result claiming real inference.")
    print("=" * 78)
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
