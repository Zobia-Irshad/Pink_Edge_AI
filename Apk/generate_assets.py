#!/usr/bin/env python3
"""
Desktop-only helper — run this BEFORE `buildozer android debug`, from a machine that has the real
datasets under ../App/Models/*/Data Set/ (i.e. the normal repo checkout, not the phone).

Reuses ../App/offline_cv.py's own template-building code (so the mobile app's offline heuristic is
built from the exact same averaged reference images as the desktop app's, not a re-derivation that
could quietly drift) and exports the two small 256x256 PNGs per modality into assets/<modality>/ —
that's what actually ships inside the APK; the multi-megabyte source datasets never do.

Usage:
    cd Apk
    python generate_assets.py
"""
import os
import sys

APK_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(APK_DIR)
APP_DIR = os.path.join(REPO_ROOT, "App")
sys.path.insert(0, APP_DIR)

import offline_cv  # noqa: E402  (../App/offline_cv.py — path adjusted above)

MODALITIES = ["tb", "maternal", "mammography"]


def main():
    any_built = False
    for modality in MODALITIES:
        pos_tpl, neg_tpl = offline_cv._get_templates(modality)
        out_dir = os.path.join(APK_DIR, "assets", modality)
        if pos_tpl is None or neg_tpl is None:
            print(f"[{modality}] no local dataset found under ../App/Models/ — skipping "
                  f"(mobile app will fall back to Roboflow-only for this modality)")
            continue
        os.makedirs(out_dir, exist_ok=True)
        offline_cv.Image.fromarray(pos_tpl).save(os.path.join(out_dir, "positive.png"))
        offline_cv.Image.fromarray(neg_tpl).save(os.path.join(out_dir, "negative.png"))
        print(f"[{modality}] wrote {out_dir}/positive.png + negative.png")
        any_built = True

    if not any_built:
        print("\nNo templates were built — check that ../App/Models/*/Data Set/ exists (run this "
              "from a full checkout, not a stripped-down copy).")
    else:
        print("\nDone. Re-run `buildozer android debug` to bundle these into the APK.")


if __name__ == "__main__":
    main()
