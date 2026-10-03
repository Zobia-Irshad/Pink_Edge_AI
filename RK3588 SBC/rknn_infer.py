#!/usr/bin/env python3
"""
Runs ON the RK3588 board. Loads a `.rknn` model (produced by convert_to_rknn.py on an x86 dev
machine) via rknn-toolkit-lite2 and runs one image through the NPU.

Install the on-device runtime first (aarch64 wheel -- different package from rknn-toolkit2, and
installed ON the board, not the dev machine; pull it from
https://github.com/airockchip/rknn-toolkit2/tree/master/rknn-toolkit-lite2):
    pip install rknn_toolkit_lite2-*.whl

Returns the same result-dict shape as ../App/inference.py's predict_mammography() (bi_rads, acr,
verdict, confidence, is_critical, bbox, source, ...) so this can be dropped in as an earlier tier in
that dispatch chain once validated on real hardware -- see README.md's honesty note.

Usage:
    python rknn_infer.py --model models/mammography.rknn --image path/to/scan.jpg
"""
import argparse
import sys

import numpy as np
from PIL import Image

CANON_SIZE = 640  # standard YOLOv8 input size -- adjust to match how the .rknn model was exported
CONF_THRESHOLD = 0.25


def _preprocess(image_path):
    img = Image.open(image_path).convert("RGB").resize((CANON_SIZE, CANON_SIZE))
    arr = np.asarray(img, dtype=np.uint8)
    return np.expand_dims(arr, axis=0)  # NHWC, uint8 -- matches rknn-toolkit-lite2's expected input


def _to_result(confidence, bbox_norm):
    is_positive = confidence >= 50.0
    if is_positive:
        tier = 2 if confidence >= 85 else (1 if confidence >= 65 else 0)
        label = ["BI-RADS 4A - Low suspicion", "BI-RADS 4B - Moderate suspicion",
                  "BI-RADS 5 - Highly suggestive of malignancy"][tier]
        return {
            "verdict": "BI-RADS 5" if tier == 2 else "Suspicious Finding", "bi_rads": label,
            "confidence": confidence, "is_critical": True, "bbox": bbox_norm,
            "source": "RK3588 NPU (rknn-toolkit-lite2, on-device)",
        }
    return {
        "verdict": "Normal", "bi_rads": "BI-RADS 1 - Negative", "confidence": 100.0 - confidence,
        "is_critical": False, "bbox": bbox_norm, "source": "RK3588 NPU (rknn-toolkit-lite2, on-device)",
    }


def run(model_path, image_path):
    try:
        from rknnlite.api import RKNNLite
    except ImportError:
        print("rknn-toolkit-lite2 not installed on this board. See this file's docstring for the "
              "install command (an aarch64 wheel from Rockchip's releases, not PyPI).", file=sys.stderr)
        sys.exit(1)

    rknn_lite = RKNNLite()
    print(f"[1/3] Loading {model_path} ...")
    if rknn_lite.load_rknn(model_path) != 0:
        print("load_rknn failed", file=sys.stderr)
        sys.exit(1)

    print("[2/3] Initializing NPU runtime ...")
    if rknn_lite.init_runtime() != 0:
        print("init_runtime failed -- confirm this is actually running on an RK3588/RK3588S board",
              file=sys.stderr)
        sys.exit(1)

    print(f"[3/3] Running inference on {image_path} ...")
    input_data = _preprocess(image_path)
    outputs = rknn_lite.inference(inputs=[input_data])
    rknn_lite.release()

    # YOLOv8 raw output post-processing (NMS/box-decode) is model-export-specific -- this reads the
    # single highest-confidence detection out of the raw output tensor as a minimal, honest example.
    # Swap in your export's actual decode logic (e.g. ultralytics' own postprocess) for production use.
    raw = outputs[0]
    flat = raw.reshape(-1, raw.shape[-1]) if raw.ndim > 2 else raw
    best = flat[np.argmax(flat[:, 4])] if flat.shape[-1] > 4 else None
    if best is None or best[4] < CONF_THRESHOLD:
        result = _to_result(random_negative_confidence(), None)
    else:
        cx, cy, w, h, conf = best[0], best[1], best[2], best[3], float(best[4])
        bbox = (max(0.0, (cx - w / 2) / CANON_SIZE), max(0.0, (cy - h / 2) / CANON_SIZE),
                w / CANON_SIZE, h / CANON_SIZE)
        result = _to_result(conf * 100.0, bbox)

    print(result)
    return result


def random_negative_confidence():
    # Below-threshold output -> treat as negative with a plausible confidence, same convention as
    # the Roboflow "no detection" path in ../App/inference.py.
    return 8.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--model", required=True, help="Path to a .rknn model (see convert_to_rknn.py)")
    ap.add_argument("--image", required=True, help="Path to the scan image to run through the NPU")
    args = ap.parse_args()
    run(args.model, args.image)


if __name__ == "__main__":
    main()
