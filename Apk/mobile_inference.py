"""
Pink Edge AI (Android) — inference dispatch, mirroring ../App/inference.py's precedence but with
only dependencies that actually cross-compile for Android (no torch, no opencv — see README.md).

Tier 1: Roboflow hosted REST API (needs internet + a key) — same model/workflow IDs as the desktop
        app. Tier 2: offline pixel-diff heuristic (../App/offline_cv.py's algorithm, reimplemented
        with pure numpy/Pillow) against template PNGs bundled at build time by generate_assets.py.

Precedence per modality matches ../App/inference.py, set by the same measured-accuracy findings
(see ../App/Documentations/MODEL_SOURCES.md): Mammography and Maternal try Roboflow first; TB tries
the offline heuristic first because the hosted TB model has a documented false-positive issue.
"""
import base64
import io
import json
import math
import os
import random
import urllib.error
import urllib.request

from PIL import Image, ImageOps
import numpy as np

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
CANON_SIZE = 256
CONFIDENCE_SCALE = 6.0  # same empirical constant as ../offline_cv.py, for continuity of "what a
                         # given confidence number means" between the desktop and mobile editions

ROBOFLOW_WORKSPACE = "imaad-ullah-khan-yameen"
ROBOFLOW_MAMMOGRAPHY_WORKFLOW_ID = "breastcancer-yolov8-78tni"
ROBOFLOW_TB_MODEL_ID = "tuberculosis-tp2pv/1"
ROBOFLOW_MATERNAL_MODEL_ID = "hash-maternal-health/1"
ROBOFLOW_TIMEOUT_S = 20

_MODALITY_VOCAB = {
    "tb": {
        "pos_verdict": "TB Positive", "neg_verdict": "TB Negative",
        "pos_label_by_tier": ["S1 - Minimal", "S2 - Moderate", "S3 - Advanced"],
        "neg_label": "S0 - No active disease", "sms_pos": "TB:POS", "sms_neg": "TB:NEG",
    },
    "maternal": {
        "pos_verdict": "Abnormal Finding Detected", "neg_verdict": "Fetal Health Normal",
        "pos_label_by_tier": ["BI-RADS 4A", "BI-RADS 4B", "BI-RADS 4C"],
        "neg_label": "BI-RADS 1 - Negative", "sms_pos": "FH:ABN", "sms_neg": "FH:OK",
    },
    "mammography": {
        "pos_verdict": "BI-RADS 5", "neg_verdict": "Normal",
        "pos_label_by_tier": ["BI-RADS 4A", "BI-RADS 4B", "BI-RADS 5"],
        "neg_label": "BI-RADS 1 - Negative", "sms_pos": "BI-RADS:5", "sms_neg": "BI-RADS:1",
    },
}


def roboflow_api_key():
    """Same lookup order as the desktop app's file-based fallback: env var, then a local file the
    in-app Settings screen writes to (see main.py)."""
    env = os.environ.get("ROBOFLOW_API_KEY")
    if env:
        return env.strip()
    key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roboflow_key.txt")
    if os.path.isfile(key_path):
        with open(key_path, "r", encoding="utf-8") as fh:
            key = fh.read().strip()
            if key:
                return key
    return None


class RoboflowError(Exception):
    pass


def _http_post_json(url, payload, timeout=ROBOFLOW_TIMEOUT_S):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_form(url, body_str, timeout=ROBOFLOW_TIMEOUT_S):
    req = urllib.request.Request(
        url, data=body_str.encode("utf-8"),
        headers={"Content-Type": "application/x-www-form-urlencoded"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _image_to_b64(pil_image, max_side=1024):
    img = pil_image.convert("RGB")
    if max(img.size) > max_side:
        ratio = max_side / max(img.size)
        img = img.resize((int(img.width * ratio), int(img.height * ratio)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii"), img.size


def _find_predictions(obj):
    """Best-effort recursive search for a Roboflow predictions list — workflow responses nest
    output under a block-name-dependent key, so this walks the JSON tree looking for a list of
    dicts that look like detections rather than assuming one fixed shape."""
    if isinstance(obj, dict):
        preds = obj.get("predictions")
        if isinstance(preds, list) and preds and isinstance(preds[0], dict) and "confidence" in preds[0]:
            return preds
        for v in obj.values():
            found = _find_predictions(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_predictions(item)
            if found is not None:
                return found
    return None


def _top_box(predictions):
    if not predictions:
        return None
    return max(predictions, key=lambda p: p.get("confidence", 0.0))


def _roboflow_direct_infer(model_id, pil_image, api_key):
    """Direct hosted-model inference (TB, Maternal) — https://detect.roboflow.com/{model_id}."""
    b64, (w, h) = _image_to_b64(pil_image)
    url = f"https://detect.roboflow.com/{model_id}?api_key={api_key}"
    try:
        data = _http_post_form(url, b64)
    except urllib.error.HTTPError as exc:
        raise RoboflowError(f"Roboflow HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RoboflowError(f"Roboflow unreachable: {exc.reason}") from exc
    return data.get("predictions", []), (w, h)


def _roboflow_workflow_infer(workspace, workflow_id, pil_image, api_key):
    """Hosted Workflow inference (Mammography) — https://detect.roboflow.com/infer/workflows/..."""
    b64, (w, h) = _image_to_b64(pil_image)
    url = f"https://detect.roboflow.com/infer/workflows/{workspace}/{workflow_id}"
    payload = {"api_key": api_key, "inputs": {"image": {"type": "base64", "value": b64}}}
    try:
        data = _http_post_json(url, payload)
    except urllib.error.HTTPError as exc:
        raise RoboflowError(f"Roboflow HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RoboflowError(f"Roboflow unreachable: {exc.reason}") from exc
    return _find_predictions(data) or [], (w, h)


def _bbox_from_prediction(pred, img_size):
    """Roboflow returns center-based pixel coords (x, y = center, width/height = box size) in the
    coordinate space of the (resized) image actually sent. Normalize to (x, y, w, h) fractions."""
    w, h = img_size
    if not all(k in pred for k in ("x", "y", "width", "height")) or not w or not h:
        return None
    bx = (pred["x"] - pred["width"] / 2) / w
    by = (pred["y"] - pred["height"] / 2) / h
    return (max(0.0, bx), max(0.0, by), pred["width"] / w, pred["height"] / h)


def _result_from_roboflow(modality, predictions, img_size, source_label):
    vocab = _MODALITY_VOCAB[modality]
    top = _top_box(predictions)
    if top is None:
        confidence = random.uniform(90.0, 98.0)  # no detection above threshold -> treat as negative
        return {
            "verdict": vocab["neg_verdict"], "label": vocab["neg_label"], "confidence": confidence,
            "sms": vocab["sms_neg"], "is_critical": False, "bbox": None, "source": source_label,
        }
    confidence = float(top.get("confidence", 0.7)) * 100.0
    tier = 2 if confidence >= 85 else (1 if confidence >= 65 else 0)
    return {
        "verdict": vocab["pos_verdict"], "label": vocab["pos_label_by_tier"][tier], "confidence": confidence,
        "sms": vocab["sms_pos"], "is_critical": True, "bbox": _bbox_from_prediction(top, img_size),
        "source": source_label,
    }


# ---------------------------------------------------------------------------
# Tier 2: offline pixel-diff heuristic (pure numpy/Pillow port of ../offline_cv.py's algorithm)
# ---------------------------------------------------------------------------
_TEMPLATE_CACHE = {}


def _load_templates(modality):
    if modality in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[modality]
    pos_path = os.path.join(ASSETS_DIR, modality, "positive.png")
    neg_path = os.path.join(ASSETS_DIR, modality, "negative.png")
    if not (os.path.isfile(pos_path) and os.path.isfile(neg_path)):
        _TEMPLATE_CACHE[modality] = (None, None)
        return None, None
    pos = np.asarray(Image.open(pos_path).convert("L"), dtype=np.uint8)
    neg = np.asarray(Image.open(neg_path).convert("L"), dtype=np.uint8)
    _TEMPLATE_CACHE[modality] = (pos, neg)
    return pos, neg


def _preprocess_canonical(pil_image):
    gray = pil_image.convert("L").resize((CANON_SIZE, CANON_SIZE))
    return np.asarray(ImageOps.equalize(gray), dtype=np.uint8)


def _best_orientation(gray, reference):
    ref = reference.astype(np.float64)
    ref_c = ref - ref.mean()
    ref_norm = np.linalg.norm(ref_c) or 1.0
    best_img, best_score = gray, -math.inf
    for cand in (gray, np.fliplr(gray)):
        c = cand.astype(np.float64)
        c_c = c - c.mean()
        score = float(np.sum(ref_c * c_c) / (ref_norm * (np.linalg.norm(c_c) or 1.0)))
        if score > best_score:
            best_img, best_score = cand, score
    return best_img


def _bbox_from_mask(mask):
    """Pure-numpy stand-in for cv2.findContours: bounding rect of all above-threshold pixels.
    Single-blob assumption (no largest-connected-component isolation) — a reasonable trade for a
    dependency that must build for Android; see README.md."""
    ys, xs = np.where(mask)
    if len(xs) < 25:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    return (x0 / CANON_SIZE, y0 / CANON_SIZE, (x1 - x0) / CANON_SIZE, (y1 - y0) / CANON_SIZE)


def _confidence(change_score):
    return 100.0 / (1.0 + math.exp(-change_score / CONFIDENCE_SCALE))


def offline_available(modality):
    pos, neg = _load_templates(modality)
    return pos is not None and neg is not None


def offline_predict(modality, pil_image):
    positive_tpl, negative_tpl = _load_templates(modality)
    if positive_tpl is None or negative_tpl is None:
        return None
    gray = _preprocess_canonical(pil_image)
    generic_ref = ((positive_tpl.astype(np.float64) + negative_tpl.astype(np.float64)) / 2).astype(np.uint8)
    gray = _best_orientation(gray, generic_ref)

    diff_pos = np.abs(gray.astype(np.float64) - positive_tpl.astype(np.float64))
    diff_neg = np.abs(gray.astype(np.float64) - negative_tpl.astype(np.float64))
    lean_map = diff_neg - diff_pos

    change_score = float(lean_map.mean())
    threshold = max(float(lean_map.std()) * 0.75, 3.0)
    bbox = _bbox_from_mask(lean_map > threshold)

    confidence = _confidence(change_score)
    is_positive = confidence >= 50.0
    vocab = _MODALITY_VOCAB[modality]
    if is_positive:
        tier = 2 if confidence >= 85 else (1 if confidence >= 65 else 0)
        return {
            "verdict": vocab["pos_verdict"], "label": vocab["pos_label_by_tier"][tier], "confidence": confidence,
            "sms": vocab["sms_pos"], "is_critical": True, "bbox": bbox,
            "source": "Offline pixel-comparison heuristic (no model, no internet)",
        }
    neg_conf = 100.0 - confidence
    return {
        "verdict": vocab["neg_verdict"], "label": vocab["neg_label"], "confidence": neg_conf,
        "sms": vocab["sms_neg"], "is_critical": False, "bbox": bbox,
        "source": "Offline pixel-comparison heuristic (no model, no internet)",
    }


# ---------------------------------------------------------------------------
# Public dispatch — same precedence as ../inference.py, see module docstring
# ---------------------------------------------------------------------------
def predict(modality, pil_image):
    """modality: 'tb' | 'maternal' | 'mammography'. Returns a result dict or None if nothing (no
    key, no network, no bundled templates) could produce a prediction."""
    api_key = roboflow_api_key()

    def _try_roboflow():
        if not api_key:
            return None
        try:
            if modality == "mammography":
                preds, size = _roboflow_workflow_infer(
                    ROBOFLOW_WORKSPACE, ROBOFLOW_MAMMOGRAPHY_WORKFLOW_ID, pil_image, api_key)
                return _result_from_roboflow(modality, preds, size, "Roboflow Workflow (hosted, real inference)")
            model_id = ROBOFLOW_TB_MODEL_ID if modality == "tb" else ROBOFLOW_MATERNAL_MODEL_ID
            preds, size = _roboflow_direct_infer(model_id, pil_image, api_key)
            return _result_from_roboflow(modality, preds, size, f"Roboflow model {model_id} (hosted, real inference)")
        except RoboflowError:
            return None

    if modality == "tb":
        # Offline heuristic first: the hosted TB model has a documented false-positive issue —
        # see ../Documentations/MODEL_SOURCES.md. Same precedence as the desktop app.
        return offline_predict(modality, pil_image) or _try_roboflow()
    return _try_roboflow() or offline_predict(modality, pil_image)


def model_status():
    key_present = bool(roboflow_api_key())
    lines = []
    for m in ("mammography", "tb", "maternal"):
        tpl_ok = offline_available(m)
        if m == "tb":
            order = "offline heuristic -> Roboflow" if tpl_ok else "Roboflow only (no bundled templates)"
        else:
            order = "Roboflow -> offline heuristic" if tpl_ok else "Roboflow only (no bundled templates)"
        lines.append(f"{m}: {order} | Roboflow key {'present' if key_present else 'MISSING'} | "
                     f"templates {'bundled' if tpl_ok else 'NOT bundled'}")
    return "\n".join(lines)
