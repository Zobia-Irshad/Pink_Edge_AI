#!/usr/bin/env python3
"""
Pink Edge AI — offline, non-neural triage fallback (classical pixel-difference comparison).
================================================================================================
No trained model, no internet, no API key, no huggingface_hub/torch/ultralytics dependency at
all: builds a "typical positive" and "typical negative" reference template per modality by
averaging images from the project's own local labeled datasets (Models/<Modality>/Data
Set/*.coco — the same ones inference.py's Roboflow ground-truth checks use), then classifies a
new image by:
  1. grayscale + resize to a canonical size + histogram-equalize (normalizes exposure/contrast)
  2. auto-orientation: try identity vs. horizontal-flip, keep whichever correlates better with
     a generic (positive+negative averaged) reference — handles left/right laterality without
     full image registration
  3. pixel-wise absolute difference against the positive template AND the negative template
  4. a per-pixel "leans positive" map (diff-from-negative minus diff-from-positive); its largest
     connected region above threshold becomes the highlighted bounding box, and its mean/coverage
     become the confidence score

This is deliberately simple and fully explainable — a nearest-mean-template heuristic, not a
substitute for the trained models. It exists as a dependency-light, always-available offline
tier: see CALIBRATION.md-equivalent notes below and Documentations/MODEL_SOURCES.md for measured
accuracy against held-out samples from the same datasets — be honest with yourself about what
those numbers mean before trusting this over the real models.
"""
import json
import math
import os
import random

import numpy as np
from PIL import Image

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")
CANON_SIZE = 256
MAX_TEMPLATE_SAMPLES = 150
CONFIDENCE_SCALE = 6.0  # picked empirically — see calibrate() / MODEL_SOURCES.md

# Dataset location + category grounding per modality — reuses the exact taxonomies grounded in
# inference.py (see Documentations/MODEL_SOURCES.md) rather than re-guessing them.
_DATASETS = {
    "tb": {
        "models_subdir": "TB", "dataset_dir_name": "tuberculosis.coco",
        "positive_categories": ["Tüberküloz", "Hastalıklı"],
        "negative_categories": ["Sağlıklı", "Sekelli"],
    },
    "maternal": {
        "models_subdir": "Maternal", "dataset_dir_name": "HASH Maternal Health.coco",
        "positive_categories": ["abnormal"],
        "negative_categories": [],  # single-class dataset -> unannotated images are the implicit negative set
    },
    "mammography": {
        "models_subdir": "Mammography", "dataset_dir_name": "BreastCancer-YOLOv8.coco",
        "positive_categories": ["cancer"],
        "negative_categories": ["normal"],
    },
}

_TEMPLATE_CACHE = {}  # modality -> (positive_template, negative_template) | (None, None)


def _dataset_dir(modality):
    cfg = _DATASETS[modality]
    return os.path.join(MODELS_DIR, cfg["models_subdir"], "Data Set", cfg["dataset_dir_name"])


def _load_coco_splits(dataset_dir):
    for split in ("train", "valid", "test"):
        split_dir = os.path.join(dataset_dir, split)
        ann_path = os.path.join(split_dir, "_annotations.coco.json")
        if os.path.isfile(ann_path):
            with open(ann_path, "r", encoding="utf-8") as fh:
                yield split, split_dir, json.load(fh)


def _collect_image_paths(dataset_dir, positive_categories, negative_categories, exclude_split=None):
    """Returns (positive_paths, negative_paths). `exclude_split` ('test', etc.) lets calibrate()
    hold a split out of template-building so it can validate against unseen images."""
    positive, negative = set(), set()
    for split, split_dir, coco in _load_coco_splits(dataset_dir):
        if split == exclude_split:
            continue
        id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}
        cat_name_by_id = {c["id"]: c["name"] for c in coco["categories"]}
        pos_ids = {cid for cid, name in cat_name_by_id.items() if name in positive_categories}
        neg_ids = {cid for cid, name in cat_name_by_id.items() if name in negative_categories}
        annotated_ids = set()
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            annotated_ids.add(img_id)
            file_name = id_to_file.get(img_id)
            if file_name is None:
                continue
            path = os.path.join(split_dir, file_name)
            if ann["category_id"] in pos_ids:
                positive.add(path)
            elif ann["category_id"] in neg_ids:
                negative.add(path)
        if not negative_categories:
            for im in coco["images"]:
                if im["id"] not in annotated_ids:
                    negative.add(os.path.join(split_dir, im["file_name"]))
    return sorted(positive), sorted(negative)


def _preprocess_canonical(pil_image):
    import cv2

    gray = np.asarray(pil_image.convert("L").resize((CANON_SIZE, CANON_SIZE)), dtype=np.uint8)
    return cv2.equalizeHist(gray)


def _build_template(image_paths, seed):
    if not image_paths:
        return None
    rng = random.Random(seed)
    sample = image_paths if len(image_paths) <= MAX_TEMPLATE_SAMPLES else rng.sample(image_paths, MAX_TEMPLATE_SAMPLES)
    acc, n = np.zeros((CANON_SIZE, CANON_SIZE), dtype=np.float64), 0
    for p in sample:
        try:
            acc += _preprocess_canonical(Image.open(p)).astype(np.float64)
            n += 1
        except Exception:
            continue
    return (acc / n).astype(np.uint8) if n else None


def _get_templates(modality, exclude_split=None, use_cache=True):
    """`exclude_split` bypasses the on-disk cache (used only by calibrate() for a proper
    train/test split — normal predict calls always use the cached, all-data templates)."""
    if use_cache and modality in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[modality]

    cfg = _DATASETS.get(modality)
    if cfg is None:
        return None, None
    tpl_dir = os.path.join(MODELS_DIR, cfg["models_subdir"], "templates")
    pos_path, neg_path = os.path.join(tpl_dir, "positive.png"), os.path.join(tpl_dir, "negative.png")

    if use_cache and os.path.isfile(pos_path) and os.path.isfile(neg_path):
        result = (np.asarray(Image.open(pos_path).convert("L")), np.asarray(Image.open(neg_path).convert("L")))
        _TEMPLATE_CACHE[modality] = result
        return result

    dataset_dir = _dataset_dir(modality)
    if not os.path.isdir(dataset_dir):
        if use_cache:
            _TEMPLATE_CACHE[modality] = (None, None)
        return None, None

    pos_paths, neg_paths = _collect_image_paths(
        dataset_dir, cfg["positive_categories"], cfg["negative_categories"], exclude_split=exclude_split)
    pos_tpl, neg_tpl = _build_template(pos_paths, seed=1), _build_template(neg_paths, seed=2)

    if use_cache and pos_tpl is not None and neg_tpl is not None:
        os.makedirs(tpl_dir, exist_ok=True)
        Image.fromarray(pos_tpl).save(pos_path)
        Image.fromarray(neg_tpl).save(neg_path)
        with open(os.path.join(tpl_dir, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump({
                "positive_samples_available": len(pos_paths), "negative_samples_available": len(neg_paths),
                "positive_samples_used": min(len(pos_paths), MAX_TEMPLATE_SAMPLES),
                "negative_samples_used": min(len(neg_paths), MAX_TEMPLATE_SAMPLES),
            }, fh, indent=2)

    if use_cache:
        _TEMPLATE_CACHE[modality] = (pos_tpl, neg_tpl)
    return pos_tpl, neg_tpl


def _best_orientation(gray, reference):
    ref = reference.astype(np.float64)
    ref_c = ref - ref.mean()
    ref_norm = np.linalg.norm(ref_c) or 1.0
    best_img, best_name, best_score = gray, "identity", -np.inf
    for name, cand in (("identity", gray), ("hflip", np.fliplr(gray))):
        c = cand.astype(np.float64)
        c_c = c - c.mean()
        score = float(np.sum(ref_c * c_c) / (ref_norm * (np.linalg.norm(c_c) or 1.0)))
        if score > best_score:
            best_img, best_name, best_score = cand, name, score
    return best_img, best_name


def _analyze(modality, pil_image, positive_tpl, negative_tpl):
    """Core comparison. Returns (change_score, change_pct, bbox_normalized_xywh_or_None, orientation)."""
    import cv2

    gray = _preprocess_canonical(pil_image)
    generic_ref = ((positive_tpl.astype(np.float64) + negative_tpl.astype(np.float64)) / 2).astype(np.uint8)
    gray, orientation = _best_orientation(gray, generic_ref)

    diff_pos = np.abs(gray.astype(np.float64) - positive_tpl.astype(np.float64))
    diff_neg = np.abs(gray.astype(np.float64) - negative_tpl.astype(np.float64))
    lean_map = diff_neg - diff_pos  # >0 pixel looks more like the positive/abnormal template

    change_score = float(lean_map.mean())
    change_pct = float((lean_map > 0).mean() * 100)

    threshold = max(float(lean_map.std()) * 0.75, 3.0)
    mask = (lean_map > threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= 25:
            x, y, w, h = cv2.boundingRect(largest)
            bbox = (x / CANON_SIZE, y / CANON_SIZE, w / CANON_SIZE, h / CANON_SIZE)
    return change_score, change_pct, bbox, orientation


def _confidence(change_score):
    return 100.0 / (1.0 + math.exp(-change_score / CONFIDENCE_SCALE))


_MODALITY_VOCAB = {
    "tb": {
        "pos_verdict": "TB-Suggestive Finding Detected", "neg_verdict": "No Active TB-Suggestive Lesions Detected",
        "pos_bi_rads_by_tier": ["S1 - Minimal (unilateral, no cavity)", "S2 - Moderate (bilateral / cavity < 2 cm)",
                                 "S3 - Advanced (large cavity / miliary pattern)"],
        "neg_bi_rads": "S0 - No active disease", "acr_pos": "Upper Zone", "acr_neg": "Bilateral",
        "sms_pos": "TB:POS", "sms_neg": "TB:NEG",
    },
    "maternal": {
        "pos_verdict": "Standard Plane: Trans-thalamic", "neg_verdict": "Standard Plane: Trans-thalamic",
        "pos_bi_rads_by_tier": ["Standard Plane: Trans-thalamic", "Standard Plane: Trans-cerebellum", "Standard Plane: Trans-ventricular"],
        "neg_bi_rads": "Standard Plane: Trans-thalamic", "acr_pos": "Ultrasound Plane - Trans-thalamic",
        "acr_neg": "Ultrasound Plane - Trans-thalamic", "sms_pos": "US:PLANE_OK", "sms_neg": "US:PLANE_OK",
    },
    "mammography": {
        "pos_verdict": "Suspicious Finding Detected", "neg_verdict": "No Focal Suspicious Lesion Detected",
        "pos_bi_rads_by_tier": ["BI-RADS 4A - Low suspicion (Biopsy recommended)", "BI-RADS 4B - Moderate suspicion",
                                 "BI-RADS 5 - Highly suggestive of malignancy"],
        "neg_bi_rads": "BI-RADS 1 - Negative", "acr_pos": "C - Heterogeneously dense",
        "acr_neg": "B - Scattered fibroglandular density", "sms_pos": "BI-RADS:5", "sms_neg": "BI-RADS:1",
    },
}


def _to_result(modality, confidence, is_positive, bbox):
    v = _MODALITY_VOCAB[modality]
    source = f"Offline pixel-comparison heuristic vs. local {modality} dataset (no model, no internet)"
    if is_positive:
        tier = 2 if confidence >= 85 else (1 if confidence >= 65 else 0)
        return {
            "bi_rads": v["pos_bi_rads_by_tier"][tier], "acr": v["acr_pos"], "verdict": v["pos_verdict"],
            "sub": "Offline heuristic: image pattern leans toward the positive reference set",
            "css": "danger", "loc": "See bounding box", "extra": v["pos_bi_rads_by_tier"][tier],
            "vicon": "⚠️", "confidence": confidence, "sms": v["sms_pos"], "is_critical": True,
            "source": source, "bbox": bbox,
        }
    return {
        "bi_rads": v["neg_bi_rads"], "acr": v["acr_neg"], "verdict": v["neg_verdict"],
        "sub": "Offline heuristic: image pattern leans toward the negative/healthy reference set",
        "css": "success", "loc": "No notable deviation found", "extra": v["neg_bi_rads"],
        "vicon": "✅", "confidence": confidence, "sms": v["sms_neg"], "is_critical": False,
        "source": source, "bbox": bbox,
    }


def available(modality: str) -> bool:
    pos, neg = _get_templates(modality)
    return pos is not None and neg is not None


def predict(modality: str, pil_image: Image.Image):
    """modality: 'tb' | 'maternal' | 'mammography'. Returns a result-dict (with an extra `bbox`
    field — normalized (x, y, w, h) fractions, or None) or None if no local dataset is present
    to build templates from."""
    positive_tpl, negative_tpl = _get_templates(modality)
    if positive_tpl is None or negative_tpl is None:
        return None
    change_score, change_pct, bbox, orientation = _analyze(modality, pil_image, positive_tpl, negative_tpl)
    confidence = _confidence(change_score)
    is_positive = confidence >= 50.0
    result = _to_result(modality, confidence if is_positive else 100.0 - confidence, is_positive, bbox)
    result["sub"] += f" ({change_pct:.1f}% of pixels leaning positive, orientation={orientation})"
    return result


# ============================================================
# Calibration / self-test — not used at runtime, run directly to (re)validate:
#   python offline_cv.py
# Builds templates from train+valid only, tests against the held-out `test` split, reports real
# accuracy so this method's confidence isn't just taken on faith.
# ============================================================
def calibrate(modality: str, max_per_class=25):
    cfg = _DATASETS[modality]
    dataset_dir = _dataset_dir(modality)
    pos_tpl, neg_tpl = _get_templates(modality, exclude_split="test", use_cache=False)
    if pos_tpl is None or neg_tpl is None:
        print(f"  [{modality}] no dataset found, skipping")
        return

    # Held-out test-split images only, by ground-truth category.
    test_dir = os.path.join(dataset_dir, "test")
    ann_path = os.path.join(test_dir, "_annotations.coco.json")
    if not os.path.isfile(ann_path):
        print(f"  [{modality}] no test split, skipping")
        return
    with open(ann_path, "r", encoding="utf-8") as fh:
        coco = json.load(fh)
    id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}
    cat_name_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    pos_ids = {cid for cid, n in cat_name_by_id.items() if n in cfg["positive_categories"]}
    neg_ids = {cid for cid, n in cat_name_by_id.items() if n in cfg["negative_categories"]}
    annotated = set()
    pos_files, neg_files = [], []
    for ann in coco["annotations"]:
        annotated.add(ann["image_id"])
        fn = id_to_file.get(ann["image_id"])
        if fn is None:
            continue
        if ann["category_id"] in pos_ids:
            pos_files.append(fn)
        elif ann["category_id"] in neg_ids:
            neg_files.append(fn)
    if not cfg["negative_categories"]:
        neg_files = [im["file_name"] for im in coco["images"] if im["id"] not in annotated]

    rng = random.Random(0)
    pos_files = rng.sample(pos_files, min(max_per_class, len(pos_files)))
    neg_files = rng.sample(neg_files, min(max_per_class, len(neg_files)))

    def _run(files, expect_positive):
        correct = 0
        for fn in files:
            img = Image.open(os.path.join(test_dir, fn))
            score, _, _, _ = _analyze(modality, img, pos_tpl, neg_tpl)
            predicted_positive = _confidence(score) >= 50.0
            correct += predicted_positive == expect_positive
        return correct, len(files)

    pc, pn = _run(pos_files, True)
    nc, nn = _run(neg_files, False)
    total_c, total_n = pc + nc, pn + nn
    print(f"  [{modality}] positive: {pc}/{pn} correct | negative: {nc}/{nn} correct | "
          f"overall: {total_c}/{total_n} ({100 * total_c / total_n:.0f}%)")


if __name__ == "__main__":
    print("Offline CV heuristic — calibration against held-out test-split data (not used at runtime):")
    for m in _DATASETS:
        calibrate(m)
