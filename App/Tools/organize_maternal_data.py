#!/usr/bin/env python3
"""
Builds the Maternal Health training set: distributes every usable image from this project's two
raw maternal sources into three severity classes -- `Good Health/`, `Med Health/`, `Bad Health/`
-- plus a held-out `validate/` copy of the same three, under Models/Maternal/.

Author Name:  Imaad Ullah Khan
Author Email: yameenimaad@gmail.com
AI Helper:    Claude

Run:  python Tools/organize_maternal_data.py            (from App/)
      python Tools/organize_maternal_data.py --dry-run  (report the plan, write nothing)

------------------------------------------------------------------------------------------------
THE THREE CLASSES, AND WHERE THEIR LABELS COME FROM
------------------------------------------------------------------------------------------------
  Good Health  normal fetal brain, routine screening        <- FETAL_PLANES_ZENODO
  Med Health   abnormal, warrants follow-up/monitoring      <- HASH Maternal Health.coco
  Bad Health   abnormal, warrants urgent referral           <- HASH Maternal Health.coco

READ THIS BEFORE TRUSTING THE Med/Bad BOUNDARY. The COCO source carries exactly one label,
`abnormal`, for all 1,384 of its images. It has no severity field. The Med/Bad split is therefore
NOT read from the dataset -- it is applied here by SEVERITY_MAP below, keyed on the pathology name
parsed out of each filename ('severe-ventriculomegaly-18b_aug_3_...jpg' -> severe ventriculomegaly
-> Bad). The grouping into follow-up vs urgent is a clinical convention, not a radiologist's
reading of these particular images, and the filenames themselves are inconsistently spelled
(`cerebellah-`, `anold-chiari`, `intracnanial-`, `hydracenphay`), which SEVERITY_MAP normalises.
Any model trained on this inherits that mapping: it predicts the severity convention encoded here,
not a verified clinical grade.

`Good Health` has a second caveat of its own: it comes from a different dataset than the other two
classes, so a model can score well by recognising which dataset an image came from rather than by
reading anatomy. That is what the normalisation below and the control test in
Tests/dataset_integrity_test.py exist for. Read the control number before trusting any accuracy.

------------------------------------------------------------------------------------------------
THE IMAGE PIPELINE, AND WHY EACH STEP IS THERE
------------------------------------------------------------------------------------------------
Applied identically to all three classes:

  1. grayscale            ultrasound carries no colour; drops Zenodo's RGBA vs COCO's RGB
  2. centre-crop 74%      discards the burnt-in scanner overlay and the sector-fan edges (below)
  3. resize to 224        one resolution for every class
  4. histogram equalise   removes the global brightness/contrast gap between the two sources
  5. COLORMAP (turbo)     depth perception -- maps intensity to hue so subtle tissue-density
                          gradients separate visually; on a dilated ventricle the fluid reads as
                          deep blue against red/orange parenchyma instead of dark-grey on grey.
                          It also fills all three input channels with real signal rather than the
                          same grey repeated, which suits the ImageNet-pretrained backbone.
  6. contrast x1.5        widens the hue separation the colormap just created
  7. JPEG q90             one encoder, one quality, so compression artefacts carry no label

Steps 2 and 4 are not cosmetic. Side by side, the Zenodo normals carry a scanner overlay in the
top-right corner ("60/61 22Hz") and a hard sector-fan edge that the COCO abnormals do not, and the
COCO images are heavily brightness-augmented. Without the crop and the equalisation a CNN learns
"corner text and fan edge => Good Health" in about one epoch and reports ~99% while reading none of
the anatomy. Steps 5 and 6 then run on an image from which those cues are already gone.

------------------------------------------------------------------------------------------------
TWO DEFECTS IN THE RAW SOURCES THAT THE SPLIT HAS TO REPAIR
------------------------------------------------------------------------------------------------
1. THE SHIPPED COCO SPLIT LEAKS ALMOST COMPLETELY. Its 1,384 images are augmentations of only 106
   distinct base images. 104 of valid's 104 base images also appear in train, and 52 of test's 54
   appear in valid -- so nearly every "held-out" image is a recoloured copy of something trained
   on, and accuracy against it measures memorisation. The shipped boundaries are ignored and
   re-derived here over GROUPS: positives group by base image, Good Health groups by Zenodo patient
   number. A group lands wholly in train or wholly in validate.

2. IDENTICAL IMAGES APPEAR UNDER DIFFERENT NAMES. One COCO scan is filed under two different
   pathologies, and Zenodo repeats 100 images across different patient numbers. Group-splitting
   alone does not catch this, so images are de-duplicated on the content hash of the *normalised*
   image, and any two groups sharing an image are merged so they cannot straddle the split.

Provenance for every written file -- source, group, pathology, class, split -- is recorded in
dataset_manifest.json next to the folders.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import offline_cv  # the single definition of the image pipeline and the class list

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATERNAL_DIR = os.path.join(APP_DIR, "Models", "Maternal")
DATA_SET_DIR = os.path.join(MATERNAL_DIR, "Data Set")
COCO_DIR = os.path.join(DATA_SET_DIR, "HASH Maternal Health.coco")
ZENODO_DIR = os.path.join(DATA_SET_DIR, "FETAL_PLANES_ZENODO")
ZENODO_CSV = os.path.join(ZENODO_DIR, "FETAL_PLANES_DB_data.csv")
ZENODO_IMAGES = os.path.join(ZENODO_DIR, "Images")

GOOD, MED, BAD = tuple(offline_cv.class_names("maternal"))
CLASSES = (GOOD, MED, BAD)          # ordered least -> most severe; the model's class indices

# Steps 1-6 are configured in offline_cv._DATASETS["maternal"]["preprocess"] and applied by
# offline_cv.preprocess_for_model(), so that inference runs the identical pipeline. --colormap and
# --contrast below write into that config rather than shadowing it.
_PRE = offline_cv._DATASETS["maternal"]["preprocess"]
JPEG_QUALITY = 90                   # step 7, only relevant when writing the dataset to disk

NORMAL_BRAIN_PLANES = ("Trans-thalamic", "Trans-ventricular", "Trans-cerebellum")

# Pathology -> severity. Keys are matched as substrings against the filename's pathology stem after
# the spelling fixes in _pathology_of(). Order matters: the first match wins, so the more specific
# 'severe-ventriculomegaly' must be tested before 'ventriculomegaly'.
SEVERITY_MAP = (
    # --- urgent referral: destructive, structural or space-occupying lesions ---
    ("severe-ventriculomegaly", BAD),
    ("holoprosencephaly", BAD),
    ("hydranencephaly", BAD),
    ("porencephaly", BAD),
    ("encephalocele", BAD),
    ("cerebellar-hypoplasia", BAD),
    ("arnold-chiari", BAD),
    ("colpocephaly", BAD),
    ("intracranial-hemorrhage", BAD),
    ("intracranial-tumor", BAD),
    # --- follow-up / monitoring: milder or commonly benign findings ---
    ("mild-ventriculomegaly", MED),
    ("moderate-ventriculomegaly", MED),
    ("arachnoid-cyst", MED),
    ("mega-cisterna-magna", MED),
)

# The COCO filenames are inconsistently spelled; normalise before matching SEVERITY_MAP.
SPELLING_FIXES = (
    ("cerebellah-hypoplasia", "cerebellar-hypoplasia"),
    ("anold-chiari-maformation", "arnold-chiari"),
    ("anold-chiari-malformation", "arnold-chiari"),
    ("intracnanial-hemorrdge", "intracranial-hemorrhage"),
    ("intracranial-hemorrdge", "intracranial-hemorrhage"),
    ("hydracenphaly", "hydranencephaly"),
    ("hydracenphay", "hydranencephaly"),
    ("colphocephaly", "colpocephaly"),
    ("polencephaly", "porencephaly"),
    ("m-magna", "mega-cisterna-magna"),
)

VALIDATE_FRACTION = 0.25   # of GROUPS, not of images
SEED = 20251003


def class_dir(cls, split="train"):
    return os.path.join(MATERNAL_DIR, cls) if split == "train" \
        else os.path.join(MATERNAL_DIR, "validate", cls)


def _base_image(filename):
    """The distinct source scan an augmentation came from.
    'mild-ventriculomegaly-18b_aug_3_png_jpg.rf.<hash>.jpg' -> 'mild-ventriculomegaly-18b'."""
    return re.sub(r"_aug_\d+.*$", "", os.path.basename(filename))


def _pathology_of(base):
    """Pathology name from a base image name, with the source's misspellings normalised.
    'severe-ventriculomegaly-18b' -> 'severe-ventriculomegaly'."""
    name = base.lower()
    for wrong, right in SPELLING_FIXES:
        name = name.replace(wrong, right)
    # strip the trailing case identifier: '-18b', '24a', '-26a-image', '_'
    name = re.sub(r"[-_]?\d+[a-z]?([-_]image)?[-_]?\d*$", "", name).strip("-_ ")
    return name


def severity_of(base):
    """Which class a COCO base image belongs to, or None if its pathology is unmapped."""
    pathology = _pathology_of(base)
    for keyword, cls in SEVERITY_MAP:
        if keyword in pathology:
            return cls, pathology
    return None, pathology


def collect_abnormal():
    """Every `abnormal` COCO image, keyed by (class, base image). The shipped train/valid/test
    boundaries are deliberately ignored and re-derived later -- they leak (defect 1)."""
    groups, unmapped = defaultdict(list), Counter()
    for split in ("train", "valid", "test"):
        ann_path = os.path.join(COCO_DIR, split, "_annotations.coco.json")
        if not os.path.isfile(ann_path):
            continue
        with open(ann_path, "r", encoding="utf-8") as fh:
            coco = json.load(fh)
        abnormal_ids = {c["id"] for c in coco["categories"] if c["name"].lower() == "abnormal"}
        annotated = {a["image_id"] for a in coco["annotations"] if a["category_id"] in abnormal_ids}
        for img in coco["images"]:
            if img["id"] not in annotated:
                continue
            path = os.path.join(COCO_DIR, split, img["file_name"])
            if not os.path.isfile(path):
                continue
            base = _base_image(img["file_name"])
            cls, pathology = severity_of(base)
            if cls is None:
                unmapped[pathology] += 1
                continue
            groups[(cls, base)].append(path)
    if unmapped:
        print(f"  note: {sum(unmapped.values())} image(s) skipped, pathology not in SEVERITY_MAP: "
              f"{dict(unmapped)}")
    return groups


def collect_good():
    """Normal fetal-brain planes from Zenodo, keyed by patient -- several images of one mother are
    near-duplicates, so the patient is the group, not the image."""
    groups = defaultdict(list)
    if not os.path.isfile(ZENODO_CSV):
        return groups
    with open(ZENODO_CSV, "r", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh, delimiter=";"))
    header = [h.strip() for h in rows[0]]
    i_name, i_patient = header.index("Image_name"), header.index("Patient_num")
    i_plane = header.index("Brain_plane")
    for row in rows[1:]:
        if len(row) <= max(i_name, i_patient, i_plane):
            continue
        if row[i_plane].strip() not in NORMAL_BRAIN_PLANES:
            continue
        path = os.path.join(ZENODO_IMAGES, row[i_name].strip() + ".png")
        if os.path.isfile(path):
            groups[(GOOD, f"patient{row[i_patient].strip()}")].append(path)
    return groups


def normalised_bytes(src_path):
    """Steps 1-6 of the pipeline, then JPEG (step 7). Returns encoded bytes rather than writing, so
    the caller can hash the normalised image to find duplicates before deciding where it goes.

    Steps 1-6 are NOT implemented here. They live in offline_cv.preprocess_for_model(), which
    inference.py also calls before running the trained weights -- one definition, so the images
    this builds the model on and the images the app later feeds it cannot drift apart."""
    from PIL import Image

    with Image.open(src_path) as im:
        rgb = offline_cv.preprocess_for_model("maternal", im)            # steps 1-6
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=JPEG_QUALITY)                          # step 7
    return buf.getvalue()


def _colormap_choices():
    import cv2
    return sorted(k for k in ("turbo", "jet", "inferno", "viridis", "bone", "magma")
                  if hasattr(cv2, f"COLORMAP_{k.upper()}"))


def dedupe_and_merge(groups):
    """Normalise every image once, then repair defect 2 using the content hash of the normalised
    image: an image appearing in two groups means they are the same underlying scan, so the groups
    are MERGED and will land on the same side of the split (dropping the duplicate alone would not
    do -- the other augmentations of that scan are near-identical and would still straddle it).
    A duplicate inside one group is dropped."""
    blobs, owner, parent = {}, {}, {}

    def find(g):
        while parent.setdefault(g, g) != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    n_dupe, n_merged = 0, 0
    for group in sorted(groups):
        find(group)
        for path in sorted(groups[group]):
            try:
                data = normalised_bytes(path)
            except Exception as e:
                print(f"  warn: unreadable, skipped {os.path.basename(path)} ({e})")
                continue
            digest = hashlib.sha256(data).hexdigest()
            if digest in owner:
                other = owner[digest]
                n_dupe += 1
                if find(other) != find(group):
                    union(other, group)
                    n_merged += 1
                continue
            owner[digest] = group
            blobs[path] = data

    merged = defaultdict(list)
    for group, paths in groups.items():
        for path in paths:
            if path in blobs:
                merged[find(group)].append(path)
    stats = {"duplicates_dropped": n_dupe, "group_pairs_merged": n_merged,
             "groups_before": len(groups), "groups_after": len(merged)}
    print(f"  dropped {n_dupe} duplicate image(s); merged {n_merged} group pair(s) sharing an "
          f"identical image ({stats['groups_before']} -> {stats['groups_after']} groups)")
    return merged, blobs, stats


def _clear(path):
    """Empty a destination folder. Everything this script writes is derived from Data Set/, so a
    rebuild is lossless -- but anything NOT matching the generated naming is kept and reported, so
    a manually added image is never silently destroyed."""
    kept = []
    if not os.path.isdir(path):
        return kept
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if os.path.isfile(full):
            if re.match(r"^(good|med|bad)_", name):
                os.remove(full)
            else:
                kept.append(name)
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    ap.add_argument("--validate-fraction", type=float, default=VALIDATE_FRACTION)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--colormap", default=_PRE["colormap"], choices=_colormap_choices(),
                    help="depth-perception colormap applied at step 5 (default: turbo)")
    ap.add_argument("--contrast", type=float, default=_PRE["contrast"])
    ap.add_argument("--balance", action="store_true", default=True,
                    help="cap each class to the smallest class per split (default on)")
    ap.add_argument("--no-balance", dest="balance", action="store_false")
    args = ap.parse_args()

    # Write through to the shared config, so preprocess_for_model() -- and therefore inference --
    # uses exactly what this build used.
    _PRE["colormap"], _PRE["contrast"] = args.colormap, args.contrast

    print("=" * 78)
    print("MATERNAL HEALTH - DATASET ORGANISATION (Good / Med / Bad Health)")
    print("=" * 78)

    print("\nReading sources")
    groups = collect_abnormal()
    groups.update(collect_good())
    if not groups:
        print("No images found - check Data Set/ is present.", file=sys.stderr)
        sys.exit(1)

    by_class = defaultdict(lambda: [0, 0])
    for (cls, _), paths in groups.items():
        by_class[cls][0] += len(paths)
        by_class[cls][1] += 1
    for cls in CLASSES:
        n_img, n_grp = by_class[cls]
        src = "FETAL_PLANES_ZENODO" if cls == GOOD else "HASH Maternal Health.coco"
        unit = "patients" if cls == GOOD else "base scans"
        print(f"  {cls:12s}: {n_img:5d} images from {n_grp:4d} {unit:11s} <- {src}")

    print(f"\nNormalising and de-duplicating "
          f"(grayscale -> crop {_PRE['crop_fraction']:.0%} -> {_PRE['img_size']}px -> equalise -> "
          f"{_PRE['colormap']} colormap -> contrast x{_PRE['contrast']} -> JPEG q{JPEG_QUALITY})")
    groups, blobs, dedupe_stats = dedupe_and_merge(groups)

    rng = random.Random(args.seed)
    plan = defaultdict(list)
    for cls in CLASSES:
        keys = sorted(k for k in groups if k[0] == cls)
        rng.shuffle(keys)
        n_val = max(1, round(len(keys) * args.validate_fraction)) if keys else 0
        for split, chosen in (("validate", keys[:n_val]), ("train", keys[n_val:])):
            for key in chosen:
                plan[(split, cls)].extend((p, key[1]) for p in sorted(groups[key]))

    # Class balance: without it a model scores well by leaning on the majority class. Capped
    # group-wise, so whole scans/patients drop out rather than half of one.
    if args.balance:
        for split in ("train", "validate"):
            target = min(len(plan[(split, c)]) for c in CLASSES)
            for cls in CLASSES:
                items = plan[(split, cls)]
                if len(items) <= target:
                    continue
                by_group = defaultdict(list)
                for p, g in items:
                    by_group[g].append((p, g))
                keys = sorted(by_group)
                rng.shuffle(keys)
                kept = []
                for k in keys:
                    if len(kept) >= target:
                        break
                    kept.extend(by_group[k])
                plan[(split, cls)] = kept[:target]

    print(f"\nGroup-wise split (seed {args.seed}, {args.validate_fraction:.0%} of groups held out)")
    for split in ("train", "validate"):
        for cls in CLASSES:
            items = plan[(split, cls)]
            print(f"  {split:8s} {cls:12s}: {len(items):5d} images from "
                  f"{len({g for _, g in items}):4d} groups")

    leaks = []
    for cls in CLASSES:
        tr = {g for _, g in plan[("train", cls)]}
        va = {g for _, g in plan[("validate", cls)]}
        if tr & va:
            leaks.append(f"{cls}: {len(tr & va)}")
    print(f"\n  group overlap train/validate: {leaks or 'none'}   "
          f"(must be none; the shipped COCO split had 104)")
    assert not leaks, f"group leaked across the split: {leaks}"

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    print("\nWriting")
    prefixes = {GOOD: "good", MED: "med", BAD: "bad"}
    manifest_files, written = [], Counter()
    for split in ("train", "validate"):
        for cls in CLASSES:
            dest_dir = class_dir(cls, split)
            os.makedirs(dest_dir, exist_ok=True)
            kept = _clear(dest_dir)
            if kept:
                print(f"  note: kept {len(kept)} non-generated file(s) in {dest_dir}: {kept[:3]}")
            for src, group in plan[(split, cls)]:
                stem = os.path.splitext(os.path.basename(src))[0]
                stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)[:90]
                dest = os.path.join(dest_dir, f"{prefixes[cls]}_{stem}.jpg")
                with open(dest, "wb") as fh:          # already normalised during de-duplication
                    fh.write(blobs[src])
                written[(split, cls)] += 1
                manifest_files.append({
                    "file": os.path.relpath(dest, MATERNAL_DIR).replace("\\", "/"),
                    "source": os.path.relpath(src, MATERNAL_DIR).replace("\\", "/"),
                    "group": group, "split": split, "class": cls,
                    "pathology": None if cls == GOOD else _pathology_of(group),
                })
            print(f"  {split:8s} {cls:12s}: {written[(split, cls)]:5d} written -> "
                  f"{os.path.relpath(dest_dir, MATERNAL_DIR)}")

    pathologies = Counter(e["pathology"] for e in manifest_files if e["pathology"])
    manifest = {
        "generated_by": "Tools/organize_maternal_data.py",
        "seed": args.seed, "validate_fraction": args.validate_fraction, "balanced": args.balance,
        "classes": list(CLASSES),
        "class_definitions": {
            GOOD: {"meaning": "normal fetal brain plane, routine screening",
                   "source": f"Data Set/FETAL_PLANES_ZENODO (Brain_plane in {list(NORMAL_BRAIN_PLANES)})",
                   "group_key": "Zenodo Patient_num"},
            MED: {"meaning": "abnormal; warrants follow-up or monitoring",
                  "source": "Data Set/HASH Maternal Health.coco", "group_key": "base scan name"},
            BAD: {"meaning": "abnormal; warrants urgent referral",
                  "source": "Data Set/HASH Maternal Health.coco", "group_key": "base scan name"},
        },
        "severity_labels_caveat":
            "The COCO source carries a single label, 'abnormal', and no severity field. The "
            "Med/Bad boundary is applied by SEVERITY_MAP in Tools/organize_maternal_data.py from "
            "the pathology name parsed out of each filename. It is a clinical convention, not a "
            "radiologist's reading of these images. 'Good Health' comes from a different dataset "
            "than the other two classes - see Tests/dataset_integrity_test.py for how much "
            "accuracy that alone can buy.",
        "pipeline": {
            "steps": ["grayscale", f"centre-crop {_PRE['crop_fraction']}",
                      f"resize {_PRE['img_size']}", "histogram equalise",
                      f"{_PRE['colormap']} colormap (depth perception)",
                      f"contrast x{_PRE['contrast']}", f"JPEG q{JPEG_QUALITY}"],
            "config": dict(_PRE),
            "implemented_by": "offline_cv.preprocess_for_model('maternal', img) - the same call "
                              "inference.py makes, so training and inference cannot drift apart",
            "rationale": "identical for every class, so source format, resolution, scanner overlay "
                         "and global brightness cannot carry the label",
        },
        "deduplication": dedupe_stats,
        "counts": {f"{s}/{c}": n for (s, c), n in written.items()},
        "pathology_counts": dict(pathologies.most_common()),
        "files": manifest_files,
    }
    path = os.path.join(MATERNAL_DIR, "dataset_manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nmanifest: {os.path.relpath(path, APP_DIR)} ({len(manifest_files)} files, "
          f"each traceable to its source and pathology)")
    print("\nNext: python Tests/dataset_integrity_test.py   (leakage + shortcut control)")
    print("      python train_local_model.py maternal")


if __name__ == "__main__":
    main()
