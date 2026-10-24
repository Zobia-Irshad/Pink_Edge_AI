#!/usr/bin/env python3
"""
Builds the Mammography training set: distributes the usable images of the
BreastCancer-YOLOv8 COCO export into `positive/` and `negative/`, plus a held-out
`validate/{positive,negative}/`, under Models/Mammography/.

Author Name:  Imaad Ullah Khan
Author Email: yameenimaad@gmail.com
AI Helper:    Claude

Run:  python Tools/organize_mammography_data.py            (from App/)
      python Tools/organize_mammography_data.py --dry-run  (report the plan, write nothing)

------------------------------------------------------------------------------------------------
THE TWO CLASSES
------------------------------------------------------------------------------------------------
  positive   cancer present     <- COCO category `cancer`
  negative   no cancer          <- COCO category `normal`

The export also declares a third category, `breastcancer-yolov8-cancer`, which is Roboflow's
super-category wrapper and carries no annotations of its own; it is ignored.

------------------------------------------------------------------------------------------------
THE IMAGE PIPELINE
------------------------------------------------------------------------------------------------
  1. greyscale          a mammogram carries no colour
  2. resize to 224      one resolution for every image
  3. histogram equalise a mammogram occupies a compressed bright band; spreading it is what
                        separates dense fibroglandular tissue from a mass or a spiculated margin
  4. contrast x1.5      widens that separation further
  5. JPEG q90           one encoder, one quality

No crop and no colormap, matching TB: a lesion can sit anywhere in the field, including hard
against the chest wall, so a centre crop risks cutting away the very finding; and a mammogram is
read on fine greyscale texture (microcalcification clusters) that a false-colour map would
quantise rather than clarify.

Steps 1-4 are NOT implemented here. They live in offline_cv.preprocess_for_model(), which
inference.py also calls before running the trained weights -- one definition, so the images the
model is built on and the images the app later feeds it cannot drift apart.

Before this script existed, Mammography had NO preprocessing configured at all:
preprocess_for_model() fell through to a bare .convert("RGB"), so the modality was the only one
without train/inference parity or an enforceable pipeline.

------------------------------------------------------------------------------------------------
TWO DEFECTS IN THE RAW SOURCE THAT THIS REPAIRS
------------------------------------------------------------------------------------------------
1. THE SHIPPED SPLIT LEAKS. Its 1,642 images are augmentations of only 515 distinct base scans,
   and 59 of test's 100 base scans -- plus 91 of validate's 171 (53.2%) -- also appear in train.
   Accuracy measured against those splits is measuring memorisation of images already trained on.
   This is the same defect that made TB's old 82.5% meaningless. The split here is re-derived over
   GROUPS keyed on the base scan name (MIAS ids such as `mdb001lm`, and `normal (107)` for the
   healthy set), so every augmented copy of one scan lands wholly in train or wholly in validate.

2. THE CLASS FOLDERS WERE THEMSELVES A LEAK SOURCE. Models/Mammography/{positive,negative}/ held
   235 Roboflow-hashed copies taken straight from the export -- including `mdb001lm`, a base scan
   that also appears in train, test AND validate. Because the trainer reads those folders, scans
   held out for evaluation were being trained on from a second direction. They are regenerated
   here from the group-disjoint plan. (They were dataset copies, not hand-added images: 155 of the
   156 files in positive/ matched the Roboflow hash pattern and the remaining one was an empty
   files.txt, so nothing contributed by hand is lost.)

Identical images under different names are de-duplicated on the content hash of the NORMALISED
image, and any two groups sharing an image are merged so they cannot straddle the split.

Provenance for every written file is recorded in dataset_manifest.json next to the folders.
"""

import argparse
import io
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_tools
import offline_cv  # the single definition of the image pipeline and the class list

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAMMO_DIR = os.path.join(APP_DIR, "Models", "Mammography")
COCO_DIR = offline_cv._dataset_dir("mammography")

POSITIVE, NEGATIVE = "positive", "negative"
CLASSES = (NEGATIVE, POSITIVE)      # index order == the model's output index
PREFIXES = {POSITIVE: "pos", NEGATIVE: "neg"}

# Roboflow's super-category wrapper: declared in `categories` but never annotated. Listing it
# keeps it out of the "unlabelled" tally, which would otherwise look like missing data.
SUPERCATEGORY = "breastcancer-yolov8-cancer"

_PRE = offline_cv._DATASETS["mammography"]["preprocess"]
JPEG_QUALITY = 90                   # step 5, only relevant when writing the dataset to disk

VALIDATE_FRACTION = 0.25            # of GROUPS, not of images
SEED = 20251004


def class_dir(cls, split="train"):
    return os.path.join(MAMMO_DIR, cls) if split == "train" \
        else os.path.join(MAMMO_DIR, "validate", cls)


def _base_scan(file_name):
    """The distinct scan an augmented copy came from.
    'mdb001lm_jpg.rf.<hash>.jpg'      -> 'mdb001lm'      (MIAS: patient 001, left mediolateral)
    'normal (107)_jpg.rf.<hash>.jpg'  -> 'normal (107)'
    Grouping on this is what stops an augmentation of a validation scan being trained on."""
    name = os.path.basename(file_name)
    name = re.sub(r"_aug_\d+.*$", "", name)
    return re.sub(r"(_png)?_jpg\.rf\..*$", "", name)


def collect():
    """Every usable image keyed by (class, base scan). The shipped train/test/validate boundaries
    are deliberately ignored and re-derived later -- they leak (defect 1)."""
    cfg = offline_cv._DATASETS["mammography"]
    pos_names = set(cfg["positive_categories"])
    neg_names = set(cfg["negative_categories"])

    groups = defaultdict(list)
    stats = Counter()
    for split, split_dir, coco in offline_cv._load_coco_splits(COCO_DIR):
        cat_name = {c["id"]: c["name"] for c in coco["categories"]}
        labels = defaultdict(set)
        for ann in coco["annotations"]:
            labels[ann["image_id"]].add(cat_name.get(ann["category_id"], ""))

        for img in coco["images"]:
            names = {n for n in labels.get(img["id"], set()) if n != SUPERCATEGORY}
            is_pos, is_neg = bool(names & pos_names), bool(names & neg_names)
            if is_pos and is_neg:
                stats["conflicting_labels"] += 1
                continue
            if not (is_pos or is_neg):
                stats["excluded_unlabelled"] += 1
                continue
            path = os.path.join(split_dir, img["file_name"])
            if not os.path.isfile(path):
                stats["missing_file"] += 1
                continue
            cls = POSITIVE if is_pos else NEGATIVE
            groups[(cls, _base_scan(img["file_name"]))].append(path)
            stats["kept"] += 1
            stats["from_" + split] += 1
    return groups, stats


def normalised_bytes(src_path):
    """Steps 1-4 via the shared definition, then JPEG (step 5). Returns encoded bytes rather than
    writing, so the caller can hash the normalised image to find duplicates before placing it."""
    from PIL import Image

    with Image.open(src_path) as im:
        rgb = offline_cv.preprocess_for_model("mammography", im)
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    ap.add_argument("--validate-fraction", type=float, default=VALIDATE_FRACTION)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--contrast", type=float, default=_PRE["contrast"])
    ap.add_argument("--max-per-class", type=int, default=0,
                    help="cap images per class per split (0 = no cap, after balancing)")
    ap.add_argument("--balance", action="store_true", default=True,
                    help="cap each class to the smaller one per split (default on)")
    ap.add_argument("--no-balance", dest="balance", action="store_false")
    ap.add_argument("--keep-stale-copies", action="store_true",
                    help="do NOT remove the pre-existing Roboflow-hashed copies in the class "
                         "folders (defect 2). Off by default because they leak across the split.")
    ap.add_argument("--max-per-group-train", type=int, default=2,
                    help="cap augmented copies per base scan in the TRAIN split (0 = no cap). "
                         "Defaults to 2: the positive class averages ~7.7 copies per scan against "
                         "the negative class's 1, so an uncapped balance spends the whole positive "
                         "budget on 35 of its 167 scans.")
    ap.add_argument("--max-per-group-validate", type=int, default=1,
                    help="cap copies per base scan in the HELD-OUT split (0 = no cap). Defaults to "
                         "1 -- one vote per scan, so the reported accuracy is not several "
                         "correlated augmentations of the same patient voting repeatedly.")
    args = ap.parse_args()

    # Write through to the shared config, so preprocess_for_model() -- and therefore inference --
    # uses exactly what this build used.
    _PRE["contrast"] = args.contrast

    print("=" * 78)
    print("MAMMOGRAPHY - DATASET ORGANISATION (positive / negative)")
    print("=" * 78)

    print(f"\nReading source: {os.path.relpath(COCO_DIR, APP_DIR)}")
    groups, stats = collect()
    if not groups:
        print(f"No images found under {COCO_DIR!r}", file=sys.stderr)
        sys.exit(1)

    by_class = defaultdict(lambda: [0, 0])
    for (cls, _), paths in groups.items():
        by_class[cls][0] += len(paths)
        by_class[cls][1] += 1
    for cls in CLASSES:
        n_img, n_grp = by_class[cls]
        print(f"  {cls:9s}: {n_img:5d} images from {n_grp:5d} distinct base scans")
    if stats["conflicting_labels"]:
        print(f"  excluded : {stats['conflicting_labels']:5d} with BOTH a cancer and a normal label")
    if stats["excluded_unlabelled"]:
        print(f"             {stats['excluded_unlabelled']:5d} with no usable label")

    print(f"\nNormalising and de-duplicating "
          f"(greyscale -> {_PRE['img_size']}px -> equalise -> contrast x{_PRE['contrast']} "
          f"-> JPEG q{JPEG_QUALITY})")
    groups, blobs, dedupe_stats = dataset_tools.dedupe_and_merge(
        groups, normalised_bytes, on_warning=lambda m: print(f"  warn: {m}"))
    print(f"  dropped {dedupe_stats['duplicates_dropped']} duplicate image(s); merged "
          f"{dedupe_stats['group_pairs_merged']} group pair(s) sharing an identical image "
          f"({dedupe_stats['groups_before']} -> {dedupe_stats['groups_after']} groups)")

    rng = random.Random(args.seed)
    caps = {"train": args.max_per_group_train, "validate": args.max_per_group_validate}
    plan = {}
    for cls in CLASSES:
        keys = [k for k in groups if k[0] == cls]
        train_keys, val_keys = dataset_tools.split_groups(keys, args.validate_fraction, rng)
        for split, chosen in (("train", train_keys), ("validate", val_keys)):
            items = [(p, key[1]) for key in chosen for p in sorted(groups[key])]
            # Cap BEFORE balancing: balancing drops whole groups to hit an image target, so an
            # uncapped heavily-augmented class loses scan diversity rather than copies.
            plan[(split, cls)] = dataset_tools.cap_per_group(items, caps[split], rng)

    if args.balance:
        for split in ("train", "validate"):
            capped = dataset_tools.balance_group_wise(
                {c: plan[(split, c)] for c in CLASSES}, rng,
                target=min(args.max_per_class, min(len(plan[(split, c)]) for c in CLASSES))
                if args.max_per_class else None)
            for c in CLASSES:
                plan[(split, c)] = capped[c]

    print(f"\nGroup-wise split (seed {args.seed}, {args.validate_fraction:.0%} of groups held out)")
    for split in ("train", "validate"):
        for cls in CLASSES:
            items = plan[(split, cls)]
            print(f"  {split:8s} {cls:9s}: {len(items):5d} images from "
                  f"{len({g for _, g in items}):5d} groups")

    leaks = []
    for cls in CLASSES:
        tr = {g for _, g in plan[("train", cls)]}
        va = {g for _, g in plan[("validate", cls)]}
        if tr & va:
            leaks.append(f"{cls}: {len(tr & va)}")
    print(f"\n  group overlap train/validate: {leaks or 'none'}   "
          f"(must be none; the shipped split had 91)")
    assert not leaks, f"group leaked across the split: {leaks}"

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    print("\nWriting")
    manifest_files, written = [], Counter()
    stale_removed = 0
    for split in ("train", "validate"):
        for cls in CLASSES:
            dest_dir = class_dir(cls, split)
            os.makedirs(dest_dir, exist_ok=True)
            kept = dataset_tools.clear_generated(dest_dir, PREFIXES.values())
            # Defect 2: the pre-existing copies carry Roboflow hash names rather than this
            # script's pos_/neg_ prefixes, so clear_generated() would preserve them and the leak
            # would survive. Remove them explicitly unless asked not to.
            if not args.keep_stale_copies:
                for name in list(kept):
                    if re.search(r"_jpg\.rf\.", name):
                        os.remove(os.path.join(dest_dir, name))
                        kept.remove(name)
                        stale_removed += 1
            if kept:
                print(f"  note: kept {len(kept)} non-generated file(s) in "
                      f"{os.path.relpath(dest_dir, MAMMO_DIR)}: {kept[:3]}")
            for src, group in plan[(split, cls)]:
                stem = os.path.splitext(os.path.basename(src))[0]
                stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)[:90]
                dest = os.path.join(dest_dir, f"{PREFIXES[cls]}_{stem}.jpg")
                with open(dest, "wb") as fh:      # already normalised during de-duplication
                    fh.write(blobs[src])
                written[(split, cls)] += 1
                manifest_files.append({
                    "file": os.path.relpath(dest, MAMMO_DIR).replace("\\", "/"),
                    "source": os.path.relpath(src, MAMMO_DIR).replace("\\", "/"),
                    "group": group, "split": split, "class": cls,
                })
            print(f"  {split:8s} {cls:9s}: {written[(split, cls)]:5d} written -> "
                  f"{os.path.relpath(dest_dir, MAMMO_DIR)}")
    if stale_removed:
        print(f"  removed {stale_removed} pre-existing Roboflow-hashed copy/copies from the class "
              f"folders (defect 2 -- they leaked across the split)")

    manifest = {
        "generated_by": "Tools/organize_mammography_data.py",
        "seed": args.seed, "validate_fraction": args.validate_fraction, "balanced": args.balance,
        "classes": list(CLASSES),
        "class_definitions": {
            POSITIVE: {"meaning": "cancer present",
                       "source_categories": offline_cv._DATASETS["mammography"]["positive_categories"]},
            NEGATIVE: {"meaning": "no cancer",
                       "source_categories": offline_cv._DATASETS["mammography"]["negative_categories"]},
        },
        "excluded": {
            "conflicting_cancer_and_normal_labels": stats["conflicting_labels"],
            "unlabelled": stats["excluded_unlabelled"],
            "ignored_supercategory": SUPERCATEGORY,
        },
        "split_caveat":
            "The shipped train/test/validate boundaries are NOT used: 1,642 images are "
            "augmentations of 515 base scans, and 59 of test's 100 base scans (plus 91 of "
            "validate's 171) also appear in train. The split here is over base-scan groups, so no "
            "augmented copy of a validation scan is trained on. The previously reported 98.7% was "
            "measured against the leaking split, and additionally the class folders held 235 "
            "Roboflow-hashed copies of export images that the trainer also read.",
        "stale_copies_removed_from_class_folders": stale_removed,
        "pipeline": {
            "steps": ["greyscale", f"resize {_PRE['img_size']}", "histogram equalise",
                      f"contrast x{_PRE['contrast']}", f"JPEG q{JPEG_QUALITY}"],
            "config": dict(_PRE),
            "implemented_by": "offline_cv.preprocess_for_model('mammography', img) - the same call "
                              "inference.py makes, so training and inference cannot drift apart. "
                              "Mammography had no pipeline configured at all before this.",
        },
        "deduplication": dedupe_stats,
        "counts": {f"{s}/{c}": n for (s, c), n in written.items()},
        "files": manifest_files,
    }
    path = os.path.join(MAMMO_DIR, "dataset_manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"\nmanifest: {os.path.relpath(path, APP_DIR)} ({len(manifest_files)} files, "
          f"each traceable to its source)")
    print("\nNext: python Tests/dataset_integrity_test.py mammography")
    print("      python train_local_model.py mammography")


if __name__ == "__main__":
    main()
