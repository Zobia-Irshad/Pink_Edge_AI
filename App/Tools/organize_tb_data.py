#!/usr/bin/env python3
"""
Builds the Tuberculosis training set: distributes the usable images of
Models/TB/Data Set/tuberculosis.coco into `positive/` and `negative/`, plus a held-out
`validate/{positive,negative}/`, under Models/TB/.

Author Name:  Imaad Ullah Khan
Author Email: yameenimaad@gmail.com
AI Helper:    Claude

Run:  python Tools/organize_tb_data.py            (from App/)
      python Tools/organize_tb_data.py --dry-run  (report the plan, write nothing)

------------------------------------------------------------------------------------------------
THE TWO CLASSES
------------------------------------------------------------------------------------------------
  positive   active tuberculosis        <- COCO categories Tüberküloz, Hastalıklı
  negative   not active tuberculosis    <- COCO categories Sağlıklı, Sekelli

`Sekelli` is sequelae: healed post-TB scarring. It is radiographically abnormal but it is not
active disease, so it belongs in `negative` for a detector that answers "is this active TB" -- and
it makes the negative class usefully hard, since a model that merely spots "this chest looks
abnormal" will get these wrong.

`Gizli` (latent TB) is in NEITHER class and is excluded. Latent infection is usually
radiographically silent, so a chest X-ray cannot support either answer; training it as positive
would teach the model to call normal-looking films positive, and as negative would teach it to
ignore real findings. 520 images are dropped for this reason. The old config excluded them too,
but silently -- here the exclusion is deliberate and counted.

------------------------------------------------------------------------------------------------
THE IMAGE PIPELINE
------------------------------------------------------------------------------------------------
  1. greyscale          a chest X-ray carries no colour
  2. resize to 224      one resolution for every image
  3. histogram equalise spreads the compressed mid-grey range a plain X-ray occupies, so
                        parenchymal texture -- infiltrate, cavitation, effusion -- separates from
                        the background instead of sitting in a narrow band of near-identical greys
  4. contrast x1.5      widens that separation further
  5. JPEG q90           one encoder, one quality

No crop and no colormap, unlike Maternal: both TB classes come from a single Roboflow export at a
uniform 640x640 RGB, so there is no scanner overlay or source cue to crop away, and a chest film is
read on fine greyscale texture that a false-colour map would quantise rather than clarify.

Steps 1-4 are NOT implemented here. They live in offline_cv.preprocess_for_model(), which
inference.py also calls before running the trained weights -- one definition, so the images the
model is built on and the images the app later feeds it cannot drift apart.

------------------------------------------------------------------------------------------------
THREE DEFECTS IN THE RAW SOURCE THAT THIS REPAIRS
------------------------------------------------------------------------------------------------
1. THE SHIPPED SPLIT LEAKS ALMOST COMPLETELY. Its 9,917 images are augmentations of only 1,507
   distinct base scans, and 699 of valid's 703 base scans -- and 419 of test's 422 -- also appear
   in train. Accuracy measured against those splits is measuring memorisation of images already
   trained on. (The 82.5% previously reported for TB's local model was measured that way.) The
   split here is re-derived over GROUPS keyed on the base scan name, so every augmented copy of one
   scan lands wholly in train or wholly in validate.

2. 82 IMAGES CARRY BOTH A POSITIVE AND A NEGATIVE LABEL. offline_cv._collect_image_paths() adds
   such an image to both sets, so the same picture trains as active TB and as healthy at once.
   They are excluded here and counted.

3. IDENTICAL IMAGES UNDER DIFFERENT NAMES. Images are de-duplicated on the content hash of the
   NORMALISED image, and any two groups sharing an image are merged so they cannot straddle the
   split.

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
TB_DIR = os.path.join(APP_DIR, "Models", "TB")
COCO_DIR = offline_cv._dataset_dir("tb")

POSITIVE, NEGATIVE = "positive", "negative"
CLASSES = (NEGATIVE, POSITIVE)      # index order == the model's output index
PREFIXES = {POSITIVE: "pos", NEGATIVE: "neg"}

EXCLUDED_CATEGORIES = ("Gizli",)    # latent TB -- see the module docstring
_PRE = offline_cv._DATASETS["tb"]["preprocess"]
JPEG_QUALITY = 90                   # step 5, only relevant when writing the dataset to disk

VALIDATE_FRACTION = 0.25            # of GROUPS, not of images
SEED = 20251003


def class_dir(cls, split="train"):
    return os.path.join(TB_DIR, cls) if split == "train" \
        else os.path.join(TB_DIR, "validate", cls)


def _base_scan(file_name):
    """The distinct scan an augmented copy came from.
    'tb0065_png_jpg.rf.<hash>.jpg' -> 'tb0065';  'h0504_png_jpg.rf.<hash>.jpg' -> 'h0504'.
    Grouping on this is what stops an augmentation of a validation scan being trained on."""
    name = os.path.basename(file_name)
    name = re.sub(r"_aug_\d+.*$", "", name)
    return re.sub(r"(_png)?_jpg\.rf\..*$", "", name)


def collect():
    """Every usable image keyed by (class, base scan). The shipped train/valid/test boundaries are
    deliberately ignored and re-derived later -- they leak (defect 1)."""
    cfg = offline_cv._DATASETS["tb"]
    pos_names = set(cfg["positive_categories"])
    neg_names = set(cfg["negative_categories"])

    groups = defaultdict(list)
    stats = Counter()
    for split in ("train", "valid", "test"):
        ann_path = os.path.join(COCO_DIR, split, "_annotations.coco.json")
        if not os.path.isfile(ann_path):
            continue
        with open(ann_path, "r", encoding="utf-8") as fh:
            coco = json.load(fh)
        cat_name = {c["id"]: c["name"] for c in coco["categories"]}
        labels = defaultdict(set)
        for ann in coco["annotations"]:
            labels[ann["image_id"]].add(cat_name.get(ann["category_id"], ""))

        for img in coco["images"]:
            names = labels.get(img["id"], set())
            is_pos, is_neg = bool(names & pos_names), bool(names & neg_names)
            if is_pos and is_neg:
                stats["conflicting_labels"] += 1          # defect 2
                continue
            if not (is_pos or is_neg):
                stats["excluded_" + ("latent" if names & set(EXCLUDED_CATEGORIES)
                                     else "unlabelled")] += 1
                continue
            path = os.path.join(COCO_DIR, split, img["file_name"])
            if not os.path.isfile(path):
                stats["missing_file"] += 1
                continue
            cls = POSITIVE if is_pos else NEGATIVE
            groups[(cls, _base_scan(img["file_name"]))].append(path)
            stats["kept"] += 1
    return groups, stats


def normalised_bytes(src_path):
    """Steps 1-4 via the shared definition, then JPEG (step 5). Returns encoded bytes rather than
    writing, so the caller can hash the normalised image to find duplicates before placing it."""
    from PIL import Image

    with Image.open(src_path) as im:
        rgb = offline_cv.preprocess_for_model("tb", im)
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
                    help="do NOT remove raw Roboflow-hashed copies sitting in the class folders. "
                         "Off by default: clear_generated() deliberately preserves files it did "
                         "not write (to protect hand-added images), but a raw export copy is not a "
                         "hand-added image -- it bypasses the pipeline and can re-introduce a "
                         "scan that belongs to the held-out split.")
    args = ap.parse_args()

    # Write through to the shared config, so preprocess_for_model() -- and therefore inference --
    # uses exactly what this build used.
    _PRE["contrast"] = args.contrast

    print("=" * 78)
    print("TUBERCULOSIS - DATASET ORGANISATION (positive / negative)")
    print("=" * 78)

    print("\nReading source")
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
    print(f"  excluded : {stats['excluded_latent']:5d} latent (Gizli) - radiographically silent, "
          f"supports neither answer")
    if stats["conflicting_labels"]:
        print(f"             {stats['conflicting_labels']:5d} with BOTH a positive and a negative "
              f"label (contradictory source annotation)")
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
    plan = {}
    for cls in CLASSES:
        keys = [k for k in groups if k[0] == cls]
        train_keys, val_keys = dataset_tools.split_groups(keys, args.validate_fraction, rng)
        for split, chosen in (("train", train_keys), ("validate", val_keys)):
            plan[(split, cls)] = [(p, key[1]) for key in chosen for p in sorted(groups[key])]

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
          f"(must be none; the shipped split had 699)")
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
            # Raw export copies carry Roboflow hash names rather than this script's pos_/neg_
            # prefixes, so clear_generated() preserves them -- correct for a hand-added image,
            # wrong for an unprocessed duplicate of the source. Six such files once reached these
            # folders (three per class) and one of them, h0993, belonged to the held-out split.
            if not args.keep_stale_copies:
                for name in list(kept):
                    if re.search(r"_jpg\.rf\.", name):
                        os.remove(os.path.join(dest_dir, name))
                        kept.remove(name)
                        stale_removed += 1
            if kept:
                print(f"  note: kept {len(kept)} non-generated file(s) in {dest_dir}: {kept[:3]}")
            for src, group in plan[(split, cls)]:
                stem = os.path.splitext(os.path.basename(src))[0]
                stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem)[:90]
                dest = os.path.join(dest_dir, f"{PREFIXES[cls]}_{stem}.jpg")
                with open(dest, "wb") as fh:      # already normalised during de-duplication
                    fh.write(blobs[src])
                written[(split, cls)] += 1
                manifest_files.append({
                    "file": os.path.relpath(dest, TB_DIR).replace("\\", "/"),
                    "source": os.path.relpath(src, TB_DIR).replace("\\", "/"),
                    "group": group, "split": split, "class": cls,
                })
            print(f"  {split:8s} {cls:9s}: {written[(split, cls)]:5d} written -> "
                  f"{os.path.relpath(dest_dir, TB_DIR)}")
    if stale_removed:
        print(f"  removed {stale_removed} raw Roboflow-hashed copy/copies from the class folders "
              f"(they bypass the pipeline and can re-introduce a held-out scan)")

    manifest = {
        "generated_by": "Tools/organize_tb_data.py",
        "seed": args.seed, "validate_fraction": args.validate_fraction, "balanced": args.balance,
        "classes": list(CLASSES),
        "class_definitions": {
            POSITIVE: {"meaning": "active tuberculosis",
                       "source_categories": offline_cv._DATASETS["tb"]["positive_categories"]},
            NEGATIVE: {"meaning": "not active tuberculosis (healthy, or healed post-TB scarring)",
                       "source_categories": offline_cv._DATASETS["tb"]["negative_categories"]},
        },
        "excluded": {
            "latent_Gizli": stats["excluded_latent"],
            "conflicting_positive_and_negative_labels": stats["conflicting_labels"],
            "unlabelled": stats["excluded_unlabelled"],
            "why": "Gizli (latent TB) is radiographically silent, so a chest X-ray supports "
                   "neither answer. Images carrying both a positive and a negative annotation are "
                   "contradictory in the source and would train as both classes at once.",
        },
        "split_caveat":
            "The shipped train/valid/test boundaries are NOT used: 9,917 images are augmentations "
            "of 1,507 base scans, and 699 of valid's 703 base scans also appear in train. The "
            "split here is over base-scan groups, so no augmented copy of a validation scan is "
            "trained on.",
        "pipeline": {
            "steps": ["greyscale", f"resize {_PRE['img_size']}", "histogram equalise",
                      f"contrast x{_PRE['contrast']}", f"JPEG q{JPEG_QUALITY}"],
            "config": dict(_PRE),
            "implemented_by": "offline_cv.preprocess_for_model('tb', img) - the same call "
                              "inference.py makes, so training and inference cannot drift apart",
        },
        "deduplication": dedupe_stats,
        "counts": {f"{s}/{c}": n for (s, c), n in written.items()},
        "files": manifest_files,
    }
    path = os.path.join(TB_DIR, "dataset_manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"\nmanifest: {os.path.relpath(path, APP_DIR)} ({len(manifest_files)} files, "
          f"each traceable to its source)")
    print("\nNext: python Tests/dataset_integrity_test.py tb")
    print("      python train_local_model.py tb")


if __name__ == "__main__":
    main()
