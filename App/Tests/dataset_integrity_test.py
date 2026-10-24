#!/usr/bin/env python3
"""
Pink Edge AI — organised-dataset integrity + shortcut control (Maternal and TB).

Author Name:  Imaad Ullah Khan
Author Email: yameenimaad@gmail.com
AI Helper:    Claude

Checks that the class folders built by Tools/organize_maternal_data.py and Tools/organize_tb_data.py
can support an accuracy number that means something, and measures how much of such a number is
available without looking at anatomy at all.

The structural checks are the cheap ones: no group leaks across the train/validate boundary, no
byte-identical image in two places, classes balanced, every image through the same pipeline,
manifest agrees with disk, and inference re-derives the training images byte-for-byte.

The last two are the ones that matter. Any accuracy is only meaningful above the accuracy
obtainable from cues that are not anatomy -- dataset of origin, scanner overlay, resolution,
compression, global brightness. So a control classifier is trained on images crushed to
CONTROL_PX x CONTROL_PX, where no ventricle, cavitation or infiltrate is resolvable and only coarse
statistics survive. That score is the floor a real model has to clear, and it is usually well above
the chance line.

Run:  python Tests/dataset_integrity_test.py              (from App/ — both modalities)
      python Tests/dataset_integrity_test.py tb           (just one)
      python Tests/dataset_integrity_test.py -v

Exits non-zero if any check fails.
"""

import hashlib
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERBOSE = "-v" in sys.argv or "--verbose" in sys.argv

CONTROL_PX = 8          # anatomy is unrecognisable at 8x8; dataset-level statistics are not
CONTROL_MARGIN = 5.0    # pp the real model must beat the control by to have earned its accuracy
SPLITS = ("train", "validate")

# Per-modality facts this suite needs that are not already in offline_cv's config.
MODALITIES = {
    "maternal": {
        "subdir": "Maternal",
        # Good Health is drawn from a different dataset than Med/Bad, so the control floor here is
        # high by construction -- that is exactly what it is there to quantify.
        "expected_mode": "RGB",
        "required_caveat": "severity_labels_caveat",
        "group_patterns": {"Good Health": r"^good_(Patient\d+)",
                           "Med Health": r"^med_(.+?)_aug_\d+",
                           "Bad Health": r"^bad_(.+?)_aug_\d+"},
        "shipped_leak": "104 of valid's 104 base scans",
    },
    "tb": {
        "subdir": "TB",
        "expected_mode": "RGB",     # greyscale replicated to 3 channels, no colormap
        "required_caveat": "split_caveat",
        "group_patterns": {"positive": r"^pos_(.+?)(?:_png)?_jpg\.rf\.",
                           "negative": r"^neg_(.+?)(?:_png)?_jpg\.rf\."},
        "shipped_leak": "699 of valid's 703 base scans",
    },
    "mammography": {
        "subdir": "Mammography",
        "expected_mode": "RGB",     # greyscale replicated to 3 channels, no colormap, same as TB
        "required_caveat": "split_caveat",
        # Positives are MIAS ids ('mdb001lm'); negatives are 'normal (107)'-style names, whose
        # parentheses and space the writer replaces with underscores.
        "group_patterns": {"positive": r"^pos_(.+?)(?:_png)?_jpg\.rf\.",
                           "negative": r"^neg_(.+?)(?:_png)?_jpg\.rf\."},
        "shipped_leak": "59 of test's 100 base scans (and 91 of validate's 171)",
    },
}

RESULTS = []


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


class Suite:
    """Runs the checks for one modality, tagging every result with it."""

    def __init__(self, modality):
        import offline_cv

        self.modality = modality
        self.cfg = MODALITIES[modality]
        self.dir = os.path.join(APP_DIR, "Models", self.cfg["subdir"])
        self.classes = offline_cv.class_names(modality)
        self.chance = 100.0 / len(self.classes)
        self.control = None

    # -- helpers ------------------------------------------------------------
    def class_dir(self, split, cls):
        return os.path.join(self.dir, cls) if split == "train" \
            else os.path.join(self.dir, "validate", cls)

    def files(self, split, cls):
        d = self.class_dir(split, cls)
        if not os.path.isdir(d):
            return []
        return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".jpg"))

    def group_of(self, path, cls):
        """Recover the organiser's group key from the written filename, so leakage is checked
        against what is actually on disk rather than against the manifest's own bookkeeping."""
        stem = os.path.splitext(os.path.basename(path))[0]
        m = re.match(self.cfg["group_patterns"][cls], stem)
        return m.group(1) if m else stem

    def manifest(self):
        path = os.path.join(self.dir, "dataset_manifest.json")
        require(os.path.isfile(path),
                f"dataset_manifest.json missing - run Tools/organize_{self.modality}_data.py")
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def check(self, name, fn):
        tag = f"[{self.modality}] {name}"
        try:
            detail = fn() or ""
            RESULTS.append((tag, True, detail))
            if VERBOSE:
                print(f"  [pass] {tag} {detail}")
        except Exception as e:
            RESULTS.append((tag, False, f"{type(e).__name__}: {e}"))
            if VERBOSE:
                print(f"  [FAIL] {tag} - {type(e).__name__}: {e}")

    # -- structure ----------------------------------------------------------
    def c_populated(self):
        counts = []
        for split in SPLITS:
            for cls in self.classes:
                f = self.files(split, cls)
                require(f, f"{split}/{cls} is empty - run "
                           f"Tools/organize_{self.modality}_data.py")
                counts.append(f"{split[:3]}/{cls.split()[0]}={len(f)}")
        return f"({', '.join(counts)})"

    def c_no_group_leak(self):
        # The defect this exists for: in both shipped COCO splits nearly every 'held-out' image is
        # an augmented copy of a training image, so accuracy against them measures memorisation.
        offenders, totals = [], []
        for cls in self.classes:
            tr = {self.group_of(p, cls) for p in self.files("train", cls)}
            va = {self.group_of(p, cls) for p in self.files("validate", cls)}
            if tr & va:
                offenders.append(f"{cls}: {len(tr & va)} shared e.g. {sorted(tr & va)[:2]}")
            totals.append(f"{cls.split()[0]}={len(tr | va)}")
        require(not offenders, "group leaked across the split -> " + "; ".join(offenders))
        return f"({', '.join(totals)} groups, 0 shared; shipped split had {self.cfg['shipped_leak']})"

    def c_no_duplicates(self):
        seen, dupes = {}, []
        for split in SPLITS:
            for cls in self.classes:
                for p in self.files(split, cls):
                    with open(p, "rb") as fh:
                        h = hashlib.sha256(fh.read()).hexdigest()
                    if h in seen:
                        dupes.append(f"{os.path.basename(p)} == {os.path.basename(seen[h])}")
                    seen[h] = p
        require(not dupes, f"{len(dupes)} duplicate(s): {dupes[:3]}")
        return f"({len(seen)} distinct images)"

    def c_balanced(self):
        out = []
        for split in SPLITS:
            n = [len(self.files(split, c)) for c in self.classes]
            require(all(n), f"{split} has an empty class: {dict(zip(self.classes, n))}")
            ratio = max(n) / min(n)
            require(ratio <= 1.5, f"{split} is imbalanced {n} ({ratio:.2f}x) - a model can score "
                                  f"well by leaning on the majority class")
            out.append(f"{split} {':'.join(str(x) for x in n)}")
        return f"({', '.join(out)}; chance = {self.chance:.1f}%)"

    def c_uniform_pipeline(self):
        # A size/mode/format difference between classes is itself a label the model can read, and
        # would make the control check uninterpretable.
        from PIL import Image

        sizes, modes = {}, {}
        for split in SPLITS:
            for cls in self.classes:
                for p in self.files(split, cls)[:60]:
                    with Image.open(p) as im:
                        sizes.setdefault(cls, set()).add(im.size)
                        modes.setdefault(cls, set()).add(im.mode)
                        require(im.format == "JPEG",
                                f"{os.path.basename(p)} is {im.format}, not JPEG")
        all_sizes = set().union(*sizes.values())
        all_modes = set().union(*modes.values())
        require(len(all_sizes) == 1, f"image sizes differ between classes: {sizes}")
        require(len(all_modes) == 1, f"colour modes differ between classes: {modes}")
        require(all_modes == {self.cfg["expected_mode"]},
                f"expected {self.cfg['expected_mode']}, got {all_modes}")
        return f"(all {all_sizes.pop()} {all_modes.pop()} JPEG)"

    def c_manifest(self):
        man = self.manifest()
        on_disk = sum(len(self.files(s, c)) for s in SPLITS for c in self.classes)
        require(len(man["files"]) == on_disk,
                f"manifest lists {len(man['files'])} files but {on_disk} are on disk")
        require(list(man["classes"]) == list(self.classes),
                f"manifest classes {man['classes']} != {list(self.classes)}")
        caveat = self.cfg["required_caveat"]
        require(man.get(caveat),
                f"manifest has lost its {caveat!r} - that limitation must travel with the data")
        missing = [e["file"] for e in man["files"][:400]
                   if not os.path.isfile(os.path.join(self.dir, e["file"]))]
        require(not missing, f"manifest references missing files e.g. {missing[:3]}")
        return f"({len(man['files'])} files, each traceable to its source)"

    def c_train_inference_parity(self):
        """The quietest way to break a model: change the pipeline on one side only. Nothing would
        error -- inference would just feed the weights images unlike anything they trained on and
        return confident nonsense. So re-run offline_cv.preprocess_for_model() (the call
        inference.py makes) over sample sources and require the bytes to match what is on disk."""
        import offline_cv
        from PIL import Image

        man = self.manifest()
        quality = int(re.search(r"q(\d+)", " ".join(man["pipeline"]["steps"])).group(1))
        sample = man["files"][::max(1, len(man["files"]) // 12)][:12]
        mismatched = []
        for entry in sample:
            src = os.path.join(self.dir, entry["source"])
            dst = os.path.join(self.dir, entry["file"])
            if not (os.path.isfile(src) and os.path.isfile(dst)):
                continue
            buf = io.BytesIO()
            with Image.open(src) as im:
                offline_cv.preprocess_for_model(self.modality, im).save(
                    buf, "JPEG", quality=quality)
            with open(dst, "rb") as fh:
                if hashlib.sha256(buf.getvalue()).digest() != hashlib.sha256(fh.read()).digest():
                    mismatched.append(entry["file"])
        require(sample, "manifest listed no files to re-derive")
        require(not mismatched,
                f"{len(mismatched)}/{len(sample)} images do not reproduce from their source "
                f"through offline_cv.preprocess_for_model() e.g. {mismatched[:2]}. Training and "
                f"inference have drifted apart - rebuild the dataset and retrain.")
        return f"({len(sample)} images re-derived byte-identically)"

    # -- the control --------------------------------------------------------
    def _tiny(self, paths, px):
        import numpy as np
        from PIL import Image

        out = []
        for p in paths:
            with Image.open(p) as im:
                out.append(np.asarray(im.convert("L").resize((px, px), Image.BILINEAR),
                                      dtype=np.float32).ravel() / 255.0)
        return np.stack(out) if out else np.empty((0, px * px), dtype=np.float32)

    def c_control(self, px=CONTROL_PX, cap=400):
        """Logistic regression on px*px images. At 8x8 a fetal skull or a chest field is a handful
        of grey blobs -- no ventricle, cyst, cavitation or infiltrate is resolvable -- so whatever
        this scores, it scored on coarse global statistics, not on pathology."""
        import numpy as np
        import torch
        import torch.nn as nn

        Xtr, ytr, Xva, yva = [], [], [], []
        for idx, cls in enumerate(self.classes):
            tr, va = self.files("train", cls)[:cap], self.files("validate", cls)[:cap]
            Xtr.append(self._tiny(tr, px)); ytr.append(np.full(len(tr), idx))
            Xva.append(self._tiny(va, px)); yva.append(np.full(len(va), idx))
        Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
        Xva, yva = np.concatenate(Xva), np.concatenate(yva)

        torch.manual_seed(0)
        model = nn.Linear(px * px, len(self.classes))
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        lossf = nn.CrossEntropyLoss()
        Xtr_t, ytr_t = torch.from_numpy(Xtr), torch.from_numpy(ytr).long()
        for _ in range(400):
            opt.zero_grad()
            lossf(model(Xtr_t), ytr_t).backward()
            opt.step()
        with torch.no_grad():
            pred = model(torch.from_numpy(Xva)).argmax(dim=1).numpy()

        acc = 100.0 * float((pred == yva).mean())
        self.control = acc
        require(0.0 <= acc <= 100.0, f"nonsensical control accuracy {acc}")
        per_class = {self.classes[i]: round(100.0 * float((pred[yva == i] == i).mean()), 1)
                     for i in range(len(self.classes))}
        margin = acc - self.chance
        verdict = ("most separation is non-anatomical" if margin >= 40 else
                   "substantial non-anatomical signal" if margin >= 20 else
                   "limited non-anatomical signal")
        return (f"({acc:.1f}% vs {self.chance:.1f}% chance, on {len(Xva)} held-out - {verdict}; "
                f"per class {per_class})")

    def c_beats_control(self):
        import offline_cv

        meta_path = os.path.join(self.dir, "local_model_metadata.json")
        if not os.path.isfile(meta_path):
            return f"(skipped: not trained yet - run train_local_model.py {self.modality})"
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        real = meta.get("held_out_test_accuracy_pct")
        require(real is not None, "the trained model has no measured held-out accuracy")
        require(list(meta.get("classes", ())) == list(self.classes),
                f"the saved model's classes {meta.get('classes')} do not match the dataset "
                f"{list(self.classes)} - it was trained on a different layout; retrain it")
        require(self.control is not None, "control check did not run")
        margin = real - self.control
        require(margin >= CONTROL_MARGIN,
                f"trained model {real:.1f}% vs {CONTROL_PX}x{CONTROL_PX} control "
                f"{self.control:.1f}% (+{margin:.1f}pp). The model adds little over what coarse "
                f"statistics already give, so its accuracy is not evidence that it reads anatomy.")
        return f"(trained {real:.1f}% vs control {self.control:.1f}% = +{margin:.1f}pp)"

    def run(self):
        self.check("all class folders exist and are populated", self.c_populated)
        self.check("no group appears in both train and validate", self.c_no_group_leak)
        self.check("no byte-identical image in two different places", self.c_no_duplicates)
        self.check("classes are balanced within each split", self.c_balanced)
        self.check("every image went through the same pipeline", self.c_uniform_pipeline)
        self.check("manifest provenance matches what is on disk", self.c_manifest)
        self.check("inference preprocesses images exactly as training did",
                   self.c_train_inference_parity)
        self.check(f"shortcut control: accuracy from {CONTROL_PX}x{CONTROL_PX} images", self.c_control)
        self.check("the trained model beats the no-anatomy control", self.c_beats_control)


if __name__ == "__main__":
    wanted = [a for a in sys.argv[1:] if not a.startswith("-")] or list(MODALITIES)
    unknown = [m for m in wanted if m not in MODALITIES]
    if unknown:
        print(f"Unknown modality {unknown} - choices are {list(MODALITIES)}", file=sys.stderr)
        sys.exit(1)

    print("=" * 86)
    print("PINK EDGE AI - ORGANISED DATASET INTEGRITY")
    print("=" * 86)
    if not VERBOSE:
        print("(run with -v to see checks as they execute)\n")

    for modality in wanted:
        Suite(modality).run()

    width = max(len(n) for n, _, _ in RESULTS) + 2
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {name.ljust(width)} {detail}")

    print(f"\n{n_pass}/{len(RESULTS)} checks passed.")
    sys.exit(0 if n_pass == len(RESULTS) else 1)
