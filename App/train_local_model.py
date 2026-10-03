#!/usr/bin/env python3
"""
Trains a real, locally-owned binary classifier per modality directly on this project's own labeled
data (Models/<Modality>/Data Set/*.coco train+valid splits, plus any images manually dropped into
Models/<Modality>/positive|negative/ — see offline_cv.py's own docstring for that mechanism),
evaluates it on the dataset's own held-out `test` split, and saves the trained weights to
Models/<Modality>/local_model.pt.

Architecture: torchvision's MobileNetV3-Small with an ImageNet-pretrained backbone (frozen) plus a
freshly-trained linear classifier head — a standard, fast transfer-learning setup that's practical
on CPU-only hardware (no GPU assumed here) while still starting from real learned visual features
instead of random weights, which a from-scratch CNN on a few hundred images would struggle with.

This is a genuine ADDITION to the model lineup, not an assumed replacement: inference.py only moves
a tier ahead of an existing one when it's MEASURABLY better on the same held-out split, matching
every other precedence decision in this project (see offline_cv.py's calibrate() and
Documentations/MODEL_SOURCES.md) — the accuracy this script prints is exactly that measurement.

Usage:
    python train_local_model.py                          # trains all 3 modalities
    python train_local_model.py tb mammography            # trains just these
    python train_local_model.py --epochs 10 --max-per-class 150 --img-size 128
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import offline_cv  # reuse its dataset location / COCO parsing / manual-folder helpers

MODELS_DIR = offline_cv.MODELS_DIR
DEFAULT_IMG_SIZE = 128
DEFAULT_EPOCHS = 6
DEFAULT_MAX_PER_CLASS = 150  # same sample budget as offline_cv.py's MAX_TEMPLATE_SAMPLES
DEFAULT_MAX_TEST_PER_CLASS = 40
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


def _gather_train_paths(modality):
    """Train-time image paths: COCO train+valid (never `test` — held out for evaluation below)
    plus any manually added positive/negative images (Models/<Modality>/positive|negative/)."""
    cfg = offline_cv._DATASETS[modality]
    dataset_dir = offline_cv._dataset_dir(modality)
    pos, neg = set(), set()
    if os.path.isdir(dataset_dir):
        coco_pos, coco_neg = offline_cv._collect_image_paths(
            dataset_dir, cfg["positive_categories"], cfg["negative_categories"], exclude_split="test")
        pos.update(coco_pos)
        neg.update(coco_neg)
    pos.update(offline_cv._manual_image_paths(modality, "positive"))
    neg.update(offline_cv._manual_image_paths(modality, "negative"))
    return sorted(pos), sorted(neg)


def _test_only_paths(modality):
    """Images that exist only in the dataset's own held-out `test` split (never used for
    training) — the same set-difference trick offline_cv.gather_from_dataset() uses."""
    cfg = offline_cv._DATASETS[modality]
    dataset_dir = offline_cv._dataset_dir(modality)
    if not os.path.isdir(dataset_dir):
        return [], []
    all_pos, all_neg = offline_cv._collect_image_paths(
        dataset_dir, cfg["positive_categories"], cfg["negative_categories"], exclude_split=None)
    trainval_pos, trainval_neg = offline_cv._collect_image_paths(
        dataset_dir, cfg["positive_categories"], cfg["negative_categories"], exclude_split="test")
    return sorted(set(all_pos) - set(trainval_pos)), sorted(set(all_neg) - set(trainval_neg))


def _build_dataset(paths_labels, img_size):
    from torch.utils.data import Dataset
    from torchvision import transforms
    from PIL import Image

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])

    class _Ds(Dataset):
        def __len__(self):
            return len(paths_labels)

        def __getitem__(self, idx):
            path, label = paths_labels[idx]
            return tf(Image.open(path).convert("RGB")), label

    return _Ds()


def _build_model():
    import torch.nn as nn
    from torchvision import models

    net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    for p in net.parameters():
        p.requires_grad = False
    in_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_features, 2)  # only this new layer trains
    return net


def train_one_modality(modality, epochs, max_per_class, max_test_per_class, img_size, batch_size=16, lr=1e-3, seed=0):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    subdir = offline_cv._DATASETS[modality]["models_subdir"]
    print(f"\n=== {modality} ===")
    train_pos, train_neg = _gather_train_paths(modality)
    test_pos, test_neg = _test_only_paths(modality)

    rng = random.Random(seed)
    if len(train_pos) > max_per_class:
        train_pos = rng.sample(train_pos, max_per_class)
    if len(train_neg) > max_per_class:
        train_neg = rng.sample(train_neg, max_per_class)
    if len(test_pos) > max_test_per_class:
        test_pos = rng.sample(test_pos, max_test_per_class)
    if len(test_neg) > max_test_per_class:
        test_neg = rng.sample(test_neg, max_test_per_class)

    if not train_pos or not train_neg:
        print(f"  Skipping: need at least one positive AND one negative training image "
              f"(have {len(train_pos)} positive, {len(train_neg)} negative). Add real images to "
              f"Models/{subdir}/positive/ and Models/{subdir}/negative/ (or `python offline_cv.py "
              f"--gather`) and re-run.")
        return None

    train_items = [(p, 1) for p in train_pos] + [(p, 0) for p in train_neg]
    rng.shuffle(train_items)
    test_items = [(p, 1) for p in test_pos] + [(p, 0) for p in test_neg]

    print(f"  train: {len(train_pos)} positive + {len(train_neg)} negative = {len(train_items)} images")
    print(f"  test (held-out, never trained on): {len(test_pos)} positive + {len(test_neg)} negative "
          f"= {len(test_items)} images")

    train_loader = DataLoader(_build_dataset(train_items, img_size), batch_size=batch_size, shuffle=True)

    model = _build_model()
    model.train()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss, n = 0.0, 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            n += images.size(0)
        print(f"  epoch {epoch + 1}/{epochs}: loss {total_loss / max(n, 1):.4f}")

    accuracy = None
    if test_items:
        model.eval()
        test_loader = DataLoader(_build_dataset(test_items, img_size), batch_size=batch_size, shuffle=False)
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                correct += (model(images).argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        accuracy = 100.0 * correct / total if total else None
        print(f"  held-out test accuracy: {correct}/{total} ({accuracy:.1f}%)")
    else:
        print("  no held-out test split available for this dataset — accuracy NOT measured; "
              "be cautious trusting this model until it can be evaluated on unseen data.")

    out_dir = os.path.join(MODELS_DIR, subdir)
    os.makedirs(out_dir, exist_ok=True)
    weights_path = os.path.join(out_dir, "local_model.pt")
    torch.save(model.state_dict(), weights_path)
    meta = {
        "architecture": "mobilenet_v3_small (ImageNet-pretrained backbone, frozen) + linear head (trained)",
        "img_size": img_size, "classes": ["negative", "positive"],
        "train_positive": len(train_pos), "train_negative": len(train_neg),
        "test_positive": len(test_pos), "test_negative": len(test_neg),
        "held_out_test_accuracy_pct": accuracy, "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "local_model_metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  saved: {weights_path}")
    return accuracy


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("modalities", nargs="*", help="Which modalities to train (default: all three)")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--max-per-class", type=int, default=DEFAULT_MAX_PER_CLASS)
    ap.add_argument("--max-test-per-class", type=int, default=DEFAULT_MAX_TEST_PER_CLASS)
    ap.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    modalities = args.modalities or list(offline_cv._DATASETS)
    for m in modalities:
        if m not in offline_cv._DATASETS:
            print(f"Unknown modality {m!r} — choices are {list(offline_cv._DATASETS)}", file=sys.stderr)
            sys.exit(1)

    results = {}
    for m in modalities:
        results[m] = train_one_modality(
            m, args.epochs, args.max_per_class, args.max_test_per_class, args.img_size,
            args.batch_size, args.lr)

    print("\nSummary (held-out test accuracy):")
    for m, acc in results.items():
        print(f"  {m}: {f'{acc:.1f}%' if acc is not None else 'not trained/measured'}")
    print("\nCompare these numbers against the other tiers in Documentations/MODEL_SOURCES.md before "
          "changing any precedence in inference.py — a new model only moves ahead of an existing one "
          "when it measurably beats it on the same held-out split.")


if __name__ == "__main__":
    main()
