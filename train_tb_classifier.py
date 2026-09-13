from __future__ import annotations

import random
import shutil
import zipfile
from hashlib import sha256
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATASET_ZIP = ROOT / "dataset" / "tbx11k-simplified.zip"
EXTRACT_DIR = ROOT / "dataset" / "tbx11k_raw"
OUTPUT_DIR = ROOT / "dataset" / "tbx11k_cls"
USER_TB_GLOB = "tb chest x ray data*.jpeg"
TRAIN_SPLIT = 0.8
MODEL_OUT = ROOT / "models" / "tb_classifier.pt"


def prepare_dataset() -> Path:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
        if OUTPUT_DIR.exists():
            raise OSError(f"Could not clear dataset directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_ZIP.exists():
        raise FileNotFoundError(f"Dataset zip not found: {DATASET_ZIP}")

    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(DATASET_ZIP) as z:
        z.extractall(EXTRACT_DIR)

    csv_path = EXTRACT_DIR / "data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    class_names = sorted(df["target"].dropna().unique().tolist())
    image_index = {path.name: path for path in (EXTRACT_DIR / "images").rglob("*") if path.is_file()}
    train_dir = OUTPUT_DIR / "train"
    val_dir = OUTPUT_DIR / "val"

    for split_dir in [train_dir, val_dir]:
        for cls in class_names:
            (split_dir / cls).mkdir(parents=True, exist_ok=True)

    skipped = 0
    for cls in class_names:
        class_rows = []
        for _, row in df[df["target"] == cls].iterrows():
            filename = Path(str(row["fname"]).strip()).name
            src = image_index.get(filename)
            if src is None:
                skipped += 1
                continue
            image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            if image is None or image.size == 0:
                skipped += 1
                continue
            class_rows.append((row, src, filename))

        if not class_rows:
            raise ValueError(f"No valid images found for class: {cls}")

        n = len(class_rows)
        train_count = max(1, int(round(n * TRAIN_SPLIT)))
        random.Random(42).shuffle(class_rows)
        train_rows = class_rows[:train_count]
        val_rows = class_rows[train_count:]

        for _, src, filename in train_rows:
            shutil.copy2(src, train_dir / cls / filename)

        for _, src, filename in val_rows:
            shutil.copy2(src, val_dir / cls / filename)

    # User-supplied images are explicitly labeled TB and remain training-only.
    raw_hashes = {sha256(path.read_bytes()).hexdigest() for path in image_index.values()}
    added_user_images = 0
    for src in sorted((ROOT / "dataset").glob(USER_TB_GLOB)):
        image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0 or min(image.shape[:2]) < 128:
            skipped += 1
            continue
        if sha256(src.read_bytes()).hexdigest() in raw_hashes:
            skipped += 1
            continue
        destination = train_dir / "tb" / f"user_{src.name.replace(' ', '_')}"
        shutil.copy2(src, destination)
        added_user_images += 1

    data_yaml = OUTPUT_DIR / "data.yaml"
    data_yaml.write_text(
        "train: ./train\n"
        "val: ./val\n"
        "nc: " + str(len(class_names)) + "\n"
        "names:\n"
        + "\n".join(f"  {i}: {cls}" for i, cls in enumerate(class_names))
        + "\n",
        encoding="utf-8",
    )

    print(f"Prepared dataset with classes: {class_names}")
    print(f"Skipped missing or unreadable images: {skipped}")
    print(f"Added user-labeled TB images to training: {added_user_images}")
    print(f"Train samples: {sum(len(list((OUTPUT_DIR / 'train' / cls).glob('*'))) for cls in class_names)}")
    print(f"Val samples: {sum(len(list((OUTPUT_DIR / 'val' / cls).glob('*'))) for cls in class_names)}")
    print(f"Dataset path: {OUTPUT_DIR}")
    return OUTPUT_DIR


def train_model() -> Path:
    dataset_dir = prepare_dataset()

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n-cls.pt")
    results = model.train(
        data=str(dataset_dir),
        epochs=3,
        imgsz=160,
        batch=64,
        project=str(ROOT / "runs"),
        name="tb_classifier",
        device="cpu",
        fraction=0.25,
        workers=0,
        verbose=True,
        seed=42,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"Best model not found after training: {best_weights}")

    shutil.copy2(best_weights, MODEL_OUT)
    print(f"Saved trained model to {MODEL_OUT}")
    return MODEL_OUT


if __name__ == "__main__":
    random.seed(42)
    train_model()
