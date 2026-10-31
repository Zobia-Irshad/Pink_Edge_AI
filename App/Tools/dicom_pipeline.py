"""
Pink Edge AI - DICOM inference pipeline.

Purpose:
    1. Read a received DICOM file.
    2. Convert its PixelData to a PIL image.
    3. Route the image to the existing Pink Edge inference functions.
    4. Return the inference result together with DICOM metadata.

This file does NOT replace inference.py.
It is only the interoperability layer between:
    DICOM/PACS -> received_scans -> existing inference.py
"""

"""
Pink Edge AI - DICOM to AI inference pipeline.

Flow:
Hospital modality
    -> DICOM C-STORE receiver
    -> received_scans/*.dcm
    -> DICOM pixel extraction
    -> PIL image
    -> existing Pink Edge inference functions
    -> *_result.json

This module does not replace the existing inference system.
It only connects DICOM input to the existing inference functions.
"""

import os
import sys
import json

import numpy as np
import pydicom
from PIL import Image

# Memory-safe DICOM helpers (Chunk 1 fail-safes): context-managed file handles
# and a forced gc cycle after extracting the high-dimensional pixel matrix.
try:
    from dicom_safe import dicom_reader, extract_pixel_array
except ImportError:  # pragma: no cover - fallback when imported as a package
    from .dicom_safe import dicom_reader, extract_pixel_array


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.abspath(os.path.join(TOOLS_DIR, ".."))
PROJECT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)


# Existing Pink Edge inference module
import inference


# ---------------------------------------------------------
# DICOM -> PIL
# ---------------------------------------------------------

def dicom_to_pil(dicom_path):
    """
    Read a DICOM file and convert its PixelData to a PIL RGB image.

    The dataset is read inside a context manager and released before this
    returns, so only the image plus a lightweight metadata snapshot survive.

    Returns:
        image, metadata_dict
    """

    if not os.path.isfile(dicom_path):
        raise FileNotFoundError(dicom_path)

    # Read inside a context manager so the file handle is released immediately,
    # and extract the pixel matrix through the helper that forces a gc cycle
    # right after the high-dimensional decode (OOM / heap protection).
    with dicom_reader(dicom_path) as ds:
        pixels = extract_pixel_array(ds).astype(np.float32)

        # Capture the lightweight metadata we need before the dataset is
        # released on context exit.
        ds_meta = _snapshot_metadata(ds)

    # Handle multi-frame DICOM.
    if pixels.ndim > 2:
        pixels = pixels[0]

    minimum = float(pixels.min())
    maximum = float(pixels.max())

    if maximum > minimum:
        pixels = (
            (pixels - minimum)
            / (maximum - minimum)
            * 255.0
        )
    else:
        pixels = np.zeros_like(pixels)

    pixels = pixels.astype(np.uint8)

    image = Image.fromarray(pixels).convert("RGB")

    return image, ds_meta


# Metadata blocks the task calls for (Patient ID, Study/Series ID, Modality).
# Snapshotting them into a plain dict lets the heavy dataset be released by the
# dicom_reader context manager before inference runs.
_METADATA_FIELDS = (
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "Modality",
)


def _snapshot_metadata(ds):
    """Copy the lightweight metadata blocks out of a DICOM dataset to a dict."""
    return {
        field: str(getattr(ds, field, ""))
        for field in _METADATA_FIELDS
    }


# ---------------------------------------------------------
# Existing Pink Edge inference routing
# ---------------------------------------------------------

def run_inference(image, modality):
    """
    Route the DICOM image to the existing Pink Edge inference
    function according to DICOM Modality.
    """

    modality = str(modality).upper().strip()

    # Ultrasound
    if modality == "US":
        return inference.predict_maternal(image)

    # Mammography
    if modality in ("MG", "MAMMOGRAPHY"):
        return inference.predict_mammography(image)

    # Chest X-ray / radiography
    if modality in ("CR", "DX"):
        return inference.predict_tb(image)

    # CT
    # There is currently no dedicated CT inference function
    # in the inspected Pink Edge inference.py.
    if modality == "CT":
        return {
            "status": "not_implemented",
            "message": "DICOM CT received, but no dedicated CT inference function is currently configured."
        }

    return {
        "status": "unsupported_modality",
        "message": f"Unsupported DICOM modality: {modality}"
    }


# ---------------------------------------------------------
# Main DICOM processing function
# ---------------------------------------------------------

def process_dicom(dicom_path):
    """
    Complete DICOM -> Pink Edge inference pipeline.

    Returns a JSON-serializable dictionary.
    """

    image, ds_meta = dicom_to_pil(dicom_path)

    modality = (ds_meta.get("Modality") or "UNKNOWN").upper().strip()

    patient_id = ds_meta.get("PatientID") or "UNKNOWN"

    study_uid = ds_meta.get("StudyInstanceUID") or ""

    series_uid = ds_meta.get("SeriesInstanceUID") or ""

    sop_uid = ds_meta.get("SOPInstanceUID") or ""

    print(">>> DICOM PIPELINE")
    print("Patient ID:", patient_id)
    print("Modality:", modality)
    print("Image size:", list(image.size))
    print("Running Pink Edge inference...")

    inference_result = run_inference(
        image,
        modality
    )

    # Governance (Chunk 2 + 4): stamp an immutable model-provenance block on the
    # inference output and derive a surrogate id so results can be correlated
    # downstream without exposing the raw PatientID. Best-effort: never let a
    # governance import failure break the core inference path.
    surrogate_id = None
    try:
        from governance.provenance import stamp_provenance
        from governance.outbound_guard import build_surrogate_id

        if isinstance(inference_result, dict):
            inference_result = stamp_provenance(inference_result)
        surrogate_id = build_surrogate_id(
            {"PatientID": patient_id, "StudyInstanceUID": study_uid}
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(">>> GOVERNANCE STAMP SKIPPED:", type(exc).__name__, exc)

    result = {
        "dicom_path": os.path.abspath(dicom_path),
        "patient_id": patient_id,
        "surrogate_id": surrogate_id,
        "study_instance_uid": study_uid,
        "series_instance_uid": series_uid,
        "sop_instance_uid": sop_uid,
        "modality": modality,
        "image_size": [
            image.width,
            image.height
        ],
        "inference": inference_result
    }

    return result


# ---------------------------------------------------------
# Save result JSON
# ---------------------------------------------------------

def save_result(dicom_path, result):
    """
    Save inference result beside the original DICOM file.
    """

    base, _ = os.path.splitext(dicom_path)

    result_path = base + "_result.json"

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    return result_path


# ---------------------------------------------------------
# Command-line testing
# ---------------------------------------------------------

if __name__ == "__main__":

    received_dir = os.path.join(
        PROJECT_DIR,
        "received_scans"
    )

    if not os.path.isdir(received_dir):
        print("No received_scans directory found.")
        sys.exit(1)

    dicom_files = [
        os.path.join(received_dir, name)
        for name in os.listdir(received_dir)
        if name.lower().endswith(".dcm")
    ]

    if not dicom_files:
        print("No DICOM files found.")
        sys.exit(1)

    dicom_path = sorted(
        dicom_files,
        key=os.path.getmtime
    )[-1]

    print("Testing DICOM:")
    print(dicom_path)
    print()

    result = process_dicom(dicom_path)

    result_path = save_result(
        dicom_path,
        result
    )

    print()
    print("========== FINAL PIPELINE RESULT ==========")
    print(json.dumps(
        result,
        indent=2,
        ensure_ascii=False
    ))

    print()
    print("Result JSON:")
    print(result_path)