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
    Returns:
        image, dataset
    """

    if not os.path.isfile(dicom_path):
        raise FileNotFoundError(dicom_path)

    ds = pydicom.dcmread(dicom_path)

    if not hasattr(ds, "PixelData"):
        raise ValueError("DICOM contains no PixelData.")

    pixels = ds.pixel_array.astype(np.float32)

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

    return image, ds


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

    image, ds = dicom_to_pil(dicom_path)

    modality = str(
        getattr(ds, "Modality", "UNKNOWN")
    ).upper().strip()

    patient_id = str(
        getattr(ds, "PatientID", "UNKNOWN")
    )

    study_uid = str(
        getattr(ds, "StudyInstanceUID", "")
    )

    series_uid = str(
        getattr(ds, "SeriesInstanceUID", "")
    )

    sop_uid = str(
        getattr(ds, "SOPInstanceUID", "")
    )

    print(">>> DICOM PIPELINE")
    print("Patient ID:", patient_id)
    print("Modality:", modality)
    print("Image size:", list(image.size))
    print("Running Pink Edge inference...")

    inference_result = run_inference(
        image,
        modality
    )

    result = {
        "dicom_path": os.path.abspath(dicom_path),
        "patient_id": patient_id,
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