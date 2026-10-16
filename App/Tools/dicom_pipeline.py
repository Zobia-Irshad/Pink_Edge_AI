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
import tempfile

import pynetdicom.sop_class as sop_classes

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.abspath(os.path.join(TOOLS_DIR, ".."))
PROJECT_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)


# Existing Pink Edge inference module
import inference

if __package__:
    from .dicom_adapter import dicom_to_pil as _adapter_dicom_to_pil
else:
    from dicom_adapter import dicom_to_pil as _adapter_dicom_to_pil


SOP_CLASS_MODALITY_NAMES = (
    ("UltrasoundImageStorage", "US"),
    ("DigitalMammographyXRayImageStorageForPresentation", "MG"),
    ("DigitalMammographyXRayImageStorageForProcessing", "MG"),
    ("ComputedRadiographyImageStorage", "CR"),
    ("DigitalXRayImageStorageForPresentation", "DX"),
    ("DigitalXRayImageStorageForProcessing", "DX"),
)


def _expected_modality_for_sop_class(sop_class_uid):
    for sop_class_name, modality in SOP_CLASS_MODALITY_NAMES:
        sop_class = getattr(sop_classes, sop_class_name, None)
        if sop_class is not None and str(sop_class) == sop_class_uid:
            return modality
    return None


def _validate_sop_class_modality(dataset):
    dataset_sop_class = getattr(dataset, "SOPClassUID", None)
    file_meta = getattr(dataset, "file_meta", None)
    file_meta_sop_class = getattr(file_meta, "MediaStorageSOPClassUID", None)

    dataset_sop_class = str(dataset_sop_class) if dataset_sop_class else None
    file_meta_sop_class = str(file_meta_sop_class) if file_meta_sop_class else None

    if (
        dataset_sop_class is not None
        and file_meta_sop_class is not None
        and dataset_sop_class != file_meta_sop_class
    ):
        return None, {"status": "sop_class_mismatch"}

    sop_class_uid = dataset_sop_class or file_meta_sop_class
    expected_modality = _expected_modality_for_sop_class(sop_class_uid)
    if expected_modality is None:
        return None, {"status": "unsupported_sop_class"}

    modality = str(getattr(dataset, "Modality", "")).upper().strip()
    if modality == "MAMMOGRAPHY":
        modality = "MG"
    if modality != expected_modality:
        return None, {"status": "modality_mismatch"}

    return expected_modality, None


# ---------------------------------------------------------
# DICOM -> PIL
# ---------------------------------------------------------

def dicom_to_pil(dicom_path):
    if not os.path.isfile(dicom_path):
        raise FileNotFoundError(dicom_path)

    image, ds = _adapter_dicom_to_pil(dicom_path)
    image = image.convert("RGB")
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

    if modality == "US":
        predictor = inference.predict_maternal
    elif modality in ("MG", "MAMMOGRAPHY"):
        predictor = inference.predict_mammography
    elif modality in ("CR", "DX"):
        predictor = inference.predict_tb
    else:
        return {"status": "unsupported_modality"}

    try:
        result = predictor(image, allow_network=False)
    except Exception as exc:
        return {"status": "inference_failed", "error_type": type(exc).__name__}

    required_fields = ("verdict", "confidence", "source", "is_critical", "css")
    if result is None:
        return {"status": "inference_unavailable"}
    if not isinstance(result, dict) or not all(field in result for field in required_fields):
        return {"status": "inference_failed", "error_type": "invalid_result"}

    return {"status": "analysis_completed", "inference": result}


# ---------------------------------------------------------
# Main DICOM processing function
# ---------------------------------------------------------

def process_dicom(dicom_path):
    """
    Complete DICOM -> Pink Edge inference pipeline.

    Returns a JSON-serializable dictionary.
    """

    try:
        image, ds = dicom_to_pil(dicom_path)
    except Exception as exc:
        return {"status": "invalid_dicom", "error_type": type(exc).__name__}

    modality = str(
        getattr(ds, "Modality", "UNKNOWN")
    ).upper().strip()

    expected_modality, validation_error = _validate_sop_class_modality(ds)
    if validation_error is not None:
        return {
            **validation_error,
            "image_size": [image.width, image.height],
        }
    if modality == "MAMMOGRAPHY":
        modality = "MG"

    inference_result = run_inference(image, modality)
    result = {
        "status": inference_result["status"],
        "modality": modality,
        "image_size": [image.width, image.height],
    }
    if "inference" in inference_result:
        result["inference"] = inference_result["inference"]
    if "error_type" in inference_result:
        result["error_type"] = inference_result["error_type"]
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
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".result-",
        suffix=".tmp",
        dir=os.path.dirname(os.path.abspath(result_path)),
    )

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as result_file:
            json.dump(result, result_file, indent=2, ensure_ascii=False)
        os.replace(temporary_path, result_path)
        return result_path
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


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

    result = process_dicom(dicom_path)
    save_result(dicom_path, result)
    print(f"DICOM analysis status: {result['status']}.")