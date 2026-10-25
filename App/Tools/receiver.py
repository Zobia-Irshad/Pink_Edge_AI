"""
Pink Edge AI - DICOM C-STORE interoperability bridge.

Receives DICOM studies from imaging equipment,
saves them locally, then sends the received study
through the existing Pink Edge DICOM inference pipeline.
"""

import os
import sys
import json
import logging

from pynetdicom import AE, evt
from pynetdicom.sop_class import (
    Verification,
    CTImageStorage,
    MRImageStorage,
    UltrasoundImageStorage,
    DigitalMammographyXRayImageStorageForPresentation,
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "received_scans"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------
# Allow importing App/Tools/dicom_pipeline.py
# ---------------------------------------------------------

TOOLS_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from dicom_pipeline import process_dicom


# ---------------------------------------------------------
# C-STORE handler
# ---------------------------------------------------------

def handle_store(event):

    print()
    print(">>> C-STORE EVENT RECEIVED")

    try:

        ds = event.dataset
        ds.file_meta = event.file_meta

        patient_id = getattr(
            ds,
            "PatientID",
            "UNKNOWN"
        )

        modality = getattr(
            ds,
            "Modality",
            "UNKNOWN"
        )

        sop_uid = getattr(
            ds,
            "SOPInstanceUID",
            "UNKNOWN"
        )

        print("Patient ID:", patient_id)
        print("Modality:", modality)
        print("SOP Instance UID:", sop_uid)

        # -------------------------------------------------
        # Save DICOM
        # -------------------------------------------------

        filepath = os.path.abspath(
            os.path.join(
                OUTPUT_DIR,
                f"{sop_uid}.dcm"
            )
        )

        print("Saving to:", filepath)

        ds.save_as(
            filepath,
            write_like_original=False
        )

        print(">>> DICOM SAVED SUCCESSFULLY")

        # -------------------------------------------------
        # Run Pink Edge DICOM pipeline
        # -------------------------------------------------

        print()
        print(">>> STARTING PINK EDGE DICOM PIPELINE")

        result = process_dicom(filepath)

        print(">>> DICOM PIPELINE COMPLETED")

        # -------------------------------------------------
        # Save pipeline result
        # -------------------------------------------------

        result_file = os.path.splitext(
            filepath
        )[0] + "_result.json"

        with open(
            result_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                result,
                f,
                indent=2,
                ensure_ascii=False
            )

        print("Result saved to:", result_file)

        # -------------------------------------------------
        # Show important result information
        # -------------------------------------------------

        print()
        print("========== PINK EDGE RESULT ==========")
        print("Modality:", result.get("modality"))
        print("Patient ID:", result.get("patient_id"))
        print("Image size:", result.get("image_size"))
        print("Inference:")

        print(result.get("inference"))

        print("======================================")
        print()

        return 0x0000

    except Exception as exc:

        print()
        print(">>> C-STORE / PIPELINE ERROR")
        print(type(exc).__name__, ":", exc)
        print()

        return 0xC211


# ---------------------------------------------------------
# Application Entity
# ---------------------------------------------------------

ae = AE(
    ae_title=b"MY_LOCAL_PACS"
)

ae.add_supported_context(
    Verification
)

ae.add_supported_context(
    CTImageStorage
)

ae.add_supported_context(
    MRImageStorage
)

ae.add_supported_context(
    UltrasoundImageStorage
)

ae.add_supported_context(
    DigitalMammographyXRayImageStorageForPresentation
)


# ---------------------------------------------------------
# Start DICOM listener
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO
)

print(
    "Bridge listening for hospital scans on port 11112..."
)

ae.start_server(
    ("0.0.0.0", 11112),
    evt_handlers=[
        (evt.EVT_C_STORE, handle_store)
    ]
)