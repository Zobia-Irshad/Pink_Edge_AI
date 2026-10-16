"""
Pink Edge AI - DICOM C-STORE interoperability bridge.

Receives DICOM studies from imaging equipment,
saves them locally, then sends the received study
through the existing Pink Edge DICOM inference pipeline.
"""

import hashlib
import json
import os
import re
import sys
import threading
import tempfile
from contextlib import contextmanager

from pynetdicom import AE, evt
from pynetdicom.sop_class import Verification
import pynetdicom.sop_class as sop_classes


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

from dicom_pipeline import process_dicom, save_result

STORAGE_SOP_CLASS_NAMES = (
    "UltrasoundImageStorage",
    "DigitalMammographyXRayImageStorageForPresentation",
    "DigitalMammographyXRayImageStorageForProcessing",
    "ComputedRadiographyImageStorage",
    "DigitalXRayImageStorageForPresentation",
    "DigitalXRayImageStorageForProcessing",
)


class DuplicateSOPInstanceError(Exception):
    pass


_SOP_INSTANCE_LOCKS = {}
_SOP_INSTANCE_LOCKS_GUARD = threading.Lock()


def _safe_log(message):
    try:
        print(message)
    except Exception:
        pass


@contextmanager
def _sop_instance_lock(sop_instance_uid):
    with _SOP_INSTANCE_LOCKS_GUARD:
        entry = _SOP_INSTANCE_LOCKS.get(sop_instance_uid)
        if entry is None:
            entry = {"lock": threading.Lock(), "users": 0}
            _SOP_INSTANCE_LOCKS[sop_instance_uid] = entry
        entry["users"] += 1

    acquired = False
    try:
        entry["lock"].acquire()
        acquired = True
        yield
    finally:
        try:
            if acquired:
                entry["lock"].release()
        finally:
            with _SOP_INSTANCE_LOCKS_GUARD:
                entry["users"] -= 1
                if (
                    entry["users"] == 0
                    and _SOP_INSTANCE_LOCKS.get(sop_instance_uid) is entry
                ):
                    del _SOP_INSTANCE_LOCKS[sop_instance_uid]


def _validate_sop_instance_uid(sop_instance_uid):
    if (
        not isinstance(sop_instance_uid, str)
        or len(sop_instance_uid) > 64
        or re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", sop_instance_uid) is None
    ):
        raise ValueError("invalid SOP Instance UID")


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def _save_received_instance(dataset, sop_instance_uid):
    _validate_sop_instance_uid(sop_instance_uid)

    destination = os.path.join(OUTPUT_DIR, f"{sop_instance_uid}.dcm")
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=".incoming-",
        suffix=".dcm",
        dir=OUTPUT_DIR,
    )
    durably_stored = False
    try:
        os.close(descriptor)
        dataset.save_as(temporary_path, write_like_original=False)
        try:
            os.link(temporary_path, destination)
            durably_stored = True
            return destination, True
        except FileExistsError:
            if _sha256_file(temporary_path) == _sha256_file(destination):
                durably_stored = True
                return destination, False
            raise DuplicateSOPInstanceError from None
    finally:
        exception_in_flight = sys.exc_info()[0] is not None
        try:
            os.unlink(temporary_path)
        except FileNotFoundError:
            pass
        except OSError:
            _safe_log("Temporary DICOM cleanup failed.")
            if not exception_in_flight and not durably_stored:
                raise


def _has_completed_analysis(dicom_path):
    result_path = os.path.splitext(dicom_path)[0] + "_result.json"
    try:
        with open(result_path, "r", encoding="utf-8") as result_file:
            result = json.load(result_file)
    except Exception:
        return False

    if not isinstance(result, dict) or result.get("status") != "analysis_completed":
        return False

    inference_result = result.get("inference")
    required_fields = ("verdict", "confidence", "source", "is_critical", "css")
    return isinstance(inference_result, dict) and all(
        field in inference_result for field in required_fields
    )


# ---------------------------------------------------------
# C-STORE handler
# ---------------------------------------------------------

def handle_store(event):
    durably_stored = False
    try:
        ds = event.dataset
        ds.file_meta = event.file_meta
        sop_instance_uid = str(getattr(ds, "SOPInstanceUID", ""))
        _validate_sop_instance_uid(sop_instance_uid)

        with _sop_instance_lock(sop_instance_uid):
            filepath, is_new = _save_received_instance(ds, sop_instance_uid)
            durably_stored = True
            if not is_new and _has_completed_analysis(filepath):
                _safe_log("Duplicate DICOM instance ignored; completed analysis preserved.")
                return 0x0000

            try:
                result = process_dicom(filepath)
            except Exception:
                result = {"status": "inference_failed", "error_type": "processing_error"}

            try:
                save_result(filepath, result)
            except Exception:
                _safe_log("DICOM stored; analysis result could not be saved.")
                return 0x0000

            _safe_log(
                f"DICOM stored; analysis status: {result.get('status', 'unknown')}."
            )
            return 0x0000
    except DuplicateSOPInstanceError:
        _safe_log("Conflicting duplicate DICOM instance rejected; existing scan preserved.")
        return 0xC211
    except Exception:
        _safe_log("DICOM receipt handling failed.")
        return 0x0000 if durably_stored else 0xC211


# ---------------------------------------------------------
# Application Entity
# ---------------------------------------------------------

ae = AE(
    ae_title=b"MY_LOCAL_PACS"
)

ae.add_supported_context(
    Verification
)

for sop_class_name in STORAGE_SOP_CLASS_NAMES:
    sop_class = getattr(sop_classes, sop_class_name, None)
    if sop_class is not None:
        ae.add_supported_context(sop_class)


# ---------------------------------------------------------
# Start DICOM listener
# ---------------------------------------------------------

def main():
    print(
        "DICOM C-STORE receiver ready on port 11112."
    )

    ae.start_server(
        ("0.0.0.0", 11112),
        evt_handlers=[
            (evt.EVT_C_STORE, handle_store)
        ]
    )


if __name__ == "__main__":
    main()