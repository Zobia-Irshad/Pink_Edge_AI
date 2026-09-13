"""
dicom_anonymizer.py
--------------------
Lightweight DICOM privacy-hash utility for Pink Edge AI.
Generates a deterministic hex hash from patient PII fields so that
the original identifiers are never stored in logs or session state.

No external dependencies — uses Python's built-in `hashlib`.
"""

import hashlib
import json


def generate_hex_privacy_hash(pii_dict: dict, algorithm: str = "sha256") -> str:
    """
    Accepts a dictionary of PII fields (e.g. pat_id, cnic, dob …) and
    returns a deterministic hex digest that can be stored safely in logs
    without exposing the original values.

    Parameters
    ----------
    pii_dict : dict
        Key-value pairs of patient-identifiable information.
        Example: {"pat_id": 12345678, "cnic": "35201-12345678-1"}
    algorithm : str
        Hash algorithm supported by hashlib (default: "sha256").

    Returns
    -------
    str
        Uppercase hex digest string, e.g. "A3F2…".
    """
    # Serialize to a canonical JSON string (sorted keys → deterministic)
    canonical = json.dumps(pii_dict, sort_keys=True, ensure_ascii=True)
    h = hashlib.new(algorithm)
    h.update(canonical.encode("utf-8"))
    return h.hexdigest().upper()


def anonymize_dicom_tags(tag_dict: dict) -> dict:
    """
    Replace sensitive DICOM tag values with their privacy hashes.
    Returns a new dict safe for logging / export.

    Parameters
    ----------
    tag_dict : dict
        Raw DICOM tags, e.g. {"PatientName": "John Doe", "PatientID": "12345"}

    Returns
    -------
    dict
        Same keys, but sensitive values replaced by hex hashes.
    """
    SENSITIVE_TAGS = {
        "PatientName", "PatientID", "PatientBirthDate",
        "PatientAddress", "PatientTelephoneNumbers",
        "ReferringPhysicianName", "InstitutionName",
    }
    result = {}
    for key, value in tag_dict.items():
        if key in SENSITIVE_TAGS:
            result[key] = generate_hex_privacy_hash({key: str(value)})
        else:
            result[key] = value
    return result
