"""
dicom_anonymizer.py
-------------------
Lightweight DICOM privacy utility for Pink Edge AI.

Provides deterministic SHA-256 privacy hashes for patient identifiers
and helpers for anonymizing sensitive DICOM metadata.

No external dependencies are required.
"""

import hashlib
import json
from typing import Any, Dict


SENSITIVE_TAGS = {
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "ReferringPhysicianName",
    "InstitutionName",
}


def generate_hex_privacy_hash(
    pii_dict: Dict[str, Any],
    algorithm: str = "sha256",
) -> str:
    """Generate a deterministic uppercase hexadecimal privacy hash."""
    if not isinstance(pii_dict, dict):
        raise TypeError("pii_dict must be a dictionary")

    canonical = json.dumps(
        pii_dict,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from exc

    hasher.update(canonical.encode("utf-8"))
    return hasher.hexdigest().upper()


def anonymize_dicom_metadata(tag_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Replace sensitive DICOM metadata values with privacy hashes."""
    if not isinstance(tag_dict, dict):
        raise TypeError("tag_dict must be a dictionary")

    result = {}

    for key, value in tag_dict.items():
        if key in SENSITIVE_TAGS:
            result[key] = generate_hex_privacy_hash({key: str(value)})
        else:
            result[key] = value

    return result


def anonymize_dicom_tags(tag_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Backward-compatible alias for anonymize_dicom_metadata()."""
    return anonymize_dicom_metadata(tag_dict)


if __name__ == "__main__":
    sample = {
        "PatientName": "John Doe",
        "PatientID": "12345",
        "PatientBirthDate": "19900101",
        "Modality": "CT",
    }

    print(generate_hex_privacy_hash({
        "pat_id": 12345,
        "cnic": "35201-12345-1",
    }))
    print(anonymize_dicom_metadata(sample))
