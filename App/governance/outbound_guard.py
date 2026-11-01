"""
outbound_guard.py
-----------------
Mandatory data-cleaning layer for anything leaving the device (Chunk 2).

Goal: completely seal the *unanonymized* outbound path to external
infrastructure (e.g. Roboflow). Before any model transmission we:

  1. Wipe every identifiable Protected Health Information (PHI) field -
     names, institution names, precise birthdates, addresses, phone numbers,
     referring physicians, accession numbers, raw patient/study identifiers.
  2. Replace true identity fields with a securely generated *surrogate*
     identifier hash (deterministic SHA-256), so a study can still be
     correlated downstream without exposing the real identity.
  3. Send only the raw pixel matrix plus that surrogate id and the handful of
     non-identifying technical fields the model actually needs (Modality,
     Rows, Columns).

This reuses the existing stdlib-only ``dicom_anonymizer`` hashing so there is
one canonical privacy-hash implementation across the project.
"""

import os
import sys
from typing import Any, Dict, Optional

# Reuse the project's canonical privacy-hash helper. It lives at the App root;
# add that to the path when imported as a loose module.
_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

try:  # package import first, then flat import
    from ..dicom_anonymizer import generate_hex_privacy_hash
except (ImportError, ValueError):  # pragma: no cover - flat layout fallback
    from dicom_anonymizer import generate_hex_privacy_hash


# Fields that must NEVER cross the network boundary. Anything matching these
# (case-insensitive, substring) is dropped before transmission.
PHI_FIELDS = frozenset(
    {
        "patientname",
        "patientid",
        "patientbirthdate",
        "patientbirthtime",
        "patientaddress",
        "patienttelephonenumbers",
        "patientmothersbirthname",
        "otherpatientids",
        "otherpatientnames",
        "referringphysicianname",
        "performingphysicianname",
        "operatorsname",
        "institutionname",
        "institutionaddress",
        "accessionnumber",
        "ethnicgroup",
        "occupation",
        "additionalpatienthistory",
        "name",
        "address",
        "phone",
        "birthdate",
        "dob",
        "ssn",
        "mrn",
        "cnic",
    }
)

# Non-identifying technical fields that are safe to forward to the model.
SAFE_TECHNICAL_FIELDS = frozenset(
    {"Modality", "Rows", "Columns", "PhotometricInterpretation", "BitsAllocated"}
)


class PHILeakError(Exception):
    """Raised when a payload still contains PHI after cleaning (defence in depth)."""


def _is_phi(field_name: str) -> bool:
    lowered = str(field_name).replace("_", "").replace(" ", "").lower()
    return any(marker in lowered for marker in PHI_FIELDS)


def build_surrogate_id(identity: Dict[str, Any]) -> str:
    """Generate a securely-derived surrogate id from true identity fields.

    Deterministic so the same patient/study maps to the same surrogate across
    calls, enabling downstream correlation without exposing the real identity.
    """
    if not identity:
        # Still return a stable, non-empty token rather than leaking emptiness.
        identity = {"_": "anonymous"}
    return generate_hex_privacy_hash(identity)


def scrub_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Drop every PHI field, keep only safe technical fields."""
    if not metadata:
        return {}
    cleaned: Dict[str, Any] = {}
    for key, value in metadata.items():
        if _is_phi(key):
            continue
        if key in SAFE_TECHNICAL_FIELDS:
            cleaned[key] = value
    return cleaned


def build_outbound_payload(
    pixel_matrix: Any,
    metadata: Optional[Dict[str, Any]] = None,
    identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construct the only structure allowed to leave the device.

    Parameters
    ----------
    pixel_matrix:
        The raw image array / PIL image to run inference on. Passed through
        untouched - pixels are not PHI once metadata is stripped.
    metadata:
        Full DICOM metadata dict. PHI is removed; safe technical fields kept.
    identity:
        True identity fields (PatientID, PatientName, ...) used *only* to derive
        the surrogate id. Never included verbatim in the result.

    Returns
    -------
    dict with keys ``surrogate_id``, ``pixels`` and ``metadata`` (scrubbed).
    """
    if identity is None:
        # Derive identity from whatever identifying fields are in metadata.
        identity = {
            k: v for k, v in (metadata or {}).items() if _is_phi(k)
        }

    payload = {
        "surrogate_id": build_surrogate_id(identity),
        "pixels": pixel_matrix,
        "metadata": scrub_metadata(metadata),
    }

    assert_no_phi(payload)
    return payload


def assert_no_phi(payload: Dict[str, Any]) -> None:
    """Defence-in-depth: verify no PHI field name survived in the payload.

    Only inspects the metadata block (pixels are a binary matrix, surrogate_id
    is a hash). Raises ``PHILeakError`` if a PHI key is found.
    """
    meta = payload.get("metadata") or {}
    leaked = [k for k in meta if _is_phi(k)]
    if leaked:
        raise PHILeakError(
            f"Outbound payload still contains PHI fields: {sorted(leaked)}"
        )
