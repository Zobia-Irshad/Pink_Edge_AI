
"""Build a structured clinical-context packet.

This module validates patient context using the existing preprocessing
module. It does not alter or run any AI model.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping


try:
    from .preprocessing import (
        FEATURE_NAMES,
        preprocess_patient_context,
    )
except ImportError:
    from preprocessing import (
        FEATURE_NAMES,
        preprocess_patient_context,
    )


@dataclass
class ClinicalContextPacket:
    """Validated patient context associated with an imaging workflow."""

    schema_version: str
    patient_id: str
    modality: str
    created_at: str
    clinical_context: Dict[str, Any]
    preprocessing_result: Dict[str, Any]
    missing_fields: list

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary suitable for workflow/report integration."""
        return asdict(self)


def build_context_packet(
    patient_id: str,
    modality: str,
    clinical_context: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate clinical fields and create a structured context packet.

    Args:
        patient_id: Existing patient identifier.
        modality: Workflow modality, e.g. 'tuberculosis'.
        clinical_context: Patient fields supported by preprocessing.py.

    Returns:
        A dictionary containing the original context, preprocessing output,
        and fields that were not supplied.

    Raises:
        TypeError: If arguments have the wrong type.
        ValueError: If patient_id or modality is empty.
        Validation errors from preprocess_patient_context are propagated.
    """
    if not isinstance(patient_id, str) or not patient_id.strip():
        raise ValueError("patient_id must be a non-empty string.")

    if not isinstance(modality, str) or not modality.strip():
        raise ValueError("modality must be a non-empty string.")

    if not isinstance(clinical_context, Mapping):
        raise TypeError("clinical_context must be a dictionary or mapping.")

    raw_context = deepcopy(dict(clinical_context))

    # Use the existing validator/normalizer; do not duplicate its logic.
    processed = preprocess_patient_context(deepcopy(raw_context))

    if not isinstance(processed, Mapping):
        raise TypeError(
            "preprocess_patient_context() must return a dictionary-like "
            "result containing the processed features and missing-value mask."
        )

    norm = processed["normalized"]
    missing_fields = [k for k, v in norm.items() if v is None or v == "unknown"]

    packet = ClinicalContextPacket(
        schema_version="1.0",
        patient_id=patient_id.strip(),
        modality=modality.strip(),
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        clinical_context=raw_context,
        preprocessing_result=deepcopy(dict(processed)),
        missing_fields=missing_fields,
    )

    return packet.to_dict()
