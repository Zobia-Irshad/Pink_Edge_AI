
"""Safe preprocessing for optional TB patient context.

This module validates and encodes metadata only.
It does NOT diagnose patients or modify the existing AI model.
"""

from typing import Any, Mapping


SYMPTOM_FIELDS = ("cough", "fever", "hemoptysis")
HISTORY_FIELDS = ("prior_tb",)

SMOKING_VALUES = {
    "never": "never",
    "former": "former",
    "current": "current",
    "unknown": "unknown",
}

# Feature order is fixed. Do not change it without updating tests
# and any future model trained to use this exact feature schema.
FEATURE_NAMES = (
    "age_normalized",
    "cough",
    "fever",
    "hemoptysis",
    "prior_tb",
    "smoking_never",
    "smoking_former",
    "smoking_current",
)


def _optional_bool(value: Any, field: str):
    """Return True, False, or None for missing/invalid values."""
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    if isinstance(value, int) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized in ("yes", "true", "1"):
            return True

        if normalized in ("no", "false", "0"):
            return False

        if normalized in ("", "unknown", "not provided", "n/a"):
            return None

    raise ValueError(
        f"{field} must be yes/no, true/false, 1/0, or missing."
    )


def _optional_age(value: Any):
    """Return a valid age in years, or None when missing."""
    if value is None:
        return None

    if isinstance(value, bool):
        raise ValueError("Age must be a number between 0 and 120.")

    if isinstance(value, str) and not value.strip():
        return None

    try:
        age = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Age must be a number between 0 and 120."
        ) from exc

    if not 0 <= age <= 120:
        raise ValueError("Age must be between 0 and 120.")

    return age


def _optional_smoking(value: Any):
    """Normalize smoking status without guessing missing information."""
    if value is None:
        return None

    if not isinstance(value, str):
        raise ValueError(
            "Smoking status must be never, former, current, or unknown."
        )

    normalized = value.strip().lower()

    if normalized in ("", "not provided", "n/a"):
        return None

    if normalized not in SMOKING_VALUES:
        raise ValueError(
            "Smoking status must be never, former, current, or unknown."
        )

    return SMOKING_VALUES[normalized]


def preprocess_patient_context(
    data: Mapping[str, Any],
) -> dict:
    """Validate patient metadata and return features plus a missing mask.

    Missing or invalid fields receive a zero placeholder and mask value 0.
    Valid fields receive mask value 1.

    This mask is for preprocessing/testing only. It is NOT yet connected
    to a trained multimodal AI model.
    """
    if not isinstance(data, Mapping):
        raise TypeError("Patient context must be a mapping/dictionary.")

    allowed_fields = {
        "age",
        *SYMPTOM_FIELDS,
        *HISTORY_FIELDS,
        "smoking_status",
    }

    unexpected = set(data.keys()) - allowed_fields
    if unexpected:
        raise ValueError(
            f"Unexpected patient fields: {sorted(map(str, unexpected))}"
        )

    warnings = []
    normalized = {}

    # Validate age.
    try:
        normalized["age"] = _optional_age(data.get("age"))
    except ValueError as exc:
        normalized["age"] = None
        warnings.append(str(exc))

    # Validate symptoms and medical history.
    for field in (*SYMPTOM_FIELDS, *HISTORY_FIELDS):
        try:
            normalized[field] = _optional_bool(
                data.get(field), field
            )
        except ValueError as exc:
            normalized[field] = None
            warnings.append(str(exc))

    # Validate smoking status.
    try:
        normalized["smoking_status"] = _optional_smoking(
            data.get("smoking_status")
        )
    except ValueError as exc:
        normalized["smoking_status"] = None
        warnings.append(str(exc))

    # Create a stable, eight-element numeric feature vector.
    age = normalized["age"]
    features = [
        age / 120.0 if age is not None else 0.0,
    ]

    for field in (*SYMPTOM_FIELDS, *HISTORY_FIELDS):
        value = normalized[field]
        features.append(
            float(value) if value is not None else 0.0
        )

    smoking = normalized["smoking_status"]
    for category in ("never", "former", "current"):
        features.append(
            float(smoking == category)
            if smoking is not None and smoking != "unknown"
            else 0.0
        )

    # One mask value per feature: 1 = valid, 0 = missing/invalid.
    mask = [
        int(age is not None),
    ]

    for field in (*SYMPTOM_FIELDS, *HISTORY_FIELDS):
        mask.append(int(normalized[field] is not None))

    smoking_known = smoking in ("never", "former", "current")
    mask.extend([int(smoking_known)] * 3)

    assert len(features) == len(FEATURE_NAMES) == 8
    assert len(mask) == len(features)
    assert all(value in (0, 1) for value in mask)

    return {
        "normalized": normalized,
        "feature_names": list(FEATURE_NAMES),
        "features": features,
        "mask": mask,
        "warnings": warnings,
        "has_any_clinical_data": any(mask),
    }
