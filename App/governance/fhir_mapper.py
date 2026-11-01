"""
fhir_mapper.py
--------------
Map signed Pink Edge records to HL7 FHIR resources (Chunk 4).

Exposes interoperable, machine-readable output so hospital systems can index
results. A completed, clinician-signed record is mapped into a FHIR
``DiagnosticReport`` plus one or more ``Observation`` resources, bundled in a
FHIR ``collection`` Bundle.

Design notes
------------
* Pure stdlib (json / datetime) - no FHIR SDK needed for this subset.
* Patient reference uses the *surrogate id* only; no PHI is embedded, keeping
  the export consistent with the outbound privacy guard.
* The mapping follows the DiagnosticReport / Observation field guidelines
  (status, code, subject, effectiveDateTime, result references, conclusion).
"""

import time
from typing import Any, Dict, List

FHIR_TIME_FMT = "%Y-%m-%dT%H:%M:%SZ"

# Map an internal clinical decision to a FHIR report status.
_STATUS_MAP = {
    "accepted": "final",
    "modified": "amended",
    "rejected": "cancelled",
    "draft": "preliminary",
}


def _now() -> str:
    return time.strftime(FHIR_TIME_FMT, time.gmtime())


def _observation(
    surrogate_id: str,
    obs_id: str,
    code_text: str,
    value: Any,
    effective: str,
) -> Dict[str, Any]:
    """Build one FHIR Observation resource."""
    obs: Dict[str, Any] = {
        "resourceType": "Observation",
        "id": obs_id,
        "status": "final",
        "code": {"text": code_text},
        "subject": {"reference": f"Patient/{surrogate_id}"},
        "effectiveDateTime": effective,
    }
    # Numeric -> valueQuantity; everything else -> valueString.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        obs["valueQuantity"] = {"value": value, "unit": "%"}
    else:
        obs["valueString"] = str(value)
    return obs


def record_to_fhir(record: Dict[str, Any]) -> Dict[str, Any]:
    """Map a signed secure-store record to a FHIR collection Bundle.

    Parameters
    ----------
    record:
        A record dict as returned by ``SecureStore.get_record`` - must carry
        ``surrogate_id``, ``status`` and a ``payload`` (the inference result).

    Returns
    -------
    A FHIR Bundle (type=collection) containing one DiagnosticReport and the
    Observations it references.
    """
    surrogate_id = record.get("surrogate_id", "unknown")
    status = record.get("status", "draft")
    payload = record.get("payload", {}) or {}
    record_id = record.get("record_id", "report")
    effective = record.get("created_at") or _now()
    report_status = _STATUS_MAP.get(status, "preliminary")

    # Build observations from the clinically meaningful fields when present.
    observations: List[Dict[str, Any]] = []
    obs_candidates = {
        "confidence": "AI model confidence",
        "verdict": "AI verdict",
        "bi_rads": "BI-RADS / severity category",
        "acr": "ACR / anatomical zone",
        "referral_priority": "Referral priority",
    }
    idx = 0
    for field, label in obs_candidates.items():
        if field in payload and payload[field] not in (None, ""):
            idx += 1
            observations.append(
                _observation(
                    surrogate_id,
                    f"{record_id}-obs-{idx}",
                    label,
                    payload[field],
                    effective,
                )
            )

    diagnostic_report: Dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "id": record_id,
        "status": report_status,
        "code": {
            "text": payload.get("modality")
            or record.get("modality")
            or "AI imaging screening",
        },
        "subject": {"reference": f"Patient/{surrogate_id}"},
        "effectiveDateTime": effective,
        "issued": _now(),
        "conclusion": payload.get("recommendation")
        or payload.get("verdict")
        or "See referenced observations.",
        "result": [
            {"reference": f"Observation/{obs['id']}"} for obs in observations
        ],
    }

    # Carry the provenance block through as a contained extension-ish note so
    # downstream indexers can see which model configuration produced the result.
    provenance = payload.get("_provenance")
    if provenance:
        diagnostic_report["extension"] = [
            {
                "url": "https://pinkedge.ai/fhir/StructureDefinition/model-provenance",
                "valueString": (
                    f"{provenance.get('model_name')} "
                    f"v{provenance.get('model_version')} "
                    f"@ {provenance.get('initialized_at')}"
                ),
            }
        ]

    entries = [{"resource": diagnostic_report}] + [
        {"resource": obs} for obs in observations
    ]
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "timestamp": _now(),
        "entry": entries,
    }
