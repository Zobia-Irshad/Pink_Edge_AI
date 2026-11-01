"""
provenance.py
-------------
Immutable model-provenance stamping for every prediction (Chunk 4).

Enforces the auditing-trail requirement: append an unalterable structure to
every model output identifying exactly which model configuration created the
prediction - Model Name, Version, and an initialization timestamp.

The stamp is attached under a reserved ``_provenance`` key and, to make
tampering detectable, carries a SHA-256 integrity digest computed over the
inference payload + the stamp fields. ``verify_provenance()`` recomputes that
digest so any later edit of the result is caught.
"""

import hashlib
import json
import time
from typing import Any, Dict, Optional

PROVENANCE_KEY = "_provenance"

# Default identity of the Pink Edge inference stack. Callers should pass the
# concrete model name/version that actually produced a given result (e.g. the
# Roboflow model id in the result's ``source`` field).
DEFAULT_MODEL_NAME = "PinkEdgeAI"
DEFAULT_MODEL_VERSION = "1.0.0"


def _digest(inference_payload: Dict[str, Any], stamp_wo_digest: Dict[str, Any]) -> str:
    """Deterministic SHA-256 over the payload + the stamp (minus the digest)."""
    canonical = json.dumps(
        {"inference": inference_payload, "stamp": stamp_wo_digest},
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


def stamp_provenance(
    result: Dict[str, Any],
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    *,
    initialized_at: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return a copy of ``result`` with an immutable provenance block appended.

    The original dict is not mutated. The model name is auto-derived from the
    result's ``source`` field when not supplied, so real Roboflow/local model
    ids are captured without the caller restating them.
    """
    if not isinstance(result, dict):
        raise TypeError("result must be a dict")

    # Derive the model name from the inference source when not given explicitly.
    if model_name is None:
        source = str(result.get("source", "")) or DEFAULT_MODEL_NAME
        model_name = source
    if model_version is None:
        model_version = DEFAULT_MODEL_VERSION
    if initialized_at is None:
        initialized_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    stamp: Dict[str, Any] = {
        "model_name": model_name,
        "model_version": model_version,
        "initialized_at": initialized_at,
        "stamped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        stamp["extra"] = extra

    # Compute integrity digest over the payload (without the provenance key) and
    # the stamp so any later tampering with either is detectable.
    payload_wo_prov = {k: v for k, v in result.items() if k != PROVENANCE_KEY}
    stamp["integrity_sha256"] = _digest(payload_wo_prov, stamp)

    stamped = dict(payload_wo_prov)
    stamped[PROVENANCE_KEY] = stamp
    return stamped


def verify_provenance(result: Dict[str, Any]) -> bool:
    """Recompute the integrity digest; return True iff the result is untampered."""
    if not isinstance(result, dict) or PROVENANCE_KEY not in result:
        return False
    stamp = dict(result[PROVENANCE_KEY])
    recorded = stamp.pop("integrity_sha256", None)
    if recorded is None:
        return False
    payload_wo_prov = {k: v for k, v in result.items() if k != PROVENANCE_KEY}
    return _digest(payload_wo_prov, stamp) == recorded
