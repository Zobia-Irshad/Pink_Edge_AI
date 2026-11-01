"""
verification.py
---------------
Clinical sign-off state machine (Chunk 4).

Gates clinical outputs so the software never emits a *finalized* diagnosis
without human oversight. A clinician moves each AI result through one of three
decisions:

    ACCEPT  -> the record is signed and FROZEN read-only (legally-safe audit).
    MODIFY  -> the clinician's corrected payload is written as a NEW record that
               supersedes the original; the original stays intact.
    REJECT  -> the record is marked rejected (and frozen); no diagnosis is
               emitted from it.

Immutability rule (the heart of the audit trail): once a record is ACCEPTED it
is frozen. Any later correction must create a new entry rather than edit the
signed one, so the history is always reconstructable.

This module is a thin, well-tested orchestration layer over
``secure_store.SecureStore``; it owns the allowed-transition rules.
"""

import uuid
from typing import Any, Dict, Optional

from .secure_store import ImmutableRecordError, SecureStore

# Decisions / statuses.
DRAFT = "draft"
ACCEPTED = "accepted"
REJECTED = "rejected"
MODIFIED = "modified"  # status carried by a superseding (corrected) record

# A result may only be signed off from DRAFT. Accepted/rejected are terminal.
_SIGNABLE_FROM = {DRAFT}


class VerificationError(Exception):
    """Raised on an illegal clinical state transition."""


class VerificationService:
    """Accept / Modify / Reject workflow backed by the secure store."""

    def __init__(self, store: SecureStore):
        self.store = store

    # ------------------------------------------------------------------
    def submit_for_review(
        self,
        surrogate_id: str,
        payload: Dict[str, Any],
        modality: Optional[str] = None,
        record_id: Optional[str] = None,
    ) -> str:
        """Persist a fresh AI result in DRAFT, awaiting clinician review."""
        record_id = record_id or uuid.uuid4().hex
        self.store.save_record(
            record_id=record_id,
            surrogate_id=surrogate_id,
            payload=payload,
            modality=modality,
            status=DRAFT,
        )
        return record_id

    # ------------------------------------------------------------------
    def accept(self, record_id: str, clinician: str) -> Dict[str, Any]:
        """Sign off a draft: freeze it read-only. Terminal."""
        rec = self._require_signable(record_id)
        payload = dict(rec["payload"])
        payload["_signoff"] = {"decision": ACCEPTED, "clinician": clinician}
        # Re-save the signoff into the payload, then freeze.
        self._rewrite_draft_payload(record_id, rec, payload)
        self.store.update_status(record_id, ACCEPTED, freeze=True)
        return self.store.get_record(record_id)

    def reject(self, record_id: str, clinician: str, reason: str = "") -> Dict[str, Any]:
        """Reject a draft: freeze it so no diagnosis is emitted. Terminal."""
        rec = self._require_signable(record_id)
        payload = dict(rec["payload"])
        payload["_signoff"] = {
            "decision": REJECTED,
            "clinician": clinician,
            "reason": reason,
        }
        self._rewrite_draft_payload(record_id, rec, payload)
        self.store.update_status(record_id, REJECTED, freeze=True)
        return self.store.get_record(record_id)

    def modify(
        self,
        record_id: str,
        clinician: str,
        corrected_payload: Dict[str, Any],
    ) -> str:
        """Record a clinician correction as a NEW record superseding the original.

        The original draft is frozen as ``accepted`` with the clinician's name
        (it was reviewed), and a fresh record carrying the corrected payload is
        inserted with ``supersedes`` pointing at it. Returns the new record id.
        """
        rec = self._require_signable(record_id)

        # Freeze the reviewed original so the pre-correction state is preserved.
        original_payload = dict(rec["payload"])
        original_payload["_signoff"] = {
            "decision": MODIFIED,
            "clinician": clinician,
            "note": "superseded by clinician correction",
        }
        self._rewrite_draft_payload(record_id, rec, original_payload)
        self.store.update_status(record_id, MODIFIED, freeze=True)

        new_id = uuid.uuid4().hex
        new_payload = dict(corrected_payload)
        new_payload["_signoff"] = {
            "decision": ACCEPTED,
            "clinician": clinician,
            "corrected_from": record_id,
        }
        self.store.save_record(
            record_id=new_id,
            surrogate_id=rec["surrogate_id"],
            payload=new_payload,
            modality=rec.get("modality"),
            status=ACCEPTED,
            supersedes=record_id,
        )
        # Freeze the corrected, clinician-signed record too.
        self.store.update_status(new_id, ACCEPTED, freeze=True)
        return new_id

    # ------------------------------------------------------------------
    def is_final(self, record_id: str) -> bool:
        """A diagnosis may only be emitted from an accepted, frozen record."""
        rec = self.store.get_record(record_id)
        return bool(rec) and rec["status"] == ACCEPTED and rec["frozen"]

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _require_signable(self, record_id: str) -> Dict[str, Any]:
        rec = self.store.get_record(record_id)
        if rec is None:
            raise VerificationError(f"Unknown record: {record_id}")
        if rec["frozen"] or rec["status"] not in _SIGNABLE_FROM:
            raise VerificationError(
                f"Record {record_id} is already finalized "
                f"(status={rec['status']}, frozen={rec['frozen']}); "
                f"corrections must create a new record."
            )
        return rec

    def _rewrite_draft_payload(
        self, record_id: str, rec: Dict[str, Any], payload: Dict[str, Any]
    ) -> None:
        """Replace a still-draft record's payload in place (pre-freeze only).

        Safe because the record is guaranteed draft+unfrozen by the caller.
        Implemented as a direct UPDATE of the payload column.
        """
        import json

        with self.store._lock:  # reuse the store's connection lock
            self.store._conn.execute(
                "UPDATE records SET payload_json = ?, updated_at = ? "
                "WHERE record_id = ? AND frozen = 0",
                (
                    json.dumps(payload, ensure_ascii=False, default=str),
                    self.store._now(),
                    record_id,
                ),
            )
            self.store._conn.commit()
