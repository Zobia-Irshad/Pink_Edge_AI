"""
Pink Edge AI - clinical governance package.

Lightweight, edge-friendly modules that add data-privacy, resilience, async
processing, secure storage and clinical-governance guarantees on top of the
existing inference pipeline. Every module is stdlib-first; any heavy runtime
dependency (Celery, SQLCipher, ...) is imported lazily and degrades gracefully
so the core offline path keeps working on RAM-constrained edge targets.

Submodules
----------
circuit_breaker : trip external calls open after repeated failures.
outbound_guard  : seal PHI out of anything sent to external endpoints.
task_queue      : non-blocking inference job submission (202-style).
secure_store    : WAL-mode (optionally SQLCipher-encrypted) record store.
provenance      : immutable model/version/timestamp stamping.
verification    : Accept / Modify / Reject clinical sign-off state machine.
fhir_mapper     : map signed records to HL7 FHIR DiagnosticReport/Observation.
"""

__all__ = [
    "circuit_breaker",
    "outbound_guard",
    "task_queue",
    "secure_store",
    "provenance",
    "verification",
    "fhir_mapper",
]
