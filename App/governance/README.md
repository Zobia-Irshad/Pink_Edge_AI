# Pink Edge AI — Clinical Governance Package

Lightweight, edge-friendly modules that layer data-privacy, resilience, async
processing, secure storage, and clinical-governance guarantees on top of the
existing inference pipeline.

**Design principle:** every module is *stdlib-first*. Any heavy runtime
dependency (Celery, Redis, SQLCipher) is imported lazily and the code degrades
gracefully when it is absent, so the core offline path keeps running on
RAM-constrained edge targets (e.g. RK3588-class SBCs).

## Modules

| Module | Chunk | Purpose |
| --- | --- | --- |
| `circuit_breaker.py` | 2 | Pure-Python circuit breaker. Trips **open after 3 consecutive failures**, stays open **10 min**, then half-opens for one trial. No `pybreaker` dependency. |
| `outbound_guard.py` | 2 | Mandatory PHI-cleaning layer. Strips identifiers, attaches a deterministic **surrogate SHA-256 id**, forwards only the pixel matrix + safe technical fields. `assert_no_phi()` is defence-in-depth. |
| `task_queue.py` | 3 | Non-blocking job submission. Returns an **HTTP 202** ticket immediately. Celery+Redis backend when configured, else a bounded in-process thread pool. |
| `secure_store.py` | 3 | Transactional record store in **WAL mode** (no `database is locked`), optional **SQLCipher AES-256** at rest, append/version-oriented schema with a `frozen` flag. |
| `provenance.py` | 4 | Appends an **immutable** `_provenance` block (model name, version, init timestamp) with a SHA-256 integrity digest; `verify_provenance()` detects tampering. |
| `verification.py` | 4 | **Accept / Modify / Reject** sign-off state machine. Accept freezes the record read-only; a correction creates a **new superseding record** (legally-safe audit trail). |
| `fhir_mapper.py` | 4 | Maps a signed record to an HL7 **FHIR DiagnosticReport + Observation** bundle. Uses the surrogate id only — no PHI leaves in the export. |

## How it is wired in

* `App/Tools/dicom_safe.py` — context-managed DICOM reads + forced `gc.collect()`
  after extracting the pixel matrix (OOM/heap protection). Used by
  `App/Tools/dicom_pipeline.py`.
* `App/Tools/dicom_pipeline.py` — stamps provenance on every inference result
  and attaches a surrogate id (best-effort; never breaks the inference path).
* `inference.py` / `App/inference.py` — the external Roboflow retry helper
  (`_with_retries`) is wrapped by the shared circuit breaker, so a dead endpoint
  short-circuits to the local fallback instead of hanging the thread.

## Optional backends

See the `OPTIONAL` section of `requirements.txt`. Installing `celery`+`redis`
enables multi-process async offload; installing `pysqlcipher3` + setting
`PINKEDGE_DB_KEY` enables AES-256 at rest. `docker-compose.yml` wires the whole
stack with hard `mem_limit` ceilings for process-boundary sandboxing.

## Environment variables

| Variable | Used by | Default |
| --- | --- | --- |
| `PINKEDGE_CELERY_BROKER` | task_queue | *(unset → thread pool)* |
| `PINKEDGE_INFER_WORKERS` | task_queue | `2` |
| `PINKEDGE_DB_PATH` | secure_store | `App/pink_edge_secure.db` |
| `PINKEDGE_DB_KEY` | secure_store | *(unset → plaintext sqlite, WAL)* |

## Tests

```bash
cd App
python Tests/test_governance.py   # 25 tests, no external services needed
python Tests/test_dicom_safe.py   # DICOM fail-safe reader (needs pydicom)
```
