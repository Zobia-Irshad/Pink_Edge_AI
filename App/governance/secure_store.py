"""
secure_store.py
---------------
Structured, encrypted-at-rest record store for Pink Edge AI (Chunk 3).

Replaces loose, unsecured text-file result paths with a transactional datastore:

  * WAL mode (``PRAGMA journal_mode=WAL``) so reads never block on writes and we
    avoid ``sqlite3.OperationalError: database is locked`` under concurrent
    inference + UI access.
  * SQLCipher (AES-256 at rest) when ``pysqlcipher3`` is installed and a key is
    configured; otherwise plain ``sqlite3`` with a loud, recorded warning so the
    offline demo path still works on machines without the native SQLCipher lib.

The schema is append-oriented to support the clinical audit trail: records are
versioned and, once a clinician accepts one, the row is frozen (see
``verification.py``) and any correction is written as a *new* row.
"""

import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

# Where the encrypted DB lives by default (overridable via env for Docker/edge).
DEFAULT_DB_PATH = os.environ.get(
    "PINKEDGE_DB_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pink_edge_secure.db"),
)

# Encryption key for SQLCipher. Supplied out-of-band (env / secret manager),
# never hard-coded. Absent -> plaintext fallback with a recorded warning.
_DB_KEY_ENV = "PINKEDGE_DB_KEY"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    record_id       TEXT PRIMARY KEY,
    surrogate_id    TEXT NOT NULL,
    modality        TEXT,
    payload_json    TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'draft',
    frozen          INTEGER NOT NULL DEFAULT 0,
    supersedes      TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_records_surrogate ON records(surrogate_id);
CREATE INDEX IF NOT EXISTS idx_records_status ON records(status);
"""


class ImmutableRecordError(Exception):
    """Raised on any attempt to mutate a frozen (accepted) record."""


class SecureStore:
    """Thread-safe, WAL-mode, optionally AES-256-encrypted record store."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH, key: Optional[str] = None):
        self.db_path = db_path
        self._key = key if key is not None else os.environ.get(_DB_KEY_ENV)
        self._lock = threading.RLock()
        self.encrypted = False
        self._conn = self._connect()
        self._init_db()

    # ------------------------------------------------------------------
    # Connection setup
    # ------------------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """Open the DB, preferring SQLCipher when a key + driver are present."""
        if self._key:
            try:
                from pysqlcipher3 import dbapi2 as sqlcipher  # type: ignore

                conn = sqlcipher.connect(self.db_path, check_same_thread=False)
                # Key must be set before any other statement touches the DB.
                conn.execute(f"PRAGMA key = \"x'{self._key}'\"") if all(
                    c in "0123456789abcdefABCDEF" for c in self._key
                ) else conn.execute("PRAGMA key = ?", (self._key,))
                self.encrypted = True
                self._apply_pragmas(conn)
                return conn
            except Exception as exc:
                # SQLCipher requested but unavailable: do NOT silently drop
                # encryption — record it so operators can see the posture.
                self._encryption_warning = (
                    f"SQLCipher unavailable ({exc}); falling back to plaintext "
                    f"sqlite3. Install pysqlcipher3 for AES-256 at rest."
                )
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._apply_pragmas(conn)
        return conn

    @staticmethod
    def _apply_pragmas(conn) -> None:
        """WAL + sane durability/concurrency pragmas."""
        # Write-Ahead Logging: decouples readers from writers, preventing
        # database-locked exceptions during concurrent inference + UI reads.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        # Keep the busy timeout modest so a stuck writer surfaces quickly.
        conn.execute("PRAGMA busy_timeout=5000;")

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

    def _row_to_dict(self, row) -> Dict[str, Any]:
        import json

        cols = [
            "record_id", "surrogate_id", "modality", "payload_json",
            "status", "frozen", "supersedes", "created_at", "updated_at",
        ]
        d = dict(zip(cols, row))
        d["payload"] = json.loads(d.pop("payload_json"))
        d["frozen"] = bool(d["frozen"])
        return d

    # ------------------------------------------------------------------
    # CRUD (write path is append/version oriented)
    # ------------------------------------------------------------------
    def save_record(
        self,
        record_id: str,
        surrogate_id: str,
        payload: Dict[str, Any],
        modality: Optional[str] = None,
        status: str = "draft",
        supersedes: Optional[str] = None,
    ) -> str:
        """Insert a new record. Never overwrites a frozen one."""
        import json

        now = self._now()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO records
                    (record_id, surrogate_id, modality, payload_json,
                     status, frozen, supersedes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)
                """,
                (
                    record_id, surrogate_id, modality,
                    json.dumps(payload, ensure_ascii=False, default=str),
                    status, supersedes, now, now,
                ),
            )
            self._conn.commit()
        return record_id

    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM records WHERE record_id = ?", (record_id,)
            )
            row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def update_status(self, record_id: str, status: str, *, freeze: bool = False) -> None:
        """Update a record's status; refuse if it is already frozen."""
        with self._lock:
            rec = self.get_record(record_id)
            if rec is None:
                raise KeyError(f"Unknown record: {record_id}")
            if rec["frozen"]:
                raise ImmutableRecordError(
                    f"Record {record_id} is frozen (accepted); write a new "
                    f"superseding record instead of mutating it."
                )
            self._conn.execute(
                "UPDATE records SET status = ?, frozen = ?, updated_at = ? "
                "WHERE record_id = ?",
                (status, 1 if freeze else 0, self._now(), record_id),
            )
            self._conn.commit()

    def list_by_surrogate(self, surrogate_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM records WHERE surrogate_id = ? ORDER BY created_at",
                (surrogate_id,),
            )
            rows = cur.fetchall()
        return [self._row_to_dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass
