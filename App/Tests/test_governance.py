"""
Pink Edge AI - unit tests for the clinical governance package.

Covers all four chunks' added logic with no external services required
(Celery/Redis/SQLCipher are optional; these tests exercise the stdlib paths):

  * Chunk 2: circuit_breaker, outbound_guard
  * Chunk 3: task_queue (thread backend), secure_store (WAL + immutability)
  * Chunk 4: provenance, verification, fhir_mapper

Run:  python Tests/test_governance.py        (from App/)
"""

import os
import sys
import tempfile
import time
import unittest

# Tests/ lives one level below App/, where the `governance` package lives.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from governance.circuit_breaker import (  # noqa: E402
    CircuitBreaker, CircuitBreakerOpen, CLOSED, OPEN, HALF_OPEN,
)
from governance.outbound_guard import (  # noqa: E402
    build_outbound_payload, scrub_metadata, assert_no_phi, PHILeakError,
)
from governance import task_queue as tq  # noqa: E402
from governance.secure_store import SecureStore, ImmutableRecordError  # noqa: E402
from governance.provenance import (  # noqa: E402
    stamp_provenance, verify_provenance, PROVENANCE_KEY,
)
from governance.verification import VerificationService, VerificationError  # noqa: E402
from governance.fhir_mapper import record_to_fhir  # noqa: E402


class TestCircuitBreaker(unittest.TestCase):
    def setUp(self):
        self.clock = [0.0]
        self.cb = CircuitBreaker(
            failure_threshold=3, reset_timeout=600, time_fn=lambda: self.clock[0]
        )

    def _boom(self):
        raise RuntimeError("endpoint down")

    def test_trips_open_after_three_consecutive_failures(self):
        for _ in range(2):
            self.cb.call(self._boom, fallback=lambda: "local")
            self.assertEqual(self.cb.state, CLOSED)
        # third failure trips it
        self.cb.call(self._boom, fallback=lambda: "local")
        self.assertEqual(self.cb.state, OPEN)

    def test_open_skips_call_and_uses_fallback(self):
        for _ in range(3):
            self.cb.call(self._boom, fallback=lambda: "local")
        # While open, the real fn must NOT be invoked at all.
        called = {"n": 0}

        def should_not_run():
            called["n"] += 1
            return "cloud"

        self.assertEqual(self.cb.call(should_not_run, fallback=lambda: "local"), "local")
        self.assertEqual(called["n"], 0)

    def test_open_without_fallback_raises(self):
        for _ in range(3):
            self.cb.call(self._boom, fallback=lambda: "local")
        with self.assertRaises(CircuitBreakerOpen):
            self.cb.call(lambda: "cloud")

    def test_recovers_after_reset_timeout(self):
        for _ in range(3):
            self.cb.call(self._boom, fallback=lambda: "local")
        self.assertEqual(self.cb.state, OPEN)
        self.clock[0] = 601  # advance past the 10-minute window
        self.assertEqual(self.cb.state, HALF_OPEN)
        self.assertEqual(self.cb.call(lambda: "cloud", fallback=lambda: "local"), "cloud")
        self.assertEqual(self.cb.state, CLOSED)

    def test_half_open_failure_reopens(self):
        for _ in range(3):
            self.cb.call(self._boom, fallback=lambda: "local")
        self.clock[0] = 601
        self.assertEqual(self.cb.state, HALF_OPEN)
        self.cb.call(self._boom, fallback=lambda: "local")
        self.assertEqual(self.cb.state, OPEN)

    def test_success_resets_failure_count(self):
        self.cb.call(self._boom, fallback=lambda: "local")
        self.cb.call(self._boom, fallback=lambda: "local")
        self.cb.call(lambda: "ok")  # success clears the streak
        self.assertEqual(self.cb.snapshot()["consecutive_failures"], 0)
        # two more failures must not trip (streak was reset)
        self.cb.call(self._boom, fallback=lambda: "local")
        self.cb.call(self._boom, fallback=lambda: "local")
        self.assertEqual(self.cb.state, CLOSED)


class TestOutboundGuard(unittest.TestCase):
    def test_strips_phi_keeps_safe_fields(self):
        meta = {
            "PatientName": "John Doe",
            "PatientID": "12345",
            "PatientBirthDate": "19900101",
            "InstitutionName": "City Hospital",
            "Modality": "US",
            "Rows": 512,
            "Columns": 512,
        }
        cleaned = scrub_metadata(meta)
        self.assertEqual(set(cleaned), {"Modality", "Rows", "Columns"})

    def test_payload_has_surrogate_and_no_phi(self):
        meta = {"PatientName": "Jane", "PatientID": "99", "Modality": "MG"}
        payload = build_outbound_payload([[0, 1]], metadata=meta)
        self.assertIn("surrogate_id", payload)
        self.assertTrue(payload["surrogate_id"])
        self.assertNotIn("PatientName", payload["metadata"])
        # pixels pass through untouched
        self.assertEqual(payload["pixels"], [[0, 1]])
        assert_no_phi(payload)  # must not raise

    def test_surrogate_is_deterministic(self):
        ident = {"PatientID": "42"}
        a = build_outbound_payload([], identity=ident)["surrogate_id"]
        b = build_outbound_payload([], identity=ident)["surrogate_id"]
        self.assertEqual(a, b)

    def test_assert_no_phi_catches_leak(self):
        with self.assertRaises(PHILeakError):
            assert_no_phi({"metadata": {"PatientName": "leak"}})


class TestTaskQueue(unittest.TestCase):
    def tearDown(self):
        tq.shutdown()

    def test_submit_returns_202_immediately(self):
        def slow(x):
            time.sleep(0.1)
            return x + 1

        ticket = tq.submit(slow, 10)
        self.assertEqual(ticket.http_status, 202)
        self.assertEqual(ticket.backend, "thread")
        # result resolves later
        self.assertEqual(tq.get_result(ticket.id, timeout=5), 11)

    def test_status_transitions(self):
        done = []

        def job():
            time.sleep(0.2)
            done.append(True)
            return "ok"

        ticket = tq.submit(job)
        # Immediately after submit it is pending/started, not success.
        self.assertIn(tq.get_status(ticket.id)["status"], {"PENDING", "STARTED"})
        tq.get_result(ticket.id, timeout=5)
        self.assertEqual(tq.get_status(ticket.id)["status"], "SUCCESS")

    def test_failure_is_captured(self):
        def boom():
            raise ValueError("bad")

        ticket = tq.submit(boom)
        with self.assertRaises(ValueError):
            tq.get_result(ticket.id, timeout=5)
        self.assertEqual(tq.get_status(ticket.id)["status"], "FAILURE")


class TestSecureStore(unittest.TestCase):
    def setUp(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.store = SecureStore(db_path=self.db)

    def tearDown(self):
        self.store.close()
        for p in (self.db, self.db + "-wal", self.db + "-shm"):
            if os.path.exists(p):
                os.remove(p)

    def test_wal_mode_enabled(self):
        mode = self.store._conn.execute("PRAGMA journal_mode;").fetchone()[0]
        self.assertEqual(mode.lower(), "wal")

    def test_save_and_get(self):
        self.store.save_record("r1", "SUR1", {"verdict": "TB"}, modality="CR")
        rec = self.store.get_record("r1")
        self.assertEqual(rec["payload"]["verdict"], "TB")
        self.assertFalse(rec["frozen"])

    def test_freeze_blocks_mutation(self):
        self.store.save_record("r1", "SUR1", {"v": 1})
        self.store.update_status("r1", "accepted", freeze=True)
        with self.assertRaises(ImmutableRecordError):
            self.store.update_status("r1", "modified")

    def test_history_by_surrogate(self):
        self.store.save_record("r1", "SUR1", {"v": 1})
        self.store.save_record("r2", "SUR1", {"v": 2}, supersedes="r1")
        self.assertEqual(len(self.store.list_by_surrogate("SUR1")), 2)


class TestProvenance(unittest.TestCase):
    def test_stamp_appends_immutable_block(self):
        r = {"verdict": "TB", "source": "Roboflow tb/1"}
        s = stamp_provenance(r, model_version="2.0.0")
        self.assertIn(PROVENANCE_KEY, s)
        self.assertEqual(s[PROVENANCE_KEY]["model_version"], "2.0.0")
        self.assertEqual(s[PROVENANCE_KEY]["model_name"], "Roboflow tb/1")
        # original untouched
        self.assertNotIn(PROVENANCE_KEY, r)

    def test_verify_detects_tampering(self):
        s = stamp_provenance({"confidence": 90})
        self.assertTrue(verify_provenance(s))
        s["confidence"] = 10
        self.assertFalse(verify_provenance(s))

    def test_verify_false_without_stamp(self):
        self.assertFalse(verify_provenance({"confidence": 1}))


class TestVerificationWorkflow(unittest.TestCase):
    def setUp(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.store = SecureStore(db_path=self.db)
        self.vs = VerificationService(self.store)

    def tearDown(self):
        self.store.close()
        for p in (self.db, self.db + "-wal", self.db + "-shm"):
            if os.path.exists(p):
                os.remove(p)

    def test_accept_freezes_and_finalizes(self):
        rid = self.vs.submit_for_review("SUR", {"verdict": "TB"}, modality="CR")
        self.assertFalse(self.vs.is_final(rid))
        self.vs.accept(rid, "Dr.A")
        self.assertTrue(self.vs.is_final(rid))
        # cannot re-decide a finalized record
        with self.assertRaises(VerificationError):
            self.vs.accept(rid, "Dr.A")

    def test_reject_freezes_not_final(self):
        rid = self.vs.submit_for_review("SUR", {"verdict": "TB"})
        self.vs.reject(rid, "Dr.A", reason="artifact")
        self.assertFalse(self.vs.is_final(rid))
        self.assertEqual(self.store.get_record(rid)["status"], "rejected")
        self.assertTrue(self.store.get_record(rid)["frozen"])

    def test_modify_creates_superseding_record(self):
        rid = self.vs.submit_for_review("SUR", {"verdict": "BIRADS-2"})
        new_id = self.vs.modify(rid, "Dr.B", {"verdict": "BIRADS-4"})
        self.assertNotEqual(rid, new_id)
        # original preserved + frozen; new one is the final accepted record
        self.assertEqual(self.store.get_record(rid)["status"], "modified")
        self.assertTrue(self.store.get_record(rid)["frozen"])
        self.assertTrue(self.vs.is_final(new_id))
        self.assertEqual(self.store.get_record(new_id)["supersedes"], rid)


class TestFHIRMapper(unittest.TestCase):
    def test_maps_to_bundle_with_report_and_observations(self):
        record = {
            "record_id": "rec1",
            "surrogate_id": "SURZ",
            "status": "accepted",
            "modality": "CR",
            "created_at": "2026-01-01T00:00:00Z",
            "payload": {
                "verdict": "TB-Suggestive",
                "confidence": 88,
                "recommendation": "Confirm with GeneXpert",
                "modality": "CR",
            },
        }
        bundle = record_to_fhir(record)
        self.assertEqual(bundle["resourceType"], "Bundle")
        self.assertEqual(bundle["type"], "collection")
        report = bundle["entry"][0]["resource"]
        self.assertEqual(report["resourceType"], "DiagnosticReport")
        self.assertEqual(report["status"], "final")  # accepted -> final
        self.assertEqual(report["subject"]["reference"], "Patient/SURZ")
        # observations referenced match the number of resources after the report
        obs_entries = [e for e in bundle["entry"] if e["resource"]["resourceType"] == "Observation"]
        self.assertEqual(len(report["result"]), len(obs_entries))
        self.assertGreater(len(obs_entries), 0)

    def test_no_phi_in_bundle(self):
        record = {
            "record_id": "rec2",
            "surrogate_id": "SURH",
            "status": "accepted",
            "payload": {"verdict": "Normal", "confidence": 12},
        }
        bundle = record_to_fhir(record)
        text = str(bundle).lower()
        for phi in ("patientname", "john", "birthdate"):
            self.assertNotIn(phi, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
