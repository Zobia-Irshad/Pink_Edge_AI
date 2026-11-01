"""
task_queue.py
-------------
Non-blocking inference job submission (Chunk 3).

Heavy inference must not freeze the clinical UI. The task's design is:

    image arrives -> save to a local cache dir -> hand the path to a worker
    -> immediately return an HTTP 202 Accepted (a ticket) to the frontend.

The reference stack is Celery + a local Redis broker. Celery/Redis are heavy
for a RAM-constrained edge device and may be absent, so this module uses Celery
when it is installed *and* a broker URL is configured, and otherwise falls back
to a lightweight in-process ``ThreadPoolExecutor`` that gives the same
submit-now / poll-later contract. Either way the UI thread returns instantly.

Public API
----------
    submit(fn, *args, **kwargs) -> Ticket      # returns immediately (202-style)
    get_status(ticket_id) -> dict              # PENDING / STARTED / SUCCESS / FAILURE
    get_result(ticket_id) -> result or raises
"""

import os
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

# Job lifecycle states, mirroring Celery's vocabulary.
PENDING = "PENDING"
STARTED = "STARTED"
SUCCESS = "SUCCESS"
FAILURE = "FAILURE"

# HTTP status the frontend should receive on submit.
HTTP_ACCEPTED = 202


@dataclass
class Ticket:
    """Handle returned to the caller the instant a job is accepted."""

    id: str
    http_status: int = HTTP_ACCEPTED
    backend: str = "thread"
    status: str = PENDING
    _future: Optional[Future] = field(default=None, repr=False)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "ticket_id": self.id,
            "http_status": self.http_status,
            "backend": self.backend,
            "status": self.status,
        }


class _ThreadBackend:
    """In-process async backend: a bounded thread pool, no external broker.

    Keeps ``max_workers`` small so concurrent heavy inference cannot exhaust
    edge RAM — jobs queue instead of all running at once.
    """

    name = "thread"

    def __init__(self, max_workers: int = 2) -> None:
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="pinkedge-infer"
        )
        self._lock = threading.Lock()
        self._futures: Dict[str, Future] = {}

    def submit(self, fn: Callable, *args, **kwargs) -> Ticket:
        ticket_id = uuid.uuid4().hex
        future = self._pool.submit(fn, *args, **kwargs)
        with self._lock:
            self._futures[ticket_id] = future
        return Ticket(id=ticket_id, backend=self.name, _future=future)

    def status(self, ticket_id: str) -> str:
        with self._lock:
            future = self._futures.get(ticket_id)
        if future is None:
            return PENDING
        if future.running():
            return STARTED
        if future.done():
            return FAILURE if future.exception() is not None else SUCCESS
        return PENDING

    def result(self, ticket_id: str, timeout: Optional[float] = None) -> Any:
        with self._lock:
            future = self._futures.get(ticket_id)
        if future is None:
            raise KeyError(f"Unknown ticket: {ticket_id}")
        return future.result(timeout=timeout)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class _CeleryBackend:  # pragma: no cover - exercised only when Celery is installed
    """Celery + Redis backend. Used only if Celery imports and a broker is set."""

    name = "celery"

    def __init__(self, app) -> None:
        self._app = app

    def submit(self, fn: Callable, *args, **kwargs) -> Ticket:
        async_result = fn.delay(*args, **kwargs)
        return Ticket(id=async_result.id, backend=self.name)

    def status(self, ticket_id: str) -> str:
        return self._app.AsyncResult(ticket_id).status

    def result(self, ticket_id: str, timeout: Optional[float] = None) -> Any:
        return self._app.AsyncResult(ticket_id).get(timeout=timeout)

    def shutdown(self) -> None:
        pass


def _build_backend():
    """Prefer Celery+Redis if available and configured, else the thread pool."""
    broker = os.environ.get("PINKEDGE_CELERY_BROKER")
    if broker:
        try:
            from celery import Celery  # noqa: F401

            app = Celery("pinkedge", broker=broker, backend=broker)
            return _CeleryBackend(app)
        except Exception:
            # Celery present but misconfigured -> fall back rather than crash.
            pass
    return _ThreadBackend(
        max_workers=int(os.environ.get("PINKEDGE_INFER_WORKERS", "2"))
    )


# Lazily-created process-wide backend.
_backend = None
_backend_lock = threading.Lock()


def _get_backend():
    global _backend
    if _backend is None:
        with _backend_lock:
            if _backend is None:
                _backend = _build_backend()
    return _backend


def submit(fn: Callable, *args, **kwargs) -> Ticket:
    """Accept a job and return immediately with a 202-style Ticket."""
    return _get_backend().submit(fn, *args, **kwargs)


def get_status(ticket_id: str) -> Dict[str, Any]:
    """Poll a job's lifecycle state without blocking."""
    backend = _get_backend()
    return {"ticket_id": ticket_id, "status": backend.status(ticket_id)}


def get_result(ticket_id: str, timeout: Optional[float] = None) -> Any:
    """Fetch a finished job's result (raises if the job raised)."""
    return _get_backend().result(ticket_id, timeout=timeout)


def shutdown() -> None:
    """Tear down the backend (tests / clean exit)."""
    global _backend
    if _backend is not None:
        _backend.shutdown()
        _backend = None
