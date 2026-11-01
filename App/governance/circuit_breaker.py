"""
circuit_breaker.py
------------------
A minimal, dependency-free circuit breaker (Chunk 2 fail-safe).

The task asks for the Circuit Breaker Pattern around external requests (e.g.
Roboflow): if an external endpoint throws 3 consecutive errors (or times out),
trip the breaker OPEN and, for the next 10 minutes, skip the cloud route and
fall back to the local model without letting the application thread hang.

We implement it in pure stdlib rather than pulling in ``pybreaker`` so the edge
build stays lean (the hardware rule forbids heavy libraries). The behaviour is
the standard three-state breaker:

    CLOSED   -> calls pass through. Consecutive failures are counted.
    OPEN     -> calls are short-circuited (fallback used) until the reset
                timeout elapses.
    HALF_OPEN-> one trial call is allowed; success closes the breaker, failure
                re-opens it for another timeout window.

Thread-safe via a single lock so it can be shared across worker threads.
"""

import threading
import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised by ``call()`` when the breaker is open and no fallback is given."""


class CircuitBreaker:
    """Trip external calls open after repeated failures; auto-recover later.

    Parameters
    ----------
    failure_threshold:
        Consecutive failures that trip the breaker open. Default 3.
    reset_timeout:
        Seconds the breaker stays open before allowing a trial call.
        Default 600 (10 minutes), matching the task's fallback window.
    name:
        Human-readable label used in state-change log messages.
    time_fn:
        Injectable clock (defaults to ``time.monotonic``) for testability.
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        reset_timeout: float = 600.0,
        name: str = "external",
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be >= 1")
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.name = name
        self._time = time_fn

        self._lock = threading.RLock()
        self._state = CLOSED
        self._consecutive_failures = 0
        self._opened_at: Optional[float] = None

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def state(self) -> str:
        """Current state, lazily transitioning OPEN -> HALF_OPEN when due."""
        with self._lock:
            self._maybe_half_open()
            return self._state

    def is_closed(self) -> bool:
        return self.state != OPEN

    def snapshot(self) -> dict:
        """Return a JSON-serialisable view of the breaker's current state."""
        with self._lock:
            self._maybe_half_open()
            remaining = 0.0
            if self._state == OPEN and self._opened_at is not None:
                remaining = max(
                    0.0,
                    self.reset_timeout - (self._time() - self._opened_at),
                )
            return {
                "name": self.name,
                "state": self._state,
                "consecutive_failures": self._consecutive_failures,
                "failure_threshold": self.failure_threshold,
                "reset_timeout": self.reset_timeout,
                "seconds_until_retry": round(remaining, 3),
            }

    # ------------------------------------------------------------------
    # Core call path
    # ------------------------------------------------------------------
    def call(
        self,
        fn: Callable[..., T],
        *args,
        fallback: Optional[Callable[..., T]] = None,
        **kwargs,
    ) -> T:
        """Run ``fn`` through the breaker.

        If the breaker is OPEN (and the reset window has not elapsed), ``fn`` is
        skipped entirely and ``fallback`` is used — the calling thread never
        waits on the dead endpoint. On HALF_OPEN a single trial call is allowed.

        Raises ``CircuitBreakerOpen`` only when the breaker is open and no
        ``fallback`` was supplied.
        """
        if not self._allow_request():
            if fallback is not None:
                return fallback(*args, **kwargs)
            raise CircuitBreakerOpen(
                f"Circuit '{self.name}' is open; external call skipped."
            )

        try:
            result = fn(*args, **kwargs)
        except Exception:
            self._on_failure()
            if fallback is not None:
                return fallback(*args, **kwargs)
            raise
        else:
            self._on_success()
            return result

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    def _allow_request(self) -> bool:
        with self._lock:
            self._maybe_half_open()
            # In CLOSED or HALF_OPEN we permit the call (HALF_OPEN = one trial).
            return self._state != OPEN

    def _maybe_half_open(self) -> None:
        """Move OPEN -> HALF_OPEN once the reset timeout has elapsed."""
        if (
            self._state == OPEN
            and self._opened_at is not None
            and (self._time() - self._opened_at) >= self.reset_timeout
        ):
            self._state = HALF_OPEN

    def _on_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._opened_at = None
            self._state = CLOSED

    def _on_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            # A failed trial call in HALF_OPEN immediately re-opens the breaker.
            if self._state == HALF_OPEN:
                self._trip()
            elif self._consecutive_failures >= self.failure_threshold:
                self._trip()

    def _trip(self) -> None:
        self._state = OPEN
        self._opened_at = self._time()

    def reset(self) -> None:
        """Force the breaker closed (operational override)."""
        with self._lock:
            self._state = CLOSED
            self._consecutive_failures = 0
            self._opened_at = None
