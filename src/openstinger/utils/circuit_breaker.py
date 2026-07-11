"""
Circuit breaker — v0.9 reliability feature.

Wraps any async callable with a CLOSED → OPEN → HALF_OPEN state machine.
Prevents cascade failures when FalkorDB or the embedding provider goes down.

States:
  CLOSED    — normal operation
  OPEN      — failing; fast-fail all calls for recovery_timeout seconds
  HALF_OPEN — probing recovery; closes after success_threshold successes
"""

import time
from enum import Enum
from typing import Any, Callable


class CircuitState(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when a call is attempted while the circuit is OPEN."""
    pass


class CircuitBreaker:
    """
    Async circuit breaker. Wraps any async callable.

    Config (from resilience block in config.yaml):
      failure_threshold: int  — consecutive failures before OPEN (default: 5)
      recovery_timeout:  int  — seconds in OPEN before HALF-OPEN probe (default: 30)
      success_threshold: int  — successes in HALF-OPEN before CLOSED (default: 2)
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout:  int = 30,
        success_threshold: int = 2,
    ) -> None:
        self.name              = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout  = recovery_timeout
        self.success_threshold = success_threshold

        self._state:         CircuitState   = CircuitState.CLOSED
        self._failure_count: int            = 0
        self._success_count: int            = 0
        self._opened_at:     float | None   = None

    @property
    def state(self) -> CircuitState:
        if (
            self._state == CircuitState.OPEN
            and self._opened_at is not None
            and (time.monotonic() - self._opened_at) >= self.recovery_timeout
        ):
            self._state         = CircuitState.HALF_OPEN
            self._success_count = 0
        return self._state

    async def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        if self.state == CircuitState.OPEN:
            raise CircuitBreakerOpen(
                f"Circuit '{self.name}' is OPEN (opened at {self._opened_at:.1f}). "
                f"Retrying in {self.recovery_timeout - (time.monotonic() - (self._opened_at or 0)):.0f}s."
            )
        try:
            result = await fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception:
            self._on_failure()
            raise

    def _on_success(self) -> None:
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                self._state         = CircuitState.CLOSED
                self._failure_count = 0
                self._opened_at     = None
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def _on_failure(self) -> None:
        self._failure_count += 1
        if self._state == CircuitState.HALF_OPEN:
            self._state     = CircuitState.OPEN
            self._opened_at = time.monotonic()
        elif (
            self._state == CircuitState.CLOSED
            and self._failure_count >= self.failure_threshold
        ):
            self._state     = CircuitState.OPEN
            self._opened_at = time.monotonic()

    def status(self) -> dict:
        return {
            "name":          self.name,
            "state":         self.state.value,
            "failure_count": self._failure_count,
            "opened_at":     self._opened_at,
        }
