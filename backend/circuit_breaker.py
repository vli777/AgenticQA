"""Reusable circuit breaker for guarding against cascading LLM failures."""

from logger import logger
from exceptions import CircuitBreakerOpenError


class CircuitBreaker:
    """
    Simple circuit breaker that opens after consecutive failures.

    Usage:
        cb = CircuitBreaker("summary", threshold=3, error_message="Summary generation failed")
        cb.check()           # raises CircuitBreakerOpenError if open
        cb.record_success()  # resets failure counter
        cb.record_failure()  # increments counter, opens if threshold reached
    """

    def __init__(self, name: str, threshold: int = 3, error_message: str | None = None):
        self.name = name
        self.threshold = threshold
        self.error_message = error_message or f"Circuit breaker '{name}' is open"
        self._failure_count = 0
        self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def check(self) -> None:
        """Raise CircuitBreakerOpenError if the breaker is open."""
        if self._is_open:
            logger.error(
                f"{self.name} circuit breaker open ({self._failure_count} consecutive failures), "
                f"aborting processing"
            )
            raise CircuitBreakerOpenError(self.error_message)

    def record_success(self) -> None:
        """Reset failure counter on success."""
        self._failure_count = 0
        self._is_open = False

    def record_failure(self) -> None:
        """Increment failure counter; open breaker if threshold reached."""
        self._failure_count += 1
        if self._failure_count >= self.threshold:
            self._is_open = True
            logger.error(
                f"{self.name} circuit breaker opened after {self._failure_count} consecutive failures"
            )
