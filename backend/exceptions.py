# backend/exceptions.py

"""Shared exceptions for the application."""


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open due to repeated LLM failures."""
    pass
