"""Resilience patterns for external service calls.

Provides circuit breaker and retry logic for graceful degradation
when external services (like Replicate API) are unavailable.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    """State of a circuit breaker."""

    CLOSED = "closed"  # Normal operation, requests flow through
    OPEN = "open"  # Failures exceeded threshold, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior.

    Attributes:
        failure_threshold: Number of failures before opening circuit
        success_threshold: Number of successes in half-open to close circuit
        timeout_seconds: Time before transitioning from open to half-open
        excluded_exceptions: Exception types that don't count as failures
    """

    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 60.0
    excluded_exceptions: tuple = ()


class CircuitBreakerError(Exception):
    """Raised when circuit breaker is open and blocking requests."""

    def __init__(self, name: str, state: CircuitState, retry_after: float):
        self.name = name
        self.state = state
        self.retry_after = retry_after
        super().__init__(f"Circuit breaker '{name}' is {state.value}. " f"Retry after {retry_after:.1f} seconds.")


class CircuitBreaker:
    """Circuit breaker pattern implementation.

    Prevents cascading failures by stopping requests to failing services.
    After a timeout, allows test requests to check if service recovered.

    Usage:
        breaker = CircuitBreaker("replicate-api")

        async with breaker:
            response = await client.get("/api/endpoint")

        # Or as decorator
        @breaker
        async def call_api():
            ...
    """

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            return self._get_state_unlocked()

    def _get_state_unlocked(self) -> CircuitState:
        """Get state without acquiring lock (caller must hold lock)."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time is not None:
                elapsed = time.time() - self._last_failure_time
                if elapsed >= self.config.timeout_seconds:
                    logger.info("Circuit breaker '%s' transitioning from OPEN to HALF_OPEN", self.name)
                    self._state = CircuitState.HALF_OPEN
                    self._success_count = 0
        return self._state

    def _can_execute(self) -> tuple[bool, float]:
        """Check if execution is allowed.

        Returns:
            (allowed, retry_after_seconds)
        """
        with self._lock:
            state = self._get_state_unlocked()

            if state == CircuitState.CLOSED:
                return True, 0

            if state == CircuitState.HALF_OPEN:
                return True, 0

            # State is OPEN
            if self._last_failure_time is None:
                return True, 0

            elapsed = time.time() - self._last_failure_time
            retry_after = max(0, self.config.timeout_seconds - elapsed)
            return False, retry_after

    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    logger.info("Circuit breaker '%s' closing after %d successful calls", self.name, self._success_count)
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    self._success_count = 0

            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                if self._failure_count > 0:
                    self._failure_count = max(0, self._failure_count - 1)

    def _record_failure(self, exc: Exception) -> None:
        """Record a failed call."""
        # Check if exception should be excluded
        if isinstance(exc, self.config.excluded_exceptions):
            return

        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                logger.warning("Circuit breaker '%s' opening after failure in HALF_OPEN: %s", self.name, exc)
                self._state = CircuitState.OPEN
                self._success_count = 0

            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.config.failure_threshold:
                    logger.warning("Circuit breaker '%s' opening after %d failures: %s", self.name, self._failure_count, exc)
                    self._state = CircuitState.OPEN

    async def __aenter__(self) -> "CircuitBreaker":
        """Async context manager entry."""
        allowed, retry_after = self._can_execute()
        if not allowed:
            raise CircuitBreakerError(self.name, self._state, retry_after)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Async context manager exit."""
        if exc_val is not None:
            self._record_failure(exc_val)
        else:
            self._record_success()
        return False

    def __enter__(self) -> "CircuitBreaker":
        """Sync context manager entry."""
        allowed, retry_after = self._can_execute()
        if not allowed:
            raise CircuitBreakerError(self.name, self._state, retry_after)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Sync context manager exit."""
        if exc_val is not None:
            self._record_failure(exc_val)
        else:
            self._record_success()
        return False

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Use as decorator."""
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                async with self:
                    return await func(*args, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            return sync_wrapper

    def get_health(self) -> Dict[str, Any]:
        """Get circuit breaker health info."""
        with self._lock:
            state = self._get_state_unlocked()
            return {
                "name": self.name,
                "state": state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure": self._last_failure_time,
            }

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            logger.info("Circuit breaker '%s' manually reset", self.name)


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Attributes:
        max_attempts: Maximum number of attempts (including initial)
        base_delay: Base delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff (2.0 = doubling)
        jitter: Whether to add random jitter to delays
        retryable_exceptions: Exception types that trigger retry
    """

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    exponential_base: float = 2.0
    jitter: bool = True
    retryable_exceptions: tuple = (Exception,)
    retryable_status_codes: tuple = (429, 500, 502, 503, 504)


def calculate_delay(
    attempt: int,
    config: RetryConfig,
) -> float:
    """Calculate delay for a retry attempt.

    Uses exponential backoff with optional jitter.
    """
    delay = config.base_delay * (config.exponential_base**attempt)
    delay = min(delay, config.max_delay)

    if config.jitter:
        # Add random jitter between 0-25% of delay
        jitter_amount = delay * random.uniform(0, 0.25)
        delay += jitter_amount

    return delay


async def retry_async(
    func: Callable[..., T],
    *args,
    config: Optional[RetryConfig] = None,
    **kwargs,
) -> T:
    """Execute an async function with retry logic.

    Args:
        func: Async function to execute
        *args: Positional arguments for func
        config: Retry configuration
        **kwargs: Keyword arguments for func

    Returns:
        Result of successful function call

    Raises:
        Last exception if all retries fail
    """
    config = config or RetryConfig()
    last_exception: Optional[Exception] = None

    for attempt in range(config.max_attempts):
        try:
            return await func(*args, **kwargs)
        except config.retryable_exceptions as exc:
            last_exception = exc

            # Check for retryable HTTP status codes
            status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
            if status_code and status_code not in config.retryable_status_codes:
                raise

            if attempt < config.max_attempts - 1:
                delay = calculate_delay(attempt, config)
                logger.warning("Attempt %d/%d failed, retrying in %.2fs: %s", attempt + 1, config.max_attempts, delay, exc)
                await asyncio.sleep(delay)
            else:
                logger.error("All %d attempts failed: %s", config.max_attempts, exc)

    if last_exception:
        raise last_exception
    raise RuntimeError("Retry exhausted with no exception")


def retry(config: Optional[RetryConfig] = None) -> Callable:
    """Decorator for adding retry logic to functions.

    Usage:
        @retry(config=RetryConfig(max_attempts=5))
        async def call_api():
            ...
    """
    config = config or RetryConfig()

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await retry_async(func, *args, config=config, **kwargs)

            return async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                last_exception = None
                for attempt in range(config.max_attempts):
                    try:
                        return func(*args, **kwargs)
                    except config.retryable_exceptions as exc:
                        last_exception = exc
                        if attempt < config.max_attempts - 1:
                            delay = calculate_delay(attempt, config)
                            logger.warning(
                                "Attempt %d/%d failed, retrying in %.2fs: %s", attempt + 1, config.max_attempts, delay, exc
                            )
                            time.sleep(delay)
                if last_exception:
                    raise last_exception

            return sync_wrapper

    return decorator


# Global circuit breakers for external services
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_breakers_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> CircuitBreaker:
    """Get or create a named circuit breaker.

    Circuit breakers are cached by name, so multiple calls with the
    same name return the same instance.

    Args:
        name: Unique name for this circuit breaker
        config: Configuration (only used on first creation)

    Returns:
        CircuitBreaker instance
    """
    with _breakers_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name, config)
        return _circuit_breakers[name]


def get_replicate_circuit_breaker() -> CircuitBreaker:
    """Get the circuit breaker for Replicate API calls."""
    return get_circuit_breaker(
        "replicate-api",
        config=CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=30.0,
        ),
    )


def get_all_circuit_breaker_health() -> Dict[str, Dict[str, Any]]:
    """Get health info for all circuit breakers."""
    with _breakers_lock:
        return {name: breaker.get_health() for name, breaker in _circuit_breakers.items()}


def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers (for testing or recovery)."""
    with _breakers_lock:
        for breaker in _circuit_breakers.values():
            breaker.reset()
