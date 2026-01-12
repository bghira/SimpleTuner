"""Circuit breaker pattern for cloud provider resilience.

Prevents cascading failures when a provider is experiencing issues.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""

    failure_threshold: int = 5  # Failures before opening
    success_threshold: int = 2  # Successes to close from half-open
    timeout_seconds: float = 60.0  # Time in open state before half-open
    half_open_max_calls: int = 3  # Max concurrent calls in half-open


@dataclass
class CircuitStats:
    """Statistics for a circuit breaker."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    opened_at: Optional[float] = None
    half_open_calls: int = 0


class CircuitBreaker:
    """Circuit breaker for a single provider."""

    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._stats = CircuitStats()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._stats.state

    @property
    def is_available(self) -> bool:
        """Check if the circuit allows requests."""
        return self._stats.state != CircuitState.OPEN

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status for monitoring."""
        return {
            "name": self.name,
            "state": self._stats.state.value,
            "failure_count": self._stats.failure_count,
            "success_count": self._stats.success_count,
            "last_failure": self._stats.last_failure_time,
            "last_success": self._stats.last_success_time,
            "opened_at": self._stats.opened_at,
        }

    async def _check_state_transition(self) -> None:
        """Check if state should transition (e.g., open -> half-open)."""
        if self._stats.state == CircuitState.OPEN:
            if self._stats.opened_at:
                elapsed = time.time() - self._stats.opened_at
                if elapsed >= self.config.timeout_seconds:
                    logger.info("Circuit %s transitioning from OPEN to HALF_OPEN after %.1fs", self.name, elapsed)
                    self._stats.state = CircuitState.HALF_OPEN
                    self._stats.half_open_calls = 0
                    self._stats.success_count = 0

    async def record_success(self) -> None:
        """Record a successful call."""
        async with self._lock:
            self._stats.last_success_time = time.time()
            self._stats.success_count += 1

            if self._stats.state == CircuitState.HALF_OPEN:
                self._stats.half_open_calls = max(0, self._stats.half_open_calls - 1)
                if self._stats.success_count >= self.config.success_threshold:
                    logger.info(
                        "Circuit %s transitioning from HALF_OPEN to CLOSED after %d successes",
                        self.name,
                        self._stats.success_count,
                    )
                    self._stats.state = CircuitState.CLOSED
                    self._stats.failure_count = 0
                    self._stats.opened_at = None
            elif self._stats.state == CircuitState.CLOSED:
                # Reset failure count on success
                self._stats.failure_count = 0

    async def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record a failed call."""
        async with self._lock:
            self._stats.last_failure_time = time.time()
            self._stats.failure_count += 1

            if self._stats.state == CircuitState.HALF_OPEN:
                self._stats.half_open_calls = max(0, self._stats.half_open_calls - 1)
                # Any failure in half-open returns to open
                logger.warning("Circuit %s transitioning from HALF_OPEN to OPEN after failure: %s", self.name, error)
                self._stats.state = CircuitState.OPEN
                self._stats.opened_at = time.time()
                self._stats.success_count = 0
            elif self._stats.state == CircuitState.CLOSED:
                if self._stats.failure_count >= self.config.failure_threshold:
                    logger.warning(
                        "Circuit %s transitioning from CLOSED to OPEN after %d failures",
                        self.name,
                        self._stats.failure_count,
                    )
                    self._stats.state = CircuitState.OPEN
                    self._stats.opened_at = time.time()

    async def can_execute(self) -> bool:
        """Check if a call can be executed.

        Returns:
            True if call is allowed, False if circuit is open
        """
        async with self._lock:
            await self._check_state_transition()

            if self._stats.state == CircuitState.CLOSED:
                return True
            elif self._stats.state == CircuitState.OPEN:
                return False
            elif self._stats.state == CircuitState.HALF_OPEN:
                if self._stats.half_open_calls < self.config.half_open_max_calls:
                    self._stats.half_open_calls += 1
                    return True
                return False
        return False

    async def execute(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Execute a function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result of the function

        Raises:
            CircuitOpenError: If circuit is open
            Original exception: If function fails
        """
        if not await self.can_execute():
            raise CircuitOpenError(f"Circuit breaker '{self.name}' is open. " f"Provider may be experiencing issues.")

        try:
            result = await func(*args, **kwargs)
            await self.record_success()
            return result
        except Exception as e:
            await self.record_failure(e)
            raise


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    pass


# Global registry of circuit breakers
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_registry_lock = asyncio.Lock()


async def get_circuit_breaker(name: str) -> CircuitBreaker:
    """Get or create a circuit breaker for a provider."""
    async with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(name)
        return _circuit_breakers[name]


async def get_all_circuit_status() -> Dict[str, Dict[str, Any]]:
    """Get status of all circuit breakers for monitoring."""
    async with _registry_lock:
        return {name: cb.get_status() for name, cb in _circuit_breakers.items()}


def reset_circuit_breakers() -> None:
    """Reset all circuit breakers (for testing)."""
    global _circuit_breakers
    _circuit_breakers = {}
