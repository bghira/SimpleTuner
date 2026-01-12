"""Async Circuit Breaker Adapter.

Implements the circuit breaker pattern using the state backend.
Persists state across restarts and works in multi-worker deployments.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

from ..protocols import StateBackendProtocol

T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, requests are rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


class AsyncCircuitBreaker:
    """Circuit breaker using pluggable state backend.

    Implements the circuit breaker pattern:
        - CLOSED: Normal operation, failures are counted
        - OPEN: Service is failing, requests are rejected immediately
        - HALF_OPEN: Allow some requests to test if service recovered

    State is stored in the backend as a hash:
        {
            "state": "closed|open|half_open",
            "failure_count": 5,
            "success_count": 0,
            "last_failure_time": 1234567890.123,
            "last_state_change": 1234567890.123,
        }

    Example:
        backend = await get_state_backend()
        breaker = AsyncCircuitBreaker(backend, "external-api")

        try:
            async with breaker:
                result = await call_external_api()
        except CircuitOpenError:
            # Circuit is open, use fallback
            result = cached_result
    """

    def __init__(
        self,
        backend: StateBackendProtocol,
        name: str,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout_seconds: float = 30.0,
        half_open_max_calls: int = 3,
        key_prefix: str = "circuit:",
    ):
        """Initialize circuit breaker.

        Args:
            backend: State backend instance.
            name: Unique name for this circuit.
            failure_threshold: Failures before opening circuit.
            success_threshold: Successes in half-open before closing.
            timeout_seconds: Time before attempting half-open.
            half_open_max_calls: Max concurrent calls in half-open state.
            key_prefix: Prefix for circuit breaker keys.
        """
        self._backend = backend
        self._name = name
        self._failure_threshold = failure_threshold
        self._success_threshold = success_threshold
        self._timeout_seconds = timeout_seconds
        self._half_open_max_calls = half_open_max_calls
        self._key = f"{key_prefix}{name}"

    async def _get_state_data(self) -> dict[str, Any]:
        """Get current state data from backend."""
        data = await self._backend.hgetall(self._key)
        if not data:
            return {
                "state": CircuitState.CLOSED.value,
                "failure_count": 0,
                "success_count": 0,
                "last_failure_time": 0.0,
                "last_state_change": time.time(),
                "half_open_calls": 0,
            }

        return {
            "state": data.get(b"state", b"closed").decode(),
            "failure_count": int(data.get(b"failure_count", b"0").decode()),
            "success_count": int(data.get(b"success_count", b"0").decode()),
            "last_failure_time": float(data.get(b"last_failure_time", b"0").decode()),
            "last_state_change": float(data.get(b"last_state_change", str(time.time()).encode()).decode()),
            "half_open_calls": int(data.get(b"half_open_calls", b"0").decode()),
        }

    async def _set_state_data(self, data: dict[str, Any]) -> None:
        """Save state data to backend."""
        for key, value in data.items():
            await self._backend.hset(self._key, key, str(value).encode())

    async def get_state(self) -> CircuitState:
        """Get current circuit state.

        Automatically transitions from OPEN to HALF_OPEN after timeout.

        Returns:
            Current circuit state.
        """
        data = await self._get_state_data()
        state = CircuitState(data["state"])

        # Check if timeout expired for open circuit
        if state == CircuitState.OPEN:
            time_in_open = time.time() - data["last_state_change"]
            if time_in_open >= self._timeout_seconds:
                # Transition to half-open
                await self._transition_to(CircuitState.HALF_OPEN)
                return CircuitState.HALF_OPEN

        return state

    async def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        now = time.time()
        updates = {
            "state": new_state.value,
            "last_state_change": now,
        }

        if new_state == CircuitState.CLOSED:
            updates["failure_count"] = 0
            updates["success_count"] = 0
            updates["half_open_calls"] = 0
        elif new_state == CircuitState.HALF_OPEN:
            updates["success_count"] = 0
            updates["half_open_calls"] = 0

        await self._set_state_data(updates)

    async def is_available(self) -> bool:
        """Check if circuit allows requests.

        Returns:
            True if requests should be attempted.
        """
        state = await self.get_state()

        if state == CircuitState.CLOSED:
            return True

        if state == CircuitState.OPEN:
            return False

        # HALF_OPEN: allow limited calls
        data = await self._get_state_data()
        return data["half_open_calls"] < self._half_open_max_calls

    async def record_success(self) -> None:
        """Record a successful call.

        In HALF_OPEN state, may close the circuit.
        """
        state = await self.get_state()

        if state == CircuitState.HALF_OPEN:
            data = await self._get_state_data()
            success_count = data["success_count"] + 1

            if success_count >= self._success_threshold:
                # Circuit recovered
                await self._transition_to(CircuitState.CLOSED)
            else:
                await self._backend.hset(self._key, "success_count", str(success_count).encode())

    async def record_failure(self) -> None:
        """Record a failed call.

        May open the circuit if threshold reached.
        """
        state = await self.get_state()
        now = time.time()

        if state == CircuitState.HALF_OPEN:
            # Single failure in half-open reopens circuit
            await self._transition_to(CircuitState.OPEN)
            await self._backend.hset(self._key, "last_failure_time", str(now).encode())
            return

        if state == CircuitState.CLOSED:
            data = await self._get_state_data()
            failure_count = data["failure_count"] + 1

            await self._backend.hset(self._key, "failure_count", str(failure_count).encode())
            await self._backend.hset(self._key, "last_failure_time", str(now).encode())

            if failure_count >= self._failure_threshold:
                await self._transition_to(CircuitState.OPEN)

    async def reset(self) -> None:
        """Reset circuit to closed state."""
        await self._transition_to(CircuitState.CLOSED)

    async def force_open(self) -> None:
        """Force circuit to open state."""
        await self._transition_to(CircuitState.OPEN)

    async def __aenter__(self) -> "AsyncCircuitBreaker":
        """Context manager entry.

        Raises:
            CircuitOpenError: If circuit is open.
        """
        if not await self.is_available():
            raise CircuitOpenError(f"Circuit '{self._name}' is open")

        # Track half-open calls
        state = await self.get_state()
        if state == CircuitState.HALF_OPEN:
            data = await self._get_state_data()
            await self._backend.hset(self._key, "half_open_calls", str(data["half_open_calls"] + 1).encode())

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit.

        Records success or failure based on exception.
        """
        if exc_type is None:
            await self.record_success()
        else:
            await self.record_failure()

        return False  # Don't suppress exceptions

    async def call(
        self,
        func: Callable[..., T],
        *args,
        fallback: Optional[Callable[..., T]] = None,
        **kwargs,
    ) -> T:
        """Execute function with circuit breaker protection.

        Args:
            func: Async function to call.
            *args: Arguments for function.
            fallback: Optional fallback function if circuit is open.
            **kwargs: Keyword arguments for function.

        Returns:
            Result from func or fallback.

        Raises:
            CircuitOpenError: If circuit is open and no fallback provided.
        """
        try:
            async with self:
                return await func(*args, **kwargs)
        except CircuitOpenError:
            if fallback is not None:
                return await fallback(*args, **kwargs)
            raise

    @property
    def name(self) -> str:
        """Get circuit name."""
        return self._name


class CircuitOpenError(Exception):
    """Raised when circuit is open and request is rejected."""

    pass


# Registry for managing multiple circuit breakers
_circuit_breakers: dict[str, AsyncCircuitBreaker] = {}


async def get_circuit_breaker(
    backend: StateBackendProtocol,
    name: str,
    **kwargs,
) -> AsyncCircuitBreaker:
    """Get or create a circuit breaker by name.

    Args:
        backend: State backend instance.
        name: Circuit breaker name.
        **kwargs: Additional arguments for AsyncCircuitBreaker.

    Returns:
        AsyncCircuitBreaker instance.
    """
    if name not in _circuit_breakers:
        _circuit_breakers[name] = AsyncCircuitBreaker(backend, name, **kwargs)
    return _circuit_breakers[name]


def reset_circuit_breakers() -> None:
    """Clear the circuit breaker registry (for testing)."""
    _circuit_breakers.clear()
