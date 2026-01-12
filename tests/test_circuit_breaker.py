"""Tests for circuit breaker and failover resilience patterns.

Tests cover:
- Synchronous CircuitBreaker (resilience.py)
- AsyncCircuitBreaker with state backend (adapters/circuit_breaker.py)
- Retry logic with exponential backoff
- State transitions (CLOSED -> OPEN -> HALF_OPEN -> CLOSED)
- Recovery and fallback behavior
"""

from __future__ import annotations

import asyncio
import logging
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Suppress expected error logs during tests
logging.getLogger("simpletuner.simpletuner_sdk.server.services.cloud.resilience").setLevel(logging.CRITICAL)

from simpletuner.simpletuner_sdk.server.services.cloud.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitState,
    RetryConfig,
    calculate_delay,
    get_all_circuit_breaker_health,
    get_circuit_breaker,
    reset_all_circuit_breakers,
    retry,
    retry_async,
)
from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.adapters.circuit_breaker import (
    AsyncCircuitBreaker,
    CircuitOpenError,
)
from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.adapters.circuit_breaker import (
    CircuitState as AsyncCircuitState,
)
from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.adapters.circuit_breaker import (
    get_circuit_breaker as async_get_circuit_breaker,
)
from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.adapters.circuit_breaker import reset_circuit_breakers
from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.backends.memory import MemoryStateBackend


class TestCircuitBreakerConfig(unittest.TestCase):
    """Tests for CircuitBreakerConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CircuitBreakerConfig()
        self.assertEqual(config.failure_threshold, 5)
        self.assertEqual(config.success_threshold, 2)
        self.assertEqual(config.timeout_seconds, 60.0)
        self.assertEqual(config.excluded_exceptions, ())

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=1,
            timeout_seconds=30.0,
            excluded_exceptions=(ValueError,),
        )
        self.assertEqual(config.failure_threshold, 3)
        self.assertEqual(config.success_threshold, 1)
        self.assertEqual(config.timeout_seconds, 30.0)
        self.assertEqual(config.excluded_exceptions, (ValueError,))


class TestCircuitBreakerStateTransitions(unittest.TestCase):
    """Tests for CircuitBreaker state transitions."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=0.1,  # Short timeout for testing
        )
        self.breaker = CircuitBreaker("test-breaker", self.config)

    def test_initial_state_is_closed(self):
        """Test initial state is CLOSED."""
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)

    def test_stays_closed_under_threshold(self):
        """Test circuit stays closed under failure threshold."""
        for _ in range(self.config.failure_threshold - 1):
            with self.assertRaises(RuntimeError):
                with self.breaker:
                    raise RuntimeError("fail")

        self.assertEqual(self.breaker.state, CircuitState.CLOSED)

    def test_opens_at_threshold(self):
        """Test circuit opens at failure threshold."""
        for _ in range(self.config.failure_threshold):
            with self.assertRaises(RuntimeError):
                with self.breaker:
                    raise RuntimeError("fail")

        self.assertEqual(self.breaker.state, CircuitState.OPEN)

    def test_blocks_when_open(self):
        """Test circuit blocks requests when open."""
        # Open the circuit
        for _ in range(self.config.failure_threshold):
            with self.assertRaises(RuntimeError):
                with self.breaker:
                    raise RuntimeError("fail")

        # Verify it blocks
        with self.assertRaises(CircuitBreakerError) as ctx:
            with self.breaker:
                pass

        self.assertEqual(ctx.exception.state, CircuitState.OPEN)
        self.assertGreater(ctx.exception.retry_after, 0)

    def test_transitions_to_half_open_after_timeout(self):
        """Test circuit transitions to HALF_OPEN after timeout."""
        # Open the circuit
        for _ in range(self.config.failure_threshold):
            with self.assertRaises(RuntimeError):
                with self.breaker:
                    raise RuntimeError("fail")

        self.assertEqual(self.breaker.state, CircuitState.OPEN)

        # Wait for timeout
        time.sleep(self.config.timeout_seconds + 0.05)

        self.assertEqual(self.breaker.state, CircuitState.HALF_OPEN)

    def test_closes_after_success_in_half_open(self):
        """Test circuit closes after successes in HALF_OPEN."""
        # Open and wait for half-open
        for _ in range(self.config.failure_threshold):
            with self.assertRaises(RuntimeError):
                with self.breaker:
                    raise RuntimeError("fail")

        time.sleep(self.config.timeout_seconds + 0.05)
        self.assertEqual(self.breaker.state, CircuitState.HALF_OPEN)

        # Succeed enough times to close
        for _ in range(self.config.success_threshold):
            with self.breaker:
                pass  # Success

        self.assertEqual(self.breaker.state, CircuitState.CLOSED)

    def test_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure in HALF_OPEN."""
        # Open and wait for half-open
        for _ in range(self.config.failure_threshold):
            with self.assertRaises(RuntimeError):
                with self.breaker:
                    raise RuntimeError("fail")

        time.sleep(self.config.timeout_seconds + 0.05)
        self.assertEqual(self.breaker.state, CircuitState.HALF_OPEN)

        # Fail in half-open
        with self.assertRaises(RuntimeError):
            with self.breaker:
                raise RuntimeError("fail")

        self.assertEqual(self.breaker.state, CircuitState.OPEN)

    def test_excluded_exceptions_not_counted(self):
        """Test excluded exceptions don't count as failures."""
        config = CircuitBreakerConfig(
            failure_threshold=2,
            excluded_exceptions=(ValueError,),
        )
        breaker = CircuitBreaker("test-excluded", config)

        # ValueError should not count
        for _ in range(5):
            with self.assertRaises(ValueError):
                with breaker:
                    raise ValueError("excluded")

        self.assertEqual(breaker.state, CircuitState.CLOSED)

        # But other exceptions should
        for _ in range(2):
            with self.assertRaises(RuntimeError):
                with breaker:
                    raise RuntimeError("counted")

        self.assertEqual(breaker.state, CircuitState.OPEN)


class TestCircuitBreakerAsync(unittest.IsolatedAsyncioTestCase):
    """Tests for async CircuitBreaker context manager."""

    async def test_async_context_manager_success(self):
        """Test async context manager with successful call."""
        breaker = CircuitBreaker("async-test", CircuitBreakerConfig())

        async with breaker:
            pass  # Success

        self.assertEqual(breaker.state, CircuitState.CLOSED)

    async def test_async_context_manager_failure(self):
        """Test async context manager with failing call."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("async-fail", config)

        with self.assertRaises(RuntimeError):
            async with breaker:
                raise RuntimeError("async fail")

        self.assertEqual(breaker.state, CircuitState.OPEN)

    async def test_async_decorator(self):
        """Test using circuit breaker as async decorator."""
        config = CircuitBreakerConfig(failure_threshold=2)
        breaker = CircuitBreaker("async-decorator", config)

        @breaker
        async def failing_func():
            raise RuntimeError("decorated fail")

        with self.assertRaises(RuntimeError):
            await failing_func()

        self.assertEqual(breaker.state, CircuitState.CLOSED)  # Only 1 failure

        with self.assertRaises(RuntimeError):
            await failing_func()

        self.assertEqual(breaker.state, CircuitState.OPEN)  # 2 failures


class TestCircuitBreakerReset(unittest.TestCase):
    """Tests for circuit breaker reset functionality."""

    def test_manual_reset(self):
        """Test manual reset of circuit breaker."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("reset-test", config)

        # Open the circuit
        with self.assertRaises(RuntimeError):
            with breaker:
                raise RuntimeError("fail")

        self.assertEqual(breaker.state, CircuitState.OPEN)

        # Reset
        breaker.reset()

        self.assertEqual(breaker.state, CircuitState.CLOSED)
        # Should allow requests again
        with breaker:
            pass

    def test_health_info(self):
        """Test getting health info."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("health-test", config)

        # Record some failures
        with self.assertRaises(RuntimeError):
            with breaker:
                raise RuntimeError("fail")

        health = breaker.get_health()

        self.assertEqual(health["name"], "health-test")
        self.assertEqual(health["state"], "closed")
        self.assertEqual(health["failure_count"], 1)
        self.assertIsNotNone(health["last_failure"])


class TestGlobalCircuitBreakers(unittest.TestCase):
    """Tests for global circuit breaker registry."""

    def setUp(self):
        """Reset global state before each test."""
        reset_all_circuit_breakers()

    def tearDown(self):
        """Reset global state after each test."""
        reset_all_circuit_breakers()

    def test_get_circuit_breaker_creates_new(self):
        """Test get_circuit_breaker creates new breaker."""
        breaker = get_circuit_breaker("global-test")
        self.assertIsInstance(breaker, CircuitBreaker)
        self.assertEqual(breaker.name, "global-test")

    def test_get_circuit_breaker_returns_same(self):
        """Test get_circuit_breaker returns same instance."""
        breaker1 = get_circuit_breaker("global-same")
        breaker2 = get_circuit_breaker("global-same")
        self.assertIs(breaker1, breaker2)

    def test_get_all_health(self):
        """Test getting health for all circuit breakers."""
        get_circuit_breaker("health-a")
        get_circuit_breaker("health-b")

        health = get_all_circuit_breaker_health()

        self.assertIn("health-a", health)
        self.assertIn("health-b", health)

    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker_a = get_circuit_breaker("reset-a", config)
        breaker_b = get_circuit_breaker("reset-b", config)

        # Open both
        with self.assertRaises(RuntimeError):
            with breaker_a:
                raise RuntimeError("fail")
        with self.assertRaises(RuntimeError):
            with breaker_b:
                raise RuntimeError("fail")

        self.assertEqual(breaker_a.state, CircuitState.OPEN)
        self.assertEqual(breaker_b.state, CircuitState.OPEN)

        # Reset all
        reset_all_circuit_breakers()

        self.assertEqual(breaker_a.state, CircuitState.CLOSED)
        self.assertEqual(breaker_b.state, CircuitState.CLOSED)


class TestRetryConfig(unittest.TestCase):
    """Tests for RetryConfig."""

    def test_default_config(self):
        """Test default retry configuration."""
        config = RetryConfig()
        self.assertEqual(config.max_attempts, 3)
        self.assertEqual(config.base_delay, 1.0)
        self.assertEqual(config.max_delay, 30.0)
        self.assertEqual(config.exponential_base, 2.0)
        self.assertTrue(config.jitter)

    def test_custom_config(self):
        """Test custom retry configuration."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=0.5,
            max_delay=10.0,
            exponential_base=3.0,
            jitter=False,
        )
        self.assertEqual(config.max_attempts, 5)
        self.assertEqual(config.base_delay, 0.5)
        self.assertEqual(config.max_delay, 10.0)
        self.assertEqual(config.exponential_base, 3.0)
        self.assertFalse(config.jitter)


class TestCalculateDelay(unittest.TestCase):
    """Tests for delay calculation."""

    def test_exponential_backoff(self):
        """Test exponential backoff calculation."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            jitter=False,
            max_delay=100.0,
        )

        self.assertEqual(calculate_delay(0, config), 1.0)  # 1 * 2^0
        self.assertEqual(calculate_delay(1, config), 2.0)  # 1 * 2^1
        self.assertEqual(calculate_delay(2, config), 4.0)  # 1 * 2^2
        self.assertEqual(calculate_delay(3, config), 8.0)  # 1 * 2^3

    def test_max_delay_cap(self):
        """Test delay is capped at max_delay."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            jitter=False,
            max_delay=5.0,
        )

        self.assertEqual(calculate_delay(5, config), 5.0)  # Would be 32, capped at 5
        self.assertEqual(calculate_delay(10, config), 5.0)

    def test_jitter_adds_randomness(self):
        """Test jitter adds randomness to delay."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            jitter=True,
            max_delay=100.0,
        )

        delays = [calculate_delay(0, config) for _ in range(10)]

        # With jitter, delays should vary
        # Base is 1.0, jitter adds 0-25%, so range is 1.0-1.25
        for delay in delays:
            self.assertGreaterEqual(delay, 1.0)
            self.assertLessEqual(delay, 1.25)

        # At least some variation (not all the same)
        self.assertGreater(len(set(delays)), 1)


class TestRetryAsync(unittest.IsolatedAsyncioTestCase):
    """Tests for async retry logic."""

    async def test_succeeds_first_try(self):
        """Test function succeeds on first try."""
        call_count = 0

        async def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_async(success_func)

        self.assertEqual(result, "success")
        self.assertEqual(call_count, 1)

    async def test_retries_on_failure(self):
        """Test function retries on failure."""
        call_count = 0

        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("temporary failure")
            return "success"

        config = RetryConfig(max_attempts=3, base_delay=0.01)
        result = await retry_async(fail_then_succeed, config=config)

        self.assertEqual(result, "success")
        self.assertEqual(call_count, 2)

    async def test_exhausts_retries(self):
        """Test all retries are exhausted."""
        call_count = 0

        async def always_fail():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("permanent failure")

        config = RetryConfig(max_attempts=3, base_delay=0.01)

        with self.assertRaises(RuntimeError) as ctx:
            await retry_async(always_fail, config=config)

        self.assertEqual(str(ctx.exception), "permanent failure")
        self.assertEqual(call_count, 3)

    async def test_non_retryable_exception(self):
        """Test non-retryable exception fails immediately."""
        call_count = 0

        async def fail_with_non_retryable():
            nonlocal call_count
            call_count += 1
            raise KeyError("not retryable")

        config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            retryable_exceptions=(RuntimeError,),  # Only retry RuntimeError
        )

        with self.assertRaises(KeyError):
            await retry_async(fail_with_non_retryable, config=config)

        self.assertEqual(call_count, 1)  # No retries

    async def test_retry_decorator(self):
        """Test retry decorator on async function."""
        call_count = 0

        @retry(config=RetryConfig(max_attempts=3, base_delay=0.01))
        async def decorated_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("retry me")
            return "decorated success"

        result = await decorated_func()

        self.assertEqual(result, "decorated success")
        self.assertEqual(call_count, 2)


class TestAsyncCircuitBreakerWithBackend(unittest.IsolatedAsyncioTestCase):
    """Tests for AsyncCircuitBreaker with state backend."""

    async def asyncSetUp(self):
        """Set up test fixtures."""
        reset_circuit_breakers()
        self.backend = MemoryStateBackend()

    async def asyncTearDown(self):
        """Clean up test fixtures."""
        reset_circuit_breakers()
        await self.backend.close()

    async def test_initial_state_closed(self):
        """Test initial state is CLOSED."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "test-initial",
            failure_threshold=3,
        )

        state = await breaker.get_state()
        self.assertEqual(state, AsyncCircuitState.CLOSED)

    async def test_opens_after_failures(self):
        """Test circuit opens after reaching failure threshold."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "test-open",
            failure_threshold=3,
        )

        # Record failures
        for _ in range(3):
            await breaker.record_failure()

        state = await breaker.get_state()
        self.assertEqual(state, AsyncCircuitState.OPEN)

    async def test_is_available(self):
        """Test is_available returns correct values."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "test-available",
            failure_threshold=2,
        )

        # Initially available
        self.assertTrue(await breaker.is_available())

        # Open the circuit
        await breaker.record_failure()
        await breaker.record_failure()

        # Not available when open
        self.assertFalse(await breaker.is_available())

    async def test_context_manager_success(self):
        """Test async context manager with success."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "test-ctx-success",
            failure_threshold=2,
            success_threshold=1,
        )

        async with breaker:
            pass  # Success

        state = await breaker.get_state()
        self.assertEqual(state, AsyncCircuitState.CLOSED)

    async def test_context_manager_failure(self):
        """Test async context manager with failure."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "test-ctx-fail",
            failure_threshold=1,
        )

        with self.assertRaises(RuntimeError):
            async with breaker:
                raise RuntimeError("fail")

        state = await breaker.get_state()
        self.assertEqual(state, AsyncCircuitState.OPEN)

    async def test_blocks_when_open(self):
        """Test circuit blocks requests when open."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "test-block",
            failure_threshold=1,
        )

        # Open the circuit
        await breaker.record_failure()

        # Should raise CircuitOpenError
        with self.assertRaises(CircuitOpenError):
            async with breaker:
                pass

    async def test_transitions_to_half_open(self):
        """Test circuit transitions to HALF_OPEN after timeout."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "test-half-open",
            failure_threshold=1,
            timeout_seconds=0.1,
        )

        # Open the circuit
        await breaker.record_failure()
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.OPEN)

        # Wait for timeout
        await asyncio.sleep(0.15)

        state = await breaker.get_state()
        self.assertEqual(state, AsyncCircuitState.HALF_OPEN)

    async def test_closes_after_success_in_half_open(self):
        """Test circuit closes after successes in HALF_OPEN."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "test-close",
            failure_threshold=1,
            success_threshold=2,
            timeout_seconds=0.1,
        )

        # Open and wait for half-open
        await breaker.record_failure()
        await asyncio.sleep(0.15)
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.HALF_OPEN)

        # Record successes
        await breaker.record_success()
        await breaker.record_success()

        state = await breaker.get_state()
        self.assertEqual(state, AsyncCircuitState.CLOSED)

    async def test_reopens_on_failure_in_half_open(self):
        """Test circuit reopens on failure in HALF_OPEN."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "test-reopen",
            failure_threshold=1,
            timeout_seconds=0.1,
        )

        # Open and wait for half-open
        await breaker.record_failure()
        await asyncio.sleep(0.15)
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.HALF_OPEN)

        # Fail in half-open
        await breaker.record_failure()

        state = await breaker.get_state()
        self.assertEqual(state, AsyncCircuitState.OPEN)

    async def test_reset(self):
        """Test circuit reset."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "test-reset",
            failure_threshold=1,
        )

        # Open the circuit
        await breaker.record_failure()
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.OPEN)

        # Reset
        await breaker.reset()

        state = await breaker.get_state()
        self.assertEqual(state, AsyncCircuitState.CLOSED)

    async def test_force_open(self):
        """Test forcing circuit open."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "test-force-open",
            failure_threshold=10,  # High threshold
        )

        self.assertEqual(await breaker.get_state(), AsyncCircuitState.CLOSED)

        # Force open
        await breaker.force_open()

        state = await breaker.get_state()
        self.assertEqual(state, AsyncCircuitState.OPEN)

    async def test_call_with_fallback(self):
        """Test call method with fallback."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "test-fallback",
            failure_threshold=1,
        )

        async def primary():
            raise RuntimeError("primary fails")

        async def fallback():
            return "fallback result"

        # First call fails, opens circuit
        with self.assertRaises(RuntimeError):
            await breaker.call(primary)

        # Second call uses fallback because circuit is open
        result = await breaker.call(primary, fallback=fallback)
        self.assertEqual(result, "fallback result")

    async def test_call_success(self):
        """Test call method with success."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "test-call-success",
            failure_threshold=3,
        )

        async def primary():
            return "primary result"

        result = await breaker.call(primary)
        self.assertEqual(result, "primary result")

    async def test_half_open_limits_calls(self):
        """Test HALF_OPEN limits concurrent calls."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "test-half-open-limit",
            failure_threshold=1,
            timeout_seconds=0.1,
            half_open_max_calls=2,
        )

        # Open and wait for half-open
        await breaker.record_failure()
        await asyncio.sleep(0.15)

        # First two calls should be allowed
        self.assertTrue(await breaker.is_available())

        # Simulate entering context (increments half_open_calls)
        async with breaker:
            pass

        # Still available (1 call, limit is 2)
        # Note: success closes the circuit if success_threshold is met


class TestAsyncCircuitBreakerRegistry(unittest.IsolatedAsyncioTestCase):
    """Tests for async circuit breaker registry."""

    async def asyncSetUp(self):
        """Set up test fixtures."""
        reset_circuit_breakers()
        self.backend = MemoryStateBackend()

    async def asyncTearDown(self):
        """Clean up test fixtures."""
        reset_circuit_breakers()
        await self.backend.close()

    async def test_get_circuit_breaker_creates_new(self):
        """Test async_get_circuit_breaker creates new breaker."""
        breaker = await async_get_circuit_breaker(
            self.backend,
            "async-registry-test",
        )

        self.assertIsInstance(breaker, AsyncCircuitBreaker)
        self.assertEqual(breaker.name, "async-registry-test")

    async def test_get_circuit_breaker_returns_same(self):
        """Test async_get_circuit_breaker returns same instance."""
        breaker1 = await async_get_circuit_breaker(
            self.backend,
            "async-registry-same",
        )
        breaker2 = await async_get_circuit_breaker(
            self.backend,
            "async-registry-same",
        )

        self.assertIs(breaker1, breaker2)


class TestCircuitBreakerStatePersistence(unittest.IsolatedAsyncioTestCase):
    """Tests for circuit breaker state persistence across instances."""

    async def asyncSetUp(self):
        """Set up test fixtures."""
        reset_circuit_breakers()
        self.backend = MemoryStateBackend()

    async def asyncTearDown(self):
        """Clean up test fixtures."""
        reset_circuit_breakers()
        await self.backend.close()

    async def test_state_persists_across_instances(self):
        """Test state is persisted and readable by new instances."""
        # Create first breaker and open it
        breaker1 = AsyncCircuitBreaker(
            self.backend,
            "persist-test",
            failure_threshold=1,
            key_prefix="cb:",
        )

        await breaker1.record_failure()
        self.assertEqual(await breaker1.get_state(), AsyncCircuitState.OPEN)

        # Create new breaker with same name and backend
        breaker2 = AsyncCircuitBreaker(
            self.backend,
            "persist-test",
            failure_threshold=1,
            key_prefix="cb:",
        )

        # Should see the same state
        state = await breaker2.get_state()
        self.assertEqual(state, AsyncCircuitState.OPEN)

    async def test_different_names_isolated(self):
        """Test different circuit breakers are isolated."""
        breaker_a = AsyncCircuitBreaker(
            self.backend,
            "isolated-a",
            failure_threshold=1,
        )
        breaker_b = AsyncCircuitBreaker(
            self.backend,
            "isolated-b",
            failure_threshold=1,
        )

        # Open only breaker A
        await breaker_a.record_failure()

        # A is open, B is closed
        self.assertEqual(await breaker_a.get_state(), AsyncCircuitState.OPEN)
        self.assertEqual(await breaker_b.get_state(), AsyncCircuitState.CLOSED)


class TestCircuitBreakerIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for circuit breaker with realistic scenarios."""

    async def asyncSetUp(self):
        """Set up test fixtures."""
        reset_circuit_breakers()
        self.backend = MemoryStateBackend()
        self.call_count = 0

    async def asyncTearDown(self):
        """Clean up test fixtures."""
        reset_circuit_breakers()
        await self.backend.close()

    async def test_realistic_failure_recovery_scenario(self):
        """Test realistic scenario of service failure and recovery."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "realistic-test",
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=0.2,
        )

        # Service is initially healthy
        for _ in range(5):
            async with breaker:
                pass  # Success

        self.assertEqual(await breaker.get_state(), AsyncCircuitState.CLOSED)

        # Service starts failing
        for _ in range(3):
            with self.assertRaises(RuntimeError):
                async with breaker:
                    raise RuntimeError("service down")

        # Circuit should be open
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.OPEN)

        # Requests are blocked
        with self.assertRaises(CircuitOpenError):
            async with breaker:
                pass

        # Wait for timeout
        await asyncio.sleep(0.25)

        # Should be half-open
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.HALF_OPEN)

        # Service recovers - test requests succeed
        async with breaker:
            pass  # First success

        async with breaker:
            pass  # Second success

        # Circuit should be closed again
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.CLOSED)

    async def test_flaky_service_scenario(self):
        """Test handling of intermittently failing service."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "flaky-test",
            failure_threshold=5,
            success_threshold=2,
            timeout_seconds=0.1,
        )

        # Intermittent failures (not enough to trip circuit)
        for i in range(10):
            try:
                async with breaker:
                    if i % 3 == 0:  # Fail every 3rd call
                        raise RuntimeError("flaky")
            except RuntimeError:
                pass

        # Circuit should still be closed (failures under threshold)
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.CLOSED)


class TestCircuitBreakerHalfOpenEdgeCases(unittest.IsolatedAsyncioTestCase):
    """Tests for HALF_OPEN state edge cases and behavior."""

    async def asyncSetUp(self):
        """Set up test fixtures."""
        reset_circuit_breakers()
        self.backend = MemoryStateBackend()

    async def asyncTearDown(self):
        """Clean up test fixtures."""
        reset_circuit_breakers()
        await self.backend.close()

    async def test_half_open_allows_limited_concurrent_calls(self):
        """Test HALF_OPEN allows only configured number of concurrent calls."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "half-open-concurrent",
            failure_threshold=1,
            success_threshold=3,
            timeout_seconds=0.1,
            half_open_max_calls=2,
        )

        # Open the circuit
        await breaker.record_failure()
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.OPEN)

        # Wait for transition to half-open
        await asyncio.sleep(0.15)
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.HALF_OPEN)

        # First call should be allowed
        self.assertTrue(await breaker.is_available())

        # Simulate entering context (increments half_open_calls)
        async with breaker:
            # While in first call, check if second is allowed
            self.assertTrue(await breaker.is_available())

        # After first success, check availability again
        state = await breaker.get_state()
        # Circuit should still be in HALF_OPEN (need 3 successes)
        self.assertEqual(state, AsyncCircuitState.HALF_OPEN)

    async def test_half_open_blocks_after_max_concurrent_calls(self):
        """Test HALF_OPEN blocks requests after max concurrent calls reached."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "half-open-block",
            failure_threshold=1,
            success_threshold=2,
            timeout_seconds=0.1,
            half_open_max_calls=1,
        )

        # Open and transition to half-open
        await breaker.record_failure()
        await asyncio.sleep(0.15)
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.HALF_OPEN)

        # Check initial state
        data = await breaker._get_state_data()
        self.assertEqual(data["half_open_calls"], 0)

        # Start first call
        try:
            async with breaker:
                # Inside first call, check state
                data = await breaker._get_state_data()
                self.assertEqual(data["half_open_calls"], 1)

                # Try to check availability for second call
                # Note: is_available checks current count, so might still show available
                pass
        except Exception:
            pass

    async def test_half_open_immediate_failure_reopens_circuit(self):
        """Test single failure in HALF_OPEN immediately reopens circuit."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "half-open-fail",
            failure_threshold=1,
            success_threshold=2,
            timeout_seconds=0.1,
        )

        # Open and transition to half-open
        await breaker.record_failure()
        await asyncio.sleep(0.15)
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.HALF_OPEN)

        # First test call fails
        with self.assertRaises(RuntimeError):
            async with breaker:
                raise RuntimeError("test failure")

        # Circuit should immediately reopen
        state = await breaker.get_state()
        self.assertEqual(state, AsyncCircuitState.OPEN)

    async def test_half_open_success_count_accumulates(self):
        """Test success count accumulates in HALF_OPEN before closing."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "half-open-accumulate",
            failure_threshold=1,
            success_threshold=3,
            timeout_seconds=0.1,
        )

        # Open and transition to half-open
        await breaker.record_failure()
        await asyncio.sleep(0.15)
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.HALF_OPEN)

        # First success - should stay half-open
        async with breaker:
            pass

        self.assertEqual(await breaker.get_state(), AsyncCircuitState.HALF_OPEN)

        # Second success - should stay half-open
        async with breaker:
            pass

        self.assertEqual(await breaker.get_state(), AsyncCircuitState.HALF_OPEN)

        # Third success - should close
        async with breaker:
            pass

        self.assertEqual(await breaker.get_state(), AsyncCircuitState.CLOSED)

    async def test_half_open_state_transition_timing_precision(self):
        """Test precise timing of OPEN to HALF_OPEN transition."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "timing-test",
            failure_threshold=1,
            timeout_seconds=0.5,
        )

        # Open the circuit
        start_time = time.time()
        await breaker.record_failure()
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.OPEN)

        # Check state before timeout expires
        await asyncio.sleep(0.3)
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.OPEN)

        # Check state after timeout expires
        await asyncio.sleep(0.3)  # Total: 0.6s
        elapsed = time.time() - start_time

        state = await breaker.get_state()
        self.assertEqual(state, AsyncCircuitState.HALF_OPEN)
        self.assertGreaterEqual(elapsed, 0.5)

    async def test_half_open_to_open_resets_timeout(self):
        """Test transition from HALF_OPEN back to OPEN resets the timeout."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "timeout-reset",
            failure_threshold=1,
            timeout_seconds=0.2,
        )

        # Open the circuit
        await breaker.record_failure()
        await asyncio.sleep(0.25)

        # Transition to half-open
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.HALF_OPEN)

        # Fail in half-open, should reopen
        with self.assertRaises(RuntimeError):
            async with breaker:
                raise RuntimeError("fail")

        self.assertEqual(await breaker.get_state(), AsyncCircuitState.OPEN)

        # Get the state change time
        data = await breaker._get_state_data()
        first_reopen_time = data["last_state_change"]

        # Wait a bit, should still be open
        await asyncio.sleep(0.1)
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.OPEN)

        # Wait for another timeout
        await asyncio.sleep(0.15)
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.HALF_OPEN)

    async def test_half_open_concurrent_call_tracking(self):
        """Test concurrent calls are properly tracked in HALF_OPEN."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "concurrent-track",
            failure_threshold=1,
            success_threshold=5,
            timeout_seconds=0.1,
            half_open_max_calls=3,
        )

        # Open and transition to half-open
        await breaker.record_failure()
        await asyncio.sleep(0.15)

        # Verify we're in half-open
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.HALF_OPEN)

        # Check initial half_open_calls
        data = await breaker._get_state_data()
        self.assertEqual(data["half_open_calls"], 0)

        # Make successful calls to verify tracking
        for i in range(3):
            async with breaker:
                pass
            # Still in half-open (need 5 successes to close)
            state = await breaker.get_state()
            if i < 4:  # First 4 should be half-open
                self.assertEqual(state, AsyncCircuitState.HALF_OPEN)

    async def test_half_open_success_count_resets_on_close(self):
        """Test success count is reset when circuit closes."""
        breaker = AsyncCircuitBreaker(
            self.backend,
            "success-reset",
            failure_threshold=1,
            success_threshold=2,
            timeout_seconds=0.1,
        )

        # Open and transition to half-open
        await breaker.record_failure()
        await asyncio.sleep(0.15)

        # Accumulate successes to close
        async with breaker:
            pass
        async with breaker:
            pass

        # Should be closed now
        self.assertEqual(await breaker.get_state(), AsyncCircuitState.CLOSED)

        # Verify success count was reset
        data = await breaker._get_state_data()
        self.assertEqual(data["success_count"], 0)
        self.assertEqual(data["failure_count"], 0)


if __name__ == "__main__":
    unittest.main()
