"""Tests for the pluggable state backend system.

Tests the core backends and adapters:
- MemoryStateBackend (used for testing)
- SQLiteStateBackend (default production backend)
- Adapters: AsyncRateLimiter, AsyncCircuitBreaker, AsyncTTLCache, OIDCStateStore
"""

import asyncio
import tempfile
import time
import unittest
from pathlib import Path


class TestMemoryStateBackend(unittest.IsolatedAsyncioTestCase):
    """Test the in-memory state backend."""

    async def asyncSetUp(self):
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.backends.memory import MemoryStateBackend

        self.backend = MemoryStateBackend()

    async def asyncTearDown(self):
        await self.backend.close()

    async def test_ping(self):
        """Test ping returns True."""
        self.assertTrue(await self.backend.ping())

    async def test_get_set_delete(self):
        """Test basic key-value operations."""
        # Set a value
        await self.backend.set("test_key", b"test_value")

        # Get it back
        value = await self.backend.get("test_key")
        self.assertEqual(value, b"test_value")

        # Check exists
        self.assertTrue(await self.backend.exists("test_key"))

        # Delete it
        deleted = await self.backend.delete("test_key")
        self.assertTrue(deleted)

        # Should be gone
        self.assertIsNone(await self.backend.get("test_key"))
        self.assertFalse(await self.backend.exists("test_key"))

    async def test_set_with_ttl(self):
        """Test TTL expiration."""
        # Set with 1 second TTL
        await self.backend.set("ttl_key", b"expires_soon", ttl=1)

        # Should exist immediately
        self.assertEqual(await self.backend.get("ttl_key"), b"expires_soon")

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        self.assertIsNone(await self.backend.get("ttl_key"))

    async def test_incr(self):
        """Test atomic counter increment."""
        # Increment non-existent key
        result = await self.backend.incr("counter")
        self.assertEqual(result, 1)

        # Increment again
        result = await self.backend.incr("counter")
        self.assertEqual(result, 2)

        # Increment by amount
        result = await self.backend.incr("counter", 5)
        self.assertEqual(result, 7)

        # Get counter value
        value = await self.backend.get_counter("counter")
        self.assertEqual(value, 7)

    async def test_sliding_window(self):
        """Test sliding window operations."""
        now = time.time()
        window = 10  # 10 second window

        # Add timestamps
        count1 = await self.backend.sliding_window_add("window_key", now, window)
        self.assertEqual(count1, 1)

        count2 = await self.backend.sliding_window_add("window_key", now + 1, window)
        self.assertEqual(count2, 2)

        count3 = await self.backend.sliding_window_add("window_key", now + 2, window)
        self.assertEqual(count3, 3)

        # Check count
        count = await self.backend.sliding_window_count("window_key", window)
        self.assertEqual(count, 3)

    async def test_hash_operations(self):
        """Test hash field operations."""
        # Set fields
        await self.backend.hset("hash_key", "field1", b"value1")
        await self.backend.hset("hash_key", "field2", b"value2")

        # Get individual field
        value = await self.backend.hget("hash_key", "field1")
        self.assertEqual(value, b"value1")

        # Get all fields
        all_fields = await self.backend.hgetall("hash_key")
        self.assertEqual(len(all_fields), 2)
        self.assertEqual(all_fields[b"field1"], b"value1")
        self.assertEqual(all_fields[b"field2"], b"value2")

        # Delete field
        deleted = await self.backend.hdel("hash_key", "field1")
        self.assertTrue(deleted)

        # Field should be gone
        self.assertIsNone(await self.backend.hget("hash_key", "field1"))

    async def test_set_operations(self):
        """Test set operations."""
        # Add members
        added = await self.backend.sadd("set_key", b"member1", b"member2")
        self.assertEqual(added, 2)

        # Add duplicate
        added = await self.backend.sadd("set_key", b"member1")
        self.assertEqual(added, 0)

        # Check membership
        self.assertTrue(await self.backend.sismember("set_key", b"member1"))
        self.assertFalse(await self.backend.sismember("set_key", b"nonexistent"))

        # Get all members
        members = await self.backend.smembers("set_key")
        self.assertEqual(len(members), 2)
        self.assertIn(b"member1", members)
        self.assertIn(b"member2", members)

        # Remove member
        removed = await self.backend.srem("set_key", b"member1")
        self.assertEqual(removed, 1)

        # Should be gone
        self.assertFalse(await self.backend.sismember("set_key", b"member1"))

    async def test_mget_mset(self):
        """Test batch operations."""
        # Set multiple
        await self.backend.mset(
            {
                "key1": b"value1",
                "key2": b"value2",
                "key3": b"value3",
            }
        )

        # Get multiple
        values = await self.backend.mget("key1", "key2", "key3", "nonexistent")
        self.assertEqual(values, [b"value1", b"value2", b"value3", None])

    async def test_delete_prefix(self):
        """Test prefix deletion."""
        # Set keys with prefix
        await self.backend.set("prefix:key1", b"value1")
        await self.backend.set("prefix:key2", b"value2")
        await self.backend.set("other:key", b"value3")

        # Delete prefix
        deleted = await self.backend.delete_prefix("prefix:")
        self.assertEqual(deleted, 2)

        # Prefix keys should be gone
        self.assertIsNone(await self.backend.get("prefix:key1"))
        self.assertIsNone(await self.backend.get("prefix:key2"))

        # Other key should remain
        self.assertEqual(await self.backend.get("other:key"), b"value3")


class TestSQLiteStateBackend(unittest.IsolatedAsyncioTestCase):
    """Test the SQLite state backend."""

    async def asyncSetUp(self):
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.backends.sqlite import SQLiteStateBackend
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.config import StateBackendConfig

        # Use temp directory for test database
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_state.db"

        config = StateBackendConfig(
            backend="sqlite",
            url=f"sqlite:///{self.db_path}",
        )
        self.backend = SQLiteStateBackend(config)

    async def asyncTearDown(self):
        await self.backend.close()
        # Clean up temp files
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    async def test_ping(self):
        """Test ping returns True."""
        self.assertTrue(await self.backend.ping())

    async def test_get_set_delete(self):
        """Test basic key-value operations."""
        await self.backend.set("test_key", b"test_value")
        value = await self.backend.get("test_key")
        self.assertEqual(value, b"test_value")

        deleted = await self.backend.delete("test_key")
        self.assertTrue(deleted)
        self.assertIsNone(await self.backend.get("test_key"))

    async def test_ttl_expiration(self):
        """Test TTL works with SQLite."""
        await self.backend.set("ttl_key", b"expires", ttl=1)
        self.assertEqual(await self.backend.get("ttl_key"), b"expires")

        await asyncio.sleep(1.1)
        self.assertIsNone(await self.backend.get("ttl_key"))

    async def test_counter_persistence(self):
        """Test counters persist correctly."""
        await self.backend.incr("counter", 5)
        await self.backend.incr("counter", 3)

        value = await self.backend.get_counter("counter")
        self.assertEqual(value, 8)

    async def test_hash_with_ttl(self):
        """Test hash with TTL."""
        await self.backend.hset_with_ttl("hash_key", "field", b"value", ttl=1)
        self.assertEqual(await self.backend.hget("hash_key", "field"), b"value")

        await asyncio.sleep(1.1)
        self.assertIsNone(await self.backend.hget("hash_key", "field"))


class TestAsyncRateLimiter(unittest.IsolatedAsyncioTestCase):
    """Test the async rate limiter adapter."""

    async def asyncSetUp(self):
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.adapters.rate_limiter import AsyncRateLimiter
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.backends.memory import MemoryStateBackend

        self.backend = MemoryStateBackend()
        self.limiter = AsyncRateLimiter(
            self.backend,
            max_requests=5,
            window_seconds=10,
        )

    async def asyncTearDown(self):
        await self.backend.close()

    async def test_allows_within_limit(self):
        """Test requests within limit are allowed."""
        for i in range(5):
            allowed = await self.limiter.is_allowed("user1")
            self.assertTrue(allowed, f"Request {i+1} should be allowed")

    async def test_blocks_over_limit(self):
        """Test requests over limit are blocked."""
        # Use up the limit
        for _ in range(5):
            await self.limiter.is_allowed("user1")

        # Next request should be blocked
        allowed = await self.limiter.is_allowed("user1")
        self.assertFalse(allowed)

    async def test_separate_keys(self):
        """Test different keys have separate limits."""
        # Use up user1's limit
        for _ in range(5):
            await self.limiter.is_allowed("user1")

        # user2 should still be allowed
        allowed = await self.limiter.is_allowed("user2")
        self.assertTrue(allowed)

    async def test_check_and_update(self):
        """Test check_and_update returns details."""
        is_allowed, count, remaining = await self.limiter.check_and_update("user1")
        self.assertTrue(is_allowed)
        self.assertEqual(count, 1)
        self.assertEqual(remaining, 4)

    async def test_get_remaining(self):
        """Test get_remaining doesn't consume quota."""
        # Check remaining (shouldn't consume)
        remaining = await self.limiter.get_remaining("user1")
        self.assertEqual(remaining, 5)

        # Still have full quota
        remaining = await self.limiter.get_remaining("user1")
        self.assertEqual(remaining, 5)

    async def test_reset(self):
        """Test reset clears the limit."""
        # Use up limit
        for _ in range(5):
            await self.limiter.is_allowed("user1")

        # Reset
        await self.limiter.reset("user1")

        # Should be allowed again
        allowed = await self.limiter.is_allowed("user1")
        self.assertTrue(allowed)


class TestAsyncCircuitBreaker(unittest.IsolatedAsyncioTestCase):
    """Test the async circuit breaker adapter."""

    async def asyncSetUp(self):
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.adapters.circuit_breaker import (
            AsyncCircuitBreaker,
            CircuitState,
            reset_circuit_breakers,
        )
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.backends.memory import MemoryStateBackend

        reset_circuit_breakers()
        self.backend = MemoryStateBackend()
        self.CircuitState = CircuitState
        self.breaker = AsyncCircuitBreaker(
            self.backend,
            "test-service",
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=1,
        )

    async def asyncTearDown(self):
        await self.backend.close()

    async def test_starts_closed(self):
        """Test circuit starts in closed state."""
        state = await self.breaker.get_state()
        self.assertEqual(state, self.CircuitState.CLOSED)

    async def test_opens_after_failures(self):
        """Test circuit opens after failure threshold."""
        # Record failures
        for _ in range(3):
            await self.breaker.record_failure()

        state = await self.breaker.get_state()
        self.assertEqual(state, self.CircuitState.OPEN)

    async def test_rejects_when_open(self):
        """Test circuit rejects requests when open."""
        # Open the circuit
        for _ in range(3):
            await self.breaker.record_failure()

        # Should not be available
        available = await self.breaker.is_available()
        self.assertFalse(available)

    async def test_transitions_to_half_open(self):
        """Test circuit transitions to half-open after timeout."""
        # Open the circuit
        for _ in range(3):
            await self.breaker.record_failure()

        # Wait for timeout
        await asyncio.sleep(1.1)

        # Should be half-open now
        state = await self.breaker.get_state()
        self.assertEqual(state, self.CircuitState.HALF_OPEN)

    async def test_closes_after_successes(self):
        """Test circuit closes after success threshold in half-open."""
        # Open the circuit
        for _ in range(3):
            await self.breaker.record_failure()

        # Wait for half-open
        await asyncio.sleep(1.1)

        # Record successes
        await self.breaker.record_success()
        await self.breaker.record_success()

        state = await self.breaker.get_state()
        self.assertEqual(state, self.CircuitState.CLOSED)

    async def test_context_manager_success(self):
        """Test context manager records success."""
        async with self.breaker:
            pass  # Success

        # Should still be closed
        state = await self.breaker.get_state()
        self.assertEqual(state, self.CircuitState.CLOSED)

    async def test_context_manager_failure(self):
        """Test context manager records failure on exception."""
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.adapters.circuit_breaker import CircuitOpenError

        # Cause failures
        for _ in range(3):
            try:
                async with self.breaker:
                    raise ValueError("test error")
            except ValueError:
                pass

        # Should be open
        state = await self.breaker.get_state()
        self.assertEqual(state, self.CircuitState.OPEN)

        # Context manager should raise CircuitOpenError
        with self.assertRaises(CircuitOpenError):
            async with self.breaker:
                pass

    async def test_reset(self):
        """Test manual reset."""
        # Open the circuit
        for _ in range(3):
            await self.breaker.record_failure()

        # Reset
        await self.breaker.reset()

        state = await self.breaker.get_state()
        self.assertEqual(state, self.CircuitState.CLOSED)


class TestAsyncTTLCache(unittest.IsolatedAsyncioTestCase):
    """Test the async TTL cache adapter."""

    async def asyncSetUp(self):
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.adapters.ttl_cache import AsyncTTLCache
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.backends.memory import MemoryStateBackend

        self.backend = MemoryStateBackend()
        self.cache = AsyncTTLCache[dict](
            self.backend,
            ttl_seconds=10,
            key_prefix="test_cache:",
        )

    async def asyncTearDown(self):
        await self.backend.close()

    async def test_get_set(self):
        """Test basic cache get/set."""
        await self.cache.set("key1", {"foo": "bar"})
        value = await self.cache.get("key1")
        self.assertEqual(value, {"foo": "bar"})

    async def test_ttl_expiration(self):
        """Test cache TTL."""
        await self.cache.set("key1", {"foo": "bar"}, ttl=1)
        self.assertIsNotNone(await self.cache.get("key1"))

        await asyncio.sleep(1.1)
        self.assertIsNone(await self.cache.get("key1"))

    async def test_get_or_set(self):
        """Test get_or_set computes on miss."""
        call_count = 0

        def factory():
            nonlocal call_count
            call_count += 1
            return {"computed": True}

        # First call - should compute
        value = await self.cache.get_or_set("key1", factory)
        self.assertEqual(value, {"computed": True})
        self.assertEqual(call_count, 1)

        # Second call - should use cache
        value = await self.cache.get_or_set("key1", factory)
        self.assertEqual(value, {"computed": True})
        self.assertEqual(call_count, 1)  # Factory not called again

    async def test_get_or_set_async(self):
        """Test get_or_set_async with async factory."""
        call_count = 0

        async def async_factory():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return {"async_computed": True}

        # First call
        value = await self.cache.get_or_set_async("key1", async_factory)
        self.assertEqual(value, {"async_computed": True})
        self.assertEqual(call_count, 1)

        # Second call - cached
        value = await self.cache.get_or_set_async("key1", async_factory)
        self.assertEqual(call_count, 1)

    async def test_invalidate(self):
        """Test cache invalidation."""
        await self.cache.set("key1", {"foo": "bar"})
        self.assertIsNotNone(await self.cache.get("key1"))

        await self.cache.invalidate("key1")
        self.assertIsNone(await self.cache.get("key1"))

    async def test_invalidate_prefix(self):
        """Test prefix invalidation."""
        await self.cache.set("user:1:data", {"id": 1})
        await self.cache.set("user:2:data", {"id": 2})
        await self.cache.set("other:key", {"other": True})

        # Invalidate user prefix
        count = await self.cache.invalidate_prefix("user:")
        self.assertEqual(count, 2)

        # User keys gone
        self.assertIsNone(await self.cache.get("user:1:data"))
        self.assertIsNone(await self.cache.get("user:2:data"))

        # Other key remains
        self.assertIsNotNone(await self.cache.get("other:key"))

    async def test_clear(self):
        """Test clearing entire cache."""
        await self.cache.set("key1", {"a": 1})
        await self.cache.set("key2", {"b": 2})

        count = await self.cache.clear()
        self.assertEqual(count, 2)

        self.assertIsNone(await self.cache.get("key1"))
        self.assertIsNone(await self.cache.get("key2"))


class TestOIDCStateStore(unittest.IsolatedAsyncioTestCase):
    """Test the OIDC state store adapter."""

    async def asyncSetUp(self):
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.adapters.oidc_state import OIDCStateStore
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.backends.memory import MemoryStateBackend

        self.backend = MemoryStateBackend()
        self.store = OIDCStateStore(
            self.backend,
            pending_ttl=10,
            discovery_ttl=10,
            jwks_ttl=10,
        )

    async def asyncTearDown(self):
        await self.backend.close()

    async def test_create_pending_state(self):
        """Test creating pending auth state."""
        pending = await self.store.create_pending_state(
            provider_id="google",
            redirect_uri="/callback",
            use_pkce=True,
        )

        self.assertIsNotNone(pending.state)
        self.assertIsNotNone(pending.nonce)
        self.assertEqual(pending.provider_id, "google")
        self.assertEqual(pending.redirect_uri, "/callback")
        self.assertIsNotNone(pending.code_verifier)  # PKCE enabled

    async def test_get_pending_state(self):
        """Test retrieving pending state."""
        pending = await self.store.create_pending_state(
            provider_id="okta",
            redirect_uri="/auth/callback",
        )

        # Get without consuming
        retrieved = await self.store.get_pending_state(pending.state)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.provider_id, "okta")

        # Should still exist
        retrieved2 = await self.store.get_pending_state(pending.state)
        self.assertIsNotNone(retrieved2)

    async def test_consume_pending_state(self):
        """Test consuming (one-time use) pending state."""
        pending = await self.store.create_pending_state(
            provider_id="azure",
            redirect_uri="/callback",
        )

        # Consume
        consumed = await self.store.consume_pending_state(pending.state)
        self.assertIsNotNone(consumed)
        self.assertEqual(consumed.provider_id, "azure")

        # Should be gone
        consumed2 = await self.store.consume_pending_state(pending.state)
        self.assertIsNone(consumed2)

    async def test_pending_state_expiration(self):
        """Test pending state expires."""
        # Use short TTL store
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.adapters.oidc_state import OIDCStateStore

        short_store = OIDCStateStore(self.backend, pending_ttl=1)

        pending = await short_store.create_pending_state(
            provider_id="test",
            redirect_uri="/callback",
        )

        # Should exist
        self.assertIsNotNone(await short_store.get_pending_state(pending.state))

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        self.assertIsNone(await short_store.get_pending_state(pending.state))

    async def test_discovery_cache(self):
        """Test discovery document caching."""
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.adapters.oidc_state import OIDCDiscoveryDocument

        doc = OIDCDiscoveryDocument(
            issuer="https://example.com",
            authorization_endpoint="https://example.com/authorize",
            token_endpoint="https://example.com/token",
        )

        # Cache it
        await self.store.cache_discovery("https://example.com", doc)

        # Retrieve it
        cached = await self.store.get_discovery("https://example.com")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.issuer, "https://example.com")

    async def test_jwks_cache(self):
        """Test JWKS caching."""
        jwks = {"keys": [{"kid": "key1", "kty": "RSA", "n": "...", "e": "AQAB"}]}

        await self.store.cache_jwks("https://example.com/.well-known/jwks.json", jwks)

        cached = await self.store.get_jwks("https://example.com/.well-known/jwks.json")
        self.assertIsNotNone(cached)
        self.assertEqual(len(cached["keys"]), 1)

    async def test_clear_all(self):
        """Test clearing all OIDC state."""
        # Create some state
        await self.store.create_pending_state("p1", "/callback")
        await self.store.cache_jwks("https://example.com/jwks", {"keys": []})

        # Clear all
        count = await self.store.clear_all()
        self.assertGreater(count, 0)


class TestStateBackendFactory(unittest.IsolatedAsyncioTestCase):
    """Test the state backend factory function."""

    async def test_create_memory_backend(self):
        """Test creating memory backend."""
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend import create_state_backend
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.config import StateBackendConfig

        config = StateBackendConfig(backend="memory")
        backend = create_state_backend(config)

        self.assertTrue(await backend.ping())
        await backend.close()

    async def test_create_sqlite_backend(self):
        """Test creating SQLite backend."""
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend import create_state_backend
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.config import StateBackendConfig

        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            config = StateBackendConfig(
                backend="sqlite",
                url=f"sqlite:///{db_path}",
            )
            backend = create_state_backend(config)

            self.assertTrue(await backend.ping())
            await backend.close()

    async def test_unknown_backend_raises(self):
        """Test unknown backend raises ValueError."""
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend import create_state_backend
        from simpletuner.simpletuner_sdk.server.services.cloud.state_backend.config import StateBackendConfig

        config = StateBackendConfig(backend="unknown_backend")

        with self.assertRaises(ValueError) as ctx:
            create_state_backend(config)

        self.assertIn("Unknown backend", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
