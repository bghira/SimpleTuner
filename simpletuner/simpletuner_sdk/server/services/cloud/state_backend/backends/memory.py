"""In-Memory State Backend.

For testing and development only. Not suitable for production.
All state is lost on process restart.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from ..config import StateBackendConfig


@dataclass
class MemoryEntry:
    """Entry in memory store."""

    value: bytes
    entry_type: str = "bytes"
    expires_at: Optional[float] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class MemoryStateBackend:
    """In-memory state backend for testing.

    Features:
        - Thread-safe via asyncio.Lock
        - Supports all protocol operations
        - Automatic TTL expiration checks

    Warning:
        All data is lost on process restart.
        Not suitable for multi-worker deployments.
    """

    def __init__(self, config: Optional[StateBackendConfig] = None):
        """Initialize memory backend."""
        self._config = config or StateBackendConfig()
        self._store: Dict[str, MemoryEntry] = {}
        self._lock = asyncio.Lock()

    def _prefixed(self, key: str) -> str:
        """Add key prefix."""
        return self._config.key_prefix + key

    def _ttl_to_expires(self, ttl: Optional[int]) -> Optional[float]:
        """Convert TTL to expiration timestamp."""
        if ttl is None:
            return None
        return time.time() + ttl

    # --- Basic Key-Value Operations ---

    async def get(self, key: str) -> Optional[bytes]:
        """Get a value by key."""
        async with self._lock:
            prefixed_key = self._prefixed(key)
            entry = self._store.get(prefixed_key)
            if entry is None or entry.is_expired():
                if entry is not None:
                    del self._store[prefixed_key]
                return None
            return entry.value

    async def set(
        self,
        key: str,
        value: bytes,
        ttl: Optional[int] = None,
    ) -> None:
        """Set a value with optional TTL."""
        async with self._lock:
            prefixed_key = self._prefixed(key)
            self._store[prefixed_key] = MemoryEntry(
                value=value,
                entry_type="bytes",
                expires_at=self._ttl_to_expires(ttl),
            )

    async def delete(self, key: str) -> bool:
        """Delete a key."""
        async with self._lock:
            prefixed_key = self._prefixed(key)
            if prefixed_key in self._store:
                del self._store[prefixed_key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return await self.get(key) is not None

    # --- Atomic Counter Operations ---

    async def incr(self, key: str, amount: int = 1) -> int:
        """Atomically increment a counter."""
        async with self._lock:
            prefixed_key = self._prefixed(key)
            entry = self._store.get(prefixed_key)

            if entry is None or entry.is_expired():
                new_value = amount
                self._store[prefixed_key] = MemoryEntry(
                    value=str(new_value).encode(),
                    entry_type="counter",
                )
            else:
                current = int(entry.value.decode())
                new_value = current + amount
                entry.value = str(new_value).encode()
                entry.updated_at = time.time()

            return new_value

    async def get_counter(self, key: str) -> int:
        """Get counter value."""
        value = await self.get(key)
        if value is None:
            return 0
        try:
            return int(value.decode())
        except ValueError:
            return 0

    # --- Sliding Window Operations ---

    async def sliding_window_add(
        self,
        key: str,
        timestamp: float,
        window_seconds: int,
    ) -> int:
        """Add timestamp to sliding window."""
        async with self._lock:
            prefixed_key = self._prefixed(key)
            entry = self._store.get(prefixed_key)
            cutoff = timestamp - window_seconds

            if entry is None or entry.entry_type != "window":
                timestamps = []
            else:
                import json

                try:
                    timestamps = json.loads(entry.value.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    timestamps = []

            # Filter old and add new
            timestamps = [ts for ts in timestamps if ts > cutoff]
            timestamps.append(timestamp)

            import json

            self._store[prefixed_key] = MemoryEntry(
                value=json.dumps(timestamps).encode(),
                entry_type="window",
            )

            return len(timestamps)

    async def sliding_window_count(
        self,
        key: str,
        window_seconds: int,
    ) -> int:
        """Count timestamps in window."""
        async with self._lock:
            prefixed_key = self._prefixed(key)
            entry = self._store.get(prefixed_key)
            cutoff = time.time() - window_seconds

            if entry is None or entry.entry_type != "window":
                return 0

            import json

            try:
                timestamps = json.loads(entry.value.decode())
                return len([ts for ts in timestamps if ts > cutoff])
            except (json.JSONDecodeError, UnicodeDecodeError):
                return 0

    # --- Hash Operations ---

    async def hget(self, key: str, field: str) -> Optional[bytes]:
        """Get field from hash."""
        async with self._lock:
            prefixed_key = self._prefixed(key)
            entry = self._store.get(prefixed_key)

            if entry is None or entry.is_expired() or entry.entry_type != "hash":
                return None

            import json

            try:
                hash_data = json.loads(entry.value.decode())
                field_value = hash_data.get(field)
                if field_value is None:
                    return None
                return bytes.fromhex(field_value)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                return None

    async def hset(self, key: str, field: str, value: bytes) -> None:
        """Set field in hash."""
        await self.hset_with_ttl(key, field, value, None)

    async def hset_with_ttl(
        self,
        key: str,
        field: str,
        value: bytes,
        ttl: Optional[int] = None,
    ) -> None:
        """Set field in hash with TTL."""
        async with self._lock:
            import json

            prefixed_key = self._prefixed(key)
            entry = self._store.get(prefixed_key)

            if entry is None or entry.entry_type != "hash":
                hash_data = {}
            else:
                try:
                    hash_data = json.loads(entry.value.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    hash_data = {}

            hash_data[field] = value.hex()

            expires_at = self._ttl_to_expires(ttl)
            if entry and entry.expires_at and expires_at is None:
                expires_at = entry.expires_at

            self._store[prefixed_key] = MemoryEntry(
                value=json.dumps(hash_data).encode(),
                entry_type="hash",
                expires_at=expires_at,
            )

    async def hgetall(self, key: str) -> Dict[bytes, bytes]:
        """Get all fields from hash."""
        async with self._lock:
            prefixed_key = self._prefixed(key)
            entry = self._store.get(prefixed_key)

            if entry is None or entry.is_expired() or entry.entry_type != "hash":
                return {}

            import json

            try:
                hash_data = json.loads(entry.value.decode())
                return {k.encode(): bytes.fromhex(v) for k, v in hash_data.items()}
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                return {}

    async def hdel(self, key: str, field: str) -> bool:
        """Delete field from hash."""
        async with self._lock:
            import json

            prefixed_key = self._prefixed(key)
            entry = self._store.get(prefixed_key)

            if entry is None or entry.entry_type != "hash":
                return False

            try:
                hash_data = json.loads(entry.value.decode())
                if field not in hash_data:
                    return False
                del hash_data[field]

                if hash_data:
                    entry.value = json.dumps(hash_data).encode()
                    entry.updated_at = time.time()
                else:
                    del self._store[prefixed_key]

                return True
            except (json.JSONDecodeError, UnicodeDecodeError):
                return False

    # --- Set Operations ---

    async def sadd(self, key: str, *members: bytes) -> int:
        """Add members to set."""
        async with self._lock:
            import json

            prefixed_key = self._prefixed(key)
            entry = self._store.get(prefixed_key)

            if entry is None or entry.entry_type != "set":
                current_set: set[str] = set()
            else:
                try:
                    current_set = set(json.loads(entry.value.decode()))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    current_set = set()

            original_size = len(current_set)
            # Convert bytes to hex strings for storage
            for member in members:
                current_set.add(member.hex())
            added = len(current_set) - original_size

            self._store[prefixed_key] = MemoryEntry(
                value=json.dumps(list(current_set)).encode(),
                entry_type="set",
            )

            return added

    async def srem(self, key: str, *members: bytes) -> int:
        """Remove members from set."""
        async with self._lock:
            import json

            prefixed_key = self._prefixed(key)
            entry = self._store.get(prefixed_key)

            if entry is None or entry.entry_type != "set":
                return 0

            try:
                current_set = set(json.loads(entry.value.decode()))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return 0

            original_size = len(current_set)
            for member in members:
                current_set.discard(member.hex())
            removed = original_size - len(current_set)

            if current_set:
                entry.value = json.dumps(list(current_set)).encode()
                entry.updated_at = time.time()
            else:
                del self._store[prefixed_key]

            return removed

    async def smembers(self, key: str) -> set[bytes]:
        """Get all members of set."""
        async with self._lock:
            import json

            prefixed_key = self._prefixed(key)
            entry = self._store.get(prefixed_key)

            if entry is None or entry.entry_type != "set":
                return set()

            try:
                hex_set = json.loads(entry.value.decode())
                return {bytes.fromhex(m) for m in hex_set}
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
                return set()

    async def sismember(self, key: str, member: bytes) -> bool:
        """Check if member in set."""
        members = await self.smembers(key)
        return member in members

    # --- Batch Operations ---

    async def mget(self, *keys: str) -> List[Optional[bytes]]:
        """Get multiple keys."""
        return [await self.get(key) for key in keys]

    async def mset(
        self,
        mapping: Dict[str, bytes],
        ttl: Optional[int] = None,
    ) -> None:
        """Set multiple keys."""
        for key, value in mapping.items():
            await self.set(key, value, ttl)

    async def delete_prefix(self, prefix: str) -> int:
        """Delete keys with prefix."""
        async with self._lock:
            full_prefix = self._prefixed(prefix)
            keys_to_delete = [k for k in self._store if k.startswith(full_prefix)]
            for key in keys_to_delete:
                del self._store[key]
            return len(keys_to_delete)

    # --- Connection Management ---

    async def ping(self) -> bool:
        """Always healthy."""
        return True

    async def close(self) -> None:
        """Clear store."""
        async with self._lock:
            self._store.clear()

    async def flush_expired(self) -> int:
        """Remove expired entries."""
        async with self._lock:
            expired_keys = [k for k, v in self._store.items() if v.is_expired()]
            for key in expired_keys:
                del self._store[key]
            return len(expired_keys)

    # --- Transaction Support ---

    async def execute_atomic(
        self,
        operations: List[tuple[str, tuple[Any, ...]]],
    ) -> List[Any]:
        """Execute operations atomically."""
        results = []
        async with self._lock:
            for method_name, args in operations:
                method = getattr(self, method_name)
                result = await method(*args)
                results.append(result)
        return results
