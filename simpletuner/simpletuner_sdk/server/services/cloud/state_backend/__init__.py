"""Pluggable State Backend System.

Provides a unified interface for state storage across different backends:
    - SQLite (default): Single-node deployments, uses aiosqlite
    - PostgreSQL: Multi-node, requires asyncpg
    - MySQL: Multi-node, requires aiomysql
    - Redis: Distributed cache, requires redis
    - Memory: Testing only

Usage:
    # Default (SQLite)
    backend = await get_state_backend()

    # With configuration
    from state_backend.config import StateBackendConfig
    config = StateBackendConfig(backend="redis", url="redis://localhost:6379/0")
    backend = create_state_backend(config)

    # Set value with TTL
    await backend.set("key", b"value", ttl=3600)

    # Get value
    value = await backend.get("key")

    # Rate limiting
    count = await backend.sliding_window_add("ratelimit:user:123", time.time(), 60)

Environment Variables:
    STATE_BACKEND: Backend type (sqlite, postgresql, mysql, redis, memory)
    STATE_BACKEND_URL: Connection URL
    STATE_BACKEND_POOL_SIZE: Connection pool size (default: 10)
    STATE_BACKEND_TIMEOUT: Connection timeout in seconds (default: 30)
    STATE_BACKEND_KEY_PREFIX: Key prefix for namespacing (default: "st:")
"""

from __future__ import annotations

import asyncio
from typing import Optional

from .config import StateBackendConfig
from .protocols import StateBackendProtocol

__all__ = [
    "StateBackendProtocol",
    "StateBackendConfig",
    "create_state_backend",
    "get_state_backend",
    "set_state_backend",
    "close_state_backend",
]

# Global singleton
_state_backend: Optional[StateBackendProtocol] = None
_init_lock = asyncio.Lock()


def create_state_backend(
    config: Optional[StateBackendConfig] = None,
) -> StateBackendProtocol:
    """Create a state backend instance based on configuration.

    Args:
        config: Backend configuration. If None, loads from environment.

    Returns:
        StateBackendProtocol implementation.

    Raises:
        ValueError: Unknown backend type.
        ImportError: Required dependency not installed.

    Example:
        # Use default SQLite backend
        backend = create_state_backend()

        # Use Redis
        config = StateBackendConfig(backend="redis", url="redis://localhost:6379")
        backend = create_state_backend(config)
    """
    if config is None:
        config = StateBackendConfig.from_env()

    # Validate configuration
    config.validate()

    backend_type = config.backend.lower()

    # Backend mapping: type -> (module, class_name)
    backend_mapping = {
        "sqlite": (".backends.sqlite", "SQLiteStateBackend"),
        "memory": (".backends.memory", "MemoryStateBackend"),
        "postgresql": (".backends.postgresql", "PostgreSQLStateBackend"),
        "postgres": (".backends.postgresql", "PostgreSQLStateBackend"),
        "mysql": (".backends.mysql", "MySQLStateBackend"),
        "mariadb": (".backends.mysql", "MySQLStateBackend"),
        "redis": (".backends.redis", "RedisStateBackend"),
    }

    if backend_type not in backend_mapping:
        valid_types = sorted(set(backend_mapping.keys()))
        raise ValueError(f"Unknown state backend: {backend_type}. " f"Valid options: {', '.join(valid_types)}")

    module_path, class_name = backend_mapping[backend_type]

    # Install hints for optional dependencies
    install_hints = {
        "postgresql": "pip install 'simpletuner[state-postgresql]' or pip install asyncpg",
        "postgres": "pip install 'simpletuner[state-postgresql]' or pip install asyncpg",
        "mysql": "pip install 'simpletuner[state-mysql]' or pip install aiomysql",
        "mariadb": "pip install 'simpletuner[state-mysql]' or pip install aiomysql",
        "redis": "pip install 'simpletuner[state-redis]' or pip install redis",
    }

    try:
        # Use importlib for cleaner dynamic imports
        import importlib

        module = importlib.import_module(module_path, package=__package__)
        backend_class = getattr(module, class_name)
    except ImportError as exc:
        hint = install_hints.get(backend_type, "")
        raise ImportError(f"Cannot load {backend_type} backend: {exc}. " f"{hint}" if hint else str(exc)) from exc
    except AttributeError as exc:
        raise ImportError(f"Backend class {class_name} not found in {module_path}: {exc}") from exc

    return backend_class(config)


async def get_state_backend() -> StateBackendProtocol:
    """Get or create the global state backend singleton.

    Thread-safe initialization with async lock.

    Returns:
        StateBackendProtocol implementation.

    Usage with FastAPI:
        from fastapi import Depends

        async def get_backend():
            return await get_state_backend()

        @router.get("/test")
        async def test(backend: StateBackendProtocol = Depends(get_backend)):
            return await backend.ping()
    """
    global _state_backend

    if _state_backend is not None:
        return _state_backend

    async with _init_lock:
        # Double-check after acquiring lock
        if _state_backend is not None:
            return _state_backend

        _state_backend = create_state_backend()

        # Verify connection
        if not await _state_backend.ping():
            raise RuntimeError("Failed to connect to state backend")

        return _state_backend


def set_state_backend(backend: StateBackendProtocol) -> None:
    """Override the global state backend.

    Useful for testing with mock backends.

    Args:
        backend: Backend instance to use.

    Example:
        from state_backend.backends.memory import MemoryStateBackend

        # In test setup
        set_state_backend(MemoryStateBackend())

        # Run tests...

        # In teardown
        await close_state_backend()
    """
    global _state_backend
    _state_backend = backend


async def close_state_backend() -> None:
    """Close the global state backend and release resources.

    Should be called on application shutdown.

    Example:
        from fastapi import FastAPI

        app = FastAPI()

        @app.on_event("shutdown")
        async def shutdown():
            await close_state_backend()
    """
    global _state_backend

    if _state_backend is not None:
        await _state_backend.close()
        _state_backend = None
