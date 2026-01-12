"""State Backend Configuration.

Handles configuration loading from environment variables and config files.
Environment variables take precedence over config file settings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


def _get_env_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    value = os.environ.get(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    value = os.environ.get(key)
    if value is None:
        return default
    return value.lower() in ("true", "1", "yes", "on")


@dataclass
class StateBackendConfig:
    """Configuration for state backend.

    Environment variables:
        STATE_BACKEND: Backend type (sqlite, postgresql, mysql, redis, memory)
        STATE_BACKEND_URL: Connection URL (format varies by backend)
        STATE_BACKEND_POOL_SIZE: Connection pool size (default: 10)
        STATE_BACKEND_TIMEOUT: Connection timeout in seconds (default: 30)
        STATE_BACKEND_KEY_PREFIX: Prefix for all keys (default: "st:")

    URL formats:
        - SQLite: sqlite:///path/to/state.db or just a path
        - PostgreSQL: postgresql://user:pass@host:5432/dbname
        - MySQL: mysql://user:pass@host:3306/dbname
        - Redis: redis://[:password@]host:6379/db

    Example config file section (TOML):
        [state_backend]
        backend = "redis"
        url = "redis://localhost:6379/0"
        pool_size = 20
        key_prefix = "myapp:"
    """

    # Core settings
    backend: str = field(default_factory=lambda: os.environ.get("STATE_BACKEND", "sqlite"))
    url: Optional[str] = field(default_factory=lambda: os.environ.get("STATE_BACKEND_URL"))
    pool_size: int = field(default_factory=lambda: _get_env_int("STATE_BACKEND_POOL_SIZE", 10))
    timeout: float = field(default_factory=lambda: _get_env_float("STATE_BACKEND_TIMEOUT", 30.0))
    key_prefix: str = field(default_factory=lambda: os.environ.get("STATE_BACKEND_KEY_PREFIX", "st:"))

    # SQLite-specific options
    sqlite_path: Optional[str] = None
    sqlite_wal_mode: bool = True
    sqlite_busy_timeout: int = 30000  # milliseconds

    # PostgreSQL/MySQL-specific options
    min_pool_size: int = 2
    max_pool_size: int = 10
    pool_recycle: int = 3600  # seconds

    # Redis-specific options
    redis_db: int = 0
    redis_ssl: bool = False
    redis_cluster_mode: bool = False
    redis_socket_timeout: float = 5.0
    redis_socket_connect_timeout: float = 5.0

    # Cleanup settings
    cleanup_interval: int = 60  # seconds between expired entry cleanup
    cleanup_batch_size: int = 1000  # max entries to clean per run

    # Backend-specific extra options
    options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "StateBackendConfig":
        """Load configuration from environment variables only."""
        return cls(
            backend=os.environ.get("STATE_BACKEND", "sqlite"),
            url=os.environ.get("STATE_BACKEND_URL"),
            pool_size=_get_env_int("STATE_BACKEND_POOL_SIZE", 10),
            timeout=_get_env_float("STATE_BACKEND_TIMEOUT", 30.0),
            key_prefix=os.environ.get("STATE_BACKEND_KEY_PREFIX", "st:"),
            redis_db=_get_env_int("STATE_BACKEND_REDIS_DB", 0),
            redis_ssl=_get_env_bool("STATE_BACKEND_REDIS_SSL", False),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateBackendConfig":
        """Load from dictionary (config file section).

        Environment variables still take precedence.

        Args:
            data: Dictionary with configuration values.

        Returns:
            StateBackendConfig instance.
        """
        # Start with file config
        config_kwargs: Dict[str, Any] = {}

        # Map known keys
        key_mapping = {
            "backend": "backend",
            "url": "url",
            "pool_size": "pool_size",
            "timeout": "timeout",
            "key_prefix": "key_prefix",
            "sqlite_path": "sqlite_path",
            "sqlite_wal_mode": "sqlite_wal_mode",
            "sqlite_busy_timeout": "sqlite_busy_timeout",
            "min_pool_size": "min_pool_size",
            "max_pool_size": "max_pool_size",
            "pool_recycle": "pool_recycle",
            "redis_db": "redis_db",
            "redis_ssl": "redis_ssl",
            "redis_cluster_mode": "redis_cluster_mode",
            "cleanup_interval": "cleanup_interval",
            "cleanup_batch_size": "cleanup_batch_size",
        }

        for file_key, attr_name in key_mapping.items():
            if file_key in data:
                config_kwargs[attr_name] = data[file_key]

        # Collect unknown keys into options
        options = {k: v for k, v in data.items() if k not in key_mapping}
        if options:
            config_kwargs["options"] = options

        # Create config with file values
        config = cls(**config_kwargs)

        # Override with environment variables
        if os.environ.get("STATE_BACKEND"):
            config.backend = os.environ["STATE_BACKEND"]
        if os.environ.get("STATE_BACKEND_URL"):
            config.url = os.environ["STATE_BACKEND_URL"]
        if os.environ.get("STATE_BACKEND_POOL_SIZE"):
            config.pool_size = _get_env_int("STATE_BACKEND_POOL_SIZE", config.pool_size)
        if os.environ.get("STATE_BACKEND_TIMEOUT"):
            config.timeout = _get_env_float("STATE_BACKEND_TIMEOUT", config.timeout)
        if os.environ.get("STATE_BACKEND_KEY_PREFIX"):
            config.key_prefix = os.environ["STATE_BACKEND_KEY_PREFIX"]

        return config

    def get_sqlite_path(self, default_dir: Optional[Path] = None) -> Path:
        """Get the SQLite database path.

        Args:
            default_dir: Default directory if no path specified.

        Returns:
            Path to the SQLite database file.
        """
        if self.sqlite_path:
            return Path(self.sqlite_path)

        if self.url:
            # Parse sqlite:///path or just path
            url = self.url
            if url.startswith("sqlite:///"):
                return Path(url[10:])
            elif url.startswith("sqlite://"):
                return Path(url[9:])
            elif not url.startswith(("postgresql", "mysql", "redis")):
                return Path(url)

        # Use default directory
        if default_dir is None:
            default_dir = Path.home() / ".simpletuner" / "config" / "cloud"

        default_dir.mkdir(parents=True, exist_ok=True)
        return default_dir / "state.db"

    def get_connection_url(self) -> str:
        """Get the connection URL for the configured backend.

        Returns:
            Connection URL string.

        Raises:
            ValueError: If URL is required but not provided.
        """
        if self.backend == "sqlite":
            path = self.get_sqlite_path()
            return f"sqlite:///{path}"

        if self.backend == "memory":
            return "memory://"

        if not self.url:
            raise ValueError(f"STATE_BACKEND_URL is required for {self.backend} backend")

        return self.url

    def validate(self) -> None:
        """Validate configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        valid_backends = {"sqlite", "postgresql", "postgres", "mysql", "mariadb", "redis", "memory"}
        if self.backend.lower() not in valid_backends:
            raise ValueError(f"Unknown backend: {self.backend}. " f"Valid options: {', '.join(sorted(valid_backends))}")

        if self.backend.lower() in ("postgresql", "postgres", "mysql", "mariadb", "redis"):
            if not self.url:
                raise ValueError(f"STATE_BACKEND_URL is required for {self.backend} backend")

        if self.pool_size < 1:
            raise ValueError("pool_size must be at least 1")

        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
