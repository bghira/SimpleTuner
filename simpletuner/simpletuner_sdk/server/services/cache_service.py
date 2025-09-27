"""Caching service for SimpleTuner WebUI.

This module provides caching functionality for:
- Configuration files
- Template renders
- API responses
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class LRUCache:
    """Simple LRU (Least Recently Used) cache implementation."""

    def __init__(self, max_size: int = 100):
        """Initialize LRU cache.

        Args:
            max_size: Maximum number of items to store in cache
        """
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        if key not in self.cache:
            return None
        # Move to end to mark as recently used
        self.cache.move_to_end(key)
        return self.cache[key]

    def set(self, key: str, value: Any) -> None:
        """Set item in cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Move to end to mark as recently used
            self.cache.move_to_end(key)
        self.cache[key] = value
        # Remove oldest items if over limit
        while len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

    def clear(self) -> None:
        """Clear all items from cache."""
        self.cache.clear()


class ConfigCache:
    """Cache for configuration files with modification time checking."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize config cache.

        Args:
            cache_dir: Directory to store cache files (optional)
        """
        self.memory_cache = {}
        self.file_mtimes = {}
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_mtime(self, file_path: Path) -> float:
        """Get file modification time."""
        try:
            return file_path.stat().st_mtime
        except Exception:
            return 0

    def _is_cache_valid(self, file_path: Path, cached_mtime: float) -> bool:
        """Check if cached data is still valid based on file modification time."""
        current_mtime = self._get_file_mtime(file_path)
        return current_mtime == cached_mtime

    def get(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Get config from cache if still valid.

        Args:
            file_path: Path to config file

        Returns:
            Cached config data or None if cache miss/invalid
        """
        cache_key = str(file_path)

        # Check memory cache first
        if cache_key in self.memory_cache:
            cached_mtime = self.file_mtimes.get(cache_key, 0)
            if self._is_cache_valid(file_path, cached_mtime):
                logger.debug(f"Config cache hit for {file_path}")
                return self.memory_cache[cache_key]

        # Check disk cache if configured
        if self.cache_dir:
            cache_file = self._get_cache_file_path(file_path)
            if cache_file.exists():
                try:
                    with open(cache_file, "r") as f:
                        cache_data = json.load(f)
                    if self._is_cache_valid(file_path, cache_data.get("mtime", 0)):
                        logger.debug(f"Disk cache hit for {file_path}")
                        # Load into memory cache
                        self.memory_cache[cache_key] = cache_data["data"]
                        self.file_mtimes[cache_key] = cache_data["mtime"]
                        return cache_data["data"]
                except Exception as e:
                    logger.error(f"Error reading cache file: {e}")

        logger.debug(f"Config cache miss for {file_path}")
        return None

    def set(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Store config in cache.

        Args:
            file_path: Path to config file
            data: Config data to cache
        """
        cache_key = str(file_path)
        mtime = self._get_file_mtime(file_path)

        # Store in memory cache
        self.memory_cache[cache_key] = data
        self.file_mtimes[cache_key] = mtime

        # Store in disk cache if configured
        if self.cache_dir:
            cache_file = self._get_cache_file_path(file_path)
            try:
                cache_data = {"mtime": mtime, "data": data}
                with open(cache_file, "w") as f:
                    json.dump(cache_data, f)
            except Exception as e:
                logger.error(f"Error writing cache file: {e}")

    def _get_cache_file_path(self, file_path: Path) -> Path:
        """Get cache file path for a given config file."""
        # Create hash of file path for cache filename
        path_hash = hashlib.md5(str(file_path).encode()).hexdigest()
        return self.cache_dir / f"{path_hash}.json"

    def clear(self) -> None:
        """Clear all cached configs."""
        self.memory_cache.clear()
        self.file_mtimes.clear()
        if self.cache_dir:
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    cache_file.unlink()
                except Exception:
                    pass


def cache_response(ttl_seconds: int = 60):
    """Decorator to cache function responses.

    Args:
        ttl_seconds: Time to live for cached response

    Returns:
        Decorated function
    """
    cache = {}
    timestamps = {}

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create cache key from function args
            cache_key = f"{func.__name__}:{repr(args)}:{repr(kwargs)}"

            # Check if we have valid cached result
            if cache_key in cache:
                if time.time() - timestamps[cache_key] < ttl_seconds:
                    logger.debug(f"Cache hit for {func.__name__}")
                    return cache[cache_key]

            # Call function and cache result
            result = func(*args, **kwargs)
            cache[cache_key] = result
            timestamps[cache_key] = time.time()

            return result

        # Add cache control methods
        wrapper.clear_cache = lambda: cache.clear()
        wrapper.cache_info = lambda: {"size": len(cache), "keys": list(cache.keys())}

        return wrapper

    return decorator


# Global instances
_lru_cache = LRUCache(max_size=500)
_config_cache = ConfigCache()


def get_lru_cache() -> LRUCache:
    """Get global LRU cache instance."""
    return _lru_cache


def get_config_cache() -> ConfigCache:
    """Get global config cache instance."""
    return _config_cache
