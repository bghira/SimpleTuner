"""Wrapper for field registry to handle import issues gracefully."""

import logging
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LazyFieldRegistry:
    """Lazy-loading wrapper for field registry to handle import issues."""

    def __init__(self):
        self._registry = None
        self._import_error = None
        self._fields_cache = {}
        self._sections_cache = {}
        self._field_cache = {}  # Cache individual fields by name
        self._dependent_fields_cache = {}
        self._cache_ttl = 300  # 5 minutes cache TTL
        self._cache_timestamps = {}

    def _get_registry(self):
        """Lazy load the field registry."""
        if self._registry is None and self._import_error is None:
            try:
                from simpletuner.simpletuner_sdk.server.services.field_registry import field_registry

                self._registry = field_registry
                logger.info(f"Successfully loaded field registry with {len(self._registry._fields)} fields")
            except ImportError as e:
                self._import_error = str(e)
                logger.error(f"Failed to import field registry: {e}")
            except Exception as e:
                self._import_error = str(e)
                logger.error(f"Unexpected error importing field registry: {e}", exc_info=True)

        return self._registry

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid based on TTL."""
        if cache_key not in self._cache_timestamps:
            return False
        return (time.time() - self._cache_timestamps[cache_key]) < self._cache_ttl

    def get_fields_for_tab(self, tab_name: str) -> List[Any]:
        """Get fields for a specific tab with caching and TTL."""
        cache_key = f"tab_{tab_name}"

        # Check if we have valid cached data
        if cache_key in self._fields_cache and self._is_cache_valid(cache_key):
            logger.debug(f"Returning cached fields for tab '{tab_name}'")
            return self._fields_cache[cache_key]

        registry = self._get_registry()
        if registry is None:
            logger.warning(f"Field registry not available, returning empty list for tab '{tab_name}'")
            logger.warning(f"Import error: {self._import_error}")
            return []

        try:
            fields = registry.get_fields_for_tab(tab_name)
            self._fields_cache[cache_key] = fields
            self._cache_timestamps[cache_key] = time.time()
            logger.debug(f"Cached {len(fields)} fields for tab '{tab_name}'")
            return fields
        except Exception as e:
            logger.error(f"Error getting fields for tab '{tab_name}': {e}", exc_info=True)
            return []

    def get_field(self, field_name: str) -> Optional[Any]:
        """Get a specific field by name with caching."""
        cache_key = f"field_{field_name}"

        # Check if we have valid cached data
        if cache_key in self._field_cache and self._is_cache_valid(cache_key):
            logger.debug(f"Returning cached field '{field_name}'")
            return self._field_cache[cache_key]

        registry = self._get_registry()
        if registry is None:
            return None

        try:
            field = registry.get_field(field_name)
            if field is not None:
                self._field_cache[cache_key] = field
                self._cache_timestamps[cache_key] = time.time()
            return field
        except Exception as e:
            logger.error(f"Error getting field '{field_name}': {e}")
            return None

    def get_sections_for_tab(self, tab_name: str) -> List[Dict[str, Any]]:
        """Get sections for a specific tab with caching."""
        cache_key = f"sections_{tab_name}"

        # Check if we have valid cached data
        if cache_key in self._sections_cache and self._is_cache_valid(cache_key):
            logger.debug(f"Returning cached sections for tab '{tab_name}'")
            return self._sections_cache[cache_key]

        registry = self._get_registry()
        if registry is None:
            return []

        try:
            sections = registry.get_sections_for_tab(tab_name)
            self._sections_cache[cache_key] = sections
            self._cache_timestamps[cache_key] = time.time()
            return sections
        except Exception as e:
            logger.error(f"Error getting sections for tab '{tab_name}': {e}")
            return []

    @property
    def _fields(self) -> Dict[str, Any]:
        """Expose _fields for debug endpoints."""
        registry = self._get_registry()
        if registry is None:
            return {}
        return getattr(registry, "_fields", {})

    @property
    def _dependencies_map(self) -> Dict[str, List[str]]:
        """Expose _dependencies_map for debug endpoints."""
        registry = self._get_registry()
        if registry is None:
            return {}
        return getattr(registry, "_dependencies_map", {})

    def get_all_fields(self) -> List[Any]:
        """Get all fields from the registry."""
        registry = self._get_registry()
        if registry is None:
            return []

        try:
            # Try to get all fields from the registry
            if hasattr(registry, "get_all_fields"):
                return registry.get_all_fields()
            elif hasattr(registry, "_fields"):
                # Return all field values if _fields is a dict
                return list(registry._fields.values())
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting all fields: {e}", exc_info=True)
            return []

    def get_dependent_fields(self, field_name: str) -> List[str]:
        """Get fields that depend on the given field with caching."""
        cache_key = f"deps_{field_name}"

        # Check if we have valid cached data
        if cache_key in self._dependent_fields_cache and self._is_cache_valid(cache_key):
            logger.debug(f"Returning cached dependent fields for '{field_name}'")
            return self._dependent_fields_cache[cache_key]

        registry = self._get_registry()
        if registry is None:
            return []

        try:
            # Try to get dependent fields from the registry
            if hasattr(registry, "get_dependent_fields"):
                deps = registry.get_dependent_fields(field_name)
                self._dependent_fields_cache[cache_key] = deps
                self._cache_timestamps[cache_key] = time.time()
                return deps
            else:
                return []
        except Exception as e:
            logger.error(f"Error getting dependent fields for '{field_name}': {e}", exc_info=True)
            return []

    def clear_cache(self) -> None:
        """Clear all caches."""
        self._fields_cache.clear()
        self._sections_cache.clear()
        self._field_cache.clear()
        self._dependent_fields_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Field registry caches cleared")

    def set_cache_ttl(self, ttl_seconds: int) -> None:
        """Set cache TTL in seconds."""
        self._cache_ttl = ttl_seconds
        logger.info(f"Cache TTL set to {ttl_seconds} seconds")

    def export_field_metadata(self) -> Dict[str, Any]:
        """Export all field metadata for frontend consumption."""
        registry = self._get_registry()
        if registry is None:
            logger.warning("Field registry not available, returning empty metadata")
            return {"fields": {}, "dependencies_map": {}, "tabs": {}}

        try:
            return registry.export_field_metadata()
        except Exception as e:
            logger.error(f"Error exporting field metadata: {e}", exc_info=True)
            return {"fields": {}, "dependencies_map": {}, "tabs": {}}

    def get_webui_onboarding_fields(self) -> List[Any]:
        """Return fields that should be treated as WebUI onboarding state."""

        registry = self._get_registry()
        if registry is None:
            return []

        try:
            if hasattr(registry, "get_webui_onboarding_fields"):
                return registry.get_webui_onboarding_fields()
            return []
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.error("Error retrieving WebUI onboarding fields: %s", exc, exc_info=True)
            return []

    def validate_field_value(self, field_name: str, value: Any, context: Optional[Dict[str, Any]] = None) -> List[str]:
        """Validate a field value against its rules."""
        registry = self._get_registry()
        if registry is None:
            logger.warning(f"Field registry not available, cannot validate field '{field_name}'")
            return []

        try:
            return registry.validate_field_value(field_name, value, context)
        except Exception as e:
            logger.error(f"Error validating field '{field_name}': {e}", exc_info=True)
            return []


# Create singleton instance
lazy_field_registry = LazyFieldRegistry()
