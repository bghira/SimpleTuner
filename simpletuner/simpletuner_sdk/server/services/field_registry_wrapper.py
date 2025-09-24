"""Wrapper for field registry to handle import issues gracefully."""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class LazyFieldRegistry:
    """Lazy-loading wrapper for field registry to handle import issues."""

    def __init__(self):
        self._registry = None
        self._import_error = None
        self._fields_cache = {}

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

    def get_fields_for_tab(self, tab_name: str) -> List[Any]:
        """Get fields for a specific tab with caching."""
        cache_key = f"tab_{tab_name}"

        if cache_key in self._fields_cache:
            return self._fields_cache[cache_key]

        registry = self._get_registry()
        if registry is None:
            logger.warning(f"Field registry not available, returning empty list for tab '{tab_name}'")
            logger.warning(f"Import error: {self._import_error}")
            return []

        try:
            fields = registry.get_fields_for_tab(tab_name)
            self._fields_cache[cache_key] = fields
            return fields
        except Exception as e:
            logger.error(f"Error getting fields for tab '{tab_name}': {e}", exc_info=True)
            return []

    def get_field(self, field_name: str) -> Optional[Any]:
        """Get a specific field by name."""
        registry = self._get_registry()
        if registry is None:
            return None

        try:
            return registry.get_field(field_name)
        except Exception as e:
            logger.error(f"Error getting field '{field_name}': {e}")
            return None

    def get_sections_for_tab(self, tab_name: str) -> List[Dict[str, Any]]:
        """Get sections for a specific tab."""
        registry = self._get_registry()
        if registry is None:
            return []

        try:
            return registry.get_sections_for_tab(tab_name)
        except Exception as e:
            logger.error(f"Error getting sections for tab '{tab_name}': {e}")
            return []

    @property
    def _fields(self) -> Dict[str, Any]:
        """Expose _fields for debug endpoints."""
        registry = self._get_registry()
        if registry is None:
            return {}
        return getattr(registry, '_fields', {})

    @property
    def _dependencies_map(self) -> Dict[str, List[str]]:
        """Expose _dependencies_map for debug endpoints."""
        registry = self._get_registry()
        if registry is None:
            return {}
        return getattr(registry, '_dependencies_map', {})


# Create singleton instance
lazy_field_registry = LazyFieldRegistry()