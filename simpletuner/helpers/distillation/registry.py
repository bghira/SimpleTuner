from __future__ import annotations

from typing import Any, Dict, Optional, Type

from simpletuner.helpers.distillation.common import DistillationBase


class DistillationRegistry:
    """Registry for tracking available distillation methods and their metadata."""

    _registry: Dict[str, Type[DistillationBase]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, distiller_cls: Type[DistillationBase], **metadata: Any) -> None:
        """Register a distiller class with optional metadata."""
        key = name.lower()
        cls._registry[key] = distiller_cls
        cls._metadata[key] = dict(metadata)

    @classmethod
    def get(cls, name: str) -> Optional[Type[DistillationBase]]:
        """Return the registered distiller class for the provided name."""
        return cls._registry.get(name.lower())

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """Return metadata associated with the registered distiller."""
        return cls._metadata.get(name.lower(), {}).copy()

    @classmethod
    def items(cls) -> Dict[str, Dict[str, Any]]:
        """Return a mapping of registered distillers to their metadata."""
        return {name: {"class": distiller, "metadata": cls._metadata.get(name, {}).copy()} for name, distiller in cls._registry.items()}
