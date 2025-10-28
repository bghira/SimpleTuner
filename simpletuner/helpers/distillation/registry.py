from __future__ import annotations

from typing import Any, Dict, Optional, Type

from simpletuner.helpers.distillation.common import DistillationBase
from simpletuner.helpers.distillation.requirements import (
    EMPTY_PROFILE,
    DistillerRequirementProfile,
    parse_distiller_requirement_profile,
)


class DistillationRegistry:
    """Registry for tracking available distillation methods and their metadata."""

    _registry: Dict[str, Type[DistillationBase]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    _requirement_profiles: Dict[str, DistillerRequirementProfile] = {}

    @classmethod
    def register(cls, name: str, distiller_cls: Type[DistillationBase], **metadata: Any) -> None:
        """Register a distiller class with optional metadata."""
        key = name.lower()
        profile = parse_distiller_requirement_profile(metadata)
        cls._registry[key] = distiller_cls
        cls._metadata[key] = dict(metadata)
        cls._requirement_profiles[key] = profile

    @classmethod
    def get(cls, name: str) -> Optional[Type[DistillationBase]]:
        """Return the registered distiller class for the provided name."""
        return cls._registry.get(name.lower())

    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """Return metadata associated with the registered distiller."""
        return cls._metadata.get(name.lower(), {}).copy()

    @classmethod
    def get_metadata_with_requirements(cls, name: str) -> Dict[str, Any]:
        """Return metadata and the parsed requirement profile."""
        metadata = cls.get_metadata(name)
        metadata["requirement_profile"] = cls.get_requirement_profile(name)
        return metadata

    @classmethod
    def get_requirement_profile(cls, name: str) -> DistillerRequirementProfile:
        """Return the parsed requirement profile for the given distiller."""
        return cls._requirement_profiles.get(name.lower(), EMPTY_PROFILE)

    @classmethod
    def items(cls) -> Dict[str, Dict[str, Any]]:
        """Return a mapping of registered distillers to their metadata."""
        return {
            name: {
                "class": distiller,
                "metadata": cls._metadata.get(name, {}).copy(),
                "requirement_profile": cls._requirement_profiles.get(name, EMPTY_PROFILE),
            }
            for name, distiller in cls._registry.items()
        }
