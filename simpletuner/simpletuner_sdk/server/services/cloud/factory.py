"""Provider factory for cloud training services."""

from __future__ import annotations

from typing import Dict, Type

from .base import CloudTrainerService
from .replicate_client import ReplicateCogClient


class ProviderFactory:
    """Factory for creating cloud provider clients."""

    _providers: Dict[str, Type[CloudTrainerService]] = {}
    _instances: Dict[str, CloudTrainerService] = {}

    @classmethod
    def register(cls, name: str, provider_cls: Type[CloudTrainerService]) -> None:
        """Register a new provider class."""
        cls._providers[name] = provider_cls

    @classmethod
    def get_provider(cls, name: str) -> CloudTrainerService:
        """Get or create a provider instance."""
        if name not in cls._providers:
            raise ValueError(f"Unknown provider: {name}")

        if name not in cls._instances:
            # Instantiate the provider
            cls._instances[name] = cls._providers[name]()

        return cls._instances[name]


# Register default providers
ProviderFactory.register("replicate", ReplicateCogClient)
