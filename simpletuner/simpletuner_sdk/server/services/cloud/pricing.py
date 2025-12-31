"""GPU pricing abstraction for cloud providers.

This module provides a unified interface for GPU pricing across
different cloud providers, with configurable overrides.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class HardwareOption:
    """Represents a GPU hardware option with pricing."""

    id: str
    name: str
    cost_per_second: float
    memory_gb: Optional[int] = None
    gpu_count: int = 1
    available: bool = True

    @property
    def cost_per_hour(self) -> float:
        """Get cost per hour in USD."""
        return self.cost_per_second * 3600

    @property
    def cost_per_minute(self) -> float:
        """Get cost per minute in USD."""
        return self.cost_per_second * 60

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "cost_per_second": self.cost_per_second,
            "cost_per_hour": self.cost_per_hour,
            "memory_gb": self.memory_gb,
            "gpu_count": self.gpu_count,
            "available": self.available,
        }


class GPUPricingProvider(ABC):
    """Abstract interface for GPU pricing.

    Cloud provider clients should implement this interface to
    provide consistent pricing information across providers.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        ...

    @abstractmethod
    def get_hardware_options(self) -> List[HardwareOption]:
        """Get all available hardware options with pricing.

        Returns:
            List of HardwareOption objects.
        """
        ...

    @abstractmethod
    def get_default_hardware(self) -> HardwareOption:
        """Get the default hardware option.

        Returns:
            The default HardwareOption.
        """
        ...

    def get_hardware_by_id(self, hardware_id: str) -> Optional[HardwareOption]:
        """Get a specific hardware option by ID.

        Args:
            hardware_id: The hardware identifier.

        Returns:
            HardwareOption if found, None otherwise.
        """
        for hw in self.get_hardware_options():
            if hw.id == hardware_id:
                return hw
        return None

    def calculate_cost(self, hardware_id: str, duration_seconds: float) -> Optional[float]:
        """Calculate cost for a job.

        Args:
            hardware_id: The hardware identifier used.
            duration_seconds: Job duration in seconds.

        Returns:
            Cost in USD, or None if hardware not found.
        """
        hw = self.get_hardware_by_id(hardware_id)
        if hw is None:
            return None
        return hw.cost_per_second * duration_seconds

    def estimate_cost(
        self,
        hardware_id: str,
        estimated_minutes: float,
    ) -> Optional[Dict[str, float]]:
        """Estimate cost for a planned job.

        Args:
            hardware_id: The hardware identifier to use.
            estimated_minutes: Estimated job duration in minutes.

        Returns:
            Dict with 'estimated', 'min', 'max' costs, or None if hardware not found.
        """
        hw = self.get_hardware_by_id(hardware_id)
        if hw is None:
            return None

        estimated = hw.cost_per_minute * estimated_minutes
        return {
            "estimated": round(estimated, 4),
            "min": round(estimated * 0.8, 4),  # 20% less
            "max": round(estimated * 1.5, 4),  # 50% more
            "currency": "USD",
        }


class ConfigurablePricingProvider(GPUPricingProvider):
    """GPU pricing provider with configurable overrides.

    Loads default pricing and allows configuration-based overrides
    from the ProviderConfigStore.
    """

    def __init__(
        self,
        provider_name: str,
        default_hardware: Dict[str, Dict[str, Any]],
        default_hardware_id: str,
    ):
        """Initialize the pricing provider.

        Args:
            provider_name: Name of the cloud provider.
            default_hardware: Default hardware configuration dict.
            default_hardware_id: ID of the default hardware option.
        """
        self._provider_name = provider_name
        self._default_hardware = default_hardware
        self._default_hardware_id = default_hardware_id
        self._configured_hardware: Optional[Dict[str, Dict[str, Any]]] = None

    @property
    def provider_name(self) -> str:
        return self._provider_name

    def configure(self, hardware_config: Dict[str, Dict[str, Any]]) -> None:
        """Apply hardware configuration overrides.

        Args:
            hardware_config: Dict of hardware_id -> {name, cost_per_second, ...}
        """
        self._configured_hardware = hardware_config
        logger.debug(
            "Applied hardware config override for %s: %s",
            self._provider_name,
            list(hardware_config.keys()),
        )

    def clear_configuration(self) -> None:
        """Clear configuration overrides, reverting to defaults."""
        self._configured_hardware = None

    def _get_hardware_dict(self) -> Dict[str, Dict[str, Any]]:
        """Get the effective hardware configuration."""
        if self._configured_hardware is not None:
            return self._configured_hardware
        return self._default_hardware

    def get_hardware_options(self) -> List[HardwareOption]:
        hardware = self._get_hardware_dict()
        options = []
        for hw_id, info in hardware.items():
            options.append(
                HardwareOption(
                    id=hw_id,
                    name=info.get("name", hw_id),
                    cost_per_second=info.get("cost_per_second", 0.0),
                    memory_gb=info.get("memory_gb"),
                    gpu_count=info.get("gpu_count", 1),
                    available=info.get("available", True),
                )
            )
        return options

    def get_default_hardware(self) -> HardwareOption:
        hardware = self._get_hardware_dict()

        # Use configured default if available
        if self._default_hardware_id in hardware:
            info = hardware[self._default_hardware_id]
            return HardwareOption(
                id=self._default_hardware_id,
                name=info.get("name", self._default_hardware_id),
                cost_per_second=info.get("cost_per_second", 0.0),
                memory_gb=info.get("memory_gb"),
                gpu_count=info.get("gpu_count", 1),
                available=info.get("available", True),
            )

        # Fall back to first available
        if hardware:
            first_id = next(iter(hardware))
            info = hardware[first_id]
            return HardwareOption(
                id=first_id,
                name=info.get("name", first_id),
                cost_per_second=info.get("cost_per_second", 0.0),
                memory_gb=info.get("memory_gb"),
                gpu_count=info.get("gpu_count", 1),
                available=info.get("available", True),
            )

        # Empty config - return placeholder
        return HardwareOption(
            id="unknown",
            name="Unknown",
            cost_per_second=0.0,
            available=False,
        )


# Provider-specific default configurations

REPLICATE_DEFAULT_HARDWARE = {
    "gpu-l40s": {
        "name": "L40S (48GB)",
        "cost_per_second": 0.000975,
        "memory_gb": 48,
    },
    "gpu-a100-large": {
        "name": "A100 (80GB)",
        "cost_per_second": 0.001400,
        "memory_gb": 80,
    },
}

SIMPLETUNER_IO_DEFAULT_HARDWARE = {
    # Placeholder - pricing will be announced when SimpleTuner.io launches
    "gpu-standard": {
        "name": "Standard GPU",
        "cost_per_second": 0.0,
        "memory_gb": 48,
        "available": False,
    },
}


def get_replicate_pricing() -> ConfigurablePricingProvider:
    """Get the Replicate pricing provider."""
    return ConfigurablePricingProvider(
        provider_name="replicate",
        default_hardware=REPLICATE_DEFAULT_HARDWARE,
        default_hardware_id="gpu-l40s",
    )


def get_simpletuner_io_pricing() -> ConfigurablePricingProvider:
    """Get the SimpleTuner.io pricing provider (coming soon)."""
    return ConfigurablePricingProvider(
        provider_name="simpletuner_io",
        default_hardware=SIMPLETUNER_IO_DEFAULT_HARDWARE,
        default_hardware_id="gpu-standard",
    )


# Pricing provider registry
_pricing_providers: Dict[str, GPUPricingProvider] = {}


def register_pricing_provider(provider: GPUPricingProvider) -> None:
    """Register a pricing provider."""
    _pricing_providers[provider.provider_name] = provider


def get_pricing_provider(provider_name: str) -> Optional[GPUPricingProvider]:
    """Get a pricing provider by name."""
    return _pricing_providers.get(provider_name)


def list_pricing_providers() -> List[str]:
    """List registered pricing provider names."""
    return list(_pricing_providers.keys())


# Register default providers
register_pricing_provider(get_replicate_pricing())
register_pricing_provider(get_simpletuner_io_pricing())
