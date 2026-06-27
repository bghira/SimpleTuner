"""Centralized provider registry.

This is the single source of truth for cloud provider definitions.
All provider metadata should be defined here and referenced elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class ProviderDefinition:
    """Definition of a cloud provider."""

    id: str
    name: str
    description: str
    coming_soon: bool = False
    default_hardware: str = ""
    default_hardware_id: str = ""

    # Dynamic fields - set at runtime
    model: Optional[str] = None
    version: Optional[str] = None
    hardware: Optional[str] = None
    cost_per_hour: Optional[float] = None
    configured: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "model": self.model,
            "version": self.version,
            "hardware": self.hardware or self.default_hardware,
            "cost_per_hour": self.cost_per_hour,
            "configured": self.configured,
            "coming_soon": self.coming_soon,
        }


# Provider definitions - the single source of truth
PROVIDER_DEFINITIONS: Dict[str, ProviderDefinition] = {
    "replicate": ProviderDefinition(
        id="replicate",
        name="Replicate",
        description="Run SimpleTuner on Replicate's cloud infrastructure",
        coming_soon=False,
        default_hardware="L40S (48GB)",
        default_hardware_id="gpu-l40s",
    ),
    "simpletuner_io": ProviderDefinition(
        id="simpletuner_io",
        name="SimpleTuner.io",
        description="Managed cloud training by the SimpleTuner team",
        coming_soon=True,
        default_hardware="H100 / MI300X",
        default_hardware_id="gpu-standard",
    ),
}


def get_provider_ids() -> List[str]:
    """Get list of all provider IDs."""
    return list(PROVIDER_DEFINITIONS.keys())


def get_available_provider_ids() -> List[str]:
    """Get list of provider IDs that are not coming_soon."""
    return [p.id for p in PROVIDER_DEFINITIONS.values() if not p.coming_soon]


def get_provider_definition(provider_id: str) -> Optional[ProviderDefinition]:
    """Get a provider definition by ID."""
    return PROVIDER_DEFINITIONS.get(provider_id)


def is_valid_provider(provider_id: str) -> bool:
    """Check if a provider ID is valid."""
    return provider_id in PROVIDER_DEFINITIONS


async def get_enriched_providers() -> List[Dict[str, Any]]:
    """Get all providers with dynamic runtime data.

    This enriches the static definitions with:
    - Configuration status (is API token set?)
    - Current model/version
    - Hardware info and costs
    """
    from ...routes.cloud._shared import get_job_store
    from .replicate_client import DEFAULT_MODEL, get_default_hardware_cost_per_hour, get_hardware_info_async
    from .replicate_profiles import (
        DEFAULT_REPLICATE_HARDWARE_PROFILE,
        get_replicate_hardware_profile,
        list_replicate_hardware_profiles,
    )
    from .secrets import get_secrets_manager

    store = get_job_store()
    providers = []

    for provider_id, definition in PROVIDER_DEFINITIONS.items():
        provider_data = definition.to_dict()

        if provider_id == "replicate":
            # Enrich with Replicate-specific data
            replicate_config = await store.get_provider_config("replicate")
            version_override = replicate_config.get("version_override")
            default_profile_id = replicate_config.get("hardware_profile") or DEFAULT_REPLICATE_HARDWARE_PROFILE
            try:
                default_profile = get_replicate_hardware_profile(default_profile_id)
            except ValueError:
                default_profile = get_replicate_hardware_profile(DEFAULT_REPLICATE_HARDWARE_PROFILE)

            hardware_info = await get_hardware_info_async(store)
            l40s_info = hardware_info.get(definition.default_hardware_id, {})
            cost_per_hour = await get_default_hardware_cost_per_hour(store)

            provider_data.update(
                {
                    "model": default_profile.model or DEFAULT_MODEL,
                    "version": version_override,
                    "hardware": l40s_info.get("name", definition.default_hardware),
                    "cost_per_hour": round(cost_per_hour, 2),
                    "configured": bool(get_secrets_manager().get_replicate_token()),
                    "hardware_profile": default_profile.id,
                    "hardware_profiles": list_replicate_hardware_profiles(),
                }
            )

        if provider_id == "simpletuner_io":
            simpletuner_config = await store.get_provider_config("simpletuner_io")
            refresh_token = get_secrets_manager().get("SIMPLETUNER_IO_REFRESH_TOKEN")
            org_id = simpletuner_config.get("org_id")
            configured = bool(refresh_token and org_id)
            provider_data["configured"] = configured
            if configured:
                provider_data["coming_soon"] = False

        providers.append(provider_data)

    return providers
