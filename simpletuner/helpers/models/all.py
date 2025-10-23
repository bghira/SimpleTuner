import logging

# Import the models package to trigger auto-discovery
import simpletuner.helpers.models
from simpletuner.helpers.models.registry import ModelRegistry

logger = logging.getLogger(__name__)

model_families = ModelRegistry.model_families()


def get_all_model_flavours() -> list:
    """
    Returns a list of all model flavours available in the model families.
    """
    flavours = []
    for model_family, model_implementation in model_families.items():
        try:
            flavours.extend(list(model_implementation.get_flavour_choices()))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to collect flavours for '%s': %s", model_family, exc)
    return flavours


def get_model_flavour_choices(key_to_find: str = None):
    if key_to_find is not None:
        implementation = model_families.get(key_to_find)
        if implementation is None:
            logger.warning("Requested flavours for unknown model family '%s'", key_to_find)
            return []
        try:
            return list(implementation.get_flavour_choices())
        except Exception as exc:
            logger.error("Failed to fetch flavour choices for '%s': %s", key_to_find, exc)
            return []

    flavour_map = {}
    for model_family, model_implementation in model_families.items():
        try:
            flavour_map[model_family] = list(model_implementation.get_flavour_choices())
        except Exception as exc:
            logger.warning("Failed to collect flavours for '%s': %s", model_family, exc)
            flavour_map[model_family] = []

    # Preserve historical behaviour for consumers expecting a string summary
    formatted_lines = [f"{family}: {choices}" for family, choices in flavour_map.items()]
    return "\n".join(formatted_lines)
