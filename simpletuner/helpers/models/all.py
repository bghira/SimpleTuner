import logging

# Import the models package to trigger auto-discovery
import simpletuner.helpers.models
from simpletuner.helpers.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


def get_all_model_flavours() -> list:
    """
    Returns a list of all model flavours available in the model families.
    """
    model_families = ModelRegistry.model_families()
    flavours = []
    for model_family, model_implementation in model_families.items():
        flavours.extend(list(model_implementation.get_flavour_choices()))
    return flavours


def get_model_flavour_choices(key_to_find: str = None):
    model_families = ModelRegistry.model_families()
    if key_to_find is not None:
        implementation = model_families.get(key_to_find)
        if implementation is None:
            raise Exception(f"Unknown model family '{key_to_find}'")
        return list(implementation.get_flavour_choices())

    flavour_map = {}
    for model_family, model_implementation in model_families.items():
        flavour_map[model_family] = list(model_implementation.get_flavour_choices())

    # Preserve historical behaviour for consumers expecting a string summary
    formatted_lines = [f"{family}: {choices}" for family, choices in flavour_map.items()]
    return "\n".join(formatted_lines)
