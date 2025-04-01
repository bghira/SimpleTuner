from helpers.models.sd3.model import SD3
from helpers.models.deepfloyd.model import DeepFloydIF
from helpers.models.sana.model import Sana

model_families = {"sd3": SD3, "deepfloyd": DeepFloydIF, "sana": Sana}


def get_all_model_flavours() -> list:
    """
    Returns a list of all model flavours available in the model families.
    """
    flavours = []
    for model_family, model_implementation in model_families.items():
        flavours.extend(list(model_implementation.get_flavour_choices()))
    return flavours


def get_model_flavour_choices(key_to_find: str = None):
    flavours = ""
    for model_family, model_implementation in model_families.items():
        if key_to_find is not None and model_family == key_to_find:
            return model_implementation.get_flavour_choices()
        flavours += f"""
{model_family}: {model_implementation.get_flavour_choices()}
"""

    return flavours
