from helpers.models.sd3.model import SD3
from helpers.models.deepfloyd.model import DeepFloydIF


model_families = {"sd3": SD3, "deepfloyd": DeepFloydIF}


def get_model_flavour_choices(key_to_find: str = None):
    flavours = ""
    for model_family, model_implementation in model_families.items():
        if key_to_find is not None and model_family == key_to_find:
            return model_implementation.get_flavour_choices()
        flavours += f"""
{model_family}: {model_implementation.get_flavour_choices()}
"""

    return flavours
