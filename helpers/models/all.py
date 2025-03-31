from helpers.models.sd3.model import SD3
from helpers.models.deepfloyd.model import DeepFloydIF

model_families = {
    "sd3": SD3,
    "deepfloyd": DeepFloydIF
}

def get_model_flavour_choices(model_family: str = None):
    flavours = {}
    for model_family, model_implementation in model_families.items():
        if model_family is not None and model_family == model_family:
            return model_implementation.get_flavour_choices()
        flavours[model_family] = model_implementation.get_flavour_choices()

    return flavours