from helpers.models.sd3.model import SD3
from helpers.models.deepfloyd.model import DeepFloydIF
from helpers.models.sana.model import Sana
from helpers.models.sdxl.model import SDXL
from helpers.models.kolors.model import Kolors
from helpers.models.flux.model import Flux
from helpers.models.wan.model import Wan
from helpers.models.ltxvideo.model import LTXVideo
from helpers.models.sd1x.model import StableDiffusion1, StableDiffusion2
from helpers.models.pixart.model import PixartSigma
from helpers.models.hidream.model import HiDream
from helpers.models.omnigen.model import OmniGen
from helpers.models.auraflow.model import Auraflow
from helpers.models.lumina2.model import Lumina2
from helpers.models.cosmos.model import Cosmos2Image
from helpers.models.qwen_image.model import QwenImage

model_families = {
    "sd1x": StableDiffusion1,
    "sd2x": StableDiffusion2,
    "sd3": SD3,
    "deepfloyd": DeepFloydIF,
    "sana": Sana,
    "sdxl": SDXL,
    "kolors": Kolors,
    "flux": Flux,
    "wan": Wan,
    "ltxvideo": LTXVideo,
    "pixart_sigma": PixartSigma,
    "omnigen": OmniGen,
    "hidream": HiDream,
    "auraflow": Auraflow,
    "lumina2": Lumina2,
    "cosmos2image": Cosmos2Image,
    "qwen_image": QwenImage,
}


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
