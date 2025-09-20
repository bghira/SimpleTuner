from simpletuner.helpers.models.auraflow.model import Auraflow
from simpletuner.helpers.models.cosmos.model import Cosmos2Image
from simpletuner.helpers.models.deepfloyd.model import DeepFloydIF
from simpletuner.helpers.models.flux.model import Flux
from simpletuner.helpers.models.hidream.model import HiDream
from simpletuner.helpers.models.kolors.model import Kolors
from simpletuner.helpers.models.ltxvideo.model import LTXVideo
from simpletuner.helpers.models.lumina2.model import Lumina2
from simpletuner.helpers.models.omnigen.model import OmniGen
from simpletuner.helpers.models.pixart.model import PixartSigma
from simpletuner.helpers.models.qwen_image.model import QwenImage
from simpletuner.helpers.models.sana.model import Sana
from simpletuner.helpers.models.sd1x.model import StableDiffusion1, StableDiffusion2
from simpletuner.helpers.models.sd3.model import SD3
from simpletuner.helpers.models.sdxl.model import SDXL
from simpletuner.helpers.models.wan.model import Wan

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
