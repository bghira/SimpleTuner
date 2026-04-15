from simpletuner.helpers.models.ernie.pipeline_diffusers import ErnieImagePipeline as DiffusersErnieImagePipeline
from simpletuner.helpers.models.z_image.pipeline import ZImageLoraLoaderMixin


class ErnieImageLoraLoaderMixin(ZImageLoraLoaderMixin):
    _lora_loadable_modules = ["transformer"]
    transformer_name = "transformer"


class ErnieImagePipeline(ErnieImageLoraLoaderMixin, DiffusersErnieImagePipeline):
    pass
