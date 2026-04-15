from simpletuner.helpers.models.z_image.pipeline import ZImageLoraLoaderMixin

try:
    from diffusers import ErnieImagePipeline as DiffusersErnieImagePipeline
except ImportError as exc:
    if "ErnieImagePipeline" not in str(exc):
        raise
    DiffusersErnieImagePipeline = None
    _ERNIE_PIPELINE_IMPORT_ERROR = exc
else:
    _ERNIE_PIPELINE_IMPORT_ERROR = None


class ErnieImageLoraLoaderMixin(ZImageLoraLoaderMixin):
    _lora_loadable_modules = ["transformer"]
    transformer_name = "transformer"


if DiffusersErnieImagePipeline is None:

    def _raise_missing_pipeline_import():
        raise ImportError(
            "Diffusers does not provide ErnieImagePipeline. Update diffusers to a version "
            "with ERNIE-Image pipeline support."
        ) from _ERNIE_PIPELINE_IMPORT_ERROR

    class ErnieImagePipeline(ErnieImageLoraLoaderMixin):
        def __init__(self, *args, **kwargs):
            _raise_missing_pipeline_import()

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            _raise_missing_pipeline_import()

else:

    class ErnieImagePipeline(ErnieImageLoraLoaderMixin, DiffusersErnieImagePipeline):
        pass
