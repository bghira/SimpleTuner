# Copyright 2025 SimpleTuner contributors
# Reuse the Flux LoRA loader to provide Kandinsky-compatible LoRA handling without relying on diffusers' mixin.
from simpletuner.helpers.models.flux.pipeline import FluxLoraLoaderMixin


class KandinskyLoraLoaderMixin(FluxLoraLoaderMixin):
    """
    Minimal Kandinsky LoRA loader mixin.

    Reuses the Flux implementation but exposes both text encoders so LoRA weights can be applied consistently.
    """

    _lora_loadable_modules = ["transformer", "text_encoder", "text_encoder_2"]
    transformer_name = "transformer"
    text_encoder_name = "text_encoder"
    text_encoder_2_name = "text_encoder_2"
