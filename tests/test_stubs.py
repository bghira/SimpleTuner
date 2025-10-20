"""Lightweight compatibility stubs used across unit tests."""

import importlib.machinery
import sys
import types

_DEF_SPEC = importlib.machinery.ModuleSpec


def _ensure_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__spec__ = _DEF_SPEC(name, loader=None)
        sys.modules[name] = module
    return module


def install_peft_stub():
    if "peft" in sys.modules:
        return
    peft_module = _ensure_module("peft")

    class _DummyLoraConfig:
        def __init__(self, *args, **kwargs):
            pass

    peft_module.LoraConfig = _DummyLoraConfig
    peft_module.LoHaConfig = _DummyLoraConfig

    utils_module = _ensure_module("peft.utils")
    utils_module.set_peft_model_state_dict = lambda *args, **kwargs: None


def install_diffusers_stub():
    diffusers_module = _ensure_module("diffusers")

    class _StubPipeline:
        def __init__(self, *args, **kwargs):
            self.components = {}

    diffusers_module.DiffusionPipeline = _StubPipeline

    config_utils = _ensure_module("diffusers.configuration_utils")

    class _ConfigMixin:
        pass

    def _register_to_config(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    config_utils.ConfigMixin = _ConfigMixin
    config_utils.register_to_config = _register_to_config

    utils_module = _ensure_module("diffusers.utils")
    utils_module.export_to_gif = lambda *args, **kwargs: None
    utils_module.USE_PEFT_BACKEND = False
    utils_module.BaseOutput = dict
    utils_module.convert_state_dict_to_diffusers = lambda *args, **kwargs: {}
    utils_module.convert_state_dict_to_peft = lambda *args, **kwargs: {}
    utils_module.convert_unet_state_dict_to_peft = lambda *args, **kwargs: {}
    utils_module.get_adapter_name = lambda *args, **kwargs: "default"
    utils_module.get_peft_kwargs = lambda *args, **kwargs: {}
    utils_module.is_peft_available = lambda: False
    utils_module.is_peft_version = lambda *args, **kwargs: False
    utils_module.is_torch_version = lambda *args, **kwargs: True
    utils_module.is_torch_xla_available = lambda: False
    utils_module.is_transformers_available = lambda: False
    utils_module.is_transformers_version = lambda *args, **kwargs: False

    image_processor = _ensure_module("diffusers.image_processor")

    class _PipelineImageInput:
        pass

    class _VaeImageProcessor:
        pass

    image_processor.PipelineImageInput = _PipelineImageInput
    image_processor.VaeImageProcessor = _VaeImageProcessor

    optimization_module = _ensure_module("diffusers.optimization")
    optimization_module.get_scheduler = lambda *args, **kwargs: None

    schedulers_module = _ensure_module("diffusers.schedulers")

    class _StubScheduler:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    schedulers_module.FlowMatchEulerDiscreteScheduler = _StubScheduler

    models_module = _ensure_module("diffusers.models")
    autoencoders_module = _ensure_module("diffusers.models.autoencoders")

    class _AutoencoderKL:
        pass

    autoencoders_module.AutoencoderKL = _AutoencoderKL

    controlnets_module = _ensure_module("diffusers.models.controlnets")
    controlnet_flux_module = _ensure_module("diffusers.models.controlnets.controlnet_flux")

    class _FluxControlNetModel:
        pass

    class _FluxMultiControlNetModel:
        pass

    controlnet_flux_module.FluxControlNetModel = _FluxControlNetModel
    controlnet_flux_module.FluxMultiControlNetModel = _FluxMultiControlNetModel

    lora_module = _ensure_module("diffusers.models.lora")
    lora_module.text_encoder_attn_modules = []
    lora_module.text_encoder_mlp_modules = []

    transformers_module = _ensure_module("diffusers.models.transformers")

    class _FluxTransformer2DModel:
        pass

    transformers_module.FluxTransformer2DModel = _FluxTransformer2DModel

    pipelines_module = _ensure_module("diffusers.pipelines")
    flux_package = _ensure_module("diffusers.pipelines.flux")
    flux_pipeline_module = _ensure_module("diffusers.pipelines.flux.pipeline_flux")
    flux_pipeline_module.calculate_shift = lambda *args, **kwargs: 0.0
    flux_output_module = _ensure_module("diffusers.pipelines.flux.pipeline_output")

    class _FluxPipelineOutput(dict):
        pass

    flux_output_module.FluxPipelineOutput = _FluxPipelineOutput

    pipeline_utils = _ensure_module("diffusers.pipelines.pipeline_utils")
    pipeline_utils.DiffusionPipeline = diffusers_module.DiffusionPipeline

    loaders_module = _ensure_module("diffusers.loaders")

    class _BaseMixin:
        pass

    loaders_module.FluxIPAdapterMixin = _BaseMixin
    loaders_module.FromSingleFileMixin = _BaseMixin
    loaders_module.TextualInversionLoaderMixin = _BaseMixin
    loaders_module.LoraBaseMixin = _BaseMixin

    lora_base_module = _ensure_module("diffusers.loaders.lora_base")
    lora_base_module.LoraBaseMixin = type("LoraBaseMixin", (), {})
    lora_base_module._fetch_state_dict = lambda *args, **kwargs: {}

    lora_conv_module = _ensure_module("diffusers.loaders.lora_conversion_utils")
    lora_conv_module.convert_unet_state_dict_to_peft = lambda *args, **kwargs: {}
    lora_conv_module.convert_state_dict_to_diffusers = lambda *args, **kwargs: {}
    lora_conv_module._convert_bfl_flux_control_lora_to_diffusers = lambda *args, **kwargs: {}
    lora_conv_module._convert_kohya_flux_lora_to_diffusers = lambda *args, **kwargs: {}
    lora_conv_module._convert_xlabs_flux_lora_to_diffusers = lambda *args, **kwargs: {}


def ensure_test_stubs_installed() -> None:
    install_peft_stub()
    install_diffusers_stub()


ensure_test_stubs_installed()
