#!/usr/bin/env python
"""Focused tests for FactoryRegistry's distiller requirement enforcement."""

import contextlib
import os
import shutil
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class _DecoratorContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False

    def __call__(self, func):
        return func


def _tensor_fn(*_args, **_kwargs):
    return MagicMock()


try:  # pragma: no cover - guard against stripped torch wheels
    import torch as _torch
except ModuleNotFoundError:  # pragma: no cover
    _torch = None


USE_TORCH_STUB = _torch is None or not hasattr(_torch, "Tensor")
if USE_TORCH_STUB:  # pragma: no cover
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    torch.FloatTensor = type("FloatTensor", (), {})
    torch.IntTensor = type("IntTensor", (), {})
    torch.float32 = "float32"
    torch.bfloat16 = "bfloat16"
    torch.autocast = lambda *args, **kwargs: _DecoratorContext()
    torch.no_grad = lambda *args, **kwargs: _DecoratorContext()
    torch.tensor = _tensor_fn
    torch.zeros = _tensor_fn
    torch.randn_like = _tensor_fn
    torch.rand = _tensor_fn
    torch.randn = _tensor_fn
    torch.randint = _tensor_fn
    torch.stack = _tensor_fn
    torch.cat = _tensor_fn
    torch.is_tensor = lambda _: False
    torch.save = MagicMock()
    torch.load = MagicMock()
    torch.set_grad_enabled = MagicMock()
    torch.multinomial = MagicMock(return_value=SimpleNamespace(item=lambda: 0))
    torch.cuda = SimpleNamespace(
        is_available=lambda: False,
        get_device_properties=lambda *_: SimpleNamespace(major=0),
        current_device=lambda: 0,
        memory_allocated=lambda: 0,
        empty_cache=lambda: None,
    )
    torch.backends = SimpleNamespace(
        cuda=SimpleNamespace(allow_tf32=False),
        cudnn=SimpleNamespace(allow_tf32=False),
        mps=SimpleNamespace(is_available=lambda: False),
    )

    torch.utils = types.ModuleType("torch.utils")
    torch.utils.data = types.ModuleType("torch.utils.data")
    torch.utils.data.DataLoader = MagicMock()
    torch.utils.data.Dataset = object
    torch.utils.data.Sampler = object
    torch.utils.checkpoint = types.ModuleType("torch.utils.checkpoint")
    torch.utils.checkpoint.checkpoint = MagicMock()
    sys.modules["torch.utils"] = torch.utils
    sys.modules["torch.utils.data"] = torch.utils.data
    sys.modules["torch.utils.checkpoint"] = torch.utils.checkpoint

    torch.nn = types.ModuleType("torch.nn")
    torch.nn.Module = type("Module", (), {})

    class _FunctionalModule(types.ModuleType):
        def __getattr__(self, name):
            value = MagicMock()
            setattr(self, name, value)
            return value

    torch.nn.functional = _FunctionalModule("torch.nn.functional")
    sys.modules["torch.nn"] = torch.nn
    sys.modules["torch.nn.functional"] = torch.nn.functional

    torch.distributions = types.ModuleType("torch.distributions")
    torch.distributions.Beta = MagicMock()
    sys.modules["torch.distributions"] = torch.distributions

    torch.optim = types.ModuleType("torch.optim")
    torch.optim.lr_scheduler = types.ModuleType("torch.optim.lr_scheduler")
    torch.optim.lr_scheduler.LambdaLR = MagicMock()
    torch.optim.lr_scheduler.LRScheduler = MagicMock()
    sys.modules["torch.optim"] = torch.optim
    sys.modules["torch.optim.lr_scheduler"] = torch.optim.lr_scheduler

    torch.device = lambda *_args, **_kwargs: "cpu"

    sys.modules["torch"] = torch
else:  # pragma: no cover
    torch = _torch
    if not hasattr(torch, "cuda"):
        torch.cuda = SimpleNamespace(
            is_available=lambda: False,
            get_device_properties=lambda *_: SimpleNamespace(major=0),
            current_device=lambda: 0,
            memory_allocated=lambda: 0,
            empty_cache=lambda: None,
        )
    if not hasattr(torch, "backends"):
        torch.backends = SimpleNamespace(
            cuda=SimpleNamespace(allow_tf32=False),
            cudnn=SimpleNamespace(allow_tf32=False),
            mps=SimpleNamespace(is_available=lambda: False),
        )
    if not hasattr(torch, "no_grad"):
        torch.no_grad = lambda *args, **kwargs: _DecoratorContext()
    if not hasattr(torch, "autocast"):
        torch.autocast = lambda *args, **kwargs: _DecoratorContext()

if "torch.distributed" not in sys.modules:  # pragma: no cover
    sys.modules["torch.distributed"] = MagicMock()

if "pandas" not in sys.modules:  # pragma: no cover
    sys.modules["pandas"] = MagicMock()

if "boto3" not in sys.modules:  # pragma: no cover
    boto3 = types.ModuleType("boto3")
    boto3.client = MagicMock()
    boto3.resource = MagicMock()
    sys.modules["boto3"] = boto3

if "botocore" not in sys.modules:  # pragma: no cover
    botocore = types.ModuleType("botocore")
    botocore.config = types.ModuleType("botocore.config")
    botocore.config.Config = type("Config", (), {})
    botocore.exceptions = types.ModuleType("botocore.exceptions")
    botocore.exceptions.NoCredentialsError = type("NoCredentialsError", (Exception,), {})
    botocore.exceptions.PartialCredentialsError = type("PartialCredentialsError", (Exception,), {})
    sys.modules["botocore"] = botocore
    sys.modules["botocore.config"] = botocore.config
    sys.modules["botocore.exceptions"] = botocore.exceptions

if "atomicwrites" not in sys.modules:  # pragma: no cover
    atomicwrites = types.ModuleType("atomicwrites")

    def atomic_write(*_args, **_kwargs):
        class _Writer:
            def __enter__(self):
                return open(os.devnull, "w")

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Writer()

    atomicwrites.atomic_write = atomic_write
    sys.modules["atomicwrites"] = atomicwrites

if "diffusers" not in sys.modules:  # pragma: no cover
    diffusers = types.ModuleType("diffusers")

    class _DummyDiffusionPipeline:
        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def to(self, *args, **kwargs):
            return self

        def load_lora_weights(self, *args, **kwargs):
            return self

    diffusers.DiffusionPipeline = _DummyDiffusionPipeline
    diffusers.optimization = types.ModuleType("diffusers.optimization")
    diffusers.optimization.get_scheduler = MagicMock()
    diffusers.pipelines = types.ModuleType("diffusers.pipelines")
    diffusers.pipelines.flux = types.ModuleType("diffusers.pipelines.flux")
    diffusers.pipelines.flux.pipeline_flux = types.ModuleType("diffusers.pipelines.flux.pipeline_flux")
    diffusers.pipelines.flux.pipeline_flux.calculate_shift = MagicMock()
    diffusers.image_processor = types.ModuleType("diffusers.image_processor")
    diffusers.image_processor.PipelineImageInput = MagicMock()
    diffusers.image_processor.VaeImageProcessor = MagicMock()
    diffusers.loaders = types.ModuleType("diffusers.loaders")
    diffusers.loaders.FluxIPAdapterMixin = type("FluxIPAdapterMixin", (), {})
    diffusers.loaders.FromSingleFileMixin = type("FromSingleFileMixin", (), {})
    diffusers.loaders.TextualInversionLoaderMixin = type("TextualInversionLoaderMixin", (), {})
    diffusers.loaders.lora_base = types.ModuleType("diffusers.loaders.lora_base")
    diffusers.loaders.lora_base.LoraBaseMixin = type("LoraBaseMixin", (), {})
    diffusers.loaders.lora_base._fetch_state_dict = MagicMock()
    diffusers.loaders.lora_conversion_utils = types.ModuleType("diffusers.loaders.lora_conversion_utils")
    diffusers.loaders.lora_conversion_utils.maybe_try_to_conv_state_dict = MagicMock()
    diffusers.loaders.lora_conversion_utils.SAFETENSORS_WEIGHTS_NAME = "model.safetensors"
    diffusers.loaders.lora_conversion_utils.WEIGHTS_NAME = "pytorch_model.bin"
    diffusers.loaders.lora_conversion_utils._convert_bfl_flux_control_lora_to_diffusers = MagicMock()
    diffusers.loaders.lora_conversion_utils._convert_kohya_flux_lora_to_diffusers = MagicMock()
    diffusers.loaders.lora_conversion_utils._convert_xlabs_flux_lora_to_diffusers = MagicMock()
    diffusers.configuration_utils = types.ModuleType("diffusers.configuration_utils")
    diffusers.configuration_utils.ConfigMixin = type("ConfigMixin", (), {})

    def _register_to_config(*_args, **_kwargs):
        def decorator(obj):
            return obj

        return decorator

    diffusers.configuration_utils.register_to_config = _register_to_config
    diffusers.utils = types.ModuleType("diffusers.utils")
    diffusers.utils.BaseOutput = type("BaseOutput", (), {})
    diffusers.utils.USE_PEFT_BACKEND = False
    diffusers.utils.convert_state_dict_to_diffusers = MagicMock()
    diffusers.utils.convert_state_dict_to_peft = MagicMock()
    diffusers.utils.convert_unet_state_dict_to_peft = MagicMock()
    diffusers.utils.get_adapter_name = MagicMock(return_value="adapter")
    diffusers.utils.get_peft_kwargs = MagicMock(return_value={})
    diffusers.utils.is_peft_available = lambda: False
    diffusers.utils.is_peft_version = lambda *args, **kwargs: False
    diffusers.utils.is_torch_version = lambda *args, **kwargs: False
    diffusers.utils.is_torch_xla_available = lambda: False
    diffusers.utils.is_transformers_available = lambda: False
    diffusers.utils.is_transformers_version = lambda *args, **kwargs: False
    diffusers.utils.logging = MagicMock(get_logger=lambda *_args, **_kwargs: MagicMock())
    diffusers.utils.replace_example_docstring = lambda *args, **kwargs: (lambda obj: obj)
    diffusers.utils.scale_lora_layers = MagicMock()
    diffusers.utils.unscale_lora_layers = MagicMock()
    diffusers.utils.torch_utils = types.ModuleType("diffusers.utils.torch_utils")
    diffusers.utils.torch_utils.randn_tensor = MagicMock()
    diffusers.utils.torch_utils.is_compiled_module = lambda *_args, **_kwargs: False
    diffusers.utils.export_utils = types.ModuleType("diffusers.utils.export_utils")
    diffusers.utils.export_utils.export_to_gif = MagicMock()
    diffusers.models = types.ModuleType("diffusers.models")
    diffusers.models.autoencoders = types.ModuleType("diffusers.models.autoencoders")
    diffusers.models.autoencoders.AutoencoderKL = type("AutoencoderKL", (), {})
    diffusers.models.autoencoders.vae = types.ModuleType("diffusers.models.autoencoders.vae")
    diffusers.models.autoencoders.vae.DiagonalGaussianDistribution = type("DiagonalGaussianDistribution", (), {})
    diffusers.models.controlnets = types.ModuleType("diffusers.models.controlnets")
    diffusers.models.controlnets.controlnet_flux = types.ModuleType("diffusers.models.controlnets.controlnet_flux")
    diffusers.models.controlnets.controlnet_flux.FluxControlNetModel = type("FluxControlNetModel", (), {})
    diffusers.models.controlnets.controlnet_flux.FluxMultiControlNetModel = type("FluxMultiControlNetModel", (), {})
    diffusers.models.lora = types.ModuleType("diffusers.models.lora")
    diffusers.models.lora.text_encoder_attn_modules = []
    diffusers.models.lora.text_encoder_mlp_modules = []
    diffusers.models.transformers = types.ModuleType("diffusers.models.transformers")
    diffusers.models.transformers.FluxTransformer2DModel = type("FluxTransformer2DModel", (), {})
    diffusers.schedulers = types.ModuleType("diffusers.schedulers")
    diffusers.schedulers.FlowMatchEulerDiscreteScheduler = type("FlowMatchEulerDiscreteScheduler", (), {})
    diffusers.schedulers.scheduling_utils = types.ModuleType("diffusers.schedulers.scheduling_utils")
    diffusers.schedulers.scheduling_utils.SchedulerMixin = type("SchedulerMixin", (), {})
    diffusers.schedulers.scheduling_utils.KarrasDiffusionSchedulers = tuple()
    sys.modules["diffusers"] = diffusers
    sys.modules["diffusers.optimization"] = diffusers.optimization
    sys.modules["diffusers.pipelines"] = diffusers.pipelines
    sys.modules["diffusers.pipelines.flux"] = diffusers.pipelines.flux
    sys.modules["diffusers.pipelines.flux.pipeline_flux"] = diffusers.pipelines.flux.pipeline_flux
    sys.modules["diffusers.image_processor"] = diffusers.image_processor
    sys.modules["diffusers.loaders"] = diffusers.loaders
    sys.modules["diffusers.loaders.lora_base"] = diffusers.loaders.lora_base
    sys.modules["diffusers.loaders.lora_conversion_utils"] = diffusers.loaders.lora_conversion_utils
    sys.modules["diffusers.configuration_utils"] = diffusers.configuration_utils
    sys.modules["diffusers.schedulers"] = diffusers.schedulers
    sys.modules["diffusers.schedulers.scheduling_utils"] = diffusers.schedulers.scheduling_utils
    sys.modules["diffusers.utils"] = diffusers.utils
    sys.modules["diffusers.utils.torch_utils"] = diffusers.utils.torch_utils
    sys.modules["diffusers.utils.export_utils"] = diffusers.utils.export_utils
    sys.modules["diffusers.models"] = diffusers.models
    sys.modules["diffusers.models.autoencoders"] = diffusers.models.autoencoders
    sys.modules["diffusers.models.autoencoders.vae"] = diffusers.models.autoencoders.vae
    sys.modules["diffusers.models.controlnets"] = diffusers.models.controlnets
    sys.modules["diffusers.models.controlnets.controlnet_flux"] = diffusers.models.controlnets.controlnet_flux
    sys.modules["diffusers.models.lora"] = diffusers.models.lora
    sys.modules["diffusers.models.transformers"] = diffusers.models.transformers

if "peft" not in sys.modules:  # pragma: no cover
    peft = types.ModuleType("peft")

    class _LoraConfig:
        def __init__(self, *args, **kwargs):
            pass

    peft.LoraConfig = _LoraConfig
    sys.modules["peft"] = peft

if "torchvision" not in sys.modules:  # pragma: no cover
    torchvision = types.ModuleType("torchvision")
    transforms_module = types.ModuleType("torchvision.transforms")
    transforms_module.Compose = lambda *args, **kwargs: None
    transforms_module.ToTensor = lambda *args, **kwargs: None
    transforms_module.Normalize = lambda *args, **kwargs: None
    torchvision.transforms = transforms_module
    torchvision.io = types.ModuleType("torchvision.io")
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = transforms_module
    video_reader_module = types.ModuleType("torchvision.io.video_reader")
    video_reader_module.VideoReader = type("VideoReader", (), {})
    torchvision.io.video_reader = video_reader_module
    sys.modules["torchvision.io"] = torchvision.io
    sys.modules["torchvision.io.video_reader"] = video_reader_module

if "transformers" not in sys.modules:  # pragma: no cover
    transformers = types.ModuleType("transformers")
    sys.modules["transformers"] = transformers
else:
    transformers = sys.modules["transformers"]

if not hasattr(transformers, "utils"):
    transformers.utils = types.ModuleType("transformers.utils")
    sys.modules["transformers.utils"] = transformers.utils


class _ContextManagers:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


transformers.utils.ContextManagers = _ContextManagers


class _DummyTransformer:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


class _DummyTokenizer(_DummyTransformer):
    def __call__(self, *args, **kwargs):
        return {}


transformers.CLIPImageProcessor = _DummyTransformer
transformers.CLIPTextModel = _DummyTransformer
transformers.CLIPTokenizer = _DummyTokenizer
transformers.CLIPVisionModelWithProjection = _DummyTransformer
transformers.T5EncoderModel = _DummyTransformer
transformers.T5TokenizerFast = _DummyTokenizer

if not hasattr(transformers, "integrations"):
    transformers.integrations = types.ModuleType("transformers.integrations")
    transformers.integrations.HfDeepSpeedConfig = _DummyTransformer
    transformers.integrations.deepspeed = types.ModuleType("transformers.integrations.deepspeed")
    transformers.integrations.deepspeed.is_deepspeed_zero3_enabled = lambda: False
    transformers.integrations.deepspeed.set_hf_deepspeed_config = MagicMock()
    transformers.integrations.deepspeed.unset_hf_deepspeed_config = MagicMock()
    transformers.integrations.deepspeed._hf_deepspeed_config_weak_ref = None
    sys.modules["transformers.integrations"] = transformers.integrations
    sys.modules["transformers.integrations.deepspeed"] = transformers.integrations.deepspeed

if "safetensors" not in sys.modules:  # pragma: no cover
    safetensors = types.ModuleType("safetensors")
    safetensors.torch = types.ModuleType("safetensors.torch")
    safetensors.torch.load_file = MagicMock()
    safetensors.torch.save_file = MagicMock()
    sys.modules["safetensors"] = safetensors
    sys.modules["safetensors.torch"] = safetensors.torch

if "accelerate" not in sys.modules:  # pragma: no cover
    accelerate = types.ModuleType("accelerate")
    accelerate.utils = types.ModuleType("accelerate.utils")
    accelerate.utils.set_module_tensor_to_device = MagicMock()
    accelerate.utils.offload_state_dict = MagicMock()
    accelerate.utils.load_checkpoint_and_dispatch = MagicMock()
    accelerate.__all__ = []
    accelerate.logging = types.ModuleType("accelerate.logging")
    accelerate.logging.get_logger = MagicMock(return_value=MagicMock())
    accelerate.state = types.ModuleType("accelerate.state")

    class _AcceleratorState:
        def __init__(self, *_args, **_kwargs):
            self.deepspeed_plugin = SimpleNamespace(
                deepspeed_config={"zero_optimization": {}, "optimizer": {}, "scheduler": {}},
                zero3_init_context_manager=lambda enable=False: contextlib.nullcontext(),
            )

    accelerate.state.AcceleratorState = _AcceleratorState
    accelerate.state.is_initialized = staticmethod(lambda: False)

    sys.modules["accelerate"] = accelerate
    sys.modules["accelerate.utils"] = accelerate.utils
    sys.modules["accelerate.state"] = accelerate.state
    sys.modules["accelerate.logging"] = accelerate.logging

if "simpletuner.helpers.models.flux" not in sys.modules:  # pragma: no cover
    flux_stub = types.ModuleType("simpletuner.helpers.models.flux")

    def _calculate_shift_flux(*_args, **_kwargs):
        return 0.0

    flux_stub.calculate_shift_flux = _calculate_shift_flux
    sys.modules["simpletuner.helpers.models.flux"] = flux_stub

if "simpletuner.helpers.webhooks.events" not in sys.modules:  # pragma: no cover
    webhook_events = types.ModuleType("simpletuner.helpers.webhooks.events")

    def _lifecycle_stage_event(*_args, **_kwargs):
        return {}

    def _attach_timestamp(event):
        event["timestamp"] = "1970-01-01T00:00:00Z"
        return event

    webhook_events.lifecycle_stage_event = _lifecycle_stage_event
    webhook_events.attach_timestamp = _attach_timestamp
    sys.modules["simpletuner.helpers.webhooks.events"] = webhook_events

if "simpletuner.helpers.webhooks.config" not in sys.modules:  # pragma: no cover
    webhook_config_module = types.ModuleType("simpletuner.helpers.webhooks.config")

    class _WebhookConfig:
        def __init__(self, config):
            self.__dict__.update(config if isinstance(config, dict) else {})

    webhook_config_module.WebhookConfig = _WebhookConfig
    sys.modules["simpletuner.helpers.webhooks.config"] = webhook_config_module

if "simpletuner.helpers.webhooks.handler" not in sys.modules:  # pragma: no cover
    webhook_handler_module = types.ModuleType("simpletuner.helpers.webhooks.handler")

    class _WebhookHandler:
        def __init__(self, *args, **kwargs):
            self.backends = []
            self.webhook_handler = None

        def send_raw(self, *_args, **_kwargs):
            return None

        def send_lifecycle_stage(self, *_args, **_kwargs):
            return None

    webhook_handler_module.WebhookHandler = _WebhookHandler
    sys.modules["simpletuner.helpers.webhooks.handler"] = webhook_handler_module

if "trainingsample" not in sys.modules:  # pragma: no cover
    trainingsample_stub = types.ModuleType("trainingsample")

    def _return_first_arg(items, *_args, **_kwargs):
        return items

    trainingsample_stub.batch_resize_images = _return_first_arg
    trainingsample_stub.batch_center_crop_images = _return_first_arg
    trainingsample_stub.batch_random_crop_images = _return_first_arg
    trainingsample_stub.batch_calculate_luminance = lambda images, *_args, **_kwargs: [0.0 for _ in images]
    trainingsample_stub.batch_resize_videos = _return_first_arg
    sys.modules["trainingsample"] = trainingsample_stub

# Ensure the project root is on sys.path for direct test execution.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.data_backend.factory import FactoryRegistry
from simpletuner.helpers.distillation.requirements import DataRequirement, DistillerRequirementProfile


class DummyConfig:
    """Minimal config stub for patched create_backend_config."""

    def __init__(self, backend):
        self.backend = backend

    def validate(self, *_args, **_kwargs):
        return None


class TestFactoryDistillerRequirements(unittest.TestCase):
    """Ensure FactoryRegistry enforces distiller requirement profiles."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(lambda: os.path.isdir(self.temp_dir) and shutil.rmtree(self.temp_dir, ignore_errors=True))

        self.args = SimpleNamespace(
            cache_dir=self.temp_dir,
            cache_dir_text=os.path.join(self.temp_dir, "text"),
            cache_dir_vae=os.path.join(self.temp_dir, "vae"),
            caption_dropout_probability=0.0,
            metadata_update_interval=60,
            train_batch_size=1,
            gradient_accumulation_steps=1,
            cache_file_suffix=None,
            resolution=1024,
            resolution_type="area",
            target_downsample_size=1.0,
            minimum_image_size=0.5,
            maximum_image_size=2.0,
            vae_cache_scan_behaviour="ignore",
            vae_cache_ondemand=False,
            skip_file_discovery="",
            caption_strategy="filename",
            prepend_instance_prompt=False,
            instance_prompt=None,
            only_instance_prompt=False,
            debug_aspect_buckets=False,
            vae_batch_size=1,
            write_batch_size=1,
            read_batch_size=1,
            max_workers=1,
            image_processing_batch_size=1,
            max_train_steps=0,
            override_dataset_config=False,
            eval_dataset_id=None,
            controlnet=False,
            distillation_method="sf_dmd",
            model_family="sdxl",
            output_dir=self.temp_dir,
            metadata_backend="discovery",
        )

        self.accelerator = MagicMock()
        self.accelerator.is_main_process = True
        self.accelerator.is_local_main_process = True
        self.accelerator.num_processes = 1

        self.model = MagicMock()
        self.model.requires_conditioning_dataset.return_value = False
        self.model.requires_conditioning_latents.return_value = False
        self.model.requires_conditioning_image_embeds.return_value = False
        self.model.get_vae.return_value = MagicMock()
        self.model.AUTOENCODER_CLASS = "AutoencoderKL"

        self.state_tracker_patch = patch("simpletuner.helpers.data_backend.factory.StateTracker")
        self.mock_state_tracker = self.state_tracker_patch.start()
        self.addCleanup(self.state_tracker_patch.stop)
        self._setup_state_tracker_mocks()

        self.create_backend_config_patch = patch(
            "simpletuner.helpers.data_backend.factory.create_backend_config",
            side_effect=lambda backend, _args: DummyConfig(backend),
        )
        self.create_backend_config_patch.start()
        self.addCleanup(self.create_backend_config_patch.stop)

        self.create_backend_builder_patch = patch(
            "simpletuner.helpers.data_backend.factory.create_backend_builder",
            side_effect=self._create_backend_builder_stub,
        )
        self.create_backend_builder_patch.start()
        self.addCleanup(self.create_backend_builder_patch.stop)

        self.init_backend_config_patch = patch(
            "simpletuner.helpers.data_backend.factory.init_backend_config",
            side_effect=lambda backend, _args, _accelerator: {
                "id": backend["id"],
                "config": dict(backend),
                "dataset_type": backend.get("dataset_type", "image"),
            },
        )
        self.init_backend_config_patch.start()
        self.addCleanup(self.init_backend_config_patch.stop)

    def _setup_state_tracker_mocks(self):
        self.mock_state_tracker.get_args.return_value = self.args
        self.mock_state_tracker.get_accelerator.return_value = self.accelerator
        self.mock_state_tracker.get_webhook_handler.return_value = None
        self.mock_state_tracker.get_data_backends.return_value = {}
        self.mock_state_tracker.get_conditioning_mappings.return_value = []
        self.mock_state_tracker.clear_data_backends.return_value = None
        self.mock_state_tracker.set_data_backend_config.return_value = None
        self.mock_state_tracker.register_data_backend.return_value = None
        self.mock_state_tracker.delete_cache_files.return_value = None
        self.mock_state_tracker.load_aspect_resolution_map.return_value = None

    @staticmethod
    def _create_backend_builder_stub(*_args, **_kwargs):
        builder = MagicMock()
        builder.build.return_value = MagicMock()
        return builder

    def _caption_only_profile(self):
        return DistillerRequirementProfile(
            requirements=(DataRequirement(dataset_types=(DatasetType.CAPTION,)),),
            is_data_generator=True,
        )

    def _create_factory(self):
        factory = FactoryRegistry(
            args=self.args,
            accelerator=self.accelerator,
            text_encoders=[],
            tokenizers=[],
            model=self.model,
            distiller_profile=self._caption_only_profile(),
            distillation_method="sf_dmd",
        )
        factory.text_embed_backends = {"default": {"text_embed_cache": MagicMock()}}
        factory.default_text_embed_backend_id = "default"
        return factory

    def test_caption_profile_allows_caption_only_config(self):
        factory = self._create_factory()

        caption_backend = {
            "id": "caption_ds",
            "type": "local",
            "dataset_type": "caption",
            "instance_data_dir": "/data/captions",
        }

        factory.configure_data_backends([caption_backend])

        self.assertIn("caption_ds", factory.caption_backends)
        evaluation = factory._distiller_requirement_result
        self.assertIsNotNone(evaluation)
        self.assertTrue(evaluation.fulfilled)
        self.assertIn(DatasetType.CAPTION, evaluation.dataset_types)

    def test_caption_profile_raises_when_missing_caption_dataset(self):
        factory = self._create_factory()

        with self.assertRaisesRegex(ValueError, "requires datasets matching"):
            factory.configure_data_backends([])


if __name__ == "__main__":
    unittest.main()
