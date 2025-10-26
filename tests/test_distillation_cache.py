import contextlib
import os
import shutil
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

try:  # pragma: no cover
    import torch as _real_torch
except ModuleNotFoundError:  # pragma: no cover
    _real_torch = None


def _tensor_fn(*args, **kwargs):
    return MagicMock()


class _DecoratorContext(contextlib.AbstractContextManager):
    def __init__(self, result=None):
        self._result = result

    def __enter__(self):
        return self._result

    def __exit__(self, exc_type, exc, tb):
        return False

    def __call__(self, func):
        return func


USE_TORCH_STUB = _real_torch is None or not hasattr(_real_torch, "Tensor")

if USE_TORCH_STUB:
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    torch.float32 = "float32"
    torch.bfloat16 = "bfloat16"
    torch.autocast = lambda *args, **kwargs: _DecoratorContext()
    torch.no_grad = lambda *args, **kwargs: _DecoratorContext()
    torch.tensor = _tensor_fn
    torch.zeros = _tensor_fn
    torch.randn_like = _tensor_fn
    torch.rand = _tensor_fn
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
    )
    torch.backends = SimpleNamespace(
        cuda=SimpleNamespace(allow_tf32=False),
        cudnn=SimpleNamespace(allow_tf32=False),
    )

    torch.utils = types.ModuleType("torch.utils")
    torch.utils.data = types.ModuleType("torch.utils.data")
    torch.utils.data.DataLoader = MagicMock()
    sys.modules["torch.utils"] = torch.utils
    sys.modules["torch.utils.data"] = torch.utils.data

    torch.nn = types.ModuleType("torch.nn")

    class _FunctionalModule(types.ModuleType):
        def __getattr__(self, name):  # pragma: no cover - simple stub
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

    sys.modules["torch"] = torch
else:  # pragma: no cover
    torch = _real_torch

if "torch.distributed" not in sys.modules:
    sys.modules["torch.distributed"] = MagicMock()

if "pandas" not in sys.modules:
    sys.modules["pandas"] = MagicMock()

if "diffusers" not in sys.modules:
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
            pass

        def fuse_lora(self, *args, **kwargs):
            pass

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
    diffusers.loaders.FluxIPAdapterMixin = MagicMock()
    diffusers.loaders.FromSingleFileMixin = MagicMock()
    diffusers.loaders.TextualInversionLoaderMixin = MagicMock()
    diffusers.loaders.lora_base = types.ModuleType("diffusers.loaders.lora_base")

    class _DummyLoraBaseMixin:
        pass

    diffusers.loaders.lora_base.LoraBaseMixin = _DummyLoraBaseMixin
    diffusers.loaders.lora_base._fetch_state_dict = MagicMock()
    diffusers.loaders.lora_conversion_utils = types.ModuleType("diffusers.loaders.lora_conversion_utils")
    diffusers.loaders.lora_conversion_utils.convert_all_state_dict_to_peft = MagicMock()
    diffusers.loaders.lora_conversion_utils.convert_state_dict_to_diffusers = MagicMock()
    diffusers.loaders.lora_conversion_utils._convert_bfl_flux_control_lora_to_diffusers = MagicMock()
    diffusers.loaders.lora_conversion_utils._convert_kohya_flux_lora_to_diffusers = MagicMock()
    diffusers.loaders.lora_conversion_utils._convert_xlabs_flux_lora_to_diffusers = MagicMock()
    diffusers.models = types.ModuleType("diffusers.models")
    diffusers.models.autoencoders = types.ModuleType("diffusers.models.autoencoders")
    diffusers.models.autoencoders.AutoencoderKL = MagicMock()
    diffusers.models.autoencoders.vae = types.ModuleType("diffusers.models.autoencoders.vae")
    diffusers.models.autoencoders.vae.DiagonalGaussianDistribution = MagicMock()
    diffusers.models.controlnets = types.ModuleType("diffusers.models.controlnets")
    diffusers.models.controlnets.controlnet_flux = types.ModuleType("diffusers.models.controlnets.controlnet_flux")
    diffusers.models.controlnets.controlnet_flux.FluxControlNetModel = MagicMock()
    diffusers.models.controlnets.controlnet_flux.FluxMultiControlNetModel = MagicMock()
    diffusers.models.lora = types.ModuleType("diffusers.models.lora")

    def _empty_iter(*args, **kwargs):
        return []

    diffusers.models.lora.text_encoder_attn_modules = _empty_iter
    diffusers.models.lora.text_encoder_mlp_modules = _empty_iter
    diffusers.models.transformers = types.ModuleType("diffusers.models.transformers")
    diffusers.models.transformers.FluxTransformer2DModel = MagicMock()
    diffusers.pipelines.pipeline_utils = types.ModuleType("diffusers.pipelines.pipeline_utils")
    diffusers.pipelines.pipeline_utils.DiffusionPipeline = _DummyDiffusionPipeline
    diffusers.pipelines.flux.pipeline_output = types.ModuleType("diffusers.pipelines.flux.pipeline_output")
    diffusers.pipelines.flux.pipeline_output.FluxPipelineOutput = MagicMock()
    diffusers.schedulers = types.ModuleType("diffusers.schedulers")
    diffusers.schedulers.FlowMatchEulerDiscreteScheduler = MagicMock()
    diffusers.schedulers.scheduling_utils = types.ModuleType("diffusers.schedulers.scheduling_utils")

    class _DummySchedulerMixin:
        pass

    diffusers.schedulers.scheduling_utils.SchedulerMixin = _DummySchedulerMixin
    diffusers.schedulers.scheduling_utils.KarrasDiffusionSchedulers = [SimpleNamespace(name="dummy")]
    diffusers.utils = types.ModuleType("diffusers.utils")

    class _BaseOutput:
        pass

    diffusers.utils.USE_PEFT_BACKEND = False
    diffusers.utils.BaseOutput = _BaseOutput

    def _identity_payload(payload=None, *args, **kwargs):
        return payload

    def _false_fn(*args, **kwargs):
        return False

    def _true_fn(*args, **kwargs):
        return True

    def _decorator_factory(*args, **kwargs):
        def _decorator(func):
            return func

        return _decorator

    diffusers.utils.convert_state_dict_to_diffusers = _identity_payload
    diffusers.utils.convert_state_dict_to_peft = _identity_payload
    diffusers.utils.convert_unet_state_dict_to_peft = _identity_payload
    diffusers.utils.get_adapter_name = lambda *_, **__: "adapter"
    diffusers.utils.get_peft_kwargs = lambda *_, **__: {}
    diffusers.utils.is_peft_available = _false_fn
    diffusers.utils.is_peft_version = _false_fn
    diffusers.utils.is_torch_version = _true_fn
    diffusers.utils.is_torch_xla_available = _false_fn
    diffusers.utils.is_transformers_available = _true_fn
    diffusers.utils.is_transformers_version = _true_fn
    diffusers.utils.logging = SimpleNamespace(get_logger=lambda *_: MagicMock())
    diffusers.utils.replace_example_docstring = _decorator_factory
    diffusers.utils.scale_lora_layers = lambda *_, **__: None
    diffusers.utils.unscale_lora_layers = lambda *_, **__: None
    diffusers.utils.torch_utils = types.ModuleType("diffusers.utils.torch_utils")
    diffusers.utils.torch_utils.randn_tensor = MagicMock()
    diffusers.utils.torch_utils.is_compiled_module = lambda *_: False
    diffusers.utils.export_utils = types.ModuleType("diffusers.utils.export_utils")
    diffusers.utils.export_utils.export_to_gif = MagicMock()
    diffusers.configuration_utils = types.ModuleType("diffusers.configuration_utils")

    class _DummyConfigMixin:
        pass

    diffusers.configuration_utils.ConfigMixin = _DummyConfigMixin
    diffusers.configuration_utils.register_to_config = _decorator_factory
    sys.modules["diffusers"] = diffusers
    sys.modules["diffusers.optimization"] = diffusers.optimization
    sys.modules["diffusers.pipelines"] = diffusers.pipelines
    sys.modules["diffusers.pipelines.flux"] = diffusers.pipelines.flux
    sys.modules["diffusers.pipelines.flux.pipeline_flux"] = diffusers.pipelines.flux.pipeline_flux
    sys.modules["diffusers.image_processor"] = diffusers.image_processor
    sys.modules["diffusers.loaders"] = diffusers.loaders
    sys.modules["diffusers.loaders.lora_base"] = diffusers.loaders.lora_base
    sys.modules["diffusers.loaders.lora_conversion_utils"] = diffusers.loaders.lora_conversion_utils
    sys.modules["diffusers.models"] = diffusers.models
    sys.modules["diffusers.models.autoencoders"] = diffusers.models.autoencoders
    sys.modules["diffusers.models.autoencoders.vae"] = diffusers.models.autoencoders.vae
    sys.modules["diffusers.models.controlnets"] = diffusers.models.controlnets
    sys.modules["diffusers.models.controlnets.controlnet_flux"] = diffusers.models.controlnets.controlnet_flux
    sys.modules["diffusers.models.lora"] = diffusers.models.lora
    sys.modules["diffusers.models.transformers"] = diffusers.models.transformers
    sys.modules["diffusers.pipelines.pipeline_utils"] = diffusers.pipelines.pipeline_utils
    sys.modules["diffusers.pipelines.flux.pipeline_output"] = diffusers.pipelines.flux.pipeline_output
    sys.modules["diffusers.schedulers"] = diffusers.schedulers
    sys.modules["diffusers.schedulers.scheduling_utils"] = diffusers.schedulers.scheduling_utils
    sys.modules["diffusers.utils"] = diffusers.utils
    sys.modules["diffusers.utils.torch_utils"] = diffusers.utils.torch_utils
    sys.modules["diffusers.utils.export_utils"] = diffusers.utils.export_utils
    sys.modules["diffusers.configuration_utils"] = diffusers.configuration_utils

if "peft" not in sys.modules:
    peft = types.ModuleType("peft")
    peft.LoraConfig = MagicMock()
    sys.modules["peft"] = peft

if "torchvision" not in sys.modules:
    torchvision = types.ModuleType("torchvision")
    torchvision.transforms = MagicMock()
    torchvision.io = types.ModuleType("torchvision.io")
    torchvision.io.video_reader = types.ModuleType("torchvision.io.video_reader")
    torchvision.io.video_reader.VideoReader = MagicMock()
    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = torchvision.transforms
    sys.modules["torchvision.io"] = torchvision.io
    sys.modules["torchvision.io.video_reader"] = torchvision.io.video_reader

if "transformers" not in sys.modules:
    transformers = types.ModuleType("transformers")
    transformers.utils = types.ModuleType("transformers.utils")
    transformers.utils.ContextManagers = MagicMock()
    transformers.integrations = types.ModuleType("transformers.integrations")
    transformers.integrations.HfDeepSpeedConfig = MagicMock()
    transformers.integrations.deepspeed = types.ModuleType("transformers.integrations.deepspeed")
    transformers.integrations.deepspeed.is_deepspeed_zero3_enabled = lambda: False
    transformers.integrations.deepspeed.set_hf_deepspeed_config = lambda *_, **__: None
    transformers.integrations.deepspeed.unset_hf_deepspeed_config = lambda *_, **__: None
    transformers.integrations.deepspeed._hf_deepspeed_config_weak_ref = None
    transformers.CLIPImageProcessor = MagicMock()
    transformers.CLIPTextModel = MagicMock()
    transformers.CLIPTokenizer = MagicMock()
    transformers.CLIPVisionModelWithProjection = MagicMock()
    transformers.T5EncoderModel = MagicMock()
    transformers.T5TokenizerFast = MagicMock()
    sys.modules["transformers"] = transformers
    sys.modules["transformers.utils"] = transformers.utils
    sys.modules["transformers.integrations"] = transformers.integrations
    sys.modules["transformers.integrations.deepspeed"] = transformers.integrations.deepspeed

if "safetensors" not in sys.modules:
    safetensors = types.ModuleType("safetensors")
    safetensors.torch = types.ModuleType("safetensors.torch")
    safetensors.torch.load_file = MagicMock()
    safetensors.torch.save_file = MagicMock()
    sys.modules["safetensors"] = safetensors
    sys.modules["safetensors.torch"] = safetensors.torch

if "accelerate" not in sys.modules:
    accelerate = types.ModuleType("accelerate")
    accelerate.state = types.ModuleType("accelerate.state")
    accelerate.state.AcceleratorState = MagicMock()
    accelerate.logging = types.ModuleType("accelerate.logging")
    accelerate.logging.get_logger = lambda *_: MagicMock()
    sys.modules["accelerate"] = accelerate
    sys.modules["accelerate.state"] = accelerate.state
    sys.modules["accelerate.logging"] = accelerate.logging

if "huggingface_hub" not in sys.modules:
    huggingface_hub = types.ModuleType("huggingface_hub")
    huggingface_hub.utils = types.ModuleType("huggingface_hub.utils")
    huggingface_hub.utils.validate_hf_hub_args = MagicMock()
    sys.modules["huggingface_hub"] = huggingface_hub
    sys.modules["huggingface_hub.utils"] = huggingface_hub.utils

from simpletuner.helpers.data_backend.factory import FactoryRegistry, run_distillation_cache_generation
from simpletuner.helpers.distillation.common import DistillationBase
from simpletuner.helpers.distillation.registry import DistillationRegistry


class TestDistillationRegistry(unittest.TestCase):
    def test_registry_tracks_metadata(self):
        class DummyDistillation(DistillationBase):
            pass

        DistillationRegistry.register("unit-test-distiller", DummyDistillation, requires_distillation_cache=True)

        self.assertIs(DistillationRegistry.get("unit-test-distiller"), DummyDistillation)
        metadata = DistillationRegistry.get_metadata("unit-test-distiller")
        self.assertTrue(metadata["requires_distillation_cache"])


class TestDistillationCacheFactory(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)
        self.accelerator = MagicMock()
        self.accelerator.is_main_process = True
        self.accelerator.is_local_main_process = True
        self.accelerator.device = "cpu"
        self.accelerator.wait_for_everyone = MagicMock()
        self.model = MagicMock()
        self.model.requires_conditioning_dataset.return_value = False
        self.model.requires_conditioning_latents.return_value = False
        self.model.requires_conditioning_image_embeds.return_value = False

        self.args = SimpleNamespace(
            model_type="test",
            model_family="test",
            cache_dir=self.temp_dir,
            cache_dir_text=os.path.join(self.temp_dir, "text"),
            cache_dir_vae=os.path.join(self.temp_dir, "vae"),
            compress_disk_cache=False,
            output_dir=self.temp_dir,
        )

    def test_configures_distillation_cache_backend(self):
        backend_config = [
            {
                "id": "ode-cache-1",
                "type": "local",
                "dataset_type": "distillation_cache",
                "cache_dir": os.path.join(self.temp_dir, "ode"),
                "distillation_type": "self_forcing",
            }
        ]

        with patch("simpletuner.helpers.data_backend.factory.StateTracker") as mock_state_tracker:
            mock_state_tracker.get_data_backend_config.return_value = {}
            mock_state_tracker.get_webhook_handler.return_value = None
            mock_state_tracker.get_args.return_value = self.args
            mock_state_tracker.is_sdxl_refiner.return_value = False

            factory = FactoryRegistry(
                args=self.args,
                accelerator=self.accelerator,
                text_encoders=[],
                tokenizers=[],
                model=self.model,
            )
            factory.configure_distillation_cache_backends(backend_config)

            self.assertIn("ode-cache-1", factory.distillation_cache_backends)
            backend_entry = factory.distillation_cache_backends["ode-cache-1"]
            self.assertEqual(backend_entry["dataset_type"], "distillation_cache")
            self.assertEqual(backend_entry["distillation_type"], "self_forcing")
            self.assertTrue(hasattr(backend_entry["distillation_cache"], "write_tensor"))


class TestRunDistillationCacheGeneration(unittest.TestCase):
    def test_runs_provider_for_matching_cache(self):
        distiller = MagicMock(spec=DistillationBase)
        distiller.requires_distillation_cache.return_value = True
        distiller.get_required_distillation_cache_type.return_value = "self_forcing"
        provider = MagicMock()
        distiller.get_ode_generator_provider.return_value = provider

        cache = MagicMock()
        backend_map = {
            "ode-cache-1": {
                "distillation_type": "self_forcing",
                "distillation_cache": cache,
                "config": {"distillation_type": "self_forcing"},
            }
        }

        with patch("simpletuner.helpers.data_backend.factory.StateTracker") as mock_state_tracker:
            mock_state_tracker.get_data_backends.return_value = backend_map
            run_distillation_cache_generation(distiller)

        provider.generate.assert_called_once_with(cache, backend_config={"distillation_type": "self_forcing"})

    def test_raises_when_required_cache_missing(self):
        distiller = MagicMock(spec=DistillationBase)
        distiller.requires_distillation_cache.return_value = True
        distiller.get_required_distillation_cache_type.return_value = "expected"
        distiller.get_ode_generator_provider.return_value = MagicMock()

        backend_map = {
            "ode-cache-1": {
                "distillation_type": "other",
                "distillation_cache": MagicMock(),
                "config": {},
            }
        }

        with patch("simpletuner.helpers.data_backend.factory.StateTracker") as mock_state_tracker:
            mock_state_tracker.get_data_backends.return_value = backend_map
            with self.assertRaises(ValueError):
                run_distillation_cache_generation(distiller)
