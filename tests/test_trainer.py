# test_trainer.py

import importlib.machinery as machinery
import importlib.util as _importlib_util
import json
import shutil
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import torch

from simpletuner.helpers.training.state_tracker import StateTracker

_OPTIONAL_MODULES = {
    "wandb",
    "torchao",
    "torchao.optim",
    "torchao.quantization",
    "torchao.quantization.quant_primitives",
    "torchao.dtypes",
    "torchao.dtypes.floatx",
    "torchao.dtypes.floatx.float8_layout",
    "torchao.dtypes.uintx",
    "torchao.dtypes.uintx.uint4_layout",
    "torchao.dtypes.uintx.uintx_layout",
    "optimi",
}


_original_find_spec = _importlib_util.find_spec


def _patched_find_spec(name, *args, **kwargs):
    if name in _OPTIONAL_MODULES:
        return None
    return _original_find_spec(name, *args, **kwargs)


_importlib_util.find_spec = _patched_find_spec

try:
    from accelerate import FullyShardedDataParallelPlugin
except Exception:  # pragma: no cover - lightweight stub when accelerate is unavailable

    class FullyShardedDataParallelPlugin:  # type: ignore[override]
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
            self.fsdp_version = kwargs.get("fsdp_version")
            self.reshard_after_forward = kwargs.get("reshard_after_forward")
            self.cpu_ram_efficient_loading = kwargs.get("cpu_ram_efficient_loading")
            self.state_dict_type = kwargs.get("state_dict_type")
            self.auto_wrap_policy = kwargs.get("auto_wrap_policy")
            self.transformer_cls_names_to_wrap = kwargs.get("transformer_cls_names_to_wrap")

        def set_state_dict_type(self, value):
            self.state_dict_type = value


def _ensure_torchao_stub():
    try:  # Prefer the real library if available in the environment.
        import importlib

        importlib.import_module("torchao.optim")
        return
    except Exception:
        if "torchao.optim" in sys.modules:
            return
    torchao_module = types.ModuleType("torchao")
    torchao_module.__spec__ = machinery.ModuleSpec("torchao", loader=None)
    optim_module = types.ModuleType("torchao.optim")
    optim_module.__spec__ = machinery.ModuleSpec("torchao.optim", loader=None)
    dummy_class = type("DummyOptimizer", (), {})

    optim_module.AdamFp8 = dummy_class
    optim_module.AdamW4bit = dummy_class
    optim_module.AdamW8bit = dummy_class
    optim_module.AdamWFp8 = dummy_class
    optim_module.CPUOffloadOptimizer = dummy_class

    quant_module = types.ModuleType("torchao.quantization")
    quant_module.__spec__ = machinery.ModuleSpec("torchao.quantization", loader=None)
    quant_primitives_module = types.ModuleType("torchao.quantization.quant_primitives")
    quant_primitives_module.__spec__ = machinery.ModuleSpec("torchao.quantization.quant_primitives", loader=None)

    class _MappingType:
        pass

    def _quantize_(tensor, *_, **__):
        return tensor

    quant_primitives_module.MappingType = _MappingType
    quant_module.quant_primitives = quant_primitives_module
    quant_module.quantize_ = _quantize_

    dtypes_module = types.ModuleType("torchao.dtypes")
    dtypes_module.__spec__ = machinery.ModuleSpec("torchao.dtypes", loader=None)

    class _NF4Tensor:
        pass

    dtypes_module.NF4Tensor = _NF4Tensor

    torchao_module.optim = optim_module
    torchao_module.quantization = quant_module
    torchao_module.dtypes = dtypes_module
    sys.modules["torchao"] = torchao_module
    sys.modules["torchao.optim"] = optim_module
    sys.modules["torchao.quantization"] = quant_module
    sys.modules["torchao.quantization.quant_primitives"] = quant_primitives_module
    sys.modules["torchao.dtypes"] = dtypes_module


def _ensure_optimi_stub():
    if "optimi" in sys.modules:
        return
    optimi_module = types.ModuleType("optimi")
    optimi_module.__spec__ = machinery.ModuleSpec("optimi", loader=None)
    for cls_name in [
        "StableAdamW",
        "AdamW",
        "Lion",
        "RAdam",
        "Ranger",
        "Adan",
        "Adam",
        "SGD",
    ]:
        setattr(optimi_module, cls_name, type(cls_name, (), {}))

    def _prepare_for_gradient_release(*_args, **_kwargs):
        return None

    optimi_module.prepare_for_gradient_release = _prepare_for_gradient_release
    sys.modules["optimi"] = optimi_module


def _ensure_fastapi_stub():
    try:
        import fastapi  # noqa: F401
        import fastapi.middleware.cors  # noqa: F401
        import fastapi.responses  # noqa: F401

        return
    except Exception:
        pass
    if "fastapi" in sys.modules:
        return

    fastapi_module = types.ModuleType("fastapi")
    fastapi_module.__spec__ = machinery.ModuleSpec("fastapi", loader=None)

    class FastAPI:
        def __init__(self, *_, **__):
            self.state = types.SimpleNamespace()

        def add_middleware(self, *_, **__):
            return None

        def mount(self, *_, **__):
            return None

    class APIRouter:
        pass

    class HTTPException(Exception):
        pass

    fastapi_module.FastAPI = FastAPI
    fastapi_module.APIRouter = APIRouter
    fastapi_module.HTTPException = HTTPException

    sys.modules["fastapi"] = fastapi_module

    middleware_module = types.ModuleType("fastapi.middleware")
    middleware_module.__spec__ = machinery.ModuleSpec("fastapi.middleware", loader=None)
    sys.modules["fastapi.middleware"] = middleware_module

    cors_module = types.ModuleType("fastapi.middleware.cors")
    cors_module.__spec__ = machinery.ModuleSpec("fastapi.middleware.cors", loader=None)

    class CORSMiddleware:
        def __init__(self, *_, **__):
            pass

    cors_module.CORSMiddleware = CORSMiddleware
    sys.modules["fastapi.middleware.cors"] = cors_module

    responses_module = types.ModuleType("fastapi.responses")
    responses_module.__spec__ = machinery.ModuleSpec("fastapi.responses", loader=None)

    class FileResponse:
        def __init__(self, *_, **__):
            pass

    responses_module.FileResponse = FileResponse
    sys.modules["fastapi.responses"] = responses_module

    staticfiles_module = types.ModuleType("fastapi.staticfiles")

    class StaticFiles:
        def __init__(self, *_, **__):
            pass

    staticfiles_module.StaticFiles = StaticFiles
    sys.modules["fastapi.staticfiles"] = staticfiles_module


def _ensure_toml_stub():
    if "toml" in sys.modules:
        return
    toml_module = types.ModuleType("toml")

    class TomlDecodeError(Exception):
        pass

    def _load(*_args, **_kwargs):
        return {}

    toml_module.TomlDecodeError = TomlDecodeError
    toml_module.load = _load
    sys.modules["toml"] = toml_module


def _ensure_peft_stub():
    if "peft" in sys.modules:
        return
    peft_module = types.ModuleType("peft")

    class LoraConfig:
        pass

    def _noop(*_args, **_kwargs):
        return None

    peft_module.LoraConfig = LoraConfig
    peft_module.inject_adapter_in_model = _noop
    peft_module.set_peft_model_state_dict = _noop
    peft_module.__spec__ = machinery.ModuleSpec("peft", loader=None)
    sys.modules["peft"] = peft_module

    utils_module = types.ModuleType("peft.utils")

    def _get_state_dict(*_args, **_kwargs):
        return {}

    utils_module.get_peft_model_state_dict = _get_state_dict
    utils_module.set_peft_model_state_dict = _noop
    utils_module.__spec__ = machinery.ModuleSpec("peft.utils", loader=None)
    sys.modules["peft.utils"] = utils_module

    other_module = types.ModuleType("peft.utils.other")

    def _transpose(value):
        return value

    other_module.transpose = _transpose
    other_module.__spec__ = machinery.ModuleSpec("peft.utils.other", loader=None)
    sys.modules["peft.utils.other"] = other_module

    tuners_module = types.ModuleType("peft.tuners")
    tuners_module.__spec__ = machinery.ModuleSpec("peft.tuners", loader=None)
    sys.modules["peft.tuners"] = tuners_module

    lora_layer_module = types.ModuleType("peft.tuners.lora.layer")

    class LoraLayer:
        pass

    lora_layer_module.LoraLayer = LoraLayer
    lora_layer_module.__spec__ = machinery.ModuleSpec("peft.tuners.lora.layer", loader=None)
    sys.modules["peft.tuners.lora.layer"] = lora_layer_module

    tuners_utils_module = types.ModuleType("peft.tuners.tuners_utils")

    class BaseTunerLayer:
        pass

    def _check_adapters_to_merge(*_args, **_kwargs):
        return []

    tuners_utils_module.BaseTunerLayer = BaseTunerLayer
    tuners_utils_module.check_adapters_to_merge = _check_adapters_to_merge
    tuners_utils_module.__spec__ = machinery.ModuleSpec("peft.tuners.tuners_utils", loader=None)
    sys.modules["peft.tuners.tuners_utils"] = tuners_utils_module

    import_utils_module = types.ModuleType("peft.import_utils")

    def _is_quanto_available():
        return False

    import_utils_module.is_quanto_available = _is_quanto_available
    import_utils_module.__spec__ = machinery.ModuleSpec("peft.import_utils", loader=None)
    sys.modules["peft.import_utils"] = import_utils_module


def _ensure_torchvision_stub():
    if "torchvision" in sys.modules:
        return

    torchvision_module = types.ModuleType("torchvision")
    torchvision_module.__spec__ = machinery.ModuleSpec("torchvision", loader=None)

    class _DummyTransform:
        def __call__(self, value):
            return value

    def _compose(*_args, **_kwargs):
        return _DummyTransform()

    def _to_tensor(value):
        return value

    def _normalize(*_args, **_kwargs):
        return _DummyTransform()

    transforms_module = types.ModuleType("torchvision.transforms")
    transforms_module.__spec__ = machinery.ModuleSpec("torchvision.transforms", loader=None)
    transforms_module.Compose = _compose
    transforms_module.ToTensor = lambda *_args, **_kwargs: _DummyTransform()
    transforms_module.Normalize = _normalize
    transforms_module.functional = types.SimpleNamespace(to_tensor=_to_tensor)

    class _InterpolationMode:
        NEAREST = "nearest"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"
        LANCZOS = "lanczos"
        HAMMING = "hamming"
        BOX = "box"

    transforms_module.InterpolationMode = _InterpolationMode

    torchvision_module.transforms = transforms_module
    sys.modules["torchvision"] = torchvision_module
    sys.modules["torchvision.transforms"] = transforms_module


def _ensure_accelerate_stub():
    existing = sys.modules.get("accelerate")
    if existing is not None and not hasattr(existing, "__path__"):
        del sys.modules["accelerate"]
        for name in list(sys.modules):
            if name.startswith("accelerate."):
                del sys.modules[name]
    try:
        import accelerate  # type: ignore  # noqa: F401

        return
    except Exception:
        pass

    accelerate_module = types.ModuleType("accelerate")
    accelerate_spec = machinery.ModuleSpec("accelerate", loader=None)
    accelerate_spec.submodule_search_locations = ["accelerate_stub"]
    accelerate_module.__spec__ = accelerate_spec
    accelerate_module.__path__ = ["accelerate_stub"]

    class _Accelerator:
        def __init__(self, *_, **__):
            self.is_main_process = True
            self.is_local_main_process = True
            self.device = "cpu"
            self.state = types.SimpleNamespace(deepspeed_plugin=None)

        def prepare(self, *objects):
            return objects

        def wait_for_everyone(self):
            return None

        def accumulate(self, *_args, **_kwargs):
            class _Context:
                def __enter__(self):
                    return None

                def __exit__(self, exc_type, exc, tb):
                    return False

            return _Context()

    class _DeepSpeedPlugin:
        pass

    class _InitProcessGroupKwargs:
        pass

    class _FSDPPlugin:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def set_state_dict_type(self, value):
            self.state_dict_type = value

    accelerate_module.Accelerator = _Accelerator
    accelerate_module.DeepSpeedPlugin = _DeepSpeedPlugin
    accelerate_module.InitProcessGroupKwargs = _InitProcessGroupKwargs
    accelerate_module.FullyShardedDataParallelPlugin = _FSDPPlugin

    accelerate_utils = types.ModuleType("accelerate.utils")
    accelerate_utils_spec = machinery.ModuleSpec("accelerate.utils", loader=None)
    accelerate_utils_spec.submodule_search_locations = ["accelerate_stub/utils"]
    accelerate_utils.__spec__ = accelerate_utils_spec
    accelerate_utils.__path__ = ["accelerate_stub/utils"]
    accelerate_utils.DistributedType = types.SimpleNamespace()
    accelerate_utils.DynamoBackend = object
    accelerate_utils.ParallelismConfig = object
    accelerate_utils.TorchContextParallelConfig = object
    accelerate_utils.TorchDynamoPlugin = object
    accelerate_utils.ProjectConfiguration = type("ProjectConfiguration", (), {})
    accelerate_utils.set_seed = lambda *_, **__: None

    accelerate_state = types.ModuleType("accelerate.state")
    accelerate_state_spec = machinery.ModuleSpec("accelerate.state", loader=None)
    accelerate_state_spec.submodule_search_locations = ["accelerate_stub/state"]
    accelerate_state.__spec__ = accelerate_state_spec
    accelerate_state.__path__ = ["accelerate_stub/state"]
    accelerate_state.AcceleratorState = type("AcceleratorState", (), {})

    sys.modules["accelerate.utils"] = accelerate_utils
    sys.modules["accelerate.state"] = accelerate_state
    accelerate_module.utils = accelerate_utils
    accelerate_module.state = accelerate_state
    sys.modules["accelerate"] = accelerate_module


_ensure_torchao_stub()
_ensure_optimi_stub()
_ensure_fastapi_stub()
_ensure_toml_stub()
_ensure_peft_stub()
_ensure_torchvision_stub()
_ensure_accelerate_stub()


def _ensure_model_registry_stub():
    module_name = "simpletuner.helpers.models.registry"
    if module_name in sys.modules:
        return

    models_registry_module = types.ModuleType(module_name)
    models_registry_module.__spec__ = machinery.ModuleSpec(module_name, loader=None)

    class _StubModelRegistry:
        @staticmethod
        def model_families():
            return {}

        @staticmethod
        def get(_name):
            raise KeyError(_name)

    models_registry_module.ModelRegistry = _StubModelRegistry
    sys.modules[module_name] = models_registry_module


_ensure_model_registry_stub()


try:
    import transformers  # type: ignore

    if not hasattr(transformers, "AutoImageProcessor"):

        class _StubAutoImageProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):  # pragma: no cover - simple stub
                return cls()

            def __call__(self, *args, **kwargs):  # pragma: no cover - simple stub
                return {}

        transformers.AutoImageProcessor = _StubAutoImageProcessor  # type: ignore[attr-defined]

    if not hasattr(transformers, "PreTrainedModel"):

        class _StubPreTrainedModel:  # pragma: no cover - simple stub
            pass

        transformers.PreTrainedModel = _StubPreTrainedModel  # type: ignore[attr-defined]

    if not hasattr(transformers, "CLIPImageProcessor"):
        transformers.CLIPImageProcessor = _StubAutoImageProcessor  # type: ignore[attr-defined]

    if not hasattr(transformers, "CLIPVisionModelWithProjection"):

        class _StubCLIPVisionModelWithProjection:  # pragma: no cover - simple stub
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        transformers.CLIPVisionModelWithProjection = _StubCLIPVisionModelWithProjection  # type: ignore[attr-defined]

    if not hasattr(transformers, "SiglipImageProcessor"):
        transformers.SiglipImageProcessor = _StubAutoImageProcessor  # type: ignore[attr-defined]

    if not hasattr(transformers, "SiglipVisionModel"):

        class _StubSiglipVisionModel:  # pragma: no cover - simple stub
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

        transformers.SiglipVisionModel = _StubSiglipVisionModel  # type: ignore[attr-defined]
except Exception:
    pass


def _ensure_models_common_stub():
    try:
        import simpletuner.helpers.models.common  # noqa: F401

        return
    except Exception:
        pass
    module_name = "simpletuner.helpers.models.common"
    if module_name in sys.modules:
        return

    common_module = types.ModuleType(module_name)
    common_module.__spec__ = machinery.ModuleSpec(module_name, loader=None)

    class _BaseModelFoundation:
        def __init__(self, *args, **kwargs):  # pragma: no cover - stub
            pass

    class ImageModelFoundation(_BaseModelFoundation):
        pass

    class VideoModelFoundation(_BaseModelFoundation):
        pass

    class ModelFoundation(_BaseModelFoundation):
        pass

    class AudioModelFoundation(_BaseModelFoundation):
        pass

    class _PredictionTypes:
        EPSILON = "epsilon"
        SAMPLE = "sample"
        V_PREDICTION = "v_prediction"
        FLOW_MATCHING = "flow_matching"

        @staticmethod
        def from_str(_label):
            return _PredictionTypes.EPSILON

    class _PipelineTypes:
        IMG2IMG = "img2img"
        TEXT2IMG = "text2img"
        IMG2VIDEO = "img2video"
        CONTROLNET = "controlnet"
        CONTROL = "control"

    class _ModelTypes:
        UNET = "unet"
        TRANSFORMER = "transformer"
        VAE = "vae"
        TEXT_ENCODER = "text_encoder"

    class _TextEmbedCacheKey:
        CAPTION = "caption"
        DATASET_AND_FILENAME = "dataset_and_filename"

    common_module.ImageModelFoundation = ImageModelFoundation
    common_module.VideoModelFoundation = VideoModelFoundation
    common_module.ModelFoundation = ModelFoundation
    common_module.AudioModelFoundation = AudioModelFoundation
    common_module.PredictionTypes = _PredictionTypes
    common_module.PipelineTypes = _PipelineTypes
    common_module.ModelTypes = _ModelTypes
    common_module.TextEmbedCacheKey = _TextEmbedCacheKey

    sys.modules[module_name] = common_module


_ensure_models_common_stub()

wandb_module = types.ModuleType("wandb")
wandb_module.__spec__ = machinery.ModuleSpec("wandb", loader=None)
wandb_module.init = lambda *_, **__: None
wandb_module.login = lambda *_, **__: None
wandb_module.finish = lambda *_, **__: None
wandb_module.log = lambda *_, **__: None
wandb_module.config = {}
sys.modules.setdefault("wandb", wandb_module)

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.training.attention_backend import AttentionBackendMode
from simpletuner.helpers.training.trainer import Trainer

# Import test configuration to suppress logging/warnings
try:
    from . import test_config
except ImportError:
    # Fallback for when running tests individually
    import test_config


class TestTrainer(unittest.TestCase):
    def _build_trainer_for_grad_logging(self, grad_clip_method: str, use_deepspeed: bool, grad_value):
        trainer = object.__new__(Trainer)
        trainer.config = SimpleNamespace(
            grad_clip_method=grad_clip_method,
            use_deepspeed_optimizer=use_deepspeed,
        )
        trainer.grad_norm = grad_value
        return trainer

    @patch("simpletuner.helpers.training.trainer.load_config")
    @patch("simpletuner.helpers.training.trainer.safety_check")
    @patch("simpletuner.helpers.training.state_tracker.StateTracker")
    @patch(
        "simpletuner.helpers.training.state_tracker.StateTracker.set_model_family",
        return_value=True,
    )
    @patch("torch.set_num_threads")
    @patch("simpletuner.helpers.training.trainer.Accelerator")
    @patch(
        "simpletuner.helpers.training.trainer.Trainer.parse_arguments",
        return_value=Mock(),
    )
    @patch("simpletuner.helpers.training.trainer.Trainer._misc_init", return_value=Mock())
    def test_config_to_obj(
        self,
        mock_misc_init,
        mock_parse_args,
        mock_accelerator,
        mock_set_num_threads,
        mock_set_model_family,
        mock_state_tracker,
        mock_safety_check,
        mock_load_config,
    ):
        trainer = Trainer()
        trainer.model = Mock()
        config_dict = {"a": 1, "b": 2}
        config_obj = trainer._config_to_obj(config_dict)
        self.assertEqual(config_obj.a, 1)
        self.assertEqual(config_obj.b, 2)

        config_none = trainer._config_to_obj(None)
        self.assertIsNone(config_none)

    @patch("simpletuner.helpers.training.trainer.Trainer._misc_init", return_value=Mock())
    @patch(
        "simpletuner.helpers.training.trainer.Trainer.parse_arguments",
        return_value=Mock(),
    )
    @patch("simpletuner.helpers.training.trainer.set_seed")
    def test_init_seed_with_value(self, mock_set_seed, mock_parse_args, mock_misc_init):
        trainer = Trainer()
        trainer.model = Mock()
        trainer.config = Mock(seed=42, seed_for_each_device=False)
        trainer.init_seed()
        mock_set_seed.assert_called_with(42, False)

    def test_prepare_batch_requires_caption_aware_distiller(self):
        trainer = object.__new__(Trainer)
        trainer.model = Mock()
        trainer.model.prepare_batch = Mock()
        trainer.state = {}
        trainer.distiller = None
        caption_batch = {"dataset_type": DatasetType.CAPTION, "data_backend_id": "captions", "captions": ["hello"]}
        with self.assertRaises(ValueError):
            trainer.prepare_batch(caption_batch)

        trainer.distiller = Mock()
        trainer.distiller.consumes_caption_batches.return_value = False
        with self.assertRaises(ValueError):
            trainer.prepare_batch(caption_batch)

    def test_prepare_batch_routes_captions_to_distiller(self):
        trainer = object.__new__(Trainer)
        trainer.model = Mock()
        trainer.model.prepare_batch = Mock()
        trainer.state = {}
        distiller = Mock()
        distiller.consumes_caption_batches.return_value = True
        prepared = {"latents": torch.zeros(1, 4, 4, 4), "timesteps": torch.zeros(1)}
        distiller.prepare_caption_batch.return_value = prepared
        trainer.distiller = distiller

        batch = {"dataset_type": DatasetType.CAPTION, "data_backend_id": "captions", "captions": ["hello world"]}
        result = trainer.prepare_batch(batch)
        self.assertIs(result, prepared)
        trainer.model.prepare_batch.assert_not_called()
        distiller.prepare_batch.assert_not_called()
        distiller.prepare_caption_batch.assert_called_once_with(batch, trainer.model, trainer.state)

    def test_run_trainer_job_aborts_promptly(self):
        from simpletuner.helpers.training import trainer as trainer_module

        class DummyTrainer:
            last_instance = None

            def __init__(self, config=None, job_id=None):
                self.config = config
                self.job_id = job_id
                self.should_abort = False
                self._external_abort_checker = None
                self.abort_called = False
                DummyTrainer.last_instance = self

            def run(self):
                deadline = time.time() + 1.0
                while not self.should_abort and time.time() < deadline:
                    time.sleep(0.01)
                if not self.should_abort:
                    raise AssertionError("Trainer run did not observe abort signal")

            def abort(self):
                self.abort_called = True
                self.should_abort = True

        with patch.object(trainer_module, "Trainer", DummyTrainer):
            result = trainer_module.run_trainer_job(
                {
                    "should_abort": lambda: True,
                    "__job_id__": "unit-test",
                }
            )

        self.assertEqual(result["status"], "completed")
        instance = DummyTrainer.last_instance
        self.assertIsNotNone(instance)
        self.assertTrue(instance.abort_called)
        self.assertTrue(instance.should_abort)

    def test_run_trainer_job_honours_single_device_selection(self):
        from simpletuner.helpers.training import trainer as trainer_module

        captured = {}

        class DummyStdout:
            def readline(self):
                return ""

            def close(self):
                captured["stdout_closed"] = True

        class DummyProcess:
            def __init__(self):
                self.stdout = DummyStdout()
                self.pid = 1234

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

            def terminate(self):
                captured["terminated"] = True

            def kill(self):
                captured["killed"] = True

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs.get("env", {})
            captured["kwargs"] = kwargs
            return DummyProcess()

        with patch("subprocess.Popen", side_effect=fake_popen):
            result = trainer_module.run_trainer_job(
                {
                    "accelerate_visible_devices": [1],
                    "--num_processes": 1,
                }
            )

            self.assertEqual(result, 0)
        self.assertIn("cmd", captured)
        self.assertEqual(captured["cmd"][0], "accelerate")
        self.assertEqual(captured["env"].get("CUDA_VISIBLE_DEVICES"), "1")
        self.assertFalse(any("--accelerate_visible_devices" in arg for arg in captured["cmd"]))

    def test_accelerate_manual_triggers_are_relayed(self):
        from simpletuner.helpers.training import trainer as trainer_module

        captured = {}
        trigger_calls = {"count": 0}

        def manual_validation_consumer():
            trigger_calls["count"] += 1
            return trigger_calls["count"] == 1

        class DummyStdout:
            def readline(self):
                return ""

            def close(self):
                captured["stdout_closed"] = True

        class DummyProcess:
            def __init__(self):
                self.stdout = DummyStdout()
                self.pid = 4321
                self._poll_calls = 0

            def poll(self):
                self._poll_calls += 1
                return None if self._poll_calls < 2 else 0

            def wait(self, timeout=None):
                return 0

            def terminate(self):
                captured["terminated"] = True

            def kill(self):
                captured["killed"] = True

        def fake_popen(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["env"] = kwargs.get("env", {})
            return DummyProcess()

        removed_paths = []

        def fake_rmtree(path, ignore_errors=True):
            removed_paths.append(path)

        with (
            patch("subprocess.Popen", side_effect=fake_popen),
            patch("simpletuner.helpers.training.trainer.shutil.rmtree", side_effect=fake_rmtree),
        ):
            result = trainer_module.run_trainer_job(
                {
                    "accelerate_visible_devices": [0],
                    "consume_manual_validation_request": manual_validation_consumer,
                }
            )

        self.assertEqual(result, 0)
        signal_file = captured.get("env", {}).get("SIMPLETUNER_ACCELERATE_SIGNAL_FILE")
        self.assertIsNotNone(signal_file)
        with open(signal_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.assertGreaterEqual(payload.get("manual_validation", 0), 1)
        self.assertTrue(removed_paths)
        for path in removed_paths:
            shutil.rmtree(path, ignore_errors=True)

    def test_accelerate_failure_summary_highlights_oom(self):
        from simpletuner.helpers.training import trainer as trainer_module

        lines = [
            "2025-11-04 16:21:12,847 - SimpleTuner - INFO - starting...",
            "2025-11-04 16:21:12,846 [ERROR] Error encoding images ['16.jpg']: CUDA out of memory. Tried to allocate 256.00 MiB.",
            "RuntimeError: Accelerate launch exited with status 1",
        ]

        summary, excerpt = trainer_module._summarize_accelerate_failure(1, lines)

        self.assertIn("CUDA out of memory", summary)
        self.assertIsNotNone(excerpt)
        if excerpt is None:  # pragma: no cover - defensive guard
            self.fail("Expected excerpt to be populated")
        self.assertIn("CUDA out of memory", excerpt)

    def test_accelerate_failure_summary_highlights_signal_exit(self):
        from simpletuner.helpers.training import trainer as trainer_module

        lines = [
            "2025-11-04 16:21:12,847 - SimpleTuner - INFO - starting...",
        ]

        summary, excerpt = trainer_module._summarize_accelerate_failure(-9, lines)

        self.assertIn("Accelerate launch exited with status -9", summary)
        self.assertTrue("SIGKILL" in summary or "signal 9" in summary)
        self.assertIsNotNone(excerpt)

    def test_accelerate_failure_summary_highlights_signal_received_by_pid(self):
        """Test parsing of accelerate's 'Signal N (SIGNAME) received by PID' format."""
        from simpletuner.helpers.training import trainer as trainer_module

        lines = [
            "2025-11-04 16:21:12,847 - SimpleTuner - INFO - starting...",
            "traceback : Signal 9 (SIGKILL) received by PID 2963915",
            "RuntimeError: Some wrapper error",
        ]

        summary, excerpt = trainer_module._summarize_accelerate_failure(1, lines)

        # Should extract the SIGKILL message even though exit code is 1
        self.assertIn("SIGKILL", summary)
        self.assertIsNotNone(excerpt)

    def test_launch_with_accelerate_fallback_imports(self):
        from simpletuner.helpers.training import trainer as trainer_module

        helper = getattr(trainer_module, "_summarize_accelerate_failure", None)
        self.assertIsNotNone(helper)

        returned = helper(1, ["foo", "bar"])
        self.assertIsInstance(returned, tuple)

    @patch("simpletuner.helpers.training.trainer.Trainer._misc_init", return_value=Mock())
    @patch(
        "simpletuner.helpers.training.trainer.Trainer.parse_arguments",
        return_value=Mock(),
    )
    @patch("simpletuner.helpers.training.trainer.set_seed")
    def test_init_seed_none(self, mock_set_seed, mock_parse_args, mock_misc_init):
        trainer = Trainer()
        trainer.model = Mock()
        trainer.config = Mock(seed=None, seed_for_each_device=False)
        trainer.init_seed()
        mock_set_seed.assert_not_called()

    def test_update_grad_metrics_skips_absmax_with_deepspeed(self):
        trainer = self._build_trainer_for_grad_logging(
            grad_clip_method="value",
            use_deepspeed=True,
            grad_value=torch.tensor(1.2),
        )
        logs = {}
        trainer._update_grad_metrics(logs)
        self.assertNotIn("grad_absmax", logs)
        self.assertNotIn("grad_norm", logs)

    def test_update_grad_metrics_logs_absmax_without_deepspeed(self):
        trainer = self._build_trainer_for_grad_logging(
            grad_clip_method="value",
            use_deepspeed=False,
            grad_value=torch.tensor(1.2),
        )
        logs = {}
        trainer._update_grad_metrics(logs)
        self.assertIn("grad_absmax", logs)
        self.assertIs(logs["grad_absmax"], trainer.grad_norm)

    def test_update_grad_metrics_clones_norm_value_when_requested(self):
        trainer = self._build_trainer_for_grad_logging(
            grad_clip_method="norm",
            use_deepspeed=False,
            grad_value=torch.tensor(1.2),
        )
        logs = {}
        trainer._update_grad_metrics(logs, clone_norm_value=True)
        self.assertIn("grad_norm", logs)
        self.assertEqual(logs["grad_norm"], float(trainer.grad_norm.clone().detach()))

    def test_update_grad_metrics_requires_value_method_when_requested(self):
        trainer = self._build_trainer_for_grad_logging(
            grad_clip_method="clip",
            use_deepspeed=False,
            grad_value=torch.tensor(1.2),
        )
        logs = {}
        trainer._update_grad_metrics(logs, require_value_method=True)
        self.assertNotIn("grad_absmax", logs)

    def _build_trainer_for_progress_metrics(self, grad_clip_method: str, use_deepspeed: bool, grad_value):
        """Build a minimal trainer for testing _compose_training_progress_metrics."""
        from simpletuner.helpers.training.iteration_tracker import IterationTracker

        trainer = object.__new__(Trainer)
        trainer.config = SimpleNamespace(
            grad_clip_method=grad_clip_method,
            use_deepspeed_optimizer=use_deepspeed,
            total_batch_size=4,
        )
        trainer.grad_norm = grad_value
        trainer.state = {"global_step": 10}
        trainer.train_loss = 0.5
        trainer.lr = 0.0001
        trainer.iteration_tracker = IterationTracker()
        trainer.iteration_tracker.mark_start()
        return trainer

    def test_compose_training_progress_metrics_includes_grad_norm(self):
        """Test that _compose_training_progress_metrics includes grad_norm when using norm clipping."""
        trainer = self._build_trainer_for_progress_metrics(
            grad_clip_method="norm",
            use_deepspeed=False,
            grad_value=torch.tensor(1.5),
        )
        metrics = trainer._compose_training_progress_metrics(epoch=1)
        self.assertIn("grad_norm", metrics)
        self.assertEqual(metrics["grad_norm"], float(trainer.grad_norm.clone().detach()))
        self.assertNotIn("grad_absmax", metrics)

    def test_compose_training_progress_metrics_includes_grad_absmax(self):
        """Test that _compose_training_progress_metrics includes grad_absmax when using value clipping."""
        trainer = self._build_trainer_for_progress_metrics(
            grad_clip_method="value",
            use_deepspeed=False,
            grad_value=torch.tensor(2.3),
        )
        metrics = trainer._compose_training_progress_metrics(epoch=1)
        self.assertIn("grad_absmax", metrics)
        self.assertIs(metrics["grad_absmax"], trainer.grad_norm)
        self.assertNotIn("grad_norm", metrics)

    def test_compose_training_progress_metrics_excludes_grad_with_deepspeed(self):
        """Test that _compose_training_progress_metrics excludes grad_absmax when deepspeed is enabled."""
        trainer = self._build_trainer_for_progress_metrics(
            grad_clip_method="value",
            use_deepspeed=True,
            grad_value=torch.tensor(1.8),
        )
        metrics = trainer._compose_training_progress_metrics(epoch=1)
        self.assertNotIn("grad_absmax", metrics)
        self.assertNotIn("grad_norm", metrics)

    def test_compose_training_progress_metrics_no_grad_when_none(self):
        """Test that _compose_training_progress_metrics excludes grad metrics when grad_norm is None."""
        trainer = self._build_trainer_for_progress_metrics(
            grad_clip_method="norm",
            use_deepspeed=False,
            grad_value=None,
        )
        metrics = trainer._compose_training_progress_metrics(epoch=1)
        self.assertNotIn("grad_norm", metrics)
        self.assertNotIn("grad_absmax", metrics)

    def test_load_fsdp_plugin_maps_options(self):
        trainer = object.__new__(Trainer)
        trainer.config = SimpleNamespace(
            fsdp_enable=True,
            fsdp_version=2,
            fsdp_reshard_after_forward=True,
            fsdp_cpu_ram_efficient_loading=True,
            fsdp_state_dict_type="SHARDED_STATE_DICT",
            fsdp_auto_wrap_policy="transformer_based_wrap",
            fsdp_transformer_layer_cls_to_wrap=["TransformerBlock"],
            deepspeed_config=None,
        )

        plugin = trainer._load_fsdp_plugin()
        self.assertIsInstance(plugin, FullyShardedDataParallelPlugin)
        self.assertEqual(plugin.fsdp_version, 2)
        self.assertTrue(plugin.reshard_after_forward)
        self.assertTrue(plugin.cpu_ram_efficient_loading)
        self.assertEqual(plugin.state_dict_type, "sharded_state_dict")
        self.assertEqual(plugin.auto_wrap_policy, "transformer_based_wrap")
        self.assertEqual(plugin.transformer_cls_names_to_wrap, ["TransformerBlock"])

    def test_load_fsdp_plugin_rejects_deepspeed(self):
        trainer = object.__new__(Trainer)
        trainer.config = SimpleNamespace(
            fsdp_enable=True,
            fsdp_version=2,
            fsdp_reshard_after_forward=True,
            fsdp_cpu_ram_efficient_loading=False,
            fsdp_state_dict_type="SHARDED_STATE_DICT",
            fsdp_auto_wrap_policy="transformer_based_wrap",
            fsdp_transformer_layer_cls_to_wrap=None,
            deepspeed_config={"zero_optimization": {"stage": 2}},
        )

        with self.assertRaises(ValueError):
            trainer._load_fsdp_plugin()

    @patch("simpletuner.helpers.training.trainer.Validation")
    def test_init_validations_enabled_for_fsdp_full_shard(self, mock_validation):
        """Test that FSDP with reshard_after_forward now supports validation"""
        trainer = object.__new__(Trainer)
        trainer.accelerator = SimpleNamespace(
            state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(deepspeed_config={"zero_optimization": {"stage": 2}}))
        )
        trainer.accelerator.wait_for_everyone = lambda *_, **__: None
        trainer.accelerator.is_main_process = True
        trainer.config = SimpleNamespace(
            use_fsdp=True,
            fsdp_reshard_after_forward=True,
            validation_disable=False,
            eval_steps_interval=None,
            weight_dtype=torch.float32,
            use_deepspeed_optimizer=False,
            vae_path=None,
            controlnet=False,
            control=False,
            validation_using_datasets=False,
            model_family="sdxl",
            model_flavour="base",
            output_dir="/tmp",
            use_ema=False,
            ema_validation="none",
            num_eval_images=1,
            eval_dataset_id=None,
            validation_prompt_library=None,
            user_prompt_library=None,
            validation_prompt=None,
            train_text_encoder=False,
            disable_benchmark=True,
        )
        trainer.validation = None
        trainer.evaluation = None
        trainer.model = MagicMock()
        trainer.validation_prompt_metadata = None
        trainer.distiller = None
        trainer._get_trainable_parameters = None
        trainer.ema_model = None

        validation_instance = MagicMock()
        validation_instance.benchmark_exists.return_value = True
        mock_validation.return_value = validation_instance
        trainer._emit_event = lambda *_, **__: None

        trainer.init_validations()

        # FSDP validation is now supported - validation_disable should remain False
        self.assertFalse(trainer.config.validation_disable)
        mock_validation.assert_called_once()

    @patch("simpletuner.helpers.training.trainer.Validation")
    def test_init_validations_creates_eval_for_epoch_interval(self, mock_validation):
        trainer = object.__new__(Trainer)
        trainer.accelerator = SimpleNamespace(
            state=SimpleNamespace(deepspeed_plugin=SimpleNamespace(deepspeed_config={"zero_optimization": {"stage": 2}})),
            wait_for_everyone=lambda *_, **__: None,
            is_main_process=True,
        )
        trainer.validation_prompt_metadata = None
        trainer.validation = None
        trainer.evaluation = None
        trainer.model = MagicMock()
        trainer.distiller = None
        trainer._get_trainable_parameters = None
        trainer.ema_model = None
        config = SimpleNamespace(
            use_fsdp=False,
            fsdp_reshard_after_forward=False,
            validation_disable=False,
            eval_steps_interval=None,
            eval_epoch_interval=1.0,
            weight_dtype=torch.float32,
            use_deepspeed_optimizer=False,
            vae_path=None,
            controlnet=False,
            control=False,
            validation_using_datasets=False,
            model_family="sdxl",
            model_flavour="base",
            output_dir="/tmp",
            use_ema=False,
            ema_validation="none",
            num_eval_images=1,
            eval_dataset_id=None,
            validation_prompt_library=None,
            user_prompt_library=None,
            validation_prompt=None,
            train_text_encoder=False,
            disable_benchmark=True,
        )
        prev_args = StateTracker.get_args()
        StateTracker.set_args(config)
        trainer.config = config
        trainer._emit_event = lambda *_, **__: None

        try:
            trainer.init_validations()
            self.assertIsNotNone(trainer.evaluation)
        finally:
            StateTracker.set_args(prev_args)

    @patch("simpletuner.helpers.training.trainer.Trainer._misc_init", return_value=Mock())
    @patch(
        "simpletuner.helpers.training.trainer.Trainer.parse_arguments",
        return_value=Mock(),
    )
    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=1024**3)
    def test_stats_memory_used_cuda(self, mock_memory_allocated, mock_is_available, mock_parse_args, mock_misc_init):
        trainer = Trainer()
        trainer.model = Mock()
        memory_used = trainer.stats_memory_used()
        self.assertEqual(memory_used, 1.0)

    @patch("simpletuner.helpers.training.trainer.Trainer._misc_init", return_value=Mock())
    @patch(
        "simpletuner.helpers.training.trainer.Trainer.parse_arguments",
        return_value=Mock(),
    )
    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=True)
    @patch("torch.mps.current_allocated_memory", return_value=1024**3)
    def test_stats_memory_used_mps(
        self,
        mock_current_allocated_memory,
        mock_mps_is_available,
        mock_cuda_is_available,
        mock_parse_args,
        mock_misc_init,
    ):
        trainer = Trainer()
        trainer.model = Mock()
        memory_used = trainer.stats_memory_used()
        self.assertEqual(memory_used, 1.0)

    @patch("simpletuner.helpers.training.trainer.Trainer._misc_init", return_value=Mock())
    @patch(
        "simpletuner.helpers.training.trainer.Trainer.parse_arguments",
        return_value=Mock(),
    )
    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    @patch("simpletuner.helpers.training.trainer.logger")
    def test_stats_memory_used_none(
        self,
        mock_logger,
        mock_mps_is_available,
        mock_cuda_is_available,
        mock_parse_args,
        mock_misc_init,
    ):
        trainer = Trainer()
        trainer.model = Mock()
        memory_used = trainer.stats_memory_used()
        self.assertEqual(memory_used, 0)
        mock_logger.warning.assert_called_with(
            "CUDA, ROCm, or Apple MPS not detected here. We cannot report VRAM reductions."
        )

    @patch("simpletuner.helpers.training.trainer.load_config")
    @patch("simpletuner.helpers.training.trainer.safety_check")
    @patch("torch.set_num_threads")
    @patch("simpletuner.helpers.training.state_tracker.StateTracker.set_global_step")
    @patch("simpletuner.helpers.training.state_tracker.StateTracker.set_args")
    @patch("simpletuner.helpers.training.state_tracker.StateTracker.set_weight_dtype")
    @patch("simpletuner.helpers.training.trainer.Trainer.set_model_family")
    @patch("simpletuner.helpers.training.trainer.Trainer.init_noise_schedule")
    @patch(
        "accelerate.accelerator.Accelerator",
        return_value=Mock(device=Mock(type="cuda")),
    )
    @patch("accelerate.state.AcceleratorState", Mock())
    @patch(
        "argparse.ArgumentParser.parse_args",
        return_value=Mock(
            torch_num_threads=2,
            train_batch_size=1,
            weight_dtype=torch.float32,
            model_type="full",
            optimizer="adamw_bf16",
            optimizer_config=None,
            max_train_steps=2,
            num_train_epochs=0,
            timestep_bias_portion=0,
            metadata_update_interval=100,
            gradient_accumulation_steps=1,
            validation_resolution=1024,
            mixed_precision="bf16",
            report_to="none",
            output_dir="output_dir",
            logging_dir="logging_dir",
            learning_rate=1,
            flow_schedule_shift=3,
            user_prompt_library=None,
            flow_schedule_auto_shift=False,
            validation_guidance_skip_layers=None,
            pretrained_model_name_or_path="some/path",
            pretrained_vae_model_name_or_path="some/other/path",
            base_model_precision="no_change",
            gradient_checkpointing_interval=None,
            validation_num_video_frames=None,
            eval_steps_interval=None,
            controlnet=False,
            text_encoder_4_precision="no_change",
            attention_mechanism="diffusers",
            distillation_config=None,
        ),
    )
    def test_misc_init(
        self,
        mock_argparse,
        # mock_accelerator_state,
        mock_accelerator,
        mock_init_noise_schedule,
        mock_set_model_family,
        mock_set_weight_dtype,
        mock_set_args,
        mock_set_global_step,
        mock_set_num_threads,
        mock_safety_check,
        mock_load_config,
    ):
        # Configure the mock_load_config to return a proper config object
        mock_config = Mock()
        mock_config.lr_scale = False  # Disable learning rate scaling to avoid the format error
        mock_config.learning_rate = 1e-4
        mock_config.train_batch_size = 1
        mock_config.gradient_accumulation_steps = 1
        mock_config.controlnet = False
        mock_config.model_family = "stable_diffusion"
        mock_config.base_model_precision = "no_change"
        mock_config.torch_num_threads = 2
        mock_config.weight_dtype = torch.bfloat16
        mock_config.attention_mechanism = "diffusers"
        mock_config.sageattention_usage = AttentionBackendMode.INFERENCE
        mock_config.publishing_config = None
        mock_load_config.return_value = mock_config

        with test_config.QuietLogs():
            trainer = Trainer(disable_accelerator=True)
        trainer.model = Mock()
        trainer.config = MagicMock(
            torch_num_threads=2,
            train_batch_size=1,
            base_model_precision="no_change",
            weight_dtype=torch.bfloat16,
        )
        trainer._misc_init()
        mock_set_num_threads.assert_called_with(trainer.config.torch_num_threads)
        self.assertEqual(
            trainer.state,
            {
                "lr": 0.0,
                "global_step": 0,
                "global_resume_step": 0,
                "first_epoch": 1,
                "args": trainer.config.__dict__,
            },
        )
        self.assertEqual(trainer.timesteps_buffer, [])
        self.assertEqual(trainer.guidance_values_list, [])
        self.assertEqual(trainer.train_loss, 0.0)
        self.assertIsNone(trainer.bf)
        self.assertIsNone(trainer.grad_norm)
        self.assertEqual(trainer.extra_lr_scheduler_kwargs, {})
        mock_set_global_step.assert_called_with(0)
        mock_set_weight_dtype.assert_called_with(trainer.config.weight_dtype)
        mock_set_model_family.assert_called()
        mock_init_noise_schedule.assert_called()

    @patch("simpletuner.helpers.training.trainer.logger")
    @patch(
        "simpletuner.helpers.training.trainer.model_classes",
        {"full": ["sdxl", "sd3", "legacy"]},
    )
    @patch("simpletuner.helpers.training.state_tracker.StateTracker")
    def test_set_model_family_default(self, mock_state_tracker, mock_logger):
        with patch("simpletuner.helpers.training.trainer.Trainer._misc_init"):
            with patch("simpletuner.helpers.training.trainer.Trainer.parse_arguments"):
                trainer = Trainer()
        trainer.config = Mock(model_family=None)
        trainer.config.pretrained_model_name_or_path = "some/path"
        trainer.config.pretrained_vae_model_name_or_path = None
        trainer.config.vae_path = None
        trainer.config.text_encoder_path = None
        trainer.config.text_encoder_subfolder = None
        trainer.config.model_family = "sdxl"
        trainer.model = Mock()

        with patch(
            "simpletuner.helpers.training.state_tracker.StateTracker.is_sdxl_refiner",
            return_value=False,
        ):
            trainer.set_model_family()
            self.assertEqual(trainer.config.model_type_label, "Stable Diffusion XL")
            mock_logger.warning.assert_not_called()

    @patch("simpletuner.helpers.training.trainer.Trainer._misc_init", return_value=Mock())
    @patch(
        "simpletuner.helpers.training.trainer.Trainer.parse_arguments",
        return_value=Mock(),
    )
    def test_set_model_family_invalid(self, mock_parse_args, mock_misc_init):
        trainer = Trainer()
        trainer.model = Mock()
        trainer.config = Mock(model_family="invalid_model_family")
        trainer.config.pretrained_model_name_or_path = "some/path"
        with self.assertRaises(ValueError) as context:
            trainer.set_model_family()
        self.assertIn(
            "Invalid model family specified: invalid_model_family",
            str(context.exception),
        )

    @patch("simpletuner.helpers.training.trainer.Trainer._misc_init", return_value=Mock())
    @patch(
        "simpletuner.helpers.training.trainer.Trainer.parse_arguments",
        return_value=Mock(),
    )
    @patch("simpletuner.helpers.training.trainer.logger")
    @patch("simpletuner.helpers.training.state_tracker.StateTracker")
    def test_epoch_rollover(self, mock_state_tracker, mock_logger, mock_parse_args, mock_misc_init):
        trainer = Trainer()
        trainer.model = Mock()
        trainer.state = {"first_epoch": 1, "current_epoch": 1}
        trainer.config = Mock(
            num_train_epochs=5,
            aspect_bucket_disable_rebuild=False,
            lr_scheduler="cosine_with_restarts",
        )
        trainer.extra_lr_scheduler_kwargs = {}
        with patch(
            "simpletuner.helpers.training.state_tracker.StateTracker.get_data_backends",
            return_value={},
        ):
            trainer._epoch_rollover(2)
            self.assertEqual(trainer.state["current_epoch"], 2)
            self.assertEqual(trainer.extra_lr_scheduler_kwargs["epoch"], 2)

    @patch(
        "simpletuner.helpers.training.trainer.Trainer.parse_arguments",
        return_value=Mock(),
    )
    @patch("simpletuner.helpers.training.trainer.Trainer._misc_init", return_value=Mock())
    def test_epoch_rollover_same_epoch(self, mock_misc_init, mock_parse_args):
        trainer = Trainer(
            config={
                "--num_train_epochs": 0,
                "--model_family": "pixart_sigma",
                "--optimizer": "adamw_bf16",
                "--pretrained_model_name_or_path": "some/path",
            }
        )
        trainer.model = Mock()
        trainer.state = {"first_epoch": 1, "current_epoch": 1}
        trainer._epoch_rollover(1)
        self.assertEqual(trainer.state["current_epoch"], 1)

    @patch("simpletuner.helpers.training.trainer.Trainer._misc_init", return_value=Mock())
    @patch(
        "simpletuner.helpers.training.trainer.Trainer.parse_arguments",
        return_value=Mock(),
    )
    @patch("simpletuner.helpers.training.trainer.os.makedirs")
    @patch("simpletuner.helpers.training.state_tracker.StateTracker.delete_cache_files")
    def test_init_clear_backend_cache_preserve(
        self, mock_delete_cache_files, mock_makedirs, mock_parse_args, mock_misc_init
    ):
        trainer = Trainer()
        trainer.model = Mock()
        trainer.config = Mock(output_dir="/path/to/output", preserve_data_backend_cache=True)
        trainer.init_clear_backend_cache()
        mock_makedirs.assert_called_with("/path/to/output", exist_ok=True)
        mock_delete_cache_files.assert_not_called()

    @patch("simpletuner.helpers.training.trainer.Trainer._misc_init", return_value=Mock())
    @patch(
        "simpletuner.helpers.training.trainer.Trainer.parse_arguments",
        return_value=Mock(),
    )
    @patch("simpletuner.helpers.training.trainer.os.makedirs")
    @patch("simpletuner.helpers.training.state_tracker.StateTracker.delete_cache_files")
    def test_init_clear_backend_cache_delete(self, mock_delete_cache_files, mock_makedirs, mock_parse_args, mock_misc_init):
        trainer = Trainer()
        trainer.accelerator = MagicMock(is_local_main_process=True)
        trainer.model = Mock()
        trainer.config = Mock(output_dir="/path/to/output", preserve_data_backend_cache=False)
        trainer.init_clear_backend_cache()
        mock_makedirs.assert_called_with("/path/to/output", exist_ok=True)
        mock_delete_cache_files.assert_called_with(preserve_data_backend_cache=False)

    @patch("simpletuner.helpers.training.trainer.Trainer._misc_init", return_value=Mock())
    @patch(
        "simpletuner.helpers.training.trainer.Trainer.parse_arguments",
        return_value=Mock(),
    )
    @patch("simpletuner.helpers.training.trainer.logger")
    @patch(
        "simpletuner.helpers.training.trainer.os.path.basename",
        return_value="checkpoint-100",
    )
    @patch(
        "simpletuner.helpers.training.trainer.os.listdir",
        return_value=["checkpoint-100", "checkpoint-200"],
    )
    @patch(
        "simpletuner.helpers.training.trainer.os.path.join",
        side_effect=lambda *args: "/".join(args),
    )
    @patch("simpletuner.helpers.training.trainer.os.path.exists", return_value=True)
    @patch("simpletuner.helpers.training.trainer.Accelerator")
    @patch("simpletuner.helpers.training.state_tracker.StateTracker")
    def test_init_resume_checkpoint(
        self,
        mock_state_tracker,
        mock_accelerator_class,
        mock_path_exists,
        mock_path_join,
        mock_os_listdir,
        mock_path_basename,
        mock_logger,
        mock_parse_args,
        mock_misc_init,
    ):
        trainer = Trainer()
        trainer.model = Mock()
        trainer.config = Mock(
            output_dir="/path/to/output",
            resume_from_checkpoint="latest",
            total_steps_remaining_at_start=100,
            global_resume_step=1,
            num_train_epochs=0,
            max_train_steps=100,
            musubi_blocks_to_swap=0,
        )
        trainer.accelerator = Mock(num_processes=1)
        trainer.state = {"global_step": 0, "first_epoch": 1, "current_epoch": 1}
        trainer.optimizer = Mock()
        trainer.config.lr_scheduler = "constant"
        trainer.config.learning_rate = 0.001
        trainer.config.is_schedulefree = False
        trainer.config.overrode_max_train_steps = False

        # Mock lr_scheduler
        lr_scheduler = Mock()
        lr_scheduler.state_dict.return_value = {"base_lrs": [0.1], "_last_lr": [0.1]}

        with patch(
            "simpletuner.helpers.training.state_tracker.StateTracker.get_data_backends",
            return_value={},
        ):
            with patch(
                "simpletuner.helpers.training.state_tracker.StateTracker.get_global_step",
                return_value=100,
            ):
                trainer.init_resume_checkpoint(lr_scheduler=lr_scheduler)
                mock_logger.info.assert_called()
                trainer.accelerator.load_state.assert_called_with("/path/to/output/checkpoint-200")

    @patch("simpletuner.helpers.training.trainer.logger")
    def test_init_resume_checkpoint_prodigy_without_split_groups(self, mock_logger):
        """Test that prodigy optimizer works with optimizers that don't have split_groups attribute"""
        trainer = object.__new__(Trainer)
        trainer.model = Mock()
        trainer.config = Mock(
            output_dir="/path/to/output",
            resume_from_checkpoint="checkpoint-100",
            num_train_epochs=1,
            ignore_final_epochs=False,
            optimizer="prodigy",
            lr_scheduler="constant",
            is_schedulefree=False,
            learning_rate=0.001,
            musubi_blocks_to_swap=0,
        )
        trainer.accelerator = Mock(num_processes=1)
        trainer.accelerator.wait_for_everyone = Mock()
        trainer.accelerator.load_state = Mock()
        trainer.state = {"global_step": 0, "first_epoch": 1, "current_epoch": 1, "global_resume_step": 0}
        trainer.distiller = None

        # Create a mock optimizer that simulates ProdigyPlusScheduleFree (no split_groups attribute)
        mock_inner_optimizer = Mock(spec=[])  # Empty spec means no attributes
        # Explicitly ensure split_groups doesn't exist
        if hasattr(mock_inner_optimizer, "split_groups"):
            delattr(mock_inner_optimizer, "split_groups")

        mock_optimizer = Mock()
        mock_optimizer.optimizer = mock_inner_optimizer

        # Create mock parameter group with device-movable tensors
        mock_param = torch.randn(10, 10)
        param_group = {
            "params": [mock_param],
            "running_d_numerator": torch.tensor(1.0),
            "running_d_denom": torch.tensor(1.0),
        }
        mock_optimizer.param_groups = [param_group]

        trainer.optimizer = mock_optimizer
        trainer._emit_event = Mock()
        trainer.job_id = "test-job"

        # Mock StateTracker methods
        with patch("simpletuner.helpers.training.state_tracker.StateTracker.get_data_backends", return_value={}):
            with patch("simpletuner.helpers.training.state_tracker.StateTracker.get_global_step", return_value=0):
                with patch("simpletuner.helpers.training.state_tracker.StateTracker.set_global_resume_step"):
                    with patch(
                        "simpletuner.helpers.training.state_tracker.StateTracker.get_training_state", return_value={}
                    ):
                        with patch("simpletuner.helpers.training.state_tracker.StateTracker.get_epoch", return_value=1):
                            with patch("simpletuner.helpers.training.state_tracker.StateTracker.set_epoch"):
                                # This should not raise AttributeError
                                result = trainer.init_resume_checkpoint(lr_scheduler=None)

        # Verify the tensors were moved to the correct device
        self.assertEqual(param_group["running_d_numerator"].device, mock_param.device)
        self.assertEqual(param_group["running_d_denom"].device, mock_param.device)
        self.assertFalse(param_group["use_focus"])

    @patch("simpletuner.helpers.training.trainer.logger")
    def test_init_resume_checkpoint_prodigy_with_split_groups(self, mock_logger):
        """Test that prodigy optimizer works correctly when split_groups=True"""
        trainer = object.__new__(Trainer)
        trainer.model = Mock()
        trainer.config = Mock(
            output_dir="/path/to/output",
            resume_from_checkpoint="checkpoint-100",
            num_train_epochs=1,
            ignore_final_epochs=False,
            optimizer="prodigy",
            lr_scheduler="constant",
            is_schedulefree=False,
            learning_rate=0.001,
            musubi_blocks_to_swap=0,
        )
        trainer.accelerator = Mock(num_processes=1)
        trainer.accelerator.wait_for_everyone = Mock()
        trainer.accelerator.load_state = Mock()
        trainer.state = {"global_step": 0, "first_epoch": 1, "current_epoch": 1, "global_resume_step": 0}
        trainer.distiller = None

        # Create a mock optimizer with split_groups=True
        mock_inner_optimizer = Mock()
        mock_inner_optimizer.split_groups = True

        mock_optimizer = Mock()
        mock_optimizer.optimizer = mock_inner_optimizer

        # Create multiple mock parameter groups
        mock_param1 = torch.randn(10, 10)
        mock_param2 = torch.randn(5, 5)
        param_groups = [
            {
                "params": [mock_param1],
                "running_d_numerator": torch.tensor(1.0),
                "running_d_denom": torch.tensor(1.0),
            },
            {
                "params": [mock_param2],
                "running_d_numerator": torch.tensor(2.0),
                "running_d_denom": torch.tensor(2.0),
            },
        ]
        mock_optimizer.param_groups = param_groups

        trainer.optimizer = mock_optimizer
        trainer._emit_event = Mock()
        trainer.job_id = "test-job"

        # Mock StateTracker methods
        with patch("simpletuner.helpers.training.state_tracker.StateTracker.get_data_backends", return_value={}):
            with patch("simpletuner.helpers.training.state_tracker.StateTracker.get_global_step", return_value=0):
                with patch("simpletuner.helpers.training.state_tracker.StateTracker.set_global_resume_step"):
                    with patch(
                        "simpletuner.helpers.training.state_tracker.StateTracker.get_training_state", return_value={}
                    ):
                        with patch("simpletuner.helpers.training.state_tracker.StateTracker.get_epoch", return_value=1):
                            with patch("simpletuner.helpers.training.state_tracker.StateTracker.set_epoch"):
                                # This should process all parameter groups when split_groups=True
                                result = trainer.init_resume_checkpoint(lr_scheduler=None)

        # Verify all parameter groups were processed
        for i, group in enumerate(param_groups):
            self.assertEqual(group["running_d_numerator"].device, group["params"][0].device)
            self.assertEqual(group["running_d_denom"].device, group["params"][0].device)
            self.assertFalse(group["use_focus"])

    def test_epoch_checkpoint_persists_next_epoch_without_state_mutation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoint-10"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            training_state_path = checkpoint_dir / "training_state.json"
            training_state_path.write_text(json.dumps({"epoch": 2, "global_step": 10}), encoding="utf-8")

            trainer = object.__new__(Trainer)
            trainer.job_id = None
            trainer.hub_manager = None
            trainer.webhook_handler = None
            trainer.model_hooks = SimpleNamespace(training_state_path="training_state.json")
            trainer._send_webhook_msg = Mock()
            trainer._prepare_training_progress_payload = Mock(return_value=({}, {}))
            trainer._emit_event = Mock()
            trainer._run_post_upload_script = Mock()
            trainer.checkpoint_state_cleanup = Mock()
            trainer.state = {"global_step": 10, "global_resume_step": 0, "current_epoch": 2}
            trainer.config = SimpleNamespace(
                output_dir=str(tmpdir),
                use_deepspeed_optimizer=False,
                fsdp_enable=False,
                checkpoints_total_limit=None,
                num_train_epochs=5,
            )
            trainer.accelerator = SimpleNamespace(is_main_process=True)
            trainer.checkpoint_state_save = Mock(return_value=str(checkpoint_dir))

            checkpoint_path = trainer._run_standard_checkpoint(
                webhook_message="Epoch checkpoint",
                parent_loss=None,
                epoch=2,
                upload_to_hub=False,
            )

            self.assertEqual(str(checkpoint_dir), checkpoint_path)
            trainer._write_checkpoint_epoch(checkpoint_path, 3)

            persisted_state = json.loads(training_state_path.read_text())
            self.assertEqual(3, persisted_state["epoch"])
            self.assertEqual(10, persisted_state["global_step"])
            self.assertEqual(2, trainer.state["current_epoch"])
            trainer._run_post_upload_script.assert_called_once_with(local_path=str(checkpoint_dir), remote_path=None)
            trainer.checkpoint_state_save.assert_called_once_with(str(tmpdir))

    @patch("simpletuner.helpers.training.state_tracker.StateTracker")
    @patch("simpletuner.helpers.training.trainer.tqdm")
    @patch("simpletuner.helpers.training.trainer.logger")
    @patch("simpletuner.helpers.training.trainer.training_logger")
    def test_accumulate_unpacks_training_models_for_fsdp2(
        self, mock_training_logger, mock_logger, mock_tqdm, mock_state_tracker
    ):
        """Regression test: trainer.train() should unpack training_models when calling accumulate()

        This test verifies the fix for the bug where training_models was passed as a list
        directly to accumulate(), causing AttributeError with FSDP2 in Accelerate 1.11.0+.
        FSDP2's no_sync() calls model.set_requires_gradient_sync(False) which fails on lists.
        """
        trainer = object.__new__(Trainer)
        trainer.config = SimpleNamespace(
            model_type="full",
            train_text_encoder=False,
            lora_type="standard",
            num_train_epochs=1,
            max_train_steps=1000,
            train_batch_size=1,
            dataloader_prefetch=False,
            lr_scheduler="constant",
            eval_dataset_id=None,
            num_update_steps_per_epoch=1,
            ignore_final_epochs=False,
            distillation_method=None,
            disable_accelerator=False,
            optimizer="adamw",
            gradient_precision="fp32",
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
        )

        mock_trained_model = Mock()
        mock_trained_model.set_requires_gradient_sync = Mock()
        mock_trained_model.train = Mock()

        trainer.model = Mock()
        trainer.model.get_trained_component = Mock(return_value=mock_trained_model)
        trainer.model.loss = Mock(return_value=torch.tensor(0.5))
        trainer.model.auxiliary_loss = Mock(return_value=(torch.tensor(0.5), {}))
        trainer.text_encoders = []
        trainer.model_predict = Mock(return_value={"model_prediction": torch.zeros(1, 4, 4, 4)})
        trainer._max_grad_value = Mock(return_value=torch.tensor(0.1))
        trainer.params_to_optimize = []

        def prepare_batch_side_effect(batch):
            if batch is False:
                return False
            return {
                "latents": torch.zeros(1, 4, 4, 4),
                "timesteps": torch.zeros(1),
            }

        trainer.prepare_batch = Mock(side_effect=prepare_batch_side_effect)

        trainer.state = {
            "global_step": 0,
            "global_resume_step": 0,
            "current_epoch": 1,
            "first_epoch": 1,
            "lr": 0.0001,
        }
        trainer.timesteps_buffer = []
        trainer.distiller = None
        trainer.bf = None
        trainer.extra_lr_scheduler_kwargs = {}
        trainer.train_loss = 0.0
        trainer._epoch_rollover = Mock()
        trainer._should_checkpoint_epoch = Mock(return_value=False)
        trainer._exit_on_signal = Mock()
        trainer.init_trackers = Mock()
        trainer._train_initial_msg = Mock()
        trainer._get_trainable_parameters = Mock(return_value=[])
        trainer.iteration_tracker = Mock()

        received_accumulate_args = []
        accumulate_call_count = 0

        class StopTrainingException(Exception):
            pass

        def mock_accumulate(*models):
            nonlocal accumulate_call_count
            accumulate_call_count += 1
            received_accumulate_args.extend(models)

            class _AccumulateContext:
                def __enter__(self):
                    for model in models:
                        if isinstance(model, list):
                            raise AttributeError(
                                "'list' object has no attribute 'set_requires_gradient_sync'. "
                                "This simulates the FSDP2 error."
                            )
                        if hasattr(model, "set_requires_gradient_sync"):
                            model.set_requires_gradient_sync(False)
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    for model in models:
                        if hasattr(model, "set_requires_gradient_sync"):
                            model.set_requires_gradient_sync(True)
                    raise StopTrainingException("Captured accumulate call, stopping test")

            return _AccumulateContext()

        mock_loss_tensor = torch.tensor(0.5)
        mock_accelerator = Mock()
        mock_accelerator.is_main_process = True
        mock_accelerator.accumulate = mock_accumulate
        mock_accelerator.wait_for_everyone = Mock()
        mock_accelerator.gather = Mock(return_value=mock_loss_tensor)
        mock_accelerator.backward = Mock()
        mock_accelerator.sync_gradients = False
        trainer.accelerator = mock_accelerator

        mock_progress_bar = Mock()
        mock_progress_bar.set_description = Mock()
        mock_tqdm.return_value.__enter__ = Mock(return_value=mock_progress_bar)
        mock_tqdm.return_value.__exit__ = Mock(return_value=False)

        call_count = [0]

        def mock_iterator(step, backends):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"latents": torch.zeros(1, 4, 4, 4), "timesteps": torch.zeros(1)}
            return False

        mock_state_tracker.get_data_backends = Mock(
            return_value={
                "backend1": {"train_dataloader": [{"latents": torch.zeros(1, 4, 4, 4), "timesteps": torch.zeros(1)}]}
            }
        )
        mock_state_tracker.backend_status = Mock(return_value=False)

        with patch("simpletuner.helpers.training.trainer.random_dataloader_iterator", mock_iterator):
            try:
                trainer.train()
            except StopTrainingException:
                pass

        self.assertGreater(accumulate_call_count, 0, "accumulate() should have been called")
        self.assertGreater(len(received_accumulate_args), 0, "accumulate() should have received arguments")
        self.assertIs(
            received_accumulate_args[0],
            mock_trained_model,
            "accumulate() should receive the model object, not a list",
        )
        self.assertNotIsInstance(
            received_accumulate_args[0],
            list,
            "accumulate() must not receive a list - would cause AttributeError in FSDP2",
        )
        mock_trained_model.set_requires_gradient_sync.assert_any_call(False)

    def test_build_init_tracker_kwargs_filters_to_active_trackers(self):
        trainer = object.__new__(Trainer)
        trainer.accelerator = SimpleNamespace(log_with=["tensorboard", "wandb"])

        init_kwargs = trainer._build_init_tracker_kwargs("run-name", "abc123")

        self.assertIn("wandb", init_kwargs)
        self.assertEqual(init_kwargs["wandb"]["name"], "run-name")
        self.assertEqual(init_kwargs["wandb"]["id"], "abc123")
        self.assertEqual(init_kwargs["wandb"]["resume"], "allow")
        self.assertTrue(init_kwargs["wandb"]["allow_val_change"])
        self.assertNotIn("tensorboard", init_kwargs)

    def test_build_init_tracker_kwargs_supports_swanlab(self):
        tracker_enum_like = SimpleNamespace(value="swanlab")
        trainer = object.__new__(Trainer)
        trainer.accelerator = SimpleNamespace(log_with=[tracker_enum_like])

        init_kwargs = trainer._build_init_tracker_kwargs("swan-run", "hash456")

        self.assertIn("swanlab", init_kwargs)
        self.assertEqual(
            init_kwargs["swanlab"],
            {
                "experiment_name": "swan-run",
                "id": "hash456",
                "resume": "allow",
            },
        )

    @patch("simpletuner.helpers.training.trainer.TorchDynamoPlugin")
    @patch("simpletuner.helpers.training.trainer.Accelerator")
    def test_dynamo_plugin_created_when_advanced_options_enabled(self, mock_accelerator, mock_dynamo_plugin):
        """Test that TorchDynamoPlugin is created when dynamo advanced options are used"""
        from simpletuner.helpers.training.trainer import DynamoBackend

        mock_backend_enum = Mock()
        mock_backend_enum.INDUCTOR = Mock(value="inductor")
        mock_backend_enum.NO = Mock()

        mock_dynamo_instance = Mock()
        mock_dynamo_plugin.return_value = mock_dynamo_instance

        accelerator_kwargs = {}

        resolved_dynamo_backend = mock_backend_enum.INDUCTOR
        will_create_dynamo_plugin = resolved_dynamo_backend and resolved_dynamo_backend != mock_backend_enum.NO
        dynamo_backend_env = "inductor"

        if not will_create_dynamo_plugin and dynamo_backend_env:
            accelerator_kwargs["dynamo_backend"] = dynamo_backend_env

        if will_create_dynamo_plugin:
            plugin_kwargs = {"backend": resolved_dynamo_backend}
            plugin_kwargs["mode"] = "max-autotune"
            plugin_kwargs["dynamic"] = True
            plugin_kwargs["use_regional_compilation"] = True

            mock_dynamo_plugin(**plugin_kwargs)
            accelerator_kwargs["dynamo_plugin"] = mock_dynamo_instance

        self.assertIn("dynamo_plugin", accelerator_kwargs)
        self.assertNotIn("dynamo_backend", accelerator_kwargs)
        mock_dynamo_plugin.assert_called_once()
        call_args = mock_dynamo_plugin.call_args[1]
        self.assertEqual(call_args["mode"], "max-autotune")
        self.assertTrue(call_args["dynamic"])
        self.assertTrue(call_args["use_regional_compilation"])

    @patch("simpletuner.helpers.training.trainer.Accelerator")
    def test_dynamo_backend_string_used_when_no_advanced_options(self, mock_accelerator):
        """Test that simple dynamo_backend string is used when no advanced options are specified"""
        trainer = object.__new__(Trainer)
        trainer.config = SimpleNamespace(
            dynamo_backend="inductor",
            dynamo_mode=None,
            dynamo_dynamic=None,
            dynamo_fullgraph=None,
            dynamo_use_regional_compilation=None,
            gradient_accumulation_steps=1,
            mixed_precision="bf16",
            fsdp_enable=False,
            deepspeed_config=None,
            report_to="none",
        )

        from simpletuner.helpers.training.trainer import DynamoBackend

        accelerator_kwargs = {}

        resolved_dynamo_backend = None
        will_create_dynamo_plugin = resolved_dynamo_backend and resolved_dynamo_backend != DynamoBackend.NO
        dynamo_backend_env = "inductor"

        if not will_create_dynamo_plugin and dynamo_backend_env:
            accelerator_kwargs["dynamo_backend"] = dynamo_backend_env

        self.assertIn("dynamo_backend", accelerator_kwargs)
        self.assertNotIn("dynamo_plugin", accelerator_kwargs)
        self.assertEqual(accelerator_kwargs["dynamo_backend"], "inductor")

    @patch("simpletuner.helpers.training.trainer.Accelerator")
    def test_dynamo_conflict_error_message(self, mock_accelerator):
        """Test that a clear error message is shown when both dynamo_backend and dynamo_plugin are set"""
        mock_accelerator.side_effect = ValueError(
            "You cannot pass in both dynamo_plugin and dynamo_backend, please only pass in one."
        )

        trainer = object.__new__(Trainer)
        trainer._should_force_bf16_override = Mock(return_value=False)

        accelerator_kwargs = {
            "dynamo_backend": "inductor",
            "dynamo_plugin": Mock(),
        }

        try:
            mock_accelerator(**accelerator_kwargs)
        except ValueError as err:
            if "dynamo_plugin and dynamo_backend" in str(err):
                has_backend = "dynamo_backend" in accelerator_kwargs
                has_plugin = "dynamo_plugin" in accelerator_kwargs
                backend_val = accelerator_kwargs.get("dynamo_backend", "not set")
                plugin_val = "set" if has_plugin else "not set"

                with self.assertRaises(ValueError) as context:
                    raise ValueError(
                        f"Conflicting Torch Dynamo configuration detected. "
                        f"Cannot pass both dynamo_plugin and dynamo_backend to Accelerator. "
                        f"Current state: dynamo_backend={backend_val} (present={has_backend}), "
                        f"dynamo_plugin={plugin_val} (present={has_plugin}). "
                        f"Original error: {err}"
                    ) from err

                self.assertIn("Conflicting Torch Dynamo configuration", str(context.exception))
                self.assertIn("dynamo_backend=inductor", str(context.exception))
                self.assertIn("dynamo_plugin=set", str(context.exception))


if __name__ == "__main__":
    unittest.main()
