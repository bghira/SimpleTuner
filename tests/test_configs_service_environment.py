import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import tests.test_stubs  # noqa: F401

if "torch" not in sys.modules:
    torch_stub = ModuleType("torch")
    dist_stub = ModuleType("torch.distributed")
    dist_stub.is_available = lambda: False  # type: ignore[attr-defined]
    dist_stub.is_initialized = lambda: False  # type: ignore[attr-defined]
    dist_stub.get_rank = lambda: 0  # type: ignore[attr-defined]
    torch_stub.distributed = dist_stub  # type: ignore[attr-defined]
    torch_stub.Tensor = type("Tensor", (), {})  # type: ignore[attr-defined]
    torch_stub.FloatTensor = type("FloatTensor", (), {})  # type: ignore[attr-defined]
    torch_stub.no_grad = lambda: (lambda fn: fn)  # type: ignore[attr-defined]
    cuda_stub = SimpleNamespace(
        is_available=lambda: False,
        get_device_properties=lambda *_: SimpleNamespace(major=0),
        device_count=lambda: 0,
        current_device=lambda: 0,
    )
    torch_stub.cuda = cuda_stub  # type: ignore[attr-defined]
    torch_stub.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))  # type: ignore[attr-defined]
    sys.modules["torch"] = torch_stub
    sys.modules["torch.distributed"] = dist_stub
    sys.modules["torch.nn"] = ModuleType("torch.nn")
    sys.modules["torch.nn.functional"] = ModuleType("torch.nn.functional")
    torch_distributions_stub = ModuleType("torch.distributions")
    torch_distributions_stub.Beta = type("Beta", (), {})  # type: ignore[attr-defined]
    sys.modules["torch.distributions"] = torch_distributions_stub
    torch_optim_stub = ModuleType("torch.optim")
    torch_optim_stub.Optimizer = type("Optimizer", (), {})  # type: ignore[attr-defined]
    sys.modules["torch.optim"] = torch_optim_stub
    torch_optim_optimizer_stub = ModuleType("torch.optim.optimizer")
    torch_optim_optimizer_stub.Optimizer = torch_optim_stub.Optimizer  # type: ignore[attr-defined]
    sys.modules["torch.optim.optimizer"] = torch_optim_optimizer_stub
    torch_lr_scheduler_stub = ModuleType("torch.optim.lr_scheduler")
    torch_lr_scheduler_stub.LambdaLR = type("LambdaLR", (), {})  # type: ignore[attr-defined]
    torch_lr_scheduler_stub.LRScheduler = type("LRScheduler", (), {})  # type: ignore[attr-defined]
    sys.modules["torch.optim.lr_scheduler"] = torch_lr_scheduler_stub
    # Provide nested elastic multiprocessing module tree used by logging setup
    for qualified in (
        "torch.distributed.elastic",
        "torch.distributed.elastic.multiprocessing",
        "torch.distributed.elastic.multiprocessing.redirects",
    ):
        if qualified not in sys.modules:
            sys.modules[qualified] = ModuleType(qualified)

if "fastapi" not in sys.modules:
    fastapi_stub = ModuleType("fastapi")
    status_stub = ModuleType("fastapi.status")
    status_stub.HTTP_400_BAD_REQUEST = 400  # type: ignore[attr-defined]
    status_stub.HTTP_404_NOT_FOUND = 404  # type: ignore[attr-defined]
    status_stub.HTTP_409_CONFLICT = 409  # type: ignore[attr-defined]
    status_stub.HTTP_422_UNPROCESSABLE_CONTENT = 422  # type: ignore[attr-defined]
    status_stub.HTTP_500_INTERNAL_SERVER_ERROR = 500  # type: ignore[attr-defined]
    fastapi_stub.status = status_stub  # type: ignore[attr-defined]

    class _HTTPException(Exception):
        def __init__(self, status_code=400, detail=None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    def _depends(callable=None):  # type: ignore[unused-argument]
        return callable

    fastapi_stub.HTTPException = _HTTPException  # type: ignore[attr-defined]
    fastapi_stub.Depends = _depends  # type: ignore[attr-defined]
    sys.modules["fastapi"] = fastapi_stub
    sys.modules["fastapi.status"] = status_stub
    requests_stub = ModuleType("fastapi.requests")

    class _Request:
        def __init__(self, *args, **kwargs):
            self.state = SimpleNamespace()

    requests_stub.Request = _Request  # type: ignore[attr-defined]
    sys.modules["fastapi.requests"] = requests_stub

app_stub_name = "simpletuner.simpletuner_sdk.server.app"
if app_stub_name not in sys.modules:
    server_app_stub = ModuleType(app_stub_name)
    server_app_stub.ServerMode = SimpleNamespace(TRAINER="trainer", CALLBACK="callback", UNIFIED="unified")  # type: ignore[attr-defined]

    def _create_app(*args, **kwargs):  # type: ignore[unused-argument]
        raise RuntimeError("Server app not available in test environment")

    server_app_stub.create_app = _create_app  # type: ignore[attr-defined]
    sys.modules[app_stub_name] = server_app_stub

if "pydantic" not in sys.modules:
    pydantic_stub = ModuleType("pydantic")

    class _FieldInfo:
        def __init__(self, default=None, default_factory=None, **_kwargs):
            self.default = default
            self.default_factory = default_factory

    def Field(*, default=None, default_factory=None, **kwargs):  # type: ignore[unused-argument]
        return _FieldInfo(default=default, default_factory=default_factory)

    class BaseModel:
        def __init__(self, **kwargs):
            annotations = getattr(self.__class__, "__annotations__", {})
            for name in annotations:
                if name in kwargs:
                    value = kwargs.pop(name)
                else:
                    attr = getattr(self.__class__, name, None)
                    if isinstance(attr, _FieldInfo):
                        if attr.default_factory is not None:
                            value = attr.default_factory()
                        else:
                            value = attr.default
                    else:
                        value = attr
                setattr(self, name, value)
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self):
            return dict(self.__dict__)

        @classmethod
        def model_validate(cls, data, **_kwargs):
            if isinstance(data, cls):
                return data
            if isinstance(data, dict):
                return cls(**data)
            return cls()

    pydantic_stub.BaseModel = BaseModel  # type: ignore[attr-defined]
    pydantic_stub.Field = Field  # type: ignore[attr-defined]
    sys.modules["pydantic"] = pydantic_stub

if "accelerate" not in sys.modules:
    accelerate_stub = ModuleType("accelerate")
    accelerate_stub.state = SimpleNamespace()  # type: ignore[attr-defined]
    accelerate_stub.InitProcessGroupKwargs = type("InitProcessGroupKwargs", (), {})  # type: ignore[attr-defined]
    sys.modules["accelerate"] = accelerate_stub
    accelerate_utils_stub = ModuleType("accelerate.utils")
    accelerate_utils_stub.ProjectConfiguration = type("ProjectConfiguration", (), {})  # type: ignore[attr-defined]
    accelerate_utils_stub.get_launch_config = lambda *args, **kwargs: {}  # type: ignore[attr-defined]
    sys.modules["accelerate.utils"] = accelerate_utils_stub
    sys.modules["accelerate.accelerator"] = ModuleType("accelerate.accelerator")

optimizer_stub_name = "simpletuner.helpers.training.optimizers.adamw_bfloat16"
if optimizer_stub_name not in sys.modules:
    optimizer_stub = ModuleType(optimizer_stub_name)
    optimizer_stub.AdamWBF16 = type("AdamWBF16", (), {})  # type: ignore[attr-defined]
    sys.modules[optimizer_stub_name] = optimizer_stub

soap_stub_name = "simpletuner.helpers.training.optimizers.soap"
if soap_stub_name not in sys.modules:
    soap_stub = ModuleType(soap_stub_name)
    soap_stub.SOAP = type("SOAP", (), {})  # type: ignore[attr-defined]
    sys.modules[soap_stub_name] = soap_stub

if "torchao" not in sys.modules:
    torchao_stub = ModuleType("torchao")
    sys.modules["torchao"] = torchao_stub
    torchao_optim_stub = ModuleType("torchao.optim")
    torchao_optim_stub.AdamFp8 = type("AdamFp8", (), {})  # type: ignore[attr-defined]
    torchao_optim_stub.AdamW4bit = type("AdamW4bit", (), {})  # type: ignore[attr-defined]
    torchao_optim_stub.AdamW8bit = type("AdamW8bit", (), {})  # type: ignore[attr-defined]
    torchao_optim_stub.AdamWFp8 = type("AdamWFp8", (), {})  # type: ignore[attr-defined]
    torchao_optim_stub.CPUOffloadOptimizer = type("CPUOffloadOptimizer", (), {})  # type: ignore[attr-defined]
    sys.modules["torchao.optim"] = torchao_optim_stub

if "optimi" not in sys.modules:
    optimi_stub = ModuleType("optimi")
    optimi_stub.StableAdamW = type("StableAdamW", (), {})  # type: ignore[attr-defined]
    optimi_stub.FusedAdamW = type("FusedAdamW", (), {})  # type: ignore[attr-defined]
    optimi_stub.AdamW = type("AdamW", (), {})  # type: ignore[attr-defined]
    optimi_stub.Lion = type("Lion", (), {})  # type: ignore[attr-defined]
    optimi_stub.RAdam = type("RAdam", (), {})  # type: ignore[attr-defined]
    optimi_stub.Ranger = type("Ranger", (), {})  # type: ignore[attr-defined]
    optimi_stub.Adan = type("Adan", (), {})  # type: ignore[attr-defined]
    optimi_stub.Adam = type("Adam", (), {})  # type: ignore[attr-defined]
    optimi_stub.SGD = type("SGD", (), {})  # type: ignore[attr-defined]
    sys.modules["optimi"] = optimi_stub

if "simpletuner.configure" not in sys.modules:
    configure_stub = ModuleType("simpletuner.configure")
    configure_stub.model_classes = {"full": ["testfamily"]}  # type: ignore[attr-defined]
    sys.modules["simpletuner.configure"] = configure_stub

if "numpy" not in sys.modules:
    numpy_stub = ModuleType("numpy")
    numpy_stub.array = lambda data, *args, **kwargs: data  # type: ignore[attr-defined]
    numpy_stub.ndarray = list  # type: ignore[attr-defined]
    numpy_stub.zeros = lambda shape, dtype=None: [0] * (shape if isinstance(shape, int) else 0)  # type: ignore[attr-defined]
    sys.modules["numpy"] = numpy_stub

if "torchvision" not in sys.modules:
    torchvision_stub = ModuleType("torchvision")
    torchvision_transforms_stub = ModuleType("torchvision.transforms")
    torchvision_transforms_stub.Compose = lambda funcs: funcs  # type: ignore[attr-defined]
    torchvision_stub.transforms = torchvision_transforms_stub  # type: ignore[attr-defined]
    sys.modules["torchvision"] = torchvision_stub
    sys.modules["torchvision.transforms"] = torchvision_transforms_stub

if "transformers" not in sys.modules:
    transformers_stub = ModuleType("transformers")
    sys.modules["transformers"] = transformers_stub
    transformers_utils_stub = ModuleType("transformers.utils")
    transformers_utils_stub.ContextManagers = type("ContextManagers", (), {})  # type: ignore[attr-defined]
    sys.modules["transformers.utils"] = transformers_utils_stub

if "safetensors" not in sys.modules:
    safetensors_stub = ModuleType("safetensors")
    sys.modules["safetensors"] = safetensors_stub
    sys.modules["safetensors.torch"] = ModuleType("safetensors.torch")

if "huggingface_hub" not in sys.modules:
    huggingface_stub = ModuleType("huggingface_hub")
    sys.modules["huggingface_hub"] = huggingface_stub
    huggingface_utils_stub = ModuleType("huggingface_hub.utils")
    huggingface_utils_stub.validate_hf_hub_args = lambda *args, **kwargs: (lambda fn: fn)  # type: ignore[attr-defined]
    sys.modules["huggingface_hub.utils"] = huggingface_utils_stub

dependencies_stub_name = "simpletuner.simpletuner_sdk.server.dependencies.common"
if dependencies_stub_name not in sys.modules:
    dependencies_stub = ModuleType(dependencies_stub_name)
    dependencies_stub._load_active_config_cached = SimpleNamespace(clear_cache=lambda: None)  # type: ignore[attr-defined]
    sys.modules[dependencies_stub_name] = dependencies_stub

field_service_stub_name = "simpletuner.simpletuner_sdk.server.services.field_service"
if field_service_stub_name not in sys.modules:
    field_service_stub = ModuleType(field_service_stub_name)
    field_service_stub.FieldFormat = SimpleNamespace  # type: ignore[attr-defined]

    class _FieldServiceStub:
        _WEBUI_ONLY_FIELDS = set()

        @staticmethod
        def normalize_config(payload):
            return payload

    field_service_stub.FieldService = _FieldServiceStub  # type: ignore[attr-defined]
    sys.modules[field_service_stub_name] = field_service_stub

if "simpletuner.helpers.training.custom_schedule" not in sys.modules:
    custom_schedule_stub = ModuleType("simpletuner.helpers.training.custom_schedule")
    custom_schedule_stub.apply_flow_schedule_shift = lambda *args, **kwargs: None  # type: ignore[attr-defined]
    custom_schedule_stub.generate_timestep_weights = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    custom_schedule_stub.segmented_timestep_selection = lambda *args, **kwargs: []  # type: ignore[attr-defined]
    sys.modules["simpletuner.helpers.training.custom_schedule"] = custom_schedule_stub

if "simpletuner.helpers.models.flux" not in sys.modules:
    flux_stub = ModuleType("simpletuner.helpers.models.flux")
    flux_stub.calculate_shift_flux = lambda *args, **kwargs: 0.0  # type: ignore[attr-defined]
    sys.modules["simpletuner.helpers.models.flux"] = flux_stub
    sys.modules["simpletuner.helpers.models.flux.pipeline"] = ModuleType("simpletuner.helpers.models.flux.pipeline")

field_service_stub_name = "simpletuner.simpletuner_sdk.server.services.field_service"
from simpletuner.helpers.models.registry import ModelRegistry

try:
    from simpletuner.simpletuner_sdk.server.services.config_store import ConfigStore
    from simpletuner.simpletuner_sdk.server.services.configs_service import ConfigServiceError, ConfigsService
    from simpletuner.simpletuner_sdk.server.services.example_configs_service import ExampleConfigInfo
    from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
    ConfigStore = None  # type: ignore[assignment]
    ConfigsService = None  # type: ignore[assignment]
    ConfigServiceError = None  # type: ignore[assignment]
    ExampleConfigInfo = None  # type: ignore[assignment]
    _SKIP_REASON = f"Dependencies unavailable: {exc}"
else:
    _SKIP_REASON = ""


class _DummyModel:
    DEFAULT_MODEL_NAME = "dummy/model"
    DEFAULT_MODEL_FLAVOUR = "default"
    HUGGINGFACE_PATHS = {"default": "dummy/model"}


@unittest.skipIf(
    ConfigStore is None or ConfigsService is None or ExampleConfigInfo is None or ConfigServiceError is None, _SKIP_REASON
)
class ConfigsServiceEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self._instances_backup = ConfigStore._instances.copy()
        ConfigStore._instances = {}
        self.addCleanup(self._restore_config_store_instances)

        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.config_root = Path(self._tempdir.name).resolve()

        # Create isolated config stores for model and dataloader configs
        self.model_store = ConfigStore(config_dir=self.config_root, config_type="model")

        # Register a dummy model family so _resolve_pretrained_path works
        ModelRegistry.register("testfamily", _DummyModel)
        self.addCleanup(lambda: ModelRegistry._registry.pop("testfamily", None))  # type: ignore[attr-defined]

    def _restore_config_store_instances(self) -> None:
        ConfigStore._instances = self._instances_backup

    def test_create_environment_saves_flat_config(self) -> None:
        service = ConfigsService()

        request = SimpleNamespace(
            name="test-env",
            model_family="testfamily",
            model_flavour="default",
            model_type="lora",
            lora_type="standard",
            description="Test environment",
            example=None,
            dataloader_path=None,
            create_dataloader=True,
        )

        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            result = service.create_environment(request)

        config_file = self.config_root / "test-env" / "config.json"
        self.assertTrue(config_file.exists(), "Training config not written to environment directory")

        with config_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        self.assertIsInstance(payload, dict)
        self.assertNotIn("_metadata", payload)
        self.assertNotIn("config", payload)
        self.assertIn("--model_family", payload)
        self.assertEqual(payload.get("--model_family"), "testfamily")

        dataloader_file = self.config_root / "test-env" / "multidatabackend.json"
        self.assertTrue(dataloader_file.exists(), "Default dataloader plan not created alongside environment")

        expected_rel_path = "test-env/multidatabackend.json"
        self.assertEqual(result["dataloader"]["path"], expected_rel_path)

    def test_example_environment_rewrites_lycoris_path(self) -> None:
        service = ConfigsService()

        example_dir = Path(self._tempdir.name) / "example-demo"
        example_dir.mkdir(parents=True, exist_ok=True)

        example_config = {
            "lycoris_config": "config/examples/pixart.lycoris-lokr/lycoris_config.json",
            "data_backend_config": "config/examples/pixart.lycoris-lokr/multidatabackend.json",
        }

        (example_dir / "config.json").write_text(json.dumps(example_config, indent=2), encoding="utf-8")
        (example_dir / "lycoris_config.json").write_text("{}", encoding="utf-8")
        (example_dir / "multidatabackend.json").write_text("{}", encoding="utf-8")

        example_info = ExampleConfigInfo(
            name="demo",
            config_path=example_dir / "config.json",
            defaults={},
            description=None,
            dataloader_path=None,
            dataloader_payload=None,
        )

        request = SimpleNamespace(
            name="tidal-timber",
            model_family="testfamily",
            model_flavour="default",
            model_type="lora",
            lora_type="lycoris",
            description=None,
            example="demo",
            dataloader_path=None,
            create_dataloader=True,
        )

        with (
            patch.object(ConfigsService, "_get_store", return_value=self.model_store),
            patch(
                "simpletuner.simpletuner_sdk.server.services.configs_service.EXAMPLE_CONFIGS_SERVICE.get_example",
                return_value=example_info,
            ),
        ):
            result = service.create_environment(request)

        env_config_path = self.config_root / "tidal-timber" / "config.json"
        self.assertTrue(env_config_path.exists())

        with env_config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        expected_relative = "tidal-timber/lycoris_config.json"
        self.assertEqual(payload.get("lycoris_config"), expected_relative)
        self.assertNotIn("--lycoris_config", payload)
        self.assertEqual(result["config"].get("lycoris_config"), expected_relative)

        lycoris_abs = self.config_root / expected_relative
        self.assertTrue(lycoris_abs.exists())

        stray_root = self.config_root / "lycoris_config.json.json"
        self.assertFalse(stray_root.exists())
        self.assertFalse(stray_root.with_suffix(".metadata.json").exists())

    def test_prompt_library_paths_are_saved_absolute(self) -> None:
        service = ConfigsService()
        libraries_dir = self.config_root / "validation_prompt_libraries"
        libraries_dir.mkdir(parents=True, exist_ok=True)
        library_path = libraries_dir / "user_prompt_library-smoke.json"
        library_path.write_text("{}", encoding="utf-8")

        form_data = {
            "user_prompt_library": f"validation_prompt_libraries/{library_path.name}",
        }

        normalized = service.normalize_form_to_config(form_data, configs_dir=str(self.config_root))
        self.assertEqual(normalized.get("--user_prompt_library"), str(library_path))

    def test_example_environment_copies_dataloader_payload(self) -> None:
        service = ConfigsService()

        example_dir = Path(self._tempdir.name) / "example-dataloader"
        example_dir.mkdir(parents=True, exist_ok=True)
        config_payload = {
            "data_backend_config": "config/examples/multidatabackend-demo.json",
            "--model_family": "testfamily",
        }
        (example_dir / "config.json").write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

        dataloader_payload = [{"id": "demo-images", "type": "local", "instance_data_dir": "/data/demo"}]
        dataloader_path = example_dir / "multidatabackend-demo.json"
        dataloader_path.write_text(json.dumps(dataloader_payload, indent=2), encoding="utf-8")
        example_info = ExampleConfigInfo(
            name="demo-dataloader",
            config_path=example_dir / "config.json",
            defaults={"model_family": "testfamily", "model_type": "lora"},
            description=None,
            dataloader_path=dataloader_path,
            dataloader_payload=None,
        )

        request = SimpleNamespace(
            name="ember-grove",
            model_family="testfamily",
            model_flavour="default",
            model_type="lora",
            lora_type="standard",
            description=None,
            example="demo-dataloader",
            dataloader_path=None,
            create_dataloader=True,
        )

        with (
            patch.object(ConfigsService, "_get_store", return_value=self.model_store),
            patch(
                "simpletuner.simpletuner_sdk.server.services.configs_service.EXAMPLE_CONFIGS_SERVICE.get_example",
                return_value=example_info,
            ),
        ):
            result = service.create_environment(request)

        dataloader_file = self.config_root / "ember-grove" / "multidatabackend.json"
        self.assertTrue(dataloader_file.exists(), "Example dataloader JSON should be copied into environment")

        with dataloader_file.open("r", encoding="utf-8") as handle:
            written_payload = json.load(handle)

        self.assertEqual(written_payload, dataloader_payload)
        expected_rel = "ember-grove/multidatabackend.json"
        self.assertEqual(result["dataloader"]["path"], expected_rel)
        self.assertEqual(result["dataloader"]["absolute_path"], str(dataloader_file))

    def test_create_environment_rejects_json_suffix_name(self) -> None:
        service = ConfigsService()

        request = SimpleNamespace(
            name="invalid-name.json",
            model_family="testfamily",
            model_flavour="default",
            model_type="lora",
            lora_type="lycoris",
            description=None,
            example=None,
            dataloader_path=None,
            create_dataloader=True,
        )

        with patch.object(ConfigsService, "_get_store", return_value=self.model_store):
            # The implementation now strips .json suffix instead of rejecting it
            result = service.create_environment(request)

        # Verify the environment was created without the .json suffix
        self.assertIn("environment", result)
        self.assertEqual(result["environment"]["name"], "invalid-name")  # .json was stripped

    def test_config_store_rejects_json_suffix_directly(self) -> None:
        store = ConfigStore(config_dir=self.config_root, config_type="model")
        metadata = store.save_config(
            "bad.json",
            config={"--model_family": "testfamily"},
            metadata=None,
            overwrite=False,
        )
        self.assertEqual(metadata.name, "bad")

    def test_webui_defaults_clears_missing_active_config(self) -> None:
        webui_dir = self.config_root / "webui_state"
        webui_dir.mkdir(parents=True, exist_ok=True)
        defaults_path = webui_dir / "defaults.json"
        defaults_payload = {
            "active_config": "ghost-config",
            "configs_dir": str(self.config_root),
            "output_dir": str(self.config_root / "output"),
            "event_polling_interval": 5,
            "event_stream_enabled": True,
            "auto_preserve_defaults": True,
            "theme": "dark",
        }
        defaults_path.write_text(json.dumps(defaults_payload), encoding="utf-8")

        store = WebUIStateStore(base_dir=webui_dir)
        loaded = store.load_defaults()
        self.assertIsNone(loaded.active_config)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
