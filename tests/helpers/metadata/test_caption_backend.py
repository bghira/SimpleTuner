import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

torch.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
    is_available=lambda: False,
    get_device_properties=lambda *_: SimpleNamespace(major=0),
)
torch.backends = types.SimpleNamespace(  # type: ignore[attr-defined]
    mps=SimpleNamespace(is_available=lambda: False)
)
if not hasattr(torch, "no_grad"):
    class _NoGradContext:
        def __call__(self, func):
            def wrapped(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapped

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _no_grad(func=None):
        context = _NoGradContext()
        if func is not None:
            return context(func)
        return context

    torch.no_grad = _no_grad
sys.modules.setdefault(
    "torch.distributed",
    SimpleNamespace(is_initialized=lambda: False, is_available=lambda: False),
)
dummy_transforms = SimpleNamespace(Compose=lambda *args, **kwargs: None)
sys.modules.setdefault("torchvision", SimpleNamespace(transforms=dummy_transforms))
sys.modules.setdefault("torchvision.transforms", dummy_transforms)
nn_functional = SimpleNamespace()
nn_module = SimpleNamespace(functional=nn_functional)
sys.modules.setdefault("torch.nn", nn_module)
sys.modules.setdefault("torch.nn.functional", nn_functional)
diffusers_module = types.ModuleType("diffusers")
diffusers_module.DiffusionPipeline = object
sys.modules.setdefault("diffusers", diffusers_module)
diffusers_optimization = types.ModuleType("diffusers.optimization")
diffusers_optimization.get_scheduler = lambda *args, **kwargs: None
sys.modules.setdefault("diffusers.optimization", diffusers_optimization)
diffusers_pipelines = types.ModuleType("diffusers.pipelines")
sys.modules.setdefault("diffusers.pipelines", diffusers_pipelines)
diffusers_flux = types.ModuleType("diffusers.pipelines.flux")
sys.modules.setdefault("diffusers.pipelines.flux", diffusers_flux)
diffusers_flux_pipeline = types.ModuleType("diffusers.pipelines.flux.pipeline_flux")
diffusers_flux_pipeline.calculate_shift = lambda *args, **kwargs: None
sys.modules.setdefault("diffusers.pipelines.flux.pipeline_flux", diffusers_flux_pipeline)
diffusers_image_processor = types.ModuleType("diffusers.image_processor")
diffusers_image_processor.PipelineImageInput = object
diffusers_image_processor.VaeImageProcessor = object
sys.modules.setdefault("diffusers.image_processor", diffusers_image_processor)
diffusers_loaders = types.ModuleType("diffusers.loaders")
diffusers_loaders.FluxIPAdapterMixin = object
diffusers_loaders.FromSingleFileMixin = object
diffusers_loaders.TextualInversionLoaderMixin = object
sys.modules.setdefault("diffusers.loaders", diffusers_loaders)
diffusers_loaders_lora = types.ModuleType("diffusers.loaders.lora_base")
diffusers_loaders_lora.LoraBaseMixin = object
diffusers_loaders_lora._fetch_state_dict = lambda *args, **kwargs: {}
sys.modules.setdefault("diffusers.loaders.lora_base", diffusers_loaders_lora)
common_models = types.ModuleType("simpletuner.helpers.models.common")
common_models.ImageModelFoundation = object
common_models.VideoModelFoundation = object
sys.modules.setdefault("simpletuner.helpers.models.common", common_models)
flux_models = types.ModuleType("simpletuner.helpers.models.flux")
flux_models.calculate_shift_flux = lambda *args, **kwargs: None
flux_models.FluxPipeline = object
sys.modules.setdefault("simpletuner.helpers.models.flux", flux_models)
sys.modules.setdefault("peft", SimpleNamespace(LoraConfig=object, TaskType=object))
sys.modules.setdefault("torch.distributions", SimpleNamespace(Beta=object))
transformers_utils = SimpleNamespace(ContextManagers=object)
sys.modules.setdefault("transformers", SimpleNamespace(utils=transformers_utils))
sys.modules.setdefault("transformers.utils", transformers_utils)
sys.modules.setdefault("safetensors", SimpleNamespace(torch=SimpleNamespace()))
sys.modules.setdefault("safetensors.torch", SimpleNamespace(load_file=lambda *args, **kwargs: {}))
sys.modules.setdefault("accelerate", SimpleNamespace(Accelerator=object))
sys.modules.setdefault("torch.optim", SimpleNamespace(lr_scheduler=SimpleNamespace()))
sys.modules.setdefault("torch.optim.lr_scheduler", SimpleNamespace(LambdaLR=object, LRScheduler=object))

from simpletuner.helpers.metadata.backends.caption import CaptionMetadataBackend
from simpletuner.helpers.training.state_tracker import StateTracker


class InMemoryDataBackend:
    def __init__(self):
        self.id = "caption"
        self._storage = {}

    def _key(self, identifier):
        return str(identifier)

    def read(self, identifier):
        key = self._key(identifier)
        if key not in self._storage:
            raise FileNotFoundError(identifier)
        return self._storage[key]

    def write(self, identifier, data):
        identifier = self._key(identifier)
        if isinstance(data, str):
            data = data.encode("utf-8")
        self._storage[identifier] = data

    def exists(self, identifier):
        return self._key(identifier) in self._storage

    def list_files(self, instance_data_dir, file_extensions):
        results = []
        for path in self._storage:
            if not path.startswith(instance_data_dir):
                continue
            suffix = Path(path).suffix.lstrip(".").lower()
            if file_extensions and suffix not in file_extensions:
                continue
            parent = str(Path(path).parent)
            results.append((parent, [], [path]))
        return results

    def get_abs_path(self, sample_path):
        return sample_path


class TestCaptionMetadataBackend(unittest.TestCase):
    def setUp(self):
        StateTracker.set_args(SimpleNamespace(output_dir="/tmp"))
        StateTracker.data_backends = {}
        StateTracker.all_caption_files = {}
        self.backend = InMemoryDataBackend()
        self.accelerator = SimpleNamespace(is_local_main_process=True)
        self.instance_dir = "/datasets/captions"
        self.cache_file = "/cache/captions.cache"
        self.metadata_file = "/cache/captions.meta"

    def _create_backend(self):
        return CaptionMetadataBackend(
            id="caption",
            instance_data_dir=self.instance_dir,
            cache_file=self.cache_file,
            metadata_file=self.metadata_file,
            data_backend=self.backend,
            accelerator=self.accelerator,
            batch_size=1,
        )

    def test_ingest_from_various_formats(self):
        self.backend.write(f"{self.instance_dir}/basic.txt", "first caption\n\nsecond")
        json_payload = json.dumps({"captions": ["json one", "json two"]})
        self.backend.write(f"{self.instance_dir}/data.json", json_payload)
        jsonl_payload = '\n'.join([json.dumps({"caption": "line a"}), '"line b"'])
        self.backend.write(f"{self.instance_dir}/mixed.jsonl", jsonl_payload)

        caption_backend = self._create_backend()
        file_cache = {
            f"{self.instance_dir}/basic.txt": False,
            f"{self.instance_dir}/data.json": False,
            f"{self.instance_dir}/mixed.jsonl": False,
        }

        created = caption_backend.ingest_from_file_cache(file_cache)
        self.assertEqual(created, 6)
        self.assertEqual(len(list(caption_backend.iter_records())), 6)

        payload = self.backend.read(str(caption_backend.metadata_file))
        stored = json.loads(payload.decode("utf-8"))
        self.assertEqual(len(stored), 6)
        self.assertTrue(all(entry["caption_text"] for entry in stored))

    def test_metadata_round_trip(self):
        self.backend.write(f"{self.instance_dir}/only.txt", "one caption")
        caption_backend = self._create_backend()
        caption_backend.ingest_from_file_cache({f"{self.instance_dir}/only.txt": False})

        reloaded_backend = self._create_backend()
        reloaded_backend.load_image_metadata()
        records = list(reloaded_backend.iter_records())

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].caption_text, "one caption")
        self.assertEqual(records[0].data_backend_id, "caption")


if __name__ == "__main__":
    unittest.main()
