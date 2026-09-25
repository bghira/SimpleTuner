import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from PIL import Image

from simpletuner.helpers.data_backend.caption_generator import CaptionImageGenerator, generated_image_backend_configs
from simpletuner.helpers.data_backend.config import create_backend_config
from simpletuner.helpers.data_backend.config.validators import validate_caption_data_generator
from simpletuner.helpers.data_backend.factory import FactoryRegistry
from simpletuner.helpers.metadata.captions import CaptionRecord
from simpletuner.helpers.training.state_tracker import StateTracker


class _Pipeline:
    def __init__(self, limits=None):
        self.calls = []
        self.limits = limits or {}
        self.transformer = None
        self.encoder = torch.nn.Linear(1, 1)
        self.failure = None

    def to(self, _device):
        raise AssertionError("The generator must preserve placement of the resident pipeline components.")

    def __call__(self, **kwargs):
        width, height = kwargs["width"], kwargs["height"]
        seeds = [generator.initial_seed() for generator in kwargs["generator"]]
        self.calls.append((width, height, list(kwargs["prompt"]), seeds))
        self.assert_base_state()
        if self.failure is not None:
            raise self.failure
        if len(seeds) > self.limits.get((width, height), 100):
            # Consume generators before failure, as a real OOM may happen after noise allocation.
            for generator in kwargs["generator"]:
                torch.rand(1, generator=generator)
            raise torch.OutOfMemoryError("CUDA out of memory")
        return SimpleNamespace(images=[Image.new("RGB", (width, height), (seed % 256, 0, 0)) for seed in seeds])

    def assert_base_state(self):
        if self.transformer.training or torch.is_grad_enabled():
            raise AssertionError("Generation must run in eval mode with gradients disabled.")


class _Model:
    MODEL_TYPE = SimpleNamespace(value="transformer")

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.pipelines = {"text2img": pipeline}
        self.model = None
        self.config = SimpleNamespace(pretrained_model_name_or_path="org/model", model_flavour="2.1")
        self.accelerator = SimpleNamespace(device=torch.device("cpu"))
        self.loads = 0

    def load_model(self, move_to_device):
        self.loads += 1
        self.model = torch.nn.Linear(1, 1)

    def get_trained_component(self):
        return self.model

    def get_pipeline(self, load_base_model):
        if load_base_model:
            raise AssertionError("Generation must reuse the resident model components.")
        self.pipeline.transformer = self.model
        return self.pipeline

    def unwrap_model(self, model):
        return model

    def update_pipeline_call_kwargs(self, kwargs):
        return kwargs


class CaptionImageGeneratorTests(unittest.TestCase):
    def setUp(self):
        identity = patch("simpletuner.helpers.data_backend.caption_generator.HfApi")
        identity.start().return_value.model_info.return_value.sha = "model-commit"
        self.addCleanup(identity.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.records = [CaptionRecord(str(index), f"caption {index}", "prompts") for index in range(5)]
        self.config = {"batch_size": 4, "resolutions": ["64x64", "128x64"], "seed": 31}

    def generator(self, pipeline=None, config=None, records=None):
        model = _Model(pipeline or _Pipeline())
        return CaptionImageGenerator(model, config or self.config, records or self.records, self.temp.name)

    def test_oom_retries_same_seeds_and_remembers_each_bucket(self):
        pipeline = _Pipeline({(64, 64): 2, (128, 64): 1})
        generator = self.generator(pipeline)
        directories = generator.generate()
        self.assertEqual(generator.manifest["batch_sizes"], {"64x64": 2, "128x64": 1})
        self.assertEqual(pipeline.calls[0][3], [31, 32, 33, 34])
        self.assertEqual(pipeline.calls[1][3], [31, 32])
        self.assertEqual(pipeline.calls[2][3], [33, 34])
        for bucket_index, directory in enumerate(directories.values()):
            files = sorted(Path(directory).glob("*.png"))
            self.assertEqual(len(files), 5)
            for index, path in enumerate(files):
                with Image.open(path) as image:
                    self.assertEqual(image.getpixel((0, 0))[0], 31 + bucket_index * 5 + index)
                self.assertEqual(path.with_suffix(".txt").read_text(), f"caption {index}")
        self.assertIsNone(generator.model.model)
        self.assertIsNone(pipeline.transformer)
        self.assertIsNotNone(pipeline.encoder)
        self.assertEqual(generator.model.loads, 1)

        # An incomplete prior write is retried with its original seed and the known bucket limit.
        (Path(directories["128x64"]) / "00000000.txt").unlink()
        resumed_pipeline = _Pipeline({(64, 64): 2, (128, 64): 1})
        resumed = self.generator(resumed_pipeline)
        resumed.generate()
        self.assertEqual(resumed_pipeline.calls, [(128, 64, ["caption 0"], [36])])

    def test_complete_cache_skips_model_loading(self):
        generator = self.generator()
        generator.generate()
        cached = self.generator()
        cached.generate()
        self.assertEqual(cached.model.loads, 0)
        self.assertFalse(cached.model.pipeline.calls)

    def test_batch_one_oom_surfaces_and_does_not_write_sample(self):
        pipeline = _Pipeline({(64, 64): 0})
        generator = self.generator(pipeline)
        with self.assertRaises(torch.OutOfMemoryError):
            generator.generate()
        self.assertFalse(list(generator.output_dir.rglob("*.png")))
        self.assertIsNone(generator.model.model)

    def test_unrelated_runtime_error_is_not_retried(self):
        pipeline = _Pipeline()
        pipeline.failure = RuntimeError("invalid tensor dimensions")
        generator = self.generator(pipeline)
        with self.assertRaisesRegex(RuntimeError, "invalid tensor"):
            generator.generate()
        self.assertEqual(len(pipeline.calls), 1)
        self.assertIsNone(generator.model.model)

    def test_existing_model_training_state_is_restored(self):
        generator = self.generator()
        model = torch.nn.Linear(1, 1)
        model.weight.requires_grad_(False)
        generator.model.model = model
        generator.generate()
        self.assertIs(generator.model.model, model)
        self.assertTrue(model.training)
        self.assertFalse(model.weight.requires_grad)
        self.assertTrue(model.bias.requires_grad)
        self.assertEqual(generator.model.loads, 0)

    def test_recipe_or_caption_changes_isolate_outputs_but_batch_size_does_not(self):
        original = self.generator().output_dir
        self.assertEqual(self.generator(config={**self.config, "batch_size": 2}).output_dir, original)
        self.assertNotEqual(self.generator(config={**self.config, "seed": 2}).output_dir, original)
        changed = self.records.copy()
        changed[0] = CaptionRecord("0", "changed caption", "prompts")
        self.assertNotEqual(self.generator(records=changed).output_dir, original)

    def test_manifest_does_not_contain_model_or_source_paths(self):
        generator = self.generator()
        generator.model.config.pretrained_model_name_or_path = "/private/model"
        generator.generate()
        manifest = generator.manifest_path.read_text()
        self.assertNotIn("/private/", manifest)
        self.assertNotIn(self.temp.name, manifest)

    def test_qwen_guidance_uses_native_parameter_and_ignores_validation_guidance(self):
        calls = []

        class QwenPipeline(_Pipeline):
            def __call__(
                self,
                prompt,
                width,
                height,
                num_inference_steps,
                generator,
                output_type,
                true_cfg_scale=1.0,
                negative_prompt=None,
            ):
                calls.append((true_cfg_scale, negative_prompt))
                return super().__call__(prompt=prompt, width=width, height=height, generator=generator)

        generator = self.generator(QwenPipeline(), config={**self.config, "guidance_scale": 2.5})
        generator.model.update_pipeline_call_kwargs = lambda kwargs: {**kwargs, "true_cfg_scale": 7.0}
        generator.generate()
        self.assertTrue(calls)
        self.assertTrue(all(scale == 2.5 and all(prompt == "" for prompt in negatives) for scale, negatives in calls))

    def test_local_weights_stat_change_invalidates_cache_identity(self):
        path = Path(self.temp.name) / "model.safetensors"
        path.write_bytes(b"first")
        first = CaptionImageGenerator._source_identity(str(path), None)
        path.write_bytes(b"changed")
        self.assertNotEqual(CaptionImageGenerator._source_identity(str(path), None), first)

    def test_generated_buckets_keep_native_training_resolution(self):
        configs = generated_image_backend_configs(
            {
                "id": "captions",
                "type": "huggingface",
                "dataset_type": "caption",
                "data_generator": self.config,
                "resolution": 1024,
                "resolution_type": "pixel_area",
                "crop": True,
                "probability": 0.6,
                "cache_dir_vae": "cache/vae",
                "huggingface": {"path": "org/captions"},
            },
            {"512x512": "generated/512x512", "768x1024": "generated/768x1024"},
        )
        self.assertEqual([config["resolution"] for config in configs], [512, 768])
        for config in configs:
            self.assertEqual(config["type"], "local")
            self.assertEqual(config["dataset_type"], "image")
            self.assertEqual(config["resolution_type"], "pixel")
            self.assertFalse(config["crop"])
            self.assertEqual(config["probability"], 0.3)
            self.assertNotIn("data_generator", config)
            self.assertNotIn("huggingface", config)
            self.assertEqual(config["caption_strategy"], "textfile")
        self.assertNotEqual(configs[0]["cache_dir_vae"], configs[1]["cache_dir_vae"])

    def test_qwen_aspect_ratio_buckets_preserve_32_pixel_alignment(self):
        resolutions = ["384x672", "672x384"]
        generator = self.generator(config={**self.config, "resolutions": resolutions})
        directories = generator.generate()
        self.assertEqual(list(directories), resolutions)
        for resolution, directory in directories.items():
            expected = tuple(map(int, resolution.split("x")))
            images = list(Path(directory).glob("*.png"))
            self.assertEqual(len(images), len(self.records))
            for path in images:
                with Image.open(path) as image:
                    self.assertEqual(image.size, expected)


class CaptionGeneratorConfigTests(unittest.TestCase):
    def test_rejects_generator_on_every_non_caption_dataset_type(self):
        for dataset_type in (
            "image",
            "video",
            "audio",
            "conditioning",
            "eval",
            "text_embeds",
            "image_embeds",
            "distillation_cache",
        ):
            with self.subTest(dataset_type=dataset_type), self.assertRaisesRegex(ValueError, "only valid"):
                create_backend_config(
                    {
                        "id": "data",
                        "type": "local",
                        "dataset_type": dataset_type,
                        "data_generator": {"resolutions": ["512x512"]},
                    },
                    {},
                )

    def test_caption_config_round_trip(self):
        config = create_backend_config(
            {
                "id": "prompts",
                "type": "local",
                "dataset_type": "caption",
                "data_generator": {"resolutions": ["512x512"], "batch_size": 3},
            },
            {},
        )
        config.validate({})
        self.assertEqual(config.to_dict()["config"]["data_generator"]["batch_size"], 3)

    def test_webui_accepts_generated_captions_and_rejects_invalid_generator(self):
        from simpletuner.simpletuner_sdk.server.services.dataset_plan import compute_validations

        config = [
            {
                "id": "prompts",
                "type": "local",
                "dataset_type": "caption",
                "instance_data_dir": "data/prompts",
                "data_generator": {"resolutions": ["512x512"]},
            },
            {"id": "text", "type": "local", "dataset_type": "text_embeds", "default": True, "cache_dir": "cache/text"},
        ]
        errors = [message for message in compute_validations(config) if message.level == "error"]
        self.assertFalse(errors, errors)
        config[0]["data_generator"]["batch_size"] = 0
        errors = [message for message in compute_validations(config) if message.level == "error"]
        self.assertTrue(any(message.field == "prompts.data_generator" for message in errors))
        config[0]["data_generator"]["batch_size"] = 1
        config[0]["dataset_type"] = "image"
        errors = [message for message in compute_validations(config) if message.level == "error"]
        self.assertTrue(any("only valid" in message.message for message in errors))

    def test_rejects_invalid_recipes(self):
        values = [
            False,
            [],
            {},
            {"resolutions": []},
            {"resolutions": [512]},
            {"resolutions": ["500x512"]},
            {"resolutions": ["48x64"]},
            {"resolutions": ["0x64"]},
            {"resolutions": ["512x512", "512x512"]},
            {"resolutions": ["512x512"], "batch_size": 0},
            {"resolutions": ["512x512"], "batch_size": True},
            {"resolutions": ["512x512"], "seed": -1},
            {"resolutions": ["512x512"], "guidance_scale": float("nan")},
            {"resolutions": ["512x512"], "num_inference_steps": 1.5},
            {"resolutions": ["512x512"], "unknown": 1},
        ]
        for value in values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_caption_data_generator(value)

    def test_factory_accepts_generated_captions_without_a_distiller(self):
        with tempfile.TemporaryDirectory() as directory:
            factory = FactoryRegistry.__new__(FactoryRegistry)
            factory.args = SimpleNamespace(cache_dir=directory, distillation_method=None)
            factory.model = _Model(_Pipeline())
            factory.accelerator = SimpleNamespace(is_main_process=True, wait_for_everyone=MagicMock())
            factory.metrics = {"backend_counts": {}}
            factory.text_embed_backends = {"text": {}}
            factory.caption_backends = {}
            factory.data_backends = {}
            factory.distiller_requirement_profile = None
            factory.distillation_method = None
            factory._log_performance_metrics = MagicMock()
            for name in (
                "_requires_conditioning_dataset",
                "_requires_s2v_datasets",
                "_connect_conditioning_datasets",
                "_connect_s2v_datasets",
            ):
                setattr(factory, name, MagicMock(return_value=False))
            for name in (
                "synchronize_conditioning_settings",
                "_validate_audio_only_datasets",
                "_validate_edit_model_conditioning_type",
                "_process_deferred_text_embeddings",
            ):
                setattr(factory, name, MagicMock())
            factory._configure_caption_backend = MagicMock()
            record = CaptionRecord("one", "a fox", "prompts")
            factory._build_caption_backend = MagicMock(
                return_value={"metadata_backend": SimpleNamespace(iter_records=lambda: iter([record]))}
            )

            def configure_image(backend, *_args):
                self.assertEqual(backend["dataset_type"], "image")
                image_paths = list(Path(backend["instance_data_dir"]).glob("*.png"))
                self.assertEqual(len(image_paths), 1)
                factory.data_backends[backend["id"]] = backend

            factory._configure_single_data_backend = MagicMock(side_effect=configure_image)
            config = [
                {
                    "id": "prompts",
                    "type": "local",
                    "dataset_type": "caption",
                    "instance_data_dir": "data/prompts",
                    "data_generator": {"resolutions": ["64x64", "128x64"]},
                }
            ]
            with (
                patch("simpletuner.helpers.data_backend.factory.StateTracker") as state,
                patch("simpletuner.helpers.data_backend.caption_generator.HfApi") as api,
            ):
                state.get_data_backends.return_value = {}
                api.return_value.model_info.return_value.sha = "model-commit"
                self.assertFalse(factory._caption_batches_supported())
                factory.configure_data_backends(config)
            self.assertEqual(factory._configure_single_data_backend.call_count, 2)
            factory._configure_caption_backend.assert_not_called()
            self.assertEqual(factory.metrics["backend_counts"]["data_backends"], 2)
            self.assertEqual(factory.metrics["backend_counts"]["caption"], 0)
            self.assertEqual(factory.model.loads, 1)

    def test_rejects_generated_outputs_inside_source_before_caption_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "captions"
            source.mkdir()
            (source / "prompts.jsonl").write_text('"a fox"\n', encoding="utf-8")
            generated = source / "generated"
            generated.mkdir()
            (generated / "00000000.txt").write_text("must not become another prompt", encoding="utf-8")
            factory = FactoryRegistry.__new__(FactoryRegistry)
            factory.args = SimpleNamespace(cache_dir=str(source / "cache"))
            factory._build_caption_backend = MagicMock()
            for output_dir in (str(source), str(generated), None):
                recipe = {"resolutions": ["512x512"]}
                if output_dir is not None:
                    recipe["output_dir"] = output_dir
                with self.subTest(output_dir=output_dir), self.assertRaisesRegex(ValueError, "outside instance_data_dir"):
                    factory._materialize_caption_generators(
                        [
                            {
                                "id": "prompts",
                                "type": "local",
                                "dataset_type": "caption",
                                "instance_data_dir": str(source),
                                "data_generator": recipe,
                            }
                        ]
                    )
            factory._build_caption_backend.assert_not_called()

    def test_factory_materializes_before_normal_image_configuration(self):
        factory = FactoryRegistry.__new__(FactoryRegistry)
        factory.args = SimpleNamespace(cache_dir="cache")
        factory.model = _Model(_Pipeline())
        factory.accelerator = SimpleNamespace(is_main_process=True, wait_for_everyone=MagicMock())
        record = CaptionRecord("one", "a fox", "prompts")

        def build_caption(backend):
            StateTracker.set_data_backend_config(backend["id"], dict(backend))
            return {"metadata_backend": SimpleNamespace(iter_records=lambda: iter([record]))}

        factory._build_caption_backend = MagicMock(side_effect=build_caption)
        config = [
            {
                "id": "prompts",
                "type": "local",
                "dataset_type": "caption",
                "instance_data_dir": "data/prompts",
                "data_generator": {"resolutions": ["512x512", "768x1024"]},
            }
        ]
        with (
            patch("simpletuner.helpers.data_backend.caption_generator.CaptionImageGenerator") as generator,
            patch.object(StateTracker, "data_backends", {}),
        ):
            generator.return_value.bucket_directories.return_value = {"512x512": "cache/a", "768x1024": "cache/b"}
            factory._materialize_caption_generators(config)
            self.assertNotIn("prompts", StateTracker.data_backends)
        generator.return_value.generate.assert_called_once_with()
        self.assertEqual([entry["dataset_type"] for entry in config], ["image", "image"])
        self.assertEqual([entry["id"] for entry in config], ["prompts-generated-512x512", "prompts-generated-768x1024"])
        factory.accelerator.wait_for_everyone.assert_not_called()

    def test_multiprocess_generator_rejected_before_caption_discovery_on_every_rank(self):
        for is_main_process in (True, False):
            with self.subTest(is_main_process=is_main_process):
                factory = FactoryRegistry.__new__(FactoryRegistry)
                factory.accelerator = SimpleNamespace(
                    is_main_process=is_main_process, num_processes=2, wait_for_everyone=MagicMock()
                )
                factory._build_caption_backend = MagicMock()
                config = [{"id": "prompts", "dataset_type": "caption", "data_generator": {"resolutions": ["512x512"]}}]
                with self.assertRaisesRegex(ValueError, "single-process preparation"):
                    factory._materialize_caption_generators(config)
                factory._build_caption_backend.assert_not_called()
                factory.accelerator.wait_for_everyone.assert_not_called()
                self.assertEqual(config[0]["dataset_type"], "caption")

    def test_multiprocess_training_allows_disabled_generators_and_generated_image_datasets(self):
        factory = FactoryRegistry.__new__(FactoryRegistry)
        factory.accelerator = SimpleNamespace(num_processes=2)
        factory._build_caption_backend = MagicMock()
        config = [
            {"id": "images", "dataset_type": "image", "instance_data_dir": "data/generated/512x512"},
            {"id": "prompts", "dataset_type": "caption", "disabled": True, "data_generator": {"resolutions": ["512x512"]}},
        ]
        original = list(config)
        factory._materialize_caption_generators(config)
        self.assertEqual(config, original)
        factory._build_caption_backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
