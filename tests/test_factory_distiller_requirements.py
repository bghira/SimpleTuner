#!/usr/bin/env python
"""Focused tests for FactoryRegistry's distiller requirement enforcement."""

import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.data_backend.factory import FactoryRegistry
from simpletuner.helpers.distillation.requirements import DataRequirement, DistillerRequirementProfile


class DummyConfig:
    """Minimal config stub for patched create_backend_config."""

    def __init__(self, backend):
        self.backend = backend
        self.id = backend.get("id")
        self.backend_type = backend.get("type", "local")
        self.dataset_type = backend.get("dataset_type")
        self.config = dict(backend)

    def validate(self, *_args, **_kwargs):
        return None

    def to_dict(self):
        return {"config": dict(self.config)}


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
            vae_cache_disable=False,
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

        from simpletuner.helpers.training.state_tracker import StateTracker

        self.enterContext(patch.object(StateTracker, "args", self.args))
        self.enterContext(patch.object(StateTracker, "accelerator", self.accelerator))
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

        self.build_backend_from_config_patch = patch(
            "simpletuner.helpers.data_backend.factory.build_backend_from_config",
            side_effect=lambda *_args, **_kwargs: {
                "data_backend": MagicMock(),
                "metadata_backend": MagicMock(),
                "instance_data_dir": "/data/captions",
                "config": {},
            },
        )
        self.build_backend_from_config_patch.start()
        self.addCleanup(self.build_backend_from_config_patch.stop)

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
