import inspect
import math
import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.common import ModelFoundation
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.training.validation import Evaluation


class _DummyModel(ModelFoundation):
    # Bypass abstract __init__ by using __new__ in tests.
    PREDICTION_TYPE = None
    MODEL_TYPE = None
    NAME = "Dummy"
    DEFAULT_PIPELINE_TYPE = None
    PIPELINE_CLASSES = {}
    VALIDATION_USES_NEGATIVE_PROMPT = False

    def model_predict(self, prepared_batch, custom_timesteps: list = None):
        raise NotImplementedError

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        raise NotImplementedError

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        raise NotImplementedError

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        raise NotImplementedError

    def _get_patch_size_for_dynamic_shift(self, noise_scheduler):
        return getattr(self, "_test_patch_size", None)


class _DefaultPatchModel(ModelFoundation):
    # Bypass abstract __init__ by using __new__ in tests.
    PREDICTION_TYPE = None
    MODEL_TYPE = None
    NAME = "DefaultPatch"
    DEFAULT_PIPELINE_TYPE = None
    PIPELINE_CLASSES = {}
    VALIDATION_USES_NEGATIVE_PROMPT = False

    def model_predict(self, prepared_batch, custom_timesteps: list = None):
        raise NotImplementedError

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        raise NotImplementedError

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        raise NotImplementedError

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        raise NotImplementedError


class DynamicShiftTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = MagicMock()
        self.scheduler.config = types.SimpleNamespace(
            base_image_seq_len=256,
            max_image_seq_len=512,
            base_shift=0.5,
            max_shift=1.0,
        )

    def test_calculate_dynamic_shift_mu_uses_patch_size_and_resolution(self):
        model = _DummyModel.__new__(_DummyModel)
        model._test_patch_size = 4
        model.config = types.SimpleNamespace()

        latents = torch.zeros(1, 4, 8, 8)
        mu = model.calculate_dynamic_shift_mu(self.scheduler, latents)

        expected = 0.5 / (512 - 256) * 4  # linear shift per seq len
        self.assertAlmostEqual(mu, expected)

    def test_calculate_dynamic_shift_mu_uses_temporal_patch_size_for_video_latents(self):
        model = _DummyModel.__new__(_DummyModel)
        model._test_patch_size = (4, 2, 2)
        model.config = types.SimpleNamespace()

        latents = torch.zeros(1, 4, 8, 4, 4)
        mu = model.calculate_dynamic_shift_mu(self.scheduler, latents)

        expected = 0.5 / (512 - 256) * 8  # (8 / 4) * (4 / 2) * (4 / 2)
        self.assertAlmostEqual(mu, expected)

    def test_calculate_dynamic_shift_mu_uses_component_patch_size_t(self):
        model = _DefaultPatchModel.__new__(_DefaultPatchModel)
        model.config = types.SimpleNamespace(controlnet=False)
        model.accelerator = None
        model.model = types.SimpleNamespace(config=types.SimpleNamespace(patch_size=2, patch_size_t=4))

        latents = torch.zeros(1, 4, 8, 4, 4)
        mu = model.calculate_dynamic_shift_mu(self.scheduler, latents)

        expected = 0.5 / (512 - 256) * 8  # (8 / 4) * (4 / 2) * (4 / 2)
        self.assertAlmostEqual(mu, expected)

    def test_calculate_dynamic_shift_mu_errors_when_config_missing(self):
        model = _DummyModel.__new__(_DummyModel)
        model._test_patch_size = 2
        model.config = types.SimpleNamespace()

        bad_scheduler = MagicMock()
        bad_scheduler.config = types.SimpleNamespace(
            base_image_seq_len=None,
            max_image_seq_len=512,
            base_shift=0.5,
            max_shift=1.0,
        )
        with self.assertRaises(ValueError):
            model.calculate_dynamic_shift_mu(bad_scheduler, torch.zeros(1, 4, 4, 4))

    @patch("simpletuner.helpers.training.validation.StateTracker")
    def test_evaluation_passes_mu_from_model(self, mock_state_tracker):
        config = types.SimpleNamespace(flow_schedule_auto_shift=True, eval_timesteps=3)
        mock_state_tracker.get_args.return_value = config

        eval_helper = Evaluation(accelerator=types.SimpleNamespace(is_main_process=True))
        mock_model = MagicMock()
        mock_model.calculate_dynamic_shift_mu.return_value = 0.42
        mock_state_tracker.get_model.return_value = mock_model

        class _Scheduler:
            def __init__(self):
                self.config = types.SimpleNamespace(use_dynamic_shifting=True)
                self.timesteps = [1, 2, 3]
                self.calls = []

            def set_timesteps(self, num_inference_steps=None, mu=None, **kwargs):
                self.calls.append({"num": num_inference_steps, "mu": mu, "kwargs": kwargs})

        scheduler = _Scheduler()

        result = eval_helper.get_timestep_schedule(scheduler, latents=torch.zeros(1, 4, 4, 4))

        self.assertEqual(len(scheduler.calls), 1)
        self.assertEqual(scheduler.calls[0]["num"], config.eval_timesteps)
        self.assertEqual(scheduler.calls[0]["mu"], 0.42)
        self.assertEqual(result, scheduler.timesteps)

    @patch("simpletuner.helpers.training.validation.StateTracker")
    def test_evaluation_raises_when_mu_missing_for_dynamic_shift(self, mock_state_tracker):
        config = types.SimpleNamespace(flow_schedule_auto_shift=False, eval_timesteps=4)
        mock_state_tracker.get_args.return_value = config

        eval_helper = Evaluation(accelerator=types.SimpleNamespace(is_main_process=True))
        mock_model = MagicMock()
        mock_model.calculate_dynamic_shift_mu.return_value = None
        mock_state_tracker.get_model.return_value = mock_model

        class _Scheduler:
            def __init__(self):
                self.config = types.SimpleNamespace(use_dynamic_shifting=True)

            def set_timesteps(self, num_inference_steps=None, mu=None, **kwargs):
                self.called = True

        scheduler = _Scheduler()

        with self.assertRaises(ValueError):
            eval_helper.get_timestep_schedule(scheduler, latents=torch.zeros(1, 4, 4, 4))

    def _dynamic_shift_model_classes(self):
        dynamic_shift_model_classes = {}
        for family, registry_entry in ModelRegistry.model_families().items():
            if hasattr(registry_entry, "get_real_class"):
                model_cls = registry_entry.get_real_class()
            else:
                model_cls = registry_entry
            if getattr(model_cls, "USES_DYNAMIC_SHIFT", False):
                dynamic_shift_model_classes[family] = model_cls
        return dynamic_shift_model_classes

    def _model_class_declares_patch_size(self, model_cls):
        component_cls = getattr(model_cls, "MODEL_CLASS", None)
        self.assertIsNotNone(component_cls, f"{model_cls.__name__} must define MODEL_CLASS")
        init_method = getattr(component_cls, "__init__", None)
        try:
            parameters = inspect.signature(init_method).parameters
        except (TypeError, ValueError):
            return False
        return "patch_size" in parameters

    def test_dynamic_shift_models_expose_patch_geometry_or_sequence_length(self):
        model_classes = self._dynamic_shift_model_classes()
        self.assertGreater(len(model_classes), 0)

        for family, model_cls in model_classes.items():
            with self.subTest(family=family):
                has_model_sequence_length = "_latent_sequence_length" in model_cls.__dict__
                has_component_patch_size = self._model_class_declares_patch_size(model_cls)
                self.assertTrue(
                    has_model_sequence_length or has_component_patch_size,
                    f"{model_cls.__name__} uses dynamic shift but does not expose patch geometry",
                )

                model = model_cls.__new__(model_cls)
                model.config = types.SimpleNamespace(controlnet=False)
                model.accelerator = None
                model.model = types.SimpleNamespace(config=types.SimpleNamespace())
                if not has_model_sequence_length:
                    model.model.config.patch_size = 2
                    model.model.config.patch_size_t = 1

                latent_channels = getattr(model_cls, "LATENT_CHANNEL_COUNT", 4)
                latents = torch.zeros(1, latent_channels, 8, 8)
                mu = model.calculate_dynamic_shift_mu(self.scheduler, latents)

                self.assertTrue(math.isfinite(mu), f"{model_cls.__name__} produced non-finite dynamic shift mu")

    def test_dynamic_scheduler_configs_declare_dynamic_shift_flag(self):
        for family, registry_entry in ModelRegistry.model_families().items():
            if hasattr(registry_entry, "get_real_class"):
                model_cls = registry_entry.get_real_class()
            else:
                model_cls = registry_entry
            scheduler_config = getattr(inspect.getmodule(model_cls), "SCHEDULER_CONFIG", None)
            if not isinstance(scheduler_config, dict) or not scheduler_config.get("use_dynamic_shifting"):
                continue

            with self.subTest(family=family):
                self.assertTrue(
                    getattr(model_cls, "USES_DYNAMIC_SHIFT", False),
                    f"{model_cls.__name__} scheduler config enables dynamic shift without USES_DYNAMIC_SHIFT",
                )


if __name__ == "__main__":
    unittest.main()
