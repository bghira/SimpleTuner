import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from simpletuner.helpers.models.common import ModelFoundation
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


if __name__ == "__main__":
    unittest.main()
