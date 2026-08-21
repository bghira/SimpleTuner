import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from simpletuner.helpers.models.common import ImageModelFoundation, PredictionTypes
from simpletuner.helpers.training.custom_schedule import apply_flow_schedule_shift


class _TestImageModel(ImageModelFoundation):
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING

    def _encode_prompts(self, prompts, is_negative_prompt=False):
        raise NotImplementedError

    def convert_negative_text_embed_for_pipeline(self, text_embedding):
        raise NotImplementedError

    def convert_text_embed_for_pipeline(self, text_embedding):
        raise NotImplementedError

    def model_predict(self, prepared_batch):
        raise NotImplementedError


class MixFlowTest(unittest.TestCase):
    def _model(self, *, gamma=0.8, shift=1.0):
        model = _TestImageModel.__new__(_TestImageModel)
        model.config = SimpleNamespace(
            mixflow_enabled=True,
            mixflow_gamma=gamma,
            flow_schedule_shift=shift,
            flow_schedule_auto_shift=False,
            flow_custom_timesteps=None,
            flux_fast_schedule=False,
            flow_use_beta_schedule=False,
            flow_use_uniform_schedule=False,
        )
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.noise_schedule = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        return model

    def test_mixflow_samples_beta_one_two_noise_sigmas(self):
        model = self._model()
        batch = {
            "latents": torch.zeros(3, 1, 1, 1),
            "noise": torch.zeros(3, 1, 1, 1),
        }

        with patch("torch.rand", return_value=torch.tensor([0.0, 0.25, 1.0])):
            sigmas, timesteps = model.sample_flow_sigmas(batch, state={})

        expected = torch.tensor([1.0, 0.5, 0.0])
        torch.testing.assert_close(sigmas, expected)
        torch.testing.assert_close(timesteps, expected * 1000.0)

    def test_mixflow_slows_interpolation_without_changing_model_timestep(self):
        model = self._model(gamma=0.8)
        batch = {
            "latents": torch.zeros(2, 1, 1, 1),
            "input_noise": torch.ones(2, 1, 1, 1),
            "sigmas": torch.tensor([0.25, 0.75]),
            "timesteps": torch.tensor([250.0, 750.0]),
        }

        with patch("torch.rand_like", return_value=torch.tensor([0.5, 1.0])):
            model._prepare_flow_noisy_latents(batch)

        torch.testing.assert_close(batch["timesteps"], torch.tensor([250.0, 750.0]))
        torch.testing.assert_close(batch["sigmas"].flatten(), torch.tensor([0.25, 0.75]))
        torch.testing.assert_close(batch["mixflow_interpolation_sigmas"], torch.tensor([0.55, 0.95]))
        torch.testing.assert_close(batch["noisy_latents"].flatten(), torch.tensor([0.55, 0.95]))

    def test_mixflow_gamma_zero_matches_standard_flow_input(self):
        model = self._model(gamma=0.0)
        batch = {
            "latents": torch.tensor([[[[2.0]]]]),
            "input_noise": torch.tensor([[[[6.0]]]]),
            "sigmas": torch.tensor([0.25]),
            "timesteps": torch.tensor([250.0]),
        }

        model._prepare_flow_noisy_latents(batch)

        torch.testing.assert_close(batch["noisy_latents"], torch.tensor([[[[3.0]]]]))

    def test_mixflow_applies_native_flow_schedule_shift_before_slowing(self):
        model = self._model(gamma=0.8, shift=3.0)
        batch = {
            "latents": torch.zeros(1, 1, 1, 1),
            "noise": torch.zeros(1, 1, 1, 1),
        }

        with patch("torch.rand", return_value=torch.tensor([0.25])):
            sigmas, _ = model.sample_flow_sigmas(batch, state={})

        expected = apply_flow_schedule_shift(model.config, model.noise_schedule, torch.tensor([0.5]), batch["noise"])
        torch.testing.assert_close(sigmas, expected)

    def test_mixflow_gamma_must_be_between_zero_and_one(self):
        model = self._model(gamma=1.1)
        batch = {
            "latents": torch.zeros(1, 1, 1, 1),
            "input_noise": torch.ones(1, 1, 1, 1),
            "sigmas": torch.tensor([0.5]),
            "timesteps": torch.tensor([500.0]),
        }

        with self.assertRaisesRegex(ValueError, "mixflow_gamma"):
            model._prepare_flow_noisy_latents(batch)

    def test_mixflow_rejects_competing_timestep_schedule(self):
        model = self._model()
        model.config.flow_use_uniform_schedule = True

        with self.assertRaisesRegex(ValueError, "flow_use_uniform_schedule"):
            model.validate_mixflow_config()

    def test_mixflow_rejects_canonical_self_flow_mode(self):
        model = self._model()
        model.config.crepa_feature_source = "self_flow"

        with self.assertRaisesRegex(ValueError, "crepa_self_flow"):
            model.validate_mixflow_config()

    def test_mixflow_rejects_non_flow_model(self):
        model = self._model()
        model.PREDICTION_TYPE = PredictionTypes.EPSILON

        with self.assertRaisesRegex(ValueError, "flow-matching"):
            model.validate_mixflow_config()

    def test_omnigen_keeps_dataward_model_time_distinct_from_noiseward_sigma(self):
        from simpletuner.helpers.models.omnigen.model import OmniGen

        model = OmniGen.__new__(OmniGen)
        model.config = self._model().config
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.noise_schedule = SimpleNamespace(config=SimpleNamespace(num_train_timesteps=1000))
        batch = {
            "latents": torch.zeros(1, 1, 1, 1),
            "noise": torch.zeros(1, 1, 1, 1),
        }

        with patch("torch.rand", return_value=torch.tensor([0.0625])):
            sigmas, timesteps = model.sample_flow_sigmas(batch, state={})

        torch.testing.assert_close(sigmas, torch.tensor([0.75]))
        torch.testing.assert_close(timesteps, torch.tensor([0.25]))

    def test_chroma_applies_native_sigma_transform_before_slowing(self):
        from simpletuner.helpers.models.chroma.model import Chroma

        model = Chroma.__new__(Chroma)
        model.config = self._model().config
        batch = {
            "latents": torch.zeros(1, 1, 1, 1),
            "input_noise": torch.ones(1, 1, 1, 1),
            "sigmas": torch.tensor([[[[0.25]]]]),
            "timesteps": torch.tensor([250.0]),
            "mixflow_slowdown_factors": torch.tensor([0.5]),
        }

        model._apply_chroma_flow_schedule(batch)

        torch.testing.assert_close(batch["sigmas"].flatten(), torch.tensor([0.0625]))
        torch.testing.assert_close(batch["mixflow_interpolation_sigmas"], torch.tensor([0.4375]))
        torch.testing.assert_close(batch["noisy_latents"].flatten(), torch.tensor([0.4375]))
        torch.testing.assert_close(batch["timesteps"], torch.tensor([62.5]))

    def test_cosmos3_converts_noise_sigma_to_dataward_model_timestep(self):
        from simpletuner.helpers.models.cosmos3.model import Cosmos3Image

        model = Cosmos3Image.__new__(Cosmos3Image)

        timesteps = model.flow_matching_timesteps_from_sigmas(torch.tensor([0.2, 0.8]))

        torch.testing.assert_close(timesteps, torch.tensor([800.0, 200.0]))


class MixFlowFieldTest(unittest.TestCase):
    def test_mixflow_fields_are_registered(self):
        from simpletuner.simpletuner_sdk.server.services.field_registry.registry import FieldRegistry

        registry = FieldRegistry()
        enabled = registry.get_field("mixflow_enabled")
        gamma = registry.get_field("mixflow_gamma")

        self.assertIsNotNone(enabled)
        self.assertEqual(enabled.arg_name, "--mixflow_enabled")
        self.assertFalse(enabled.default_value)
        self.assertIsNotNone(gamma)
        self.assertEqual(gamma.arg_name, "--mixflow_gamma")
        self.assertEqual(gamma.default_value, 0.8)


if __name__ == "__main__":
    unittest.main()
