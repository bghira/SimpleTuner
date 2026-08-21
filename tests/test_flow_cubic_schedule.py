import math
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

import tests.test_stubs  # noqa: F401
from simpletuner.helpers.configuration.cli_utils import mapping_to_cli_args
from simpletuner.helpers.configuration.cmd_args import parse_cmdline_args
from simpletuner.helpers.models.ace_step.model import ACEStep
from simpletuner.helpers.models.common import ImageModelFoundation
from simpletuner.helpers.models.cosmos3.model import Cosmos3Image
from simpletuner.helpers.models.ideogram.model import Ideogram4, get_schedule_for_resolution
from simpletuner.helpers.models.minimaxmusic.model import MiniMaxMusic
from simpletuner.helpers.models.omnigen.model import OmniGen
from simpletuner.helpers.models.sana.model import Sana
from simpletuner.helpers.models.sanavideo.model import SanaVideo
from simpletuner.helpers.training.timestep_distribution import CubicSplineDistribution, parse_cubic_spline_weights
from simpletuner.simpletuner_sdk.server.services.field_registry.registry import FieldRegistry
from simpletuner.simpletuner_sdk.server.services.field_registry.types import FieldType


def _base_args():
    return [
        "--model_family=pixart",
        "--output_dir=/tmp/output",
        "--model_type=lora",
        "--optimizer=adamw_bf16",
        "--data_backend_config=/tmp/config.json",
    ]


def _flow_model(weights, shift=None):
    model = SimpleNamespace(
        config=SimpleNamespace(
            flow_cubic_schedule_weights=weights,
            flow_custom_timesteps=None,
            flow_timesteps_mode="fixed-list",
            flow_schedule_shift=shift,
            flow_schedule_auto_shift=False,
        ),
        accelerator=SimpleNamespace(device=torch.device("cpu")),
        noise_schedule=SimpleNamespace(config=SimpleNamespace()),
    )
    model._normalize_flow_custom_timesteps = ImageModelFoundation._normalize_flow_custom_timesteps.__get__(model)
    model._flow_cubic_schedule_weights = ImageModelFoundation._flow_cubic_schedule_weights.__get__(model)
    model._uses_flow_cubic_schedule = ImageModelFoundation._uses_flow_cubic_schedule.__get__(model)
    model._sample_flow_cubic_values = ImageModelFoundation._sample_flow_cubic_values.__get__(model)
    model.sample_flow_sigmas = ImageModelFoundation.sample_flow_sigmas.__get__(model)
    return model


class CubicSplineDistributionTests(unittest.TestCase):
    def test_empty_and_single_weight_schedules_are_uniform(self):
        for weights in ([], [0.0], [4.0]):
            torch.manual_seed(17)
            expected = torch.rand(64)
            torch.manual_seed(17)
            actual = CubicSplineDistribution(weights).sample((64,))
            self.assertTrue(torch.equal(actual, expected))

    def test_two_weights_define_linear_density(self):
        distribution = CubicSplineDistribution([0.0, 1.0])
        density = distribution.log_prob(torch.tensor([0.25, 0.75])).exp()
        self.assertTrue(torch.allclose(density, torch.tensor([0.5, 1.5]), atol=1e-3))

        torch.manual_seed(11)
        samples = distribution.sample((100_000,))
        self.assertAlmostEqual(samples.mean().item(), 2.0 / 3.0, places=2)

    def test_cubic_density_is_bounded_and_finite(self):
        distribution = CubicSplineDistribution([0.0, 1.0, 0.1, 2.0, 0.0])
        samples = distribution.sample((10_000,))

        self.assertGreaterEqual(samples.min().item(), 0.0)
        self.assertLessEqual(samples.max().item(), 1.0)
        self.assertTrue(torch.isfinite(distribution.log_prob(samples)).all())
        self.assertEqual(distribution.log_prob(torch.tensor([-0.1, 1.1])).tolist(), [-math.inf, -math.inf])

    def test_invalid_weights_raise(self):
        invalid_values = ([1.0, -1.0], [0.0, 0.0], [1.0, math.inf], "1,,2", {"weight": 1})
        for value in invalid_values:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_cubic_spline_weights(value)


class FlowCubicScheduleIntegrationTests(unittest.TestCase):
    def test_common_flow_sampler_uses_distribution_and_shift(self):
        model = _flow_model([0.0, 1.0], shift=2.0)
        batch = {
            "latents": torch.zeros(32, 1, 2, 2),
            "noise": torch.zeros(32, 1, 2, 2),
        }

        torch.manual_seed(7)
        raw = CubicSplineDistribution([0.0, 1.0]).sample((32,))
        expected = (raw * 2.0) / (1.0 + raw)
        torch.manual_seed(7)
        sigmas, timesteps = model.sample_flow_sigmas(batch=batch, state={})

        self.assertTrue(torch.allclose(sigmas, expected))
        self.assertTrue(torch.allclose(timesteps, expected * 1000.0))

    def test_json_config_remains_one_cli_argument(self):
        cli_args = mapping_to_cli_args({"flow_cubic_schedule_weights": [0.0, 1.0, 0.25]})
        matching = [value for value in cli_args if value.startswith("--flow_cubic_schedule_weights")]

        self.assertEqual(matching, ["--flow_cubic_schedule_weights=[0.0, 1.0, 0.25]"])

    def test_command_line_parses_json_and_comma_separated_weights(self):
        json_args = parse_cmdline_args(
            input_args=_base_args() + ["--flow_cubic_schedule_weights=[0, 1, 0.5]"],
            exit_on_error=False,
        )
        comma_args = parse_cmdline_args(
            input_args=_base_args() + ["--flow_cubic_schedule_weights=0,1,0.5"],
            exit_on_error=False,
        )

        self.assertEqual(json_args.flow_cubic_schedule_weights, [0.0, 1.0, 0.5])
        self.assertEqual(comma_args.flow_cubic_schedule_weights, [0.0, 1.0, 0.5])

    def test_command_line_rejects_competing_schedule(self):
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            parse_cmdline_args(
                input_args=_base_args() + ["--flow_cubic_schedule_weights=[0, 1]", "--flow_use_uniform_schedule"],
                exit_on_error=False,
            )

    def test_webui_field_is_registered_as_json(self):
        field = FieldRegistry().get_field("flow_cubic_schedule_weights")

        self.assertIsNotNone(field)
        self.assertEqual(field.field_type, FieldType.TEXT_JSON)
        self.assertEqual(field.arg_name, "--flow_cubic_schedule_weights")


class ModelSpecificCubicScheduleTests(unittest.TestCase):
    def test_ace_step_uses_density_before_scheduler_lookup(self):
        model = ACEStep.__new__(ACEStep)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(flow_cubic_schedule_weights=[0.0, 1.0], logit_mean=0.0, logit_std=1.0)
        model.noise_schedule = SimpleNamespace(
            sigmas=torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]),
            timesteps=torch.tensor([1000.0, 750.0, 500.0, 250.0, 0.0]),
        )
        model._sample_flow_cubic_values = MagicMock(return_value=torch.tensor([0.0, 0.99]))

        sigmas, timesteps = model.sample_flow_sigmas({"latents": torch.zeros(2, 1, 2, 2)}, {})

        self.assertTrue(torch.equal(sigmas, torch.tensor([1.0, 0.25])))
        self.assertTrue(torch.equal(timesteps, torch.tensor([1000.0, 250.0])))

    def test_cosmos3_uses_density_as_native_timestep(self):
        model = Cosmos3Image.__new__(Cosmos3Image)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(
            flow_cubic_schedule_weights=[0.0, 1.0],
            weight_dtype=torch.float32,
            model_flavour="nano",
        )
        model._sample_flow_cubic_values = MagicMock(return_value=torch.tensor([0.2, 0.8]))
        model.prepare_batch_conditions = lambda batch, state: batch

        result = model.prepare_batch({"latent_batch": torch.ones(2, 1, 1, 1)}, {})

        self.assertTrue(torch.equal(result["sigmas"], torch.tensor([0.2, 0.8])))
        self.assertTrue(torch.equal(result["timesteps"], torch.tensor([200.0, 800.0])))

    def test_ideogram_uses_density_before_resolution_schedule(self):
        model = Ideogram4.__new__(Ideogram4)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(
            flow_cubic_schedule_weights=[0.0, 1.0],
            ideogram_schedule_mu=0.0,
            ideogram_schedule_std=1.5,
        )
        schedule_u = torch.tensor([0.25, 0.75])
        model._sample_flow_cubic_values = MagicMock(return_value=schedule_u)
        batch = {"latents": torch.zeros(2, 128, 64, 64)}

        sigmas, timesteps = model.sample_flow_sigmas(batch, {})

        model_t = get_schedule_for_resolution((1024, 1024), known_mean=0.0, std=1.5)(schedule_u)
        self.assertTrue(torch.allclose(sigmas, 1.0 - model_t))
        self.assertTrue(torch.allclose(timesteps, (1.0 - model_t) * 1000.0))

    def test_minimax_music_preserves_data_ward_timestep_convention(self):
        model = MiniMaxMusic.__new__(MiniMaxMusic)
        model.accelerator = SimpleNamespace(device=torch.device("cpu"))
        model.config = SimpleNamespace(
            flow_cubic_schedule_weights=[0.0, 1.0],
            weight_dtype=torch.float32,
            logit_mean=0.0,
            logit_std=1.0,
        )
        model._sample_flow_cubic_values = MagicMock(return_value=torch.tensor([0.2, 0.8]))

        sigmas, timesteps = model.sample_flow_sigmas({"latents": torch.zeros(2, 1, 2)}, {})

        self.assertTrue(torch.allclose(sigmas, torch.tensor([0.8, 0.2])))
        self.assertTrue(torch.equal(timesteps, torch.tensor([0.2, 0.8])))

    def test_omnigen_uses_density_as_unscaled_timestep(self):
        model = OmniGen.__new__(OmniGen)
        model.config = SimpleNamespace(flow_cubic_schedule_weights=[0.0, 1.0])
        model._sample_flow_cubic_values = MagicMock(return_value=torch.tensor([0.2, 0.8]))

        sigmas, timesteps = model.sample_flow_sigmas({"latents": torch.zeros(2, 1, 2, 2)}, {})

        self.assertTrue(torch.equal(sigmas, torch.tensor([0.2, 0.8])))
        self.assertTrue(torch.equal(timesteps, torch.tensor([0.2, 0.8])))

    def test_sana_families_use_density_before_scheduler_lookup(self):
        for model_class in (Sana, SanaVideo):
            with self.subTest(model_class=model_class.__name__):
                model = model_class.__new__(model_class)
                model.accelerator = SimpleNamespace(device=torch.device("cpu"))
                model.config = SimpleNamespace(flow_cubic_schedule_weights=[0.0, 1.0])
                model.noise_schedule = SimpleNamespace(
                    config=SimpleNamespace(num_train_timesteps=5),
                    sigmas=torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0]),
                    timesteps=torch.tensor([1000.0, 750.0, 500.0, 250.0, 0.0]),
                )
                model._sample_flow_cubic_values = MagicMock(return_value=torch.tensor([0.0, 0.99]))

                sigmas, timesteps = model.sample_flow_sigmas({"latents": torch.zeros(2, 1, 2, 2)}, {})

                self.assertTrue(torch.equal(sigmas, torch.tensor([1.0, 0.0])))
                self.assertTrue(torch.equal(timesteps, torch.tensor([1000.0, 0.0])))


if __name__ == "__main__":
    unittest.main()
