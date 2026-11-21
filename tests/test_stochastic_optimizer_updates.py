import unittest
from unittest.mock import patch

import torch

from simpletuner.helpers.training.optimizers.adamw_bfloat16 import AdamWBF16
from simpletuner.helpers.training.optimizers.adamw_bfloat16.stochastic import add_stochastic_, copy_stochastic_
from simpletuner.helpers.training.optimizers.muon import MuonClip


class StochasticOptimizerUpdateTests(unittest.TestCase):
    def devices(self):
        yield "cpu"
        if torch.cuda.is_available():
            yield "cuda"
        if torch.backends.mps.is_available():
            yield "mps"

    def test_rounding_probability_matches_fractional_bfloat16_spacing(self):
        samples = 1 << 16
        for device in self.devices():
            random_bits = torch.arange(samples, dtype=torch.int32, device=device)
            for sign in (1.0, -1.0):
                for fraction in (0.0, 1 / 128, 0.25, 0.5, 0.75):
                    with self.subTest(device=device, sign=sign, fraction=fraction):
                        source = torch.full((samples,), sign * (1 + fraction / 128), device=device)
                        target = torch.empty_like(source, dtype=torch.bfloat16)
                        with patch(
                            "simpletuner.helpers.training.optimizers.adamw_bfloat16.stochastic.torch.randint_like",
                            return_value=random_bits.clone(),
                        ) as randint:
                            copy_stochastic_(target, source)
                        randint.assert_called_once_with(source, dtype=torch.int32, low=0, high=samples)
                        farther = sign * (1 + 1 / 128)
                        self.assertTrue(torch.all((target == sign) | (target == farther)).item())
                        self.assertEqual((target == farther).sum().item(), int(samples * fraction))

    def test_rounding_preserves_nonfinite_values_and_seeded_finite_results(self):
        source_bits = torch.tensor(
            [0x7FFFFFFF, -1, 0x7F800001, -0x007FFFFF, 0x7F800000, -0x00800000],
            dtype=torch.int32,
        )
        nonfinite_bits = source_bits.repeat_interleave(3)
        nonfinite_draws = torch.tensor([0, 1, 0xFFFF], dtype=torch.int32).repeat(source_bits.numel())
        finite = torch.linspace(-2.0, 2.0, 1025)
        generator = torch.Generator().manual_seed(42)
        finite_draws = torch.randint(0, 1 << 16, finite.shape, dtype=torch.int32, generator=generator)
        expected = (finite.view(torch.int32) + finite_draws).bitwise_and(-65536).view(torch.float32).bfloat16()

        for device in self.devices():
            with self.subTest(device=device):
                source = torch.cat((nonfinite_bits, finite.view(torch.int32))).to(device).view(torch.float32)
                draws = torch.cat((nonfinite_draws, finite_draws)).to(device)
                target = torch.empty_like(source, dtype=torch.bfloat16)
                with patch.object(torch, "randint_like", return_value=draws):
                    copy_stochastic_(target, source)

                actual_nonfinite, actual_finite = target.cpu().split((nonfinite_bits.numel(), finite.numel()))
                self.assertTrue(actual_nonfinite[:12].isnan().all().item())
                torch.testing.assert_close(
                    actual_nonfinite[12:],
                    torch.tensor([float("inf")] * 3 + [float("-inf")] * 3, dtype=torch.bfloat16),
                )
                torch.testing.assert_close(actual_finite.view(torch.int16), expected.view(torch.int16), rtol=0, atol=0)

    def test_scaled_add_matches_tensor_add_and_preserves_source(self):
        for device in self.devices():
            for dtype in (torch.bfloat16, torch.float32):
                for alpha in (0.0, 0.5, 1.0, -0.25):
                    with self.subTest(device=device, source_dtype=dtype, alpha=alpha):
                        target = torch.tensor([4.0, -4.0, 8.0], device=device, dtype=torch.bfloat16)
                        source = torch.tensor([8.0, 16.0, -16.0], device=device, dtype=dtype)
                        original = source.clone()
                        expected = target.float().add(source.float(), alpha=alpha).bfloat16()
                        add_stochastic_(target, source, alpha=alpha)
                        torch.testing.assert_close(target, expected, rtol=0, atol=0)
                        torch.testing.assert_close(source, original, rtol=0, atol=0)

    def test_scaled_add_supports_aliased_weight_decay(self):
        for device in self.devices():
            with self.subTest(device=device):
                target = torch.tensor([4.0, -8.0, 16.0], device=device, dtype=torch.bfloat16)
                expected = target.float().mul(0.875).bfloat16()
                add_stochastic_(target, target, alpha=-0.125)
                torch.testing.assert_close(target, expected, rtol=0, atol=0)

    def test_adamw_first_moment_retains_the_requested_momentum(self):
        for device in self.devices():
            with self.subTest(device=device):
                parameter = torch.nn.Parameter(torch.full((3,), 16.0, device=device, dtype=torch.bfloat16))
                optimizer = AdamWBF16([parameter], lr=0.125, betas=(0.5, 0.5))
                for gradient, expected in ((8.0, 4.0), (4.0, 4.0), (-4.0, 0.0)):
                    parameter.grad = torch.full_like(parameter, gradient)
                    optimizer.step()
                    torch.testing.assert_close(
                        optimizer.state[parameter]["exp_avg"],
                        torch.full_like(parameter, expected),
                        rtol=0,
                        atol=0,
                    )

    def test_muon_vector_step_applies_learning_rate_to_update(self):
        for device in self.devices():
            with self.subTest(device=device):
                parameter = torch.nn.Parameter(torch.full((3,), 8.0, device=device, dtype=torch.bfloat16))
                parameter.grad = torch.full_like(parameter, 2.0)
                optimizer = MuonClip([parameter], lr=0.25, momentum=0.0, weight_decay=0.0)
                optimizer.step()
                torch.testing.assert_close(parameter, torch.full_like(parameter, 7.5), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
