import math
import unittest
from types import SimpleNamespace

import torch

from simpletuner.helpers.models.qwen_image.model import QwenImage
from simpletuner.helpers.training.custom_schedule import apply_flow_schedule_shift


class QwenFlowShiftTests(unittest.TestCase):
    def scheduler(self, flavour):
        model = QwenImage.__new__(QwenImage)
        model.config = SimpleNamespace(model_flavour=flavour)
        return model.setup_training_noise_schedule()[1]

    def test_21_shift_counts_each_latent_pixel(self):
        args = SimpleNamespace(flow_schedule_shift=0, flow_schedule_auto_shift=True)
        scheduler = self.scheduler("v2.1")
        for height, width in ((32, 32), (64, 64), (48, 80)):
            with self.subTest(height=height, width=width):
                noise = torch.zeros(1, 64, height, width)
                sigmas = torch.tensor([0.0, 0.25, 0.5, 1.0])
                mu = 0.5 + (height * width - 256) * (0.9 - 0.5) / (8192 - 256)
                shift = math.exp(mu)
                expected = sigmas * shift / (1 + (shift - 1) * sigmas)
                torch.testing.assert_close(apply_flow_schedule_shift(args, scheduler, sigmas, noise), expected)

    def test_older_flavours_retain_patch_two_shift(self):
        for flavour in ("v1.0", "v2.0", "edit-v3"):
            with self.subTest(flavour=flavour):
                scheduler = self.scheduler(flavour)
                self.assertEqual(getattr(scheduler.config, "patch_size", 2), 2)

    def test_explicit_static_shift_still_wins(self):
        args = SimpleNamespace(flow_schedule_shift=3.0, flow_schedule_auto_shift=True)
        sigma = torch.tensor([0.5])
        for flavour in ("v2.1", "v2.0"):
            shifted = apply_flow_schedule_shift(args, self.scheduler(flavour), sigma, torch.zeros(1, 64, 64, 64))
            torch.testing.assert_close(shifted, torch.tensor([0.75]))
