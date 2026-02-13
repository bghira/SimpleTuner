import unittest
from types import SimpleNamespace

import torch

from simpletuner.helpers.training.peft_init import init_lokr_network_with_perturbed_normal

try:
    from torchao.prototype.quantized_training.int8 import Int8QuantizedTrainingLinearWeight
except Exception:  # pragma: no cover - optional dependency
    Int8QuantizedTrainingLinearWeight = None


@unittest.skipIf(Int8QuantizedTrainingLinearWeight is None, "torchao int8 quantized training is not available")
class LoKrInitTorchAOTests(unittest.TestCase):
    def test_init_lokr_with_torchao_int8_weight(self):
        org_weight = Int8QuantizedTrainingLinearWeight.from_float(torch.randn(8, 8))
        lokr_module = SimpleNamespace(
            org_weight=org_weight,
            lokr_w1=torch.zeros(8, 8),
            lokr_w2=torch.zeros(8, 8),
        )
        network = SimpleNamespace(loras=[lokr_module])

        init_lokr_network_with_perturbed_normal(network, scale=1e-3)

        self.assertTrue(torch.allclose(lokr_module.lokr_w1, torch.ones_like(lokr_module.lokr_w1)))
        self.assertGreater(lokr_module.lokr_w2.abs().sum().item(), 0.0)
