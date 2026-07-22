import logging
import unittest
from unittest.mock import patch

import torch
from torch import nn

from simpletuner.helpers.musubi_block_swap import MusubiBlockSwapManager, _module_on_device


class MusubiBlockSwapTests(unittest.TestCase):
    def _accelerator_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        self.skipTest("No accelerator device available for block swap transfer tests")

    def test_quanto_qlinear_streams_without_apply_swap(self):
        try:
            from optimum.quanto import freeze, qint8, quantize
            from optimum.quanto.nn.qlinear import QLinear
        except ImportError as exc:
            self.skipTest(f"Quanto int8 quantization is unavailable: {exc}")

        device = self._accelerator_device()
        block = nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Linear(4, 4))
        quantize(block, weights=qint8)
        freeze(block)
        qlinear = block[0]
        self.assertIsInstance(qlinear, QLinear)

        manager = MusubiBlockSwapManager(
            block_indices=[0],
            offload_device=torch.device("cpu"),
            logger=logging.getLogger(__name__),
        )

        with patch.object(QLinear, "_apply", side_effect=RuntimeError("_apply(): Couldn't swap QLinear.weight")):
            manager.stream_in(block, device)
            self.assertTrue(_module_on_device(block, device))
            self.assertEqual(qlinear.weight.device.type, device.type)
            self.assertEqual(qlinear.weight._data.device.type, device.type)
            self.assertEqual(qlinear.weight._scale.device.type, device.type)
            output = block(torch.randn(2, 4, device=device))
            self.assertEqual(output.device.type, device.type)

            manager.stream_out(block)
            self.assertTrue(_module_on_device(block, torch.device("cpu")))
            self.assertEqual(qlinear.weight.device.type, "cpu")
            self.assertEqual(qlinear.weight._data.device.type, "cpu")
            self.assertEqual(qlinear.weight._scale.device.type, "cpu")

    def test_stream_out_keeps_trainable_params_on_accelerator(self):
        device = self._accelerator_device()
        block = nn.Sequential(nn.Linear(4, 4), nn.SiLU(), nn.Linear(4, 4))
        for param in block.parameters():
            param.requires_grad_(False)
        block.register_parameter("adapter_weight", nn.Parameter(torch.ones(4, device=device)))
        block.register_buffer("adapter_scalar", torch.ones((), device=device))
        block.to(device)

        manager = MusubiBlockSwapManager(
            block_indices=[0],
            offload_device=torch.device("cpu"),
            logger=logging.getLogger(__name__),
        )

        manager.stream_out(block)

        self.assertEqual(block.adapter_weight.device.type, device.type)
        self.assertEqual(block.adapter_scalar.device.type, device.type)
        self.assertEqual(block[0].weight.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
