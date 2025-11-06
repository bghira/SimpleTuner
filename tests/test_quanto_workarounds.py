import tempfile
import unittest

import torch

try:
    from diffusers.hooks import apply_group_offloading

    _DIFFUSERS_AVAILABLE = True
except ImportError:  # pragma: no cover - test skips when diffusers is missing
    apply_group_offloading = None  # type: ignore[assignment]
    _DIFFUSERS_AVAILABLE = False

try:
    from optimum.quanto import freeze, quantize, qint4, qint8

    _QUANTO_AVAILABLE = True
except ImportError:  # pragma: no cover - test skips when optimum-quanto is missing
    freeze = quantize = qint4 = qint8 = None  # type: ignore[assignment]
    _QUANTO_AVAILABLE = False


class _LinearWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@unittest.skipUnless(_QUANTO_AVAILABLE, "Optimum Quanto not installed")
class QuantoWorkaroundsTests(unittest.TestCase):
    def setUp(self):
        # importing applies the monkey patches we want to validate
        from simpletuner.helpers.training.quantisation import quanto_workarounds  # noqa: F401

    def _quantized_linear(self, quant_scheme):
        model = _LinearWrapper()
        quantize(model, weights=quant_scheme)
        freeze(model)
        return model

    def _assert_storage_matches_backing_tensor(self, tensor):
        self.assertNotEqual(tensor.data_ptr(), 0)
        backing = tensor._data
        self.assertEqual(tensor.data_ptr(), backing.data_ptr())
        self.assertEqual(tensor.untyped_storage().data_ptr(), backing.untyped_storage().data_ptr())
        self.assertEqual(tensor.storage().data_ptr(), backing.storage().data_ptr())

    def test_qbytes_tensor_exposes_backing_storage(self):
        model = self._quantized_linear(qint8)
        self._assert_storage_matches_backing_tensor(model.linear.weight.data)

    def test_qbits_tensor_exposes_backing_storage(self):
        model = self._quantized_linear(qint4)
        self._assert_storage_matches_backing_tensor(model.linear.weight.data)

    @unittest.skipUnless(_DIFFUSERS_AVAILABLE, "Diffusers group offloading unavailable")
    def test_quantized_module_can_be_group_offloaded(self):
        model = self._quantized_linear(qint8)
        with tempfile.TemporaryDirectory() as tmp_dir:
            apply_group_offloading(
                module=model,
                onload_device=torch.device("cpu"),
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=1,
                offload_to_disk_path=tmp_dir,
            )


if __name__ == "__main__":
    unittest.main()
