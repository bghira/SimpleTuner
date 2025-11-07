import tempfile
import unittest

import torch
from diffusers.hooks import apply_group_offloading
from optimum.quanto import freeze, qint4, qint8, quantize

import simpletuner.helpers.training.quantisation.quanto_workarounds  # noqa: F401


class _LinearWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class QuantoWorkaroundsTests(unittest.TestCase):
    def _accelerator_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        self.skipTest("No accelerator device available for device transfer tests")

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

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_quantized_module_can_be_group_offloaded_with_cuda(self):
        model = self._quantized_linear(qint8)
        with tempfile.TemporaryDirectory() as tmp_dir:
            apply_group_offloading(
                module=model,
                onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="block_level",
                num_blocks_per_group=1,
                offload_to_disk_path=tmp_dir,
            )
            input_tensor = torch.randn(2, 4, device="cuda")
            output = model(input_tensor)
            self.assertEqual(output.device.type, "cuda")

    def test_param_data_move_keeps_backing_tensors_in_sync(self):
        device = self._accelerator_device()
        model = self._quantized_linear(qint8)
        weight = model.linear.weight
        weight.data = weight.data.to(device)
        self.assertEqual(weight.device.type, device.type)
        if device.index is not None:
            self.assertEqual(weight.device.index, device.index)
        self.assertEqual(weight._data.device.type, device.type)
        if device.index is not None:
            self.assertEqual(weight._data.device.index, device.index)
        self.assertEqual(weight._scale.device.type, device.type)
        if device.index is not None:
            self.assertEqual(weight._scale.device.index, device.index)


if __name__ == "__main__":
    unittest.main()
