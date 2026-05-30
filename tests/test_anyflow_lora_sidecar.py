import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from simpletuner.helpers.training.adapter import load_lora_weights


class _ConditionEmbedder(torch.nn.Module):
    def __init__(self, *, with_delta: bool):
        super().__init__()
        if with_delta:
            self.delta_embedder = torch.nn.Sequential()
            self.delta_embedder.add_module("linear_1", torch.nn.Linear(2, 2))


class _Transformer(torch.nn.Module):
    def __init__(self, *, with_delta: bool):
        super().__init__()
        self.condition_embedder = _ConditionEmbedder(with_delta=with_delta)


class AnyFlowLoraSidecarTests(unittest.TestCase):
    def _write_sidecar(self, directory: Path) -> Path:
        path = directory / "adapter.safetensors"
        save_file(
            {
                "transformer.condition_embedder.delta_embedder.linear_1.weight": torch.full((2, 2), 3.0),
                "transformer.condition_embedder.delta_embedder.linear_1.bias": torch.full((2,), 4.0),
            },
            str(path),
        )
        return path

    def test_anyflow_sidecar_requires_enabled_delta_embedder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_sidecar(Path(temp_dir))
            model = _Transformer(with_delta=False)

            with self.assertRaisesRegex(ValueError, "enable_flowmap_time_conditioning"):
                load_lora_weights({"transformer": model}, str(path))

    def test_anyflow_sidecar_loads_delta_embedder_tensors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._write_sidecar(Path(temp_dir))
            model = _Transformer(with_delta=True)

            load_lora_weights({"transformer": model}, str(path))

            self.assertTrue(torch.equal(model.condition_embedder.delta_embedder.linear_1.weight, torch.full((2, 2), 3.0)))
            self.assertTrue(torch.equal(model.condition_embedder.delta_embedder.linear_1.bias, torch.full((2,), 4.0)))


if __name__ == "__main__":
    unittest.main()
