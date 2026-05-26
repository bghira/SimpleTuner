import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import load_file, safe_open, save_file

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from extract_adapter_common import TensorSource  # noqa: E402


class ExtractAdapterScriptsTests(unittest.TestCase):
    def _write_pair(self, tmpdir: Path) -> tuple[Path, Path, torch.Tensor]:
        base = {
            "layer.weight": torch.zeros(2, 2, dtype=torch.float32),
            "ignored.bias": torch.ones(2, dtype=torch.float32),
        }
        delta = torch.tensor([[3.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        target = {
            "layer.weight": delta.clone(),
            "ignored.bias": torch.ones(2, dtype=torch.float32),
        }
        base_path = tmpdir / "base.safetensors"
        target_path = tmpdir / "target.safetensors"
        save_file(base, base_path)
        save_file(target, target_path)
        return base_path, target_path, delta

    def test_peft_lora_extracts_rank_approximation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            base_path, target_path, delta = self._write_pair(tmpdir)
            output_path = tmpdir / "adapter.safetensors"

            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "extract_peft_lora.py"),
                    str(base_path),
                    str(target_path),
                    str(output_path),
                    "--rank",
                    "2",
                    "--dtype",
                    "float32",
                    "--target-modules",
                    "all-linear",
                ],
                cwd=REPO_ROOT,
                check=True,
            )

            state = load_file(output_path)
            self.assertEqual(
                set(state),
                {
                    "transformer.layer.lora_A.weight",
                    "transformer.layer.lora_B.weight",
                    "transformer.layer.alpha",
                },
            )
            reconstructed = state["transformer.layer.lora_B.weight"] @ state["transformer.layer.lora_A.weight"]
            self.assertTrue(torch.allclose(reconstructed, delta, atol=1e-5))

    def test_lycoris_extracts_locon_state_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            base_path, target_path, delta = self._write_pair(tmpdir)
            output_path = tmpdir / "adapter.safetensors"

            subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "extract_lycoris_adapter.py"),
                    str(base_path),
                    str(target_path),
                    str(output_path),
                    "--rank",
                    "2",
                    "--dtype",
                    "float32",
                    "--target-modules",
                    "all-linear",
                ],
                cwd=REPO_ROOT,
                check=True,
            )

            state = load_file(output_path)
            self.assertEqual(
                set(state),
                {
                    "lycoris_layer.lora_down.weight",
                    "lycoris_layer.lora_up.weight",
                    "lycoris_layer.alpha",
                },
            )
            reconstructed = state["lycoris_layer.lora_up.weight"] @ state["lycoris_layer.lora_down.weight"]
            self.assertTrue(torch.allclose(reconstructed, delta, atol=1e-5))

            with safe_open(output_path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
            self.assertIn("lycoris_config", metadata)

    def test_tensor_source_reuses_open_handles(self):
        class FakeHandle:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def get_tensor(self, key):
                return torch.tensor([1.0 if key == "a" else 2.0])

        path = Path("/tmp/test.safetensors")
        calls = []

        def fake_safe_open(file_path, *, framework, device):
            calls.append((file_path, framework, device))
            return FakeHandle()

        with patch("extract_adapter_common.safe_open", fake_safe_open):
            source = TensorSource("test", {"a": path, "b": path})
            with source:
                self.assertEqual(source.get_tensor("a").item(), 1.0)
                self.assertEqual(source.get_tensor("b").item(), 2.0)

        self.assertEqual(calls, [(path, "pt", "cpu")])


if __name__ == "__main__":
    unittest.main()
