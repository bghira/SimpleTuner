import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch.nn as nn

from simpletuner.helpers.configuration import cmd_args
from simpletuner.helpers.utils import ramtorch as ramtorch_utils


class _StubLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, skip_init=False):
        super().__init__(in_features, out_features, bias=bias, device="cpu", dtype=dtype)
        self._device_arg = device
        self._skip_init = skip_init


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 2)
        self.block = nn.Sequential(nn.Linear(2, 2), nn.ReLU())


class RamTorchUtilsTests(unittest.TestCase):
    def _build_stub_imports(self, replace_all_fn):
        return {
            "Linear": _StubLinear,
            "replace_all": replace_all_fn,
            "broadcast_zero_params": None,
            "create_zero_param_groups": None,
            "setup_grad_sharding_hooks": None,
        }

    def test_replace_linear_with_target_patterns(self):
        model = _SimpleModel()
        replace_all_calls = {"count": 0}

        def fake_replace_all(mod, device=None):
            replace_all_calls["count"] += 1

        stub_imports = self._build_stub_imports(fake_replace_all)
        with patch.object(ramtorch_utils, "ensure_available", return_value=stub_imports):
            replaced = ramtorch_utils.replace_linear_layers_with_ramtorch(
                model, device="cuda", target_patterns=["linear1", "block.*"]
            )

        self.assertEqual(replaced, 2)
        self.assertEqual(replace_all_calls["count"], 0)
        self.assertIsInstance(model.linear1, _StubLinear)
        self.assertIsInstance(model.block[0], _StubLinear)
        self.assertIsInstance(model.block[1], nn.ReLU)
        self.assertTrue(getattr(model.linear1.weight, "is_ramtorch", False))
        self.assertTrue(getattr(model.block[0].weight, "is_ramtorch", False))

    def test_replace_linear_uses_replace_all_when_no_patterns(self):
        model = _SimpleModel()
        replace_all_calls = {"count": 0}

        def fake_replace_all(mod, device=None):
            replace_all_calls["count"] += 1

        stub_imports = self._build_stub_imports(fake_replace_all)
        with patch.object(ramtorch_utils, "ensure_available", return_value=stub_imports):
            replaced = ramtorch_utils.replace_linear_layers_with_ramtorch(model, device="cuda", target_patterns=None)

        self.assertEqual(replaced, 2)
        self.assertEqual(replace_all_calls["count"], 1)
        # No replacements performed because replace_all is stubbed.
        self.assertIsInstance(model.linear1, nn.Linear)
        self.assertIsInstance(model.block[0], nn.Linear)

    def test_mark_ddp_ignore_params_marks_ramtorch_names(self):
        class _IgnoreModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_a = nn.Linear(2, 2)
                self.linear_b = nn.Linear(2, 2)
                setattr(self.linear_b.weight, "is_ramtorch", True)
                if self.linear_b.bias is not None:
                    setattr(self.linear_b.bias, "is_ramtorch", True)

        model = _IgnoreModel()
        ignored = ramtorch_utils.mark_ddp_ignore_params(model)
        self.assertEqual(ignored, 2)
        ignore_set = getattr(model, "_ddp_params_and_buffers_to_ignore", set())
        self.assertIn("linear_b.weight", ignore_set)
        self.assertIn("linear_b.bias", ignore_set)


class RamTorchConfigTests(unittest.TestCase):
    def _base_args(self, tmp_dir: str) -> list[str]:
        data_config = Path(tmp_dir) / "backend.json"
        data_config.write_text("{}")
        return [
            f"--output_dir={tmp_dir}",
            "--model_type=full",
            "--optimizer=adamw_bf16",
            f"--data_backend_config={data_config}",
            "--model_family=sdxl",
            "--pretrained_model_name_or_path=stub-model",
        ]

    def test_group_offload_mutually_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_args = self._base_args(tmp_dir)
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.backends.mps.is_available", return_value=False),
            ):
                with self.assertRaises(ValueError):
                    cmd_args.parse_cmdline_args(["--ramtorch", "--enable_group_offload", *base_args], exit_on_error=True)

    def test_set_grads_to_none_forced_for_ramtorch(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_args = self._base_args(tmp_dir)
            with (
                patch("torch.cuda.is_available", return_value=True),
                patch("torch.backends.mps.is_available", return_value=False),
            ):
                args = cmd_args.parse_cmdline_args(["--ramtorch", *base_args], exit_on_error=True)
        self.assertTrue(getattr(args, "set_grads_to_none", False))


class RamTorchTextEncoderEmbeddingTests(unittest.TestCase):
    """Test that embedding layers work correctly with ramtorch text encoders."""

    @unittest.skipUnless(
        __import__("torch").cuda.is_available(),
        "CUDA not available",
    )
    def test_embedding_device_mismatch_with_ramtorch(self):
        """
        When ramtorch is applied to a text encoder model, only nn.Linear layers
        are converted to RamTorch (kept in CPU RAM). The nn.Embedding layers
        remain as regular PyTorch modules on CPU. If the caller sends input_ids
        to GPU, the embedding lookup fails with a device mismatch error.

        This test reproduces the issue seen in Chroma/T5 encoding:
        RuntimeError: Expected all tensors to be on the same device,
        but got index is on cuda:0, different from other tensors on cpu
        """
        import torch
        import torch.nn as nn

        # Simple model mimicking T5 structure: embedding + linear layers
        class SimpleTextEncoder(nn.Module):
            def __init__(self, vocab_size=100, embed_dim=64, hidden_dim=128):
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
                self.linear1 = nn.Linear(embed_dim, hidden_dim)
                self.linear2 = nn.Linear(hidden_dim, embed_dim)

            def forward(self, input_ids):
                x = self.embed_tokens(input_ids)
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        model = SimpleTextEncoder()

        # Simulate what ramtorch does: convert Linear layers but not Embedding
        replace_all_calls = {"count": 0}

        def fake_replace_all(mod, device=None):
            replace_all_calls["count"] += 1

        stub_imports = {
            "Linear": _StubLinear,
            "replace_all": fake_replace_all,
            "broadcast_zero_params": None,
            "create_zero_param_groups": None,
            "setup_grad_sharding_hooks": None,
        }

        with patch.object(ramtorch_utils, "ensure_available", return_value=stub_imports):
            ramtorch_utils.replace_linear_layers_with_ramtorch(model, device="cuda", target_patterns=None)

        # Model stays on CPU (ramtorch doesn't move the model)
        # This is the state after ramtorch is applied to text encoders

        # Now simulate what the pipeline does: send input_ids to GPU
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        input_ids_cuda = input_ids.to("cuda")

        # This should fail because embed_tokens weights are on CPU
        # but input_ids are on CUDA
        with self.assertRaises(RuntimeError) as ctx:
            model(input_ids_cuda)

        self.assertIn("device", str(ctx.exception).lower())

    @unittest.skipUnless(
        __import__("torch").cuda.is_available(),
        "CUDA not available",
    )
    def test_embedding_moved_to_gpu_with_ramtorch(self):
        """
        After ramtorch is applied to text encoder, embedding layers should be
        moved to GPU so that input_ids (which come from GPU) work correctly.
        This tests the fix for the device mismatch issue.
        """
        import torch
        import torch.nn as nn

        class SimpleTextEncoder(nn.Module):
            def __init__(self, vocab_size=100, embed_dim=64, hidden_dim=128):
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
                self.linear1 = nn.Linear(embed_dim, hidden_dim)
                self.linear2 = nn.Linear(hidden_dim, embed_dim)

            def forward(self, input_ids):
                x = self.embed_tokens(input_ids)
                x = self.linear1(x)
                x = self.linear2(x)
                return x

        model = SimpleTextEncoder()

        # Verify embedding starts on CPU
        self.assertEqual(model.embed_tokens.weight.device.type, "cpu")

        # Simulate ramtorch being applied - mark Linear layers as ramtorch
        # (In real usage, replace_linear_layers_with_ramtorch does this)
        for name, child in model.named_modules():
            if isinstance(child, nn.Linear):
                setattr(child.weight, "is_ramtorch", True)
                if child.bias is not None:
                    setattr(child.bias, "is_ramtorch", True)

        # Apply the fix: move non-ramtorch modules to GPU
        moved = ramtorch_utils.move_embeddings_to_device(model, "cuda")

        # Verify one module was moved (only the embedding, not the ramtorch linears)
        self.assertEqual(moved, 1)

        # Verify embedding is now on GPU
        self.assertEqual(model.embed_tokens.weight.device.type, "cuda")

        # Verify Linear layers stayed on CPU (they're ramtorch)
        self.assertEqual(model.linear1.weight.device.type, "cpu")
        self.assertEqual(model.linear2.weight.device.type, "cpu")

        # Test that embedding lookup works with GPU input_ids
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        input_ids_cuda = input_ids.to("cuda")

        # Embedding lookup should work (no device mismatch)
        embed_output = model.embed_tokens(input_ids_cuda)
        self.assertEqual(embed_output.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
