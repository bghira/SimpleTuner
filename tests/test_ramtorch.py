import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
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


if __name__ == "__main__":
    unittest.main()
