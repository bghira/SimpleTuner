import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from scripts.convert_krea2_lokr_to_comfyui import split_fused_lokr


class Krea2LoKRExportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.config = dict(
            attention_head_dim=4,
            num_attention_heads=8,
            num_key_value_heads=2,
            text_hidden_dim=32,
            text_num_attention_heads=8,
            text_num_key_value_heads=8,
        )

    def test_export_preserves_full_matrix_and_output_dora_across_factor_boundaries(self):
        for prefix, sizes in (
            ("lycoris_transformer_blocks_0_attn_to_qkv", (32, 8, 8)),
            ("lycoris_text_fusion_layerwise_blocks_0_attn_to_qkv", (32, 32, 32)),
            ("lycoris_text_fusion_refiner_blocks_1_attn_to_qkv", (32, 32, 32)),
        ):
            for dtype in (torch.float32, torch.bfloat16):
                with self.subTest(prefix=prefix, dtype=dtype):
                    w1 = torch.randn(16, 2).to(dtype)
                    w2 = torch.randn(sum(sizes) // 16, 16).to(dtype)
                    delta = torch.kron(w1.float(), w2.float())
                    base = torch.randn_like(delta)
                    dora = torch.rand(sum(sizes), 1)
                    unrelated = torch.randn(2, 3)
                    weights = {
                        f"{prefix}.lokr_w1": w1,
                        f"{prefix}.lokr_w2": w2,
                        f"{prefix}.alpha": torch.tensor(10000.0),
                        f"{prefix}.dora_scale": dora,
                        "lycoris_transformer_blocks_0_ff_up.lokr_w1": unrelated,
                    }
                    converted = split_fused_lokr(weights, self.config)
                    self.assertIs(converted["lycoris_transformer_blocks_0_ff_up.lokr_w1"], unrelated)
                    self.assertFalse(any("to_qkv" in key for key in converted))
                    original_merged = (base + delta) * dora / (base + delta).norm(dim=1, keepdim=True)
                    offset = 0
                    for projection, size in zip("qkv", sizes):
                        target = prefix.removesuffix("qkv") + projection
                        actual_delta = torch.kron(converted[f"{target}.lokr_w1"], converted[f"{target}.lokr_w2"])
                        torch.testing.assert_close(actual_delta, delta[offset : offset + size], rtol=0, atol=0)
                        merged = base[offset : offset + size] + actual_delta
                        merged = merged * converted[f"{target}.dora_scale"] / merged.norm(dim=1, keepdim=True)
                        torch.testing.assert_close(merged, original_merged[offset : offset + size], rtol=0, atol=0)
                        offset += size

    def test_rejects_config_mismatch_decomposed_factors_and_input_normalized_dora(self):
        prefix = "lycoris_transformer_blocks_0_attn_to_qkv"
        valid = {f"{prefix}.lokr_w1": torch.ones(16, 2), f"{prefix}.lokr_w2": torch.ones(3, 16)}
        for changes in (
            {f"{prefix}.lokr_w2": torch.ones(4, 16)},
            {f"{prefix}.lokr_w1_a": torch.ones(16, 2)},
            {f"{prefix}.dora_scale": torch.ones(1, 32)},
            {"lycoris_transformer_blocks_0_attn_to_q.lokr_w1": torch.ones(1, 1)},
        ):
            with self.subTest(keys=list(changes)), self.assertRaises(ValueError):
                split_fused_lokr(valid | changes, self.config)

    def test_cli_preserves_metadata_and_native_checkpoint(self):
        import sys

        prefix = "lycoris_transformer_blocks_0_attn_to_qkv"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target, config = root / "native.safetensors", root / "comfy.safetensors", root / "config.json"
            weights = {f"{prefix}.lokr_w1": torch.ones(16, 2), f"{prefix}.lokr_w2": torch.ones(3, 16)}
            save_file(weights, source, metadata={"format": "pt", "modelspec.title": "test"})
            original = source.read_bytes()
            config.write_text(json.dumps(self.config))
            command = [
                sys.executable,
                "scripts/convert_krea2_lokr_to_comfyui.py",
                "--input",
                str(source),
                "--output",
                str(target),
                "--transformer-config",
                str(config),
            ]
            subprocess.run(command, check=True, capture_output=True)
            self.assertEqual(len(load_file(target)), 6)
            with safe_open(target, framework="pt") as checkpoint:
                self.assertEqual(checkpoint.metadata(), {"format": "pt", "modelspec.title": "test"})
            command[command.index(str(target))] = str(source)
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.assertEqual(source.read_bytes(), original)
