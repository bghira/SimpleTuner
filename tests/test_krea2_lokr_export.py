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
                    dora = torch.rand(sum(sizes), 1).to(dtype)
                    unrelated = torch.randn(2, 3)
                    weights = {
                        f"{prefix}.lokr_w1": w1,
                        f"{prefix}.lokr_w2": w2,
                        f"{prefix}.alpha": torch.tensor(10000.0),
                        f"{prefix}.dora_scale": dora,
                        "lycoris_transformer_blocks_0_ff_up.lokr_w1": unrelated,
                    }
                    converted = split_fused_lokr(weights, self.config, {prefix: base})
                    self.assertIs(converted["lycoris_transformer_blocks_0_ff_up.lokr_w1"], unrelated)
                    self.assertFalse(any("to_qkv" in key for key in converted))
                    original_merged = (base + delta).to(dora.dtype)
                    original_merged = original_merged * (
                        dora / (original_merged.norm(dim=1, keepdim=True) + torch.finfo(dora.dtype).eps)
                    )
                    offset = 0
                    for projection, size in zip("qkv", sizes):
                        target = prefix.removesuffix("qkv") + projection
                        actual_delta = torch.kron(converted[f"{target}.lokr_w1"], converted[f"{target}.lokr_w2"])
                        merged = base[offset : offset + size] + actual_delta
                        self.assertNotIn(f"{target}.dora_scale", converted)
                        torch.testing.assert_close(
                            merged, original_merged[offset : offset + size].float(), rtol=1e-5, atol=1e-6
                        )
                        offset += size
                    without_dora = {key: value for key, value in weights.items() if not key.endswith(".dora_scale")}
                    converted = split_fused_lokr(without_dora, self.config)
                    actual_delta = torch.cat(
                        [converted[f"{prefix.removesuffix('qkv') + projection}.lokr_w2"] for projection in "qkv"]
                    )
                    torch.testing.assert_close(actual_delta, delta, rtol=0, atol=0)

    def test_unused_separate_projections_do_not_change_fused_export(self):
        for prefix, rows in (
            ("lycoris_transformer_blocks_0_attn_to_qkv", 48),
            ("lycoris_text_fusion_layerwise_blocks_0_attn_to_qkv", 96),
            ("lycoris_text_fusion_refiner_blocks_0_attn_to_qkv", 96),
        ):
            for with_dora in (False, True):
                for zero_factor in ("lokr_w1", "lokr_w2"):
                    with self.subTest(prefix=prefix, with_dora=with_dora, zero_factor=zero_factor):
                        fused = {f"{prefix}.lokr_w1": torch.randn(16, 2), f"{prefix}.lokr_w2": torch.randn(rows // 16, 16)}
                        if with_dora:
                            fused[f"{prefix}.dora_scale"] = torch.rand(rows, 1)
                        base = {prefix: torch.randn(rows, 32)}
                        weights = dict(fused)
                        for projection, size in zip("qkv", (32, (rows - 32) // 2, (rows - 32) // 2)):
                            target = prefix.removesuffix("qkv") + projection
                            weights.update(
                                {
                                    f"{target}.lokr_w1": torch.randn(2, 2),
                                    f"{target}.lokr_w2": torch.randn(size // 2, 16),
                                    f"{target}.alpha": torch.tensor(10000.0),
                                }
                            )
                            weights[f"{target}.{zero_factor}"].zero_()
                            if with_dora:
                                weights[f"{target}.dora_scale"] = torch.rand(size, 1)
                        originals = {key: value.clone() for key, value in weights.items()}
                        actual = split_fused_lokr(weights, self.config, base)
                        expected = split_fused_lokr(fused, self.config, base)
                        self.assertEqual(actual.keys(), expected.keys())
                        for key in expected:
                            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
                        self.assertEqual(weights.keys(), originals.keys())
                        for key in weights:
                            torch.testing.assert_close(weights[key], originals[key], rtol=0, atol=0)

    def test_rejects_nonzero_and_unsupported_separate_projections(self):
        prefix = "lycoris_text_fusion_layerwise_blocks_0_attn_to_qkv"
        fused = {f"{prefix}.lokr_w1": torch.ones(16, 2), f"{prefix}.lokr_w2": torch.ones(6, 16)}
        for projection in "qkv":
            target = prefix.removesuffix("qkv") + projection
            separate = {f"{target}.lokr_w1": torch.ones(2, 2), f"{target}.lokr_w2": torch.ones(16, 16)}
            with self.subTest(projection=projection), self.assertRaisesRegex(ValueError, "nonzero separate projection"):
                split_fused_lokr(fused | separate, self.config)
            separate[f"{target}.lokr_w2"].zero_()
            separate[f"{target}.lokr_w1_a"] = torch.ones(2, 2)
            with self.subTest(projection=projection), self.assertRaisesRegex(ValueError, "full-matrix LoKR"):
                split_fused_lokr(fused | separate, self.config)

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
                split_fused_lokr(valid | changes, self.config, {prefix: torch.ones(48, 32)})

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

    def test_cli_materializes_fused_and_unfused_dora_against_original_base(self):
        import sys

        prefix = "lycoris_transformer_blocks_0_attn_to_qkv"
        other = "lycoris_transformer_blocks_0_ff_up"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target, config, base_path = (
                root / name for name in ("native.safetensors", "comfy.safetensors", "config.json", "raw.safetensors")
            )
            base = torch.randn(48, 32)
            base_other = torch.randn(32, 32)
            weights = {
                f"{prefix}.lokr_w1": torch.randn(16, 2),
                f"{prefix}.lokr_w2": torch.randn(3, 16),
                f"{prefix}.dora_scale": torch.rand(48, 1),
                f"{other}.lokr_w1": torch.randn(2, 2),
                f"{other}.lokr_w2": torch.randn(16, 16),
                f"{other}.dora_scale": torch.rand(32, 1),
            }
            for projection, rows in zip("qkv", (32, 8, 8)):
                target_prefix = prefix.removesuffix("qkv") + projection
                weights.update(
                    {
                        f"{target_prefix}.lokr_w1": torch.randn(2, 2),
                        f"{target_prefix}.lokr_w2": torch.zeros(rows // 2, 16),
                        f"{target_prefix}.dora_scale": torch.rand(rows, 1),
                        f"{target_prefix}.alpha": torch.tensor(10000.0),
                    }
                )
            save_file(weights, source)
            save_file(
                {
                    "blocks.0.attn.wq.weight": base[:32].clone(),
                    "blocks.0.attn.wk.weight": base[32:40].clone(),
                    "blocks.0.attn.wv.weight": base[40:].clone(),
                    "blocks.0.mlp.up.weight": base_other,
                },
                base_path,
            )
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
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
            subprocess.run(command + ["--base-model", str(base_path)], capture_output=True, check=True)
            converted = load_file(target)
            self.assertFalse(any(key.endswith(".dora_scale") for key in converted))
            for name, original in ((prefix, base), (other, base_other)):
                delta = torch.kron(weights[f"{name}.lokr_w1"], weights[f"{name}.lokr_w2"])
                merged = original + delta
                expected = merged * (
                    weights[f"{name}.dora_scale"] / (merged.norm(dim=1, keepdim=True) + torch.finfo(merged.dtype).eps)
                )
                if name == prefix:
                    exported = torch.cat(
                        [converted[f"{prefix.removesuffix('qkv') + projection}.lokr_w2"] for projection in "qkv"]
                    )
                else:
                    exported = converted[f"{name}.lokr_w2"]
                torch.testing.assert_close(original + exported, expected, rtol=1e-5, atol=1e-6)
