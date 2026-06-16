import argparse
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from scripts.prompt2effect.data import Prompt2EffectTargetDataset, collate_prompt2effect_batch
from scripts.prompt2effect.lora_utils import discover_lora_modules, lora_delta, save_generated_lora
from scripts.prompt2effect.model import Prompt2EffectConfig, Prompt2EffectHyperNetwork, prompt2effect_loss
from scripts.prompt2effect.prepare import prepare_prompt2effect_targets
from scripts.prompt2effect.schema import TARGETS_FILENAME, load_schema


class Prompt2EffectScriptsTest(unittest.TestCase):
    def _write_base_model(self, root: Path):
        transformer = root / "transformer"
        transformer.mkdir(parents=True)
        save_file(
            {
                "blocks.0.attn.to_q.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10,
                "blocks.0.attn.to_v.weight": torch.arange(15, dtype=torch.float32).reshape(5, 3) / 10,
            },
            transformer / "diffusion_pytorch_model.safetensors",
        )

    def _write_lora(self, path: Path, scale: float):
        a_q = torch.tensor([[1.0, 0.0, 2.0], [0.5, -1.0, 0.25]], dtype=torch.float32) * scale
        b_q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-0.5, 0.25]], dtype=torch.float32)
        a_v = torch.tensor([[0.25, 1.0, 0.0], [1.5, -0.25, 0.5]], dtype=torch.float32) * scale
        b_v = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, -1.0], [0.5, 0.5], [-0.25, 1.0]],
            dtype=torch.float32,
        )
        save_file(
            {
                "transformer.blocks.0.attn.to_q.lora_A.weight": a_q,
                "transformer.blocks.0.attn.to_q.lora_B.weight": b_q,
                "transformer.blocks.0.attn.to_q.alpha": torch.tensor(2.0),
                "transformer.blocks.0.attn.to_v.lora_A.weight": a_v,
                "transformer.blocks.0.attn.to_v.lora_B.weight": b_v,
                "transformer.blocks.0.attn.to_v.alpha": torch.tensor(2.0),
            },
            path,
        )

    def test_prepare_canonicalizes_lora_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_dir = root / "base"
            self._write_base_model(base_dir)
            lora_0 = root / "effect0.safetensors"
            lora_1 = root / "effect1.safetensors"
            self._write_lora(lora_0, 1.0)
            self._write_lora(lora_1, 0.5)
            manifest = root / "manifest.jsonl"
            with manifest.open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": "effect0", "effect_prompt": "blue mood", "lora_path": str(lora_0)}) + "\n")
                handle.write(json.dumps({"id": "effect1", "effect_prompt": "red mood", "lora_path": str(lora_1)}) + "\n")

            prepared = root / "prepared"
            prepare_prompt2effect_targets(
                argparse.Namespace(
                    manifest=str(manifest),
                    output_dir=str(prepared),
                    model_family="wan",
                    base_model=str(base_dir),
                    model_flavour=None,
                    base_revision=None,
                    cache_dir=None,
                    component_subfolder="transformer",
                    target_modules="to_q,to_v",
                    rank=None,
                )
            )

            schema = load_schema(prepared)
            self.assertEqual(schema["model_family"], "wan")
            self.assertEqual(schema["rank"], 2)
            self.assertEqual(len(schema["samples"]), 2)
            self.assertEqual(len(schema["layers"]), 2)

            targets = load_file(prepared / TARGETS_FILENAME)
            lora_state = load_file(lora_0)
            first_layer = schema["layers"][0]
            module_name = first_layer["module_name"]
            module = next(
                module
                for module in discover_lora_modules(lora_state, component_prefix="transformer").values()
                if module.module_name == module_name
            )
            delta = lora_delta(lora_state, module)
            reconstructed = targets["samples.0.layers.0.B"] @ targets["samples.0.layers.0.A"]
            self.assertTrue(torch.allclose(reconstructed, delta, atol=1e-5, rtol=1e-5))

            dataset = Prompt2EffectTargetDataset(prepared)
            batch = collate_prompt2effect_batch([dataset[0], dataset[1]])
            self.assertEqual(batch["effect_prompts"], ["blue mood", "red mood"])
            self.assertEqual(batch["targets"][0]["A"].shape, (2, 2, 3))

    def test_model_forward_loss_and_generated_lora_format(self):
        config = Prompt2EffectConfig(
            rank=2,
            hidden_dim=8,
            text_hidden_dim=6,
            compressed_tokens=3,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            layer_count=2,
            module_types=["to_q", "to_v"],
            layer_shapes=[(4, 3), (5, 3)],
        )
        model = Prompt2EffectHyperNetwork(config)
        base_weights = [torch.randn(4, 3), torch.randn(5, 3)]
        text_hidden = torch.randn(2, 4, 6)
        predictions = model(
            text_hidden,
            base_weights,
            module_types_for_layer=["to_q", "to_v"],
            text_attention_mask=torch.ones(2, 4, dtype=torch.long),
        )
        self.assertEqual(predictions[0]["A"].shape, (2, 2, 3))
        self.assertEqual(predictions[0]["B"].shape, (2, 4, 2))
        self.assertEqual(predictions[1]["A"].shape, (2, 2, 3))
        self.assertEqual(predictions[1]["B"].shape, (2, 5, 2))

        targets = [
            {"A": torch.randn_like(predictions[0]["A"]), "B": torch.randn_like(predictions[0]["B"])},
            {"A": torch.randn_like(predictions[1]["A"]), "B": torch.randn_like(predictions[1]["B"])},
        ]
        loss = prompt2effect_loss(predictions, targets)
        self.assertTrue(torch.isfinite(loss))

        with tempfile.TemporaryDirectory() as tmp:
            output = save_generated_lora(
                Path(tmp),
                [{"A": predictions[0]["A"][0], "B": predictions[0]["B"][0]}],
                [{"module_name": "blocks.0.attn.to_q"}],
                component_prefix="transformer",
                rank=2,
                dtype=torch.float32,
            )
            state = load_file(output)
            self.assertIn("transformer.blocks.0.attn.to_q.lora_A.weight", state)
            self.assertIn("transformer.blocks.0.attn.to_q.lora_B.weight", state)
            self.assertIn("transformer.blocks.0.attn.to_q.alpha", state)


if __name__ == "__main__":
    unittest.main()
