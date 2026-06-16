from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import safe_open
from torch.utils.data import Dataset

from scripts.prompt2effect.schema import TARGETS_FILENAME, load_schema


class Prompt2EffectTargetDataset(Dataset):
    def __init__(self, prepared_dir: str | Path):
        self.prepared_dir = Path(prepared_dir).expanduser()
        self.schema = load_schema(self.prepared_dir)
        self.targets_path = self.prepared_dir / TARGETS_FILENAME
        if not self.targets_path.is_file():
            raise FileNotFoundError(f"Prompt2Effect targets not found: {self.targets_path}")
        self.samples = list(self.schema.get("samples", []))
        self.layers = list(self.schema.get("layers", []))
        if not self.samples:
            raise ValueError(f"Prompt2Effect schema contains no samples: {self.prepared_dir}")
        if not self.layers:
            raise ValueError(f"Prompt2Effect schema contains no layers: {self.prepared_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        targets = []
        with safe_open(self.targets_path, framework="pt", device="cpu") as handle:
            for layer_idx, _layer in enumerate(self.layers):
                targets.append(
                    {
                        "A": handle.get_tensor(f"samples.{index}.layers.{layer_idx}.A"),
                        "B": handle.get_tensor(f"samples.{index}.layers.{layer_idx}.B"),
                    }
                )
        return {
            "id": sample["id"],
            "effect_prompt": sample["effect_prompt"],
            "targets": targets,
        }


def collate_prompt2effect_batch(examples: list[dict[str, Any]]) -> dict[str, Any]:
    if not examples:
        raise ValueError("Prompt2Effect dataloader produced an empty batch.")
    layer_count = len(examples[0]["targets"])
    targets = []
    for layer_idx in range(layer_count):
        targets.append(
            {
                "A": torch.stack([example["targets"][layer_idx]["A"] for example in examples], dim=0),
                "B": torch.stack([example["targets"][layer_idx]["B"] for example in examples], dim=0),
            }
        )
    return {
        "ids": [example["id"] for example in examples],
        "effect_prompts": [example["effect_prompt"] for example in examples],
        "targets": targets,
    }
