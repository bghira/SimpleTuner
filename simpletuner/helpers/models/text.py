"""Collation of cached variable-length text embeddings and attention masks."""

import torch
import torch.nn.functional as F


def collate_embeddings(entries):
    embeddings, masks = [], []
    for entry in entries:
        value = entry["prompt_embeds"]
        mask = entry["attention_masks"]
        if value.ndim == 2:
            value, mask = value.unsqueeze(0), mask.reshape(1, -1)
        embeddings.extend(value.unbind(0))
        masks.extend(mask.bool().unbind(0))
    length = max(value.shape[0] for value in embeddings)
    return {
        "prompt_embeds": torch.stack([F.pad(value, (0, 0, 0, length - value.shape[0])) for value in embeddings]),
        "attention_masks": torch.stack([F.pad(value, (0, length - value.shape[0])) for value in masks]),
    }
