"""Collate helper for caption-only datasets."""

from __future__ import annotations

from typing import Dict, List

from simpletuner.helpers.data_backend.dataset_types import DatasetType


def collate_caption_batch(examples: List[Dict]) -> Dict:
    """
    Convert CaptionDataset outputs into a training batch payload.

    Each example already represents a logical batch because the dataloader
    runs with batch_size=1. We unwrap the first element and expose caption
    strings and metadata ids for downstream components.
    """

    if not examples:
        return {
            "captions": [],
            "metadata_ids": [],
            "data_backend_id": None,
            "dataset_type": DatasetType.CAPTION,
            "records": [],
        }

    batch = examples[0] or {}
    raw_records = list(batch.get("records", []))
    records = []
    for record in raw_records:
        if not record:
            continue
        entry = dict(record)
        entry.setdefault("data_backend_id", batch.get("data_backend_id"))
        records.append(entry)

    return {
        "captions": [entry.get("caption_text", "") for entry in records],
        "metadata_ids": [entry.get("metadata_id") for entry in records],
        "data_backend_id": batch.get("data_backend_id"),
        "dataset_type": DatasetType.CAPTION,
        "records": records,
    }


__all__ = ["collate_caption_batch"]
