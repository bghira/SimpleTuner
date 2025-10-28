"""Dataset wrapper for caption-only metadata."""

from __future__ import annotations

from typing import List, Sequence

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.metadata.backends.caption import CaptionMetadataBackend
from simpletuner.helpers.metadata.captions import CaptionRecord

try:  # pragma: no cover - exercised indirectly in environments without torch
    from torch.utils.data import Dataset
except Exception:  # noqa: BLE001

    class Dataset:  # type: ignore[override]
        """Minimal shim used when torch isn't installed in lightweight test environments."""

        pass


class CaptionDataset(Dataset):
    """Thin Dataset wrapper that returns caption metadata batches."""

    def __init__(self, id: str, metadata_backend: CaptionMetadataBackend):
        self.id = id
        self.metadata_backend = metadata_backend

    def __len__(self) -> int:
        return len(self.metadata_backend)

    def __getitem__(self, metadata_batch):
        metadata_ids = self._normalize_batch(metadata_batch)
        records: List[dict] = []
        for entry in metadata_ids:
            if isinstance(entry, dict):
                payload = dict(entry)
            elif isinstance(entry, CaptionRecord):
                payload = entry.to_payload()
            else:
                record = self.metadata_backend.get_record(str(entry))
                if record is None:
                    raise KeyError(f"Caption metadata {entry} not found in backend {self.id}.")
                payload = record.to_payload()
            payload.setdefault("data_backend_id", self.id)
            records.append(payload)

        return {
            "data_backend_id": self.id,
            "dataset_type": DatasetType.CAPTION,
            "records": records,
        }

    def _normalize_batch(self, metadata_batch) -> Sequence:
        if metadata_batch is None:
            return []
        if isinstance(metadata_batch, (list, tuple)):
            return metadata_batch
        return [metadata_batch]
