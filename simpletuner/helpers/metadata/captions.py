from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from pandas import isna
from pandas.api.types import is_scalar


@dataclass
class CaptionRecord:
    """Canonical representation of a caption-only metadata entry."""

    metadata_id: str
    caption_text: str
    data_backend_id: str
    source_path: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> Dict[str, Any]:
        payload = {
            "metadata_id": self.metadata_id,
            "caption_text": self.caption_text,
            "data_backend_id": self.data_backend_id,
        }
        if self.source_path:
            payload["source_path"] = self.source_path
        if self.extras:
            payload["extras"] = self.extras
        return payload

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "CaptionRecord":
        extras = payload.get("extras") or {}
        return cls(
            metadata_id=str(payload["metadata_id"]),
            caption_text=str(payload["caption_text"]),
            data_backend_id=str(payload["data_backend_id"]),
            source_path=payload.get("source_path"),
            extras=dict(extras),
        )


def caption_records_revision(records: Iterable[CaptionRecord]) -> str:
    """Hash the ordered IDs and text in the caption checkpoint format."""
    digest = hashlib.sha256(b"[")
    for index, record in enumerate(records):
        if index:
            digest.update(b", ")
        digest.update(json.dumps([record.metadata_id, record.caption_text], ensure_ascii=False).encode("utf-8"))
    digest.update(b"]")
    return digest.hexdigest()


def parquet_sample_key(path, roots, *, includes_extension: bool, local: bool) -> str:
    filename = str(path)
    candidate = os.path.abspath(filename) if local else filename
    separator = os.sep if local else "/"
    for root in roots if isinstance(roots, (list, tuple)) else (roots,):
        if not root:
            continue
        prefix = os.path.abspath(root) if local else str(root)
        prefix = prefix.rstrip(separator) + separator
        if candidate.startswith(prefix):
            filename = candidate[len(prefix) :]
            break
    return filename if includes_extension else os.path.splitext(filename)[0]


def is_missing_caption(value: Any) -> bool:
    return is_scalar(value) and bool(isna(value))


def normalize_caption_text(value: Any) -> str:
    """
    Convert arbitrary input into a clean caption string.

    Empty strings are rejected so downstream components never receive blank prompts.
    """

    if is_missing_caption(value):
        raise ValueError("Caption text cannot be null.")
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip()
    if not text:
        raise ValueError("Caption text cannot be empty after stripping whitespace.")
    return text


__all__ = ["CaptionRecord", "caption_records_revision", "is_missing_caption", "normalize_caption_text"]
