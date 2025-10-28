from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


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


def normalize_caption_text(value: Any) -> str:
    """
    Convert arbitrary input into a clean caption string.

    Empty strings are rejected so downstream components never receive blank prompts.
    """

    if value is None:
        raise ValueError("Caption text cannot be None.")
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = str(value).strip()
    if not text:
        raise ValueError("Caption text cannot be empty after stripping whitespace.")
    return text


__all__ = ["CaptionRecord", "normalize_caption_text"]
