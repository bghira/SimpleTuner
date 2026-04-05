"""BboxMetadata: parse per-entity bounding box annotations from various formats."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from simpletuner.helpers.training.grounding.types import BboxEntity

logger = logging.getLogger(__name__)


class BboxMetadata:
    """Static helpers to load entity annotations from sidecar files or string data."""

    @staticmethod
    def from_file(path: str) -> list[BboxEntity]:
        """Load from a ``.bbox`` sidecar file.

        Reads the file content then delegates to :meth:`from_string`.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Bbox sidecar file not found: {path}")
        content = p.read_text(encoding="utf-8").strip()
        if not content:
            return []
        return BboxMetadata.from_string(content)

    @staticmethod
    def from_string(data: str) -> list[BboxEntity]:
        """Parse entity annotations from a string.

        Supported formats:
        - JSON array: ``[{"label": "...", "bbox": [x1,y1,x2,y2], "mask": "path"}, ...]``
        - JSON lines: one JSON object per line
        - YOLO txt: ``class_id x_center y_center w h`` per line
        """
        data = data.strip()
        if not data:
            return []

        # Try JSON array first
        if data.startswith("["):
            return BboxMetadata._parse_json_array(data)

        # Try JSON lines (first line starts with '{')
        first_line = data.split("\n", 1)[0].strip()
        if first_line.startswith("{"):
            return BboxMetadata._parse_json_lines(data)

        # Fall back to YOLO txt format
        return BboxMetadata._parse_yolo_txt(data)

    # ------------------------------------------------------------------
    # Internal parsers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json_array(data: str) -> list[BboxEntity]:
        raw = json.loads(data)
        if not isinstance(raw, list):
            raise ValueError("Expected a JSON array of entity objects.")
        return [BboxMetadata._entity_from_dict(obj) for obj in raw]

    @staticmethod
    def _parse_json_lines(data: str) -> list[BboxEntity]:
        entities: list[BboxEntity] = []
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            entities.append(BboxMetadata._entity_from_dict(obj))
        return entities

    @staticmethod
    def _parse_yolo_txt(data: str) -> list[BboxEntity]:
        entities: list[BboxEntity] = []
        for line in data.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                logger.warning(f"Skipping malformed YOLO line: {line!r}")
                continue
            class_id = parts[0]
            x_center, y_center, w, h = (float(v) for v in parts[1:5])
            # Convert XYWH-center to XYXY
            x1 = x_center - w / 2.0
            y1 = y_center - h / 2.0
            x2 = x_center + w / 2.0
            y2 = y_center + h / 2.0
            bbox = BboxMetadata._clamp_and_validate(x1, y1, x2, y2)
            entities.append(BboxEntity(label=str(class_id), bbox=bbox))
        return entities

    @staticmethod
    def _entity_from_dict(obj: dict) -> BboxEntity:
        label = obj.get("label", "")
        raw_bbox = obj.get("bbox")
        if raw_bbox is None or len(raw_bbox) != 4:
            raise ValueError(f"Entity must have a 'bbox' with 4 elements, got: {raw_bbox}")
        x1, y1, x2, y2 = (float(v) for v in raw_bbox)
        bbox = BboxMetadata._clamp_and_validate(x1, y1, x2, y2)
        mask_path = obj.get("mask") or obj.get("mask_path")
        return BboxEntity(label=label, bbox=bbox, mask_path=mask_path)

    @staticmethod
    def _clamp_and_validate(x1: float, y1: float, x2: float, y2: float) -> tuple[float, float, float, float]:
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))
        x2 = max(0.0, min(1.0, x2))
        y2 = max(0.0, min(1.0, y2))
        if x1 >= x2 or y1 >= y2:
            raise ValueError(
                f"Invalid bbox after clamping: ({x1}, {y1}, {x2}, {y2}). "
                "Coordinates must satisfy x1 < x2 and y1 < y2 in [0, 1]."
            )
        return (x1, y1, x2, y2)
