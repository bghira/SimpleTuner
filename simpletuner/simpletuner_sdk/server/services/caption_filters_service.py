"""Caption filter management service for the WebUI."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from fastapi import status

from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore
from simpletuner.simpletuner_sdk.server.utils.paths import get_config_directory

logger = logging.getLogger(__name__)

_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


class CaptionFilterError(Exception):
    """Domain-specific error signalling caption filter failures."""

    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


@dataclass
class CaptionFilterRecord:
    """Serializable representation of a caption filter definition."""

    name: str
    label: Optional[str]
    description: Optional[str]
    entries: List[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    path: str

    def to_public_dict(self) -> Dict[str, Any]:
        """Return JSON-serialisable payload for API responses."""

        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "entries": list(self.entries),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "path": self.path,
        }


class CaptionFiltersService:
    """Persistent storage helper for caption filter definitions."""

    def __init__(
        self,
        filters_dir: Optional[Path | str] = None,
        base_dir: Optional[Path | str] = None,
    ) -> None:
        self._base_dir = self._resolve_base_dir(base_dir)
        self._filters_dir = self._resolve_filters_dir(filters_dir)
        self._filters_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _resolve_base_dir(base_dir: Optional[Path | str]) -> Path:
        if base_dir:
            return Path(base_dir).expanduser()
        try:
            defaults = WebUIStateStore().load_defaults()
            if defaults.configs_dir:
                return Path(defaults.configs_dir).expanduser()
        except Exception:  # pragma: no cover - defaults unavailable
            logger.debug("Falling back to default config directory for caption filters", exc_info=True)
        return Path(get_config_directory()).expanduser()

    def _resolve_filters_dir(self, filters_dir: Optional[Path | str]) -> Path:
        if filters_dir:
            return Path(filters_dir).expanduser()
        # // ASSUMPTION: caption filters are stored as JSON files under a dedicated caption_filters directory.
        return self._base_dir / "caption_filters"

    @staticmethod
    def _normalise_name(name: Optional[str]) -> str:
        candidate = (name or "").strip()
        if not candidate:
            raise CaptionFilterError("Filter name is required", status.HTTP_400_BAD_REQUEST)
        if not _NAME_PATTERN.match(candidate):
            raise CaptionFilterError(
                "Filter name may only contain letters, numbers, '.', '_' or '-'",
                status.HTTP_400_BAD_REQUEST,
            )
        return candidate

    @staticmethod
    def _normalise_entries(entries: Optional[Iterable[str]]) -> List[str]:
        if entries is None:
            return []
        normalised: List[str] = []
        for entry in entries:
            if entry is None:
                continue
            line = str(entry).strip()
            if not line:
                continue
            normalised.append(line)
        return normalised

    @staticmethod
    def _normalise_optional_text(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    def _filter_path(self, name: str) -> Path:
        return self._filters_dir / f"{name}.json"

    def _relative_path(self, path: Path) -> str:
        try:
            return path.resolve(strict=False).relative_to(self._base_dir.resolve(strict=False)).as_posix()
        except Exception:
            try:
                return path.resolve(strict=False).as_posix()
            except Exception:  # pragma: no cover - fallback when resolve fails
                return path.as_posix()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _load_record(self, path: Path) -> Optional[CaptionFilterRecord]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return None
        except Exception as exc:  # pragma: no cover - corrupted file
            logger.warning("Failed to load caption filter from %s: %s", path, exc)
            return None

        if isinstance(payload, dict):
            name = payload.get("name") or path.stem
            entries = payload.get("entries", [])
            label = payload.get("label")
            description = payload.get("description")
            created_at = payload.get("created_at")
            updated_at = payload.get("updated_at")
        else:  # legacy plain list support
            name = path.stem
            entries = payload
            label = None
            description = None
            created_at = None
            updated_at = None

        normalised_entries = self._normalise_entries(entries)
        return CaptionFilterRecord(
            name=name,
            label=self._normalise_optional_text(label),
            description=self._normalise_optional_text(description),
            entries=normalised_entries,
            created_at=created_at,
            updated_at=updated_at,
            path=self._relative_path(path),
        )

    def list_filters(self) -> List[CaptionFilterRecord]:
        records: List[CaptionFilterRecord] = []
        if not self._filters_dir.exists():
            return records
        for path in sorted(self._filters_dir.glob("*.json")):
            record = self._load_record(path)
            if record:
                records.append(record)
        records.sort(key=lambda item: (item.label or item.name or "").lower())
        return records

    def get_filter(self, name: str) -> CaptionFilterRecord:
        normalised = self._normalise_name(name)
        path = self._filter_path(normalised)
        record = self._load_record(path)
        if not record:
            raise CaptionFilterError("Caption filter not found", status.HTTP_404_NOT_FOUND)
        return record

    def create_filter(self, payload: Dict[str, Any]) -> CaptionFilterRecord:
        name = self._normalise_name(payload.get("name"))
        path = self._filter_path(name)
        if path.exists():
            raise CaptionFilterError(
                f"Caption filter '{name}' already exists",
                status.HTTP_409_CONFLICT,
            )

        label = self._normalise_optional_text(payload.get("label"))
        description = self._normalise_optional_text(payload.get("description"))
        entries = self._normalise_entries(payload.get("entries"))
        if not entries:
            raise CaptionFilterError("Caption filter requires at least one entry", status.HTTP_400_BAD_REQUEST)

        timestamp = self._timestamp()
        record_payload = {
            "name": name,
            "label": label,
            "description": description,
            "entries": entries,
            "created_at": timestamp,
            "updated_at": timestamp,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(record_payload, handle, indent=2)
            handle.write("\n")
        logger.debug("Created caption filter '%s' at %s", name, path)
        return self.get_filter(name)

    def update_filter(self, original_name: str, payload: Dict[str, Any]) -> CaptionFilterRecord:
        existing = self.get_filter(original_name)
        new_name = self._normalise_name(payload.get("name") or existing.name)
        entries = self._normalise_entries(payload.get("entries") or existing.entries)
        if not entries:
            raise CaptionFilterError("Caption filter requires at least one entry", status.HTTP_400_BAD_REQUEST)

        label = self._normalise_optional_text(payload.get("label"))
        description = self._normalise_optional_text(payload.get("description"))

        current_path = self._filter_path(existing.name)
        new_path = self._filter_path(new_name)
        if new_path != current_path and new_path.exists():
            raise CaptionFilterError(
                f"Caption filter '{new_name}' already exists",
                status.HTTP_409_CONFLICT,
            )

        record_payload = {
            "name": new_name,
            "label": label,
            "description": description,
            "entries": entries,
            "created_at": existing.created_at or self._timestamp(),
            "updated_at": self._timestamp(),
        }

        with new_path.open("w", encoding="utf-8") as handle:
            json.dump(record_payload, handle, indent=2)
            handle.write("\n")

        if new_path != current_path and current_path.exists():
            current_path.unlink()

        logger.debug("Updated caption filter '%s' -> '%s'", original_name, new_name)
        return self.get_filter(new_name)

    def delete_filter(self, name: str) -> None:
        normalised = self._normalise_name(name)
        path = self._filter_path(normalised)
        if not path.exists():
            raise CaptionFilterError("Caption filter not found", status.HTTP_404_NOT_FOUND)
        path.unlink()
        logger.debug("Deleted caption filter '%s'", normalised)

    @staticmethod
    def apply_filters(entries: Iterable[str], sample: str) -> str:
        """Apply filter rules to a caption sample following CLI semantics."""

        result = sample or ""
        for raw_entry in entries:
            if raw_entry is None:
                continue
            entry = str(raw_entry)
            if not entry:
                continue

            if entry.startswith("s/") and entry.count("/") >= 2:
                try:
                    _, pattern, replacement, *_ = entry.split("/", 3)
                except ValueError:
                    pattern = ""
                    replacement = ""
                if pattern:
                    try:
                        result = re.sub(pattern, replacement, result)
                    except re.error as exc:  # pragma: no cover - invalid regex
                        logger.debug("Failed to apply sed-like replacement '%s': %s", entry, exc)

            if entry and result:
                result = result.replace(entry, "")

            try:
                regex = re.compile(entry)
            except re.error:
                continue
            try:
                result = regex.sub("", result)
            except re.error:
                logger.debug("Regex substitution failed for '%s'", entry)
        return result

    def test_entries(self, entries: Iterable[str], sample: str) -> str:
        return self.apply_filters(entries, sample)


CAPTION_FILTERS_SERVICE = CaptionFiltersService()
