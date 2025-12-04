"""Service managing validation prompt library files."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import status

from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore
from simpletuner.simpletuner_sdk.server.utils.paths import get_config_directory

logger = logging.getLogger(__name__)


class PromptLibraryError(Exception):
    """Domain error for prompt library operations."""

    def __init__(self, message: str, status_code: int = status.HTTP_400_BAD_REQUEST) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


@dataclass
class PromptLibraryRecord:
    filename: str
    relative_path: str
    absolute_path: str
    display_name: str
    library_name: str
    prompt_count: int
    updated_at: str


@dataclass
class PromptLibraryEntry:
    prompt: str
    adapter_strength: Optional[float] = None

    @classmethod
    def from_payload(cls, payload: Union[str, Dict[str, Any]]) -> "PromptLibraryEntry":
        if isinstance(payload, str):
            return cls(prompt=payload, adapter_strength=None)
        if not isinstance(payload, dict):
            raise PromptLibraryError("Prompt entries must be strings or objects with a prompt field.")
        prompt_value = payload.get("prompt")
        if prompt_value is None:
            raise PromptLibraryError("Prompt entry objects must include a 'prompt' field.")
        strength = payload.get("adapter_strength", None)
        try:
            strength_value = None if strength is None else float(strength)
        except (TypeError, ValueError):
            raise PromptLibraryError("adapter_strength must be numeric when provided.")
        return cls(prompt=str(prompt_value), adapter_strength=strength_value)

    def serialise(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {"prompt": self.prompt}
        if self.adapter_strength is not None:
            data["adapter_strength"] = self.adapter_strength
        return data


class PromptLibraryService:
    """Manages prompt library files stored under a configs directory."""

    _PREFIX = "user_prompt_library"
    _LIBRARIES_SUBDIR = "validation_prompt_libraries"
    _FILENAME_PATTERN = re.compile(rf"^{_PREFIX}(?:-([A-Za-z0-9._-]+))?\.json$")

    def __init__(self, config_dir: Optional[Path] = None, libraries_dir: Optional[Path] = None) -> None:
        self._config_dir = Path(config_dir) if config_dir else self._resolve_config_dir()
        self._config_dir.mkdir(parents=True, exist_ok=True)
        if libraries_dir:
            self._libraries_dir = Path(libraries_dir)
        else:
            self._libraries_dir = self._config_dir / self._LIBRARIES_SUBDIR
        self._libraries_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_config_dir(self) -> Path:
        try:
            defaults = WebUIStateStore().load_defaults()
            if defaults.configs_dir:
                candidate = Path(defaults.configs_dir).expanduser()
                candidate.mkdir(parents=True, exist_ok=True)
                return candidate
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.debug("Unable to resolve configs_dir from WebUI defaults", exc_info=exc)
        default = get_config_directory()
        default.mkdir(parents=True, exist_ok=True)
        return default

    def _validate_filename(self, filename: str) -> str:
        if not filename:
            raise PromptLibraryError("Prompt library filename is required.")
        candidate = Path(filename).name
        match = self._FILENAME_PATTERN.fullmatch(candidate)
        if not match:
            raise PromptLibraryError(
                "Prompt library filenames must be user_prompt_library[-name].json and only contain letters, numbers, '.', '_', or '-'."
            )
        return candidate

    @staticmethod
    def parse_entries(payload: Any) -> Dict[str, PromptLibraryEntry]:
        if not isinstance(payload, dict):
            raise PromptLibraryError("Prompt library entries must be an object with ID -> prompt mappings.")
        normalized: Dict[str, PromptLibraryEntry] = {}
        for key, value in payload.items():
            shortname = str(key).strip()
            if not shortname:
                continue
            if isinstance(value, PromptLibraryEntry):
                normalized[shortname] = value
                continue
            try:
                normalized[shortname] = PromptLibraryEntry.from_payload(value)
            except PromptLibraryError:
                raise
            except Exception as exc:
                raise PromptLibraryError(f"Invalid prompt entry for '{shortname}': {exc}")
        return normalized

    @staticmethod
    def serialise_entries(entries: Dict[str, PromptLibraryEntry]) -> Dict[str, Any]:
        serialised: Dict[str, Any] = {}
        for key, entry in entries.items():
            if entry.adapter_strength is None:
                serialised[key] = entry.prompt
            else:
                serialised[key] = entry.serialise()
        return serialised

    def _load_entries(self, path: Path) -> Dict[str, PromptLibraryEntry]:
        if not path.exists():
            raise PromptLibraryError(f"Prompt library '{path.name}' not found", status.HTTP_404_NOT_FOUND)
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError as exc:
            raise PromptLibraryError(f"Invalid JSON in '{path.name}': {exc}", status.HTTP_422_UNPROCESSABLE_CONTENT) from exc
        except OSError as exc:
            raise PromptLibraryError(f"Failed to read '{path.name}': {exc}", status.HTTP_500_INTERNAL_SERVER_ERROR) from exc
        return self.parse_entries(payload)

    def _build_metadata(self, path: Path, entries: Dict[str, str]) -> PromptLibraryRecord:
        match = self._FILENAME_PATTERN.fullmatch(path.name)
        library_name = match.group(1) if match else ""
        relative_path = self._relative_path(path)
        display_name = library_name if library_name else "default"
        absolute_path = str(path.resolve())
        updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        return PromptLibraryRecord(
            filename=path.name,
            relative_path=relative_path,
            absolute_path=absolute_path,
            display_name=display_name,
            library_name=library_name or "",
            prompt_count=len(entries),
            updated_at=updated_at,
        )

    def _relative_path(self, path: Path) -> str:
        try:
            return path.relative_to(self._config_dir).as_posix()
        except Exception:
            return path.name

    def list_libraries(self) -> List[PromptLibraryRecord]:
        records: List[PromptLibraryRecord] = []
        if not self._libraries_dir.exists():
            return records
        for path in sorted(self._libraries_dir.iterdir()):
            if not path.is_file():
                continue
            if not self._FILENAME_PATTERN.fullmatch(path.name):
                continue
            try:
                entries = self._load_entries(path)
            except PromptLibraryError as exc:
                logger.warning("Skipping prompt library '%s': %s", path.name, exc.message)
                continue
            records.append(self._build_metadata(path, entries))
        records.sort(key=lambda record: record.display_name.lower())
        return records

    def read_library(self, filename: str) -> Dict[str, Any]:
        sanitized = self._validate_filename(filename)
        path = self._libraries_dir / sanitized
        entries = self._load_entries(path)
        metadata = self._build_metadata(path, entries)
        return {
            "entries": self.serialise_entries(entries),
            "library": metadata,
        }

    def save_library(
        self,
        filename: str,
        entries: Dict[str, Union[str, Dict[str, Any], PromptLibraryEntry]],
        previous_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized = self.parse_entries(entries)
        sanitized = self._validate_filename(filename)
        target = self._libraries_dir / sanitized
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            with target.open("w", encoding="utf-8") as handle:
                json.dump(self.serialise_entries(normalized), handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        except OSError as exc:
            raise PromptLibraryError(
                f"Failed to write '{target.name}': {exc}", status.HTTP_500_INTERNAL_SERVER_ERROR
            ) from exc

        if previous_filename:
            previous = self._validate_filename(previous_filename)
            if previous != sanitized:
                previous_path = self._libraries_dir / previous
                try:
                    if previous_path.exists():
                        previous_path.unlink()
                except OSError as exc:
                    logger.warning("Could not remove old prompt library '%s': %s", previous, exc)

        metadata = self._build_metadata(target, normalized)
        return {"entries": self.serialise_entries(normalized), "library": metadata}
