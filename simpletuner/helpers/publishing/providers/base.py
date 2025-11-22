from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class PublishingResult:
    provider: str
    name: Optional[str]
    uri: Optional[str]
    artifact_path: Path
    metadata: dict[str, Any]


class PublishingProvider(ABC):
    """Base class for publishing providers."""

    def __init__(self, provider_type: str, config: dict[str, Any], *, display_name: str | None = None):
        self.provider_type = provider_type
        self.config = config or {}
        self.display_name = display_name or provider_type
        self._last_result: Optional[PublishingResult] = None
        self._warned_missing_uri = False
        self.logger = logging.getLogger(f"{__name__}.{self.provider_type}")

    @abstractmethod
    def publish(
        self,
        artifact_path: str | Path,
        *,
        artifact_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PublishingResult:
        """Upload the provided artifact path and return the most recent publish result."""

    def _record_result(
        self, *, artifact_path: Path, uri: str | None, metadata: dict[str, Any] | None = None
    ) -> PublishingResult:
        result = PublishingResult(
            provider=self.provider_type,
            name=self.display_name,
            uri=uri,
            artifact_path=artifact_path,
            metadata=metadata or {},
        )
        self._last_result = result
        return result

    def get_last_uri(self) -> Optional[str]:
        """Return the last published URI if available, otherwise log once then return None."""
        if self._last_result and self._last_result.uri:
            return self._last_result.uri
        if not self._warned_missing_uri:
            self.logger.warning(
                "%s provider does not expose a retrievable URI for the last upload.",
                self.display_name,
            )
            self._warned_missing_uri = True
        return None

    @staticmethod
    def _normalize_path_fragment(fragment: str | None) -> str:
        if not fragment:
            return ""
        return fragment.strip("/").strip()

    def _build_destination_root(self, artifact_name: str | None) -> str:
        base_path = self._normalize_path_fragment(self.config.get("base_path"))
        clean_artifact = artifact_name.strip("/").strip() if artifact_name else None
        parts = [part for part in (base_path, clean_artifact) if part]
        return "/".join(parts)

    @staticmethod
    def _iter_files(path: Path) -> list[Path]:
        if path.is_file():
            return [path]
        if not path.is_dir():
            raise FileNotFoundError(f"Artifact path does not exist: {path}")
        return [candidate for candidate in path.rglob("*") if candidate.is_file()]
