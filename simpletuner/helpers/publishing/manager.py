from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, List, Optional

from .providers import (
    AzureBlobPublishingProvider,
    BackblazeB2PublishingProvider,
    DropboxPublishingProvider,
    PublishingProvider,
    PublishingResult,
    S3PublishingProvider,
)

logger = logging.getLogger(__name__)


class PublishingManager:
    """Coordinate uploads across multiple publishing providers."""

    def __init__(
        self,
        publishing_configs: Iterable[dict[str, Any]] | dict[str, Any] | None,
        *,
        provider_registry: dict[str, type[PublishingProvider]] | None = None,
        logger_instance: logging.Logger | None = None,
    ):
        self.logger = logger_instance or logger
        self.providers: list[PublishingProvider] = []
        self._registry = provider_registry or {
            "s3": S3PublishingProvider,
            "s3-compatible": S3PublishingProvider,
            "backblaze_b2": BackblazeB2PublishingProvider,
            "b2": BackblazeB2PublishingProvider,
            "azure_blob": AzureBlobPublishingProvider,
            "azure": AzureBlobPublishingProvider,
            "dropbox": DropboxPublishingProvider,
        }
        self._configure(publishing_configs)

    @staticmethod
    def _normalize_configs(
        configs: Iterable[dict[str, Any]] | dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if configs is None:
            return []
        if isinstance(configs, dict):
            return [configs]
        if isinstance(configs, (str, bytes)):
            raise ValueError("publishing_config string values should be parsed before reaching the PublishingManager.")
        if not isinstance(configs, Iterable):
            raise ValueError(f"publishing_config must be a dict or list, got {type(configs).__name__}")
        return list(configs)

    def _configure(self, configs: Iterable[dict[str, Any]] | dict[str, Any] | None) -> None:
        normalized = self._normalize_configs(configs)
        for config in normalized:
            provider_name = str(config.get("provider") or config.get("type") or "").strip().lower()
            if not provider_name:
                raise ValueError("Each publishing_config entry must define a provider.")
            provider_cls = self._registry.get(provider_name)
            if provider_cls is None:
                raise ValueError(f"Unsupported publishing provider '{provider_name}'.")
            try:
                provider = provider_cls(config)
            except Exception as exc:
                self.logger.error("Failed to initialise publishing provider '%s': %s", provider_name, exc)
                raise
            self.providers.append(provider)

    def publish(
        self,
        artifact_path: str | Path,
        *,
        artifact_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> List[PublishingResult]:
        if not self.providers:
            self.logger.debug("No publishing providers configured; skipping publish.")
            return []

        results: list[PublishingResult] = []
        path = Path(artifact_path)
        metadata = dict(metadata) if metadata else {}
        resolved_artifact_name = artifact_name or path.name

        for provider in self.providers:
            try:
                result = provider.publish(path, artifact_name=resolved_artifact_name, metadata=metadata)
                results.append(result)
            except Exception as exc:
                self.logger.error("Publishing via %s failed: %s", provider.display_name, exc)
        return results

    def latest_uri(self) -> Optional[str]:
        for provider in self.providers:
            uri = provider.get_last_uri()
            if uri:
                return uri
        self.logger.warning("No publishing provider returned a retrievable URI for the last upload.")
        return None

    @property
    def configured(self) -> bool:
        return len(self.providers) > 0
