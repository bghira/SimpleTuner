"""Provider configuration storage.

This module handles storage and retrieval of provider-specific
configuration (webhook URLs, cost limits, hardware info, etc.).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .base import BaseSQLiteStore, get_default_db_path

logger = logging.getLogger(__name__)


class ProviderConfigStore(BaseSQLiteStore):
    """Storage for provider-specific configuration.

    Provides CRUD operations for per-provider settings like:
    - Webhook URLs and secrets
    - Cost limits and billing config
    - Hardware pricing overrides
    - SSL/TLS settings
    """

    def _get_default_db_path(self) -> Path:
        return get_default_db_path("jobs.db")

    def _init_schema(self) -> None:
        """Initialize the provider_config table schema."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_config (
                    provider TEXT PRIMARY KEY,
                    config TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                )
            """
            )

            conn.commit()
        except Exception as exc:
            logger.error("Failed to initialize provider_config schema: %s", exc)
            raise
        finally:
            conn.close()

    async def get(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider.

        Args:
            provider: Provider name (e.g., "replicate").

        Returns:
            Configuration dictionary, empty if not found.
        """
        loop = asyncio.get_running_loop()

        def _get():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT config FROM provider_config WHERE provider = ?", (provider,))
                row = cursor.fetchone()
                if row:
                    try:
                        return json.loads(row["config"])
                    except json.JSONDecodeError:
                        return {}
                return {}
            finally:
                conn.close()

        return await loop.run_in_executor(None, _get)

    def get_sync(self, provider: str) -> Dict[str, Any]:
        """Get configuration for a specific provider (synchronous).

        Args:
            provider: Provider name (e.g., "replicate").

        Returns:
            Configuration dictionary, empty if not found.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT config FROM provider_config WHERE provider = ?", (provider,))
            row = cursor.fetchone()
            if row:
                try:
                    return json.loads(row["config"])
                except json.JSONDecodeError:
                    return {}
            return {}
        finally:
            conn.close()

    async def save(self, provider: str, config: Dict[str, Any]) -> None:
        """Save configuration for a specific provider.

        Args:
            provider: Provider name.
            config: Configuration dictionary.
        """
        loop = asyncio.get_running_loop()

        def _save():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO provider_config (provider, config, updated_at)
                    VALUES (?, ?, ?)
                """,
                    (provider, json.dumps(config), datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
            finally:
                conn.close()

        await loop.run_in_executor(None, _save)

    def save_sync(self, provider: str, config: Dict[str, Any]) -> None:
        """Save configuration for a specific provider (synchronous).

        Args:
            provider: Provider name.
            config: Configuration dictionary.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO provider_config (provider, config, updated_at)
                VALUES (?, ?, ?)
            """,
                (provider, json.dumps(config), datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

    async def update(self, provider: str, updates: Dict[str, Any]) -> None:
        """Merge updates into existing provider configuration.

        Args:
            provider: Provider name.
            updates: Fields to update (merged with existing config).
        """
        current = await self.get(provider)
        current.update(updates)
        await self.save(provider, current)

    async def delete(self, provider: str) -> bool:
        """Delete configuration for a provider.

        Args:
            provider: Provider name.

        Returns:
            True if deleted, False if not found.
        """
        loop = asyncio.get_running_loop()

        def _delete():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM provider_config WHERE provider = ?", (provider,))
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

        return await loop.run_in_executor(None, _delete)

    async def list_providers(self) -> list[str]:
        """List all configured providers.

        Returns:
            List of provider names.
        """
        loop = asyncio.get_running_loop()

        def _list():
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT provider FROM provider_config ORDER BY provider")
                return [row["provider"] for row in cursor.fetchall()]
            finally:
                conn.close()

        return await loop.run_in_executor(None, _list)

    # Convenience methods for common config patterns

    async def get_webhook_config(self, provider: str) -> Dict[str, Any]:
        """Get webhook-specific configuration.

        Returns:
            Dict with webhook_url, webhook_secret, webhook_require_signature, etc.
        """
        config = await self.get(provider)
        return {
            "webhook_url": config.get("webhook_url"),
            "webhook_secret": config.get("webhook_secret"),
            "webhook_require_signature": config.get("webhook_require_signature", True),
            "webhook_allowed_ips": config.get("webhook_allowed_ips", []),
        }

    async def get_cost_limit_config(self, provider: str) -> Dict[str, Any]:
        """Get cost limit configuration.

        Returns:
            Dict with cost_limit_enabled, cost_limit_amount, cost_limit_period, etc.
        """
        config = await self.get(provider)
        return {
            "enabled": config.get("cost_limit_enabled", False),
            "amount": config.get("cost_limit_amount", 0.0),
            "period": config.get("cost_limit_period", "daily"),
            "action": config.get("cost_limit_action", "warn"),
        }

    async def get_hardware_info(self, provider: str) -> Optional[Dict[str, Dict[str, Any]]]:
        """Get hardware pricing configuration.

        Returns:
            Dict of hardware_id -> {name, cost_per_second}, or None if not configured.
        """
        config = await self.get(provider)
        return config.get("hardware_info")


# Singleton accessor
_provider_config_store: Optional[ProviderConfigStore] = None


def get_provider_config_store() -> ProviderConfigStore:
    """Get the singleton ProviderConfigStore instance."""
    global _provider_config_store
    if _provider_config_store is None:
        _provider_config_store = ProviderConfigStore()
    return _provider_config_store
