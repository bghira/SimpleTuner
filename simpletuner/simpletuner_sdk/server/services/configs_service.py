"""Service helpers for configuration management routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import status

from simpletuner.simpletuner_sdk.server.services.config_store import ConfigMetadata, ConfigStore
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore
from simpletuner.simpletuner_sdk.server.utils.paths import resolve_config_path


class ConfigServiceError(Exception):
    """Domain error raised when configuration service operations fail."""

    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class ConfigsService:
    """Coordinator for configuration-related operations."""

    def _get_store(self, config_type: str = "model") -> ConfigStore:
        """Return a ConfigStore using user defaults when available."""
        try:
            defaults = WebUIStateStore().load_defaults()
            if defaults.configs_dir:
                expanded_dir = Path(defaults.configs_dir).expanduser()
                return ConfigStore(config_dir=expanded_dir, config_type=config_type)
        except Exception:
            pass
        return ConfigStore(config_type=config_type)

    @staticmethod
    def _allowed_directories(store: ConfigStore) -> List[Path]:
        allowed = [Path.cwd()]
        try:
            allowed.append(Path(store.config_dir))
        except Exception:
            pass
        return allowed

    @staticmethod
    def _is_relative_to(candidate: Path, base: Path) -> bool:
        try:
            candidate.relative_to(base)
            return True
        except ValueError:
            return False

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------
    def list_configs(self, config_type: str = "model") -> Dict[str, Any]:
        store = self._get_store(config_type)
        configs = store.list_configs()
        active = store.get_active_config() if config_type == "model" else None
        return {
            "configs": configs,
            "active": active,
            "count": len(configs),
            "config_type": config_type,
        }

    def list_templates(self) -> Dict[str, Any]:
        store = self._get_store()
        templates = store.list_templates()
        return {"templates": templates, "count": len(templates)}

    def get_active_config(self) -> Dict[str, Any]:
        store = self._get_store()
        active_name = store.get_active_config()
        if not active_name:
            return {"name": None, "config": {}, "metadata": None}

        try:
            config, metadata = store.load_config(active_name)
        except FileNotFoundError as exc:
            raise ConfigServiceError(str(exc), status.HTTP_404_NOT_FOUND) from exc

        return {"name": active_name, "config": config, "metadata": metadata.model_dump()}

    def get_config(self, name: str, config_type: str = "model") -> Dict[str, Any]:
        store = self._get_store(config_type)
        try:
            config, metadata = store.load_config(name)
        except FileNotFoundError as exc:
            raise ConfigServiceError(str(exc), status.HTTP_404_NOT_FOUND) from exc
        except ValueError as exc:
            raise ConfigServiceError(str(exc), status.HTTP_422_UNPROCESSABLE_ENTITY) from exc

        return {"name": name, "config": config, "metadata": metadata.model_dump(), "config_type": config_type}

    def read_data_backend_file(self, path: str) -> Any:
        store = self._get_store()
        resolved_path = resolve_config_path(path, config_dir=store.config_dir, check_cwd_first=True)
        if not resolved_path:
            raise ConfigServiceError(
                f"Data backend config file not found: {path}",
                status.HTTP_404_NOT_FOUND,
            )

        allowed_dirs = self._allowed_directories(store)
        resolved_real = resolved_path.resolve()
        if not any(self._is_relative_to(resolved_real, directory.resolve()) for directory in allowed_dirs):
            raise ConfigServiceError(
                "Resolved path is outside allowed directories",
                status.HTTP_400_BAD_REQUEST,
            )

        try:
            with resolved_real.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError as exc:
            raise ConfigServiceError(
                f"Invalid JSON in data backend config file: {exc}",
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            ) from exc
        except OSError as exc:
            raise ConfigServiceError(
                f"Error reading data backend config file: {exc}",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc

    # ------------------------------------------------------------------
    # Mutating operations
    # ------------------------------------------------------------------
    def create_config(
        self,
        name: str,
        config: Dict[str, Any],
        description: Optional[str],
        tags: List[str],
        config_type: str = "model",
    ) -> Dict[str, Any]:
        store = self._get_store(config_type)
        validation = store.validate_config(config)
        if not validation.is_valid:
            raise ConfigServiceError(
                "Configuration validation failed",
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        metadata = ConfigMetadata(
            name=name,
            description=description,
            tags=tags,
            created_at="",
            modified_at="",
        )

        try:
            saved = store.save_config(name, config, metadata, overwrite=False)
        except FileExistsError as exc:
            raise ConfigServiceError(str(exc), status.HTTP_409_CONFLICT) from exc

        return {
            "message": f"Configuration '{name}' created successfully",
            "metadata": saved.model_dump(),
            "validation": {"warnings": validation.warnings, "suggestions": validation.suggestions},
        }

    def update_config(
        self,
        name: str,
        config: Dict[str, Any],
        description: Optional[str],
        tags: List[str],
        config_type: str = "model",
    ) -> Dict[str, Any]:
        store = self._get_store(config_type)
        validation = store.validate_config(config)
        if not validation.is_valid:
            raise ConfigServiceError(
                "Configuration validation failed",
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        try:
            _, existing_metadata = store.load_config(name)
        except FileNotFoundError as exc:
            raise ConfigServiceError(str(exc), status.HTTP_404_NOT_FOUND) from exc

        existing_metadata.description = description or existing_metadata.description
        existing_metadata.tags = tags if tags else existing_metadata.tags

        saved = store.save_config(name, config, existing_metadata, overwrite=True)
        return {
            "message": f"Configuration '{name}' updated successfully",
            "metadata": saved.model_dump(),
            "validation": {"warnings": validation.warnings, "suggestions": validation.suggestions},
        }

    def delete_config(self, name: str, config_type: str = "model") -> Dict[str, Any]:
        store = self._get_store(config_type)
        if store.get_active_config() == name:
            raise ConfigServiceError("Cannot delete the active configuration", status.HTTP_400_BAD_REQUEST)
        if store.delete_config(name):
            return {"message": f"Configuration '{name}' deleted successfully"}
        raise ConfigServiceError(f"Configuration '{name}' not found", status.HTTP_404_NOT_FOUND)

    def rename_config(self, name: str, new_name: str, config_type: str = "model") -> Dict[str, Any]:
        store = self._get_store(config_type)
        try:
            metadata = store.rename_config(name, new_name)
            return {
                "message": f"Configuration renamed from '{name}' to '{new_name}'",
                "metadata": metadata.model_dump(),
            }
        except FileNotFoundError as exc:
            raise ConfigServiceError(str(exc), status.HTTP_404_NOT_FOUND) from exc
        except FileExistsError as exc:
            raise ConfigServiceError(str(exc), status.HTTP_409_CONFLICT) from exc

    def copy_config(self, name: str, target_name: str, config_type: str = "model") -> Dict[str, Any]:
        store = self._get_store(config_type)
        try:
            metadata = store.copy_config(name, target_name)
            return {
                "message": f"Configuration '{name}' copied to '{target_name}'",
                "metadata": metadata.model_dump(),
            }
        except FileNotFoundError as exc:
            raise ConfigServiceError(str(exc), status.HTTP_404_NOT_FOUND) from exc
        except FileExistsError as exc:
            raise ConfigServiceError(str(exc), status.HTTP_409_CONFLICT) from exc

    def activate_config(self, name: str) -> Dict[str, Any]:
        store = self._get_store()
        try:
            store.set_active_config(name)
            return {"message": f"Configuration '{name}' is now active", "active": name}
        except FileNotFoundError as exc:
            raise ConfigServiceError(str(exc), status.HTTP_404_NOT_FOUND) from exc

    def export_config(self, name: str, include_metadata: bool, config_type: str = "model") -> Dict[str, Any]:
        store = self._get_store(config_type)
        try:
            return store.export_config(name, include_metadata)
        except FileNotFoundError as exc:
            raise ConfigServiceError(str(exc), status.HTTP_404_NOT_FOUND) from exc

    def import_config(self, data: Dict[str, Any], name: Optional[str], overwrite: bool, config_type: str = "model") -> Dict[str, Any]:
        store = self._get_store(config_type)
        try:
            metadata = store.import_config(data, name, overwrite)
            return {
                "message": f"Configuration imported as '{metadata.name}'",
                "metadata": metadata.model_dump(),
            }
        except FileExistsError as exc:
            raise ConfigServiceError(str(exc), status.HTTP_409_CONFLICT) from exc
        except Exception as exc:  # pragma: no cover - defensive guard
            raise ConfigServiceError(
                f"Failed to import configuration: {exc}",
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            ) from exc

    def create_from_template(self, template_name: str, config_name: str, config_type: str = "model") -> Dict[str, Any]:
        store = self._get_store(config_type)
        try:
            metadata = store.create_from_template(template_name, config_name)
            return {
                "message": f"Configuration '{config_name}' created from template '{template_name}'",
                "metadata": metadata.model_dump(),
            }
        except FileNotFoundError as exc:
            raise ConfigServiceError(str(exc), status.HTTP_404_NOT_FOUND) from exc
        except FileExistsError as exc:
            raise ConfigServiceError(str(exc), status.HTTP_409_CONFLICT) from exc

    def validate_config(self, name: str, config_type: str = "model") -> Dict[str, Any]:
        store = self._get_store(config_type)
        try:
            config, _ = store.load_config(name)
        except FileNotFoundError as exc:
            raise ConfigServiceError(str(exc), status.HTTP_404_NOT_FOUND) from exc

        validation = store.validate_config(config)
        return {
            "name": name,
            "is_valid": validation.is_valid,
            "errors": validation.errors,
            "warnings": validation.warnings,
            "suggestions": validation.suggestions,
        }

    def validate_config_data(self, config: Dict[str, Any], config_type: str = "model") -> Dict[str, Any]:
        store = self._get_store(config_type)
        validation = store.validate_config(config)
        return {
            "is_valid": validation.is_valid,
            "errors": validation.errors,
            "warnings": validation.warnings,
            "suggestions": validation.suggestions,
        }


# Singleton instance used by routes
CONFIGS_SERVICE = ConfigsService()
