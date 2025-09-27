"""Service helpers for configuration management routes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import status

from simpletuner.simpletuner_sdk.server.services.config_store import ConfigMetadata, ConfigStore
from simpletuner.simpletuner_sdk.server.services.dataset_service import normalize_dataset_config_value
from simpletuner.simpletuner_sdk.server.services.field_registry import FieldType
from simpletuner.simpletuner_sdk.server.services.field_registry_wrapper import lazy_field_registry
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore
from simpletuner.simpletuner_sdk.server.utils.paths import resolve_config_path
from simpletuner.simpletuner_sdk.server.dependencies.common import _load_active_config_cached


class ConfigServiceError(Exception):
    """Domain error raised when configuration service operations fail."""

    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class ConfigsService:
    """Coordinator for configuration-related operations."""

    @staticmethod
    def _invalidate_active_config_cache() -> None:
        try:
            _load_active_config_cached.clear_cache()
        except AttributeError:
            pass

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

    @staticmethod
    def _resolve_under_base(base: Optional[str], value: str) -> str:
        """Resolve relative paths under a given base directory and normalise."""

        if not value:
            return value

        expanded_value = os.path.expanduser(value)
        if os.path.isabs(expanded_value):
            return os.path.normpath(expanded_value)

        if base:
            base_value = os.path.abspath(os.path.expanduser(base))
            rel = expanded_value.lstrip("./")
            base_name = os.path.basename(base_value.rstrip(os.sep))
            prefix = f"{base_name}{os.sep}"
            if rel.startswith(prefix):
                rel = rel[len(prefix):]
            return os.path.normpath(os.path.join(base_value, rel))

        return os.path.normpath(os.path.abspath(expanded_value))

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

        self._invalidate_active_config_cache()
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
        self._invalidate_active_config_cache()
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
            self._invalidate_active_config_cache()
            return {"message": f"Configuration '{name}' deleted successfully"}
        raise ConfigServiceError(f"Configuration '{name}' not found", status.HTTP_404_NOT_FOUND)

    def rename_config(self, name: str, new_name: str, config_type: str = "model") -> Dict[str, Any]:
        store = self._get_store(config_type)
        try:
            metadata = store.rename_config(name, new_name)
            self._invalidate_active_config_cache()
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
            self._invalidate_active_config_cache()
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
            self._invalidate_active_config_cache()
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
            self._invalidate_active_config_cache()
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
            self._invalidate_active_config_cache()
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

    # ------------------------------------------------------------------
    # Shared helpers for config normalization
    # ------------------------------------------------------------------
    @staticmethod
    def convert_value_by_type(value: Any, field_type: FieldType) -> Any:
        """Convert a raw value into the appropriate Python type."""

        if value is None:
            return None

        if field_type == FieldType.NUMBER:
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                cleaned = value.strip()
                if not cleaned:
                    return value
                try:
                    if "." in cleaned or "e" in cleaned.lower():
                        return float(cleaned)
                    return int(cleaned)
                except ValueError:
                    return value
            return value

        if field_type == FieldType.CHECKBOX:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value != 0
            if isinstance(value, str):
                return value.strip().lower() in {"true", "1", "yes", "on"}
            return bool(value)

        if field_type == FieldType.MULTI_SELECT:
            if isinstance(value, str):
                return [item.strip() for item in value.split(",") if item.strip()]
            if isinstance(value, (list, tuple, set)):
                return list(value)
            return value

        return value

    @staticmethod
    def normalize_form_to_config(
        form_data: Dict[str, Any],
        directory_fields: Optional[List[str]] = None,
        output_root: Optional[str] = None,
        configs_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Normalize submitted form data into CLI-style config dict."""

        directory_fields = directory_fields or []
        config_dict: Dict[str, Any] = {}
        excluded_fields = {"configs_dir", "__active_tab__"}

        field_types: Dict[str, FieldType] = {
            field.arg_name: field.field_type
            for field in lazy_field_registry.get_all_fields()
        }

        json_path_fields = {
            "--data_backend_config",
            "--webhooks_config",
            "--lycoris_config",
        }

        numeric_fields = {
            "--num_train_epochs",
            "--max_train_steps",
            "--lr_warmup_steps",
            "--gradient_accumulation_steps",
            "--train_batch_size",
        }

        always_include_fields = {"--model_flavour", "--optimizer_config"}

        for key, value in form_data.items():
            if key in excluded_fields:
                continue

            config_key = key if key.startswith("--") else f"--{key}"

            if config_key in numeric_fields:
                if value in (None, ""):
                    value = "0"
            elif config_key in always_include_fields:
                if value in (None, ""):
                    value = ""
            elif value in (None, ""):
                continue

            if config_key in directory_fields and value:
                base_dir = output_root if config_key == "--output_dir" else None
                expanded_value = ConfigsService._resolve_under_base(base_dir, value)
                config_dict[config_key] = expanded_value
                continue

            field_type = field_types.get(config_key, FieldType.TEXT)
            converted_value = ConfigsService.convert_value_by_type(value, field_type)

            if (
                isinstance(converted_value, str)
                and converted_value
                and converted_value.lower().endswith(".json")
            ):
                if config_key in json_path_fields:
                    converted_value = ConfigsService._resolve_under_base(configs_dir, converted_value)

                normalized_value = normalize_dataset_config_value(converted_value, configs_dir)
                if normalized_value:
                    converted_value = normalized_value

                if config_key in json_path_fields:
                    converted_value = ConfigsService._resolve_under_base(configs_dir, converted_value)

            config_dict[config_key] = converted_value

        return ConfigsService._migrate_legacy_keys(config_dict)

    @staticmethod
    def coerce_config_values_by_field(config: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce config values to the correct Python types based on field metadata."""

        coerced: Dict[str, Any] = {}
        for key, value in config.items():
            lookup_name = key[2:] if key.startswith("--") else key
            field = lazy_field_registry.get_field(lookup_name)
            if not field:
                coerced[key] = value
                continue

            coerced[key] = ConfigsService.convert_value_by_type(value, field.field_type)

        return coerced

    # ------------------------------------------------------------------
    # Legacy migrations
    # ------------------------------------------------------------------

    @staticmethod
    def _is_truthy(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"", "0", "false", "no", "off", "none"}:
                return False
            if lowered in {"1", "true", "yes", "on"}:
                return True
        return bool(value)

    @staticmethod
    def _pop_legacy_value(config: Dict[str, Any], base_key: str) -> Any:
        for variant in (base_key, f"--{base_key}"):
            if variant in config:
                return config.pop(variant)
        return None

    @staticmethod
    def _migrate_legacy_keys(config: Dict[str, Any]) -> Dict[str, Any]:
        """Rewrite legacy configuration keys to their current equivalents."""

        migrated = dict(config)

        legacy_limit = ConfigsService._pop_legacy_value(migrated, "save_total_limit")
        if legacy_limit is not None and "--checkpoints_total_limit" not in migrated:
            migrated["--checkpoints_total_limit"] = legacy_limit

        legacy_vae = ConfigsService._pop_legacy_value(migrated, "vae_cache_preprocess")
        if legacy_vae is not None and "--vae_cache_ondemand" not in migrated:
            if ConfigsService._is_truthy(legacy_vae):
                # Legacy true means preprocess; no flag required
                pass
            else:
                migrated["--vae_cache_ondemand"] = True

        # Remove dataset-level fields that should not be part of trainer CLI config
        for dataset_key in (
            "aspect_ratio_bucketing",
            "aspect_ratio_bucket_min",
            "aspect_ratio_bucket_max",
            "repeats",
        ):
            ConfigsService._pop_legacy_value(migrated, dataset_key)

        return migrated


# Singleton instance used by routes
CONFIGS_SERVICE = ConfigsService()
