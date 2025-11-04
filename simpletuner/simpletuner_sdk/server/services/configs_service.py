"""Service helpers for configuration management routes."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import status

from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.simpletuner_sdk.server.dependencies.common import _load_active_config_cached
from simpletuner.simpletuner_sdk.server.services.config_store import ConfigMetadata, ConfigStore
from simpletuner.simpletuner_sdk.server.services.dataset_service import normalize_dataset_config_value
from simpletuner.simpletuner_sdk.server.services.example_configs_service import EXAMPLE_CONFIGS_SERVICE, ExampleConfigInfo
from simpletuner.simpletuner_sdk.server.services.field_registry import FieldType
from simpletuner.simpletuner_sdk.server.services.field_registry_wrapper import lazy_field_registry
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore
from simpletuner.simpletuner_sdk.server.utils.paths import get_simpletuner_root, resolve_config_path


class ConfigServiceError(Exception):
    """Domain error raised when configuration service operations fail."""

    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class ConfigsService:
    """Coordinator for configuration-related operations."""

    _CONFIG_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")

    @classmethod
    def _validate_config_name(cls, name: str) -> str:
        candidate = (name or "").strip()
        if candidate.lower().endswith(".json"):
            candidate = candidate[:-5].strip()

        if not candidate:
            raise ConfigServiceError("Configuration name is required", status.HTTP_400_BAD_REQUEST)
        if not cls._CONFIG_NAME_PATTERN.match(candidate):
            raise ConfigServiceError(
                "Configuration name may only contain letters, numbers, '.', '_' or '-'",
                status.HTTP_400_BAD_REQUEST,
            )
        return candidate

    @staticmethod
    def _resolve_pretrained_path(model_family: Optional[str], model_flavour: Optional[str]) -> Optional[str]:
        if not model_family:
            return None

        model = ModelRegistry.get(model_family)
        if not model:
            return None

        flavour = model_flavour or getattr(model, "DEFAULT_MODEL_FLAVOUR", None)
        hf_paths = getattr(model, "HUGGINGFACE_PATHS", None)
        if isinstance(hf_paths, dict) and hf_paths:
            if flavour and flavour in hf_paths:
                return hf_paths[flavour]
            default_flavour = getattr(model, "DEFAULT_MODEL_FLAVOUR", None)
            if default_flavour and default_flavour in hf_paths:
                return hf_paths[default_flavour]
            return next(iter(hf_paths.values()))

        default_repo = getattr(model, "DEFAULT_MODEL_NAME", None)
        if isinstance(default_repo, str) and default_repo.strip():
            return default_repo.strip()

        return None

    @staticmethod
    def _dataloader_destination(base_dir: Path, name: str, requested_path: Optional[str]) -> Path:
        if requested_path:
            candidate = Path(requested_path).expanduser()
            if not candidate.is_absolute():
                candidate = base_dir / candidate
        else:
            candidate = base_dir / name / "multidatabackend.json"

        resolved = candidate.resolve(strict=False)

        try:
            resolved.relative_to(base_dir)
        except ValueError as exc:
            raise ConfigServiceError(
                "Dataloader configs must live within the configs directory",
                status.HTTP_400_BAD_REQUEST,
            ) from exc

        return resolved

    @staticmethod
    @staticmethod
    def _format_relative_to_configs(path: Path, base_dir: Path) -> str:
        try:
            return path.resolve(strict=False).relative_to(base_dir.resolve(strict=False)).as_posix()
        except Exception:
            try:
                root = get_simpletuner_root()
                return path.resolve(strict=False).relative_to(root).as_posix()
            except Exception:
                try:
                    return path.resolve(strict=False).as_posix()
                except Exception:
                    return path.as_posix()

    @staticmethod
    def _write_default_dataloader(path: Path, environment: str, include_defaults: bool = True) -> None:
        if path.exists():
            raise ConfigServiceError(
                f"Dataloader config already exists at '{path}'",
                status.HTTP_409_CONFLICT,
            )

        path.parent.mkdir(parents=True, exist_ok=True)

        if include_defaults:
            default_plan: List[Dict[str, Any]] = [
                {
                    "id": f"{environment}-images",
                    "type": "local",
                    "dataset_type": "image",
                    "instance_data_dir": "",
                    "resolution": 1024,
                    "resolution_type": "pixel_area",
                    "caption_strategy": "textfile",
                    "minimum_image_size": 256,
                    "maximum_image_size": 4096,
                    "target_downsample_size": 1024,
                    "crop": True,
                    "crop_style": "center",
                    "crop_aspect": "square",
                },
                {
                    "id": f"{environment}-text-embeds",
                    "type": "local",
                    "dataset_type": "text_embeds",
                    "default": True,
                    "cache_dir": "",
                },
            ]
        else:
            # Create empty dataloader config
            default_plan = []

        with path.open("w", encoding="utf-8") as handle:
            json.dump(default_plan, handle, indent=2)
            handle.write("\n")

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
                rel = rel[len(prefix) :]
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

    def list_examples(self) -> Dict[str, Any]:
        examples = [example.to_public_dict() for example in EXAMPLE_CONFIGS_SERVICE.list_examples()]
        return {"examples": examples, "count": len(examples)}

    def generate_project_name(self) -> Dict[str, str]:
        return {"name": EXAMPLE_CONFIGS_SERVICE.generate_project_name()}

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
            raise ConfigServiceError(str(exc), status.HTTP_422_UNPROCESSABLE_CONTENT) from exc

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
                status.HTTP_422_UNPROCESSABLE_CONTENT,
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
        name = self._validate_config_name(name)
        store = self._get_store(config_type)
        validation = store.validate_config(config)
        if not validation.is_valid:
            raise ConfigServiceError(
                "Configuration validation failed",
                status.HTTP_422_UNPROCESSABLE_CONTENT,
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
        name = self._validate_config_name(name)
        store = self._get_store(config_type)
        validation = store.validate_config(config)
        if not validation.is_valid:
            raise ConfigServiceError(
                "Configuration validation failed",
                status.HTTP_422_UNPROCESSABLE_CONTENT,
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

    def create_environment(self, request: Any) -> Dict[str, Any]:
        name = self._validate_config_name(getattr(request, "name", None))
        model_family = getattr(request, "model_family", None)
        if not model_family:
            raise ConfigServiceError("model_family is required", status.HTTP_400_BAD_REQUEST)

        model_flavour = getattr(request, "model_flavour", None)
        model_type = getattr(request, "model_type", None) or "lora"
        lora_type = getattr(request, "lora_type", None)
        description = getattr(request, "description", None)
        example = getattr(request, "example", None)
        dataloader_path = getattr(request, "dataloader_path", None)
        create_dataloader = getattr(request, "create_dataloader", True)

        if not create_dataloader and not dataloader_path:
            raise ConfigServiceError(
                "dataloader_path is required when referencing an existing configuration",
                status.HTTP_400_BAD_REQUEST,
            )

        if model_type == "lora" and not lora_type and not example:
            lora_type = "standard"

        model_store = self._get_store("model")
        env_dir = Path(model_store.config_dir) / name

        if env_dir.exists():
            raise ConfigServiceError(
                f"Environment '{name}' already exists",
                status.HTTP_409_CONFLICT,
            )

        dataloader_abs = self._dataloader_destination(Path(model_store.config_dir), name, dataloader_path)
        dataloader_rel = self._format_relative_to_configs(dataloader_abs, Path(model_store.config_dir))

        example_info: Optional[ExampleConfigInfo] = None
        if example:
            try:
                example_info = EXAMPLE_CONFIGS_SERVICE.get_example(example)
            except FileNotFoundError as exc:
                available = [info.name for info in EXAMPLE_CONFIGS_SERVICE.list_examples()]
                message = str(exc)
                if available:
                    message += f". Available examples: {', '.join(available)}"
                raise ConfigServiceError(message, status.HTTP_404_NOT_FOUND) from exc

            source_path = example_info.config_path
            if source_path.name == "config.json" and source_path.parent.is_dir():
                shutil.copytree(source_path.parent, env_dir)
            else:
                env_dir.mkdir(parents=True, exist_ok=True)
                target_config = env_dir / "config.json"
                try:
                    with source_path.open("r", encoding="utf-8") as handle:
                        data = json.load(handle)
                except Exception as exc:
                    raise ConfigServiceError(
                        f"Failed to load example config '{example}': {exc}",
                        status.HTTP_422_UNPROCESSABLE_CONTENT,
                    ) from exc
                with target_config.open("w", encoding="utf-8") as handle:
                    json.dump(data, handle, indent=2, sort_keys=True)
                    handle.write("\n")
        else:
            env_dir.mkdir(parents=True, exist_ok=True)

        # Determine base configuration
        if example:
            config_file = env_dir / "config.json"
            if config_file.exists():
                try:
                    with config_file.open("r", encoding="utf-8") as handle:
                        raw_config = json.load(handle)
                    if isinstance(raw_config, dict) and "_metadata" in raw_config:
                        config_payload = dict(raw_config.get("config", {}) or {})
                    else:
                        config_payload = dict(raw_config) if isinstance(raw_config, dict) else {}
                except Exception:
                    config_payload = {}
            else:
                config_payload = dict(example_info.defaults) if example_info else {}
        else:
            try:
                base_config, _ = model_store.load_config("default")
                config_payload = dict(base_config)
            except Exception:
                config_payload = {}

        if example:
            lycoris_value = config_payload.get("lycoris_config") or config_payload.get("--lycoris_config")
            candidate_names = []
            if lycoris_value:
                try:
                    candidate_names.append(Path(str(lycoris_value)).name)
                except Exception:
                    pass
            candidate_names.append("lycoris_config.json")

            lycoris_path = None
            seen_names: set[str] = set()
            for lycoris_name in candidate_names:
                if not lycoris_name or lycoris_name in seen_names:
                    continue
                seen_names.add(lycoris_name)
                candidate = env_dir / lycoris_name
                if candidate.exists():
                    lycoris_path = candidate
                    break

            if lycoris_path is not None:
                lycoris_rel = self._format_relative_to_configs(
                    lycoris_path,
                    Path(model_store.config_dir),
                )
                config_payload["lycoris_config"] = lycoris_rel
                config_payload.pop("--lycoris_config", None)

        pretrained_path = self._resolve_pretrained_path(model_family, model_flavour)
        if not pretrained_path:
            pretrained_path = model_flavour or model_family

        overrides = {
            "data_backend_config": dataloader_rel,
        }

        if not example:
            overrides.update(
                {
                    "--model_family": model_family,
                    "--model_flavour": model_flavour,
                    "--model_type": model_type,
                    "--pretrained_model_name_or_path": pretrained_path,
                    "--output_dir": f"output/{name}",
                }
            )
            if model_type == "lora" and lora_type:
                overrides["--lora_type"] = lora_type

        for key, value in overrides.items():
            if value is None:
                continue
            config_payload[key] = value

        # Ensure canonical data backend key is present and remove deprecated duplicates
        config_payload["data_backend_config"] = dataloader_rel
        config_payload.pop("--data_backend_config", None)

        overwrite = example is not None
        model_store.save_config(name, config_payload, metadata=None, overwrite=overwrite)

        # Update metadata description if provided
        _, metadata = model_store.load_config(name)
        if description:
            metadata.description = description
        if not example:
            metadata.model_family = model_family or metadata.model_family
            metadata.model_type = model_type or metadata.model_type
            metadata.model_flavour = model_flavour or metadata.model_flavour
            if model_type == "lora" and lora_type:
                metadata.lora_type = lora_type
        model_store.save_config(name, config_payload, metadata, overwrite=True)

        # Handle dataloader file
        if create_dataloader:
            payload_written = False
            if example_info:
                source_path = getattr(example_info, "dataloader_path", None)
                if source_path:
                    try:
                        source_path = Path(source_path)
                    except Exception:
                        source_path = None
                if source_path and source_path.exists():
                    dataloader_abs.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        if source_path.resolve(strict=False) != dataloader_abs.resolve(strict=False):
                            shutil.copyfile(source_path, dataloader_abs)
                        else:
                            # If source and destination resolve the same, read/write to normalise perms
                            with source_path.open("r", encoding="utf-8") as handle:
                                data = handle.read()
                            with dataloader_abs.open("w", encoding="utf-8") as handle:
                                handle.write(data)
                    except Exception as exc:
                        raise ConfigServiceError(
                            f"Failed to copy dataloader config from example: {exc}",
                            status.HTTP_500_INTERNAL_SERVER_ERROR,
                        ) from exc
                    else:
                        payload_written = True

            if not payload_written and example_info and example_info.dataloader_payload is not None:
                dataloader_abs.parent.mkdir(parents=True, exist_ok=True)
                with dataloader_abs.open("w", encoding="utf-8") as handle:
                    json.dump(example_info.dataloader_payload, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                payload_written = True

            if not payload_written:
                example_dataloader_candidates = list(env_dir.glob("multidatabackend*.json"))
                if example_dataloader_candidates:
                    selected = example_dataloader_candidates[0]
                    if selected.resolve(strict=False) != dataloader_abs:
                        dataloader_abs.parent.mkdir(parents=True, exist_ok=True)
                        # Use atomic replace operation to avoid TOCTOU race condition
                        try:
                            # Path.replace() is atomic on most filesystems
                            selected.replace(dataloader_abs)
                        except OSError:
                            # If replace fails (e.g., file exists and can't be replaced),
                            # remove and retry atomically
                            if dataloader_abs.exists():
                                dataloader_abs.unlink()
                            selected.replace(dataloader_abs)
                    payload_written = True

            if not payload_written:
                self._write_default_dataloader(dataloader_abs, name)
        else:
            if not dataloader_abs.exists():
                raise ConfigServiceError(
                    f"Dataloader configuration not found: {dataloader_rel}",
                    status.HTTP_404_NOT_FOUND,
                )

        self._invalidate_active_config_cache()

        _, metadata = model_store.load_config(name)
        metadata_dict = metadata.model_dump()
        metadata_dict["path"] = str(env_dir)

        return {
            "message": f"Environment '{name}' created",
            "environment": metadata_dict,
            "config": config_payload,
            "dataloader": {
                "path": dataloader_rel,
                "absolute_path": str(dataloader_abs),
            },
        }

    def create_environment_dataloader(
        self, environment: str, requested_path: Optional[str], include_defaults: bool = True
    ) -> Dict[str, Any]:
        env_name = self._validate_config_name(environment)
        model_store = self._get_store("model")

        try:
            config_payload, metadata = model_store.load_config(env_name)
        except FileNotFoundError as exc:
            raise ConfigServiceError(str(exc), status.HTTP_404_NOT_FOUND) from exc

        config_payload = dict(config_payload)

        dataloader_abs = self._dataloader_destination(Path(model_store.config_dir), env_name, requested_path)
        dataloader_rel = self._format_relative_to_configs(dataloader_abs, Path(model_store.config_dir))

        if dataloader_abs.exists():
            raise ConfigServiceError(
                f"Dataloader config already exists at '{dataloader_rel}'",
                status.HTTP_409_CONFLICT,
            )

        self._write_default_dataloader(dataloader_abs, env_name, include_defaults)

        config_payload["data_backend_config"] = dataloader_rel
        config_payload.pop("--data_backend_config", None)

        model_store.save_config(env_name, config_payload, metadata, overwrite=True)
        self._invalidate_active_config_cache()

        return {
            "message": f"Dataloader for environment '{env_name}' created",
            "environment": metadata.model_dump(),
            "dataloader": {
                "path": dataloader_rel,
                "absolute_path": str(dataloader_abs),
            },
        }

    def delete_config(self, name: str, config_type: str = "model") -> Dict[str, Any]:
        name = self._validate_config_name(name)
        store = self._get_store(config_type)
        if store.get_active_config() == name:
            raise ConfigServiceError("Cannot delete the active configuration", status.HTTP_400_BAD_REQUEST)
        if store.delete_config(name):
            self._invalidate_active_config_cache()
            return {"message": f"Configuration '{name}' deleted successfully"}
        raise ConfigServiceError(f"Configuration '{name}' not found", status.HTTP_404_NOT_FOUND)

    def delete_dataloader_config(self, path: str) -> Dict[str, Any]:
        """Delete a dataloader configuration file by path."""
        store = self._get_store("model")
        resolved_path = resolve_config_path(path, config_dir=store.config_dir, check_cwd_first=True)
        if not resolved_path or not resolved_path.exists():
            raise ConfigServiceError(
                f"Dataloader config file not found: {path}",
                status.HTTP_404_NOT_FOUND,
            )

        allowed_dirs = self._allowed_directories(store)
        resolved_real = resolved_path.resolve()
        if not any(self._is_relative_to(resolved_real, directory.resolve()) for directory in allowed_dirs):
            raise ConfigServiceError(
                "Cannot delete files outside allowed directories",
                status.HTTP_400_BAD_REQUEST,
            )

        try:
            resolved_path.unlink()
            self._invalidate_active_config_cache()
            return {"message": f"Dataloader configuration deleted: {path}"}
        except OSError as exc:
            raise ConfigServiceError(
                f"Error deleting dataloader config file: {exc}",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc

    def rename_config(self, name: str, new_name: str, config_type: str = "model") -> Dict[str, Any]:
        name = self._validate_config_name(name)
        new_name = self._validate_config_name(new_name)
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
        name = self._validate_config_name(name)
        target_name = self._validate_config_name(target_name)
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
        name = self._validate_config_name(name)
        store = self._get_store()
        try:
            store.set_active_config(name)
            self._invalidate_active_config_cache()
            return {"message": f"Configuration '{name}' is now active", "active": name}
        except FileNotFoundError as exc:
            raise ConfigServiceError(str(exc), status.HTTP_404_NOT_FOUND) from exc

    def export_config(self, name: str, include_metadata: bool, config_type: str = "model") -> Dict[str, Any]:
        name = self._validate_config_name(name)
        store = self._get_store(config_type)
        try:
            return store.export_config(name, include_metadata)
        except FileNotFoundError as exc:
            raise ConfigServiceError(str(exc), status.HTTP_404_NOT_FOUND) from exc

    def import_config(
        self, data: Dict[str, Any], name: Optional[str], overwrite: bool, config_type: str = "model"
    ) -> Dict[str, Any]:
        store = self._get_store(config_type)
        if name:
            name = self._validate_config_name(name)
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
                status.HTTP_422_UNPROCESSABLE_CONTENT,
            ) from exc

    def create_from_template(self, template_name: str, config_name: str, config_type: str = "model") -> Dict[str, Any]:
        config_name = self._validate_config_name(config_name)
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
        name = self._validate_config_name(name)
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
    # Lycoris configuration management
    # ------------------------------------------------------------------

    def get_lycoris_config(self, environment_id: str) -> Optional[Dict]:
        """Load Lycoris configuration from file.

        Reads the lycoris_config path from environment config,
        then loads and returns the JSON file.
        Returns None if not configured or file doesn't exist.

        Args:
            environment_id: The environment name/ID

        Returns:
            The Lycoris configuration dict, or None if not found
        """
        environment_id = self._validate_config_name(environment_id)
        store = self._get_store("model")
        try:
            config, _ = store.load_config(environment_id)
        except FileNotFoundError:
            return None

        # Check for lycoris_config path in environment config
        lycoris_path_str = config.get("lycoris_config") or config.get("--lycoris_config")
        if not lycoris_path_str:
            return None

        # Resolve the lycoris config path
        lycoris_path = resolve_config_path(lycoris_path_str, config_dir=store.config_dir, check_cwd_first=True)
        if not lycoris_path or not lycoris_path.exists():
            return None

        # Load and return the lycoris config
        try:
            with lycoris_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None

    def save_lycoris_config(self, environment_id: str, lycoris_config: Dict) -> Dict:
        """Save Lycoris configuration to file.

        Gets the lycoris_config path from environment (or creates one),
        writes the config JSON, returns success status.

        Args:
            environment_id: The environment name/ID
            lycoris_config: The Lycoris configuration to save

        Returns:
            Dict with success status and path information

        Raises:
            ConfigServiceError: If environment not found or save fails
        """
        environment_id = self._validate_config_name(environment_id)
        store = self._get_store("model")
        try:
            config, metadata = store.load_config(environment_id)
        except FileNotFoundError as exc:
            raise ConfigServiceError(
                f"Environment '{environment_id}' not found",
                status.HTTP_404_NOT_FOUND,
            ) from exc

        config = dict(config)

        # Get or create lycoris_config path
        lycoris_path_str = config.get("lycoris_config") or config.get("--lycoris_config")
        if not lycoris_path_str:
            # Create default path: config/{env_id}/lycoris_config.json
            env_dir = Path(store.config_dir) / environment_id
            lycoris_path = env_dir / "lycoris_config.json"
            config["lycoris_config"] = self._format_relative_to_configs(
                lycoris_path,
                Path(store.config_dir),
            )
        else:
            lycoris_path_candidate = Path(str(lycoris_path_str)).expanduser()
            if not lycoris_path_candidate.is_absolute():
                lycoris_path_candidate = Path(store.config_dir) / lycoris_path_candidate

            lycoris_path = lycoris_path_candidate.resolve(strict=False)

            try:
                lycoris_path.relative_to(Path(store.config_dir).resolve(strict=False))
            except ValueError as exc:
                raise ConfigServiceError(
                    "LyCORIS configuration files must reside within the configs directory",
                    status.HTTP_400_BAD_REQUEST,
                ) from exc

        # Ensure parent directory exists
        lycoris_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the lycoris config
        try:
            with lycoris_path.open("w", encoding="utf-8") as handle:
                json.dump(lycoris_config, handle, indent=2, sort_keys=True)
                handle.write("\n")
        except OSError as exc:
            raise ConfigServiceError(
                f"Failed to write Lycoris config: {exc}",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc

        lycoris_rel = self._format_relative_to_configs(lycoris_path, Path(store.config_dir))
        config["lycoris_config"] = lycoris_rel
        config.pop("--lycoris_config", None)
        store.save_config(environment_id, config, metadata, overwrite=True)

        return {
            "success": True,
            "path": lycoris_rel,
            "absolute_path": str(lycoris_path),
        }

    def validate_lycoris_config(self, lycoris_config: Dict) -> Dict:
        """Validate Lycoris configuration structure.

        Args:
            lycoris_config: The Lycoris configuration to validate

        Returns:
            Dict with validation results: {
                "valid": bool,
                "errors": list[str],
                "warnings": list[str]
            }
        """
        errors = []
        warnings = []

        # Check required field: algo
        if "algo" not in lycoris_config:
            errors.append("Missing required field: 'algo'")
        else:
            algo = lycoris_config.get("algo")
            valid_algos = ["lora", "loha", "lokr", "locon", "oft", "boft", "glora", "full"]
            if algo not in valid_algos:
                warnings.append(f"Unknown algo '{algo}'. Known algorithms: {', '.join(valid_algos)}")

        # Check multiplier
        multiplier = lycoris_config.get("multiplier")
        if multiplier is not None:
            if not isinstance(multiplier, (int, float)):
                errors.append("Field 'multiplier' must be a number")
            elif multiplier <= 0:
                errors.append("Field 'multiplier' must be greater than 0")

        # Check linear_dim and linear_alpha for most algos (except full)
        algo = lycoris_config.get("algo")
        if algo and algo != "full":
            linear_dim = lycoris_config.get("linear_dim")
            if linear_dim is not None and not isinstance(linear_dim, int):
                errors.append("Field 'linear_dim' must be an integer")
            elif linear_dim is not None and linear_dim <= 0:
                errors.append("Field 'linear_dim' must be positive")

            linear_alpha = lycoris_config.get("linear_alpha")
            if linear_alpha is not None and not isinstance(linear_alpha, (int, float)):
                errors.append("Field 'linear_alpha' must be a number")
            elif linear_alpha is not None and linear_alpha <= 0:
                errors.append("Field 'linear_alpha' must be positive")

        # Check factor for lokr algo
        if algo == "lokr":
            factor = lycoris_config.get("factor")
            if factor is not None:
                if not isinstance(factor, int):
                    errors.append("Field 'factor' must be an integer for lokr algorithm")
                elif factor <= 0:
                    errors.append("Field 'factor' must be positive")

        # Validate apply_preset structure if present
        apply_preset = lycoris_config.get("apply_preset")
        if apply_preset is not None:
            if not isinstance(apply_preset, dict):
                errors.append("Field 'apply_preset' must be a dictionary")
            else:
                target_module = apply_preset.get("target_module")
                if target_module is not None and not isinstance(target_module, list):
                    errors.append("Field 'apply_preset.target_module' must be a list")

                module_algo_map = apply_preset.get("module_algo_map")
                if module_algo_map is not None and not isinstance(module_algo_map, dict):
                    errors.append("Field 'apply_preset.module_algo_map' must be a dictionary")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # Shared helpers for config normalization
    # ------------------------------------------------------------------
    @staticmethod
    def convert_value_by_type(
        value: Any, field_type: FieldType, default_value: Any = None, allow_empty: bool = False
    ) -> Any:
        """Convert a raw value into the appropriate Python type."""

        if value is None:
            return default_value

        if field_type == FieldType.NUMBER:
            if isinstance(value, (int, float)):
                return value
            if isinstance(value, str):
                cleaned = value.strip()
                if not cleaned:
                    return default_value if default_value is not None else None
                try:
                    if "." in cleaned or "e" in cleaned.lower():
                        return float(cleaned)
                    return int(cleaned)
                except ValueError:
                    # Return default value if conversion fails
                    return default_value if default_value is not None else None
            # If it's some other type, try to convert to int/float
            try:
                if isinstance(value, bool):
                    return int(value)
                return int(value)
            except (ValueError, TypeError):
                try:
                    return float(value)
                except (ValueError, TypeError):
                    return default_value if default_value is not None else None

        if field_type == FieldType.CHECKBOX:
            if isinstance(value, (list, tuple, set)):
                if not value:
                    return False
                for item in value:
                    if ConfigsService.convert_value_by_type(item, field_type):
                        return True
                return False
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return value != 0
            if isinstance(value, str):
                if not value.strip():
                    return default_value if default_value is not None else False
                return value.strip().lower() in {"true", "1", "yes", "on"}
            return bool(value)

        if field_type == FieldType.MULTI_SELECT:
            if isinstance(value, str):
                return [item.strip() for item in value.split(",") if item.strip()]
            if isinstance(value, (list, tuple, set)):
                return list(value)
            return value

        # For TEXT, SELECT, TEXTAREA, etc.
        if isinstance(value, str):
            cleaned = value.strip()
            # Handle the case where None comes through as "None" string
            if cleaned.lower() == "none":
                return default_value if default_value is not None else None
            if cleaned:
                if isinstance(default_value, bool):
                    lowered = cleaned.lower()
                    if lowered in {"1", "true", "yes", "on"}:
                        return True
                    if lowered in {"0", "false", "no", "off"}:
                        return False
                if isinstance(default_value, int):
                    try:
                        return int(cleaned)
                    except ValueError:
                        pass
                if isinstance(default_value, float):
                    try:
                        return float(cleaned)
                    except ValueError:
                        pass
                return cleaned
            # Empty string – if allow_empty is True, return empty string; otherwise defer to default
            if allow_empty:
                return ""
            if default_value is not None:
                return default_value
            return None

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
        excluded_fields = {
            "configs_dir",
            "__active_tab__",
        }

        for onboarding_field in lazy_field_registry.get_webui_onboarding_fields():
            excluded_fields.add(onboarding_field.name)
            arg_name = onboarding_field.arg_name or ""
            if arg_name:
                excluded_fields.add(arg_name)
                if not arg_name.startswith("--"):
                    excluded_fields.add(f"--{arg_name}")

        field_types: Dict[str, FieldType] = {
            field.arg_name: field.field_type for field in lazy_field_registry.get_all_fields()
        }

        json_path_fields = {
            "--data_backend_config",
            "--webhook_config",
            "--lycoris_config",
        }

        numeric_fields = {
            "--num_train_epochs",
            "--max_train_steps",
            "--lr_warmup_steps",
            "--gradient_accumulation_steps",
            "--train_batch_size",
        }

        nullable_numeric_fields = {
            "--accelerator_cache_clear_interval",
        }

        always_include_fields = {
            "--model_flavour",
            "--optimizer_config",
            "--output_dir",
            "--tracker_project_name",
            "--tracker_run_name",
        }

        for key, value in form_data.items():
            if key in excluded_fields:
                continue

            # Filter out Alpine.js UI state variables that should never reach the backend
            if (
                key.startswith("currentDataset.")
                or key.startswith("--currentDataset.")
                or key.startswith("datasets_page_")
                or key.startswith("--datasets_page_")
            ):
                continue

            if key in {"webhooks_config", "--webhooks_config"}:
                config_key = "--webhook_config"
            else:
                config_key = key if key.startswith("--") else f"--{key}"

            if isinstance(value, (list, tuple)):
                value = list(value)

            # Prefer explicit CLI keys over aliases. If we've already captured a value
            # for the canonical "--" key, skip later alias entries to avoid overwriting
            if config_key in config_dict and not key.startswith("--"):
                continue

            if config_key in numeric_fields:
                if value in (None, ""):
                    value = "0"
            elif config_key in nullable_numeric_fields:
                if value in (None, ""):
                    config_dict[config_key] = None
                    continue
            elif config_key in always_include_fields:
                if value in (None, ""):
                    value = ""
            elif value in (None, "") or (isinstance(value, (list, tuple)) and not value):
                # Check if the field allows empty values
                lookup_name = config_key[2:] if config_key.startswith("--") else config_key
                field = lazy_field_registry.get_field(lookup_name)
                if not (field and field.allow_empty):
                    continue

            if config_key in directory_fields and value:
                base_dir = output_root if config_key == "--output_dir" else None
                expanded_value = ConfigsService._resolve_under_base(base_dir, value)
                config_dict[config_key] = expanded_value
                continue

            field_type = field_types.get(config_key, FieldType.TEXT)

            # Get field for default value
            lookup_name = config_key[2:] if config_key.startswith("--") else config_key
            field = lazy_field_registry.get_field(lookup_name)
            default_value = field.default_value if field else None
            allow_empty = field.allow_empty if field else False

            if isinstance(value, list):
                if field_type == FieldType.CHECKBOX or field_type == FieldType.MULTI_SELECT:
                    converted_value = ConfigsService.convert_value_by_type(value, field_type, default_value, allow_empty)
                    config_dict[config_key] = converted_value
                    continue
                # webhook_config should preserve list structure (can contain multiple webhook configs)
                if config_key in {"--webhook_config", "webhook_config"}:
                    config_dict[config_key] = value
                    continue
                # Filter out empty strings from list before taking last element
                non_empty = [v for v in value if v not in (None, "")]
                value = non_empty[-1] if non_empty else ""

            converted_value = ConfigsService.convert_value_by_type(value, field_type, default_value, allow_empty)
            if config_key in always_include_fields and value in (None, ""):
                converted_value = "" if value in (None, "") else converted_value

            # Skip if the final converted value is empty (unless it's a field that must always be included or allows empty)
            if config_key not in always_include_fields and config_key not in numeric_fields:
                if converted_value in (None, ""):
                    # Don't skip if the field explicitly allows empty values
                    if not allow_empty:
                        continue

            if isinstance(converted_value, str) and converted_value and converted_value.lower().endswith(".json"):
                if config_key in json_path_fields:
                    converted_value = ConfigsService._resolve_under_base(configs_dir, converted_value)

                normalized_value = normalize_dataset_config_value(converted_value, configs_dir)
                if normalized_value:
                    converted_value = normalized_value

                if config_key in json_path_fields:
                    converted_value = ConfigsService._resolve_under_base(configs_dir, converted_value)

            if config_key in {"--deepspeed_config", "deepspeed_config"} and isinstance(converted_value, str):
                trimmed = converted_value.strip()
                if trimmed and (trimmed.startswith("{") or trimmed.startswith("[")):
                    try:
                        converted_value = json.loads(trimmed)
                    except json.JSONDecodeError:
                        # Keep original string if parsing fails; validation will surface errors later
                        pass
            if config_key in {"--tread_config", "tread_config"} and isinstance(converted_value, str):
                trimmed = converted_value.strip()
                if trimmed.startswith("{") or trimmed.startswith("["):
                    try:
                        converted_value = json.loads(trimmed)
                    except json.JSONDecodeError:
                        pass

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

            coerced[key] = ConfigsService.convert_value_by_type(
                value, field.field_type, field.default_value, field.allow_empty
            )

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

        legacy_webhook = ConfigsService._pop_legacy_value(migrated, "webhooks_config")
        if legacy_webhook is not None and "--webhook_config" not in migrated:
            migrated["--webhook_config"] = legacy_webhook

        legacy_checkpoint_steps = ConfigsService._pop_legacy_value(migrated, "save_every_n_steps")
        if legacy_checkpoint_steps is not None:
            if "--checkpoint_step_interval" not in migrated and "--checkpointing_steps" not in migrated:
                migrated["--checkpoint_step_interval"] = legacy_checkpoint_steps

        legacy_max_caption = ConfigsService._pop_legacy_value(migrated, "maximum_caption_length")
        if legacy_max_caption is not None and "--tokenizer_max_length" not in migrated:
            migrated["--tokenizer_max_length"] = legacy_max_caption

        legacy_project_name = ConfigsService._pop_legacy_value(migrated, "project_name")
        if isinstance(legacy_project_name, str) and legacy_project_name.strip().lower() in {"", "none", "null"}:
            legacy_project_name = None
        if legacy_project_name is not None and "--tracker_project_name" not in migrated:
            migrated["--tracker_project_name"] = legacy_project_name

        # Drop web UI helper keys if present
        migrated.pop("__active_tab__", None)
        migrated.pop("--__active_tab__", None)

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
