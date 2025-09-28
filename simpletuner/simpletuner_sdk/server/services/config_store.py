"""Configuration store for managing training configurations."""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from simpletuner.helpers.configuration.cmd_args import get_argument_parser as _get_cli_argument_parser
    from simpletuner.helpers.configuration.cmd_args import get_model_flavour_choices as _cli_get_model_flavour_choices
    from simpletuner.helpers.configuration.cmd_args import model_family_choices as _cli_model_family_choices
except Exception:  # pragma: no cover - fallback when CLI utilities unavailable
    _get_cli_argument_parser = None
    _cli_get_model_flavour_choices = None
    _cli_model_family_choices = []

try:
    from simpletuner.helpers.models.all import get_all_model_flavours as _get_all_model_flavours
    from simpletuner.helpers.models.all import model_families as _MODEL_FAMILY_MAP
except Exception:  # pragma: no cover - fallback when model metadata unavailable
    _get_all_model_flavours = None
    _MODEL_FAMILY_MAP = {}

if _get_cli_argument_parser:
    try:
        _CLI_ARGUMENT_PARSER = _get_cli_argument_parser()
    except Exception:
        _CLI_ARGUMENT_PARSER = None
else:
    _CLI_ARGUMENT_PARSER = None

from pydantic import BaseModel, Field

from simpletuner.simpletuner_sdk.server.services.cache_service import get_config_cache
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore
from simpletuner.simpletuner_sdk.server.utils.paths import get_simpletuner_root

# Environment variables for configuration paths
_CONFIG_ENV_DIR = "SIMPLETUNER_CONFIG_DIR"
_CONFIG_ACTIVE = "SIMPLETUNER_ACTIVE_CONFIG"

# Default paths - will be resolved relative to SimpleTuner root
_DEFAULT_CONFIG_DIR = "config"
_DEFAULT_DATALOADER_DIR = "config/dataloaders"
_DEFAULT_TEMPLATE_DIR = "config/templates"
_DEFAULT_CONFIG_FILE = "config/config.json"


def _extract_parser_choices(option_name: str) -> List[str]:
    """Safely pull available choices for an argparse option."""

    if not _CLI_ARGUMENT_PARSER:
        return []

    action = _CLI_ARGUMENT_PARSER._option_string_actions.get(option_name)  # type: ignore[attr-defined]
    if not action or not getattr(action, "choices", None):
        return []

    return list(action.choices)


def _load_model_family_choices() -> List[str]:
    from simpletuner.configure import model_classes

    return model_classes["full"]


def _load_model_type_choices() -> List[str]:
    choices = _extract_parser_choices("--model_type")
    if choices:
        return choices
    # Fallback to historical defaults if parser metadata is unavailable
    return ["lora", "full", "controlnet", "embedding"]


def _load_model_flavour_choices() -> List[str]:
    choices = _extract_parser_choices("--model_flavour")
    if choices:
        return choices
    if _get_all_model_flavours:
        try:
            return list(_get_all_model_flavours())
        except Exception:
            return []
    return []


def _get_flavour_choices_for_family(model_family: Optional[str]) -> List[str]:
    if not model_family:
        return []

    if _cli_get_model_flavour_choices:
        try:
            flavours = _cli_get_model_flavour_choices(model_family)
            if isinstance(flavours, str):
                return [flavours]
            return list(flavours)
        except Exception:
            pass

    if _MODEL_FAMILY_MAP and model_family in _MODEL_FAMILY_MAP:
        try:
            return list(_MODEL_FAMILY_MAP[model_family].get_flavour_choices())
        except Exception:
            return []

    return []


_MODEL_FAMILY_CHOICES = _load_model_family_choices()
if not _MODEL_FAMILY_CHOICES:
    raise RuntimeError("No model families available - check model metadata")
_MODEL_FAMILY_SET = set(_MODEL_FAMILY_CHOICES)

_MODEL_TYPE_CHOICES = _load_model_type_choices()
_MODEL_TYPE_SET = set(_MODEL_TYPE_CHOICES)

_MODEL_FLAVOUR_CHOICES = _load_model_flavour_choices()
_MODEL_FLAVOUR_SET = set(_MODEL_FLAVOUR_CHOICES)


class ConfigMetadata(BaseModel):
    """Metadata for a configuration."""

    name: str
    description: Optional[str] = None
    model_family: Optional[str] = None
    model_type: Optional[str] = None
    model_flavour: Optional[str] = None
    lora_type: Optional[str] = None
    created_at: str
    modified_at: str
    tags: List[str] = Field(default_factory=list)
    is_template: bool = False
    parent_template: Optional[str] = None


class ConfigValidation(BaseModel):
    """Validation result for a configuration."""

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class ConfigStore:
    """Store for managing training configurations.

    Uses a singleton pattern with caching to avoid repeated filesystem hits.
    """

    _instances = {}  # Singleton instances keyed by (config_dir, config_type)

    def __new__(cls, config_dir: Optional[Path | str] = None, config_type: str = "model"):
        """Create or return existing instance (singleton pattern)."""
        # Create cache key from params
        key = (str(config_dir) if config_dir else None, config_type)

        if key not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[key] = instance
            instance._initialized = False

        return cls._instances[key]

    def __init__(self, config_dir: Optional[Path | str] = None, config_type: str = "model"):
        """Initialize the config store.

        Args:
            config_dir: Directory to store configurations. Defaults to config/environments or config/dataloaders.
            config_type: Type of configuration ('model' or 'dataloader'). Defaults to 'model'.
        """
        # Skip initialization if already done (singleton pattern)
        if self._initialized:
            return

        self.config_type = config_type
        self._cache = get_config_cache()  # Use global config cache

        if config_dir is not None:
            # Expand user home directory if present
            self.config_dir = Path(os.path.expanduser(str(config_dir)))
        else:
            self.config_dir = self._resolve_config_dir()

        self.template_dir = self._resolve_template_dir()
        self._ensure_directories()
        self._initialized = True

    def _resolve_config_dir(self) -> Path:
        """Resolve the configuration directory from environment or default."""
        env_dir = os.environ.get(_CONFIG_ENV_DIR)
        if env_dir:
            return Path(env_dir)

        try:
            state_store = WebUIStateStore()
            defaults = state_store.load_defaults()
            if self.config_type == "model" and defaults.configs_dir:
                return Path(defaults.configs_dir).expanduser()
            if self.config_type == "dataloader" and defaults.configs_dir:
                return Path(defaults.configs_dir).expanduser() / "dataloaders"
        except Exception:
            pass

        # Get SimpleTuner root and resolve relative to it
        simpletuner_root = get_simpletuner_root()
        if self.config_type == "dataloader":
            return simpletuner_root / _DEFAULT_DATALOADER_DIR
        return simpletuner_root / _DEFAULT_CONFIG_DIR

    @staticmethod
    def _resolve_template_dir() -> Path:
        """Resolve the template directory."""
        simpletuner_root = get_simpletuner_root()
        return simpletuner_root / _DEFAULT_TEMPLATE_DIR

    def _ensure_directories(self):
        """Ensure configuration directories exist."""
        # Always ensure templates exist (shared across config types)
        self.template_dir.mkdir(parents=True, exist_ok=True)

        if self.config_type == "dataloader":
            # Avoid creating a dedicated dataloaders directory unless the user explicitly saves one.
            # Dataloaders typically live alongside their environments.
            return

        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Create default config if it doesn't exist
        default_config = self.config_dir / "default.json"
        if not default_config.exists():
            self._create_default_config(default_config)

    def _create_default_config(self, path: Path):
        """Create a default configuration file."""
        if self.config_type == "dataloader":
            # Default dataloader config
            default_config: list[dict[str, Any]] = []

            metadata = self._create_metadata("default", "Default dataloader configuration")
        else:
            # Get onboarding values for default config
            output_dir = "output/models"
            try:
                state_store = WebUIStateStore()
                defaults = state_store.load_defaults()
                if defaults.output_dir:
                    output_dir = defaults.output_dir
            except Exception:
                # Fall back to hardcoded default if WebUI state is not available
                pass

            # Default model config
            default_config = {
                "--model_type": "lora",
                "--model_family": "flux",
                "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
                "--output_dir": output_dir,
                "--train_batch_size": 1,
                "--learning_rate": 0.0001,
                "--max_train_steps": 1000,
                "--checkpointing_steps": 100,
                "--seed": 42,
                "--resolution": 1024,
                "--mixed_precision": "bf16",
                "--gradient_checkpointing": True,
                "--use_ema": False,
            }

            metadata = self._create_metadata("default", "Default training configuration")

        with path.open("w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=2, sort_keys=True)
            f.write("\n")

        # Persist metadata alongside the flattened config for quick lookup
        self._save_metadata_sidecar(path, metadata)

    def _create_metadata(self, name: str, description: str = None) -> ConfigMetadata:
        """Create metadata for a configuration."""
        now = datetime.now(timezone.utc).isoformat()
        return ConfigMetadata(name=name, description=description, created_at=now, modified_at=now)

    def _get_config_path(self, name: str) -> Path:
        """Get the path for a configuration file."""
        if self.config_type == "model":
            dir_path = self.config_dir / name
            subdir_path = dir_path / "config.json"
            if dir_path.exists() or subdir_path.exists():
                return subdir_path

            # Fall back to root-level JSON file
            return self.config_dir / f"{name}.json"

        # Dataloader configs support both the dedicated dataloaders directory
        # and environment-scoped paths that live alongside the training config.
        dataloader_candidates = [
            self.config_dir / name / "multidatabackend.json",
            self.config_dir / f"{name}.json",
        ]

        dir_path = self.config_dir / name
        if dir_path.exists():
            dataloader_candidates.insert(0, dir_path / "multidatabackend.json")

        # Environment-local config folder (e.g. config/<env>/multidatabackend.json)
        env_root = self.config_dir.parent
        if env_root != self.config_dir:
            dataloader_candidates.append(env_root / name / "multidatabackend.json")
            dataloader_candidates.append(env_root / f"{name}.json")

        for candidate in dataloader_candidates:
            try:
                if candidate.exists():
                    return candidate
            except OSError:
                continue

        # If nothing matched, prefer the dedicated dataloaders directory so new
        # configs created via ConfigStore continue to live under config/dataloaders.
        return dataloader_candidates[0]

    def _get_template_path(self, name: str) -> Path:
        """Get the path for a template file."""
        return self.template_dir / f"{name}.json"

    def _is_folder_config(self, name: str) -> bool:
        """Check if a configuration is folder-based."""
        if self.config_type == "model":
            folder_path = self.config_dir / name / "config.json"
            return folder_path.exists()
        else:
            folder_path = self.config_dir / name / "multidatabackend.json"
            if folder_path.exists():
                return True

            env_root = self.config_dir.parent
            if env_root != self.config_dir:
                env_folder = env_root / name / "multidatabackend.json"
                return env_folder.exists()
            return False

    def _metadata_file_path(self, config_path: Path) -> Path:
        """Return the canonical metadata sidecar path for a configuration."""

        if config_path.name == "config.json":
            return config_path.parent / ".metadata.json"
        return config_path.with_suffix(".metadata.json")

    def _load_metadata_sidecar(self, config_path: Path) -> Optional[Dict[str, Any]]:
        if self.config_type != "model":
            return None

        metadata_path = self._metadata_file_path(config_path)
        if not metadata_path.exists():
            return None

        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                return payload
        except Exception:
            return None
        return None

    def _save_metadata_sidecar(self, config_path: Path, metadata: ConfigMetadata) -> None:
        if self.config_type != "model":
            return

        metadata_path = self._metadata_file_path(config_path)
        metadata_payload = metadata.model_dump()
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata_payload, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def _delete_metadata_sidecar(self, config_path: Path) -> None:
        if self.config_type != "model":
            return

        metadata_path = self._metadata_file_path(config_path)
        try:
            if metadata_path.exists():
                metadata_path.unlink()
        except Exception:
            pass

    def list_configs(self) -> List[Dict[str, Any]]:
        """List all available configurations.

        Returns:
            List of configuration metadata.
        """
        configs = []

        def _extract_backend_path(config_obj: Any) -> Optional[str]:
            if not isinstance(config_obj, dict):
                return None
            for key in (
                "--data_backend_config",
                "data_backend_config",
                "--dataloader_config",
                "dataloader_config",
                "dataloader_path",
            ):
                value = config_obj.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return None

        if self.config_type == "model":
            # For model configs, look for config.json in subdirectories
            if self.config_dir.exists():
                for subdir in self.config_dir.iterdir():
                    if subdir.is_dir():
                        config_file = subdir / "config.json"
                        if config_file.exists():
                            try:
                                with config_file.open("r", encoding="utf-8") as f:
                                    data = json.load(f)
                            except Exception:
                                continue

                            config_data = data.get("config", data) if isinstance(data, dict) else data
                            backend_path = _extract_backend_path(config_data)
                            metadata: Optional[Dict[str, Any]] = None
                            if isinstance(data, dict) and "_metadata" in data:
                                metadata = data["_metadata"].copy()
                                metadata["name"] = subdir.name  # Use folder name
                            else:
                                sidecar = self._load_metadata_sidecar(config_file)
                                if isinstance(sidecar, dict):
                                    metadata = sidecar.copy()
                                    metadata.setdefault("name", subdir.name)
                            if metadata is None:
                                metadata = {
                                    "name": subdir.name,
                                    "created_at": datetime.fromtimestamp(
                                        config_file.stat().st_ctime, tz=timezone.utc
                                    ).isoformat(),
                                    "modified_at": datetime.fromtimestamp(
                                        config_file.stat().st_mtime, tz=timezone.utc
                                    ).isoformat(),
                                }
                                # Extract model info from config data
                                if isinstance(config_data, dict):
                                    if "--model_family" in config_data:
                                        metadata["model_family"] = config_data["--model_family"]
                                    elif "model_family" in config_data:
                                        metadata["model_family"] = config_data["model_family"]

                                    if "--model_type" in config_data:
                                        metadata["model_type"] = config_data["--model_type"]
                                    elif "model_type" in config_data:
                                        metadata["model_type"] = config_data["model_type"]

                                    if "--model_flavour" in config_data:
                                        metadata["model_flavour"] = config_data["--model_flavour"]
                                    elif "model_flavour" in config_data:
                                        metadata["model_flavour"] = config_data["model_flavour"]

                                    if "--lora_type" in config_data:
                                        metadata["lora_type"] = config_data["--lora_type"]
                                    elif "lora_type" in config_data:
                                        metadata["lora_type"] = config_data["lora_type"]

                            if backend_path:
                                metadata.setdefault("dataloader_path", backend_path)
                                metadata.setdefault("data_backend_config", backend_path)
                                metadata["has_dataloader"] = True
                            else:
                                metadata.setdefault("has_dataloader", False)

                            configs.append(metadata)

            # Also check root-level JSON files for backward compatibility
            # But exclude files that are clearly dataloader configs
            if self.config_dir.exists():
                for config_file in self.config_dir.glob("*.json"):
                    try:
                        with config_file.open("r", encoding="utf-8") as f:
                            data = json.load(f)
                    except Exception:
                        continue
                    else:
                        # Skip files that are dataloader configs
                        if isinstance(data, list):
                            continue
                        if isinstance(data, dict) and "datasets" in data and isinstance(data["datasets"], list):
                            continue
                        if config_file.stem.startswith("multidatabackend"):
                            continue
                        if isinstance(data, dict) and "webhook_type" in data:
                            continue
                        if isinstance(data, dict) and "algo" in data:
                            continue

                        config_data = data.get("config", data) if isinstance(data, dict) else data
                        backend_path = _extract_backend_path(config_data)

                        metadata: Optional[Dict[str, Any]] = None
                        if isinstance(data, dict) and "_metadata" in data:
                            metadata = data["_metadata"].copy()
                            metadata["name"] = config_file.stem
                        else:
                            sidecar = self._load_metadata_sidecar(config_file)
                            if isinstance(sidecar, dict):
                                metadata = sidecar.copy()
                                metadata.setdefault("name", config_file.stem)
                        if metadata is None:
                            metadata = {
                                "name": config_file.stem,
                                "created_at": datetime.fromtimestamp(
                                    config_file.stat().st_ctime, tz=timezone.utc
                                ).isoformat(),
                                "modified_at": datetime.fromtimestamp(
                                    config_file.stat().st_mtime, tz=timezone.utc
                                ).isoformat(),
                            }
                            if isinstance(config_data, dict):
                                if "--model_family" in config_data:
                                    metadata["model_family"] = config_data["--model_family"]
                                elif "model_family" in config_data:
                                    metadata["model_family"] = config_data["model_family"]

                                if "--model_type" in config_data:
                                    metadata["model_type"] = config_data["--model_type"]
                                elif "model_type" in config_data:
                                    metadata["model_type"] = config_data["model_type"]

                                if "--model_flavour" in config_data:
                                    metadata["model_flavour"] = config_data["--model_flavour"]
                                elif "model_flavour" in config_data:
                                    metadata["model_flavour"] = config_data["model_flavour"]

                                if "--lora_type" in config_data:
                                    metadata["lora_type"] = config_data["--lora_type"]
                                elif "lora_type" in config_data:
                                    metadata["lora_type"] = config_data["lora_type"]

                        if backend_path:
                            metadata.setdefault("dataloader_path", backend_path)
                            metadata.setdefault("data_backend_config", backend_path)
                            metadata["has_dataloader"] = True
                        else:
                            metadata.setdefault("has_dataloader", False)

                        configs.append(metadata)
        elif self.config_type == "dataloader":
            # For dataloader configs, look for multidatabackend.json in the dedicated
            # dataloaders directory as well as alongside trainer configs.
            seen_paths: set[Path] = set()

            def _register_dataloader(config_file: Path, display_name: Optional[str] = None) -> None:
                try:
                    resolved = config_file.resolve(strict=False)
                except OSError:
                    resolved = config_file

                if resolved in seen_paths:
                    return

                try:
                    with config_file.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    return

                datasets: List[Any] = []
                if isinstance(data, list):
                    datasets = data
                elif isinstance(data, dict):
                    if "datasets" in data and isinstance(data["datasets"], list):
                        datasets = data["datasets"]
                    elif config_file.stem.startswith("multidatabackend"):
                        datasets = data if isinstance(data, list) else []
                    else:
                        return
                else:
                    return

                if isinstance(data, dict) and "_metadata" in data and not isinstance(data, list):
                    metadata = data["_metadata"].copy()
                    metadata["dataset_count"] = len(datasets)
                else:
                    metadata = {
                        "name": display_name or config_file.stem,
                        "created_at": datetime.fromtimestamp(config_file.stat().st_ctime, tz=timezone.utc).isoformat(),
                        "modified_at": datetime.fromtimestamp(config_file.stat().st_mtime, tz=timezone.utc).isoformat(),
                        "dataset_count": len(datasets),
                    }

                metadata.setdefault("name", display_name or config_file.stem)
                metadata.setdefault("path", str(config_file))
                configs.append(metadata)
                seen_paths.add(resolved)

            if self.config_dir.exists():
                for subdir in self.config_dir.iterdir():
                    if not subdir.is_dir():
                        continue
                    config_file = subdir / "multidatabackend.json"
                    if config_file.exists():
                        _register_dataloader(config_file, display_name=subdir.name)

                for config_file in self.config_dir.glob("*.json"):
                    _register_dataloader(config_file)

            env_root = self.config_dir.parent
            if env_root != self.config_dir and env_root.exists():
                for env_dir in env_root.iterdir():
                    if not env_dir.is_dir() or env_dir == self.config_dir:
                        continue
                    if env_dir.name.startswith("."):
                        continue

                    for pattern in ("multidatabackend.json", "multidatabackend" + "*.json"):
                        for config_file in env_dir.glob(pattern):
                            if not config_file.is_file():
                                continue
                            display_name = (
                                env_dir.name
                                if config_file.name == "multidatabackend.json"
                                else f"{env_dir.name}:{config_file.stem}"
                            )
                            _register_dataloader(config_file, display_name=display_name)
        elif self.config_type == "webhook":
            # For webhook configs, look for files with "webhook_type" key
            # Check subdirectories for any JSON files
            for subdir in self.config_dir.iterdir():
                if subdir.is_dir():
                    for config_file in subdir.glob("*.json"):
                        try:
                            with config_file.open("r", encoding="utf-8") as f:
                                data = json.load(f)

                            # Only include if it has webhook_type key
                            if isinstance(data, dict) and "webhook_type" in data:
                                metadata = {
                                    "name": f"{subdir.name}/{config_file.name}",
                                    "created_at": datetime.fromtimestamp(
                                        config_file.stat().st_ctime, tz=timezone.utc
                                    ).isoformat(),
                                    "modified_at": datetime.fromtimestamp(
                                        config_file.stat().st_mtime, tz=timezone.utc
                                    ).isoformat(),
                                    "webhook_type": data.get("webhook_type"),
                                    "callback_url": data.get("callback_url"),
                                }
                                configs.append(metadata)
                        except Exception:
                            # Skip invalid files
                            continue

            # Also check root-level JSON files
            for config_file in self.config_dir.glob("*.json"):
                try:
                    with config_file.open("r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Only include if it has webhook_type key
                    if isinstance(data, dict) and "webhook_type" in data:
                        metadata = {
                            "name": config_file.stem,
                            "created_at": datetime.fromtimestamp(config_file.stat().st_ctime, tz=timezone.utc).isoformat(),
                            "modified_at": datetime.fromtimestamp(config_file.stat().st_mtime, tz=timezone.utc).isoformat(),
                            "webhook_type": data.get("webhook_type"),
                            "callback_url": data.get("callback_url"),
                        }
                        configs.append(metadata)
                except Exception:
                    # Skip invalid files
                    continue
        elif self.config_type == "lycoris":
            # For lycoris configs, look for files with "algo" key
            # Check subdirectories for lycoris_config.json
            for subdir in self.config_dir.iterdir():
                if subdir.is_dir():
                    # Check for lycoris_config.json specifically
                    config_file = subdir / "lycoris_config.json"
                    if config_file.exists():
                        try:
                            with config_file.open("r", encoding="utf-8") as f:
                                data = json.load(f)

                            # Only include if it has algo key
                            if isinstance(data, dict) and "algo" in data:
                                metadata = {
                                    "name": subdir.name,
                                    "created_at": datetime.fromtimestamp(
                                        config_file.stat().st_ctime, tz=timezone.utc
                                    ).isoformat(),
                                    "modified_at": datetime.fromtimestamp(
                                        config_file.stat().st_mtime, tz=timezone.utc
                                    ).isoformat(),
                                    "algo": data.get("algo"),
                                    "factor": data.get("factor"),
                                    "multiplier": data.get("multiplier"),
                                }
                                configs.append(metadata)
                        except Exception:
                            # Skip invalid files
                            continue

                    # Also check other JSON files in subdirs
                    for other_config in subdir.glob("*.json"):
                        if other_config.name == "lycoris_config.json":
                            continue  # Already handled
                        try:
                            with other_config.open("r", encoding="utf-8") as f:
                                data = json.load(f)

                            # Only include if it has algo key
                            if isinstance(data, dict) and "algo" in data:
                                metadata = {
                                    "name": f"{subdir.name}/{other_config.name}",
                                    "created_at": datetime.fromtimestamp(
                                        other_config.stat().st_ctime, tz=timezone.utc
                                    ).isoformat(),
                                    "modified_at": datetime.fromtimestamp(
                                        other_config.stat().st_mtime, tz=timezone.utc
                                    ).isoformat(),
                                    "algo": data.get("algo"),
                                    "factor": data.get("factor"),
                                    "multiplier": data.get("multiplier"),
                                }
                                configs.append(metadata)
                        except Exception:
                            # Skip invalid files
                            continue

            # Also check root-level JSON files
            for config_file in self.config_dir.glob("*.json"):
                try:
                    with config_file.open("r", encoding="utf-8") as f:
                        data = json.load(f)

                    # Only include if it has algo key
                    if isinstance(data, dict) and "algo" in data:
                        metadata = {
                            "name": config_file.stem,
                            "created_at": datetime.fromtimestamp(config_file.stat().st_ctime, tz=timezone.utc).isoformat(),
                            "modified_at": datetime.fromtimestamp(config_file.stat().st_mtime, tz=timezone.utc).isoformat(),
                            "algo": data.get("algo"),
                            "factor": data.get("factor"),
                            "multiplier": data.get("multiplier"),
                        }
                        configs.append(metadata)
                except Exception:
                    # Skip invalid files
                    continue

        return sorted(configs, key=lambda x: x.get("name", ""))

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available templates.

        Returns:
            List of template metadata.
        """
        templates = []

        for template_file in self.template_dir.glob("**/*.json"):
            try:
                with template_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                if "_metadata" in data:
                    metadata = data["_metadata"]
                    metadata["is_template"] = True
                    templates.append(metadata)
            except Exception:
                continue

        return sorted(templates, key=lambda x: x.get("name", ""))

    def load_config(self, name: str) -> Tuple[Dict[str, Any], ConfigMetadata]:
        """Load a configuration by name.

        Args:
            name: Name of the configuration.

        Returns:
            Tuple of (config dict, metadata).

        Raises:
            FileNotFoundError: If config doesn't exist.
            ValueError: If config is invalid.
        """
        config_path = self._get_config_path(name)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration '{name}' not found")

        # Try cache first
        cached_data = self._cache.get(config_path)
        if cached_data:
            data = cached_data
        else:
            # Load from disk and cache
            try:
                with config_path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self._cache.set(config_path, data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in configuration '{name}': {e}")

        # Handle both new format (with metadata) and legacy format
        metadata_dict = data.get("_metadata") if isinstance(data, dict) else None
        if metadata_dict is not None:
            metadata = ConfigMetadata(**metadata_dict)
            if self.config_type == "dataloader":
                config = data.get("datasets", [])
            else:
                config_section = data.get("config") if isinstance(data, dict) else None
                if isinstance(config_section, dict):
                    config = config_section
                else:
                    # New flattened format stored config values alongside _metadata
                    config = {key: value for key, value in data.items() if key != "_metadata"}
        else:
            sidecar_metadata = self._load_metadata_sidecar(config_path)
            if isinstance(sidecar_metadata, dict):
                metadata = ConfigMetadata(**sidecar_metadata)
            else:
                metadata = ConfigMetadata(
                    name=name,
                    created_at=datetime.fromtimestamp(config_path.stat().st_ctime, tz=timezone.utc).isoformat(),
                    modified_at=datetime.fromtimestamp(config_path.stat().st_mtime, tz=timezone.utc).isoformat(),
                )
            config = data

            # Extract model info from legacy config for model configs
            if self.config_type == "model" and isinstance(config, dict):
                if "--model_family" in config:
                    metadata.model_family = config["--model_family"]
                elif "model_family" in config:
                    metadata.model_family = config["model_family"]

                if "--model_type" in config:
                    metadata.model_type = config["--model_type"]
                elif "model_type" in config:
                    metadata.model_type = config["model_type"]

                if "--model_flavour" in config:
                    metadata.model_flavour = config["--model_flavour"]
                elif "model_flavour" in config:
                    metadata.model_flavour = config["model_flavour"]

                if "--lora_type" in config:
                    metadata.lora_type = config["--lora_type"]
                elif "lora_type" in config:
                    metadata.lora_type = config["lora_type"]

        if self.config_type == "model" and isinstance(config, dict):
            if "data_backend_config" in config and "--data_backend_config" not in config:
                config["--data_backend_config"] = config["data_backend_config"]
            elif "--data_backend_config" in config and "data_backend_config" not in config:
                config["data_backend_config"] = config["--data_backend_config"]

        return config, metadata

    def save_config(
        self, name: str, config: Dict[str, Any], metadata: Optional[ConfigMetadata] = None, overwrite: bool = False
    ) -> ConfigMetadata:
        """Save a configuration.

        Args:
            name: Name for the configuration.
            config: Configuration dictionary.
            metadata: Optional metadata. Created if not provided.
            overwrite: Whether to overwrite existing config.

        Returns:
            Configuration metadata.

        Raises:
            FileExistsError: If config exists and overwrite is False.
        """
        config_path = self._get_config_path(name)

        if config_path.exists() and not overwrite:
            raise FileExistsError(f"Configuration '{name}' already exists")

        # Create or update metadata
        if metadata is None:
            metadata = self._create_metadata(name)
        else:
            metadata.modified_at = datetime.now(timezone.utc).isoformat()
            metadata.name = name

        if self.config_type == "dataloader":
            # Preserve original dataset payload shape (list or dict) without injecting metadata.
            data = config if config is not None else []
        else:
            combined: Dict[str, Any] = {}
            if isinstance(config, dict):
                nested = config.get("config") if isinstance(config.get("config"), dict) else None
                if nested:
                    for key, value in nested.items():
                        if key == "_metadata":
                            continue
                        if key == "--data_backend_config" and "data_backend_config" not in combined:
                            combined["data_backend_config"] = value
                            continue
                        combined[key] = value

                for key, value in config.items():
                    if key in {"_metadata", "config"}:
                        continue
                    if key == "--data_backend_config":
                        combined.setdefault("data_backend_config", value)
                        continue
                    combined[key] = value

            prepared_config = combined if combined else (config if isinstance(config, dict) else {})

            if (
                isinstance(config, dict)
                and "data_backend_config" not in prepared_config
                and "--data_backend_config" in config
            ):
                prepared_config["data_backend_config"] = config["--data_backend_config"]

            # Extract model info from config if not already present in metadata
            source_dict = prepared_config if isinstance(prepared_config, dict) else {}
            if not metadata.model_family:
                metadata.model_family = source_dict.get("--model_family") or source_dict.get("model_family")
            if not metadata.model_type:
                metadata.model_type = source_dict.get("--model_type") or source_dict.get("model_type")
            if not metadata.model_flavour:
                metadata.model_flavour = source_dict.get("--model_flavour") or source_dict.get("model_flavour")
            if not metadata.lora_type:
                metadata.lora_type = source_dict.get("--lora_type") or source_dict.get("lora_type")

            data = prepared_config if isinstance(prepared_config, dict) else {}

        # Create parent directory if it doesn't exist (for subdirectory configs)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with config_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")

        if self.config_type == "model":
            self._save_metadata_sidecar(config_path, metadata)
        else:
            self._delete_metadata_sidecar(config_path)

        try:
            self._cache.set(config_path, data)
        except Exception:
            pass

        return metadata

    def save_trainer_config(self, name: str, config: Dict[str, Any], overwrite: bool = False) -> None:
        """Save a trainer configuration without metadata wrapper.

        This is used for SimpleTuner trainer configs which expect a flat JSON structure.

        Args:
            name: Name for the configuration.
            config: Configuration dictionary.
            overwrite: Whether to overwrite existing config.

        Raises:
            FileExistsError: If config exists and overwrite is False.
        """
        config_path = self._get_config_path(name)

        if config_path.exists() and not overwrite:
            raise FileExistsError(f"Configuration '{name}' already exists")

        # Create parent directory if it doesn't exist (for subdirectory configs)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with config_path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
            f.write("\n")

    def delete_config(self, name: str) -> bool:
        """Delete a configuration.

        Args:
            name: Name of the configuration.

        Returns:
            True if deleted, False if not found.
        """
        config_path = self._get_config_path(name)
        deleted = False

        if self._is_folder_config(name):
            folder = config_path.parent
            if folder.exists():
                shutil.rmtree(folder)
                deleted = True
        elif config_path.exists():
            config_path.unlink()
            deleted = True

        if deleted:
            self._delete_metadata_sidecar(config_path)
            return True

        return False

    def rename_config(self, old_name: str, new_name: str) -> ConfigMetadata:
        """Rename a configuration.

        Args:
            old_name: Current name.
            new_name: New name.

        Returns:
            Updated metadata.

        Raises:
            FileNotFoundError: If old config doesn't exist.
            FileExistsError: If new name already exists.
        """
        # Check if old config exists
        old_path = self._get_config_path(old_name)
        if not old_path.exists():
            raise FileNotFoundError(f"Configuration '{old_name}' not found")

        # Load config before renaming
        config, metadata = self.load_config(old_name)

        # Determine if this is a folder-based config
        is_folder = self._is_folder_config(old_name)

        if is_folder:
            # For folder-based configs, rename the folder
            old_folder = self.config_dir / old_name
            new_folder = self.config_dir / new_name

            if new_folder.exists():
                raise FileExistsError(f"Configuration '{new_name}' already exists")

            # Rename the folder
            old_folder.rename(new_folder)

        else:
            # For file-based configs, rename the file (preserving .json extension)
            old_file = self.config_dir / f"{old_name}.json"
            new_file = self.config_dir / f"{new_name}.json"
            old_metadata_file = self._metadata_file_path(old_file)

            if new_file.exists():
                raise FileExistsError(f"Configuration '{new_name}' already exists")

            # Rename the file
            old_file.rename(new_file)
            if old_metadata_file.exists():
                try:
                    old_metadata_file.unlink()
                except Exception:
                    pass

        # Update metadata
        metadata.name = new_name
        metadata.modified_at = datetime.now(timezone.utc).isoformat()

        # Save updated metadata back to the file
        self.save_config(new_name, config, metadata, overwrite=True)

        return metadata

    def copy_config(self, source: str, target: str) -> ConfigMetadata:
        """Copy a configuration.

        Args:
            source: Source configuration name.
            target: Target configuration name.

        Returns:
            Metadata of the new configuration.

        Raises:
            FileNotFoundError: If source doesn't exist.
            FileExistsError: If target already exists.
        """
        source_path = self._get_config_path(source)
        target_path = self._get_config_path(target)

        if not source_path.exists():
            raise FileNotFoundError(f"Configuration '{source}' not found")

        if target_path.exists():
            raise FileExistsError(f"Configuration '{target}' already exists")

        # Load source and create new metadata
        config, old_metadata = self.load_config(source)
        new_metadata = self._create_metadata(target, f"Copy of {old_metadata.description or source}")
        new_metadata.model_family = old_metadata.model_family
        new_metadata.model_type = old_metadata.model_type
        new_metadata.tags = old_metadata.tags.copy()
        new_metadata.parent_template = old_metadata.parent_template

        # Save as new config
        self.save_config(target, config, new_metadata)

        return new_metadata

    def create_from_template(self, template_name: str, config_name: str) -> ConfigMetadata:
        """Create a configuration from a template.

        Args:
            template_name: Name of the template.
            config_name: Name for the new configuration.

        Returns:
            Metadata of the new configuration.

        Raises:
            FileNotFoundError: If template doesn't exist.
            FileExistsError: If config name already exists.
        """
        template_path = self._get_template_path(template_name)

        if not template_path.exists():
            # Check in subdirectories
            for template_file in self.template_dir.glob(f"**/{template_name}.json"):
                template_path = template_file
                break
            else:
                raise FileNotFoundError(f"Template '{template_name}' not found")

        # Load template
        with template_path.open("r", encoding="utf-8") as f:
            template_data = json.load(f)

        # Extract config and metadata
        if "_metadata" in template_data:
            template_metadata = template_data["_metadata"]
            config = template_data.get("config", {})
        else:
            template_metadata = {"name": template_name}
            config = template_data

        # Create new metadata
        metadata = self._create_metadata(
            config_name, f"Created from template: {template_metadata.get('name', template_name)}"
        )
        metadata.parent_template = template_name
        metadata.model_family = template_metadata.get("model_family")
        metadata.model_type = template_metadata.get("model_type")

        # Save new config
        self.save_config(config_name, config, metadata)

        return metadata

    def validate_config(self, config: Dict[str, Any]) -> ConfigValidation:
        """Validate a configuration.

        Args:
            config: Configuration dictionary or list for dataloader configs.

        Returns:
            Validation result.
        """
        validation = ConfigValidation(is_valid=True)

        if self.config_type == "dataloader":
            # Validate dataloader config (list of datasets)
            if not isinstance(config, list):
                validation.errors.append("Dataloader config must be a list of datasets")
                validation.is_valid = False
                return validation

            if not config:
                validation.errors.append("At least one dataset must be configured")
                validation.is_valid = False

            # Validate each dataset
            for idx, dataset in enumerate(config):
                if not isinstance(dataset, dict):
                    validation.errors.append(f"Dataset {idx} must be a dictionary")
                    validation.is_valid = False
                    continue

                if "id" not in dataset:
                    validation.errors.append(f"Dataset {idx} is missing required field 'id'")
                    validation.is_valid = False

                if "type" not in dataset:
                    validation.errors.append(f"Dataset {idx} is missing required field 'type'")
                    validation.is_valid = False
        else:
            # Check for empty config values that shouldn't be empty
            for key, value in config.items():
                if value == "" and key in ["--pretrained_model_name_or_path", "--output_dir"]:
                    validation.errors.append(f"Field '{key}' cannot be empty")
                    validation.is_valid = False

            # Model family validation
            model_family = config.get("--model_family")
            if model_family and _MODEL_FAMILY_SET and model_family not in _MODEL_FAMILY_SET:
                validation.warnings.append(f"Unknown model family: {model_family}")

            # Model type validation
            model_type = config.get("--model_type")
            if model_type and _MODEL_TYPE_SET and model_type not in _MODEL_TYPE_SET:
                validation.warnings.append(f"Unknown model type: {model_type}")

            # Model flavour validation
            model_flavour = config.get("--model_flavour")
            if model_flavour:
                valid_flavours = _get_flavour_choices_for_family(model_family)
                if not valid_flavours and _MODEL_FLAVOUR_SET:
                    valid_flavours = list(_MODEL_FLAVOUR_SET)
                valid_flavour_set = set(valid_flavours)
                if valid_flavour_set and model_flavour not in valid_flavour_set:
                    if model_family:
                        validation.warnings.append(f"Unknown model flavour '{model_flavour}' for family '{model_family}'")
                    else:
                        validation.warnings.append(f"Unknown model flavour: {model_flavour}")

            # LoRA specific validations
            if config.get("--model_type") == "lora":
                if "--lora_rank" not in config:
                    validation.suggestions.append("Consider setting '--lora_rank' for LoRA training")

            # Resolution validation
            if "--resolution" in config:
                res = config["--resolution"]
                if isinstance(res, (int, float)):
                    if res < 256:
                        validation.errors.append("Resolution must be at least 256")
                        validation.is_valid = False
                    elif res > 4096:
                        validation.warnings.append("Resolution above 4096 may cause memory issues")

        return validation

    def get_active_config(self) -> Optional[str]:
        """Get the currently active configuration name.

        Returns:
            Name of active config or None.
        """
        # Try to get from WebUI state first (most persistent)
        try:
            from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore

            state_store = WebUIStateStore()
            defaults = state_store.load_defaults()
            if defaults.active_config:
                return defaults.active_config
        except Exception:
            pass

        # Fall back to environment variable
        active_config = os.environ.get(_CONFIG_ACTIVE)
        if active_config:
            return active_config

        # If no active config is set, try to find the first available config
        configs = self.list_configs()
        if configs:
            return configs[0]["name"]

        return None

    def set_active_config(self, name: str) -> bool:
        """Set the active configuration.

        Args:
            name: Name of configuration to activate.

        Returns:
            True if successful.

        Raises:
            FileNotFoundError: If config doesn't exist.
        """
        config_path = self._get_config_path(name)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration '{name}' not found")

        # Save to WebUI state for persistence
        try:
            from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore

            state_store = WebUIStateStore()
            defaults = state_store.load_defaults()
            defaults.active_config = name
            state_store.save_defaults(defaults)
        except Exception:
            pass

        # Also set environment variable for current session
        os.environ[_CONFIG_ACTIVE] = name

        # Remove legacy config.json if it exists (no longer maintained)
        legacy_path = self.config_dir / "config.json"
        if legacy_path.exists() or legacy_path.is_symlink():
            try:
                legacy_path.unlink()
            except FileNotFoundError:
                pass

        return True

    def export_config(self, name: str, include_metadata: bool = True) -> Dict[str, Any]:
        """Export a configuration for sharing.

        Args:
            name: Name of configuration.
            include_metadata: Whether to include metadata.

        Returns:
            Configuration data.
        """
        config, metadata = self.load_config(name)

        if include_metadata:
            return {"_metadata": metadata.model_dump(), "config": config}

        return config

    def import_config(self, data: Dict[str, Any], name: Optional[str] = None, overwrite: bool = False) -> ConfigMetadata:
        """Import a configuration.

        Args:
            data: Configuration data.
            name: Optional name override.
            overwrite: Whether to overwrite existing.

        Returns:
            Configuration metadata.
        """
        # Extract config and metadata
        if "_metadata" in data:
            metadata = ConfigMetadata(**data["_metadata"])
            config = data.get("config", {})
        else:
            config = data
            metadata = None

        # Use provided name or metadata name
        if name:
            config_name = name
        elif metadata:
            config_name = metadata.name
        else:
            config_name = f"imported_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save configuration
        return self.save_config(config_name, config, metadata, overwrite=overwrite)
