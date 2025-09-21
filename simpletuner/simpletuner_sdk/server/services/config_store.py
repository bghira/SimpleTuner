"""Configuration store for managing training configurations."""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

# Environment variables for configuration paths
_CONFIG_ENV_DIR = "SIMPLETUNER_CONFIG_DIR"
_CONFIG_ACTIVE = "SIMPLETUNER_ACTIVE_CONFIG"
_DEFAULT_CONFIG_DIR = Path("config/environments")
_DEFAULT_TEMPLATE_DIR = Path("config/templates")
_DEFAULT_CONFIG_FILE = Path("config/config.json")


class ConfigMetadata(BaseModel):
    """Metadata for a configuration."""
    name: str
    description: Optional[str] = None
    model_family: Optional[str] = None
    model_type: Optional[str] = None
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
    """Store for managing training configurations."""

    def __init__(self, config_dir: Optional[Path | str] = None):
        """Initialize the config store.

        Args:
            config_dir: Directory to store configurations. Defaults to config/environments.
        """
        if config_dir is not None:
            self.config_dir = Path(config_dir)
        else:
            self.config_dir = self._resolve_config_dir()

        self.template_dir = self._resolve_template_dir()
        self._ensure_directories()

    @staticmethod
    def _resolve_config_dir() -> Path:
        """Resolve the configuration directory from environment or default."""
        env_dir = os.environ.get(_CONFIG_ENV_DIR)
        if env_dir:
            return Path(env_dir)
        return _DEFAULT_CONFIG_DIR

    @staticmethod
    def _resolve_template_dir() -> Path:
        """Resolve the template directory."""
        return _DEFAULT_TEMPLATE_DIR

    def _ensure_directories(self):
        """Ensure configuration directories exist."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.template_dir.mkdir(parents=True, exist_ok=True)

        # Create default config if it doesn't exist
        default_config = self.config_dir / "default.json"
        if not default_config.exists():
            self._create_default_config(default_config)

    def _create_default_config(self, path: Path):
        """Create a default configuration file."""
        default_config = {
            "--model_type": "lora",
            "--model_family": "flux",
            "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
            "--output_dir": "output/models",
            "--train_batch_size": 1,
            "--learning_rate": 0.0001,
            "--max_train_steps": 1000,
            "--checkpointing_steps": 100,
            "--seed": 42,
            "--resolution": 1024,
            "--mixed_precision": "bf16",
            "--gradient_checkpointing": True,
            "--use_ema": False
        }

        metadata = self._create_metadata("default", "Default training configuration")
        config_with_metadata = {
            "_metadata": metadata.model_dump(),
            "config": default_config
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(config_with_metadata, f, indent=2, sort_keys=True)
            f.write("\n")

    def _create_metadata(self, name: str, description: str = None) -> ConfigMetadata:
        """Create metadata for a configuration."""
        now = datetime.now(timezone.utc).isoformat()
        return ConfigMetadata(
            name=name,
            description=description,
            created_at=now,
            modified_at=now
        )

    def _get_config_path(self, name: str) -> Path:
        """Get the path for a configuration file."""
        return self.config_dir / f"{name}.json"

    def _get_template_path(self, name: str) -> Path:
        """Get the path for a template file."""
        return self.template_dir / f"{name}.json"

    def list_configs(self) -> List[Dict[str, Any]]:
        """List all available configurations.

        Returns:
            List of configuration metadata.
        """
        configs = []

        # List user configs
        for config_file in self.config_dir.glob("*.json"):
            try:
                with config_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)

                if "_metadata" in data:
                    metadata = data["_metadata"]
                else:
                    # Legacy config without metadata
                    metadata = {
                        "name": config_file.stem,
                        "created_at": datetime.fromtimestamp(
                            config_file.stat().st_ctime, tz=timezone.utc
                        ).isoformat(),
                        "modified_at": datetime.fromtimestamp(
                            config_file.stat().st_mtime, tz=timezone.utc
                        ).isoformat()
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

        try:
            with config_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration '{name}': {e}")

        # Handle both new format (with metadata) and legacy format
        if "_metadata" in data:
            metadata = ConfigMetadata(**data["_metadata"])
            config = data.get("config", {})
        else:
            # Legacy format - create metadata on the fly
            metadata = ConfigMetadata(
                name=name,
                created_at=datetime.fromtimestamp(
                    config_path.stat().st_ctime, tz=timezone.utc
                ).isoformat(),
                modified_at=datetime.fromtimestamp(
                    config_path.stat().st_mtime, tz=timezone.utc
                ).isoformat()
            )
            config = data

        return config, metadata

    def save_config(
        self,
        name: str,
        config: Dict[str, Any],
        metadata: Optional[ConfigMetadata] = None,
        overwrite: bool = False
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

        # Extract model info from config if not in metadata
        if not metadata.model_family and "--model_family" in config:
            metadata.model_family = config["--model_family"]
        if not metadata.model_type and "--model_type" in config:
            metadata.model_type = config["--model_type"]

        # Save with metadata
        data = {
            "_metadata": metadata.model_dump(),
            "config": config
        }

        with config_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")

        return metadata

    def delete_config(self, name: str) -> bool:
        """Delete a configuration.

        Args:
            name: Name of the configuration.

        Returns:
            True if deleted, False if not found.
        """
        config_path = self._get_config_path(name)

        if config_path.exists():
            config_path.unlink()
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
        old_path = self._get_config_path(old_name)
        new_path = self._get_config_path(new_name)

        if not old_path.exists():
            raise FileNotFoundError(f"Configuration '{old_name}' not found")

        if new_path.exists():
            raise FileExistsError(f"Configuration '{new_name}' already exists")

        # Load, update metadata, and save
        config, metadata = self.load_config(old_name)
        metadata.name = new_name
        metadata.modified_at = datetime.now(timezone.utc).isoformat()

        # Move file
        old_path.rename(new_path)

        # Update metadata in file
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
        new_metadata = self._create_metadata(
            target,
            f"Copy of {old_metadata.description or source}"
        )
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
            config_name,
            f"Created from template: {template_metadata.get('name', template_name)}"
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
            config: Configuration dictionary.

        Returns:
            Validation result.
        """
        validation = ConfigValidation(is_valid=True)

        # Required fields
        required_fields = [
            "--model_type",
            "--pretrained_model_name_or_path",
            "--output_dir"
        ]

        for field in required_fields:
            if field not in config:
                validation.errors.append(f"Required field '{field}' is missing")
                validation.is_valid = False

        # Model family validation
        if "--model_family" in config:
            valid_families = ["flux", "sd3", "sdxl", "sd", "pixart", "ltxvideo"]
            if config["--model_family"] not in valid_families:
                validation.warnings.append(
                    f"Unknown model family: {config['--model_family']}"
                )

        # Model type validation
        if "--model_type" in config:
            valid_types = ["lora", "full", "controlnet", "embedding"]
            if config["--model_type"] not in valid_types:
                validation.warnings.append(
                    f"Unknown model type: {config['--model_type']}"
                )

        # LoRA specific validations
        if config.get("--model_type") == "lora":
            if "--lora_rank" not in config:
                validation.suggestions.append(
                    "Consider setting '--lora_rank' for LoRA training"
                )

        # Resolution validation
        if "--resolution" in config:
            res = config["--resolution"]
            if isinstance(res, (int, float)):
                if res < 256:
                    validation.errors.append(
                        "Resolution must be at least 256"
                    )
                    validation.is_valid = False
                elif res > 4096:
                    validation.warnings.append(
                        "Resolution above 4096 may cause memory issues"
                    )

        return validation

    def get_active_config(self) -> Optional[str]:
        """Get the currently active configuration name.

        Returns:
            Name of active config or None.
        """
        active_config = os.environ.get(_CONFIG_ACTIVE)
        if active_config:
            return active_config

        # Check if default config symlink exists
        if _DEFAULT_CONFIG_FILE.is_symlink():
            target = _DEFAULT_CONFIG_FILE.readlink()
            if target.parent == self.config_dir:
                return target.stem

        return "default"

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

        # Set environment variable
        os.environ[_CONFIG_ACTIVE] = name

        # Update symlink or copy
        if _DEFAULT_CONFIG_FILE.exists():
            if _DEFAULT_CONFIG_FILE.is_symlink():
                _DEFAULT_CONFIG_FILE.unlink()
            else:
                # Backup existing config
                backup_path = _DEFAULT_CONFIG_FILE.with_suffix(".json.bak")
                shutil.copy2(_DEFAULT_CONFIG_FILE, backup_path)
                _DEFAULT_CONFIG_FILE.unlink()

        # Create symlink to active config (or copy on Windows)
        try:
            _DEFAULT_CONFIG_FILE.symlink_to(config_path)
        except OSError:
            # Fallback to copying on systems that don't support symlinks
            shutil.copy2(config_path, _DEFAULT_CONFIG_FILE)

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
            return {
                "_metadata": metadata.model_dump(),
                "config": config
            }

        return config

    def import_config(
        self,
        data: Dict[str, Any],
        name: Optional[str] = None,
        overwrite: bool = False
    ) -> ConfigMetadata:
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