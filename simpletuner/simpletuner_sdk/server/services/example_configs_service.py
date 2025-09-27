"""Helpers for working with bundled example configurations."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from simpletuner.simpletuner_sdk.server.utils.paths import get_simpletuner_root, resolve_config_path

try:  # pragma: no cover - optional dependency
    from simpletuner.cli import get_examples_dir as _cli_get_examples_dir
except Exception:  # pragma: no cover - fallback when CLI not available
    _cli_get_examples_dir = None


@dataclass
class ExampleConfigInfo:
    """Metadata container for a single example configuration."""

    name: str
    config_path: Path
    defaults: Dict[str, Any]
    description: Optional[str]
    dataloader_path: Optional[Path]
    dataloader_payload: Optional[Any]

    def to_public_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable view for API consumers."""

        def _format(path: Optional[Path]) -> Optional[str]:
            if not path:
                return None
            try:
                root = get_simpletuner_root()
                return path.resolve(strict=False).relative_to(root).as_posix()
            except Exception:
                try:
                    return path.resolve(strict=False).as_posix()
                except Exception:
                    return path.as_posix()

        return {
            "name": self.name,
            "defaults": self.defaults,
            "description": self.description,
            "config_path": _format(self.config_path),
            "dataloader_path": _format(self.dataloader_path),
            "has_dataloader": self.dataloader_payload is not None,
        }


class ExampleConfigsService:
    """Collect and hydrate example configuration bundles."""

    _ADJECTIVES = [
        "amber",
        "bold",
        "crimson",
        "dapper",
        "ember",
        "fuzzy",
        "golden",
        "harbor",
        "indigo",
        "jade",
        "krypton",
        "lilac",
        "mellow",
        "nebula",
        "opal",
        "plush",
        "quantum",
        "rustic",
        "stellar",
        "tidal",
        "umber",
        "velvet",
        "willow",
        "xenial",
        "young",
        "zephyr",
    ]

    _NOUNS = [
        "aurora",
        "banyan",
        "cascade",
        "dynamo",
        "ember",
        "flare",
        "grove",
        "harbor",
        "inkwell",
        "junction",
        "keystone",
        "lagoon",
        "mirage",
        "nebula",
        "oasis",
        "pavilion",
        "quartz",
        "rapids",
        "solstice",
        "timber",
        "uplink",
        "voyage",
        "whisper",
        "yonder",
        "zenith",
    ]

    def _examples_root(self) -> Path:
        if _cli_get_examples_dir:
            try:
                return Path(_cli_get_examples_dir())
            except Exception:  # pragma: no cover - fallback when CLI fails
                pass
        return get_simpletuner_root() / "examples"

    def list_examples(self) -> List[ExampleConfigInfo]:
        root = self._examples_root()
        if not root.exists():
            return []

        examples: List[ExampleConfigInfo] = []
        for item in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            info = self._build_example_info(item)
            if info is not None:
                examples.append(info)
        return examples

    def get_example(self, name: str) -> ExampleConfigInfo:
        for info in self.list_examples():
            if info.name == name:
                return info
        raise FileNotFoundError(f"Example '{name}' not found")

    def generate_project_name(self) -> str:
        return f"{random.choice(self._ADJECTIVES)}-{random.choice(self._NOUNS)}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_example_info(self, item: Path) -> Optional[ExampleConfigInfo]:
        config_path: Optional[Path] = None
        if item.is_dir():
            candidate = item / "config.json"
            if candidate.exists():
                config_path = candidate
        elif item.suffix == ".json":
            config_path = item

        if not config_path or not config_path.exists():
            return None

        try:
            with config_path.open("r", encoding="utf-8") as handle:
                config_data = json.load(handle)
        except Exception:
            return None

        if not isinstance(config_data, dict):
            # Skip dataloader-only JSON helpers or malformed configs
            return None

        defaults = self._extract_defaults(config_data)
        description = self._extract_description(config_data)
        dataloader_value = defaults.get("data_backend_config")

        dataloader_path: Optional[Path] = None
        dataloader_payload: Optional[Any] = None
        if dataloader_value:
            try:
                resolved = resolve_config_path(dataloader_value, check_cwd_first=False)
            except Exception:
                resolved = None
            if resolved and resolved.exists():
                dataloader_path = resolved
                try:
                    with resolved.open("r", encoding="utf-8") as handle:
                        dataloader_payload = json.load(handle)
                except Exception:
                    dataloader_payload = None

        return ExampleConfigInfo(
            name=item.stem if item.is_file() else item.name,
            config_path=config_path,
            defaults=defaults,
            description=description,
            dataloader_path=dataloader_path,
            dataloader_payload=dataloader_payload,
        )

    @staticmethod
    def _extract_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        def _get(*keys: str) -> Optional[Any]:
            for key in keys:
                if key in config:
                    return config[key]
            return None

        defaults: Dict[str, Any] = {}
        defaults["model_family"] = _get("--model_family", "model_family")
        defaults["model_type"] = _get("--model_type", "model_type")
        defaults["model_flavour"] = _get("--model_flavour", "model_flavour")
        defaults["lora_type"] = _get("--lora_type", "lora_type")
        defaults["data_backend_config"] = _get("--data_backend_config", "data_backend_config")
        return defaults

    @staticmethod
    def _extract_description(config: Dict[str, Any]) -> Optional[str]:
        if not isinstance(config, dict):
            return None

        metadata = config.get("_metadata")
        if isinstance(metadata, dict):
            description = metadata.get("description")
            if isinstance(description, str):
                return description
        return None


EXAMPLE_CONFIGS_SERVICE = ExampleConfigsService()
