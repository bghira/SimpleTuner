"""Helpers for working with bundled example configurations."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
                raw_data = json.load(handle)
        except Exception:
            return None

        config_data, metadata = self._normalise_config_document(raw_data)
        if not config_data:
            # Skip dataloader-only JSON helpers or malformed configs
            return None

        defaults = self._extract_defaults(config_data)
        description = self._extract_description(config_data, metadata)
        dataloader_value = defaults.get("data_backend_config")

        dataloader_path: Optional[Path] = None
        dataloader_payload: Optional[Any] = None
        if dataloader_value:
            dataloader_path = self._resolve_dataloader_path(dataloader_value, config_path)
            if dataloader_path and dataloader_path.exists():
                try:
                    with dataloader_path.open("r", encoding="utf-8") as handle:
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

    def _resolve_dataloader_path(self, dataloader_value: Any, config_path: Path) -> Optional[Path]:
        if not isinstance(dataloader_value, str) or not dataloader_value.strip():
            return None

        try:
            resolved = resolve_config_path(dataloader_value, check_cwd_first=False)
        except Exception:
            resolved = None

        candidates: List[Path] = []
        if resolved:
            candidates.append(resolved)

        value_path = Path(dataloader_value.strip())

        if not value_path.is_absolute():
            candidates.append(config_path.parent / value_path)

        if value_path.name:
            candidates.append(config_path.parent / value_path.name)

        examples_root = self._examples_root()
        parts = [part for part in value_path.parts if part not in (".", "")]
        if parts:
            candidates.append(examples_root / Path(*parts))

            trimmed_parts = parts[:]
            while trimmed_parts and trimmed_parts[0].lower() in {"config", "configs"}:
                trimmed_parts = trimmed_parts[1:]
                if trimmed_parts:
                    candidates.append(examples_root / Path(*trimmed_parts))

            if trimmed_parts and trimmed_parts[0].lower() == "examples":
                trimmed_parts = trimmed_parts[1:]
                if trimmed_parts:
                    candidates.append(examples_root / Path(*trimmed_parts))

        if value_path.name:
            candidates.append(examples_root / value_path.name)

        seen: Set[str] = set()
        for candidate in candidates:
            if not candidate:
                continue
            try:
                resolved_candidate = candidate.resolve(strict=False)
            except Exception:
                resolved_candidate = candidate

            key = resolved_candidate.as_posix().lower()
            if key in seen:
                continue
            seen.add(key)

            try:
                if resolved_candidate.exists():
                    return resolved_candidate
            except Exception:
                continue

        return None

    @staticmethod
    def _extract_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(config, dict):
            return {}

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
    def _extract_description(config: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        if isinstance(config, list):
            return None
        if not isinstance(config, dict):
            return None

        def _clean(text: Optional[str]) -> Optional[str]:
            if isinstance(text, str):
                stripped = text.strip()
                return stripped or None
            return None

        if isinstance(metadata, dict):
            for key in ("description", "summary", "notes"):
                candidate = _clean(metadata.get(key))
                if candidate:
                    return candidate

        for key in (
            "project_description",
            "_project_description",
            "description",
            "--project_description",
        ):
            candidate = _clean(config.get(key))
            if candidate:
                return candidate

        return None

    @staticmethod
    def _normalise_config_document(raw: Any) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        if not isinstance(raw, dict):
            return {}, None

        metadata: Optional[Dict[str, Any]] = None
        if isinstance(raw.get("_metadata"), dict):
            metadata = raw["_metadata"]

        combined: Dict[str, Any] = {}

        # Prefer nested "config" payloads when present but allow top-level
        for source in (raw.get("config"), raw):
            if not isinstance(source, dict):
                continue
            for key, value in source.items():
                if key == "_metadata":
                    continue
                combined[key] = value

        return combined, metadata


EXAMPLE_CONFIGS_SERVICE = ExampleConfigsService()
