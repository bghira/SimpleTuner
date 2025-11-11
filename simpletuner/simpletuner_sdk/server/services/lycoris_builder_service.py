"""Service that exposes LyCORIS algorithm metadata and presets for the WebUI."""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set

from lycoris.config import list_builtin_presets
from lycoris.config_sdk import PresetConfig, PresetValidationError, list_algorithms

from simpletuner.lycoris_defaults import lycoris_defaults


class LycorisBuilderService:
    """Aggregates LyCORIS metadata (algos, presets, defaults) for consumption by the UI."""

    _SUGGESTION_KEYS: tuple[str, ...] = (
        "target_module",
        "target_name",
        "unet_target_module",
        "unet_target_name",
        "text_encoder_target_module",
        "text_encoder_target_name",
        "exclude_name",
    )

    def __init__(self) -> None:
        self._metadata_cache: Optional[Dict[str, Any]] = None

    def _serialize_algorithms(self) -> List[Dict[str, Any]]:
        specs = list_algorithms()
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "supported_args": list(spec.supported_args),
                "required_args": list(spec.required_args),
                "notes": spec.notes,
            }
            for spec in specs
        ]

    def _serialize_presets(self) -> List[Dict[str, Any]]:
        presets = list_builtin_presets()
        serialized: List[Dict[str, Any]] = []
        for name, preset in presets.items():
            preset_dict = preset.to_dict()
            serialized.append({"name": name, "config": preset_dict})
        serialized.sort(key=lambda item: item["name"])
        return serialized

    def _build_suggestions(self, presets: Iterable[Dict[str, Any]]) -> Dict[str, List[str]]:
        buckets: Dict[str, Set[str]] = {key: set() for key in self._SUGGESTION_KEYS}
        module_keys: Set[str] = set()
        name_keys: Set[str] = set()

        for preset in presets:
            config = preset.get("config") or {}
            for key in self._SUGGESTION_KEYS:
                entries = config.get(key) or []
                if isinstance(entries, list):
                    buckets[key].update(str(value) for value in entries if value)
            module_map = config.get("module_algo_map") or {}
            if isinstance(module_map, dict):
                module_keys.update(str(key) for key in module_map.keys() if key)
            name_map = config.get("name_algo_map") or {}
            if isinstance(name_map, dict):
                name_keys.update(str(key) for key in name_map.keys() if key)

        suggestions: Dict[str, List[str]] = {key: sorted(values) for key, values in buckets.items()}
        if module_keys:
            suggestions["module_override_keys"] = sorted(module_keys)
        if name_keys:
            suggestions["name_override_keys"] = sorted(name_keys)
        return suggestions

    def get_defaults(self) -> Dict[str, Any]:
        return copy.deepcopy(lycoris_defaults)

    def get_metadata(self, force_refresh: bool = False) -> Dict[str, Any]:
        if self._metadata_cache is not None and not force_refresh:
            return copy.deepcopy(self._metadata_cache)

        presets = self._serialize_presets()
        metadata = {
            "algorithms": self._serialize_algorithms(),
            "defaults": self.get_defaults(),
            "presets": presets,
            "suggestions": self._build_suggestions(presets),
        }
        self._metadata_cache = copy.deepcopy(metadata)
        return copy.deepcopy(self._metadata_cache)

    def validate_preset(self, payload: Mapping[str, Any]) -> None:
        """Validate an apply_preset payload using LyCORIS' schema."""
        if payload is None:
            return
        PresetConfig.from_dict(payload, strict=True)


LYCORIS_BUILDER_SERVICE = LycorisBuilderService()
