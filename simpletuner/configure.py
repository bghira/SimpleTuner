#!/usr/bin/env python3
"""SimpleTuner configuration wizard driven by FieldRegistry metadata."""

from __future__ import annotations

import copy
import curses
import json
import os
import sys
import textwrap
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from fastapi import HTTPException

import simpletuner.helpers.models  # noqa: F401  # Ensure model registry population
from simpletuner.helpers.acceleration import AccelerationBackend, AccelerationPreset
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.simpletuner_sdk.server.services.webui_state import WebUIStateStore

try:
    import psutil

    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

try:  # Lazy import so the configurator still works without LyCORIS extras
    from lycoris.config_sdk import PresetValidationError
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    PresetValidationError = RuntimeError  # type: ignore[assignment]

try:
    from simpletuner.simpletuner_sdk.server.services.lycoris_builder_service import LYCORIS_BUILDER_SERVICE
except Exception as _lycoris_exc:  # pragma: no cover - optional dependency
    LYCORIS_BUILDER_SERVICE = None  # type: ignore[assignment]
    _LYCORIS_IMPORT_ERROR = str(_lycoris_exc)
else:  # pragma: no cover - passthrough for tests
    _LYCORIS_IMPORT_ERROR = ""

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

    from simpletuner.simpletuner_sdk.server.routes import webui_state as webui_routes
    from simpletuner.simpletuner_sdk.server.services.field_service import FieldService
    from simpletuner.simpletuner_sdk.server.services.tab_service import TabService


@dataclass
class TabEntry:
    """Metadata for a configuration tab."""

    name: str
    title: str
    description: str
    id: str


def _build_model_class_map() -> Dict[str, List[str]]:
    """Construct capability-aware model family listings used by legacy consumers."""

    families: List[tuple[str, str]] = []
    lora_supported = set()
    control_supported = set()

    for family, model_cls in ModelRegistry.model_families().items():
        display_name = getattr(model_cls, "NAME", family)
        families.append((family, display_name.lower()))

        try:
            if hasattr(model_cls, "supports_lora") and model_cls.supports_lora():
                lora_supported.add(family)
        except Exception:
            pass

        try:
            if hasattr(model_cls, "supports_controlnet") and model_cls.supports_controlnet():
                control_supported.add(family)
        except Exception:
            pass

    families.sort(key=lambda item: item[1])
    ordered = [family for family, _ in families]

    return {
        "full": ordered,
        "lora": [family for family in ordered if family in lora_supported],
        "controlnet": [family for family in ordered if family in control_supported],
    }


model_classes = _build_model_class_map()

default_models = {
    "flux": "black-forest-labs/FLUX.1-dev",
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "pixart_sigma": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    "kolors": "kwai-kolors/kolors-diffusers",
    "terminus": "ptx0/terminus-xl-velocity-v2",
    "sd3": "stabilityai/stable-diffusion-3.5-large",
    "sd2x": "stabilityai/stable-diffusion-2-1-base",
    "sd1x": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sana": "terminusresearch/sana-1.6b-1024px",
    "ltxvideo": "Lightricks/LTX-Video",
    "wan": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "hidream": "HiDream-ai/HiDream-I1-Full",
    "auraflow": "terminusresearch/auraflow-v0.3",
    "deepfloyd": "DeepFloyd/DeepFloyd-IF-I-XL-v1.0",
    "omnigen": "Shitao/OmniGen-v1-diffusers",
}

default_cfg = {
    "flux": 3.0,
    "sdxl": 4.2,
    "pixart_sigma": 3.4,
    "kolors": 5.0,
    "terminus": 8.0,
    "sd3": 5.0,
    "ltxvideo": 4.0,
    "hidream": 2.5,
    "wan": 4.0,
    "sana": 3.8,
    "omnigen": 3.2,
    "deepfloyd": 6.0,
    "sd2x": 7.0,
    "sd1x": 6.0,
}

model_labels = {
    "flux": "FLUX",
    "pixart_sigma": "PixArt Sigma",
    "kolors": "Kwai Kolors",
    "terminus": "Terminus",
    "sdxl": "Stable Diffusion XL",
    "sd3": "Stable Diffusion 3",
    "sd2x": "Stable Diffusion 2",
    "sd1x": "Stable Diffusion",
    "ltxvideo": "LTX Video",
    "wan": "WanX",
    "hidream": "HiDream I1",
    "sana": "Sana",
}

lora_ranks = [1, 16, 64, 128, 256]
learning_rates_by_rank = {
    1: "3e-4",
    16: "1e-4",
    64: "8e-5",
    128: "6e-5",
    256: "5.09e-5",
}

DEFAULT_LYCORIS_CONFIG_PATH = "config/lycoris_config.json"
_LYCORIS_LIST_FIELDS: Tuple[str, ...] = (
    "target_module",
    "target_name",
    "unet_target_module",
    "unet_target_name",
    "text_encoder_target_module",
    "text_encoder_target_name",
    "exclude_name",
)


def _split_csv(value: str) -> List[str]:
    if not value:
        return []
    tokens: List[str] = []
    for chunk in value.replace("\n", ",").split(","):
        cleaned = chunk.strip()
        if cleaned:
            tokens.append(cleaned)
    return tokens


def _normalize_override_value(value: str) -> Any:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if any(token in text for token in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


class LycorisBuilderSession:
    """Stateful helper for editing LyCORIS configurations via the CLI."""

    def __init__(self, service: Any, path: str | Path):
        self.service = service
        self.metadata = service.get_metadata()
        self.defaults: Dict[str, Any] = copy.deepcopy(self.metadata.get("defaults", {}))
        self.presets: List[Dict[str, Any]] = copy.deepcopy(self.metadata.get("presets", []))
        self.algorithms: List[Dict[str, Any]] = copy.deepcopy(self.metadata.get("algorithms", []))
        self.suggestions: Dict[str, List[str]] = copy.deepcopy(self.metadata.get("suggestions", {}))
        self.path = Path(path)
        self._last_loaded_path: Optional[str] = None
        self.config: Dict[str, Any] = self._initial_config()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _determine_default_algo(self) -> str:
        if "lora" in self.defaults:
            return "lora"
        if self.defaults:
            return next(iter(self.defaults.keys()))
        for algo in self.algorithms:
            name = algo.get("name")
            if isinstance(name, str) and name:
                return name
        return "lora"

    def _initial_config(self) -> Dict[str, Any]:
        algo = self._determine_default_algo()
        template = copy.deepcopy(self.defaults.get(algo)) or {}
        template.setdefault("algo", algo)
        template.setdefault("multiplier", 1.0)
        template.setdefault("linear_dim", 64)
        template.setdefault("linear_alpha", 32)
        return self._prepare_config(template)

    def _prepare_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        config = copy.deepcopy(payload) if isinstance(payload, dict) else {}
        config.setdefault("algo", self._determine_default_algo())
        apply_preset = config.get("apply_preset")
        if not isinstance(apply_preset, dict):
            apply_preset = {}
        config["apply_preset"] = copy.deepcopy(apply_preset)
        self._normalize_apply_preset(config["apply_preset"])
        return config

    def _normalize_apply_preset(self, preset: Dict[str, Any]) -> None:
        if not isinstance(preset, dict):
            return
        for field in _LYCORIS_LIST_FIELDS:
            values = preset.get(field)
            if values is None:
                continue
            if isinstance(values, (list, tuple, set)):
                cleaned = [str(item).strip() for item in values if str(item).strip()]
            else:
                cleaned = _split_csv(str(values))
            if cleaned:
                preset[field] = cleaned
            else:
                preset.pop(field, None)

        for map_key in ("module_algo_map", "name_algo_map"):
            mapping = preset.get(map_key)
            if not isinstance(mapping, dict):
                if map_key in preset:
                    preset[map_key] = {}
                continue
            cleaned_map: Dict[str, Dict[str, Any]] = {}
            for raw_key, raw_payload in mapping.items():
                key = (raw_key or "").strip()
                if not key:
                    continue
                payload = raw_payload if isinstance(raw_payload, dict) else {}
                cleaned_map[key] = copy.deepcopy(payload)
            preset[map_key] = cleaned_map

    def _apply_defaults(self, algo: str) -> None:
        template = copy.deepcopy(self.defaults.get(algo))
        if not template:
            return
        template.setdefault("algo", algo)
        self.config = self._prepare_config(template)

    def _preset_dict(self) -> Dict[str, Any]:
        preset = self.config.get("apply_preset")
        if not isinstance(preset, dict):
            preset = {}
            self.config["apply_preset"] = preset
        return preset

    def _override_key(self, scope: str) -> str:
        return "module_algo_map" if scope == "module" else "name_algo_map"

    def _override_map(self, scope: str, create: bool = True) -> Dict[str, Dict[str, Any]]:
        preset = self._preset_dict()
        map_key = self._override_key(scope)
        mapping = preset.get(map_key)
        if not isinstance(mapping, dict):
            if not create:
                return {}
            mapping = {}
            preset[map_key] = mapping
        return mapping

    def _clean_apply_preset(self, preset: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(preset, dict):
            return {}
        cleaned: Dict[str, Any] = {}
        for field in _LYCORIS_LIST_FIELDS:
            values = preset.get(field)
            if not values:
                continue
            if isinstance(values, (list, tuple, set)):
                normalized = [str(item).strip() for item in values if str(item).strip()]
            else:
                normalized = _split_csv(str(values))
            if normalized:
                cleaned[field] = normalized

        for map_key in ("module_algo_map", "name_algo_map"):
            mapping = preset.get(map_key)
            if not isinstance(mapping, dict):
                continue
            encoded: Dict[str, Dict[str, Any]] = {}
            for raw_key, raw_payload in mapping.items():
                key = (raw_key or "").strip()
                if not key or not isinstance(raw_payload, dict):
                    continue
                entry: Dict[str, Any] = {}
                algo = raw_payload.get("algo")
                if isinstance(algo, str) and algo.strip():
                    entry["algo"] = algo.strip()
                for opt_key, opt_value in raw_payload.items():
                    if opt_key == "algo":
                        continue
                    if opt_value in (None, ""):
                        continue
                    entry[opt_key] = opt_value
                if entry:
                    encoded[key] = entry
            if encoded:
                cleaned[map_key] = encoded

        for key, value in preset.items():
            if key in cleaned:
                continue
            if key in _LYCORIS_LIST_FIELDS:
                continue
            if key in {"module_algo_map", "name_algo_map"}:
                continue
            cleaned[key] = value
        return cleaned

    # ------------------------------------------------------------------
    # Public helpers used by the curses UI
    # ------------------------------------------------------------------
    def set_path(self, path: str | Path) -> None:
        self.path = Path(path)

    def get_path_display(self) -> str:
        return str(self.path)

    def load_from_file(self, path: str | Path | None = None) -> bool:
        target = Path(path) if path is not None else self.path
        if not target.exists():
            raise FileNotFoundError(f"LyCORIS config not found: {target}")
        with target.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("LyCORIS config must be a JSON object")
        self.config = self._prepare_config(payload)
        self.path = target
        self._last_loaded_path = str(target)
        return True

    def try_load_from_disk(self) -> bool:
        try:
            return self.load_from_file(self.path)
        except Exception:
            return False

    def to_serializable(self) -> Dict[str, Any]:
        payload = copy.deepcopy(self.config)
        preset = self._clean_apply_preset(payload.get("apply_preset") or {})
        if preset:
            payload["apply_preset"] = preset
        elif "apply_preset" in payload:
            payload.pop("apply_preset")
        return payload

    def validate(self) -> tuple[bool, Optional[str]]:
        algo = self.config.get("algo")
        if not isinstance(algo, str) or not algo.strip():
            return False, "Algorithm is required"
        preset = self._clean_apply_preset(self.config.get("apply_preset") or {})
        if preset:
            try:
                self.service.validate_preset(preset)
            except PresetValidationError as exc:  # type: ignore[misc]
                return False, str(exc)
        return True, None

    def save_to_file(self, path: str | Path | None = None) -> Path:
        valid, error = self.validate()
        if not valid:
            raise ValueError(error or "Invalid LyCORIS configuration")
        target = Path(path) if path is not None else self.path
        payload = self.to_serializable()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        self.path = target
        self._last_loaded_path = str(target)
        return target

    def get_algorithm_names(self) -> List[str]:
        names: List[str] = []
        for algo in self.algorithms:
            name = algo.get("name")
            if isinstance(name, str) and name:
                names.append(name)
        if not names:
            names.append(self._determine_default_algo())
        return names

    def get_algorithm_description(self, name: str) -> str:
        for algo in self.algorithms:
            if algo.get("name") == name:
                return algo.get("description") or ""
        return ""

    def set_algorithm(self, algorithm: str, *, reset: bool = False) -> None:
        normalized = algorithm.strip().lower()
        if not normalized:
            return
        self.config["algo"] = normalized
        if reset:
            self._apply_defaults(normalized)

    def set_numeric(self, field: str, value: Optional[float | int]) -> None:
        if value is None:
            self.config.pop(field, None)
            return
        self.config[field] = value

    def set_bool(self, field: str, value: Optional[bool]) -> None:
        if value is None:
            self.config.pop(field, None)
            return
        self.config[field] = bool(value)

    def get_list(self, field: str) -> List[str]:
        preset = self._preset_dict()
        values = preset.get(field)
        if isinstance(values, list):
            return [str(item) for item in values]
        return []

    def set_list(self, field: str, values: List[str]) -> None:
        preset = self._preset_dict()
        cleaned = [value.strip() for value in values if value.strip()]
        if cleaned:
            preset[field] = cleaned
        else:
            preset.pop(field, None)

    def get_override_entries(self, scope: str) -> Dict[str, Dict[str, Any]]:
        mapping = self._override_map(scope, create=False)
        return copy.deepcopy(mapping)

    def upsert_override(self, scope: str, key: str) -> Dict[str, Any]:
        key = key.strip()
        if not key:
            raise ValueError("Override key cannot be empty")
        mapping = self._override_map(scope, create=True)
        entry = mapping.get(key)
        if not isinstance(entry, dict):
            entry = {}
            mapping[key] = entry
        return entry

    def rename_override(self, scope: str, old_key: str, new_key: str) -> None:
        new_key = new_key.strip()
        if not new_key:
            return
        mapping = self._override_map(scope, create=True)
        payload = mapping.pop(old_key, None)
        if payload is None:
            return
        mapping[new_key] = payload

    def delete_override(self, scope: str, key: str) -> None:
        mapping = self._override_map(scope, create=True)
        mapping.pop(key, None)
        if not mapping:
            preset = self._preset_dict()
            preset.pop(self._override_key(scope), None)

    def set_override_algo(self, scope: str, key: str, algo: Optional[str]) -> None:
        entry = self.upsert_override(scope, key)
        if algo and algo.strip():
            entry["algo"] = algo.strip()
        else:
            entry.pop("algo", None)

    def set_override_option(self, scope: str, key: str, option: str, value: Any) -> None:
        entry = self.upsert_override(scope, key)
        if value in (None, ""):
            entry.pop(option, None)
        else:
            entry[option] = value

    def get_override_option(self, scope: str, key: str, option: str) -> Any:
        mapping = self._override_map(scope, create=False)
        entry = mapping.get(key) or {}
        return entry.get(option)

    def apply_preset(self, preset_name: str) -> bool:
        for preset in self.presets:
            if preset.get("name") == preset_name:
                config = copy.deepcopy(preset.get("config") or {})
                self.config["apply_preset"] = config
                self._normalize_apply_preset(self.config["apply_preset"])
                return True
        return False

    def get_preset_names(self) -> List[str]:
        return sorted([preset.get("name") for preset in self.presets if preset.get("name")])

    def get_summary(self) -> str:
        algo = self.config.get("algo", "?")
        multiplier = self.config.get("multiplier", "?")
        linear_dim = self.config.get("linear_dim", "?")
        return f"{algo} | mult={multiplier} dim={linear_dim}"

    def get_target_summary(self) -> str:
        preset = self._preset_dict()
        populated = [field for field in _LYCORIS_LIST_FIELDS if preset.get(field)]
        return f"{len(populated)} lists" if populated else "none"

    def get_override_summary(self, scope: str) -> str:
        mapping = self._override_map(scope, create=False)
        count = len(mapping)
        return f"{count} entr{'y' if count == 1 else 'ies'}" if count else "none"

    def last_loaded(self) -> Optional[str]:
        return self._last_loaded_path


class MemoryPresetsSession:
    """Stateful helper for selecting and applying memory optimization presets via the CLI."""

    # Backend labels for display
    BACKEND_LABELS = {
        "RAMTORCH": "RamTorch Streaming",
        "MUSUBI_BLOCK_SWAP": "Block Swap",
        "GROUP_OFFLOAD": "Group Offload",
        "DEEPSPEED_ZERO_1": "DeepSpeed ZeRO-1",
        "DEEPSPEED_ZERO_2": "DeepSpeed ZeRO-2",
        "DEEPSPEED_ZERO_3": "DeepSpeed ZeRO-3",
    }

    # These backends are mutually exclusive
    EXCLUSIVE_BACKENDS = {"RAMTORCH", "GROUP_OFFLOAD", "MUSUBI_BLOCK_SWAP"}

    def __init__(self, model_family: str):
        self.model_family = model_family
        self._model_cls = None
        self.presets: List[AccelerationPreset] = []
        self.max_swappable_blocks: Optional[int] = None
        self.unsupported_backends: set[str] = set()
        self.system_ram_gb: Optional[float] = None

        # Selection state
        self.selected_presets: Dict[str, str] = {}  # backend_name -> level
        self.custom_block_swap_count: int = 0

        self._load_presets()
        self._detect_system_ram()

    def _load_presets(self) -> None:
        """Load presets from the model class."""
        model_families = ModelRegistry.model_families()
        self._model_cls = model_families.get(self.model_family)
        if self._model_cls is None:
            return

        # Get presets from model
        if hasattr(self._model_cls, "get_acceleration_presets"):
            self.presets = self._model_cls.get_acceleration_presets()

        # Get max swappable blocks
        if hasattr(self._model_cls, "max_swappable_blocks"):
            self.max_swappable_blocks = self._model_cls.max_swappable_blocks()

        # Get unsupported backends
        unsupported = getattr(self._model_cls, "UNSUPPORTED_BACKENDS", set())
        self.unsupported_backends = {b.name if hasattr(b, "name") else str(b) for b in unsupported}

    def _detect_system_ram(self) -> None:
        """Detect system RAM if psutil is available."""
        if not _PSUTIL_AVAILABLE:
            return
        try:
            mem = psutil.virtual_memory()
            self.system_ram_gb = round(mem.total / (1024**3), 2)
        except Exception:
            pass

    @property
    def low_system_ram(self) -> bool:
        """True if system RAM is below 64GB."""
        return self.system_ram_gb is not None and self.system_ram_gb < 64

    @property
    def has_ram_intensive_selection(self) -> bool:
        """True if any selected preset requires 64GB+ RAM."""
        for backend, level in self.selected_presets.items():
            preset = self._find_preset(backend, level)
            if preset and preset.requires_min_system_ram_gb >= 64:
                return True
        return self.custom_block_swap_count > 0

    def _find_preset(self, backend: str, level: str) -> Optional[AccelerationPreset]:
        """Find a preset by backend name and level."""
        for preset in self.presets:
            if preset.backend.name == backend and preset.level == level:
                return preset
        return None

    def get_presets_for_tab(self, tab: str) -> List[AccelerationPreset]:
        """Get presets for a specific tab (basic, advanced)."""
        return [p for p in self.presets if p.tab == tab]

    def get_presets_grouped_by_backend(self, tab: str) -> Dict[str, List[AccelerationPreset]]:
        """Get presets grouped by backend for a tab."""
        groups: Dict[str, List[AccelerationPreset]] = {}
        for preset in self.get_presets_for_tab(tab):
            backend_name = preset.backend.name
            if backend_name not in groups:
                groups[backend_name] = []
            groups[backend_name].append(preset)

        # Sort presets within each group by level
        level_order = {"basic": 0, "light": 1, "conservative": 2, "balanced": 3, "aggressive": 4, "extreme": 5}
        for group in groups.values():
            group.sort(key=lambda p: level_order.get(p.level, 99))

        return groups

    def get_backend_label(self, backend: str) -> str:
        """Get human-readable label for a backend."""
        return self.BACKEND_LABELS.get(backend, backend.replace("_", " ").title())

    def is_preset_selected(self, preset: AccelerationPreset) -> bool:
        """Check if a preset is currently selected."""
        return self.selected_presets.get(preset.backend.name) == preset.level

    def toggle_preset(self, preset: AccelerationPreset) -> None:
        """Toggle a preset on/off, respecting mutual exclusivity."""
        backend_name = preset.backend.name

        if self.is_preset_selected(preset):
            # Deselect
            del self.selected_presets[backend_name]
        else:
            # If selecting an exclusive backend, deselect the others
            if backend_name in self.EXCLUSIVE_BACKENDS:
                for other in self.EXCLUSIVE_BACKENDS:
                    if other != backend_name and other in self.selected_presets:
                        del self.selected_presets[other]
                # Clear custom block swap when selecting RamTorch or Group Offload
                if backend_name in {"RAMTORCH", "GROUP_OFFLOAD"}:
                    self.custom_block_swap_count = 0
            # If preset has a group, deselect other presets in the same group
            if preset.group:
                for other_preset in self.presets:
                    if (
                        other_preset.group == preset.group
                        and other_preset.backend.name != backend_name
                        and other_preset.backend.name in self.selected_presets
                    ):
                        del self.selected_presets[other_preset.backend.name]
            # Select this preset
            self.selected_presets[backend_name] = preset.level

    def set_custom_block_swap(self, count: int) -> None:
        """Set a custom block swap count, clearing exclusive backends."""
        self.custom_block_swap_count = max(0, min(count, self.max_swappable_blocks or 0))
        if self.custom_block_swap_count > 0:
            # Clear RamTorch and Group Offload when using custom block swap
            self.selected_presets.pop("RAMTORCH", None)
            self.selected_presets.pop("GROUP_OFFLOAD", None)

    def get_selected_config(self) -> Dict[str, Any]:
        """Merge all selected presets' configs into one dictionary."""
        merged_config: Dict[str, Any] = {}

        for backend, level in self.selected_presets.items():
            preset = self._find_preset(backend, level)
            if preset and preset.config:
                merged_config.update(preset.config)

        # Apply custom block swap if set
        if self.custom_block_swap_count > 0:
            merged_config["musubi_blocks_to_swap"] = self.custom_block_swap_count
            # Clear RamTorch if using custom block swap
            merged_config.pop("ramtorch", None)
            merged_config.pop("ramtorch_target_modules", None)

        return merged_config

    @property
    def has_selection(self) -> bool:
        """True if any presets are selected or custom block swap is set."""
        return bool(self.selected_presets) or self.custom_block_swap_count > 0

    def get_summary(self) -> str:
        """Get a summary string of current selections."""
        if not self.has_selection:
            return "No presets selected"
        parts = []
        for backend, level in sorted(self.selected_presets.items()):
            parts.append(f"{self.get_backend_label(backend)}: {level}")
        if self.custom_block_swap_count > 0:
            parts.append(f"Block swap: {self.custom_block_swap_count} blocks")
        return ", ".join(parts)


class ConfigState:
    """Holds configuration values and interacts with the FieldRegistry."""

    def __init__(self, field_service: "FieldService"):
        self.field_service = field_service
        self.registry = field_service.field_registry
        self.field_defs = self._load_field_definitions()
        self.aliases = {name: self._compute_aliases(field) for name, field in self.field_defs.items()}
        self.values = self._initialize_defaults()
        self.loaded_config_path: Optional[str] = None
        self.webui_defaults: Dict[str, Any] = {}
        self.unknown_values: Dict[str, Any] = {}

    def _load_field_definitions(self) -> Dict[str, Any]:
        definitions: Dict[str, Any] = {}
        try:
            fields_iterable = self.registry.get_all_fields()
        except AttributeError:
            fields_iterable = getattr(self.registry, "_fields", {}).values()

        for field in fields_iterable or []:
            if not field:
                continue
            definitions[field.name] = field
        return definitions

    def _initialize_defaults(self) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {}
        for name, field in self.field_defs.items():
            if getattr(field, "webui_only", False):
                continue
            defaults[name] = field.default_value
        return defaults

    def _compute_aliases(self, field: Any) -> List[str]:
        aliases = set()
        aliases.add(field.name)
        arg_name = getattr(field, "arg_name", "")
        if arg_name:
            aliases.add(arg_name)
            cleaned = arg_name.lstrip("-")
            if cleaned:
                aliases.add(f"--{cleaned}")
        aliases.add(f"--{field.name}")
        for alias in getattr(field, "aliases", []) or []:
            if alias:
                aliases.add(alias)
        return list(aliases)

    def reset_to_defaults(self) -> None:
        """Reset configuration values to FieldRegistry defaults."""

        self.values = self._initialize_defaults()
        self.loaded_config_path = None
        self.unknown_values = {}

    def get_value(self, field_name: str) -> Any:
        """Return the current value for a field."""

        if field_name in self.values:
            return self.values[field_name]
        field = self.field_defs.get(field_name)
        if field and not getattr(field, "webui_only", False):
            return field.default_value
        return None

    def get_lycoris_config_path(self) -> str:
        path = self.get_value("lycoris_config")
        if isinstance(path, str) and path.strip():
            return path
        return DEFAULT_LYCORIS_CONFIG_PATH

    def set_value(self, field_name: str, value: Any) -> None:
        """Persist a value for a field."""

        if field_name not in self.field_defs:
            self.values[field_name] = value
            return

        field = self.field_defs[field_name]
        if getattr(field, "webui_only", False):
            return

        self.values[field_name] = value
        for alias in self.aliases.get(field_name, []):
            self.unknown_values.pop(alias, None)
        if field_name in {"model_family", "model_flavour"}:
            self._enforce_wan_i2v_dataset_requirements()

    def _get_config_scalar(self, *field_names: str) -> Any:
        for name in field_names:
            if name in self.values:
                value = self.values[name]
                if isinstance(value, str):
                    if value.strip():
                        return value
                    continue
                if value is not None:
                    return value
            if name in self.unknown_values:
                value = self.unknown_values[name]
                if isinstance(value, str):
                    if value.strip():
                        return value
                    continue
                if value is not None:
                    return value
        return None

    def _is_wan_i2v_model(self) -> bool:
        model_family = self._get_config_scalar("model_family", "--model_family")
        if not isinstance(model_family, str) or model_family.strip().lower() != "wan":
            return False
        model_flavour = self._get_config_scalar("model_flavour", "--model_flavour")
        if not isinstance(model_flavour, str):
            return False
        return model_flavour.strip().lower().startswith("i2v-")

    def _enforce_wan_i2v_dataset_requirements(self, payload: Optional[Any] = None) -> None:
        if not self._is_wan_i2v_model():
            return
        target = payload if payload is not None else self.unknown_values
        if isinstance(target, (dict, list)):
            self._force_video_entries_i2v(target)

    def _force_video_entries_i2v(self, node: Any) -> None:
        if isinstance(node, dict):
            dataset_type = node.get("dataset_type")
            dataset_type_str = dataset_type.strip().lower() if isinstance(dataset_type, str) else ""
            video_block = node.get("video")
            has_video_config = isinstance(video_block, dict)
            if dataset_type_str == "video" or has_video_config:
                if not isinstance(video_block, dict):
                    video_block = {}
                if video_block.get("is_i2v") is not True:
                    video_block["is_i2v"] = True
                node["video"] = video_block
            for value in node.values():
                if isinstance(value, (dict, list)):
                    self._force_video_entries_i2v(value)
        elif isinstance(node, list):
            for item in node:
                if isinstance(item, (dict, list)):
                    self._force_video_entries_i2v(item)

    def as_config_data(self) -> Dict[str, Any]:
        """Return configuration data, including aliases, for FieldService context."""

        self._enforce_wan_i2v_dataset_requirements()
        data: Dict[str, Any] = dict(self.unknown_values)
        for name, field in self.field_defs.items():
            if getattr(field, "webui_only", False):
                continue

            value = self.values.get(name, field.default_value)
            data[name] = value
            data[f"--{name}"] = value

            arg_name = getattr(field, "arg_name", "")
            if arg_name:
                data[arg_name] = value
                cleaned = arg_name.lstrip("-")
                if cleaned:
                    data.setdefault(f"--{cleaned}", value)

        self._enforce_wan_i2v_dataset_requirements(data)
        return data

    def to_serializable(self) -> Dict[str, Any]:
        """Return a serializable dictionary suitable for writing to disk."""

        self._enforce_wan_i2v_dataset_requirements()
        data = dict(self.unknown_values)
        for name, field in self.field_defs.items():
            if getattr(field, "webui_only", False):
                continue
            value = self.values.get(name, field.default_value)
            if value is None:
                continue
            data[name] = value
        self._enforce_wan_i2v_dataset_requirements(data)
        return data

    def load_from_file(self, config_path: str) -> bool:
        """Load configuration from a JSON file."""

        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return False

        if not isinstance(payload, dict):
            return False

        self.apply_config(payload)
        self.loaded_config_path = config_path
        return True

    def apply_config(self, payload: Dict[str, Any]) -> None:
        """Apply configuration values from a dictionary."""

        self.values = self._initialize_defaults()
        recognized: set[str] = set()

        for name, field in self.field_defs.items():
            if getattr(field, "webui_only", False):
                continue

            for alias in self.aliases.get(name, [name]):
                if alias in payload:
                    self.values[name] = payload[alias]
                    recognized.add(alias)

        self.unknown_values = {key: value for key, value in payload.items() if key not in recognized}
        self._enforce_wan_i2v_dataset_requirements()


class MenuNavigator:
    """Helper class for menu navigation."""

    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.h, self.w = stdscr.getmaxyx()
        self.last_selection = 0

    def show_menu(
        self,
        title: str,
        items: List[tuple[str, Any]],
        current_values: Optional[Dict[str, str]] = None,
        selected: int = 0,
    ) -> int:
        """Display a scrollable menu."""

        if not items:
            self.last_selection = 0
            return -2

        if selected < 0:
            selected = 0
        if selected >= len(items):
            selected = len(items) - 1
        self.last_selection = selected

        while True:
            self.stdscr.clear()

            self.stdscr.addstr(1, 2, title, curses.A_BOLD)
            self.stdscr.addstr(2, 2, "─" * (self.w - 4))
            self.stdscr.addstr(3, 2, "↑/↓: Navigate  Enter: Select  ←/Backspace: Back  q: Quit")
            self.stdscr.addstr(4, 2, "─" * (self.w - 4))

            start_y = 6
            for idx, (item_name, _) in enumerate(items):
                if start_y + idx >= self.h - 2:
                    break

                attr = curses.A_REVERSE if idx == selected else curses.A_NORMAL
                display_text = f"{idx + 1}. {item_name}"

                if current_values and item_name in current_values:
                    value_text = current_values[item_name]
                    max_value_len = self.w - len(display_text) - 10
                    if max_value_len > 0 and len(value_text) > max_value_len:
                        value_text = "..." + value_text[-(max_value_len - 3) :]
                    display_text += f" [{value_text}]"

                if len(display_text) > self.w - 4:
                    display_text = display_text[: self.w - 7] + "..."

                self.stdscr.addstr(start_y + idx, 4, display_text, attr)

            self.stdscr.refresh()

            key = self.stdscr.getch()
            if key == ord("q"):
                self.last_selection = selected
                return -1
            if key in [curses.KEY_LEFT, curses.KEY_BACKSPACE, 127, 8]:
                self.last_selection = selected
                return -2
            if key == curses.KEY_UP and selected > 0:
                selected -= 1
            elif key == curses.KEY_DOWN and selected < len(items) - 1:
                selected += 1
            elif key in [curses.KEY_ENTER, ord("\n"), ord("\r")]:
                self.last_selection = selected
                return selected
            elif ord("1") <= key <= ord("9"):
                num = key - ord("1")
                if num < len(items):
                    selected = num
                    self.last_selection = selected
                    return num


class SimpleTunerNCurses:
    """Interactive curses interface powered by FieldRegistry metadata."""

    def __init__(self):
        from fastapi.templating import Jinja2Templates

        from simpletuner.simpletuner_sdk.server.services.field_service import FieldService
        from simpletuner.simpletuner_sdk.server.services.tab_service import TabService

        templates_dir = Path(__file__).resolve().parent / "templates"
        self.templates = Jinja2Templates(directory=str(templates_dir))

        try:
            self.tab_service = TabService(self.templates)
        except Exception as exc:  # pragma: no cover - initialization guard
            raise RuntimeError(f"Failed to initialise TabService: {exc}") from exc

        try:
            self.field_service = FieldService()
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Missing optional dependency '{exc.name}'. Install project extras to use the configurator."
            ) from exc

        self.state = ConfigState(self.field_service)
        self.webui_store: Optional[WebUIStateStore] = None
        self._webui_store_error: Optional[str] = None
        self.lycoris_service = LYCORIS_BUILDER_SERVICE
        self._lycoris_support_error = _LYCORIS_IMPORT_ERROR
        self._lycoris_session: Optional[LycorisBuilderSession] = None
        self._lycoris_menu_index = 0
        self._memory_presets_session: Optional[MemoryPresetsSession] = None
        self._memory_presets_menu_index = 0
        self._webui_menu_index = 0
        self._refresh_webui_defaults()
        self.tab_entries = self._build_tab_entries()
        self.tab_lookup = {entry.name: entry for entry in self.tab_entries}
        self.menu_items = [
            (entry.title, partial(self.edit_tab, tab_name=entry.name), entry.description or "") for entry in self.tab_entries
        ]
        if self.lycoris_service is not None:
            self.menu_items.append(
                (
                    "LyCORIS Builder",
                    self.edit_lycoris_config,
                    "Interactively edit LyCORIS algorithm presets and overrides",
                )
            )
        self.menu_items.append(
            (
                "Memory Optimization Presets",
                self.edit_memory_presets,
                "Select and apply memory optimization presets for your model",
            )
        )
        self.menu_items.append(
            (
                "WebUI Setup",
                self.edit_webui_onboarding,
                "Update WebUI onboarding defaults and completion state",
            )
        )
        self.menu_items.append(("Review & Save", self.review_and_save, "Review the configuration and write it to disk"))
        self._menu_index = 0
        self._section_menu_indices: Dict[str, int] = {}
        self._field_menu_indices: Dict[Tuple[str, str], int] = {}

        default_config = Path("config/config.json")
        if default_config.exists():
            self.state.load_from_file(str(default_config))

    def _build_tab_entries(self) -> List[TabEntry]:
        entries: List[TabEntry] = []
        raw_tabs = self.tab_service.get_all_tabs()
        skip_tabs = {"checkpoints", "ui_settings"}

        for tab in raw_tabs:
            name = tab.get("name")
            if not name or name in skip_tabs:
                continue

            try:
                fields = self.field_service.field_registry.get_fields_for_tab(name)
            except Exception:
                fields = []

            fields = [field for field in fields if not getattr(field, "webui_only", False)]
            if not fields:
                continue

            entries.append(
                TabEntry(
                    name=name,
                    title=tab.get("title") or name.replace("_", " ").title(),
                    description=tab.get("description") or "",
                    id=tab.get("id") or name,
                )
            )

        return entries

    def run(self) -> None:
        """Launch the curses interface."""

        try:
            curses.wrapper(self._main_loop)
        except Exception as exc:
            print(f"Error: {exc}")
            traceback.print_exc()

    def _main_loop(self, stdscr) -> None:
        try:
            curses.curs_set(0)
        except curses.error:
            pass

        self.show_startup_screen(stdscr)

        while True:
            try:
                handler = self.show_main_menu(stdscr)
                if handler is None:
                    continue
                if handler == "quit":
                    break
                handler(stdscr)
            except KeyboardInterrupt:
                if self.confirm_quit(stdscr):
                    break

    def show_startup_screen(self, stdscr) -> None:
        """Display initial splash screen."""

        h, w = stdscr.getmaxyx()
        stdscr.clear()

        title_lines = [
            "╔═══════════════════════════════════════╗",
            "║      SimpleTuner Configuration        ║",
            "║       Registry-backed ncurses UI      ║",
            "╚═══════════════════════════════════════╝",
        ]

        start_y = (h - len(title_lines) - 6) // 2
        for idx, line in enumerate(title_lines):
            x = (w - len(line)) // 2
            stdscr.addstr(start_y + idx, x, line, curses.A_BOLD)

        info_y = start_y + len(title_lines) + 2

        if self.state.loaded_config_path:
            info = f"Loaded configuration: {self.state.loaded_config_path}"
            stdscr.addstr(info_y, (w - len(info)) // 2, info, curses.A_DIM)
            status = "Ready to modify existing configuration"
        else:
            status = "No configuration loaded - starting fresh"

        stdscr.addstr(info_y + 1, (w - len(status)) // 2, status)
        prompt = "Press any key to continue..."
        stdscr.addstr(h - 2, (w - len(prompt)) // 2, prompt, curses.A_DIM)
        stdscr.refresh()
        stdscr.getch()

    def show_main_menu(self, stdscr):
        """Render the main menu and return the selected handler."""

        if not self.menu_items:
            self.show_error(stdscr, "No tabs available from FieldRegistry.")
            return "quit"

        h, w = stdscr.getmaxyx()
        selected = min(self._menu_index, len(self.menu_items) - 1)
        max_visible = max(1, h - 7)
        scroll_offset = max(0, selected - max_visible + 1)

        while True:
            stdscr.clear()
            title = "SimpleTuner Configuration"
            stdscr.addstr(1, (w - len(title)) // 2, title, curses.A_BOLD)

            if self.state.loaded_config_path:
                info = f"Loaded: {self.state.loaded_config_path}"
                if len(info) > w - 4:
                    info = "..." + info[-(w - 7) :]
                stdscr.addstr(2, 2, info, curses.A_DIM)

            stdscr.addstr(3, 2, "↑/↓: Navigate  Enter: Select  'l': Load config  'q': Quit")

            visible_items = self.menu_items[scroll_offset : scroll_offset + max_visible]
            for idx, (item_name, _, _) in enumerate(visible_items):
                actual_idx = idx + scroll_offset
                y = 5 + idx
                attr = curses.A_REVERSE if actual_idx == selected else curses.A_NORMAL
                text = f"{actual_idx + 1}. {item_name}"
                if len(text) > w - 4:
                    text = text[: w - 7] + "..."
                stdscr.addstr(y, 2, text, attr)

            description = self.menu_items[selected][2]
            if description:
                desc_lines = textwrap.wrap(description, w - 4)
                if desc_lines:
                    stdscr.addstr(h - 2, 2, desc_lines[0][: w - 4], curses.A_DIM)

            if scroll_offset > 0:
                stdscr.addstr(4, w - 10, "▲ More", curses.A_DIM)
            if scroll_offset + max_visible < len(self.menu_items):
                stdscr.addstr(h - 3, w - 10, "▼ More", curses.A_DIM)

            stdscr.refresh()

            key = stdscr.getch()
            if key == ord("q"):
                if self.confirm_quit(stdscr):
                    return "quit"
            elif key == ord("l"):
                if self.load_config_dialog(stdscr):
                    selected = min(self._menu_index, len(self.menu_items) - 1)
                    scroll_offset = max(0, selected - max_visible + 1)
            elif key == curses.KEY_UP and selected > 0:
                selected -= 1
                if selected < scroll_offset:
                    scroll_offset = selected
            elif key == curses.KEY_DOWN and selected < len(self.menu_items) - 1:
                selected += 1
                if selected >= scroll_offset + max_visible:
                    scroll_offset = selected - max_visible + 1
            elif key in [curses.KEY_ENTER, ord("\n"), ord("\r")]:
                self._menu_index = selected
                return self.menu_items[selected][1]

    def show_error(self, stdscr, error_msg: str) -> None:
        """Display an error message."""

        h, w = stdscr.getmaxyx()
        error_lines = textwrap.wrap(error_msg, w - 10)
        error_h = len(error_lines) + 4
        error_w = min(80, w - 4)

        error_win = curses.newwin(error_h, error_w, (h - error_h) // 2, (w - error_w) // 2)
        error_win.box()
        error_win.addstr(0, 2, " Error ", curses.A_BOLD)

        for idx, line in enumerate(error_lines):
            error_win.addstr(idx + 1, 2, line)

        error_win.addstr(error_h - 2, 2, "Press any key to continue...")
        error_win.refresh()
        error_win.getch()

    def show_message(self, stdscr, message: str) -> None:
        """Display an informational message."""

        h, w = stdscr.getmaxyx()
        msg_lines = textwrap.wrap(message, w - 10)
        msg_h = len(msg_lines) + 4
        msg_w = min(80, w - 4)

        msg_win = curses.newwin(msg_h, msg_w, (h - msg_h) // 2, (w - msg_w) // 2)
        msg_win.box()
        msg_win.addstr(0, 2, " Info ", curses.A_BOLD)

        for idx, line in enumerate(msg_lines):
            msg_win.addstr(idx + 1, 2, line)

        msg_win.addstr(msg_h - 2, 2, "Press any key to continue...")
        msg_win.refresh()
        msg_win.getch()

    def get_input(
        self,
        stdscr,
        prompt: str,
        default: str = "",
        validation_fn=None,
        multiline: bool = False,
    ) -> str:
        """Prompt for user input."""

        h, w = stdscr.getmaxyx()
        stdscr.clear()
        wrapped_prompt = textwrap.wrap(prompt, w - 4)
        for idx, line in enumerate(wrapped_prompt):
            stdscr.addstr(2 + idx, 2, line)

        if default:
            stdscr.addstr(2 + len(wrapped_prompt) + 1, 2, f"Default: {default}")

        input_y = 2 + len(wrapped_prompt) + 3
        stdscr.addstr(input_y, 2, "> ")

        curses.echo()
        try:
            if multiline:
                user_input = stdscr.getstr(input_y, 4, w - 6).decode("utf-8")
            else:
                user_input = stdscr.getstr(input_y, 4, w - 6).decode("utf-8")
        finally:
            curses.noecho()

        if not user_input and default:
            user_input = default

        if validation_fn and not validation_fn(user_input):
            raise ValueError("Invalid input")

        return user_input

    def show_options(self, stdscr, prompt: str, options: List[str], default: int = 0) -> int:
        """Present a list of options and return the selected index."""

        stdscr.clear()
        h, w = stdscr.getmaxyx()

        wrapped_prompt = textwrap.wrap(prompt, w - 4)
        for idx, line in enumerate(wrapped_prompt):
            stdscr.addstr(2 + idx, 2, line)

        start_y = 2 + len(wrapped_prompt) + 2
        selected = default

        while True:
            for idx, option in enumerate(options):
                if start_y + idx >= h - 2:
                    break
                attr = curses.A_REVERSE if idx == selected else curses.A_NORMAL
                text = f"{idx + 1}. {option}"
                if len(text) > w - 4:
                    text = text[: w - 7] + "..."
                stdscr.addstr(start_y + idx, 4, text, attr)

            stdscr.refresh()
            key = stdscr.getch()
            if key == curses.KEY_UP and selected > 0:
                selected -= 1
            elif key == curses.KEY_DOWN and selected < len(options) - 1:
                selected += 1
            elif key in [curses.KEY_ENTER, ord("\n"), ord("\r")]:
                return selected
            elif key == 27:
                return -1

    def confirm_quit(self, stdscr) -> bool:
        """Confirm quitting the wizard."""

        return (
            self.show_options(
                stdscr,
                "Are you sure you want to quit? Unsaved changes will be lost.",
                ["No, continue", "Yes, quit"],
                0,
            )
            == 1
        )

    def find_config_files(self, base_path: str = "config") -> List[str]:
        """Return a list of discovered config.json files."""

        config_files: List[str] = []
        if os.path.exists(base_path):
            for root, _dirs, files in os.walk(base_path):
                if "config.json" in files:
                    config_files.append(os.path.join(root, "config.json"))
        return sorted(config_files)

    def load_config_dialog(self, stdscr) -> bool:
        """Allow the user to load or create a configuration file."""

        config_files = self.find_config_files()
        options = ["Create new configuration", "Enter path manually"] + config_files

        selected = self.show_options(
            stdscr,
            "Select a configuration to load:",
            options,
            2 if len(config_files) > 0 else 0,
        )

        if selected == -1:
            return False

        if selected == 0:
            self.state.reset_to_defaults()
            self.show_message(stdscr, "Started a new configuration.")
            return True
        if selected == 1:
            config_path = self.get_input(stdscr, "Enter path to config.json:", "config/config.json")
            if os.path.exists(config_path) and self.state.load_from_file(config_path):
                self.show_message(stdscr, f"Successfully loaded: {config_path}")
                return True
            self.show_error(stdscr, f"Failed to load: {config_path}")
            return False

        config_path = options[selected]
        if self.state.load_from_file(config_path):
            self.show_message(stdscr, f"Successfully loaded: {config_path}")
            return True

        self.show_error(stdscr, f"Failed to load: {config_path}")
        return False

    def _get_tab_structure(
        self,
        tab_name: str,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        config_data = self.state.as_config_data()
        tab_values = self.field_service.prepare_tab_field_values(tab_name, config_data, self.state.webui_defaults)
        fields, sections = self.field_service.build_template_tab(tab_name, tab_values, raw_config=config_data)
        return fields, sections, tab_values

    def edit_tab(self, stdscr, tab_name: str) -> None:
        """Display sections for the selected tab."""

        nav = MenuNavigator(stdscr)

        while True:
            fields, sections, _ = self._get_tab_structure(tab_name)
            section_fields = {
                section["id"]: [field for field in fields if field.get("section_id") == section["id"]]
                for section in sections
            }
            sections_with_fields = [section for section in sections if section_fields.get(section["id"])]

            if not sections_with_fields:
                self.show_message(stdscr, "No configurable fields in this tab with the current context.")
                return

            menu_items: List[tuple[str, Any]] = []
            current_values: Dict[str, str] = {}

            for section in sections_with_fields:
                title = section.get("title") or section["id"].replace("_", " ").title()
                count = len(section_fields[section["id"]])
                current_values[title] = f"{count} field{'s' if count != 1 else ''}"
                menu_items.append((title, partial(self.edit_section, tab_name=tab_name, section_id=section["id"])))

            last_selected = self._section_menu_indices.get(tab_name, 0)
            if menu_items:
                last_selected = max(0, min(last_selected, len(menu_items) - 1))
            else:
                last_selected = 0

            choice = nav.show_menu(self.tab_lookup[tab_name].title, menu_items, current_values, selected=last_selected)
            self._section_menu_indices[tab_name] = nav.last_selection

            if choice == -1:
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif choice == -2:
                return
            else:
                menu_items[choice][1](stdscr)

    def edit_section(self, stdscr, tab_name: str, section_id: str) -> None:
        """Display fields for a given section."""

        nav = MenuNavigator(stdscr)
        section_title = section_id.replace("_", " ").title()

        while True:
            fields, sections, _ = self._get_tab_structure(tab_name)
            section_fields = [field for field in fields if field.get("section_id") == section_id]

            if not section_fields:
                self.show_message(stdscr, "No fields available for this section with the current context.")
                return

            title_lookup = next((section.get("title") for section in sections if section["id"] == section_id), None)
            display_title = title_lookup or section_title

            menu_items: List[tuple[str, Any]] = []
            current_values: Dict[str, str] = {}

            for field in section_fields:
                label = field.get("label") or field["name"]
                if field.get("subsection"):
                    label = f"{label} [{field['subsection'].replace('_', ' ').title()}]"
                menu_items.append((label, partial(self.edit_field, tab_name=tab_name, field_name=field["name"])))
                current_values[label] = self._format_field_value(field)

            menu_key: Tuple[str, str] = (tab_name, section_id)
            last_selected = self._field_menu_indices.get(menu_key, 0)
            if menu_items:
                last_selected = max(0, min(last_selected, len(menu_items) - 1))
            else:
                last_selected = 0

            choice = nav.show_menu(display_title, menu_items, current_values, selected=last_selected)
            self._field_menu_indices[menu_key] = nav.last_selection

            if choice == -1:
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif choice == -2:
                return
            else:
                menu_items[choice][1](stdscr)

    def edit_field(self, stdscr, tab_name: str, field_name: str) -> None:
        """Prompt the user to edit a specific field."""

        fields, _sections, _ = self._get_tab_structure(tab_name)
        field_dict = next((field for field in fields if field["name"] == field_name), None)
        if not field_dict:
            self.show_error(stdscr, f"Field '{field_name}' is not available in the current context.")
            return

        if field_dict.get("disabled"):
            description = field_dict.get("description") or "This field is disabled based on current selections."
            self.show_message(stdscr, description)
            return

        field_def = self.state.field_defs.get(field_name)
        field_type = field_dict.get("type", "text")
        current_value = field_dict.get("value", self.state.get_value(field_name))

        prompt_parts = [field_dict.get("label", field_name)]
        if field_dict.get("description"):
            prompt_parts.append("")
            prompt_parts.append(field_dict["description"])
        prompt = "\n".join(prompt_parts)

        if field_type == "checkbox":
            new_value = self._prompt_checkbox(stdscr, prompt, current_value)
        elif field_type == "select":
            new_value = self._prompt_select(stdscr, prompt, field_dict, current_value)
        elif field_type == "multi_select":
            new_value = self._prompt_multi_select(stdscr, prompt, field_dict, current_value)
        elif field_type == "number":
            new_value = self._prompt_number(stdscr, prompt, field_def, current_value)
        elif field_type == "textarea":
            new_value = self._prompt_text(stdscr, prompt, current_value, multiline=True)
        else:
            new_value = self._prompt_text(stdscr, prompt, current_value)

        if new_value is None:
            return

        if self._validate_and_set_field(stdscr, field_name, new_value):
            self.show_message(stdscr, "Field updated.")

    # ------------------------------------------------------------------
    # WebUI onboarding support
    # ------------------------------------------------------------------

    def _get_webui_store(self) -> Optional[WebUIStateStore]:
        if self.webui_store is not None:
            return self.webui_store
        try:
            self.webui_store = WebUIStateStore()
        except Exception as exc:
            self._webui_store_error = str(exc)
            return None
        self._webui_store_error = None
        return self.webui_store

    def _refresh_webui_defaults(self) -> None:
        store = self._get_webui_store()
        if store is None:
            self.state.webui_defaults = {}
            return
        try:
            bundle = store.get_defaults_bundle()
        except ValueError as exc:
            self._webui_store_error = str(exc)
            self.state.webui_defaults = {}
            return
        self._webui_store_error = None
        resolved = bundle.get("resolved")
        self.state.webui_defaults = resolved if isinstance(resolved, dict) else {}

    @staticmethod
    def _get_webui_routes():
        from simpletuner.simpletuner_sdk.server.routes import webui_state as webui_routes

        return webui_routes

    def _summarize_onboarding_step(self, step: "webui_routes.OnboardingStepDefinition", stored: Any) -> str:
        if stored is None:
            return "Pending" if step.required else "Not set"
        is_complete = stored.completed_version >= step.version
        status = "Done" if is_complete else "Pending"
        if stored.value in (None, ""):
            return f"{status}: empty"
        if isinstance(stored.value, dict):
            summary = json.dumps(stored.value, sort_keys=True)
        else:
            summary = str(stored.value)
        return f"{status}: {summary}"

    def _prompt_onboarding_value(
        self,
        stdscr,
        step: "webui_routes.OnboardingStepDefinition",
        current_value: Optional[Any],
    ) -> Optional[Any]:
        if step.id == "default_datasets_dir" and current_value not in (None, ""):
            choice = self.show_options(
                stdscr,
                f"{step.title}\n{step.prompt}",
                ["Keep current value", "Update value", "Clear value"],
                0,
            )
            if choice == -1:
                return None
            if choice == 0:
                return current_value
            if choice == 2:
                return ""

        prompt = f"{step.title}\n{step.prompt}"
        if step.input_type == "accelerate_auto":
            prompt = f"{prompt}\nEnter JSON overrides (single line):"
            default_value = json.dumps(current_value, sort_keys=True) if current_value else ""
        else:
            default_value = "" if current_value in (None, "") else str(current_value)
        return self.get_input(stdscr, prompt, default_value)

    def edit_webui_onboarding(self, stdscr) -> None:
        store = self._get_webui_store()
        if store is None:
            error = self._webui_store_error or "WebUI state store unavailable."
            self.show_error(stdscr, error)
            return

        nav = MenuNavigator(stdscr)

        while True:
            try:
                onboarding = store.load_onboarding()
            except ValueError as exc:
                self.show_error(stdscr, f"Failed to load WebUI onboarding state: {exc}")
                return

            webui_routes = self._get_webui_routes()
            steps = webui_routes._resolve_step_definitions()
            if not steps:
                self.show_message(stdscr, "No onboarding steps available.")
                return

            menu_items: List[tuple[str, Any]] = []
            current_values: Dict[str, str] = {}

            for step in steps:
                menu_items.append((step.title, partial(self._edit_webui_onboarding_step, stdscr, step)))
                current_values[step.title] = self._summarize_onboarding_step(step, onboarding.steps.get(step.id))

            menu_items.append(("Back", None))
            choice = nav.show_menu("WebUI Onboarding", menu_items, current_values, selected=self._webui_menu_index)
            self._webui_menu_index = nav.last_selection

            if choice == -1:
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif choice == -2 or menu_items[choice][1] is None:
                return
            else:
                handler = menu_items[choice][1]
                if handler:
                    handler()

    def _edit_webui_onboarding_step(
        self,
        stdscr,
        step: "webui_routes.OnboardingStepDefinition",
    ) -> None:
        store = self._get_webui_store()
        if store is None:
            error = self._webui_store_error or "WebUI state store unavailable."
            self.show_error(stdscr, error)
            return

        try:
            onboarding = store.load_onboarding()
            defaults = store.load_defaults()
        except ValueError as exc:
            self.show_error(stdscr, f"Failed to load WebUI state: {exc}")
            return

        stored = onboarding.steps.get(step.id)
        current_value = stored.value if stored else None
        raw_value = self._prompt_onboarding_value(stdscr, step, current_value)
        if raw_value is None:
            return

        try:
            webui_routes = self._get_webui_routes()
            normalized = webui_routes._normalise_value(step, raw_value)
        except HTTPException as exc:
            self.show_error(stdscr, str(exc.detail))
            return

        if step.required and not normalized and step.id != "default_datasets_dir":
            self.show_error(stdscr, "A value is required to complete this step.")
            return

        store.record_onboarding_step(step.id, step.version, value=normalized)
        webui_routes = self._get_webui_routes()
        webui_routes._apply_step_to_defaults(defaults, step, normalized)
        store.save_defaults(defaults)
        self._refresh_webui_defaults()
        self.show_message(stdscr, f"Updated '{step.title}'.")

    # ------------------------------------------------------------------
    # Memory optimization presets support
    # ------------------------------------------------------------------

    def _get_memory_presets_session(self) -> Optional[MemoryPresetsSession]:
        """Get or create a memory presets session for the current model family."""
        model_family = self.state.get_value("model_family")
        if not model_family:
            return None

        if self._memory_presets_session is None or self._memory_presets_session.model_family != model_family:
            self._memory_presets_session = MemoryPresetsSession(model_family)

        return self._memory_presets_session

    def edit_memory_presets(self, stdscr) -> None:
        """Display the memory optimization presets menu."""
        session = self._get_memory_presets_session()
        if session is None:
            self.show_error(stdscr, "Please select a model family first (in Model Selection tab).")
            return

        if not session.presets:
            self.show_message(
                stdscr,
                f"No memory optimization presets available for model family '{session.model_family}'.",
            )
            return

        nav = MenuNavigator(stdscr)

        while True:
            # Show RAM warning if low RAM and RAM-intensive selection
            ram_warning = ""
            if session.low_system_ram and session.has_ram_intensive_selection:
                ram_warning = f"\n[WARNING: Low system RAM ({session.system_ram_gb}GB). Most strategies require 64GB+]"

            menu_items = [
                ("Basic Presets", partial(self._memory_presets_tab, stdscr, session, "basic")),
                ("Advanced Presets", partial(self._memory_presets_tab, stdscr, session, "advanced")),
            ]
            if session.max_swappable_blocks is not None:
                menu_items.append(("Custom Block Swap", partial(self._memory_presets_custom_block_swap, stdscr, session)))
            menu_items.append(("Show Preset Configuration", partial(self._memory_presets_preview, stdscr, session)))
            menu_items.append(("Apply to Configuration", partial(self._memory_presets_apply, stdscr, session)))
            menu_items.append(("Clear Selections", partial(self._memory_presets_clear, stdscr, session)))
            menu_items.append(("Back", None))

            current_values = {
                "Basic Presets": f"{len(session.get_presets_for_tab('basic'))} available",
                "Advanced Presets": f"{len(session.get_presets_for_tab('advanced'))} available",
                "Custom Block Swap": (
                    f"{session.custom_block_swap_count}/{session.max_swappable_blocks}"
                    if session.max_swappable_blocks
                    else "N/A"
                ),
                "Show Preset Configuration": f"{len(session.get_selected_config())} values",
                "Apply to Configuration": session.get_summary(),
                "Clear Selections": "",
            }

            title = f"Memory Optimization Presets ({session.model_family}){ram_warning}"
            choice = nav.show_menu(title, menu_items, current_values, selected=self._memory_presets_menu_index)
            self._memory_presets_menu_index = nav.last_selection

            if choice == -1:
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif choice == -2 or menu_items[choice][1] is None:
                return
            else:
                handler = menu_items[choice][1]
                if handler:
                    handler()

    def _memory_presets_tab(self, stdscr, session: MemoryPresetsSession, tab: str) -> None:
        """Display presets for a specific tab (basic/advanced)."""
        presets = session.get_presets_for_tab(tab)
        if not presets:
            self.show_message(stdscr, f"No {tab} presets available for this model.")
            return

        nav = MenuNavigator(stdscr)
        groups = session.get_presets_grouped_by_backend(tab)

        while True:
            menu_items: List[tuple[str, Any]] = []
            current_values: Dict[str, str] = {}

            for backend_name, backend_presets in groups.items():
                label = session.get_backend_label(backend_name)
                for preset in backend_presets:
                    item_label = f"{label}: {preset.level.capitalize()}"
                    menu_items.append((item_label, partial(self._memory_presets_toggle, stdscr, session, preset)))
                    if session.is_preset_selected(preset):
                        current_values[item_label] = "[SELECTED]"
                    else:
                        current_values[item_label] = preset.tradeoff_vram

            menu_items.append(("Back", None))

            title = f"{tab.capitalize()} Presets ({session.model_family})"
            if tab == "advanced":
                title += "\n[Advanced strategies require careful configuration]"

            choice = nav.show_menu(title, menu_items, current_values)
            if choice == -1:
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
            elif choice == -2 or menu_items[choice][1] is None:
                return
            else:
                menu_items[choice][1]()

    def _memory_presets_toggle(self, stdscr, session: MemoryPresetsSession, preset: AccelerationPreset) -> None:
        """Toggle a preset and show its info."""
        # Show preset details first
        info_lines = [
            f"Preset: {preset.name}",
            "",
            f"Description: {preset.description}",
            "",
            f"VRAM Impact: {preset.tradeoff_vram}",
            f"Speed Impact: {preset.tradeoff_speed}",
        ]
        if preset.tradeoff_notes:
            info_lines.append(f"Notes: {preset.tradeoff_notes}")
        if preset.requires_cuda:
            info_lines.append("Requires: CUDA/ROCm")
        if preset.requires_min_system_ram_gb > 0:
            info_lines.append(f"Requires: {preset.requires_min_system_ram_gb}GB+ system RAM")

        info_lines.append("")
        status = "Currently: SELECTED" if session.is_preset_selected(preset) else "Currently: Not selected"
        info_lines.append(status)

        choice = self.show_options(
            stdscr,
            "\n".join(info_lines),
            ["Toggle selection", "Cancel"],
            0,
        )
        if choice == 0:
            session.toggle_preset(preset)
            action = "Deselected" if not session.is_preset_selected(preset) else "Selected"
            self.show_message(stdscr, f"{action} '{preset.name}'")

    def _memory_presets_custom_block_swap(self, stdscr, session: MemoryPresetsSession) -> None:
        """Allow setting a custom block swap count."""
        if session.max_swappable_blocks is None:
            self.show_message(stdscr, "Block swap is not available for this model.")
            return

        current = session.custom_block_swap_count
        prompt = (
            f"Custom Block Swap Count\n\n"
            f"Maximum blocks: {session.max_swappable_blocks}\n"
            f"Current setting: {current}\n\n"
            f"Higher values save more VRAM but increase training time.\n"
            f"Enter 0 to disable custom block swap.\n\n"
            f"Enter value (0-{session.max_swappable_blocks}):"
        )

        response = self.get_input(stdscr, prompt, str(current))
        if not response.strip():
            return

        try:
            value = int(response.strip())
            if value < 0 or value > session.max_swappable_blocks:
                self.show_error(stdscr, f"Value must be between 0 and {session.max_swappable_blocks}")
                return
            session.set_custom_block_swap(value)
            if value > 0:
                self.show_message(stdscr, f"Set custom block swap to {value} blocks")
            else:
                self.show_message(stdscr, "Disabled custom block swap")
        except ValueError:
            self.show_error(stdscr, "Please enter a valid number")

    def _memory_presets_preview(self, stdscr, session: MemoryPresetsSession) -> None:
        """Show the merged configuration from selected presets as JSON."""
        config = session.get_selected_config()
        if not config:
            self.show_message(stdscr, "No presets selected. Select presets to see their configuration.")
            return

        self._display_json(stdscr, config)

    # Memory optimization fields that should be reset before applying new presets
    MEMORY_OPT_RESET_VALUES: Dict[str, Any] = {
        "ramtorch": False,
        "ramtorch_target_modules": "",
        "musubi_blocks_to_swap": 0,
        "enable_group_offload": False,
        "group_offload_type": "",
        "deepspeed_config": "",
    }

    def _memory_presets_apply(self, stdscr, session: MemoryPresetsSession) -> None:
        """Apply selected presets to the current configuration."""
        if not session.has_selection:
            self.show_error(stdscr, "No presets selected. Please select at least one preset.")
            return

        config = session.get_selected_config()
        if not config:
            self.show_error(stdscr, "No configuration values to apply.")
            return

        # Show what will be applied
        info_lines = ["The following settings will be applied:", ""]
        info_lines.append("First, these memory optimization fields will be reset:")
        for key in sorted(self.MEMORY_OPT_RESET_VALUES.keys()):
            info_lines.append(f"  {key}: {self.MEMORY_OPT_RESET_VALUES[key]}")
        info_lines.append("")
        info_lines.append("Then, these preset values will be applied:")
        for key, value in sorted(config.items()):
            info_lines.append(f"  {key}: {value}")
        info_lines.append("")
        info_lines.append("Apply these settings?")

        choice = self.show_options(stdscr, "\n".join(info_lines), ["Apply", "Cancel"], 0)
        if choice != 0:
            return

        # First, reset all memory optimization fields to defaults
        for key, value in self.MEMORY_OPT_RESET_VALUES.items():
            self.state.set_value(key, value)

        # Then apply the new preset config
        for key, value in config.items():
            self.state.set_value(key, value)

        count = len(config)
        self.show_message(stdscr, f"Applied {count} setting{'s' if count != 1 else ''} from memory presets.")

    def _memory_presets_clear(self, stdscr, session: MemoryPresetsSession) -> None:
        """Clear all selected presets."""
        if not session.has_selection:
            self.show_message(stdscr, "No presets are currently selected.")
            return

        choice = self.show_options(
            stdscr,
            f"Clear all selections?\n\nCurrent: {session.get_summary()}",
            ["Clear all", "Cancel"],
            1,
        )
        if choice == 0:
            session.selected_presets.clear()
            session.custom_block_swap_count = 0
            self.show_message(stdscr, "Cleared all preset selections.")

    # ------------------------------------------------------------------
    # LyCORIS builder support
    # ------------------------------------------------------------------

    def _get_lycoris_session(self) -> LycorisBuilderSession:
        if self.lycoris_service is None:
            raise RuntimeError("LyCORIS builder is unavailable in this environment")
        target_path = self.state.get_lycoris_config_path()
        if self._lycoris_session is None:
            self._lycoris_session = LycorisBuilderSession(self.lycoris_service, target_path)
            self._lycoris_session.try_load_from_disk()
        elif str(self._lycoris_session.path) != target_path:
            self._lycoris_session.set_path(target_path)
            self._lycoris_session.try_load_from_disk()
        return self._lycoris_session

    def edit_lycoris_config(self, stdscr) -> None:
        if self.lycoris_service is None:
            message = self._lycoris_support_error or "Install LyCORIS extras to enable this feature."
            self.show_error(stdscr, f"LyCORIS builder unavailable: {message}")
            return

        session = self._get_lycoris_session()
        nav = MenuNavigator(stdscr)

        while True:
            menu_items = [
                ("Change target file", partial(self._lycoris_change_path, stdscr, session)),
                ("Reload from disk", partial(self._lycoris_reload_config, stdscr, session)),
                ("Base settings", partial(self._lycoris_edit_base_settings, stdscr, session)),
                ("Advanced parameters", partial(self._lycoris_edit_advanced_settings, stdscr, session)),
                ("Target lists", partial(self._lycoris_edit_targets, stdscr, session)),
                (
                    "Module overrides",
                    partial(self._lycoris_manage_overrides, stdscr, session, "module"),
                ),
                (
                    "Name overrides",
                    partial(self._lycoris_manage_overrides, stdscr, session, "name"),
                ),
                ("Apply built-in preset", partial(self._lycoris_apply_preset, stdscr, session)),
                ("Preview JSON", partial(self._lycoris_preview_json, stdscr, session)),
                ("Save LyCORIS config", partial(self._lycoris_save_config, stdscr, session)),
                ("Back", None),
            ]

            current_values = {
                "Change target file": session.get_path_display(),
                "Reload from disk": session.last_loaded() or "not loaded",
                "Base settings": session.get_summary(),
                "Advanced parameters": ", ".join(
                    str(session.config.get(key))
                    for key in ("factor", "block_size", "constraint", "rescaled")
                    if session.config.get(key) not in (None, "")
                )
                or "none",
                "Target lists": session.get_target_summary(),
                "Module overrides": session.get_override_summary("module"),
                "Name overrides": session.get_override_summary("name"),
                "Apply built-in preset": f"{len(session.get_preset_names())} available",
                "Preview JSON": session.get_path_display(),
                "Save LyCORIS config": session.get_path_display(),
            }

            choice = nav.show_menu(
                "LyCORIS Builder",
                menu_items,
                current_values,
                selected=self._lycoris_menu_index,
            )
            self._lycoris_menu_index = nav.last_selection

            if choice == -1:
                if self.confirm_quit(stdscr):
                    raise KeyboardInterrupt
                continue
            if choice == -2 or menu_items[choice][1] is None:
                return

            handler = menu_items[choice][1]
            if handler:
                handler()

    def _lycoris_change_path(self, stdscr, session: LycorisBuilderSession) -> None:
        new_path = self.get_input(stdscr, "Enter path for LyCORIS config:", session.get_path_display())
        if not new_path:
            return
        session.set_path(new_path)
        self.state.set_value("lycoris_config", new_path)
        self.show_message(stdscr, f"LyCORIS config path set to {new_path}")

    def _lycoris_reload_config(self, stdscr, session: LycorisBuilderSession) -> None:
        try:
            session.load_from_file()
            self.show_message(stdscr, f"Reloaded LyCORIS config from {session.get_path_display()}")
        except Exception as exc:
            self.show_error(stdscr, f"Failed to load LyCORIS config: {exc}")

    def _lycoris_save_config(self, stdscr, session: LycorisBuilderSession) -> None:
        try:
            target = session.save_to_file()
            self.state.set_value("lycoris_config", str(target))
            self.show_message(stdscr, f"LyCORIS config written to {target}")
        except Exception as exc:
            self.show_error(stdscr, f"Failed to save LyCORIS config: {exc}")

    def _lycoris_preview_json(self, stdscr, session: LycorisBuilderSession) -> None:
        self._display_json(stdscr, session.to_serializable())

    # -- Base settings -----------------------------------------------------

    def _lycoris_edit_base_settings(self, stdscr, session: LycorisBuilderSession) -> None:
        nav = MenuNavigator(stdscr)
        options = [
            ("Algorithm", partial(self._lycoris_choose_algorithm, stdscr, session)),
            ("Multiplier", partial(self._lycoris_prompt_numeric, stdscr, session, "multiplier", True, 0.0)),
            ("Linear dimension", partial(self._lycoris_prompt_numeric, stdscr, session, "linear_dim", False, 1)),
            ("Linear alpha", partial(self._lycoris_prompt_numeric, stdscr, session, "linear_alpha", False, 0)),
            ("Back", None),
        ]

        while True:
            current_values = {
                "Algorithm": session.config.get("algo", "unknown"),
                "Multiplier": str(session.config.get("multiplier", "unset")),
                "Linear dimension": str(session.config.get("linear_dim", "unset")),
                "Linear alpha": str(session.config.get("linear_alpha", "unset")),
            }
            choice = nav.show_menu("LyCORIS Base Settings", options, current_values)
            if choice == -1 or choice == -2 or options[choice][1] is None:
                return
            options[choice][1]()

    def _lycoris_choose_algorithm(self, stdscr, session: LycorisBuilderSession) -> None:
        algorithms = session.get_algorithm_names()
        if not algorithms:
            self.show_error(stdscr, "No LyCORIS algorithms available.")
            return
        current = session.config.get("algo")
        default_idx = algorithms.index(current) if current in algorithms else 0
        selection = self.show_options(stdscr, "Select LyCORIS algorithm:", algorithms, default_idx)
        if selection == -1:
            return
        chosen = algorithms[selection]
        if chosen == current:
            return
        reset = self.show_options(
            stdscr,
            "Reset to algorithm defaults?",
            ["Yes", "No"],
            1,
        )
        session.set_algorithm(chosen, reset=reset == 0)

    def _lycoris_prompt_numeric(
        self,
        stdscr,
        session: LycorisBuilderSession,
        field: str,
        allow_float: bool,
        minimum: Optional[float],
    ) -> None:
        label = field.replace("_", " ").title()
        current_value = session.config.get(field)
        default = "" if current_value in (None, "") else str(current_value)
        while True:
            response = self.get_input(stdscr, f"Enter value for {label}:", default)
            if not response.strip():
                if minimum is None:
                    session.set_numeric(field, None)
                    return
                self.show_error(stdscr, f"{label} is required.")
                continue
            try:
                value = float(response) if allow_float else int(float(response))
            except ValueError:
                self.show_error(stdscr, "Invalid numeric value")
                continue
            if minimum is not None and value < minimum:
                self.show_error(stdscr, f"Value must be >= {minimum}")
                continue
            if not allow_float:
                value = int(value)
            session.set_numeric(field, value)
            return

    # -- Advanced settings -------------------------------------------------

    def _lycoris_edit_advanced_settings(self, stdscr, session: LycorisBuilderSession) -> None:
        nav = MenuNavigator(stdscr)
        options = [
            ("LoKr factor", partial(self._lycoris_prompt_numeric, stdscr, session, "factor", False, 1)),
            ("DyLoRA block size", partial(self._lycoris_prompt_numeric, stdscr, session, "block_size", False, 1)),
            ("Constraint", partial(self._lycoris_prompt_bool, stdscr, session, "constraint")),
            ("Rescaled", partial(self._lycoris_prompt_bool, stdscr, session, "rescaled")),
            ("Back", None),
        ]

        while True:
            current_values = {
                "LoKr factor": str(session.config.get("factor", "unset")),
                "DyLoRA block size": str(session.config.get("block_size", "unset")),
                "Constraint": str(session.config.get("constraint", "inherit")),
                "Rescaled": str(session.config.get("rescaled", "inherit")),
            }
            choice = nav.show_menu("LyCORIS Advanced", options, current_values)
            if choice == -1 or choice == -2 or options[choice][1] is None:
                return
            options[choice][1]()

    def _lycoris_prompt_bool(self, stdscr, session: LycorisBuilderSession, field: str) -> None:
        label = field.replace("_", " ").title()
        current = session.config.get(field)
        if current is True:
            default = 0
        elif current is False:
            default = 1
        else:
            default = 2
        choice = self.show_options(
            stdscr,
            f"Set {label}:",
            ["Enabled", "Disabled", "Inherit"],
            default,
        )
        if choice == -1:
            return
        if choice == 2:
            session.set_bool(field, None)
        else:
            session.set_bool(field, choice == 0)

    # -- Target lists ------------------------------------------------------

    def _lycoris_edit_targets(self, stdscr, session: LycorisBuilderSession) -> None:
        fields = [
            ("target_module", "Target modules"),
            ("target_name", "Target names"),
            ("unet_target_module", "UNet modules"),
            ("unet_target_name", "UNet names"),
            ("text_encoder_target_module", "Text encoder modules"),
            ("text_encoder_target_name", "Text encoder names"),
            ("exclude_name", "Exclude list"),
        ]
        nav = MenuNavigator(stdscr)
        while True:
            menu = [
                (label, partial(self._lycoris_edit_target_field, stdscr, session, field, label)) for field, label in fields
            ]
            menu.append(("Back", None))
            current_values = {label: ", ".join(session.get_list(field)) or "none" for field, label in fields}
            choice = nav.show_menu("LyCORIS Target Lists", menu, current_values)
            if choice == -1 or choice == -2 or menu[choice][1] is None:
                return
            menu[choice][1]()

    def _lycoris_edit_target_field(
        self,
        stdscr,
        session: LycorisBuilderSession,
        field: str,
        label: str,
    ) -> None:
        current = ", ".join(session.get_list(field))
        suggestions = session.suggestions.get(field, [])
        hint = f"\nSuggestions: {', '.join(suggestions[:6])}" if suggestions else ""
        prompt = f"Enter comma-separated values for {label}.{hint}\nLeave blank to clear."
        response = self.get_input(stdscr, prompt, current)
        session.set_list(field, _split_csv(response))

    # -- Overrides ---------------------------------------------------------

    def _lycoris_manage_overrides(self, stdscr, session: LycorisBuilderSession, scope: str) -> None:
        title = "Module Overrides" if scope == "module" else "Name Overrides"
        nav = MenuNavigator(stdscr)
        while True:
            overrides = session.get_override_entries(scope)
            menu_items = [
                (
                    f"{key} ({payload.get('algo', 'inherit')})",
                    partial(self._lycoris_edit_override, stdscr, session, scope, key),
                )
                for key, payload in sorted(overrides.items())
            ]
            menu_items.append(("Add override", partial(self._lycoris_add_override, stdscr, session, scope)))
            menu_items.append(("Back", None))
            choice = nav.show_menu(title, menu_items, None)
            if choice == -1 or choice == -2 or menu_items[choice][1] is None:
                return
            menu_items[choice][1]()

    def _lycoris_add_override(self, stdscr, session: LycorisBuilderSession, scope: str) -> None:
        key = self.get_input(stdscr, "Enter module/name to override:", "")
        if not key.strip():
            return
        try:
            session.upsert_override(scope, key.strip())
            self.show_message(stdscr, f"Added override '{key.strip()}'")
        except ValueError as exc:
            self.show_error(stdscr, str(exc))

    def _lycoris_edit_override(
        self,
        stdscr,
        session: LycorisBuilderSession,
        scope: str,
        key: str,
    ) -> None:
        nav = MenuNavigator(stdscr)
        options = [
            ("Rename", partial(self._lycoris_rename_override, stdscr, session, scope, key)),
            ("Set override algorithm", partial(self._lycoris_override_algo, stdscr, session, scope, key)),
            ("Edit options", partial(self._lycoris_edit_override_options, stdscr, session, scope, key)),
            ("Delete", partial(self._lycoris_delete_override, stdscr, session, scope, key)),
            ("Back", None),
        ]
        while True:
            choice = nav.show_menu(f"Override: {key}", options, None)
            if choice == -1 or choice == -2 or options[choice][1] is None:
                return
            options[choice][1]()

    def _lycoris_rename_override(
        self,
        stdscr,
        session: LycorisBuilderSession,
        scope: str,
        key: str,
    ) -> None:
        new_name = self.get_input(stdscr, "Enter new override name:", key)
        if not new_name.strip() or new_name.strip() == key:
            return
        session.rename_override(scope, key, new_name.strip())
        self.show_message(stdscr, f"Renamed override to '{new_name.strip()}'")

    def _lycoris_override_algo(
        self,
        stdscr,
        session: LycorisBuilderSession,
        scope: str,
        key: str,
    ) -> None:
        algorithms = session.get_algorithm_names()
        display = algorithms + ["Inherit (clear)"]
        current = session.get_override_option(scope, key, "algo")
        default = display.index(current) if isinstance(current, str) and current in display else len(display) - 1
        choice = self.show_options(stdscr, "Select override algorithm:", display, default)
        if choice == -1:
            return
        if choice == len(display) - 1:
            session.set_override_algo(scope, key, None)
        else:
            session.set_override_algo(scope, key, display[choice])

    def _lycoris_delete_override(
        self,
        stdscr,
        session: LycorisBuilderSession,
        scope: str,
        key: str,
    ) -> None:
        confirm = self.show_options(stdscr, f"Delete override '{key}'?", ["No", "Yes"], 0)
        if confirm == 1:
            session.delete_override(scope, key)
            self.show_message(stdscr, f"Deleted override '{key}'")

    def _lycoris_edit_override_options(
        self,
        stdscr,
        session: LycorisBuilderSession,
        scope: str,
        key: str,
    ) -> None:
        nav = MenuNavigator(stdscr)
        while True:
            entry = session.get_override_entries(scope).get(key, {})
            option_keys = [opt for opt in entry.keys() if opt != "algo"]
            menu_items = [
                (
                    f"{opt} = {entry.get(opt)}",
                    partial(self._lycoris_edit_single_option, stdscr, session, scope, key, opt),
                )
                for opt in sorted(option_keys)
            ]
            menu_items.append(("Add option", partial(self._lycoris_add_option, stdscr, session, scope, key)))
            menu_items.append(("Back", None))
            choice = nav.show_menu(f"Override options: {key}", menu_items, None)
            if choice == -1 or choice == -2 or menu_items[choice][1] is None:
                return
            menu_items[choice][1]()

    def _lycoris_add_option(
        self,
        stdscr,
        session: LycorisBuilderSession,
        scope: str,
        key: str,
    ) -> None:
        option = self.get_input(stdscr, "Enter option name:", "")
        if not option.strip():
            return
        raw_value = self.get_input(stdscr, f"Value for {option.strip()}:", "")
        value = _normalize_override_value(raw_value)
        session.set_override_option(scope, key, option.strip(), value)

    def _lycoris_edit_single_option(
        self,
        stdscr,
        session: LycorisBuilderSession,
        scope: str,
        key: str,
        option: str,
    ) -> None:
        current = session.get_override_option(scope, key, option)
        choice = self.show_options(
            stdscr,
            f"Option '{option}':",
            ["Rename", "Set value", "Delete", "Back"],
            1,
        )
        if choice == -1 or choice == 3:
            return
        if choice == 2:
            session.set_override_option(scope, key, option, None)
            self.show_message(stdscr, f"Removed option '{option}'")
            return
        if choice == 0:
            new_name = self.get_input(stdscr, "New option name:", option)
            if not new_name.strip() or new_name.strip() == option:
                return
            value = session.get_override_option(scope, key, option)
            session.set_override_option(scope, key, option, None)
            session.set_override_option(scope, key, new_name.strip(), value)
            self.show_message(stdscr, f"Renamed option to '{new_name.strip()}'")
            return
        raw_value = self.get_input(
            stdscr,
            f"Enter value for {option} (current: {current}):",
            str(current or ""),
        )
        value = _normalize_override_value(raw_value)
        session.set_override_option(scope, key, option, value)

    # -- Presets -----------------------------------------------------------

    def _lycoris_apply_preset(self, stdscr, session: LycorisBuilderSession) -> None:
        presets = session.get_preset_names()
        if not presets:
            self.show_message(stdscr, "No built-in presets available.")
            return
        choice = self.show_options(stdscr, "Select a preset to apply:", presets, 0)
        if choice == -1:
            return
        preset_name = presets[choice]
        confirm = self.show_options(
            stdscr,
            f"Apply preset '{preset_name}'? This replaces target lists and overrides.",
            ["Apply", "Cancel"],
            1,
        )
        if confirm == 0:
            if session.apply_preset(preset_name):
                self.show_message(stdscr, f"Applied preset '{preset_name}'")
            else:
                self.show_error(stdscr, f"Failed to apply preset '{preset_name}'")

    def _prompt_checkbox(self, stdscr, prompt: str, current_value: Any) -> Optional[bool]:
        options = ["Enabled", "Disabled"]
        default_idx = 0 if self._coerce_bool(current_value) else 1
        selected = self.show_options(stdscr, prompt, options, default_idx)
        if selected == -1:
            return None
        return selected == 0

    def _prompt_select(
        self,
        stdscr,
        prompt: str,
        field_dict: Dict[str, Any],
        current_value: Any,
    ) -> Optional[Any]:
        options = field_dict.get("options") or []
        if not options:
            text_default = "" if current_value in (None, "") else str(current_value)
            return self.get_input(stdscr, f"{prompt}\n\nEnter value:", text_default)

        display_options: List[str] = []
        values: List[Any] = []
        default_idx = 0

        for idx, option in enumerate(options):
            value = option.get("value")
            label = option.get("label") or str(value)
            display_options.append(label)
            values.append(value)
            if self._values_equal(value, current_value):
                default_idx = idx

        display_options.append("Enter custom value…")
        selected = self.show_options(stdscr, prompt, display_options, default_idx)
        if selected == -1:
            return None
        if selected == len(display_options) - 1:
            text_default = "" if current_value in (None, "") else str(current_value)
            custom_value = self.get_input(stdscr, "Enter custom value:", text_default)
            return custom_value if custom_value != "" else None
        return values[selected]

    def _prompt_multi_select(
        self,
        stdscr,
        prompt: str,
        field_dict: Dict[str, Any],
        current_value: Any,
    ) -> Optional[List[str]]:
        if isinstance(current_value, (list, tuple)):
            default_text = ", ".join(str(item) for item in current_value)
        elif current_value:
            default_text = str(current_value)
        else:
            default_text = ""

        options = field_dict.get("options") or []
        option_labels = ", ".join(str(option.get("value")) for option in options)
        extended_prompt = prompt
        if option_labels:
            extended_prompt = f"{prompt}\n\nAvailable options: {option_labels}"
        response = self.get_input(stdscr, extended_prompt, default_text)
        if response.strip() == "":
            return []
        return [item.strip() for item in response.split(",") if item.strip()]

    def _prompt_number(
        self,
        stdscr,
        prompt: str,
        field_def: Optional[Any],
        current_value: Any,
    ) -> Optional[Any]:
        default_value = field_def.default_value if field_def else None
        default_str = "" if current_value in (None, "") else str(current_value)
        response = self.get_input(stdscr, prompt, default_str)
        if response.strip() == "":
            return default_value

        try:
            if field_def and isinstance(field_def.default_value, float):
                return float(response)
            if "." in response.strip():
                return float(response)
            return int(response)
        except ValueError:
            self.show_error(stdscr, "Please enter a valid numeric value.")
            return None

    def _prompt_text(
        self,
        stdscr,
        prompt: str,
        current_value: Any,
        multiline: bool = False,
    ) -> Optional[str]:
        default_str = "" if current_value in (None, "None") else str(current_value)
        response = self.get_input(stdscr, prompt, default_str, multiline=multiline)
        return response

    def _validate_and_set_field(self, stdscr, field_name: str, value: Any) -> bool:
        context = self.state.as_config_data()
        context[field_name] = value
        context[f"--{field_name}"] = value

        field_def = self.state.field_defs.get(field_name)
        if field_def:
            arg_name = getattr(field_def, "arg_name", "")
            if arg_name:
                context[arg_name] = value

        try:
            errors = self.field_service.field_registry.validate_field_value(field_name, value, context)
        except Exception:
            errors = []

        if errors:
            self.show_error(stdscr, "\n".join(errors))
            return False

        self.state.set_value(field_name, value)
        return True

    def _format_field_value(self, field_dict: Dict[str, Any]) -> str:
        value = field_dict.get("value")
        field_type = field_dict.get("type", "text")

        if field_type == "checkbox":
            return "Enabled" if self._coerce_bool(value) else "Disabled"
        if field_type == "number":
            return str(value) if value not in (None, "") else "Not set"
        if field_type == "select":
            options = field_dict.get("options") or []
            label = next(
                (opt.get("label") for opt in options if self._values_equal(opt.get("value"), value)),
                None,
            )
            if label:
                return label
        if field_type == "multi_select":
            if not value:
                return "Not set"
            if isinstance(value, (list, tuple, set)):
                return ", ".join(str(item) for item in value)
        if value in (None, "", "Not configured"):
            return "Not set"
        return str(value)

    def review_and_save(self, stdscr) -> None:
        """Show current configuration and optionally write it to disk."""

        config_data = self.state.to_serializable()
        self._display_json(stdscr, config_data)
        choice = self.show_options(
            stdscr,
            "Choose an action:",
            ["Save to file", "Back"],
            0,
        )
        if choice == 0:
            self._save_config(stdscr, config_data)

    def _display_json(self, stdscr, data: Dict[str, Any]) -> None:
        """Render configuration data with simple scrolling."""

        text = json.dumps(data, indent=2, sort_keys=True)
        lines = text.splitlines()
        start = 0

        while True:
            stdscr.clear()
            h, w = stdscr.getmaxyx()
            title = "Review Configuration"
            stdscr.addstr(1, (w - len(title)) // 2, title, curses.A_BOLD)

            max_lines = h - 5
            y = 3
            displayed = 0
            idx = start

            while displayed < max_lines and idx < len(lines):
                wrapped = textwrap.wrap(lines[idx], w - 4) or [""]
                for chunk in wrapped:
                    if displayed >= max_lines:
                        break
                    stdscr.addstr(y + displayed, 2, chunk)
                    displayed += 1
                idx += 1

            stdscr.addstr(h - 2, 2, "↑/↓ scroll  Enter to continue", curses.A_DIM)
            stdscr.refresh()

            key = stdscr.getch()
            if key in (curses.KEY_ENTER, ord("\n"), ord("\r"), 27, ord("q")):
                return
            if key == curses.KEY_UP and start > 0:
                start -= 1
            elif key == curses.KEY_DOWN and start < max(0, len(lines) - 1):
                start += 1

    def _save_config(self, stdscr, config_data: Dict[str, Any]) -> None:
        """Persist configuration to disk."""

        default_path = self.state.loaded_config_path or "config/config.json"
        path = self.get_input(stdscr, "Enter file path to save config:", default_path)
        if not path:
            self.show_error(stdscr, "Path cannot be empty.")
            return

        try:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("w", encoding="utf-8") as handle:
                json.dump(config_data, handle, indent=4, sort_keys=True)
            self.state.loaded_config_path = str(target)
            self.show_message(stdscr, f"Configuration saved to {target}")
        except Exception as exc:
            self.show_error(stdscr, f"Failed to save configuration: {exc}")

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "on"}
        return False

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        if left == right:
            return True
        if isinstance(left, str) and isinstance(right, str):
            return left.strip() == right.strip()
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            return float(left) == float(right)
        return False


def main() -> None:
    """CLI entry point."""

    config_path: Optional[str] = None
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    try:
        configurator = SimpleTunerNCurses()
    except Exception as exc:
        print(f"Unable to start configurator: {exc}")
        traceback.print_exc()
        sys.exit(1)

    if config_path:
        if not os.path.exists(config_path):
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
        if configurator.state.load_from_file(config_path):
            print(f"Loaded configuration from: {config_path}")
        else:
            print(f"Error: Failed to load config: {config_path}")
            sys.exit(1)
    elif configurator.state.loaded_config_path:
        print(f"Loaded existing configuration from: {configurator.state.loaded_config_path}")
    else:
        print("No existing configuration found. Starting fresh setup.")

    configurator.run()


if __name__ == "__main__":
    main()
