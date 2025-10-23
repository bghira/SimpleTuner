#!/usr/bin/env python3
"""SimpleTuner configuration wizard driven by FieldRegistry metadata."""

from __future__ import annotations

import curses
import json
import os
import sys
import textwrap
import traceback
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import simpletuner.helpers.models  # noqa: F401  # Ensure model registry population
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.simpletuner_sdk.server.fields import Field

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

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
        if not getattr(model_cls, "ENABLED_IN_WIZARD", True):
            continue

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


class ConfigState:
    """Holds configuration values and interacts with the FieldRegistry."""

    def __init__(self, field_service: "FieldService"):
        self.field_service = field_service
        self.registry = field_service.field_registry
        self.field_defs: Dict[str, Field] = self._load_field_definitions()
        self.aliases: Dict[str, List[str]] = {name: self._compute_aliases(field) for name, field in self.field_defs.items()}
        self.values: Dict[str, Any] = self._initialize_defaults()
        self.loaded_config_path: Optional[str] = None
        self.webui_defaults: Dict[str, Any] = {}
        self.unknown_values: Dict[str, Any] = {}

    def _load_field_definitions(self) -> Dict[str, Field]:
        definitions: Dict[str, Field] = {}
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

    def _compute_aliases(self, field: Field) -> List[str]:
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
        field: Optional[Field] = self.field_defs.get(field_name)
        if field and not getattr(field, "webui_only", False):
            return field.default_value
        return None

    def set_value(self, field_name: str, value: Any) -> None:
        """Persist a value for a field."""

        if field_name not in self.field_defs:
            self.unknown_values[field_name] = value
            return

        field: Field = self.field_defs[field_name]
        if getattr(field, "webui_only", False):
            return

        self.values[field_name] = value
        for alias in self.aliases.get(field_name, []):
            self.unknown_values.pop(alias, None)

    def as_config_data(self) -> Dict[str, Any]:
        """Return configuration data, including aliases, for FieldService context."""

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

        return data

    def to_serializable(self) -> Dict[str, Any]:
        """Return a serializable dictionary suitable for writing to disk."""

        data = dict(self.unknown_values)
        for name, field in self.field_defs.items():
            if getattr(field, "webui_only", False):
                continue
            value = self.values.get(name, field.default_value)
            if value is None:
                continue
            data[name] = value
        return data

    def load_from_file(self, config_path: str) -> None:
        """Load configuration values from a JSON file."""
        self.reset_to_defaults()
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return

        self.loaded_config_path = config_path
        all_aliases = {alias: name for name, aliases in self.aliases.items() for alias in aliases}

        for key, value in data.items():
            field_name = all_aliases.get(key)
            if field_name:
                self.set_value(field_name, value)
            # Also check if the key is a canonical field name itself
            elif key in self.field_defs:
                self.set_value(key, value)
            else:
                self.unknown_values[key] = value