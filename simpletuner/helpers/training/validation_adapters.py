from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple

DEFAULT_LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
VALID_ADAPTER_MODES = {"adapter_only", "comparison", "none"}


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower())
    slug = slug.strip("_")
    return slug or "adapter"


def _stem_from_path(path_value: str) -> str:
    basename = os.path.basename(path_value.rstrip("/"))
    stem, _ = os.path.splitext(basename)
    return stem or basename or "adapter"


@dataclass(frozen=True)
class ValidationAdapterSpec:
    """Represents a single adapter load instruction for validation sampling."""

    is_local: bool
    location: str
    weight_name: str | None
    strength: float
    adapter_name: str | None = None

    @property
    def repo_id(self) -> str | None:
        return None if self.is_local else self.location

    @property
    def path(self) -> str | None:
        return self.location if self.is_local else None


@dataclass(frozen=True)
class ValidationAdapterRun:
    """Represents one validation pass that may enable zero or more adapters."""

    label: str | None
    slug: str
    adapters: Tuple[ValidationAdapterSpec, ...]
    is_base: bool = False

    @classmethod
    def base(cls) -> "ValidationAdapterRun":
        return cls(label=None, slug="", adapters=tuple(), is_base=True)


def _extract_repo_and_weight(raw_value: str) -> Tuple[str, str]:
    repo_id = raw_value
    weight_name = DEFAULT_LORA_WEIGHT_NAME
    if ":" in raw_value:
        repo_id, weight_name = raw_value.split(":", 1)
    return repo_id.strip(), weight_name.strip() or DEFAULT_LORA_WEIGHT_NAME


def _build_adapter_spec(raw_value: str, strength: float, adapter_name: str | None = None) -> ValidationAdapterSpec:
    if raw_value is None:
        raise ValueError("Adapter path cannot be None.")
    cleaned = str(raw_value).strip()
    if cleaned == "":
        raise ValueError("Adapter path cannot be empty.")
    if adapter_name is not None:
        adapter_name = adapter_name.strip() or None

    expanded = os.path.abspath(os.path.expanduser(cleaned))
    path_exists = os.path.exists(expanded)
    drive, _ = os.path.splitdrive(expanded)
    if path_exists or drive:
        return ValidationAdapterSpec(
            is_local=True,
            location=expanded,
            weight_name=None,
            strength=float(strength),
            adapter_name=adapter_name,
        )

    repo_id, weight_name = _extract_repo_and_weight(cleaned)
    return ValidationAdapterSpec(
        is_local=False,
        location=repo_id,
        weight_name=weight_name,
        strength=float(strength),
        adapter_name=adapter_name,
    )


def _norm_strength(value: Any, default: float = 1.0) -> float:
    if value is None:
        return float(default)
    try:
        scale = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid adapter scale value: {value}") from None
    if scale <= 0:
        raise ValueError("Adapter scale must be greater than zero.")
    return scale


def _ensure_list(entry: Any, *, field_name: str) -> List[Any]:
    if entry is None:
        return []
    if isinstance(entry, list):
        return entry
    raise ValueError(f"Expected a list for '{field_name}', got {type(entry).__name__}.")


def _normalize_run_entry(entry: Any) -> Tuple[str | None, List[ValidationAdapterSpec]]:
    if isinstance(entry, str):
        spec = _build_adapter_spec(entry, 1.0)
        return _stem_from_path(entry), [spec]

    if not isinstance(entry, dict):
        raise ValueError(f"Invalid adapter config entry: {entry!r}")

    label = entry.get("label") or entry.get("name")
    base_strength = _norm_strength(entry.get("strength", entry.get("scale")), 1.0)
    base_adapter_name = entry.get("adapter_name")

    if "path" in entry and "paths" not in entry and "adapters" not in entry:
        spec = _build_adapter_spec(
            entry["path"], entry.get("strength", entry.get("scale", base_strength)), base_adapter_name
        )
        return label or _stem_from_path(entry["path"]), [spec]

    adapter_entries = entry.get("adapters")
    if adapter_entries is None:
        adapter_entries = entry.get("paths")
    adapter_entries = _ensure_list(adapter_entries, field_name="paths")

    specs: List[ValidationAdapterSpec] = []
    for adapter in adapter_entries:
        if isinstance(adapter, str):
            specs.append(_build_adapter_spec(adapter, base_strength, base_adapter_name))
            continue
        if not isinstance(adapter, dict):
            raise ValueError(f"Invalid adapter specification: {adapter!r}")
        path_value = adapter.get("path")
        if path_value is None:
            raise ValueError(f"Adapter specification is missing 'path': {adapter!r}")
        adapter_strength = _norm_strength(adapter.get("strength", adapter.get("scale")), base_strength)
        adapter_name = adapter.get("adapter_name") or base_adapter_name
        specs.append(_build_adapter_spec(path_value, adapter_strength, adapter_name))

    if not specs:
        raise ValueError("Adapter run must include at least one adapter path.")

    if label is None and specs:
        first = specs[0]
        label = _stem_from_path(first.path or first.repo_id or "adapter")

    return label, specs


def _iter_config_entries(config: Any) -> Iterable[Any]:
    if config is None:
        return []
    if isinstance(config, dict):
        if "runs" in config:
            return config["runs"]
        return [config]
    if isinstance(config, list):
        return config
    raise ValueError("validation_adapter_config must be a list or a dict containing 'runs'.")


def build_validation_adapter_runs(
    adapter_path: str | None,
    adapter_config: Any,
    *,
    adapter_name: str | None = None,
    adapter_strength: float = 1.0,
    adapter_mode: str | None = None,
) -> List[ValidationAdapterRun]:
    """
    Build adapter run definitions from CLI inputs.
    """

    mode = (adapter_mode or "adapter_only").strip().lower()
    if mode not in VALID_ADAPTER_MODES:
        raise ValueError(f"Invalid adapter mode '{adapter_mode}'. Expected one of {sorted(VALID_ADAPTER_MODES)}")

    runs: List[ValidationAdapterRun] = []
    seen_slugs: set[str] = set()

    def _make_run(label: str | None, specs: Sequence[ValidationAdapterSpec]) -> ValidationAdapterRun:
        primary_spec = specs[0]
        if primary_spec.is_local:
            fallback_label = _stem_from_path(primary_spec.location)
        else:
            fallback_label = _stem_from_path(primary_spec.repo_id or "adapter")
        display_label = label or primary_spec.adapter_name or fallback_label
        slug = _slugify(display_label)
        original_slug = slug
        counter = 2
        while slug in seen_slugs:
            slug = f"{original_slug}_{counter}"
            counter += 1
        seen_slugs.add(slug)
        return ValidationAdapterRun(
            label=display_label,
            slug=slug,
            adapters=tuple(specs),
            is_base=False,
        )

    if adapter_path and mode != "none":
        specs = [_build_adapter_spec(adapter_path, adapter_strength, adapter_name)]
        preferred_label = adapter_name or _stem_from_path(adapter_path)
        runs.append(_make_run(preferred_label, specs))

    for entry in _iter_config_entries(adapter_config):
        label, specs = _normalize_run_entry(entry)
        if not specs:
            continue
        runs.append(_make_run(label, specs))

    include_base = True
    if adapter_path and mode == "adapter_only" and adapter_config in (None, [], {}):
        include_base = False

    ordered_runs: List[ValidationAdapterRun] = []
    if include_base:
        ordered_runs.append(ValidationAdapterRun.base())
    ordered_runs.extend(runs)

    if not ordered_runs:
        ordered_runs.append(ValidationAdapterRun.base())

    return ordered_runs
