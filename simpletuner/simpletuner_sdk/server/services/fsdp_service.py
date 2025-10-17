"""Services and utilities for FSDP-related operations."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from simpletuner.helpers.models.all import model_families

logger = logging.getLogger(__name__)


class FSDPServiceError(Exception):
    """Domain error raised when FSDP service operations fail."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass
class DetectedClassInfo:
    """Summary information for a detected module class."""

    class_name: str
    occurrences: int
    total_params: int
    sample_paths: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "occurrences": self.occurrences,
            "total_params": self.total_params,
            "sample_paths": self.sample_paths,
        }


class FSDPService:
    """Service responsible for detecting and caching model block information."""

    def __init__(self) -> None:
        self.cache_path = Path.home() / ".simpletuner" / "fsdp_block_cache.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load_cache()

    # --------------------------------------------------------------------- #
    # Cache helpers
    # --------------------------------------------------------------------- #
    def _load_cache(self) -> Dict[str, Any]:
        if not self.cache_path.exists():
            return {"entries": {}}
        try:
            with self.cache_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, dict):
                return {"entries": {}}
            data.setdefault("entries", {})
            return data
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load FSDP block cache; starting with an empty cache.")
            return {"entries": {}}

    def _save_cache(self) -> None:
        temp_path = self.cache_path.with_suffix(".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(self._cache, handle, indent=2, sort_keys=True)
                handle.write("\n")
            temp_path.replace(self.cache_path)
        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def clear_cache(self) -> Dict[str, Any]:
        """Remove cached detection results."""
        self._cache = {"entries": {}}
        try:
            if self.cache_path.exists():
                self.cache_path.unlink()
        except OSError as exc:
            logger.debug("Unable to remove FSDP block cache file %s: %s", self.cache_path, exc, exc_info=True)
        return {"cleared": True}

    # --------------------------------------------------------------------- #
    # Detection logic
    # --------------------------------------------------------------------- #
    def detect_block_classes(
        self,
        model_family: str,
        *,
        pretrained_model: Optional[str] = None,
        model_flavour: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Detect transformer/block classes for the supplied model family."""

        model_cls = model_families.get(model_family)
        if model_cls is None:
            raise FSDPServiceError(f"Unknown model_family '{model_family}'.")

        resolved_path = self._resolve_pretrained_path(model_cls, pretrained_model, model_flavour)
        cache_key = self._cache_key(model_family, resolved_path, model_flavour or "")
        cached_entry = self._cache["entries"].get(cache_key)

        if cached_entry and not force_refresh:
            return {**cached_entry, "cached": True}

        detection = self._run_detection(model_cls, resolved_path)
        detection.update(
            {
                "model_family": model_family,
                "pretrained_model": resolved_path,
                "model_flavour": model_flavour,
                "detected_at": time.time(),
                "cached": False,
            }
        )

        self._cache["entries"][cache_key] = detection
        self._save_cache()
        return detection

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _resolve_pretrained_path(self, model_cls: Any, explicit_path: Optional[str], model_flavour: Optional[str]) -> str:
        path = (explicit_path or "").strip()
        if path:
            return path

        flavour_map = getattr(model_cls, "HUGGINGFACE_PATHS", {}) or {}
        if model_flavour and model_flavour in flavour_map:
            return flavour_map[model_flavour]

        default_flavour = getattr(model_cls, "DEFAULT_MODEL_FLAVOUR", None)
        if default_flavour and default_flavour in flavour_map:
            return flavour_map[default_flavour]

        if flavour_map:
            # Fall back to the first available flavour
            for value in flavour_map.values():
                if value:
                    return value

        raise FSDPServiceError(
            "Unable to resolve a pretrained model path for detection. Please provide --pretrained_model_name_or_path."
        )

    def _cache_key(self, model_family: str, model_path: str, flavour: str) -> str:
        return "|".join([model_family or "", model_path or "", flavour or ""])

    def _run_detection(self, model_cls: Any, model_path: str) -> Dict[str, Any]:
        model_class = getattr(model_cls, "MODEL_CLASS", None)
        if model_class is None:
            raise FSDPServiceError(f"Model family '{model_cls.__name__}' does not expose a MODEL_CLASS attribute.")

        load_kwargs: Dict[str, Any] = {}
        model_subfolder = getattr(model_cls, "MODEL_SUBFOLDER", None)
        if model_subfolder and str(model_subfolder).strip().lower() not in {"", "none"}:
            load_kwargs["subfolder"] = model_subfolder

        try:
            config = model_class.load_config(model_path, **load_kwargs)
        except Exception as exc:  # pragma: no cover - defensive
            raise FSDPServiceError(f"Unable to load configuration for '{model_path}': {exc}") from exc

        try:
            with torch.no_grad():
                model = model_class.from_config(config)
        except Exception as exc:  # pragma: no cover - defensive
            raise FSDPServiceError(f"Failed to instantiate model '{model_class.__name__}': {exc}") from exc

        class_map: Dict[str, DetectedClassInfo] = {}
        parameter_total = 0

        for module_name, module in model.named_modules():
            if not module_name:  # skip root
                continue
            try:
                params = list(module.parameters(recurse=False))
            except Exception:
                params = []
            param_count = sum(param.numel() for param in params if param.requires_grad)
            if param_count == 0:
                continue
            parameter_total += param_count
            class_name = module.__class__.__name__
            info = class_map.get(class_name)
            if info is None:
                info = DetectedClassInfo(class_name=class_name, occurrences=0, total_params=0, sample_paths=[])
                class_map[class_name] = info
            info.occurrences += 1
            info.total_params += param_count
            if len(info.sample_paths) < 5:
                info.sample_paths.append(module_name)

        no_split_modules = []
        try:
            attr = getattr(model, "_no_split_modules", None)
            if attr:
                if isinstance(attr, (list, tuple, set)):
                    no_split_modules = sorted({str(item) for item in attr if item})
                else:
                    no_split_modules = [str(attr)]
        except Exception:
            no_split_modules = []

        del model  # free references early

        detected_classes = sorted(
            (info.to_dict() for info in class_map.values()),
            key=lambda item: (-item["total_params"], item["class_name"]),
        )

        return {
            "transformer_classes": detected_classes,
            "total_parameter_count": parameter_total,
            "no_split_modules": no_split_modules,
        }


FSDP_SERVICE = FSDPService()
