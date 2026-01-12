"""Service helpers for model metadata routes."""

from __future__ import annotations

import inspect
import re
from types import SimpleNamespace
from typing import Any, Dict, Optional

from fastapi import status

from simpletuner.helpers.models.common import AudioModelFoundation, ModelFoundation, PipelineTypes, VideoModelFoundation
from simpletuner.helpers.models.registry import ModelRegistry


class ModelServiceError(Exception):
    """Domain error raised when model service operations fail."""

    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class ModelsService:
    """Coordinator for model metadata operations."""

    _ATTRIBUTE_WHITELIST = {
        "NAME",
        "MODEL_DESCRIPTION",
        "ENABLED_IN_WIZARD",
        "MODEL_LICENSE",
        "DEFAULT_MODEL_FLAVOUR",
        "HUGGINGFACE_PATHS",
        "DEFAULT_LORA_TARGET",
        "DEFAULT_LYCORIS_TARGET",
        "MODEL_CLASS",
        "MODEL_SUBFOLDER",
        "MAXIMUM_CANVAS_SIZE",
        "CONTROLNET_LORA_STATE_DICT_PREFIX",
        "PREDICTION_TYPE",
        "MODEL_TYPE",
        "TEXT_ENCODER_CONFIGURATION",
        "SUPPORTS_TEXT_ENCODER_TRAINING",
        "AUTOENCODER_CLASS",
        "LATENT_CHANNEL_COUNT",
        "REQUIRES_FLAVOUR",
    }

    def __init__(self) -> None:
        self._models_loaded = False

    def list_families(self) -> Dict[str, Any]:
        self._ensure_models_loaded()
        families = sorted(ModelRegistry.model_families().keys())
        return {"families": families}

    def list_wizard_models(self) -> Dict[str, Any]:
        """Return model families enabled for the wizard with their descriptions."""
        self._ensure_models_loaded()
        wizard_models = []
        category_order = {"image": 0, "video": 1, "audio": 2, "other": 3}

        for family_name, model_cls in ModelRegistry.model_families().items():
            # Skip entries that aren't actual classes
            if not isinstance(model_cls, type):
                continue

            # Check if model is enabled in wizard (default to True if not specified)
            enabled = getattr(model_cls, "ENABLED_IN_WIZARD", True)
            if not enabled:
                continue

            # Get model metadata
            display_name = getattr(model_cls, "NAME", family_name)
            description = getattr(model_cls, "MODEL_DESCRIPTION", "")
            try:
                if issubclass(model_cls, AudioModelFoundation):
                    category = "audio"
                elif issubclass(model_cls, VideoModelFoundation):
                    category = "video"
                else:
                    category = "image"
            except TypeError:
                category = "image"

            wizard_models.append(
                {
                    "family": family_name,
                    "name": display_name,
                    "description": description,
                    "category": category,
                }
            )

        # Sort by display name for consistent ordering
        wizard_models.sort(key=lambda x: (category_order.get(x.get("category"), 99), x["name"]))

        return {"models": wizard_models}

    def get_model_flavours(self, model_family: str) -> Dict[str, Any]:
        model_cls = self._get_model_class(model_family)
        flavours: list[str] = []
        if hasattr(model_cls, "get_flavour_choices") and callable(model_cls.get_flavour_choices):
            try:
                flavours = list(model_cls.get_flavour_choices())
            except Exception as exc:  # pragma: no cover - defensive
                raise ModelServiceError(
                    f"Failed to gather flavours for '{model_family}': {exc}",
                    status.HTTP_500_INTERNAL_SERVER_ERROR,
                ) from exc
        return {"family": model_family, "flavours": flavours}

    def get_model_details(self, model_family: str) -> Dict[str, Any]:
        model_cls = self._get_model_class(model_family)

        attributes: Dict[str, Any] = {}
        for attr in self._ATTRIBUTE_WHITELIST:
            if not hasattr(model_cls, attr):
                continue
            if attr == "NAME":
                continue  # handled separately
            value = getattr(model_cls, attr)
            attributes[self._normalise_key(attr)] = self._serialize_value(value)

        display_name: Optional[str] = getattr(model_cls, "NAME", None)

        flavours = []
        try:
            flavours = list(model_cls.get_flavour_choices())
        except Exception:  # pragma: no cover - defensive
            pass

        pipelines_raw = getattr(model_cls, "PIPELINE_CLASSES", None)
        pipeline_types: list[str] = []
        pipeline_mapping: Dict[str, Optional[str]] = {}
        if isinstance(pipelines_raw, dict):
            for key, implementation in pipelines_raw.items():
                pipeline_key = key.value if isinstance(key, PipelineTypes) else str(key)
                pipeline_types.append(pipeline_key)
                pipeline_mapping[pipeline_key] = (
                    self._class_path(implementation) if inspect.isclass(implementation) else None
                )

        capabilities = {
            "overrides_requires_conditioning_dataset": self._is_method_overridden(
                model_cls, "requires_conditioning_dataset"
            ),
            "overrides_requires_conditioning_latents": self._is_method_overridden(
                model_cls, "requires_conditioning_latents"
            ),
            "overrides_requires_conditioning_image_embeds": self._is_method_overridden(
                model_cls, "requires_conditioning_image_embeds"
            ),
            "overrides_requires_conditioning_validation_inputs": self._is_method_overridden(
                model_cls, "requires_conditioning_validation_inputs"
            ),
            "overrides_requires_validation_edit_captions": self._is_method_overridden(
                model_cls, "requires_validation_edit_captions"
            ),
            "overrides_requires_validation_i2v_samples": self._is_method_overridden(
                model_cls, "requires_validation_i2v_samples"
            ),
            "overrides_conditioning_validation_dataset_type": self._is_method_overridden(
                model_cls, "conditioning_validation_dataset_type"
            ),
            "has_controlnet_pipeline": any(
                pt in pipeline_types for pt in {PipelineTypes.CONTROLNET.value, PipelineTypes.CONTROL.value}
            ),
            "is_video_model": isinstance(model_cls, type) and issubclass(model_cls, VideoModelFoundation),
            "is_audio_model": isinstance(model_cls, type) and issubclass(model_cls, AudioModelFoundation),
        }

        # Check if model supports lyrics by examining caption_field_preferences
        supports_lyrics = False
        if hasattr(model_cls, "caption_field_preferences") and callable(model_cls.caption_field_preferences):
            try:
                # Check with audio dataset type (most likely to have lyrics)
                audio_fields = model_cls.caption_field_preferences(dataset_type="audio")
                if "lyrics" in (audio_fields or []):
                    supports_lyrics = True
                # Also check without dataset_type for models that always support lyrics
                if not supports_lyrics:
                    default_fields = model_cls.caption_field_preferences(dataset_type=None)
                    if "lyrics" in (default_fields or []):
                        supports_lyrics = True
            except Exception:
                supports_lyrics = False
        capabilities["supports_lyrics"] = supports_lyrics

        strict_i2v_flavours: list[str] = []
        try:
            if hasattr(model_cls, "strict_i2v_flavours") and callable(model_cls.strict_i2v_flavours):
                strict_i2v_flavours = list(model_cls.strict_i2v_flavours())
        except Exception:
            strict_i2v_flavours = []
        capabilities["strict_i2v_flavours"] = strict_i2v_flavours
        capabilities["strict_i2v_all_flavours"] = bool(getattr(model_cls, "STRICT_I2V_FOR_ALL_FLAVOURS", False))
        preview_spec = getattr(model_cls, "VALIDATION_PREVIEW_SPEC", None)
        supports_preview = preview_spec is not None
        if not supports_preview:
            base_method = getattr(ModelFoundation, "get_validation_preview_spec")
            current_method = getattr(model_cls, "get_validation_preview_spec", base_method)
            supports_preview = current_method is not base_method
        capabilities["supports_validation_preview"] = supports_preview

        default_flavour = getattr(model_cls, "DEFAULT_MODEL_FLAVOUR", None)
        if default_flavour is None:
            hf_paths = getattr(model_cls, "HUGGINGFACE_PATHS", None)
            if isinstance(hf_paths, dict) and hf_paths:
                try:
                    default_flavour = next(iter(hf_paths.keys()))
                except StopIteration:  # pragma: no cover - defensive
                    default_flavour = None

        details = {
            "family": model_family,
            "class": self._class_path(model_cls),
            "display_name": display_name or model_family,
            "flavours": flavours,
            "default_flavour": default_flavour,
            "attributes": attributes,
            "pipelines": {
                "types": pipeline_types,
                "implementations": pipeline_mapping,
            },
            "capabilities": capabilities,
        }
        return details

    def evaluate_requirements(
        self,
        model_family: str,
        config: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Evaluate conditioning requirements for a model configuration."""

        if not model_family:
            raise ModelServiceError("model_family is required", status.HTTP_400_BAD_REQUEST)

        model_cls = self._get_model_class(model_family)

        normalized: Dict[str, Any] = {}
        normalized.update(self._normalise_config_dict(config or {}))
        if metadata:
            normalized.update(self._normalise_config_dict(metadata))

        normalized.setdefault("model_family", model_family)

        config_namespace = _ConfigNamespace()
        for key, value in normalized.items():
            sanitised = self._sanitize_key(key)
            if not sanitised:
                continue
            config_namespace.setdefault(sanitised, value)

        config_namespace.setdefault("model_family", model_family)
        config_namespace.setdefault("model_flavour", normalized.get("model_flavour") or normalized.get("modelflavour"))
        config_namespace.setdefault("model_type", normalized.get("model_type"))
        config_namespace.setdefault("controlnet", bool(config_namespace.get("controlnet")))
        config_namespace.setdefault("control", bool(config_namespace.get("control")))

        placeholder = SimpleNamespace(config=config_namespace)

        def _safe_call(attr: str, default):
            method = getattr(model_cls, attr, None)
            if not callable(method):
                return default
            try:
                result = method(placeholder)
            except Exception:
                return default
            return result if result is not None else default

        requires_dataset = bool(_safe_call("requires_conditioning_dataset", False))
        requires_latents = bool(_safe_call("requires_conditioning_latents", False))
        requires_image_embeds = bool(_safe_call("requires_conditioning_image_embeds", False))
        requires_validation_inputs = bool(_safe_call("requires_conditioning_validation_inputs", False))
        requires_edit_captions = bool(_safe_call("requires_validation_edit_captions", False))
        requires_i2v_validation_samples = bool(_safe_call("requires_validation_i2v_samples", False))
        supports_conditioning_dataset = bool(_safe_call("supports_conditioning_dataset", False))
        dataset_type = _safe_call("conditioning_validation_dataset_type", "conditioning")
        if not isinstance(dataset_type, str) or not dataset_type:
            dataset_type = "conditioning"

        return {
            "requires_conditioning_dataset": requires_dataset,
            "requires_conditioning_latents": requires_latents,
            "requires_conditioning_image_embeds": requires_image_embeds,
            "requires_conditioning_validation_inputs": requires_validation_inputs,
            "requires_validation_edit_captions": requires_edit_captions,
            "requires_validation_i2v_samples": requires_i2v_validation_samples,
            "conditioning_dataset_type": dataset_type,
            "supports_conditioning_dataset": supports_conditioning_dataset,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _ensure_models_loaded(self) -> None:
        if self._models_loaded:
            return
        try:
            import simpletuner.helpers.models.all  # noqa: F401 - side effects populate registry
        except Exception as exc:  # pragma: no cover - defensive
            raise ModelServiceError(
                f"Failed to load model registry: {exc}",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            ) from exc
        self._models_loaded = True

    def _get_model_class(self, model_family: str):
        self._ensure_models_loaded()
        cls = ModelRegistry.get(model_family)
        if cls is None:
            raise ModelServiceError(
                f"Model family '{model_family}' not found",
                status.HTTP_404_NOT_FOUND,
            )
        return cls

    @staticmethod
    def _normalise_key(name: str) -> str:
        return name.lower()

    @staticmethod
    def _class_path(obj: Any) -> Optional[str]:
        if not inspect.isclass(obj):
            return None
        return f"{obj.__module__}.{obj.__name__}"

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if isinstance(value, list) or isinstance(value, tuple) or isinstance(value, set):
            return [ModelsService._serialize_value(item) for item in value]
        if isinstance(value, dict):
            serialized: Dict[str, Any] = {}
            for key, item in value.items():
                if isinstance(key, PipelineTypes):
                    key_str = key.value
                else:
                    key_str = str(key)
                serialized[key_str] = ModelsService._serialize_value(item)
            return serialized
        if isinstance(value, bool):
            return value
        if inspect.isclass(value):
            return ModelsService._class_path(value)
        if hasattr(value, "value") and not isinstance(value, (str, bytes)):
            # Enum-like
            try:
                return value.value
            except Exception:
                pass
        return value

    @staticmethod
    def _is_method_overridden(model_cls: type, method_name: str) -> bool:
        method = getattr(model_cls, method_name, None)
        base_method = getattr(ModelFoundation, method_name, None)
        if not callable(method) or not callable(base_method):
            return False
        return getattr(method, "__code__", None) is not getattr(base_method, "__code__", None)

    @staticmethod
    def _sanitize_key(key: str) -> str:
        if not key or not isinstance(key, str):
            return ""
        trimmed = key.strip()
        if trimmed.startswith("--"):
            trimmed = trimmed[2:]
        trimmed = trimmed.replace("-", "_")
        trimmed = trimmed.replace(" ", "_")
        trimmed = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", trimmed)
        trimmed = re.sub(r"__+", "_", trimmed)
        trimmed = re.sub(r"[^0-9a-zA-Z_]", "_", trimmed)
        trimmed = trimmed.lower()
        if trimmed and trimmed[0].isdigit():
            trimmed = f"_{trimmed}"
        return trimmed

    @staticmethod
    def _normalise_config_dict(config: Dict[str, Any]) -> Dict[str, Any]:
        normalised: Dict[str, Any] = {}
        for raw_key, value in config.items():
            if not isinstance(raw_key, str):
                continue
            key_variants = set()
            trimmed = raw_key.strip()
            key_variants.add(trimmed)
            no_prefix = trimmed[2:] if trimmed.startswith("--") else trimmed
            key_variants.add(no_prefix)
            key_variants.add(no_prefix.lower())
            snake = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", no_prefix)
            snake = re.sub(r"[-\s]+", "_", snake).lower()
            key_variants.add(snake)
            key_variants.add(snake.replace("__", "_"))
            key_variants.add(snake.replace("_", ""))

            for variant in key_variants:
                if variant:
                    normalised[variant] = value
        return normalised


MODELS_SERVICE = ModelsService()


class _ConfigNamespace(dict):
    """Dictionary with attribute-style access used for model config stubs."""

    __slots__ = ()

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc

    def __setattr__(self, key, value):
        self[key] = value
