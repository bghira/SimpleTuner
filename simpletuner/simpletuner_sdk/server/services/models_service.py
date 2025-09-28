"""Service helpers for model metadata routes."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

from fastapi import status

from simpletuner.helpers.models.common import ModelFoundation, PipelineTypes
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
            "overrides_requires_conditioning_validation_inputs": self._is_method_overridden(
                model_cls, "requires_conditioning_validation_inputs"
            ),
            "overrides_requires_validation_edit_captions": self._is_method_overridden(
                model_cls, "requires_validation_edit_captions"
            ),
            "overrides_conditioning_validation_dataset_type": self._is_method_overridden(
                model_cls, "conditioning_validation_dataset_type"
            ),
            "has_controlnet_pipeline": any(
                pt in pipeline_types for pt in {PipelineTypes.CONTROLNET.value, PipelineTypes.CONTROL.value}
            ),
        }

        details = {
            "family": model_family,
            "class": self._class_path(model_cls),
            "display_name": display_name or model_family,
            "flavours": flavours,
            "attributes": attributes,
            "pipelines": {
                "types": pipeline_types,
                "implementations": pipeline_mapping,
            },
            "capabilities": capabilities,
        }
        return details

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


MODELS_SERVICE = ModelsService()
