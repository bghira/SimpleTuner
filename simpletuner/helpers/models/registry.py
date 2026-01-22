import json
import os
from typing import Any, Dict, Type


class LazyModelClass:
    def __init__(self, metadata):
        self._metadata = metadata
        self._real_class = None

    @property
    def NAME(self):
        return self._metadata.get("name", "Unknown Model")

    @property
    def PREDICTION_TYPE(self):
        """Return prediction type from metadata without importing the module."""
        prediction_type = self._metadata.get("prediction_type")
        if prediction_type is None:
            return None
        # Return a simple object with a .value attribute to match the enum pattern
        # used in loss.py: getattr(getattr(cls, "PREDICTION_TYPE"), "value", ...)
        return type("PredictionType", (), {"value": prediction_type})()

    def get_flavour_choices(self):
        return self._metadata.get("flavour_choices", [])

    def __call__(self, *args, **kwargs):
        if self._real_class is None:
            import importlib

            module = importlib.import_module(self._metadata["module_path"])
            self._real_class = getattr(module, self._metadata["class_name"])
        return self._real_class(*args, **kwargs)

    def __getattr__(self, name):
        if self._real_class is None:
            import importlib

            module = importlib.import_module(self._metadata["module_path"])
            self._real_class = getattr(module, self._metadata["class_name"])
        return getattr(self._real_class, name)

    def get_real_class(self):
        """Resolve and return the actual model class."""
        if self._real_class is None:
            import importlib

            module = importlib.import_module(self._metadata["module_path"])
            self._real_class = getattr(module, self._metadata["class_name"])
        return self._real_class


class ModelRegistry:
    _registry: Dict[str, Type[Any]] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    _loaded_metadata: bool = False

    @classmethod
    def _load_metadata(cls):
        if cls._loaded_metadata:
            return
        metadata_path = os.path.join(os.path.dirname(__file__), "model_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r") as f:
                    cls._metadata = json.load(f)
            except Exception:
                pass
        cls._loaded_metadata = True

    @classmethod
    def register(cls, family: str, model_class: Type[Any]) -> None:
        cls._registry[family.lower()] = model_class

    @classmethod
    def get(cls, family: str) -> Type[Any]:
        family_lower = family.lower()
        if family_lower in cls._registry:
            return cls._registry[family_lower]

        cls._load_metadata()
        if family_lower in cls._metadata:
            return LazyModelClass(cls._metadata[family_lower])

        return None

    @classmethod
    def model_families(cls) -> Dict[str, Type[Any]]:
        cls._load_metadata()
        combined = {}
        # Metadata-based lazy classes
        for family, meta in cls._metadata.items():
            combined[family] = LazyModelClass(meta)
        # Explicitly registered classes override metadata
        for family, model_cls in cls._registry.items():
            combined[family] = model_cls
        return combined
