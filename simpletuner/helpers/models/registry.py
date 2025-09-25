from typing import Dict, Type, Any


class ModelRegistry:
    _registry: Dict[str, Type[Any]] = {}

    @classmethod
    def register(cls, family: str, model_class: Type[Any]) -> None:
        cls._registry[family.lower()] = model_class

    @classmethod
    def get(cls, family: str) -> Type[Any]:
        return cls._registry.get(family.lower())

    @classmethod
    def model_families(cls) -> Dict[str, Type[Any]]:
        return cls._registry.copy()