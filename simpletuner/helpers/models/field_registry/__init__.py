"""Model-specific field registry discovery."""

import importlib
import pkgutil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simpletuner.simpletuner_sdk.server.services.field_registry.registry import FieldRegistry


def register_model_field_registries(registry: "FieldRegistry") -> None:
    package = importlib.import_module(__name__)

    for module_info in sorted(pkgutil.iter_modules(package.__path__), key=lambda item: item.name):
        if module_info.ispkg or module_info.name.startswith("_"):
            continue

        module = importlib.import_module(f"{__name__}.{module_info.name}")
        register_fields = getattr(module, "register_fields")
        register_fields(registry)
