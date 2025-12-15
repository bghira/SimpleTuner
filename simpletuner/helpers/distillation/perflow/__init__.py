__all__ = ["PerFlowDistiller"]
from .distiller import PerFlowDistiller


def __getattr__(name):
    if name == "PerFlowDistiller":
        from .distiller import PerFlowDistiller

        return PerFlowDistiller
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
