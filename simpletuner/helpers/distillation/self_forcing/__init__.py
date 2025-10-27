__all__ = ["SelfForcingDistillation"]


def __getattr__(name):
    if name == "SelfForcingDistillation":
        from .distiller import SelfForcingDistillation

        return SelfForcingDistillation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
