"""Audio helpers for SimpleTuner."""


def __getattr__(name: str):
    """Lazy import audio functions to avoid requiring torchaudio at import time."""
    if name in ("load_audio", "load_audio_from_video", "generate_zero_audio"):
        from . import load

        return getattr(load, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["load_audio", "load_audio_from_video", "generate_zero_audio"]
