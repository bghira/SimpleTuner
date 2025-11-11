"""Audio loading utilities."""

from __future__ import annotations

from io import BytesIO
from typing import IO, Tuple, Union

import torch

try:
    import torchaudio
except ModuleNotFoundError as exc:  # pragma: no cover - import error surfaces early
    raise ModuleNotFoundError("torchaudio is required for audio dataset support.") from exc


AudioSource = Union[str, bytes, bytearray, IO[bytes]]


def _coerce_to_stream(source: AudioSource) -> Union[str, IO[bytes]]:
    if isinstance(source, (bytes, bytearray)):
        buffer = BytesIO(source)
        buffer.seek(0)
        return buffer

    if hasattr(source, "read"):
        stream = source  # type: ignore[assignment]
        try:
            stream.seek(0)
        except (AttributeError, OSError):
            data = stream.read()
            stream = BytesIO(data)
        return stream

    return source


def load_audio(source: AudioSource) -> Tuple[torch.Tensor, int]:
    """
    Load an audio source into a waveform tensor and sample rate using torchaudio.

    Args:
        source: Path to an audio file, raw bytes, bytearray, or a file-like object.

    Returns:
        Tuple of waveform tensor shaped (channels, samples) and the sample rate.
    """
    stream = _coerce_to_stream(source)
    if isinstance(stream, str):
        waveform, sample_rate = torchaudio.load(stream)
    else:
        waveform, sample_rate = torchaudio.load(stream)
    return waveform, sample_rate
