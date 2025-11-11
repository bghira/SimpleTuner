"""Audio loading utilities."""

from __future__ import annotations

import os
import tempfile
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
    format_hint = None
    if isinstance(stream, str):
        _, ext = os.path.splitext(stream)
        if ext:
            format_hint = ext.lstrip(".")
    try:
        waveform, sample_rate = torchaudio.load(stream, format=format_hint)
    except RuntimeError:
        if hasattr(stream, "read"):
            stream.seek(0)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(stream.read())
                tmp_path = tmp_file.name
            try:
                waveform, sample_rate = torchaudio.load(tmp_path, format="wav")
            finally:
                stream.seek(0)
                os.unlink(tmp_path)
        else:
            raise
    return waveform, sample_rate
