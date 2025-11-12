"""Audio loading utilities."""

from __future__ import annotations

import os
import tempfile
import wave
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


def _load_with_wave(source: AudioSource) -> Tuple[torch.Tensor, int]:
    """Fallback WAV reader used when torchaudio cannot decode a source."""
    if isinstance(source, (bytes, bytearray)):
        payload = bytes(source)
    elif isinstance(source, str):
        with open(source, "rb") as file:
            payload = file.read()
    elif hasattr(source, "read"):
        stream = source  # type: ignore[assignment]
        position = None
        if hasattr(stream, "tell") and hasattr(stream, "seek"):
            try:
                position = stream.tell()
                stream.seek(0)
            except (OSError, AttributeError):
                position = None
        payload = stream.read()
        if position is not None:
            stream.seek(position)
    else:
        raise TypeError(f"Unsupported audio source type: {type(source)}")

    buffer = BytesIO(payload)
    with wave.open(buffer, "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        num_channels = wav_file.getnchannels()
        num_frames = wav_file.getnframes()
        frames = wav_file.readframes(num_frames)

    frame_buffer = bytearray(frames)
    waveform = torch.frombuffer(frame_buffer, dtype=torch.int16).to(torch.float32)
    if num_channels > 1:
        waveform = waveform.view(-1, num_channels).t()
    else:
        waveform = waveform.view(1, -1)
    waveform /= 32767.0
    return waveform.contiguous(), sample_rate


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
    except (RuntimeError, ImportError, OSError):
        if hasattr(stream, "read"):
            try:
                stream.seek(0)
            except (AttributeError, OSError):
                pass
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_file.write(stream.read())
                tmp_path = tmp_file.name
            try:
                waveform, sample_rate = torchaudio.load(tmp_path, format="wav")
            except Exception:
                waveform, sample_rate = _load_with_wave(tmp_path)
            finally:
                if hasattr(stream, "seek"):
                    try:
                        stream.seek(0)
                    except (AttributeError, OSError):
                        pass
                os.unlink(tmp_path)
        else:
            waveform, sample_rate = _load_with_wave(stream)
    return waveform, sample_rate
