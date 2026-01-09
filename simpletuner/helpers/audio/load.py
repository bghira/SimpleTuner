"""Audio loading utilities."""

from __future__ import annotations

import logging
import os
import tempfile
import wave
from io import BytesIO
from pathlib import Path
from typing import IO, Optional, Tuple, Union

import torch

try:
    import torchaudio
except ModuleNotFoundError as exc:  # pragma: no cover - import error surfaces early
    raise ModuleNotFoundError("torchaudio is required for audio dataset support.") from exc

logger = logging.getLogger(__name__)


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
    waveform = torch.frombuffer(frame_buffer, dtype=torch.int16).to(torch.float32) / 32767.0
    if num_channels > 1:
        waveform = waveform.view(-1, num_channels).t()
    else:
        waveform = waveform.view(1, -1)
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
                # Some streams are not seekable; continue from current position.
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
                        # Safe to ignore failing to reset non-seekable streams.
                        pass
                os.unlink(tmp_path)
        else:
            waveform, sample_rate = _load_with_wave(stream)
    return waveform, sample_rate


def generate_zero_audio(
    duration_seconds: float,
    sample_rate: int = 16000,
    channels: int = 1,
) -> Tuple[torch.Tensor, int]:
    """
    Generate a zero-filled audio tensor for videos without audio streams.

    Args:
        duration_seconds: Duration of the zero audio in seconds.
        sample_rate: Sample rate in Hz (default: 16000).
        channels: Number of audio channels (default: 1 for mono).

    Returns:
        Tuple of (waveform tensor, sample_rate).
    """
    num_samples = int(duration_seconds * sample_rate)
    waveform = torch.zeros(channels, num_samples)
    return waveform, sample_rate


def load_audio_from_video(
    source: Union[str, Path, bytes, BytesIO],
    target_sample_rate: int = 16000,
    target_channels: int = 1,
) -> Tuple[torch.Tensor, int]:
    """
    Extract audio stream from a video file.

    Works with local file paths or byte streams (for S3/HuggingFace backends).

    Args:
        source: Path to video file, or raw bytes/BytesIO from a remote backend.
        target_sample_rate: Target sample rate in Hz (default: 16000 for Wav2Vec2).
        target_channels: Target number of channels (default: 1 for mono).

    Returns:
        Tuple of (waveform tensor, sample_rate).

    Raises:
        ValueError: If the video has no audio stream.
        RuntimeError: If ffmpeg extraction fails for other reasons.
    """
    import subprocess

    # Handle byte streams (from S3/HuggingFace backends)
    cleanup_source = False
    if isinstance(source, (bytes, BytesIO)):
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            if isinstance(source, bytes):
                tmp.write(source)
            else:
                source.seek(0)
                tmp.write(source.read())
            source_path = Path(tmp.name)
        cleanup_source = True
    else:
        source_path = Path(source) if isinstance(source, str) else source

    try:
        # Use ffmpeg to extract audio from video to avoid torchaudio video decode crashes.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav_path = tmp.name

        try:
            cmd = [
                "ffmpeg",
                "-i",
                str(source_path),
                "-vn",
                "-acodec",
                "pcm_s16le",
                "-ar",
                str(target_sample_rate),
                "-ac",
                str(target_channels),
                "-y",
                tmp_wav_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)

            if result.returncode != 0:
                stderr = result.stderr or ""
                if "does not contain any stream" in stderr or "Output file is empty" in stderr:
                    raise ValueError(f"Video has no audio stream: {source_path}")
                raise RuntimeError(f"ffmpeg failed: {stderr[:500]}")

            # Check if output file is empty or very small (no audio extracted)
            if not os.path.exists(tmp_wav_path) or os.path.getsize(tmp_wav_path) < 100:
                raise ValueError(f"Video has no audio stream: {source_path}")

            waveform, sample_rate = _load_with_wave(tmp_wav_path)
            return waveform, sample_rate
        finally:
            if os.path.exists(tmp_wav_path):
                os.unlink(tmp_wav_path)
    finally:
        if cleanup_source and source_path.exists():
            source_path.unlink()
