import base64
import logging
import os
from io import BytesIO
from typing import Any

import numpy as np
import scipy.io.wavfile
import torch
import torchaudio
import wandb

from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)


def save_audio(save_dir, validation_audios, validation_shortname, sample_rate=44100):
    """
    Save validation audio to disk.
    validation_audios: dict where key is shortname, value is list of audio tensors (C, T) or (T,)
    """
    audio_list = validation_audios.get(validation_shortname, [])
    saved_paths = []

    for idx, audio in enumerate(audio_list):
        # Ensure audio is on CPU
        if hasattr(audio, "cpu"):
            audio = audio.cpu()

        # Ensure audio is 2D (Channels, Time)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        filename = f"step_{StateTracker.get_global_step()}_{validation_shortname}_{idx}.wav"
        save_path = os.path.join(save_dir, filename)

        try:
            torchaudio.save(save_path, audio, sample_rate)
            saved_paths.append(save_path)
        except Exception as e:
            logger.error(f"Failed to save validation audio to {save_path}: {e}")

    return saved_paths


def _coerce_audio_tensor(audio: Any) -> torch.Tensor | None:
    """Normalise incoming audio payloads to a 2D CPU tensor for logging."""
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    if not torch.is_tensor(audio):
        logger.warning("Skipping validation audio of unsupported type %s", type(audio))
        return None

    tensor = audio.detach().cpu()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        logger.warning("Validation audio tensor must be 2D (channels, time); received shape %s", tuple(tensor.shape))
        return None
    return tensor


def _tensor_to_wav_buffer(audio: Any, sample_rate: int) -> BytesIO | None:
    """Encode a tensor-like audio clip into an in-memory WAV buffer."""
    tensor = _coerce_audio_tensor(audio)
    if tensor is None:
        return None

    buffer = BytesIO()
    try:
        # Transpose to (Time, Channels) for scipy and ensure numpy
        audio_np = tensor.numpy().T
        if audio_np.shape[1] == 1:
            audio_np = audio_np.squeeze(1)
        scipy.io.wavfile.write(buffer, sample_rate, audio_np)
    except Exception as exc:
        logger.warning("Unable to encode validation audio for webhook delivery: %s", exc)
        return None
    buffer.seek(0)
    return buffer


def log_audio_to_trackers(accelerator, validation_audios, validation_shortname, sample_rate=44100):
    """
    Log validation audio to trackers (WandB).
    """
    audio_list = validation_audios.get(validation_shortname, [])

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            wandb_audios = []
            for audio in audio_list:
                # WandB expects numpy array or path
                if hasattr(audio, "cpu"):
                    audio = audio.cpu().numpy()
                # If (C, T), wandb expects (T,) for mono or (T, C) ?
                # WandB docs: "numpy array of audio data"
                # Usually expects (samples,) or (samples, channels)
                if audio.ndim == 2:
                    if audio.shape[0] < audio.shape[1]:
                        # Assume (C, T) -> Transpose to (T, C)
                        audio = audio.transpose(1, 0)

                wandb_audios.append(wandb.Audio(audio, sample_rate=sample_rate, caption=validation_shortname))

            # Log as a list of audios? Or one by one?
            # Usually we log a table or just the media
            # For simplicity, let's log them individually with index
            log_dict = {}
            for idx, wa in enumerate(wandb_audios):
                log_dict[f"validation_audio/{validation_shortname}_{idx}"] = wa

            tracker.log(log_dict, step=StateTracker.get_global_step())


def log_audio_to_webhook(validation_audios, validation_shortname, validation_prompt, sample_rate=44100):
    """
    Log validation audio to webhook.
    """
    webhook_handler = StateTracker.get_webhook_handler()
    if webhook_handler is None:
        return

    message = (
        f"Validation audio for `{validation_shortname if validation_shortname != '' else '(blank shortname)'}`"
        f"\\nValidation prompt: `{validation_prompt if validation_prompt != '' else '(blank prompt)'}`"
    )

    audio_list = validation_audios.get(validation_shortname, [])
    audio_buffers_for_discord = []
    audio_payloads_for_raw = []

    for idx, audio in enumerate(audio_list):
        buffer = _tensor_to_wav_buffer(audio, sample_rate)
        if buffer is None:
            continue
        buffer.name = getattr(buffer, "name", None) or f"{validation_shortname or 'validation'}_{idx}.wav"
        wav_bytes = buffer.getvalue()
        data_uri = f"data:audio/wav;base64,{base64.b64encode(wav_bytes).decode('utf-8')}"
        # Ensure the buffer is rewound before sending to Discord
        buffer.seek(0)
        audio_buffers_for_discord.append(buffer)
        audio_payloads_for_raw.append({"src": data_uri, "mime_type": "audio/wav"})

    webhook_handler.send(
        message,
        audios=audio_buffers_for_discord,
    )

    webhook_handler.send_raw(
        structured_data={"message": f"Validation: {validation_shortname}"},
        message_type="training.validation",
        message_level="info",
        job_id=StateTracker.get_job_id(),
        audios=audio_payloads_for_raw,
    )
