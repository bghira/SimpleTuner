import logging
import os

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

    # Webhook might not support audio uploads directly in the same way as images/videos depending on implementation
    # But we can try sending a message.
    # If the webhook handler supports generic files, we could upload.
    # For now, just send the text notification.

    webhook_handler.send(message)
