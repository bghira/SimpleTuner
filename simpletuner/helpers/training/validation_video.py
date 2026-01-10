import base64
import logging
import os
import shutil
import subprocess
from io import BytesIO

from diffusers.utils.export_utils import export_to_video

from simpletuner.helpers.multiaspect.image import MultiaspectImage
from simpletuner.helpers.training import validation_audio
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)


def _mux_audio_into_video(video_path, audio, sample_rate):
    if sample_rate is None:
        raise ValueError("audio_sample_rate is required to mux validation audio into video.")
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required to mux validation audio into video.")

    audio_buffer = validation_audio._tensor_to_wav_buffer(audio, sample_rate)
    if audio_buffer is None:
        raise ValueError("Unable to coerce validation audio for muxing.")
    base_path, ext = os.path.splitext(video_path)
    if not ext:
        ext = ".mp4"
    temp_video_path = f"{base_path}.tmp{ext}"
    try:
        result = subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-loglevel",
                "error",
                "-i",
                video_path,
                "-f",
                "wav",
                "-i",
                "pipe:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                temp_video_path,
            ],
            input=audio_buffer.getvalue(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"ffmpeg failed with exit code {result.returncode}: {stderr}")
        os.replace(temp_video_path, video_path)
    finally:
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)


def save_videos(
    save_dir,
    validation_images,
    validation_shortname,
    validation_resolutions,
    config,
    validation_audios=None,
    audio_sample_rate=None,
):
    """
    Save validation videos to disk (with audio if provided).
    Returns a list of video paths.
    """
    validation_img_idx = 0
    video_paths = []
    audio_list = None
    if validation_audios is not None:
        audio_list = validation_audios.get(validation_shortname)
        if audio_list is not None:
            expected = len(validation_images.get(validation_shortname, []))
            if len(audio_list) != expected:
                raise ValueError(f"Validation audio count ({len(audio_list)}) does not match video count ({expected}).")

    # validation_images[validation_shortname] is a list of image lists (frames) or single images
    for validation_image in validation_images.get(validation_shortname, []):
        # Get the validation resolution for this index
        if validation_img_idx < len(validation_resolutions):
            resolution = validation_resolutions[validation_img_idx]
            if isinstance(resolution, str):
                if "x" in resolution:
                    res_label = resolution
                else:
                    res_label = f"{resolution}x{resolution}"
            elif isinstance(resolution, tuple):
                res_label = f"{resolution[0]}x{resolution[1]}"
            else:
                res_label = f"{resolution}x{resolution}"
        else:
            # Fallback to actual size if somehow out of bounds
            logger.warning(f"Image index {validation_img_idx} exceeds validation resolutions list")
            if isinstance(validation_image, list) and len(validation_image) > 0:
                size_x, size_y = validation_image[0].size
            elif hasattr(validation_image, "size"):
                size_x, size_y = validation_image.size
            else:
                size_x, size_y = (0, 0)
            res_label = f"{size_x}x{size_y}"

        # convert array of numpy to array of pil:
        validation_image = MultiaspectImage.numpy_list_to_pil(validation_image)

        if not isinstance(validation_image, list):
            # save as single image instead
            filename = f"step_{StateTracker.get_global_step()}_{validation_shortname}_{validation_img_idx}_{res_label}.png"
            save_path = os.path.join(save_dir, filename)
            try:
                validation_image.save(save_path)
            except Exception as e:
                logger.error(f"Failed to save validation image to {save_path}: {e}")
            validation_img_idx += 1
            continue

        filename = f"step_{StateTracker.get_global_step()}_{validation_shortname}_{validation_img_idx}_{res_label}.mp4"
        video_path = os.path.join(save_dir, filename)

        try:
            export_to_video(
                validation_image,
                video_path,
                fps=config.framerate,
            )
            video_paths.append(video_path)
            if audio_list is not None:
                _mux_audio_into_video(video_path, audio_list[validation_img_idx], audio_sample_rate)
        except Exception as e:
            logger.error(f"Failed to save validation video to {video_path}: {e}")

        validation_img_idx += 1

    return video_paths


def log_videos_to_webhook(validation_images, validation_video_paths, validation_shortname, validation_prompt, eval_scores):
    """
    Log validation videos to webhook.
    """
    webhook_handler = StateTracker.get_webhook_handler()
    if webhook_handler is None:
        return

    message = (
        f"Validation video for `{validation_shortname if validation_shortname != '' else '(blank shortname)'}`"
        f"\\nValidation prompt: `{validation_prompt if validation_prompt != '' else '(blank prompt)'}`"
        f"\\nEvaluation score: {eval_scores.get(validation_shortname, 'N/A')}"
    )

    video_paths = validation_video_paths.get(validation_shortname, [])
    videos_for_discord = []
    videos_for_raw = []

    if video_paths:
        for path in video_paths:
            try:
                with open(path, "rb") as handle:
                    video_bytes = handle.read()
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.warning("Failed to read validation video %s: %s", path, exc)
                continue
            video_buffer = BytesIO(video_bytes)
            video_buffer.name = os.path.basename(path)
            videos_for_discord.append(video_buffer)
            data_uri = f"data:video/mp4;base64,{base64.b64encode(video_bytes).decode('utf-8')}"
            videos_for_raw.append({"src": data_uri, "mime_type": "video/mp4"})

    webhook_handler.send(
        message,
        videos=videos_for_discord,
    )

    webhook_handler.send_raw(
        structured_data={"message": f"Validation: {validation_shortname}"},
        message_type="training.validation",
        message_level="info",
        job_id=StateTracker.get_job_id(),
        videos=videos_for_raw,
    )
