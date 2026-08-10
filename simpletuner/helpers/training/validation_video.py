import base64
import importlib.util
import logging
import os
import shutil
import subprocess
from functools import lru_cache
from io import BytesIO

import numpy as np
import torch
import wandb
from diffusers.utils.export_utils import export_to_video
from PIL import Image

from simpletuner.helpers.multiaspect.image import MultiaspectImage
from simpletuner.helpers.training import validation_audio
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)


def _resolve_ffmpeg_path():
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        return ffmpeg_path
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


def _mux_audio_into_video(video_path, audio, sample_rate):
    if sample_rate is None:
        raise ValueError("audio_sample_rate is required to mux validation audio into video.")
    ffmpeg_path = _resolve_ffmpeg_path()
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
    audio_only=False,
):
    """
    Save validation videos to disk (with audio if provided).
    If audio_only=True and there are no videos but there is audio, saves audio files directly.
    Returns a list of video/audio paths.
    """
    validation_img_idx = 0
    video_paths = []
    audio_list = None
    if validation_audios is not None:
        audio_list = validation_audios.get(validation_shortname)

    # Handle audio-only mode: save audio files directly without video
    video_count = len(validation_images.get(validation_shortname, []))
    if audio_only and video_count == 0 and audio_list is not None:
        audio_paths = validation_audio.save_audio(
            save_dir,
            validation_audios,
            validation_shortname,
            sample_rate=audio_sample_rate,
        )
        return audio_paths

    # Validate audio/video count match for normal video+audio mode
    if audio_list is not None:
        if len(audio_list) != video_count:
            raise ValueError(f"Validation audio count ({len(audio_list)}) does not match video count ({video_count}).")

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
                fps=int(getattr(config, "framerate", None) or 16),
            )
            video_paths.append(video_path)
            if audio_list is not None:
                _mux_audio_into_video(video_path, audio_list[validation_img_idx], audio_sample_rate)
        except Exception as e:
            logger.error(f"Failed to save validation video to {video_path}: {e}")

        validation_img_idx += 1

    return video_paths


def _frame_to_rgb_array(frame):
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    if torch.is_tensor(frame):
        frame = frame.detach().cpu()
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
            frame = frame.permute(1, 2, 0)
        frame = frame.numpy()
    array = np.asarray(frame)
    if array.ndim == 2:
        array = array[..., np.newaxis]
    if array.ndim != 3:
        raise ValueError(f"Expected a video frame with 2 or 3 dimensions, got shape {array.shape}.")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    elif array.shape[-1] == 4:
        array = array[..., :3]
    elif array.shape[-1] != 3:
        raise ValueError(f"Expected RGB-like video frame data, got shape {array.shape}.")
    return _normalise_array_to_uint8(array)


def _normalise_array_to_uint8(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.integer):
        return np.clip(array, 0, 255).astype(np.uint8)
    array = array.astype(np.float32, copy=False)
    if array.size and float(np.nanmin(array)) < 0.0:
        array = (array + 1.0) / 2.0
    if array.size and float(np.nanmax(array)) > 1.0:
        array = array / 255.0
    return np.clip(array * 255.0, 0, 255).astype(np.uint8)


def _video_to_thwc_array(media):
    if isinstance(media, Image.Image):
        return None
    if isinstance(media, list):
        if not media:
            return None
        return np.stack([_frame_to_rgb_array(frame) for frame in media], axis=0)
    if torch.is_tensor(media):
        media = media.detach().cpu()
        if media.ndim == 5:
            if media.shape[-1] in (1, 3, 4):  # B, T, H, W, C
                media = media[0]
            elif media.shape[1] in (1, 3, 4):  # B, C, T, H, W
                media = media[0].permute(1, 0, 2, 3)
            elif media.shape[2] in (1, 3, 4):  # B, T, C, H, W
                media = media[0]
            else:
                return None
        media = media.numpy()
    array = np.asarray(media)
    if array.ndim == 5:
        if array.shape[-1] in (1, 3, 4):  # B, T, H, W, C
            array = array[0]
        elif array.shape[1] in (1, 3, 4):  # B, C, T, H, W
            array = np.moveaxis(array[0], 0, -1)
        elif array.shape[2] in (1, 3, 4):  # B, T, C, H, W
            array = np.moveaxis(array[0], 1, -1)
        else:
            return None
    elif array.ndim == 4:
        if array.shape[-1] in (1, 3, 4):  # T, H, W, C
            pass
        elif array.shape[1] in (1, 3, 4):  # T, C, H, W
            array = np.moveaxis(array, 1, -1)
        elif array.shape[0] in (1, 3, 4):  # C, T, H, W
            array = np.moveaxis(array, 0, -1)
        else:
            return None
    else:
        return None
    return np.stack([_frame_to_rgb_array(frame) for frame in array], axis=0)


def _image_to_chw_batch(media):
    if isinstance(media, list):
        if not media:
            return None
        media = media[0]
    try:
        image = _frame_to_rgb_array(media)
    except Exception:
        return None
    return np.moveaxis(image, -1, 0)[np.newaxis, ...]


def _tensorboard_video(media):
    video = _video_to_thwc_array(media)
    if video is None:
        return None
    tensor = torch.from_numpy(video).permute(0, 3, 1, 2).unsqueeze(0).float() / 255.0
    return tensor


@lru_cache(maxsize=1)
def _tensorboard_video_supported() -> bool:
    """TensorBoard's video writer requires the legacy moviepy.editor module."""
    try:
        return importlib.util.find_spec("moviepy.editor") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _wandb_video(media):
    video = _video_to_thwc_array(media)
    if video is None:
        return None
    return np.moveaxis(video, -1, 1)


def _first_frame_image(media):
    video = _video_to_thwc_array(media)
    if video is not None and len(video) > 0:
        return Image.fromarray(video[0])
    try:
        return Image.fromarray(_frame_to_rgb_array(media))
    except Exception:
        return None


def log_videos_to_trackers(accelerator, validation_images, validation_resolutions, config):
    """
    Log validation video media to trackers.

    TensorBoard's image path expects NCHW and rejects 5D video tensors, so video
    samples are sent through SummaryWriter.add_video as N,T,C,H,W.
    """
    fps = int(getattr(config, "framerate", None) or 16)
    global_step = StateTracker.get_global_step()
    for tracker in accelerator.trackers:
        if tracker.name == "comet_ml":
            experiment = accelerator.get_tracker("comet_ml").tracker
            for shortname, media_list in validation_images.items():
                for idx, media in enumerate(media_list):
                    image = _first_frame_image(media)
                    if image is None:
                        continue
                    res_label = str(validation_resolutions[idx]) if idx < len(validation_resolutions) else "unknown"
                    experiment.log_image(image, name=f"{shortname} - {res_label} - frame 0")
        elif tracker.name == "tensorboard":
            tracker = accelerator.get_tracker("tensorboard")
            image_logs = {}
            video_supported = _tensorboard_video_supported()
            for shortname, media_list in validation_images.items():
                for idx, media in enumerate(media_list):
                    res_label = validation_resolutions[idx] if idx < len(validation_resolutions) else "unknown"
                    tag = f"{shortname} - {res_label}"
                    if video_supported:
                        video = _tensorboard_video(media)
                        if video is not None:
                            tracker.writer.add_video(tag, video, global_step=global_step, fps=fps)
                            continue
                    image = _image_to_chw_batch(media)
                    if image is None and not video_supported:
                        first_frame = _first_frame_image(media)
                        if first_frame is not None:
                            image = _image_to_chw_batch(first_frame)
                    if image is not None:
                        image_logs[tag] = image
            if image_logs:
                tracker.log_images(image_logs, step=global_step)
        elif tracker.name == "wandb":
            resolution_list = []
            for res in validation_resolutions:
                if isinstance(res, tuple):
                    resolution_list.append(f"{res[0]}x{res[1]}")
                else:
                    resolution_list.append(str(res))

            logs = {}
            for shortname, media_list in validation_images.items():
                for idx, media in enumerate(media_list):
                    res_label = resolution_list[idx] if idx < len(resolution_list) else "unknown"
                    caption = f"{shortname} - {res_label}"
                    video = _wandb_video(media)
                    if video is not None:
                        logs[caption] = wandb.Video(video, fps=fps, format="mp4", caption=caption)
                    else:
                        image = _first_frame_image(media)
                        if image is not None:
                            logs[caption] = wandb.Image(image, caption=caption)
            if logs:
                tracker.log(logs, step=global_step)


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
