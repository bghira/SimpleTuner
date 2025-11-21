import base64
import logging
import os
from io import BytesIO

from diffusers.utils.export_utils import export_to_video

from simpletuner.helpers.multiaspect.image import MultiaspectImage
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)


def save_videos(save_dir, validation_images, validation_shortname, validation_resolutions, config):
    """
    Save validation videos to disk.
    Returns a list of video paths.
    """
    validation_img_idx = 0
    video_paths = []

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
