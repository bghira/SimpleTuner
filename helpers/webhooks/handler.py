from helpers.webhooks.config import WebhookConfig
from helpers.multiaspect.image import MultiaspectImage
from pathlib import Path
import requests
import os
import json
import logging
import numpy as np
import time
from io import BytesIO
import base64

import imageio  # <-- for in-memory video encoding (MP4 or GIF)

# Define log levels
log_levels = {"critical": 0, "error": 1, "warning": 2, "info": 3, "debug": 4}

logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))


class WebhookHandler:
    def __init__(
        self,
        config_path: str,
        accelerator,
        project_name: str,
        args,
        mock_webhook_config: WebhookConfig = None,
        send_video: bool = False,
    ):
        self.accelerator = accelerator
        self.config = mock_webhook_config or WebhookConfig(config_path)
        self.webhook_url = self.config.values.get(
            "webhook_url", self.config.values.get("callback_url", None)
        )
        self.webhook_type = self.config.webhook_type  # "discord" or "raw"
        self.message_prefix = (
            f"`({self.config.message_prefix})` "
            if self.config.message_prefix is not None
            else f"`({project_name})` "
        )
        self.log_level = log_levels.get(
            self.config.log_level or "info", log_levels["info"]
        )
        self.stored_response = None
        self.send_video = send_video
        self.video_framerate = args.framerate

    def _check_level(self, level: str) -> bool:
        """Check if the message level meets the configured log level."""
        return log_levels.get(level, "info") <= self.log_level

    def _send_request(
        self,
        message: str,
        images: list = None,
        store_response: bool = False,
        raw_request: bool = False,
    ):
        """Send the webhook request based on the webhook type."""
        if self.webhook_type == "discord":
            # Prepare Discord-style payload
            data = {"content": f"{self.message_prefix}{message}"}
            if self.send_video:
                # images is actually a list of "videos" in this usage
                files = self._prepare_videos(images)
            else:
                # images is a list of PIL Images
                files = self._prepare_images(images)

            request_args = {"data": data, "files": files}

        elif self.webhook_type == "raw":
            # Prepare raw data payload for direct POST
            if raw_request:
                # If already fully formed JSON or dict, just send raw
                data = message
                files = None
            else:
                # Convert images to base64 for a generic "raw" JSON
                data = {
                    "message": message,
                    "images": (
                        [self._convert_image_to_base64(img) for img in images]
                        if images
                        else []
                    ),
                }
                files = None

            request_args = {"json": data, "files": files}

        else:
            logger.error(f"Unsupported webhook type: {self.webhook_type}")
            return

        # Send request
        try:
            logger.debug(f"Sending webhook request: {request_args}")
            post_result = requests.post(self.webhook_url, **request_args)
            post_result.raise_for_status()
        except Exception as e:
            logger.error(f"Could not send webhook request: {e}")
            return

        if store_response:
            self.stored_response = post_result.headers

    def _prepare_videos(self, videos: list):
        """
        Convert in-memory video frames to file objects for Discord uploads.

        We assume each item in `videos` is either:
        1) A list of frames (PIL Images or NumPy arrays) for a single video, or
        2) Already an in-memory BytesIO with encoded MP4 data.
        """
        files = {}
        if not videos:
            return files
        if not isinstance(videos, list):
            raise ValueError(f"Videos must be a list. Received: {type(videos)}")

        for index, vid_data in enumerate(videos):
            # If vid_data is a BytesIO, assume it's already an mp4
            if isinstance(vid_data, BytesIO):
                vid_data.seek(0)
                files[f"file{index}"] = (f"video{index}.mp4", vid_data, "video/mp4")
            else:
                # Otherwise, assume vid_data is a list of frames (PIL or np.ndarray).
                # We'll convert each PIL Image to a NumPy array, then pass to mimwrite.
                frames = []
                for frame in vid_data:
                    if hasattr(frame, "convert"):  # i.e. PIL image
                        frames.append(np.asarray(frame))
                    else:
                        # assume it's already a NumPy array
                        frames.append(frame)

                video_byte_array = BytesIO()
                imageio.v3.imwrite(
                    video_byte_array,
                    frames,  # a list of NumPy arrays
                    plugin="pyav",  # or "ffmpeg"
                    fps=self.video_framerate,
                    extension=".mp4",
                    codec="libx264",
                )
                video_byte_array.seek(0)

                files[f"file{index}"] = (
                    f"video{index}.mp4",
                    video_byte_array,
                    "video/mp4",
                )
        return files

    def _prepare_images(self, images: list):
        """Convert PIL images to file objects for Discord uploads."""
        files = {}
        if not images:
            return files
        if not isinstance(images, list):
            raise ValueError(f"Images must be a list of PIL images. Received: {images}")

        for index, img in enumerate(images):
            img = MultiaspectImage.numpy_list_to_pil(img)
            img_byte_array = BytesIO()
            img.save(img_byte_array, format="PNG")
            img_byte_array.seek(0)
            files[f"file{index}"] = (
                f"image{index}.png",
                img_byte_array,
                "image/png",
            )
        return files

    def _convert_image_to_base64(self, image):
        """Convert PIL image to a base64 string (for 'raw' webhook type)."""
        img_byte_array = BytesIO()
        image.save(img_byte_array, format="PNG")
        img_byte_array.seek(0)
        return base64.b64encode(img_byte_array.read()).decode("utf-8")

    def send(
        self,
        message: str,
        images: list = None,
        message_level: str = "info",
        store_response: bool = False,
    ):
        """
        Send a message through the webhook with optional images/videos.
        If self.send_video is True, `images` is interpreted as `videos`.
        """
        # Only send from main process if it's Discord (to avoid duplicates).
        if not self.accelerator.is_main_process or self.webhook_type != "discord":
            return
        if not self._check_level(message_level):
            return

        if images is not None and not isinstance(images, list):
            images = [images]

        # Discord limits: max 10 attachments
        max_attachments = 10
        if images and len(images) > max_attachments:
            for i in range(0, len(images), max_attachments):
                try:
                    self._send_request(
                        message,
                        images[i : i + max_attachments],
                        store_response=store_response,
                    )
                except Exception as e:
                    logger.error(f"Error sending webhook: {e}")
        else:
            self._send_request(message, images, store_response=store_response)

    def send_raw(
        self,
        structured_data: dict,
        message_type: str,
        message_level: str = "info",
        job_id: str = None,
    ):
        """
        Send structured data to a "raw" webhook, e.g. for step progress.
        Ignores 'images' entirely, uses JSON payload only.
        """
        if (
            self.webhook_type != "raw"
            or not self.accelerator.is_main_process
            or not self._check_level(message_level)
        ):
            return
        structured_data["message_type"] = message_type
        structured_data["job_id"] = job_id
        structured_data["timestamp"] = int(time.time())
        self._send_request(
            message=structured_data, images=None, store_response=False, raw_request=True
        )
