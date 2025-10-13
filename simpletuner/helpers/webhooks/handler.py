import base64
import json
import logging
import os
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import imageio  # <-- for in-memory video encoding (MP4 or GIF)
import numpy as np
import requests

from simpletuner.helpers.multiaspect.image import MultiaspectImage
from simpletuner.helpers.webhooks.config import WebhookConfig

# Define log levels
log_levels = {"critical": 0, "error": 1, "warning": 2, "info": 3, "debug": 4}

logger = logging.getLogger(__name__)
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


def _truncate_for_log(
    obj,
    *,
    max_length: int = 256,
    preview_length: int = 64,
    suffix: str = "...[truncated]...",
    _seen: set | None = None,
):
    """Return a copy of *obj* with long strings shortened for debug logging."""
    if _seen is None:
        _seen = set()

    if isinstance(obj, str):
        return obj if len(obj) <= max_length else f"{obj[:preview_length]}{suffix}"

    if obj is None or isinstance(obj, (int, float, bool)):
        return obj

    obj_id = id(obj)
    if obj_id in _seen:
        return "<recursion>"
    _seen.add(obj_id)

    if isinstance(obj, dict):
        return {
            (
                _truncate_for_log(key, max_length=max_length, preview_length=preview_length, suffix=suffix, _seen=_seen)
                if isinstance(key, str)
                else key
            ): _truncate_for_log(value, max_length=max_length, preview_length=preview_length, suffix=suffix, _seen=_seen)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple, set)):
        items = [
            _truncate_for_log(item, max_length=max_length, preview_length=preview_length, suffix=suffix, _seen=_seen)
            for item in obj
        ]
        if isinstance(obj, tuple):
            return tuple(items)
        if isinstance(obj, set):
            return items
        return items

    if hasattr(obj, "__dict__") and isinstance(getattr(obj, "__dict__", None), dict):
        return _truncate_for_log(vars(obj), max_length=max_length, preview_length=preview_length, suffix=suffix, _seen=_seen)

    return str(obj)


class WebhookHandler:
    def __init__(
        self,
        accelerator,
        project_name: str,
        webhook_config: dict | list = None,
        mock_webhook_config: WebhookConfig = None,
        send_video: bool = False,
        video_framerate: int = None,
    ):
        self.accelerator = accelerator
        self.send_video = send_video
        self.video_framerate = video_framerate
        self.stored_response = None

        # Handle mock config for testing
        if mock_webhook_config is not None:
            self.backends = [self._create_backend(mock_webhook_config, project_name)]
        else:
            # Normalize webhook_config to list format
            if webhook_config is None:
                self.backends = []
            elif isinstance(webhook_config, dict):
                # Single dict → wrap in list
                self.backends = [self._create_backend(WebhookConfig(webhook_config), project_name)]
            elif isinstance(webhook_config, list):
                # List of dicts → create backend for each
                self.backends = [self._create_backend(WebhookConfig(config), project_name) for config in webhook_config]
            else:
                raise ValueError(f"webhook_config must be dict or list, got {type(webhook_config)}")

        # For backward compatibility, expose first backend's properties
        if self.backends:
            first_backend = self.backends[0]
            self.config = first_backend["config"]
            self.webhook_url = first_backend["webhook_url"]
            self.webhook_type = first_backend["webhook_type"]
            self.message_prefix = first_backend["message_prefix"]
            self.log_level = first_backend["log_level"]
        else:
            self.config = None
            self.webhook_url = None
            self.webhook_type = None
            self.message_prefix = f"`({project_name})` "
            self.log_level = log_levels["info"]

    def _create_backend(self, config: WebhookConfig, project_name: str) -> dict:
        """Create a webhook backend configuration."""
        return {
            "config": config,
            "webhook_url": config.webhook_url,
            "webhook_type": config.webhook_type,
            "message_prefix": (
                f"`({config.message_prefix})` " if config.message_prefix is not None else f"`({project_name})` "
            ),
            "log_level": log_levels.get(config.log_level or "info", log_levels["info"]),
            "ssl_no_verify": getattr(config, "ssl_no_verify", False)
            or os.environ.get("SIMPLETUNER_SSL_NO_VERIFY", "false").lower() == "true",
        }

    @staticmethod
    def from_unprocessed_config(accelerator, project_name: str, raw_json_config: str, send_video: bool = False):
        """Create a WebhookHandler from a raw JSON string config."""
        try:
            config_dict = json.loads(raw_json_config)
            config = WebhookConfig(config_dict)
            return WebhookHandler(accelerator, project_name, config, send_video=send_video)
        except Exception as e:
            logging.error(f"Could not parse webhook configuration: {e}")
            return None

    def _check_level(self, level: str, backend_log_level: int) -> bool:
        """Check if the message level meets the backend's configured log level."""
        return log_levels.get(level, log_levels["info"]) <= backend_log_level

    def _send_request_to_backend(
        self,
        backend: dict,
        message: str | dict,
        images: list = None,
        store_response: bool = False,
        raw_request: bool = False,
    ):
        """Send a webhook request to a specific backend."""
        webhook_type = backend["webhook_type"]
        webhook_url = backend["webhook_url"]
        message_prefix = backend["message_prefix"]

        if webhook_type == "discord":
            # Prepare Discord-style payload
            data = {"content": f"{message_prefix}{message}"}
            if self.send_video:
                # images is actually a list of "videos" in this usage
                files = self._prepare_videos(images)
            else:
                # images is a list of PIL Images
                files = self._prepare_images(images)

            request_args = {"data": data, "files": files}

        elif webhook_type == "raw":
            # Prepare raw data payload for direct POST
            if raw_request:
                # If already fully formed JSON or dict, sanitize for safe JSON encoding first
                data = self._sanitize_for_json(message)
                files = None
            else:
                # Convert images to base64 for a generic "raw" JSON
                # Convert images, filtering out any that fail
                converted_images = []
                if images:
                    for img in images:
                        converted = self._convert_image_to_base64(img)
                        if converted:
                            converted_images.append(converted)
                data = {
                    "message": message,
                    "images": converted_images,
                }
                files = None

            request_args = {"json": data, "files": files}

        else:
            logging.error(f"Unsupported webhook type: {webhook_type}")
            return

        # Send request
        try:
            logging.debug("Sending webhook request to %s: %s", webhook_url, _truncate_for_log(request_args))
            # Configure SSL verification
            verify = not backend.get("ssl_no_verify", False)
            post_result = requests.post(webhook_url, **request_args, timeout=5, verify=verify)
            post_result.raise_for_status()
        except (requests.exceptions.ConnectionError, BrokenPipeError) as e:
            # Connection errors are expected when WebUI is refreshed/closed
            # Silently ignore to avoid confusing the UI
            logging.debug(f"Webhook connection unavailable (expected during page refresh): {e}")
            return
        except requests.exceptions.Timeout:
            # Timeout is also benign - just means WebUI is slow/unresponsive
            logging.debug("Webhook request timed out (WebUI may be busy)")
            return
        except Exception as e:
            # Log other errors at warning level since they might indicate real issues
            logging.warning(f"Could not send webhook request to {webhook_url}: {e}")
            return

        if store_response:
            self.stored_response = post_result.headers

    def _sanitize_for_json(self, payload, _seen=None):
        """Convert objects to JSON-serializable structures."""
        if _seen is None:
            _seen = set()

        if payload is None or isinstance(payload, (str, int, float, bool)):
            return payload

        if isinstance(payload, Path):
            return str(payload)

        if isinstance(payload, np.generic):
            return payload.item()

        if isinstance(payload, np.ndarray):
            return payload.tolist()

        # Avoid infinite recursion on cyclic references
        obj_id = id(payload)
        if obj_id in _seen:
            return str(payload)
        _seen.add(obj_id)

        if isinstance(payload, dict):
            return {str(key): self._sanitize_for_json(value, _seen) for key, value in payload.items()}

        if isinstance(payload, (list, tuple, set)):
            return [self._sanitize_for_json(item, _seen) for item in payload]

        if hasattr(payload, "dict") and callable(payload.dict):
            try:
                return self._sanitize_for_json(payload.dict(), _seen)
            except Exception:
                pass

        if hasattr(payload, "__dict__"):
            try:
                return self._sanitize_for_json(vars(payload), _seen)
            except TypeError:
                pass

        # Fallback: string representation
        return str(payload)

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
        from PIL import Image

        # Handle string paths
        if isinstance(image, str):
            try:
                image = Image.open(image)
            except Exception as e:
                logging.error(f"Failed to open image from path {image}: {e}")
                return None

        # Handle PIL Image objects
        if hasattr(image, "save"):
            img_byte_array = BytesIO()
            image.save(img_byte_array, format="PNG")
            img_byte_array.seek(0)
            return base64.b64encode(img_byte_array.read()).decode("utf-8")

        logging.error(f"Unsupported image type: {type(image)}")
        return None

    def send(
        self,
        message: str,
        images: list = None,
        message_level: str = "info",
        store_response: bool = False,
    ):
        """
        Send a message through all configured webhooks with optional images/videos.
        If self.send_video is True, `images` is interpreted as `videos`.
        """
        # Only send from main process
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return

        if images is not None and not isinstance(images, list):
            images = [images]

        # Send to each backend that meets the log level requirement
        for backend in self.backends:
            if not self._check_level(message_level, backend["log_level"]):
                continue

            # Skip Discord on non-main process
            if (
                backend["webhook_type"] == "discord"
                and self.accelerator is not None
                and not self.accelerator.is_main_process
            ):
                continue

            # Discord limits: max 10 attachments
            max_attachments = 10
            if images and len(images) > max_attachments:
                for i in range(0, len(images), max_attachments):
                    try:
                        self._send_request_to_backend(
                            backend,
                            message,
                            images[i : i + max_attachments],
                            store_response=store_response,
                        )
                    except Exception as e:
                        logging.error(f"Error sending webhook to {backend['webhook_url']}: {e}")
            else:
                try:
                    self._send_request_to_backend(backend, message, images, store_response=store_response)
                except Exception as e:
                    logging.error(f"Error sending webhook to {backend['webhook_url']}: {e}")

    def send_raw(
        self,
        structured_data: dict,
        message_type: str | None = None,
        message_level: str = "info",
        job_id: str | None = None,
        images: list | None = None,
    ):
        """
        Send structured data to all "raw" webhooks (JSON payload).
        """
        # Only send from main process
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return

        if not isinstance(structured_data, dict):
            logging.error("send_raw expects a mapping payload.")
            return

        payload = dict(structured_data)

        if message_type and "type" not in payload:
            payload["type"] = message_type

        if job_id and payload.get("job_id") is None:
            payload["job_id"] = job_id

        if "severity" not in payload and message_level:
            payload["severity"] = message_level

        if "timestamp" not in payload:
            payload["timestamp"] = datetime.now(tz=timezone.utc).isoformat()

        # Send to all raw webhook backends that meet the log level
        for backend in self.backends:
            if backend["webhook_type"] != "raw":
                continue
            if not self._check_level(message_level, backend["log_level"]):
                continue

            try:
                self._send_request_to_backend(
                    backend, message=payload, images=images, store_response=False, raw_request=True
                )
            except Exception as e:
                logging.error(f"Error sending raw webhook to {backend['webhook_url']}: {e}")
