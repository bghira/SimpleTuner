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

from simpletuner.helpers.logging import get_logger
from simpletuner.helpers.multiaspect.image import MultiaspectImage
from simpletuner.helpers.webhooks.config import WebhookConfig

# Define log levels
log_levels = {"critical": 0, "error": 1, "warning": 2, "info": 3, "debug": 4}

logger = get_logger(__name__, disable_webhook=True)


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
            "auth_token": getattr(config, "auth_token", None),
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
        videos: list = None,
        audios: list = None,
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
            use_video_attachments = bool(self.send_video and videos)
            attachments = []
            attachment_type = None
            if audios:
                attachments = audios
                attachment_type = "audio"
            elif use_video_attachments:
                attachments = videos or []
                attachment_type = "video"
            else:
                attachments = images or []
                attachment_type = "image"

            files = {}
            if attachments:
                if attachment_type == "audio":
                    files = self._prepare_audios(attachments)
                elif attachment_type == "video":
                    files = self._prepare_videos(attachments)
                else:
                    files = self._prepare_images(attachments)

            request_args = {"data": data, "files": files}

        elif webhook_type == "raw":
            # Prepare raw data payload for direct POST
            # Convert images to base64 for inclusion in JSON
            converted_images = []
            if images:
                for img in images:
                    converted = self._convert_image_to_base64(img)
                    if converted:
                        converted_images.append(converted)
            converted_videos = []
            if videos:
                for vid in videos:
                    converted = self._convert_video_to_base64(vid)
                    if converted:
                        converted_videos.append(converted)
            converted_audios = []
            if audios:
                for audio in audios:
                    converted = self._convert_audio_to_base64(audio)
                    if converted:
                        converted_audios.append(converted)

            if raw_request:
                # If already fully formed JSON or dict, sanitize for safe JSON encoding first
                data = self._sanitize_for_json(message)
                # Add images to the structured data if they exist
                if converted_images and isinstance(data, dict):
                    data["images"] = converted_images
                if converted_videos and isinstance(data, dict):
                    data["videos"] = converted_videos
                if converted_audios and isinstance(data, dict):
                    data["audios"] = converted_audios
                files = None
            else:
                data = {
                    "message": message,
                    "images": converted_images,
                }
                if converted_videos:
                    data["videos"] = converted_videos
                if converted_audios:
                    data["audios"] = converted_audios
                files = None

            request_args = {"json": data, "files": files}

        else:
            logging.error(f"Unsupported webhook type: {webhook_type}")
            return

        # Send request
        try:
            # logging.debug("Sending webhook request to %s: %s", webhook_url, _truncate_for_log(request_args))
            # Configure SSL verification
            verify = not backend.get("ssl_no_verify", False)

            # Add authentication header if auth token is provided
            headers = {}
            auth_token = backend.get("auth_token")
            if auth_token:
                headers["X-API-Key"] = auth_token

            post_result = requests.post(webhook_url, **request_args, headers=headers, timeout=5, verify=verify)
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

        # Support Pydantic v2 (model_dump) and v1 (dict) style serialization
        if hasattr(payload, "model_dump") and callable(payload.model_dump):
            try:
                return self._sanitize_for_json(payload.model_dump(), _seen)
            except Exception:
                pass
        elif hasattr(payload, "dict") and callable(payload.dict):
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

    def _prepare_audios(self, audios: list):
        """Convert audio sources to file objects for Discord uploads."""
        files = {}
        if not audios:
            return files
        if not isinstance(audios, list):
            raise ValueError(f"Audios must be a list. Received: {type(audios)}")

        for index, audio in enumerate(audios):
            filename = f"audio{index}.wav"
            mime_type = "audio/wav"

            if isinstance(audio, BytesIO):
                audio.seek(0)
                buffer = audio
                filename = getattr(audio, "name", None) or filename
            elif isinstance(audio, (bytes, bytearray)):
                buffer = BytesIO(bytes(audio))
                buffer.seek(0)
            elif isinstance(audio, str):
                if not os.path.isfile(audio):
                    raise ValueError(f"Unsupported audio path provided: {audio}")
                with open(audio, "rb") as handle:
                    buffer = BytesIO(handle.read())
                buffer.seek(0)
                filename = os.path.basename(audio) or filename
            elif hasattr(audio, "read"):
                try:
                    payload = audio.read()
                except Exception as exc:
                    raise ValueError(f"Could not read audio-like object for webhook payload: {exc}") from exc
                buffer = BytesIO(payload)
                buffer.seek(0)
                filename = getattr(audio, "name", None) or filename
            else:
                raise ValueError(f"Unsupported audio type for webhook payload: {type(audio)}")

            files[f"file{index}"] = (
                filename,
                buffer,
                mime_type,
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

        if isinstance(image, list):
            if not image:
                logging.error("Unsupported image type: empty list")
                return None
            image = MultiaspectImage.numpy_list_to_pil(image[0])

        # Handle PIL Image objects
        if hasattr(image, "save"):
            img_byte_array = BytesIO()
            image.save(img_byte_array, format="PNG")
            img_byte_array.seek(0)
            return base64.b64encode(img_byte_array.read()).decode("utf-8")

        logging.error(f"Unsupported image type: {type(image)}")
        return None

    def _convert_video_to_base64(self, video, mime_type: str = "video/mp4"):
        """Convert video sources to data URIs for raw webhooks."""
        if video is None:
            return None

        if isinstance(video, dict):
            src = (
                video.get("src")
                or video.get("url")
                or video.get("data")
                or video.get("base64")
                or video.get("video")
                or video.get("video_base64")
            )
            if isinstance(src, str) and src.strip():
                src = src.strip()
                resolved_mime = video.get("mime_type") or video.get("mime") or mime_type
                return {"src": src, "mime_type": resolved_mime}
            return None

        if isinstance(video, str):
            value = video.strip()
            if not value:
                return None
            if value.startswith("data:"):
                return {"src": value, "mime_type": mime_type}
            if value.startswith(("http://", "https://", "//")):
                return {"src": value, "mime_type": mime_type}
            if os.path.isfile(value):
                try:
                    with open(value, "rb") as handle:
                        data = handle.read()
                except Exception:
                    return None
                encoded = base64.b64encode(data).decode("utf-8")
                return {"src": f"data:{mime_type};base64,{encoded}", "mime_type": mime_type}
            return None

        data_bytes = None
        if isinstance(video, BytesIO):
            position = video.tell()
            video.seek(0)
            data_bytes = video.read()
            video.seek(position)
        elif hasattr(video, "read"):
            try:
                data_bytes = video.read()
            except Exception:
                data_bytes = None
            if hasattr(video, "seek"):
                try:
                    video.seek(0)
                except Exception:
                    pass
        elif isinstance(video, (bytes, bytearray)):
            data_bytes = bytes(video)

        if not data_bytes:
            return None

        encoded = base64.b64encode(data_bytes).decode("utf-8")
        return {"src": f"data:{mime_type};base64,{encoded}", "mime_type": mime_type}

    def _convert_audio_to_base64(self, audio, mime_type: str = "audio/wav"):
        """Convert audio sources to data URIs for raw webhooks."""
        if audio is None:
            return None

        if isinstance(audio, dict):
            src = (
                audio.get("src")
                or audio.get("url")
                or audio.get("data")
                or audio.get("base64")
                or audio.get("audio")
                or audio.get("audio_base64")
            )
            if isinstance(src, str) and src.strip():
                resolved_mime = audio.get("mime_type") or audio.get("mime") or mime_type
                return {"src": src.strip(), "mime_type": resolved_mime}
            return None

        if isinstance(audio, str):
            value = audio.strip()
            if not value:
                return None
            if value.startswith("data:") or value.startswith(("http://", "https://", "//")):
                return {"src": value, "mime_type": mime_type}
            if os.path.isfile(value):
                try:
                    with open(value, "rb") as handle:
                        data = handle.read()
                except Exception:
                    return None
                encoded = base64.b64encode(data).decode("utf-8")
                return {"src": f"data:{mime_type};base64,{encoded}", "mime_type": mime_type}
            return None

        data_bytes = None
        if isinstance(audio, BytesIO):
            position = audio.tell()
            audio.seek(0)
            data_bytes = audio.read()
            audio.seek(position)
        elif hasattr(audio, "read"):
            try:
                data_bytes = audio.read()
            except Exception:
                data_bytes = None
            if hasattr(audio, "seek"):
                try:
                    audio.seek(0)
                except Exception:
                    pass
        elif isinstance(audio, (bytes, bytearray)):
            data_bytes = bytes(audio)

        if not data_bytes:
            return None

        encoded = base64.b64encode(data_bytes).decode("utf-8")
        return {"src": f"data:{mime_type};base64,{encoded}", "mime_type": mime_type}

    def send(
        self,
        message: str,
        images: list = None,
        message_level: str = "info",
        store_response: bool = False,
        videos: list | None = None,
        audios: list | None = None,
    ):
        """
        Send a message through Discord webhooks with optional images/videos/audios.
        Raw webhooks (like WebUI callback) should use send_raw() with typed events.
        If self.send_video is True, `images` is interpreted as `videos`.
        """
        # Only send from main process
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return

        if images is not None and not isinstance(images, list):
            images = [images]
        if videos is not None and not isinstance(videos, list):
            videos = [videos]
        if audios is not None and not isinstance(audios, list):
            audios = [audios]

        # Send ONLY to Discord backends - raw backends should use send_raw()
        for backend in self.backends:
            # Only send to Discord backends
            if backend["webhook_type"] != "discord":
                continue

            if not self._check_level(message_level, backend["log_level"]):
                continue

            # Skip Discord on non-main process
            if self.accelerator is not None and not self.accelerator.is_main_process:
                continue

            # Discord limits: max 10 attachments
            max_attachments = 10
            use_videos = bool(self.send_video and videos)
            attachment_type = None
            attachments = []
            if audios:
                attachments = audios
                attachment_type = "audio"
            elif use_videos:
                attachments = videos or []
                attachment_type = "video"
            else:
                attachments = images or []
                attachment_type = "image"

            if attachments and len(attachments) > max_attachments:
                for i in range(0, len(attachments), max_attachments):
                    chunk = attachments[i : i + max_attachments]
                    try:
                        self._send_request_to_backend(
                            backend,
                            message,
                            images=chunk if attachment_type == "image" else None,
                            videos=chunk if attachment_type == "video" else None,
                            audios=chunk if attachment_type == "audio" else None,
                            store_response=store_response,
                        )
                    except Exception as e:
                        logging.error(f"Error sending webhook to {backend['webhook_url']}: {e}")
                continue

            try:
                self._send_request_to_backend(
                    backend,
                    message,
                    images=attachments if attachment_type == "image" else None,
                    videos=attachments if attachment_type == "video" else None,
                    audios=attachments if attachment_type == "audio" else None,
                    store_response=store_response,
                )
            except Exception as e:
                logging.error(f"Error sending webhook to {backend['webhook_url']}: {e}")

    def send_raw(
        self,
        structured_data: dict,
        message_type: str | None = None,
        message_level: str = "info",
        job_id: str | None = None,
        images: list | None = None,
        videos: list | None = None,
        audios: list | None = None,
        exclude_webhook_urls: list[str] | set[str] | tuple[str, ...] | str | None = None,
    ):
        """
        Send structured data to all "raw" webhooks (JSON payload) with optional media attachments.
        """
        # Only send from main process
        if self.accelerator is not None and not self.accelerator.is_main_process:
            return

        if not isinstance(structured_data, dict):
            logging.error("send_raw expects a mapping payload.")
            return

        if images is not None and not isinstance(images, list):
            images = [images]
        if videos is not None and not isinstance(videos, list):
            videos = [videos]
        if audios is not None and not isinstance(audios, list):
            audios = [audios]

        payload = dict(structured_data)

        if message_type and "type" not in payload:
            payload["type"] = message_type

        if job_id and payload.get("job_id") is None:
            payload["job_id"] = job_id

        if "severity" not in payload and message_level:
            payload["severity"] = message_level

        if "timestamp" not in payload:
            payload["timestamp"] = datetime.now(tz=timezone.utc).isoformat()

        exclude_urls = set()
        if exclude_webhook_urls:
            if isinstance(exclude_webhook_urls, str):
                exclude_urls.add(exclude_webhook_urls)
            else:
                exclude_urls.update([url for url in exclude_webhook_urls if isinstance(url, str) and url])

        # Send to all raw webhook backends that meet the log level
        for backend in self.backends:
            if backend["webhook_type"] != "raw":
                continue
            if exclude_urls and backend.get("webhook_url") in exclude_urls:
                continue
            if not self._check_level(message_level, backend["log_level"]):
                continue

            try:
                self._send_request_to_backend(
                    backend,
                    message=payload,
                    images=images,
                    videos=videos,
                    audios=audios,
                    store_response=False,
                    raw_request=True,
                )
            except Exception as e:
                logging.error(f"Error sending raw webhook to {backend['webhook_url']}: {e}")

    def send_lifecycle_stage(
        self,
        stage_key: str,
        stage_label: str,
        stage_status: str = "running",
        message: str | None = None,
        progress_current: int | None = None,
        progress_total: int | None = None,
        progress_percent: float | None = None,
    ):
        """
        Send a lifecycle stage event to raw webhooks.

        Args:
            stage_key: Unique identifier for the stage (e.g., "validation", "checkpoint_save")
            stage_label: Human-readable label for the stage
            stage_status: Status of the stage ("running", "completed", "failed")
            message: Optional message describing the stage
            progress_current: Optional current progress value
            progress_total: Optional total progress value
            progress_percent: Optional percentage complete
        """
        from simpletuner.helpers.training.state_tracker import StateTracker

        stage_data = {
            "key": stage_key,
            "label": stage_label,
            "status": stage_status,
            "progress": {
                "label": stage_label,
            },
        }

        if progress_current is not None:
            stage_data["progress"]["current"] = progress_current
        if progress_total is not None:
            stage_data["progress"]["total"] = progress_total
        if progress_percent is not None:
            stage_data["progress"]["percent"] = progress_percent

        payload = {
            "type": "lifecycle.stage",
            "stage": stage_data,
        }

        if message:
            payload["message"] = message
            payload["title"] = message

        self.send_raw(
            structured_data=payload,
            message_type="lifecycle.stage",
            message_level="info",
            job_id=StateTracker.get_job_id(),
        )
