from helpers.webhooks.config import WebhookConfig
import requests
import os
import json
import logging
import time
from io import BytesIO

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
        mock_webhook_config: WebhookConfig = None,
    ):
        self.accelerator = accelerator
        self.config = mock_webhook_config or WebhookConfig(config_path)
        self.webhook_url = self.config.values.get(
            "webhook_url", self.config.values.get("callback_url", None)
        )
        self.webhook_type = (
            self.config.webhook_type
        )  # Use webhook_type to differentiate behavior
        self.message_prefix = (
            f"`({self.config.message_prefix})` "
            if self.config.message_prefix is not None
            else f"`({project_name})` "
        )
        self.log_level = log_levels.get(
            self.config.log_level or "info", log_levels["info"]
        )
        self.stored_response = None

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
            files = self._prepare_images(images)
            request_args = {
                "data": data,
                "files": files if self.webhook_type == "discord" else None,
            }
        elif self.webhook_type == "raw":
            # Prepare raw data payload for direct POST
            if raw_request:
                data = message
                files = None
            else:
                data = {
                    "message": message,
                    "images": (
                        [self._convert_image_to_base64(img) for img in images]
                        if images
                        else []
                    ),
                }
            files = None
            request_args = {
                "json": data,
                "files": None,
            }
        else:
            logger.error(f"Unsupported webhook type: {self.webhook_type}")
            return

        # Send request
        try:
            logger.debug(f"Sending webhook request: {request_args}")
            post_result = requests.post(
                self.webhook_url,
                **request_args,
            )
            post_result.raise_for_status()
        except Exception as e:
            logger.error(f"Could not send webhook request: {e}")
            return

        if store_response:
            self.stored_response = post_result.headers

    def _prepare_images(self, images: list):
        """Convert images to file objects for Discord uploads."""
        files = {}
        if not images:
            return files
        if type(images) is not list:
            raise ValueError(f"Images must be a list of PIL images. Received: {images}")
        if images:
            for index, img in enumerate(images):
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
        import base64

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
        """Send a message through the webhook with optional images."""
        if not self.accelerator.is_main_process or "discord" != self.webhook_type:
            return
        if not self._check_level(message_level):
            return
        if images is not None and not isinstance(images, list):
            images = [images]

        # Split the images into smaller chunks if there are too many (Discord limitation)
        if images and len(images) > 10:
            for i in range(0, len(images), 9):
                self._send_request(
                    message, images[i : i + 9], store_response=store_response
                )
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
        for sending structured dict to the callback for eg. training step progress updates
        """
        if (
            "raw" != self.webhook_type
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
