from helpers.webhooks.config import WebhookConfig
import requests
from io import BytesIO
from PIL import Image

log_levels = {"critical": 0, "error": 1, "warning": 2, "info": 3, "debug": 4}


class WebhookHandler:
    def __init__(self, config_path: str, accelerator, project_name: str):
        self.accelerator = accelerator
        self.config = WebhookConfig(config_path)
        self.webhook_url = self.config.webhook_url
        self.message_prefix = (
            f"`({self.config.message_prefix})` "
            if self.config.message_prefix is not None
            else f"`({project_name})` "
        )
        self.log_level = log_levels.get(self.config.log_level or "info", log_levels["info"])
        self.stored_response = None

    def _check_level(self, level: str) -> bool:
        return log_levels.get(level, "info") >= self.log_level

    def _send_request(
        self, message: str, images: list = None, store_response: bool = False
    ):
        # Prepare the request data
        data = {"content": f"{self.message_prefix}{message}"}
        files = {}

        if images:
            # Convert PIL images to BytesIO and add to files dictionary
            for index, img in enumerate(images):
                img_byte_array = BytesIO()
                img.save(img_byte_array, format="PNG")
                img_byte_array.seek(0)
                files[f"file{index}"] = (
                    f"image{index}.png",
                    img_byte_array,
                    "image/png",
                )

        # Send request to webhook URL with images if present
        post_result = requests.post(self.webhook_url, data=data, files=files)
        if store_response:
            self.stored_response = post_result.headers
            print(f"Stored result: {self.stored_response}")

    def send(
        self,
        message: str,
        images: list = None,
        message_level: str = "info",
        store_response: bool = False,
    ):
        if not self.accelerator.is_main_process:
            return
        if images is not None and not isinstance(images, list):
            images = [images]
        # Send webhook message
        if self._check_level(message_level):
            if images and len(images) <= 10:
                self._send_request(message, images, store_response=store_response)
            elif images and len(images) > 10:
                for i in range(0, len(images), 9):
                    self._send_request(
                        message, images[i : i + 9], store_response=store_response
                    )
            else:
                self._send_request(message, store_response=store_response)
