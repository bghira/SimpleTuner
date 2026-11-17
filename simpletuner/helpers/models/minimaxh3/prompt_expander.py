"""MiniMax's released prompt-expansion requests for an OpenAI-compatible PE server."""

import base64
import copy
import io
import json
import math
import os
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

from simpletuner.helpers.models.prompt_upsampling import PromptUpsampler

PROMPT_EXPANDER_MODEL_ID = "MiniMaxAI/MiniMax-H3-Image-PE"


class MiniMaxH3ImagePromptExpander(PromptUpsampler):
    def __init__(self, url):
        self.url = url.rstrip("/")
        if not self.url.endswith("/v1"):
            self.url += "/v1"
        self.prompts = {}
        for name in ("t2i/system_prompt.txt", "t2i/task_prompt.txt", "t2i/request_params.json", "edit/task_prompt.txt"):
            path = hf_hub_download(PROMPT_EXPANDER_MODEL_ID, f"prompt/{name}")
            self.prompts[name] = Path(path).read_text(encoding="utf-8")

    def build_request(self, request):
        body = json.loads(self.prompts["t2i/request_params.json"])
        body.update(model=PROMPT_EXPANDER_MODEL_ID, stream=False)
        ratio_gcd = math.gcd(request.width, request.height)
        ratio = f"{request.width // ratio_gcd}:{request.height // ratio_gcd}"
        if not request.reference_images:
            prompt = f"{request.prompt}\nFinal output canvas: {request.width}x{request.height} pixels."
            body["messages"] = [
                {"role": "system", "content": self.prompts["t2i/system_prompt.txt"].rstrip("\n")},
                {"role": "user", "content": self.prompts["t2i/task_prompt.txt"].format(prompt)},
            ]
            return body
        content = [{"type": "text", "text": self.prompts["edit/task_prompt.txt"]}]
        count = len(request.reference_images)
        if count > 16:
            raise ValueError("MiniMax-H3-Image supports at most sixteen reference images.")
        content.append(
            {
                "type": "text",
                "text": f"Reference Image Count:\n{count} input image(s), labeled <Picture 1> through <Picture {count}>.",
            }
        )
        for index, image in enumerate(request.reference_images, 1):
            divisor = math.gcd(image.width, image.height)
            content.append(
                {
                    "type": "text",
                    "text": f"Reference Media:\n- <Picture {index}> [aspect_ratio {image.width // divisor}:{image.height // divisor}]",
                }
            )
            buffer = io.BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
            encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}})
        content.append(
            {
                "type": "text",
                "text": f"\nUser Editing Instruction (JSON string):\n{json.dumps(request.prompt)}\n\nUser Aspect Ratio Selection:\n{ratio}\n\nOutput only the required two-field JSON object.",
            }
        )
        body["messages"] = [{"role": "user", "content": content}]
        schema = copy.deepcopy(body["response_format"]["json_schema"]["schema"])
        schema["properties"] = {"expanded_caption": {"type": "string"}, "target_aspect_ratio": {"type": "string"}}
        schema["required"] = ["expanded_caption", "target_aspect_ratio"]
        body["response_format"]["json_schema"]["schema"] = schema
        return body

    def upsample(self, requests):
        headers = {"Content-Type": "application/json"}
        token = os.environ.get("MINIMAX_H3_PROMPT_EXPANDER_API_KEY")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        captions = []
        for request in requests:
            payload = self.build_request(request)
            query = urllib.request.Request(
                f"{self.url}/chat/completions", data=json.dumps(payload).encode(), headers=headers
            )
            with urllib.request.urlopen(query, timeout=1200) as response:
                result = json.load(response)
            (choice,) = result["choices"]
            if choice["finish_reason"] != "stop":
                raise ValueError("MiniMax-H3 prompt expansion did not finish.")
            caption = json.loads(choice["message"]["content"])["expanded_caption"]
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError("MiniMax-H3 prompt expansion returned an empty caption.")
            captions.append(caption.strip())
        return captions
