"""The FLUX.3 Image Klein Qwen3.5 prompt-upsample backend."""

import json
import math

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoProcessor, LogitsProcessor, Qwen3_5ForConditionalGeneration

from simpletuner.helpers.models.prompt_upsampling import PromptUpsampler

PROMPT_UPSAMPLER_MODEL_ID = "black-forest-labs/flux-3-image-klein-pup"


class GeneratedTokenPresencePenalty(LogitsProcessor):
    def __init__(self, prompt_length, penalty=1.5):
        self.prompt_length = prompt_length
        self.penalty = penalty

    def __call__(self, input_ids, scores):
        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, input_ids[:, self.prompt_length :], True)
        return scores - seen.to(scores.dtype) * self.penalty


class Flux3ImagePromptUpsampler(PromptUpsampler):
    def __init__(self, device, dtype, text_encoders):
        self.device = device
        self.dtype = dtype
        self.text_encoders = text_encoders
        self.processor = AutoProcessor.from_pretrained(PROMPT_UPSAMPLER_MODEL_ID)
        self.model = (
            Qwen3_5ForConditionalGeneration.from_pretrained(PROMPT_UPSAMPLER_MODEL_ID, dtype=dtype)
            .eval()
            .requires_grad_(False)
        )
        with open(hf_hub_download(PROMPT_UPSAMPLER_MODEL_ID, "system_prompts.json")) as handle:
            self.system_prompts = json.load(handle)

    @torch.no_grad()
    def upsample(self, requests):
        for encoder in self.text_encoders:
            encoder.to("cpu")
        captions = []
        try:
            self.model.to(self.device)
            for request in requests:
                references = request.reference_images
                if len(references) > 10:
                    raise ValueError("FLUX.3 prompt upsampling supports at most ten reference images.")
                content = request.prompt
                if references:
                    content = []
                    pixel_budget = 1048576 // len(references)
                    for image in references:
                        scale = min(1.0, math.sqrt(pixel_budget / (image.width * image.height)))
                        if scale < 1:
                            image = image.resize(
                                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                                Image.Resampling.LANCZOS,
                            )
                        content.append({"type": "image", "image": image})
                    content.append(
                        {
                            "type": "text",
                            "text": (
                                "Reference images follow in order (the first image, the second image, ...).\n"
                                f"User request: {request.prompt}\nOutput the caption."
                            ),
                        }
                    )
                messages = [
                    {"role": "system", "content": self.system_prompts["i2i" if references else "t2i"]},
                    {"role": "user", "content": content},
                ]
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=False,
                ).to(self.device)
                length = inputs["input_ids"].shape[1]
                processors = [] if references else [GeneratedTokenPresencePenalty(length)]
                output = self.model.generate(**inputs, max_new_tokens=3584, do_sample=False, logits_processor=processors)
                captions.append(self.processor.decode(output[0, length:], skip_special_tokens=True).strip())
        finally:
            self.model.to("cpu")
        return captions
