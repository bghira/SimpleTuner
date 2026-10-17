"""Image-aware prompt rewriting before text conditioning is encoded."""

import hashlib
import json
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from PIL import Image

from simpletuner.helpers.training.state_tracker import StateTracker


@dataclass(frozen=True)
class PromptUpsamplingRequest:
    prompt: str
    height: int
    width: int
    reference_images: tuple[Image.Image, ...] = ()


class PromptUpsampler(ABC):
    @abstractmethod
    def upsample(self, requests: list[PromptUpsamplingRequest]) -> list[str]:
        """Return one rewritten prompt for each image-generation request."""


def reference_images_from_context(context: dict) -> tuple[Image.Image, ...]:
    images = context.get("reference_images")
    if images is not None:
        images = [images] if isinstance(images, Image.Image) else images
        return tuple(image.convert("RGB") for image in images)

    pixels = context.get("conditioning_pixel_values")
    if pixels is None:
        paths = context.get("image_paths")
        if paths:
            backend_ids = context["data_backend_ids"]
            if len(paths) != len(backend_ids):
                raise ValueError("Reference image paths and backend IDs must have equal lengths.")
            return tuple(
                StateTracker.get_data_backend(backend_id)["data_backend"].read_image(path).convert("RGB")
                for path, backend_id in zip(paths, backend_ids)
            )

        return ()
    if torch.is_tensor(pixels):
        pixels = [pixels] if pixels.ndim == 3 else list(pixels.unbind(0))
    low, high = context.get("conditioning_pixel_values_range", (0.0, 1.0))
    images = []
    for value in pixels:
        if value.ndim != 3 or value.shape[0] != 3:
            raise ValueError("Reference pixels must have shape [3, height, width].")
        array = ((value.detach().float().cpu() - low) / (high - low)).clamp(0, 1)
        array = (array.permute(1, 2, 0) * 255).round().to(torch.uint8).numpy()
        images.append(Image.fromarray(array))
    return tuple(images)


def resize_reference_images(images, max_total_pixels=1048576):
    if not images:
        return ()
    pixel_budget = max_total_pixels // len(images)
    output = []
    for image in images:
        scale = min(1.0, math.sqrt(pixel_budget / (image.width * image.height)))
        if scale < 1:
            image = image.resize(
                (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        output.append(image)
    return tuple(output)


def prompt_upsampling_context_key(metadata, height, width):
    context = {
        "image_paths": metadata.get("image_paths", []),
        "data_backend_ids": metadata.get("data_backend_ids", []),
        "height": metadata.get("height", height),
        "width": metadata.get("width", width),
    }
    if "conditioning_pixel_values" in metadata or "reference_images" in metadata:
        references = []
        for image in reference_images_from_context(metadata):
            references.append((image.size, hashlib.sha256(image.tobytes()).hexdigest()))
        context.pop("image_paths")
        context.pop("data_backend_ids")
        context["references"] = references
    return hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()


def upsample_prompt_batch(upsampler, prompts, *, height, width, prompt_contexts=None):
    if prompt_contexts is not None and len(prompt_contexts) != len(prompts):
        raise ValueError("Prompt context must contain one entry per prompt.")
    active = [index for index, prompt in enumerate(prompts) if str(prompt).strip()]
    if not active:
        return list(prompts)
    requests = []
    for index in active:
        context = (prompt_contexts[index] or {}) if prompt_contexts is not None else {}
        requests.append(
            PromptUpsamplingRequest(
                prompt=str(prompts[index]),
                height=int(context.get("height", height)),
                width=int(context.get("width", width)),
                reference_images=reference_images_from_context(context),
            )
        )
    rewritten = upsampler.upsample(requests)
    if len(rewritten) != len(requests):
        raise ValueError("A prompt upsampler must return one caption per request.")
    output = list(prompts)
    for index, caption in zip(active, rewritten):
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("A prompt upsampler returned an empty or invalid caption.")
        output[index] = caption
    return output


class ImagePromptUpsamplingMixin:
    PROMPT_UPSAMPLING_CONFIG_KEY = None
    PROMPT_UPSAMPLING_MODEL_ID = None

    def prompt_upsampling_enabled(self) -> bool:
        key = self.PROMPT_UPSAMPLING_CONFIG_KEY
        return bool(key and getattr(self.config, key, False))

    def text_embed_cache_key(self):
        from simpletuner.helpers.models.common import TextEmbedCacheKey

        if self.prompt_upsampling_enabled():
            return TextEmbedCacheKey.DATASET_AND_FILENAME
        return super().text_embed_cache_key()

    def get_prompt_upsampler_model_id(self):
        return self.PROMPT_UPSAMPLING_MODEL_ID

    def text_embed_cache_namespace(self) -> str | None:
        namespace = super().text_embed_cache_namespace()
        if self.prompt_upsampling_enabled():
            upsampler = f"prompt-upsampling:{self.get_prompt_upsampler_model_id()}"
            return f"{upsampler}:{namespace}" if namespace else upsampler
        return namespace

    def text_embed_cache_context_key(self, metadata, *, prompt=None):
        if not self.prompt_upsampling_enabled() or (prompt is not None and not str(prompt).strip()):
            return None
        height, width = self._prompt_upsample_resolution()
        return prompt_upsampling_context_key(metadata, height, width)

    def requires_text_embed_resolution_context(self) -> bool:
        converter = self.get_json_schema_converter()
        return self.prompt_upsampling_enabled() and converter is not None and converter.USES_PIXEL_COORDINATES

    def validation_prompt_image_context(self, original_image, prepared_image):
        if self.prompt_upsampling_enabled():
            return original_image
        return super().validation_prompt_image_context(original_image, prepared_image)

    def text_embed_cache_metadata_for_filepath(self, *, init_backend, image_path, **kwargs):
        if not self.prompt_upsampling_enabled():
            return {}
        metadata = init_backend["metadata_backend"].get_metadata_by_filepath(image_path)
        size = metadata.get("target_size")
        return {"width": int(size[0]), "height": int(size[1])} if size else {}

    def text_embed_cache_metadata_for_sample(self, *, example, **kwargs):
        if not self.prompt_upsampling_enabled():
            return {}
        size = example.get("target_size")
        return {"width": int(size[0]), "height": int(size[1])} if size else {}

    def requires_text_embed_image_context(self) -> bool:
        return self.prompt_upsampling_enabled() or super().requires_text_embed_image_context()

    def requires_text_embed_conditioning_pixels(self) -> bool:
        return not self.prompt_upsampling_enabled() and super().requires_text_embed_conditioning_pixels()

    def create_prompt_upsampler(self):
        raise NotImplementedError("This image model must provide its prompt upsampler.")

    def _prompt_upsample_resolution(self) -> tuple[int, int]:
        resolution = (
            getattr(self.config, "resolution", None)
            or getattr(self.config, "validation_resolution", None)
            or getattr(self.config, "maximum_image_size", None)
            or "1024x1024"
        )
        if isinstance(resolution, (tuple, list)) and len(resolution) >= 2:
            return int(resolution[0]), int(resolution[1])
        if isinstance(resolution, int):
            return resolution, resolution
        if isinstance(resolution, str):
            parts = resolution.lower().replace(",", "x").split("x")
            if len(parts) >= 2:
                return int(parts[0].strip()), int(parts[1].strip())
            return int(resolution), int(resolution)
        return 1024, 1024

    @torch.no_grad()
    def encode_text_batch(self, text_batch, is_negative_prompt=False, prompt_contexts=None):
        if self.prompt_upsampling_enabled() and not is_negative_prompt and any(str(p).strip() for p in text_batch):
            if not self.text_encoders:
                self.load_text_encoder()
            if getattr(self, "_prompt_upsampler", None) is None:
                self._prompt_upsampler = self.create_prompt_upsampler()
            height, width = self._prompt_upsample_resolution()
            text_batch = upsample_prompt_batch(
                self._prompt_upsampler,
                text_batch,
                height=height,
                width=width,
                prompt_contexts=prompt_contexts,
            )
        return super().encode_text_batch(text_batch, is_negative_prompt, prompt_contexts)

    def unload_text_encoder(self):
        self._prompt_upsampler = None
        super().unload_text_encoder()


class TransformersJSONPromptUpsampler(PromptUpsampler):
    def __init__(self, *, model_id, converter, device, dtype, text_encoders):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.converter = converter
        self.device = device
        self.text_encoders = text_encoders
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).eval().requires_grad_(False)

    @torch.no_grad()
    def upsample(self, requests):
        for encoder in self.text_encoders:
            encoder.to("cpu")
        captions = []
        try:
            self.model.to(self.device)
            for request in requests:
                if request.prompt.lstrip().startswith("{"):
                    captions.append(self.converter.convert(request.prompt, height=request.height, width=request.width))
                    continue
                instruction = (
                    self.converter.generation_instructions(height=request.height, width=request.width)
                    + "Return exactly one JSON object matching this schema, without Markdown: "
                    + json.dumps(self.converter.json_schema())
                )
                content = [{"type": "image", "image": image} for image in resize_reference_images(request.reference_images)]
                content.append({"type": "text", "text": instruction + "\nOriginal caption: " + request.prompt})
                inputs = self.processor.apply_chat_template(
                    [{"role": "user", "content": content}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=True,
                ).to(self.device)
                length = inputs["input_ids"].shape[1]
                output = self.model.generate(**inputs, max_new_tokens=16384, do_sample=False)
                tokens = output[0, length:]
                end_thinking = self.processor.tokenizer.get_vocab().get("</think>")
                if end_thinking is not None:
                    positions = (tokens == end_thinking).nonzero(as_tuple=True)[0]
                    if positions.numel():
                        tokens = tokens[int(positions[-1]) + 1 :]
                text = self.processor.decode(tokens, skip_special_tokens=True).strip()
                caption = json.loads(text)
                if not isinstance(caption, dict):
                    raise ValueError("A JSON prompt upsampler must return an object.")
                captions.append(
                    self.converter.convert(self.converter.serialize(caption), height=request.height, width=request.width)
                )
        finally:
            self.model.to("cpu")
        return captions
