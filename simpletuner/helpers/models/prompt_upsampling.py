"""Image-aware prompt rewriting before text conditioning is encoded."""

import hashlib
import json
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

    def text_embed_cache_namespace(self) -> str | None:
        if self.prompt_upsampling_enabled():
            return f"prompt-upsampling:{self.PROMPT_UPSAMPLING_MODEL_ID}"
        return super().text_embed_cache_namespace()

    def text_embed_cache_context_key(self, metadata, *, prompt=None):
        if not self.prompt_upsampling_enabled() or (prompt is not None and not str(prompt).strip()):
            return None
        height, width = self._prompt_upsample_resolution()
        return prompt_upsampling_context_key(metadata, height, width)

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
