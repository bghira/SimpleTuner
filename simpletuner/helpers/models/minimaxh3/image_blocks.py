"""Image-specific presentation and decoding over the shared H3 denoising blocks."""

import math

import numpy as np
import torch
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam
from PIL import Image, ImageOps

from .before_denoise import MiniMaxH3PrepareLayoutStep, _clear_negative_layout_state, _set_layout_state
from .before_encoder import MiniMaxH3SetupStep
from .decoders import MiniMaxH3VideoDecodeStep
from .encoders import MiniMaxH3KeyframeVaeEncoderStep
from .modular_blocks_minimax_h3 import MiniMaxH3AutoKeyframeVaeEncoderStep, MiniMaxH3Blocks
from .packing import (
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MiniMaxH3PackedSequence,
    _spatial_position_grid,
    keyframe_condition_noise,
    patchify_video_latents,
)


def resize_reference(image, minimum, maximum):
    image = ImageOps.exif_transpose(image).convert("RGB")
    short = min(image.size)
    scale = min(max(short, minimum), maximum) / short
    size = tuple(max(32, round(dim * scale / 32) * 32) for dim in image.size)
    return image.resize(size, Image.Resampling.LANCZOS) if size != image.size else image


def build_image_layout(text_token_tags, reference_grids, latent_height, latent_width, patch_size):
    if text_token_tags.ndim == 2:
        text_token_tags = text_token_tags[0]
    text_count = len(text_token_tags)
    patch_h, patch_w = patch_size[1:]
    grids = [*reference_grids, (latent_height, latent_width)]
    row_counts = [(height // patch_h) * (width // patch_w) for height, width in grids]
    length = text_count + sum(row_counts)
    position_ids = torch.zeros(length, 3, dtype=torch.float64)
    position_ids[:text_count, 0] = torch.arange(text_count, dtype=torch.float64)
    cursor = text_count
    for index, ((height, width), count) in enumerate(zip(grids, row_counts)):
        sqrt_area = math.sqrt(height * width)
        h_grid = _spatial_position_grid(height, patch_h, sqrt_area)
        w_grid = _spatial_position_grid(width, patch_w, sqrt_area)
        frame_grid = torch.stack(torch.meshgrid(h_grid, w_grid, indexing="ij"), dim=-1).reshape(-1, 2)
        position_ids[cursor : cursor + count, 0] = text_count + index
        position_ids[cursor : cursor + count, 1:] = frame_grid
        cursor += count
    tags = torch.zeros(length, dtype=torch.long)
    tags[:text_count] = text_token_tags
    return MiniMaxH3PackedSequence(
        length,
        position_ids,
        tags,
        torch.arange(text_count, length),
        torch.empty(0, dtype=torch.long),
        torch.arange(text_count),
        sum(row_counts[:-1]),
        0,
    )


class MiniMaxH3ImageSetupStep(MiniMaxH3SetupStep):
    @property
    def inputs(self):
        return [
            InputParam("image", type_hint=Image.Image | list[Image.Image]),
            InputParam("height", type_hint=int, default=2048),
            InputParam("width", type_hint=int, default=2048),
            InputParam("num_frames", type_hint=int, default=1),
        ]

    @property
    def intermediate_outputs(self):
        return [*super().intermediate_outputs, OutputParam("reference_images", type_hint=list)]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        if block_state.num_frames != 1:
            raise ValueError("MiniMax-H3-Image generates exactly one frame.")
        if block_state.height <= 0 or block_state.width <= 0 or block_state.height % 32 or block_state.width % 32:
            raise ValueError("MiniMax-H3-Image canvas dimensions must be positive multiples of 32.")
        images = block_state.image or []
        images = [images] if isinstance(images, Image.Image) else images
        if len(images) > 16:
            raise ValueError("MiniMax-H3-Image supports at most sixteen reference images.")
        block_state.reference_images = [resize_reference(image, 1024, 2048) for image in images]
        block_state.keyframes = [resize_reference(image, 1024, 1024) for image in images]
        ratio = components.vae_spatial_compression_ratio
        block_state.num_latent_frames = 1
        block_state.latent_height, block_state.latent_width = block_state.height // ratio, block_state.width // ratio
        block_state.num_audio_latents = 0
        block_state.keyframe_anchors = ()
        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3ImageReferenceEncoderStep(MiniMaxH3KeyframeVaeEncoderStep):
    @property
    def inputs(self):
        return [*super().inputs, InputParam("reference_images", type_hint=list, required=True)]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        device = components._execution_device
        mean = torch.tensor(components.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(components.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        pixel_mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=device).view(1, 3, 1, 1)
        pixel_std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=device).view(1, 3, 1, 1)
        rows, shapes = [], []
        for image in block_state.reference_images:
            pixels = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None].float() / 255
            latents = components.vae.encode((pixels - pixel_mean) / pixel_std).latent_dist.mode().unsqueeze(2)
            rows.append(patchify_video_latents((latents - mean) / std, components.patch_size))
            shapes.append(tuple(latents.shape[-3:]))
        condition = torch.cat(rows)
        noise = keyframe_condition_noise(
            tuple(shapes),
            components.patch_size,
            components.vae_latent_channels,
            generator=block_state.generator,
            device=device,
        )
        block_state.condition_latents = components.scheduler.scale_noise(condition, MINIMAX_H3_KEYFRAME_NOISE_AUG, noise)
        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3ImageAutoReferenceEncoderStep(MiniMaxH3AutoKeyframeVaeEncoderStep):
    block_classes = [MiniMaxH3ImageReferenceEncoderStep]
    block_trigger_inputs = ["image"]


class MiniMaxH3ImagePrepareLayoutStep(MiniMaxH3PrepareLayoutStep):
    @property
    def inputs(self):
        return [*super().inputs, InputParam("reference_images", type_hint=list, default=[])]

    @torch.no_grad()
    def __call__(self, components, state):
        block_state = self.get_block_state(state)
        ratio = components.vae_spatial_compression_ratio
        grids = [(image.height // ratio, image.width // ratio) for image in block_state.reference_images]
        layout = build_image_layout(
            block_state.text_token_tags, grids, block_state.latent_height, block_state.latent_width, components.patch_size
        )
        _set_layout_state(block_state, layout, components._execution_device)
        _clear_negative_layout_state(block_state)
        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3ImageDecodeStep(MiniMaxH3VideoDecodeStep):
    @torch.no_grad()
    def __call__(self, components, state):
        components, state = super().__call__(components, state)
        if state.get("output_type") == "pil":
            state.set("videos", [frames[0] for frames in state.get("videos")])
        return components, state


class MiniMaxH3ImageBlocks(MiniMaxH3Blocks):
    block_classes = [MiniMaxH3ImageSetupStep, *MiniMaxH3Blocks.block_classes[1:]]
    block_classes[2] = MiniMaxH3ImageAutoReferenceEncoderStep
    block_classes[3] = MiniMaxH3ImagePrepareLayoutStep
    block_classes[-1] = MiniMaxH3ImageDecodeStep
    _workflow_map = {"t2i": {"prompt": True}, "edit": {"prompt": True, "image": True}}
