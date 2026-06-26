import logging
from typing import Optional

import numpy as np
import torch
from diffusers import AutoencoderKLQwenImage
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLModel

from simpletuner.helpers.models.common import (
    ImageModelFoundation,
    ModelTypes,
    PipelineTypes,
    PredictionTypes,
    TextEmbedCacheKey,
)
from simpletuner.helpers.models.krea2.pipeline import Krea2Pipeline
from simpletuner.helpers.models.krea2.transformer import Krea2Transformer2DModel
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.models.tae.types import VideoTAESpec
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)


class Krea2(ImageModelFoundation):
    SUPPORTS_MUON_CLIP = True
    NAME = "Krea 2"
    MODEL_DESCRIPTION = "Krea 2 flow-matching transformer"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    USES_DYNAMIC_SHIFT = True
    AUTOENCODER_CLASS = AutoencoderKLQwenImage
    AUTOENCODER_SCALING_FACTOR = 1.0
    LATENT_CHANNEL_COUNT = 16
    VALIDATION_PREVIEW_SPEC = VideoTAESpec(filename="taew2_1.pth", description="Wan 2.1 VAE compatible")

    MODEL_CLASS = Krea2Transformer2DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: Krea2Pipeline,
    }
    DEFAULT_MODEL_FLAVOUR = "raw"
    HUGGINGFACE_PATHS = {
        "raw": "krea/Krea-2-Raw",
        "turbo": "krea/Krea-2-Turbo",
    }
    MODEL_LICENSE = "other"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Qwen3VL",
            "tokenizer": AutoTokenizer,
            "tokenizer_subfolder": "tokenizer",
            "model": Qwen3VLModel,
            "subfolder": "text_encoder",
        },
    }
    PROCESSOR_CLASS = AutoProcessor
    PROCESSOR_PATH = "Qwen/Qwen3-VL-4B-Instruct"
    PROCESSOR_SUBFOLDER = None
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    FUSED_LORA_TARGET = ["to_qkv", "to_out.0"]

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        return 27

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        self.processor = None
        self.vae_scale_factor = 8

    def _uses_reference_latents(self) -> bool:
        return bool(getattr(self.config, "krea2_reference_latents", False))

    def supports_conditioning_dataset(self) -> bool:
        return True

    def requires_conditioning_dataset(self) -> bool:
        return self._uses_reference_latents()

    def requires_conditioning_validation_inputs(self) -> bool:
        return self._uses_reference_latents()

    def requires_validation_edit_captions(self) -> bool:
        return self._uses_reference_latents()

    def should_precompute_validation_negative_prompt(self) -> bool:
        return not self._uses_reference_latents()

    def text_embed_cache_key(self) -> TextEmbedCacheKey:
        if self._uses_reference_latents():
            return TextEmbedCacheKey.DATASET_AND_FILENAME
        return super().text_embed_cache_key()

    def requires_text_embed_image_context(self) -> bool:
        return self._uses_reference_latents()

    def requires_conditioning_latents(self) -> bool:
        return self._uses_reference_latents()

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        if self._uses_reference_latents() and "image" in pipeline_kwargs and "reference_image" not in pipeline_kwargs:
            pipeline_kwargs["reference_image"] = pipeline_kwargs.pop("image")
        return pipeline_kwargs

    def get_lora_target_layers(self):
        if getattr(self.config, "fuse_qkv_projections", False):
            return self.FUSED_LORA_TARGET
        return super().get_lora_target_layers()

    def pre_vae_encode_transform_sample(self, sample):
        if sample.dim() == 4:
            sample = sample.unsqueeze(2)
        return sample

    def post_vae_encode_transform_sample(self, sample):
        vae = self.get_vae()
        if vae is None:
            raise ValueError("Cannot normalize Krea 2 latents without a loaded VAE.")

        sample_latents = sample.latent_dist.sample()
        if sample_latents.dim() == 5:
            sample_latents = sample_latents.squeeze(2)

        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1)
            .to(sample_latents.device, sample_latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1).to(
            sample_latents.device, sample_latents.dtype
        )
        return (sample_latents - latents_mean) * latents_std

    def _load_processor_for_pipeline(self):
        if self.processor is not None:
            return self.processor

        processor_path = getattr(self.config, "processor_pretrained_model_name_or_path", None) or self.PROCESSOR_PATH
        processor_subfolder = getattr(self.config, "processor_subfolder", self.PROCESSOR_SUBFOLDER)
        processor_revision = getattr(self.config, "processor_revision", getattr(self.config, "revision", None))

        processor_kwargs = {"pretrained_model_name_or_path": processor_path}
        if processor_subfolder:
            processor_kwargs["subfolder"] = processor_subfolder
        if processor_revision is not None:
            processor_kwargs["revision"] = processor_revision
        if getattr(self.config, "local_files_only", False):
            processor_kwargs["local_files_only"] = True

        self.processor = self.PROCESSOR_CLASS.from_pretrained(**processor_kwargs)
        return self.processor

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        text_encoder = self.text_encoders[0]
        if text_encoder.device != self.accelerator.device:
            text_encoder.to(self.accelerator.device)

        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        prompt_contexts = getattr(self, "_current_prompt_contexts", None)
        encode_kwargs = {
            "device": self.accelerator.device,
            "num_images_per_prompt": 1,
        }
        if self.requires_text_embed_image_context():
            if not prompt_contexts or len(prompt_contexts) != len(prompts):
                raise ValueError("Krea 2 reference text encoding requires image context for each caption.")
            reference_images = self._prepare_prompt_image_batch(prompt_contexts, len(prompts))
            if reference_images is None:
                raise ValueError("Failed to resolve reference images for Krea 2 text encoding.")
            encode_kwargs["images"] = reference_images
            encode_kwargs["processor"] = self._load_processor_for_pipeline()

        return pipeline.encode_prompt(prompts, **encode_kwargs)

    def _prepare_prompt_image_batch(self, prompt_contexts, batch_size: int):
        if not prompt_contexts or len(prompt_contexts) != batch_size:
            return None
        images = []
        for idx, context in enumerate(prompt_contexts):
            extracted = self._extract_prompt_image_from_context(context)
            if extracted is None:
                logger.warning("Failed to extract Krea 2 reference image tensor from context %s: %s", idx, context)
                return None
            if isinstance(extracted, list):
                if len(extracted) != 1:
                    raise ValueError("Krea 2 reference text encoding expects exactly one reference image per caption.")
                extracted = extracted[0]
            images.append(self._tensor_to_pil(extracted))
        return images

    def _extract_prompt_image_from_context(self, context: dict):
        if not isinstance(context, dict):
            return None
        tensor = self._coerce_prompt_tensor(context.get("conditioning_pixel_values"))
        if tensor is not None:
            return tensor
        return self._load_prompt_image_from_backend(context)

    def _coerce_prompt_tensor(self, tensor):
        if tensor is None:
            return None
        if isinstance(tensor, Image.Image):
            return tensor
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if not torch.is_tensor(tensor):
            return None
        if tensor.dim() == 4 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        if tensor.dim() != 3:
            return None
        return tensor.to(device=self.accelerator.device, dtype=self.config.weight_dtype)

    def _load_prompt_image_from_backend(self, context: dict):
        image_paths = context.get("image_paths")
        data_backend_ids = context.get("data_backend_ids")
        if isinstance(image_paths, (list, tuple)) and image_paths:
            image_path = image_paths[0]
            if isinstance(data_backend_ids, (list, tuple)) and data_backend_ids:
                data_backend_id = data_backend_ids[0]
            else:
                data_backend_id = context.get("data_backend_id")
        else:
            image_path = context.get("image_path")
            data_backend_id = context.get("data_backend_id")
        if not image_path or not data_backend_id:
            return None
        backend_entry = StateTracker.get_data_backend(data_backend_id)
        if backend_entry is None:
            return None
        data_backend = backend_entry.get("data_backend")
        if data_backend is None:
            return None
        image = data_backend.read_image(image_path)
        return self._convert_image_to_tensor(image)

    def _convert_image_to_tensor(self, image):
        if isinstance(image, Image.Image):
            array = np.array(image.convert("RGB"), copy=True)
            tensor = torch.from_numpy(array)
        elif isinstance(image, np.ndarray):
            array = image[0] if image.ndim == 4 else image
            if array.ndim == 3 and array.shape[2] == 4:
                array = array[:, :, :3]
            tensor = torch.from_numpy(array)
        elif torch.is_tensor(image):
            tensor = image.clone().detach()
        else:
            return None
        if tensor.dim() == 3 and tensor.shape[0] not in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        elif tensor.dim() == 4 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        tensor = tensor.to(dtype=torch.float32)
        if tensor.max() > 1.0 or tensor.min() < 0.0:
            tensor = tensor / 255.0
        return tensor.clamp_(0.0, 1.0).to(device=self.accelerator.device, dtype=self.config.weight_dtype)

    def _tensor_to_pil(self, tensor: torch.Tensor | np.ndarray | Image.Image):
        if isinstance(tensor, Image.Image):
            return tensor
        if isinstance(tensor, np.ndarray):
            tensor = torch.from_numpy(tensor)
        if not torch.is_tensor(tensor):
            raise ValueError(f"Unsupported Krea 2 reference image type: {type(tensor)}")
        converted = tensor.detach().float().cpu()
        if converted.dim() == 4 and converted.size(0) == 1:
            converted = converted.squeeze(0)
        if converted.dim() != 3:
            raise ValueError(f"Expected Krea 2 reference tensor with shape (C, H, W); received {tuple(converted.shape)}.")
        if converted.max().item() > 1.0 or converted.min().item() < 0.0:
            converted = (converted + 1.0) / 2.0
        converted = converted.clamp_(0.0, 1.0)
        array = (converted.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        return Image.fromarray(array)

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        prompt_embeds, prompt_embeds_mask = text_embedding
        return {
            "prompt_embeds": prompt_embeds,
            "attention_masks": prompt_embeds_mask,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        attention_mask = text_embedding.get("attention_masks", None)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        prompt_embeds = text_embedding["prompt_embeds"]
        if prompt_embeds.dim() == 3:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        return {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_mask": attention_mask.to(dtype=torch.int64) if attention_mask is not None else None,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        attention_mask = text_embedding.get("attention_masks", None)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        prompt_embeds = text_embedding["prompt_embeds"]
        if prompt_embeds.dim() == 3:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        return {
            "negative_prompt_embeds": prompt_embeds,
            "negative_prompt_embeds_mask": attention_mask.to(dtype=torch.int64) if attention_mask is not None else None,
        }

    def collate_prompt_embeds(self, text_encoder_output: list) -> dict:
        if not text_encoder_output:
            return {}
        embeds = []
        masks = []
        for entry in text_encoder_output:
            embed = entry["prompt_embeds"]
            mask = entry.get("attention_masks")
            if embed.dim() == 3:
                embed = embed.unsqueeze(0)
            embeds.append(embed)
            if mask is not None:
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)
                masks.append(mask)

        max_seq_len = max(embed.shape[1] for embed in embeds)
        padded_embeds = []
        padded_masks = []
        for idx, embed in enumerate(embeds):
            if embed.shape[1] < max_seq_len:
                pad_len = max_seq_len - embed.shape[1]
                pad_shape = (embed.shape[0], pad_len, embed.shape[2], embed.shape[3])
                embed = torch.cat(
                    [embed, torch.zeros(pad_shape, dtype=embed.dtype, device=embed.device)],
                    dim=1,
                )
            padded_embeds.append(embed)
            if masks:
                mask = masks[idx]
                if mask.shape[1] < max_seq_len:
                    pad_len = max_seq_len - mask.shape[1]
                    mask = torch.cat(
                        [mask, torch.zeros((mask.shape[0], pad_len), dtype=mask.dtype, device=mask.device)],
                        dim=1,
                    )
                padded_masks.append(mask)

        return {
            "prompt_embeds": torch.cat(padded_embeds, dim=0),
            "attention_masks": torch.cat(padded_masks, dim=0) if padded_masks else None,
        }

    def _patch_size(self) -> int:
        transformer = self.unwrap_model(self.model) if getattr(self, "model", None) is not None else None
        config = getattr(transformer, "config", None)
        return int(max(getattr(config, "patch_size", 2), 1))

    def _pack_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        patch_size = self._patch_size()
        batch_size, channels, height, width = latents.shape
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(f"Krea 2 latent dimensions must be divisible by patch_size={patch_size}: {height}x{width}.")
        packed = latents.view(batch_size, channels, height // patch_size, patch_size, width // patch_size, patch_size)
        packed = packed.permute(0, 2, 4, 1, 3, 5)
        packed = packed.reshape(
            batch_size, (height // patch_size) * (width // patch_size), channels * patch_size * patch_size
        )
        return packed, height // patch_size, width // patch_size

    def _unpack_latents(self, latents: torch.Tensor, latent_height: int, latent_width: int) -> torch.Tensor:
        patch_size = self._patch_size()
        batch_size, _, channels = latents.shape
        unpacked = latents.view(
            batch_size,
            latent_height // patch_size,
            latent_width // patch_size,
            channels // (patch_size * patch_size),
            patch_size,
            patch_size,
        )
        unpacked = unpacked.permute(0, 3, 1, 4, 2, 5)
        return unpacked.reshape(batch_size, channels // (patch_size * patch_size), latent_height, latent_width)

    @staticmethod
    def _position_ids_for_grids(text_seq_len: int, grids: list[tuple[int, int]], device: torch.device):
        text_ids = torch.zeros(text_seq_len, 3, device=device)
        image_ids = []
        for grid_height, grid_width in grids:
            ids = torch.zeros(grid_height, grid_width, 3, device=device)
            ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
            ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
            image_ids.append(ids.reshape(grid_height * grid_width, 3))
        if image_ids:
            return torch.cat([text_ids, *image_ids], dim=0)
        return text_ids

    def _prepare_model_predict_timesteps(self, raw_timesteps, batch_size: int) -> torch.Tensor:
        if not torch.is_tensor(raw_timesteps):
            timesteps = torch.tensor(raw_timesteps, device=self.accelerator.device, dtype=torch.float32)
        else:
            timesteps = raw_timesteps.to(device=self.accelerator.device, dtype=torch.float32)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(batch_size)
        elif timesteps.ndim == 1:
            if timesteps.shape[0] == 1:
                timesteps = timesteps.expand(batch_size)
            elif timesteps.shape[0] != batch_size:
                raise ValueError(
                    f"Krea 2 expected 1 timestep or {batch_size} per-batch timesteps, got {timesteps.shape[0]}."
                )
        else:
            raise ValueError(f"Krea 2 expected scalar or 1D timesteps, got shape {tuple(timesteps.shape)}.")
        return timesteps / 1000.0

    def _prepare_reference_latents(self, prepared_batch: dict, batch_size: int, channels: int, height: int, width: int):
        reference_latents = prepared_batch.get("conditioning_latents")
        if isinstance(reference_latents, list):
            reference_latents = reference_latents[0] if reference_latents else None
        if reference_latents is None:
            raise ValueError("Krea 2 reference-latent training requires conditioning_latents in the batch.")
        if reference_latents.dim() == 5:
            if reference_latents.shape[2] != 1:
                raise ValueError(f"Krea 2 reference latents must have a single frame, got {tuple(reference_latents.shape)}.")
            reference_latents = reference_latents.squeeze(2)
        if reference_latents.shape != (batch_size, channels, height, width):
            raise ValueError(
                "Krea 2 reference latents must match target latent shape. "
                f"Got reference {tuple(reference_latents.shape)} vs target {(batch_size, channels, height, width)}."
            )
        return reference_latents.to(device=self.accelerator.device, dtype=self.config.weight_dtype)

    def model_predict(self, prepared_batch):
        latent_model_input = prepared_batch["noisy_latents"]
        target_latents = prepared_batch["latents"]
        target_ndim = target_latents.dim()

        if latent_model_input.dim() == 5:
            if latent_model_input.shape[2] != 1:
                raise ValueError(
                    f"Krea 2 image training expects a single latent frame, got {tuple(latent_model_input.shape)}."
                )
            latent_model_input = latent_model_input.squeeze(2)
        batch_size, channels, latent_height, latent_width = latent_model_input.shape

        hidden_states, grid_height, grid_width = self._pack_latents(
            latent_model_input.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        )
        target_token_count = hidden_states.shape[1]
        grids = [(grid_height, grid_width)]

        if self._uses_reference_latents():
            reference_latents = self._prepare_reference_latents(
                prepared_batch,
                batch_size=batch_size,
                channels=channels,
                height=latent_height,
                width=latent_width,
            )
            packed_reference, ref_grid_height, ref_grid_width = self._pack_latents(reference_latents)
            hidden_states = torch.cat([hidden_states, packed_reference], dim=1)
            grids.append((ref_grid_height, ref_grid_width))

        prompt_embeds = prepared_batch["prompt_embeds"].to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        prompt_embeds_mask = prepared_batch.get("encoder_attention_mask")
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask.to(self.accelerator.device, dtype=torch.int64)
            if prompt_embeds_mask.dim() == 3 and prompt_embeds_mask.size(1) == 1:
                prompt_embeds_mask = prompt_embeds_mask.squeeze(1)

        timesteps = self._prepare_model_predict_timesteps(prepared_batch["timesteps"], batch_size)
        position_ids = self._position_ids_for_grids(prompt_embeds.shape[1], grids, self.accelerator.device)

        noise_pred = self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=prompt_embeds,
            timestep=timesteps,
            position_ids=position_ids,
            encoder_attention_mask=prompt_embeds_mask,
            return_dict=False,
        )[0]
        noise_pred = noise_pred[:, :target_token_count]
        noise_pred = self._unpack_latents(noise_pred, latent_height, latent_width)

        if target_ndim == 5:
            noise_pred = noise_pred.unsqueeze(2)
        return {"model_prediction": noise_pred}


ModelRegistry.register("krea2", Krea2)
