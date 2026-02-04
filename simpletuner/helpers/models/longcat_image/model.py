import logging
import os
from typing import List, Optional

import numpy as np
import torch
from diffusers import AutoencoderKL
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from simpletuner.helpers.acceleration import (
    AccelerationBackend,
    AccelerationPreset,
    get_bitsandbytes_presets,
    get_deepspeed_presets,
    get_quanto_presets,
    get_sdnq_presets,
    get_torchao_presets,
)
from simpletuner.helpers.models.common import (
    ImageModelFoundation,
    ModelTypes,
    PipelineTypes,
    PredictionTypes,
    TextEmbedCacheKey,
    get_model_config_path,
)
from simpletuner.helpers.models.longcat_image import pack_latents, prepare_pos_ids, unpack_latents
from simpletuner.helpers.models.longcat_image.pipeline import LongCatImagePipeline
from simpletuner.helpers.models.longcat_image.pipeline_edit import LongCatImageEditPipeline
from simpletuner.helpers.models.longcat_image.transformer import LongCatImageTransformer2DModel
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.models.tae.types import ImageTAESpec
from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class LongCatImage(ImageModelFoundation):
    SUPPORTS_MUON_CLIP = True
    NAME = "LongCat-Image"
    MODEL_DESCRIPTION = "Bilingual 6B image generation and editing"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 16
    VALIDATION_PREVIEW_SPEC = ImageTAESpec(repo_id="madebyollin/taef1")

    MODEL_CLASS = LongCatImageTransformer2DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: LongCatImagePipeline,
        PipelineTypes.IMG2IMG: LongCatImageEditPipeline,
    }

    DEFAULT_MODEL_FLAVOUR = "final"
    HUGGINGFACE_PATHS = {
        "final": "meituan-longcat/LongCat-Image",
        "dev": "meituan-longcat/LongCat-Image-Dev",
        "edit": "meituan-longcat/LongCat-Image-Edit",
    }
    MODEL_LICENSE = "apache-2.0"

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        # LongCat-Image has 19 double + 38 single = 57 transformer blocks
        return 56

    @classmethod
    def get_acceleration_presets(cls) -> list[AccelerationPreset]:
        # Common settings for memory optimization presets
        _base_memory_config = {
            "base_model_precision": "no_change",
            "gradient_checkpointing": True,
        }

        return [
            # RamTorch presets (Basic tab) - 3 levels for 6B model
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="basic",
                name="RamTorch - Basic",
                description="Offloads single transformer blocks only.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~30%",
                tradeoff_speed="Increases training time by ~20%",
                tradeoff_notes="Requires 32GB+ system RAM.",
                requires_min_system_ram_gb=32,
                config={
                    **_base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "single_transformer_blocks.*",
                },
            ),
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="balanced",
                name="RamTorch - Balanced",
                description="Offloads single blocks and half of double blocks.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~45%",
                tradeoff_speed="Increases training time by ~35%",
                tradeoff_notes="Requires 48GB+ system RAM.",
                requires_min_system_ram_gb=48,
                config={
                    **_base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "single_transformer_blocks.*,transformer_blocks.10,transformer_blocks.11,transformer_blocks.12,transformer_blocks.13,transformer_blocks.14,transformer_blocks.15,transformer_blocks.16,transformer_blocks.17,transformer_blocks.18",
                },
            ),
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="aggressive",
                name="RamTorch - Aggressive",
                description="Offloads all transformer blocks to CPU RAM.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~60%",
                tradeoff_speed="Increases training time by ~50%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={
                    **_base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "transformer_blocks.*,single_transformer_blocks.*",
                },
            ),
            # Block Swap presets (Basic tab) - 3 levels for 6B model
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="conservative",
                name="Block Swap - Conservative",
                description="Swaps 18 of 57 blocks between GPU and CPU.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~25%",
                tradeoff_speed="Increases training time by ~15%",
                tradeoff_notes="Requires 32GB+ system RAM.",
                requires_min_system_ram_gb=32,
                config={**_base_memory_config, "musubi_blocks_to_swap": 18},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="balanced",
                name="Block Swap - Balanced",
                description="Swaps 28 of 57 blocks between GPU and CPU.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~40%",
                tradeoff_speed="Increases training time by ~25%",
                tradeoff_notes="Requires 48GB+ system RAM.",
                requires_min_system_ram_gb=48,
                config={**_base_memory_config, "musubi_blocks_to_swap": 28},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="aggressive",
                name="Block Swap - Aggressive",
                description="Swaps 42 of 57 blocks between GPU and CPU.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~55%",
                tradeoff_speed="Increases training time by ~40%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={**_base_memory_config, "musubi_blocks_to_swap": 42},
            ),
            # DeepSpeed presets (multi-GPU only)
            *get_deepspeed_presets(_base_memory_config),
            # SDNQ presets (works on AMD, Apple, NVIDIA)
            *get_sdnq_presets(_base_memory_config),
            # TorchAO presets (NVIDIA only)
            *get_torchao_presets(_base_memory_config),
            # Quanto presets (works on AMD, Apple, NVIDIA)
            *get_quanto_presets(_base_memory_config),
            # BitsAndBytes presets (NVIDIA only)
            *get_bitsandbytes_presets(_base_memory_config),
        ]

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Qwen-2.5 VL",
            "tokenizer": Qwen2Tokenizer,
            "tokenizer_subfolder": "tokenizer",
            "model": Qwen2_5_VLForConditionalGeneration,
            "subfolder": "text_encoder",
        },
    }

    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0", "to_qkv"]
    EDIT_FLAVOURS = frozenset({"edit"})

    def __init__(self, config: dict, accelerator):
        super().__init__(config, accelerator)
        self.vae_scale_factor = 8
        pipeline_classes = dict(self.PIPELINE_CLASSES)
        if self._is_edit_flavour():
            pipeline_classes[PipelineTypes.TEXT2IMG] = LongCatImageEditPipeline
        self.PIPELINE_CLASSES = pipeline_classes

    def _get_model_flavour(self) -> Optional[str]:
        return getattr(self.config, "model_flavour", None)

    def _load_text_processor_for_pipeline(self):
        text_processor = getattr(self, "text_processor", None)
        if text_processor is not None:
            return text_processor
        model_path = get_model_config_path(self.config.model_family, self.config.pretrained_model_name_or_path)
        text_processor = AutoProcessor.from_pretrained(model_path, subfolder="text_processor")
        self.text_processor = text_processor
        return text_processor

    def _is_edit_flavour(self) -> bool:
        flavour = self._get_model_flavour()
        return flavour in self.EDIT_FLAVOURS if flavour is not None else False

    def requires_conditioning_latents(self) -> bool:
        return self._is_edit_flavour()

    def requires_conditioning_validation_inputs(self) -> bool:
        return self._is_edit_flavour()

    def requires_validation_edit_captions(self) -> bool:
        return self._is_edit_flavour()

    def requires_text_embed_image_context(self) -> bool:
        return self._is_edit_flavour()

    def should_precompute_validation_negative_prompt(self) -> bool:
        return not self._is_edit_flavour()

    def conditioning_validation_dataset_type(self) -> bool:
        if self._is_edit_flavour():
            return "image"
        return super().conditioning_validation_dataset_type()

    def text_embed_cache_key(self) -> TextEmbedCacheKey:
        if self._is_edit_flavour():
            return TextEmbedCacheKey.DATASET_AND_FILENAME
        return super().text_embed_cache_key()

    def _create_dummy_image(self):
        from PIL import Image

        return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    def encode_dropout_caption(self, positive_prompt_embeds: dict = None):
        if not self._is_edit_flavour():
            return super().encode_dropout_caption(positive_prompt_embeds)

        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        dummy_image = self._create_dummy_image()
        prompt_embeds, text_ids = pipeline.encode_prompt(
            dummy_image,
            [""],
            device=self.accelerator.device,
            dtype=torch.float64,
        )

        return {
            "prompt_embeds": prompt_embeds[0],
            "text_ids": text_ids,
        }

    def encode_validation_negative_prompt(self, negative_prompt: str, positive_prompt_embeds: dict = None):
        if not self._is_edit_flavour():
            return super().encode_validation_negative_prompt(negative_prompt, positive_prompt_embeds)

        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        dummy_image = self._create_dummy_image()
        prompt_embeds, text_ids = pipeline.encode_prompt(
            dummy_image,
            [negative_prompt],
            device=self.accelerator.device,
            dtype=torch.float64,
        )

        return {
            "prompt_embeds": prompt_embeds[0],
            "text_ids": text_ids,
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        prompt_image_tensor = None
        if self._is_edit_flavour():
            prompt_contexts = getattr(self, "_current_prompt_contexts", None)
            if not prompt_contexts or len(prompt_contexts) != len(prompts):
                raise ValueError(
                    "LongCat edit prompt encoding requires image context for each caption, but none was provided."
                )
            prompt_image_tensor = self._prepare_prompt_image_batch(prompt_contexts, len(prompts), pipeline)
            if prompt_image_tensor is None:
                raise ValueError("Failed to resolve prompt image tensors for LongCat edit text encoding.")
            prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt_image_tensor,
                prompts,
                device=self.accelerator.device,
                dtype=torch.float64,
            )
        else:
            prompt_embeds, text_ids = pipeline.encode_prompt(prompts, device=self.accelerator.device, dtype=torch.float64)

        return prompt_embeds, text_ids

    def _prepare_prompt_image_batch(self, prompt_contexts, batch_size: int, pipeline):
        if not prompt_contexts or len(prompt_contexts) != batch_size:
            return None
        image_tensors = []
        for idx, context in enumerate(prompt_contexts):
            tensor = self._extract_prompt_image_from_context(context)
            if tensor is None:
                logger.warning(f"Failed to extract image tensor from context {idx}: {context}")
                return None
            if tensor.dim() == 4 and tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            if tensor.dim() != 3:
                raise ValueError(f"Expected conditioning tensor with shape (C, H, W); received {tensor.shape}.")
            image_tensors.append(tensor)
        pil_images = [self._tensor_to_pil(tensor) for tensor in image_tensors]
        return pil_images

    def _extract_prompt_image_from_context(self, context: dict):
        if not isinstance(context, dict):
            return None
        embed = context.get("conditioning_image_embeds")
        tensor = None
        if isinstance(embed, dict):
            tensor = self._coerce_prompt_tensor(embed.get("pixel_values"))
        elif torch.is_tensor(embed):
            tensor = self._coerce_prompt_tensor(embed)
        if tensor is not None:
            return tensor

        tensor = context.get("conditioning_pixel_values")
        tensor = self._coerce_prompt_tensor(tensor)
        if tensor is not None:
            return tensor
        return self._load_prompt_image_from_backend(context)

    def _tensor_to_pil(self, tensor: torch.Tensor | np.ndarray):
        from PIL import Image

        if isinstance(tensor, np.ndarray):
            array = tensor
            if array.ndim == 3 and array.shape[2] in (1, 3):
                if array.dtype != np.uint8:
                    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
                if array.shape[2] == 1:
                    array = np.repeat(array, 3, axis=2)
                return Image.fromarray(array)
            raise ValueError(f"Unsupported numpy image shape: {array.shape}")
        if not torch.is_tensor(tensor):
            raise ValueError(f"Unsupported prompt image type: {type(tensor)}")

        converted = tensor.detach().float().cpu()
        if converted.dim() == 4 and converted.size(0) == 1:
            converted = converted.squeeze(0)
        if converted.dim() != 3:
            raise ValueError(f"Expected conditioning tensor with shape (C, H, W); received {tuple(converted.shape)}.")
        if converted.max().item() > 1.0 or converted.min().item() < 0.0:
            converted = (converted + 1.0) / 2.0
        converted = converted.clamp_(0.0, 1.0)
        array = (converted.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(array)

    def _load_prompt_image_from_backend(self, context: dict):
        if not self._is_edit_flavour():
            return None
        image_path = context.get("image_path")
        data_backend_id = context.get("data_backend_id")
        if not image_path or not data_backend_id:
            return None
        from simpletuner.helpers.training.state_tracker import StateTracker

        backend_entry = StateTracker.get_data_backend(data_backend_id)
        if backend_entry is None:
            return None
        data_backend = backend_entry.get("data_backend")
        if data_backend is None:
            return None
        image = data_backend.read_image(image_path)
        tensor = self._convert_image_to_tensor(image)
        if tensor is None:
            return None
        return tensor.to(device=self.accelerator.device, dtype=self.config.weight_dtype)

    def _coerce_prompt_tensor(self, tensor):
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            coerced = tensor
        elif isinstance(tensor, np.ndarray):
            coerced = torch.from_numpy(tensor)
        else:
            return None
        if coerced.dim() == 4 and coerced.size(0) == 1:
            coerced = coerced.squeeze(0)
        if coerced.dim() != 3:
            return None
        return coerced.to(device=self.accelerator.device, dtype=self.config.weight_dtype)

    def _convert_image_to_tensor(self, image):
        from PIL import Image

        if isinstance(image, torch.Tensor):
            tensor = image.clone().detach()
        elif isinstance(image, Image.Image):
            array = np.array(image.convert("RGB"), copy=True)
            tensor = torch.from_numpy(array)
        elif isinstance(image, np.ndarray):
            array = image
            if array.ndim == 4:
                array = array[0]
            if array.ndim == 3 and array.shape[2] == 4:
                array = array[:, :, :3]
            tensor = torch.from_numpy(array)
        else:
            return None
        if tensor.dim() == 3 and tensor.shape[0] not in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        elif tensor.dim() == 4 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        tensor = tensor.to(dtype=torch.float32)
        if tensor.max() > 1.0 or tensor.min() < 0.0:
            tensor = tensor / 255.0
        return tensor.clamp_(0.0, 1.0)

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        prompt_embeds, text_ids = text_embedding
        return {
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
        }

    def collate_prompt_embeds(self, text_encoder_output: list) -> dict:
        """
        Collate prompt embeddings for LongCat-Image models with padding.

        For edit models, embeddings can have different sequence lengths due to
        different conditioning image sizes. This method pads all embeddings to
        the maximum sequence length in the batch.
        """
        if not text_encoder_output:
            return {}

        first_embed = text_encoder_output[0].get("prompt_embeds")
        if first_embed is None:
            return {}

        # Single sample - just ensure batch dimension
        if len(text_encoder_output) == 1:
            embed = first_embed
            if embed.dim() == 2:
                embed = embed.unsqueeze(0)
            return {"prompt_embeds": embed}

        # Normalize all embeddings to 2D [seq, hidden] for processing
        embeds = []
        for t in text_encoder_output:
            embed = t["prompt_embeds"]
            if embed.dim() == 3 and embed.shape[0] == 1:
                embed = embed.squeeze(0)
            embeds.append(embed)

        # Find max sequence length
        max_seq_len = max(e.shape[0] for e in embeds)
        hidden_dim = embeds[0].shape[-1]

        # Pad all embeddings to max length
        padded_embeds = []
        for embed in embeds:
            seq_len = embed.shape[0]
            if seq_len < max_seq_len:
                padding = torch.zeros(
                    max_seq_len - seq_len,
                    hidden_dim,
                    dtype=embed.dtype,
                    device=embed.device,
                )
                embed = torch.cat([embed, padding], dim=0)
            padded_embeds.append(embed)

        return {"prompt_embeds": torch.stack(padded_embeds, dim=0)}

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        text_ids = text_embedding.get("text_ids")

        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if text_ids is not None and text_ids.dim() == 3:
            text_ids = text_ids[0]

        return {
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        if self.config.validation_guidance is None or self.config.validation_guidance <= 1.0:
            return {}
        prompt_embeds = text_embedding["prompt_embeds"]
        text_ids = text_embedding.get("text_ids")
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if text_ids is not None and text_ids.dim() == 3:
            text_ids = text_ids[0]
        return {
            "negative_prompt_embeds": prompt_embeds,
            "negative_text_ids": text_ids,
        }

    def model_predict(self, prepared_batch):
        prompt_embeds = prepared_batch["prompt_embeds"].to(
            device=self.accelerator.device,
            dtype=self.config.base_weight_dtype if hasattr(self.config, "base_weight_dtype") else self.config.weight_dtype,
        )
        hidden_states_buffer = self._new_hidden_state_buffer()
        prompt_length = prompt_embeds.shape[1]
        timesteps = (
            torch.tensor(prepared_batch["timesteps"])
            .expand(prepared_batch["noisy_latents"].shape[0])
            .to(device=self.accelerator.device, dtype=torch.float32)
            / self.noise_schedule.config.num_train_timesteps
        )
        dtype_pos = torch.float64 if self.accelerator.device.type != "mps" else torch.float32
        text_ids = prepare_pos_ids(
            modality_id=0,
            type="text",
            start=(0, 0),
            num_token=prompt_length,
        ).to(self.accelerator.device, dtype=dtype_pos)
        img_ids = prepare_pos_ids(
            modality_id=1,
            type="image",
            start=(prompt_length, prompt_length),
            height=prepared_batch["latents"].shape[2] // 2,
            width=prepared_batch["latents"].shape[3] // 2,
        ).to(self.accelerator.device, dtype=dtype_pos)

        packed_noisy_latents = pack_latents(
            prepared_batch["noisy_latents"],
            batch_size=prepared_batch["latents"].shape[0],
            num_channels_latents=prepared_batch["latents"].shape[1],
            height=prepared_batch["latents"].shape[2],
            width=prepared_batch["latents"].shape[3],
        ).to(dtype=self.config.base_weight_dtype if hasattr(self.config, "base_weight_dtype") else self.config.weight_dtype)

        hidden_states = packed_noisy_latents
        img_ids_input = img_ids
        if self._is_edit_flavour():
            conditioning = prepared_batch.get("conditioning_latents")
            if conditioning is None:
                raise ValueError("Edit flavour requires conditioning_latents in the prepared batch.")
            packed_ref_latents = pack_latents(
                conditioning,
                batch_size=conditioning.shape[0],
                num_channels_latents=conditioning.shape[1],
                height=conditioning.shape[2],
                width=conditioning.shape[3],
            ).to(
                dtype=(
                    self.config.base_weight_dtype if hasattr(self.config, "base_weight_dtype") else self.config.weight_dtype
                )
            )
            hidden_states = torch.cat([packed_noisy_latents, packed_ref_latents], dim=1)
            ref_ids = prepare_pos_ids(
                modality_id=2,
                type="image",
                start=(prompt_length, prompt_length),
                height=conditioning.shape[2] // 2,
                width=conditioning.shape[3] // 2,
            ).to(self.accelerator.device, dtype=dtype_pos)
            img_ids_input = torch.cat([img_ids, ref_ids], dim=0)

        model_pred = self.model(
            hidden_states=hidden_states,
            timestep=timesteps,
            timestep_sign=(
                prepared_batch.get("twinflow_time_sign") if getattr(self.config, "twinflow_enabled", False) else None
            ),
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids_input,
            hidden_states_buffer=hidden_states_buffer,
            return_dict=False,
        )[0]

        if self._is_edit_flavour():
            scene_seq_len = packed_noisy_latents.shape[1]
            model_pred = model_pred[:, :scene_seq_len, :]

        crepa_hidden = None
        crepa = getattr(self, "crepa_regularizer", None)
        if crepa and crepa.enabled and hidden_states_buffer is not None:
            crepa_hidden = hidden_states_buffer.get(f"layer_{crepa.block_index}")

        return {
            "model_prediction": unpack_latents(
                model_pred,
                height=prepared_batch["latents"].shape[2] * 8,
                width=prepared_batch["latents"].shape[3] * 8,
                vae_scale_factor=16,
            ),
            "crepa_hidden_states": crepa_hidden,
            "hidden_states_buffer": hidden_states_buffer,
        }


ModelRegistry.register("longcat_image", LongCatImage)
