import contextlib
import functools
import logging
import math
import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLQwenImage, QwenImagePipeline
from diffusers.models.attention_processor import Attention
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer, Qwen2VLProcessor

from simpletuner.helpers.models.common import (
    ImageModelFoundation,
    ModelTypes,
    PipelineTypes,
    PredictionTypes,
    TextEmbedCacheKey,
)
from simpletuner.helpers.models.qwen_image.pipeline import QwenImageEditPipeline
from simpletuner.helpers.models.qwen_image.pipeline_edit_plus import (
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
    QwenImageEditPlusPipeline,
)
from simpletuner.helpers.models.qwen_image.transformer import QwenImageTransformer2DModel
from simpletuner.helpers.models.tae.types import VideoTAESpec
from simpletuner.helpers.musubi_block_swap import apply_musubi_pretrained_defaults
from simpletuner.helpers.training.multi_process import _get_rank
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class QwenImage(ImageModelFoundation):
    SUPPORTS_MUON_CLIP = True
    NAME = "Qwen-Image"
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKLQwenImage
    AUTOENCODER_SCALING_FACTOR = 1.0
    LATENT_CHANNEL_COUNT = 16
    VALIDATION_PREVIEW_SPEC = VideoTAESpec(filename="taew2_1.pth", description="Wan 2.1 VAE compatible")

    MODEL_CLASS = QwenImageTransformer2DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: QwenImagePipeline,
    }
    EDIT_PIPELINE_CLASS = QwenImageEditPipeline
    EDIT_PLUS_PIPELINE_CLASS = QwenImageEditPlusPipeline
    EDIT_V1_FLAVOURS = frozenset({"edit-v1"})
    EDIT_V2_FLAVOURS = frozenset({"edit-v2"})

    # Default model flavor
    DEFAULT_MODEL_FLAVOUR = "v1.0"
    HUGGINGFACE_PATHS = {
        "v1.0": "Qwen/Qwen-Image",
        "edit-v1": "Qwen/Qwen-Image-Edit",
        "edit-v2": "Qwen/Qwen-Image-Edit-2509",
    }
    MODEL_LICENSE = "other"

    # Qwen Image uses a different text encoder configuration
    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Qwen2.5-VL",
            "tokenizer": Qwen2Tokenizer,
            "tokenizer_subfolder": "tokenizer",
            "model": Qwen2_5_VLForConditionalGeneration,
            "subfolder": "text_encoder",
        },
    }
    PROCESSOR_CLASS = Qwen2VLProcessor
    PROCESSOR_SUBFOLDER = "processor"

    # LoRA configuration
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]

    def __init__(self, config: dict, accelerator):
        super().__init__(config, accelerator)
        self.vae_scale_factor = 8
        pipeline_classes = dict(self.PIPELINE_CLASSES)
        if self._is_edit_v1_flavour():
            pipeline_classes[PipelineTypes.TEXT2IMG] = self.EDIT_PIPELINE_CLASS
        elif self._is_edit_v2_flavour():
            pipeline_classes[PipelineTypes.TEXT2IMG] = self.EDIT_PLUS_PIPELINE_CLASS
        self.PIPELINE_CLASSES = pipeline_classes
        self._conditioning_image_embedder = None
        self._conditioning_processor = None
        self.processor = None

    def _load_processor_for_pipeline(self):
        if self.processor is not None:
            return self.processor

        processor_cls = getattr(self, "PROCESSOR_CLASS", None)
        if processor_cls is None:
            return None

        processor_path = getattr(self.config, "processor_pretrained_model_name_or_path", None) or self._model_config_path()
        processor_subfolder = getattr(self.config, "processor_subfolder", self.PROCESSOR_SUBFOLDER)
        processor_revision = getattr(self.config, "processor_revision", getattr(self.config, "revision", None))

        processor_kwargs = {"pretrained_model_name_or_path": processor_path}
        if processor_subfolder:
            processor_kwargs["subfolder"] = processor_subfolder
        if processor_revision is not None:
            processor_kwargs["revision"] = processor_revision

        self.processor = processor_cls.from_pretrained(**processor_kwargs)
        return self.processor

    @contextlib.contextmanager
    def _force_packed_transformer_output(self, transformer):
        original_untokenize = getattr(transformer, "_untokenize_hidden_states", None)
        patched = False
        if callable(original_untokenize):

            def passthrough(hidden_states, *unused_args, **unused_kwargs):
                return hidden_states

            transformer._untokenize_hidden_states = passthrough
            patched = True

        try:
            yield
        finally:
            if patched:
                transformer._untokenize_hidden_states = original_untokenize

    def get_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True):
        pipeline = super().get_pipeline(pipeline_type=pipeline_type, load_base_model=load_base_model)
        if pipeline is None:
            return None

        return pipeline

    def setup_training_noise_schedule(self):
        """
        Loads the noise schedule for Qwen Image (flow matching).
        """
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": 0.5,
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": 0.9,
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": 0.02,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False,
        }

        self.noise_schedule = FlowMatchEulerDiscreteScheduler(**scheduler_config)
        self.config.prediction_type = "flow_matching"

        return self.config, self.noise_schedule

    def _get_model_flavour(self) -> Optional[str]:
        return getattr(self.config, "model_flavour", None)

    @staticmethod
    def _resolve_flavour_from_source(source) -> Optional[str]:
        config = getattr(source, "config", source)
        if config is None:
            return None
        flavour = getattr(config, "model_flavour", None)
        if flavour is None and isinstance(config, dict):
            flavour = config.get("model_flavour")
        return flavour

    @classmethod
    def _is_edit_v1_config(cls, source) -> bool:
        flavour = cls._resolve_flavour_from_source(source)
        return flavour in cls.EDIT_V1_FLAVOURS if flavour is not None else False

    @classmethod
    def _is_edit_v2_config(cls, source) -> bool:
        flavour = cls._resolve_flavour_from_source(source)
        return flavour in cls.EDIT_V2_FLAVOURS if flavour is not None else False

    @classmethod
    def _is_edit_config(cls, source) -> bool:
        return cls._is_edit_v1_config(source) or cls._is_edit_v2_config(source)

    def _is_edit_v1_flavour(self) -> bool:
        return QwenImage._is_edit_v1_config(self)

    def _is_edit_v2_flavour(self) -> bool:
        return QwenImage._is_edit_v2_config(self)

    def _is_edit_flavour(self) -> bool:
        return QwenImage._is_edit_config(self)

    def requires_conditioning_latents(self) -> bool:
        # Edit flavours rely on conditioning latents (masked regions) alongside the base inputs.
        return QwenImage._is_edit_config(self)

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode prompts using Qwen's text encoder.

        Args:
            prompts: List of text prompts to encode
            is_negative_prompt: Whether these are negative prompts

        Returns:
            Tuple of (prompt_embeds, prompt_embeds_mask)
        """
        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        text_encoder = self.text_encoders[0]
        tokenizer = self.tokenizers[0]

        # Move to device if needed
        if text_encoder.device != self.accelerator.device:
            text_encoder.to(self.accelerator.device)

        # Get the pipeline for encoding
        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)

        prompt_image_tensor = None
        prompt_contexts = getattr(self, "_current_prompt_contexts", None)
        if self.requires_text_embed_image_context():
            if not prompt_contexts or len(prompt_contexts) != len(prompts):
                raise ValueError(
                    "Qwen edit text encoding requires prompt image context for each caption, but none was provided."
                )
            prompt_image_tensor = self._prepare_prompt_image_batch(prompt_contexts, len(prompts), pipeline)
            if prompt_image_tensor is None:
                raise ValueError("Failed to resolve prompt image tensors for Qwen edit text encoding.")

        # Use pipeline's encode_prompt method
        encode_kwargs = {"device": self.accelerator.device, "num_images_per_prompt": 1}
        if prompt_image_tensor is not None:
            encode_kwargs["image"] = prompt_image_tensor

        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(prompts, **encode_kwargs)

        return prompt_embeds, prompt_embeds_mask

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
            logger.debug(f"Prompt image {idx} tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
            image_tensors.append(tensor)
        pil_images = [self._tensor_to_pil(tensor) for tensor in image_tensors]
        logger.debug(
            f"Converted {len(pil_images)} tensors to PIL images: {[img.size if isinstance(img, Image.Image) else type(img) for img in pil_images]}"
        )
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

    def _tensor_to_pil(self, tensor: torch.Tensor | np.ndarray | Image.Image):
        if isinstance(tensor, Image.Image):
            return tensor
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

    def _batch_tensors_to_pil(self, tensor_batch: torch.Tensor) -> List[Image.Image]:
        if not torch.is_tensor(tensor_batch):
            raise ValueError(f"Unsupported batch tensor type: {type(tensor_batch)}")
        if tensor_batch.dim() == 3:
            tensor_list = [tensor_batch]
        elif tensor_batch.dim() == 4:
            tensor_list = [tensor_batch[i] for i in range(tensor_batch.shape[0])]
        else:
            raise ValueError(f"Unexpected conditioning tensor rank {tensor_batch.dim()} for prompt image conversion.")
        return [self._tensor_to_pil(entry) for entry in tensor_list]

    def _load_prompt_image_from_backend(self, context: dict):
        if not self._is_edit_v1_flavour():
            return None
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
        logger.debug(
            f"Loaded prompt image from backend: path={image_path}, type={type(image)}, size={image.size if isinstance(image, Image.Image) else (image.shape if hasattr(image, 'shape') else 'unknown')}"
        )
        tensor = self._convert_image_to_tensor(image)
        if tensor is None:
            return None
        final_tensor = tensor.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        logger.debug(f"Converted image to tensor: shape={final_tensor.shape}, dtype={final_tensor.dtype}")
        return final_tensor

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
        """
        Format the text embeddings for Qwen Image.

        Args:
            text_embedding: The embedding tuple from _encode_prompts

        Returns:
            Dictionary with formatted embeddings
        """
        prompt_embeds, prompt_embeds_mask = text_embedding

        return {
            "prompt_embeds": prompt_embeds,
            "attention_masks": prompt_embeds_mask,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        """
        Convert text embeddings for pipeline use.
        """
        attention_mask = text_embedding.get("attention_masks", None)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        return {
            "prompt_embeds": (
                text_embedding["prompt_embeds"].unsqueeze(0)
                if text_embedding["prompt_embeds"].dim() == 2
                else text_embedding["prompt_embeds"]
            ),
            "prompt_embeds_mask": (attention_mask.to(dtype=torch.int64) if attention_mask is not None else None),
        }

    def collate_prompt_embeds(self, text_encoder_output: list) -> dict:
        """
        Collate prompt embeddings for Qwen models.

        Qwen's cached embeddings already include a batch dimension, so we need to
        concatenate along batch dimension instead of stacking.

        Ensures returned tensors always have batch dimension even when batch size is 1.
        """
        if not text_encoder_output:
            return {}

        # Check if embeddings already have batch dimension
        first_embed = text_encoder_output[0].get("prompt_embeds")
        first_mask = text_encoder_output[0].get("attention_masks")

        if first_embed is None:
            return {}

        # If batch size is 1, ensure tensors have batch dimension
        if len(text_encoder_output) == 1:
            # Ensure embeddings have batch dimension [batch_size, seq_len, hidden_dim]
            if first_embed.dim() == 2:
                first_embed = first_embed.unsqueeze(0)

            # Ensure mask has batch dimension [batch_size, seq_len]
            if first_mask is not None and first_mask.dim() == 1:
                first_mask = first_mask.unsqueeze(0)

            return {
                "prompt_embeds": first_embed,
                "attention_masks": first_mask,
            }

        # For multiple samples, concatenate along batch dimension (dim=0)
        # First ensure all embeddings and masks have batch dimension
        embeds = []
        masks = []

        for t in text_encoder_output:
            embed = t["prompt_embeds"]
            mask = t.get("attention_masks")

            # Add batch dimension if missing
            if embed.dim() == 2:
                embed = embed.unsqueeze(0)
            embeds.append(embed)

            if mask is not None:
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)
                masks.append(mask)

        return {
            "prompt_embeds": torch.cat(embeds, dim=0),
            "attention_masks": torch.cat(masks, dim=0) if masks else None,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        """
        Convert negative text embeddings for pipeline use.
        """
        attention_mask = text_embedding.get("attention_masks", None)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        result = {
            "negative_prompt_embeds": (
                text_embedding["prompt_embeds"].unsqueeze(0)
                if text_embedding["prompt_embeds"].dim() == 2
                else text_embedding["prompt_embeds"]
            ),
            "negative_prompt_embeds_mask": (attention_mask.to(dtype=torch.int64) if attention_mask is not None else None),
        }

        # Map validation_guidance_real to true_cfg_scale for Qwen pipeline
        if self.config.validation_guidance_real is not None and self.config.validation_guidance_real > 1.0:
            result["true_cfg_scale"] = float(self.config.validation_guidance_real)

        return result

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        """
        Update pipeline kwargs to ensure proper parameter mapping for Qwen pipelines.
        """
        # Map guidance_scale_real to true_cfg_scale for Qwen pipeline
        if "guidance_scale_real" in pipeline_kwargs:
            pipeline_kwargs["true_cfg_scale"] = pipeline_kwargs.pop("guidance_scale_real")

        # If true_cfg_scale is not set but validation_guidance_real is configured, use it
        if "true_cfg_scale" not in pipeline_kwargs and self.config.validation_guidance_real is not None:
            if self.config.validation_guidance_real > 1.0:
                pipeline_kwargs["true_cfg_scale"] = float(self.config.validation_guidance_real)

        return pipeline_kwargs

    def requires_conditioning_dataset(self) -> bool:
        return QwenImage._is_edit_config(self)

    def requires_conditioning_validation_inputs(self) -> bool:
        """Whether this model requires conditioning inputs during validation."""
        return QwenImage._is_edit_config(self)

    def requires_validation_edit_captions(self) -> bool:
        """Whether this model requires edit captions with reference images for validation."""
        return QwenImage._is_edit_config(self)

    def should_precompute_validation_negative_prompt(self) -> bool:
        """Qwen edit models need per-sample negative prompt encoding with reference images."""
        return not self._is_edit_flavour()

    def _create_dummy_image(self):
        """Create a small zero tensor for encoding prompts without real image context."""
        import torch

        return torch.zeros((1, 3, 224, 224), device=self.accelerator.device, dtype=self.config.weight_dtype)

    def encode_validation_negative_prompt(self, negative_prompt: str, positive_prompt_embeds: dict = None):
        """For edit models, encode with dummy image."""
        if not self._is_edit_flavour():
            return super().encode_validation_negative_prompt(negative_prompt, positive_prompt_embeds)

        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        dummy_image = self._create_dummy_image()
        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
            [negative_prompt],
            image=dummy_image,
            device=self.accelerator.device,
        )

        return {
            "prompt_embeds": prompt_embeds[0],
            "attention_masks": prompt_embeds_mask[0],
        }

    def encode_dropout_caption(self, positive_prompt_embeds: dict = None):
        """For edit models, encode empty string with dummy image."""
        if not self._is_edit_flavour():
            return super().encode_dropout_caption(positive_prompt_embeds)

        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        dummy_image = self._create_dummy_image()
        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
            [""],
            image=dummy_image,
            device=self.accelerator.device,
        )

        return {
            "prompt_embeds": prompt_embeds[0],
            "attention_masks": prompt_embeds_mask[0],
        }

    def text_embed_cache_key(self) -> TextEmbedCacheKey:
        if QwenImage._is_edit_v1_config(self):
            return TextEmbedCacheKey.DATASET_AND_FILENAME
        return super().text_embed_cache_key()

    def requires_text_embed_image_context(self) -> bool:
        return QwenImage._is_edit_v1_config(self)

    def requires_conditioning_image_embeds(self) -> bool:
        return QwenImage._is_edit_v1_config(self)

    def conditioning_image_embeds_use_reference_dataset(self) -> bool:
        return QwenImage._is_edit_v1_config(self)

    def _get_conditioning_image_embedder(self):
        if not self._is_edit_v1_flavour():
            raise ValueError("Conditioning image embeds are only supported for Qwen edit-v1 flavours.")

        if self._conditioning_image_embedder is not None:
            return self._conditioning_image_embedder

        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG)
        processor = getattr(pipeline, "processor", None)
        if processor is None:
            raise ValueError("Qwen edit pipeline does not expose a processor for conditioning embeds.")
        text_encoder = getattr(pipeline, "text_encoder", None)
        dtype = getattr(text_encoder, "dtype", torch.float32)

        self._conditioning_image_embedder = self._EditV1ConditioningImageEmbedder(
            processor=processor,
            device=self.accelerator.device,
            dtype=dtype,
        )
        return self._conditioning_image_embedder

    def prepare_batch_conditions(self, batch: dict, state: dict) -> dict:
        if self._is_edit_v2_flavour():
            conditioning_multi = batch.get("conditioning_pixel_values")
            prepared = super().prepare_batch_conditions(batch, state)
            if conditioning_multi is not None:
                prepared["conditioning_pixel_values_multi"] = conditioning_multi
            return prepared
        return super().prepare_batch_conditions(batch, state)

    def _prepare_edit_batch_v1(self, batch: dict) -> dict:
        prompts = batch.get("prompts")
        if prompts is None:
            logger.warning("Edit flavour batch is missing prompts; skipping prompt re-encoding.")
            return batch

        prompt_embeds = batch.get("prompt_embeds")
        prompt_embeds_mask = batch.get("encoder_attention_mask")
        if prompt_embeds is None or prompt_embeds_mask is None:
            raise ValueError(
                "Qwen edit training requires cached prompt embeddings with image context, "
                "but prompt_embeds or encoder_attention_mask was missing from the batch."
            )
        batch["prompt_embeds"] = prompt_embeds.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        batch["encoder_attention_mask"] = prompt_embeds_mask.to(self.accelerator.device, dtype=torch.int64)

        control_latents = batch.get("conditioning_latents")
        if isinstance(control_latents, list):
            control_latents = control_latents[0] if control_latents else None
        if control_latents is None:
            raise ValueError(
                "Qwen edit training requires cached conditioning latents, "
                "but conditioning_latents was missing from the batch."
            )
        batch["edit_control_latents"] = control_latents.to(self.accelerator.device, dtype=self.config.weight_dtype)

        return batch

    def _prepare_edit_batch_v2(self, batch: dict) -> dict:
        prompts = batch.get("prompts")
        if prompts is None:
            logger.warning("Edit flavour batch is missing prompts; skipping prompt re-encoding.")
            return batch

        latents = batch.get("latents")
        if latents is None:
            logger.warning("Edit flavour batch is missing latents; skipping edit conditioning.")
            return batch
        batch_size = latents.shape[0]

        conditioning_multi = batch.get("conditioning_pixel_values_multi")
        if conditioning_multi is None:
            conditioning_single = batch.get("conditioning_pixel_values")
            if conditioning_single is not None:
                conditioning_multi = [conditioning_single]

        if conditioning_multi is None:
            logger.warning("Edit flavour batch is missing conditioning pixels; skipping edit conditioning.")
            return batch

        control_tensor_list: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        for control_tensor in conditioning_multi:
            if control_tensor is None:
                continue
            if control_tensor.dim() != 4:
                raise ValueError("Expected conditioning tensor with shape (B, C, H, W).")
            for idx in range(batch_size):
                control_tensor_list[idx].append(control_tensor[idx].to(self.accelerator.device, self.config.weight_dtype))

        if any(len(items) == 0 for items in control_tensor_list):
            raise ValueError("Each batch item must provide at least one control image for edit-v2 training.")

        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG)
        pixel_dtype = getattr(pipeline.text_encoder, "dtype", torch.float32)
        device = self.accelerator.device

        prompt_embeds_list = []
        prompt_masks_list = []

        for prompt, control_images in zip(prompts, control_tensor_list):
            processed_images = []
            for control_img in control_images:
                img = control_img
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                ratio = img.shape[2] / img.shape[3]
                width = math.sqrt(CONDITION_IMAGE_SIZE * ratio)
                height = width / ratio
                width = round(width / 32) * 32
                height = round(height / 32) * 32
                resized = F.interpolate(img, size=(int(height), int(width)), mode="bilinear", align_corners=False)
                resized = resized.squeeze(0)
                processed = ((resized + 1.0) / 2.0).clamp_(0.0, 1.0)
                processed_images.append(self._tensor_to_pil(processed))

            prompt_embed, prompt_mask = pipeline.encode_prompt(
                [prompt],
                image=processed_images,  # Don't wrap in list - already a list of PIL images
                device=device,
                num_images_per_prompt=1,
            )
            prompt_embeds_list.append(prompt_embed.squeeze(0))
            prompt_masks_list.append(prompt_mask.squeeze(0))

        prompt_embeds = torch.stack(prompt_embeds_list, dim=0).to(device=device, dtype=self.config.weight_dtype)
        prompt_masks = torch.stack(prompt_masks_list, dim=0).to(device=device, dtype=torch.int64)

        batch["prompt_embeds"] = prompt_embeds
        batch["encoder_attention_mask"] = prompt_masks
        batch["control_tensor_list"] = control_tensor_list

        return batch

    def prepare_batch(self, batch: dict, state: dict) -> dict:
        batch = super().prepare_batch(batch, state)
        if self._is_edit_v1_flavour():
            batch = self._prepare_edit_batch_v1(batch)
        elif self._is_edit_v2_flavour():
            batch = self._prepare_edit_batch_v2(batch)
        return batch

    def model_predict(self, prepared_batch):
        if self._is_edit_v1_flavour():
            return self._model_predict_edit_v1(prepared_batch)
        if self._is_edit_v2_flavour():
            return self._model_predict_edit_plus(prepared_batch)
        return self._model_predict_standard(prepared_batch)

    def _model_predict_standard(self, prepared_batch):
        latent_model_input = prepared_batch["noisy_latents"]
        timesteps = prepared_batch["timesteps"]
        target_latents = prepared_batch["latents"]

        # Handle both 4D and 5D inputs
        if latent_model_input.dim() == 5:
            batch_size, num_channels, frames, latent_height, latent_width = latent_model_input.shape
            latent_model_input = latent_model_input.squeeze(2)
        else:
            batch_size, num_channels, latent_height, latent_width = latent_model_input.shape

        # Get the pipeline class to use its static methods
        pipeline_class = self.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG]

        # Note: _unpack_latents expects pixel-space dimensions and will apply vae_scale_factor
        # So we need to convert our latent dimensions back to pixel space
        pixel_height = latent_height * self.vae_scale_factor
        pixel_width = latent_width * self.vae_scale_factor

        # Pack latents using the official method
        flat_latents = pipeline_class._pack_latents(
            latent_model_input,
            batch_size,
            num_channels,
            latent_height,
            latent_width,
        )
        latent_model_input = flat_latents

        # Prepare text embeddings
        prompt_embeds = prepared_batch["prompt_embeds"].to(
            device=self.accelerator.device,
            dtype=self.config.weight_dtype,
        )

        # Get attention mask
        prompt_embeds_mask = prepared_batch.get("encoder_attention_mask")
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask.to(self.accelerator.device, dtype=torch.int64)
            if prompt_embeds_mask.dim() == 3 and prompt_embeds_mask.size(1) == 1:
                prompt_embeds_mask = prompt_embeds_mask.squeeze(1)

        # Prepare image shapes - using the LATENT dimensions divided by 2 (for patchification)
        latent_height_for_shape = latent_height // 2
        latent_width_for_shape = latent_width // 2
        if latent_height_for_shape < 1 or latent_width_for_shape < 1:
            raise ValueError(
                f"Packed latent grid is degenerate. Latent tensor shape: {(batch_size, num_channels, latent_height, latent_width)} "
                f"-> computed patch grid ({latent_height_for_shape}, {latent_width_for_shape})."
            )
        img_shapes = [(1, latent_height_for_shape, latent_width_for_shape)] * batch_size

        # Prepare timesteps (normalize to 0-1 range)
        raw_timesteps = prepared_batch["timesteps"]
        if not torch.is_tensor(raw_timesteps):
            raw_timesteps = torch.tensor(raw_timesteps, device=self.accelerator.device, dtype=torch.float32)
        else:
            raw_timesteps = raw_timesteps.to(device=self.accelerator.device, dtype=torch.float32)
        timesteps = raw_timesteps.expand(batch_size) / 1000.0  # Normalize to [0, 1]

        # Get text sequence lengths
        txt_seq_lens = (
            prompt_embeds_mask.sum(dim=1).tolist()
            if prompt_embeds_mask is not None
            else [prompt_embeds.shape[1]] * batch_size
        )

        # Forward pass through transformer
        with self._force_packed_transformer_output(self.model):
            noise_pred = self.model(
                hidden_states=latent_model_input.to(self.accelerator.device, self.config.weight_dtype),
                timestep=timesteps,
                guidance=None,  # Qwen Image doesn't use guidance during training
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_embeds_mask,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0]

        target_ndim = target_latents.dim()

        if noise_pred.dim() == 3:
            # Older diffusers versions return packed latents; unpack them to spatial maps.
            noise_pred = pipeline_class._unpack_latents(noise_pred, pixel_height, pixel_width, self.vae_scale_factor)
        elif noise_pred.dim() not in (4, 5):
            raise ValueError(f"Unexpected noise prediction rank {noise_pred.dim()} with shape {tuple(noise_pred.shape)}")

        # Align optional frame dimension with the training targets.
        if target_ndim == 5 and noise_pred.dim() == 4:
            noise_pred = noise_pred.unsqueeze(2)
        elif target_ndim == 4 and noise_pred.dim() == 5:
            if noise_pred.size(2) == 1:
                noise_pred = noise_pred.squeeze(2)
            else:
                raise ValueError(
                    f"Cannot squeeze transformer output with non-singular frame dimension: shape {tuple(noise_pred.shape)}"
                )

        if noise_pred.shape != target_latents.shape:
            raise ValueError(
                f"Noise prediction shape {tuple(noise_pred.shape)} does not match target latents shape {tuple(target_latents.shape)}"
            )

        return {"model_prediction": noise_pred}

    class _EditV1ConditioningImageEmbedder:
        def __init__(self, processor, device, dtype):
            # keep processor reference for future use even if we currently only cache raw pixels
            self.processor = processor
            self.device = device
            self.dtype = dtype

        @torch.no_grad()
        def encode(self, images, captions=None):
            embeds: List[dict] = []
            for image in images:
                if not isinstance(image, Image.Image):
                    # convert tensors/arrays back to PIL for consistent processing
                    if isinstance(image, torch.Tensor):
                        tensor = image.detach().cpu()
                        if tensor.dim() == 4 and tensor.size(0) == 1:
                            tensor = tensor.squeeze(0)
                        if tensor.dim() == 3:
                            array = tensor.permute(1, 2, 0).numpy()
                            image = Image.fromarray((np.clip(array, 0.0, 1.0) * 255.0).astype(np.uint8))
                    elif isinstance(image, np.ndarray):
                        image = Image.fromarray(image.astype(np.uint8))
                    else:
                        raise ValueError(f"Unsupported conditioning image type: {type(image)}")

                array = np.array(image.convert("RGB"), copy=True)
                tensor = torch.from_numpy(array).permute(2, 0, 1).to(dtype=self.dtype)
                tensor = tensor / 255.0
                embeds.append({"pixel_values": tensor})
            return embeds

    def _model_predict_edit_v1(self, prepared_batch):
        latent_model_input = prepared_batch["noisy_latents"]
        control_latents = prepared_batch.get("edit_control_latents")
        if control_latents is None:
            raise ValueError("Edit training requires control latents but none were provided in the batch.")

        # Align tensor shapes for latents
        if latent_model_input.dim() == 5:
            batch_size, num_channels, _, latent_height, latent_width = latent_model_input.shape
            latent_model_input = latent_model_input.squeeze(2)
        else:
            batch_size, num_channels, latent_height, latent_width = latent_model_input.shape

        if control_latents.dim() == 5:
            _, control_channels, _, control_height, control_width = control_latents.shape
            control_latents = control_latents.squeeze(2)
        else:
            _, control_channels, control_height, control_width = control_latents.shape

        pipeline_class = self.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG]

        pixel_height = latent_height * self.vae_scale_factor
        pixel_width = latent_width * self.vae_scale_factor

        packed_latents = pipeline_class._pack_latents(
            latent_model_input,
            batch_size,
            num_channels,
            latent_height,
            latent_width,
        )
        packed_control = pipeline_class._pack_latents(
            control_latents,
            batch_size,
            control_channels,
            control_height,
            control_width,
        )
        transformer_inputs = torch.cat([packed_latents, packed_control], dim=1)

        prompt_embeds = prepared_batch["prompt_embeds"].to(
            device=self.accelerator.device,
            dtype=self.config.weight_dtype,
        )
        prompt_embeds_mask = prepared_batch.get("encoder_attention_mask")
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask.to(self.accelerator.device, dtype=torch.int64)
            if prompt_embeds_mask.dim() == 3 and prompt_embeds_mask.size(1) == 1:
                prompt_embeds_mask = prompt_embeds_mask.squeeze(1)
            txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        else:
            txt_seq_lens = [prompt_embeds.shape[1]] * batch_size

        img_shapes = [
            [
                (1, latent_height // 2, latent_width // 2),
                (1, control_height // 2, control_width // 2),
            ]
            for _ in range(batch_size)
        ]

        raw_timesteps = prepared_batch["timesteps"]
        if not torch.is_tensor(raw_timesteps):
            raw_timesteps = torch.tensor(raw_timesteps, device=self.accelerator.device, dtype=torch.float32)
        else:
            raw_timesteps = raw_timesteps.to(device=self.accelerator.device, dtype=torch.float32)
        timesteps = raw_timesteps.expand(batch_size) / 1000.0

        with self._force_packed_transformer_output(self.model):
            noise_pred = self.model(
                hidden_states=transformer_inputs.to(self.accelerator.device, self.config.weight_dtype),
                timestep=timesteps,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_embeds_mask,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0]

        noise_pred = noise_pred[:, : packed_latents.size(1)]
        noise_pred = pipeline_class._unpack_latents(noise_pred, pixel_height, pixel_width, self.vae_scale_factor)
        if noise_pred.dim() == 5:
            noise_pred = noise_pred.squeeze(2)

        return {"model_prediction": noise_pred}

    def _model_predict_edit_plus(self, prepared_batch):
        latent_model_input = prepared_batch["noisy_latents"]
        control_tensor_list = prepared_batch.get("control_tensor_list")
        if control_tensor_list is None:
            raise ValueError("Edit-v2 training requires control tensors but none were provided in the batch.")

        if latent_model_input.dim() == 5:
            batch_size, num_channels, _, latent_height, latent_width = latent_model_input.shape
            latent_model_input = latent_model_input.squeeze(2)
        else:
            batch_size, num_channels, latent_height, latent_width = latent_model_input.shape

        pipeline_class = self.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG]
        pixel_height = latent_height * self.vae_scale_factor
        pixel_width = latent_width * self.vae_scale_factor

        packed_latents = pipeline_class._pack_latents(
            latent_model_input,
            batch_size,
            num_channels,
            latent_height,
            latent_width,
        )
        base_packed_tokens = packed_latents
        packed_latents_split = torch.chunk(packed_latents, batch_size, dim=0)

        img_shapes = [[(1, latent_height // 2, latent_width // 2)] for _ in range(batch_size)]
        combined_tokens = []
        vae = self.get_vae()
        if vae is None:
            raise ValueError("Qwen edit-v2 inference requires a loaded VAE.")
        vae.to(device=self.accelerator.device, dtype=self.config.weight_dtype)

        for idx, sample_controls in enumerate(control_tensor_list):
            if not sample_controls:
                raise ValueError("Each batch item must provide at least one control image for edit-v2 training.")

            sample_tokens = [packed_latents_split[idx]]
            for control_img in sample_controls:
                control_tensor = control_img
                if control_tensor.dim() == 3:
                    control_tensor = control_tensor.unsqueeze(0)

                ratio = control_tensor.shape[2] / control_tensor.shape[3]
                width = math.sqrt(VAE_IMAGE_SIZE * ratio)
                height = width / ratio
                width = round(width / 32) * 32
                height = round(height / 32) * 32

                resized = F.interpolate(
                    control_tensor,
                    size=(int(height), int(width)),
                    mode="bilinear",
                    align_corners=False,
                )

                scaled = resized.to(device=self.accelerator.device, dtype=self.config.weight_dtype).clamp_(-1.0, 1.0)
                vae_input = scaled.unsqueeze(2)  # (1, C, 1, H, W)

                with torch.no_grad():
                    encoded = vae.encode(vae_input).latent_dist.sample()

                if encoded.dim() == 5:
                    encoded = encoded.squeeze(2)

                latents_mean = (
                    torch.tensor(vae.config.latents_mean)
                    .view(1, vae.config.z_dim, 1, 1)
                    .to(device=encoded.device, dtype=encoded.dtype)
                )
                latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1).to(
                    device=encoded.device, dtype=encoded.dtype
                )
                control_latent = (encoded - latents_mean) * latents_std

                cl_height, cl_width = control_latent.shape[2], control_latent.shape[3]
                packed_control = pipeline_class._pack_latents(
                    control_latent,
                    1,
                    control_latent.shape[1],
                    cl_height,
                    cl_width,
                )
                sample_tokens.append(packed_control)
                img_shapes[idx].append((1, cl_height // 2, cl_width // 2))

            combined_tokens.append(torch.cat(sample_tokens, dim=1))

        transformer_inputs = torch.cat(combined_tokens, dim=0)

        prompt_embeds = prepared_batch["prompt_embeds"].to(
            device=self.accelerator.device,
            dtype=self.config.weight_dtype,
        )
        prompt_embeds_mask = prepared_batch.get("encoder_attention_mask")
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask.to(self.accelerator.device, dtype=torch.int64)
            if prompt_embeds_mask.dim() == 3 and prompt_embeds_mask.size(1) == 1:
                prompt_embeds_mask = prompt_embeds_mask.squeeze(1)
            txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()
        else:
            txt_seq_lens = [prompt_embeds.shape[1]] * batch_size

        raw_timesteps = prepared_batch["timesteps"]
        if not torch.is_tensor(raw_timesteps):
            raw_timesteps = torch.tensor(raw_timesteps, device=self.accelerator.device, dtype=torch.float32)
        else:
            raw_timesteps = raw_timesteps.to(device=self.accelerator.device, dtype=torch.float32)
        timesteps = raw_timesteps.expand(batch_size) / 1000.0

        with self._force_packed_transformer_output(self.model):
            noise_pred = self.model(
                hidden_states=transformer_inputs.to(self.accelerator.device, self.config.weight_dtype),
                timestep=timesteps,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_embeds_mask,
                img_shapes=img_shapes,
                txt_seq_lens=txt_seq_lens,
                return_dict=False,
            )[0]

        noise_pred = noise_pred[:, : base_packed_tokens.size(1)]

        noise_pred = pipeline_class._unpack_latents(noise_pred, pixel_height, pixel_width, self.vae_scale_factor)
        if noise_pred.dim() == 5:
            noise_pred = noise_pred.squeeze(2)

        return {"model_prediction": noise_pred}

    def pre_vae_encode_transform_sample(self, sample):
        """
        Pre-encode transform for the sample before passing it to the VAE.
        Qwen Image VAE expects 5D input (adds frame dimension).
        """
        # Add frame dimension for Qwen VAE if needed
        if sample.dim() == 4:
            sample = sample.unsqueeze(2)  # (B, C, H, W) -> (B, C, 1, H, W)
        return sample

    def post_vae_encode_transform_sample(self, sample):
        """
        Post-encode transform for Qwen Image VAE output.
        Normalizes latents and removes frame dimension.
        """
        # Qwen Image VAE normalization
        # Remove frame dimension if present
        vae = self.get_vae()
        if vae is None:
            raise ValueError("Cannot normalize Qwen Image latents without a loaded VAE.")
        sample_latents = sample.latent_dist.sample()
        if sample_latents.dim() == 5:
            sample_latents = sample_latents.squeeze(2)  # (B, C, 1, H, W) -> (B, C, H, W)
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1)
            .to(sample_latents.device, sample_latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1).to(
            sample_latents.device, sample_latents.dtype
        )

        sample_latents = (sample_latents - latents_mean) * latents_std

        return sample_latents

    def pre_validation_preview_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Pre-process latents before passing to validation preview decoder.

        Qwen Image uses packed transformer latents that need to be untokenized/unpacked
        to spatial format, and the Wan 2.1 decoder expects rank-5 tensors (video format),
        so we need to unpack and add a frame dimension.

        Args:
            latents: The latents tensor to transform

        Returns:
            The transformed latents tensor unpacked and with frame dimension added (rank 5)
        """
        original_shape = latents.shape
        if latents.dim() == 4:
            # Already unpacked spatial latents: (B, C, H, W) -> (B, C, 1, H, W)
            latents = latents.unsqueeze(2)
            logger.debug(f"Validation preview: added frame dimension {original_shape} -> {latents.shape}")
        elif latents.dim() == 3:
            # Packed transformer latents - need to unpack to spatial format
            # The packed format is [B, num_tokens, hidden_dim]
            # We need to unpack using the pipeline's _unpack_latents method

            # Get the pipeline class for unpacking
            pipeline_class = self.PIPELINE_CLASSES.get(PipelineTypes.TEXT2IMG)
            if pipeline_class is None:
                raise ValueError("Cannot unpack latents without pipeline class")

            # Try to infer the spatial dimensions from the number of tokens
            # For Qwen, tokens = (H/2) * (W/2) where H,W are latent spatial dims
            batch_size, num_tokens, hidden_dim = latents.shape

            # Estimate spatial size: assume square aspect ratio for validation
            # num_tokens = (H/2) * (W/2), so H = W = sqrt(num_tokens) * 2
            estimated_side = int(math.sqrt(num_tokens)) * 2
            pixel_height = estimated_side * self.vae_scale_factor
            pixel_width = estimated_side * self.vae_scale_factor

            logger.debug(
                f"Validation preview: unpacking latents {original_shape} "
                f"(estimated pixel size: {pixel_height}x{pixel_width})"
            )

            try:
                # Unpack the latents to spatial format
                latents = pipeline_class._unpack_latents(latents, pixel_height, pixel_width, self.vae_scale_factor)

                # Now latents should be [B, C, H, W] - add frame dimension
                if latents.dim() == 4:
                    latents = latents.unsqueeze(2)  # [B, C, 1, H, W]
                elif latents.dim() == 5:
                    # Already has frame dimension
                    pass

                logger.debug(f"Validation preview: unpacked to {latents.shape}")
            except Exception as e:
                logger.error(
                    f"Failed to unpack validation preview latents with shape {original_shape}: {e}. "
                    f"This may indicate non-square aspect ratio or unexpected latent format."
                )
                raise

        return latents

    def check_user_config(self):
        """
        Check and validate user configuration for Qwen Image.
        """
        super().check_user_config()

        # Qwen Image specific checks
        if self.config.aspect_bucket_alignment != 32:
            if not getattr(self.config, "i_know_what_i_am_doing", False):
                logger.warning(
                    f"{self.NAME} requires an alignment value of 32px. "
                    "Overriding the value of --aspect_bucket_alignment. "
                    "If you really want to proceed without this enforcement, "
                    "supply `--i_know_what_i_am_doing`. -!-"
                )
                self.config.aspect_bucket_alignment = 32
            else:
                logger.warning(
                    f"-!- {self.NAME} requires an alignment value of 32px, but you have "
                    "supplied `--i_know_what_i_am_doing`, so this limit will not be enforced. -!-"
                )
                logger.warning(
                    "Proceeding with a non-32px alignment may cause bucketting errors, "
                    "image artifacts, or unstable training behaviour."
                )

        # Ensure we're using flow matching
        if self.config.prediction_type != "flow_matching":
            logger.warning(f"{self.NAME} uses flow matching. " "Overriding prediction_type to 'flow_matching'.")
            self.config.prediction_type = "flow_matching"

    def pretrained_load_args(self, pretrained_load_args: dict) -> dict:
        args = super().pretrained_load_args(pretrained_load_args)
        return apply_musubi_pretrained_defaults(self.config, args)


from simpletuner.helpers.models.registry import ModelRegistry

ModelRegistry.register("qwen_image", QwenImage)
