import logging
import math
import os
import random

import torch
import torch.nn.functional as F
from typing import List, Optional
from diffusers import AutoencoderKLQwenImage, QwenImagePipeline
from diffusers.models.attention_processor import Attention
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.qwen_image.pipeline import QwenImageEditPipeline
from simpletuner.helpers.models.qwen_image.pipeline_edit_plus import (
    QwenImageEditPlusPipeline,
    CONDITION_IMAGE_SIZE,
    VAE_IMAGE_SIZE,
)
from simpletuner.helpers.models.qwen_image.transformer import QwenImageTransformer2DModel
from simpletuner.helpers.training.multi_process import _get_rank

logger = logging.getLogger(__name__)
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class QwenImage(ImageModelFoundation):
    NAME = "Qwen-Image"
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKLQwenImage
    AUTOENCODER_SCALING_FACTOR = 1.0
    LATENT_CHANNEL_COUNT = 16

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

    def _is_edit_v1_flavour(self) -> bool:
        flavour = self._get_model_flavour()
        return flavour in self.EDIT_V1_FLAVOURS if flavour is not None else False

    def _is_edit_v2_flavour(self) -> bool:
        flavour = self._get_model_flavour()
        return flavour in self.EDIT_V2_FLAVOURS if flavour is not None else False

    def _is_edit_flavour(self) -> bool:
        return self._is_edit_v1_flavour() or self._is_edit_v2_flavour()
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

        # Use pipeline's encode_prompt method
        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
            prompts,
            device=self.accelerator.device,
            num_images_per_prompt=1,
        )

        return prompt_embeds, prompt_embeds_mask

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

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor, prompt: str) -> dict:
        """
        Convert negative text embeddings for pipeline use.
        """
        attention_mask = text_embedding.get("attention_masks", None)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        return {
            "negative_prompt_embeds": (
                text_embedding["prompt_embeds"].unsqueeze(0)
                if text_embedding["prompt_embeds"].dim() == 2
                else text_embedding["prompt_embeds"]
            ),
            "negative_prompt_embeds_mask": (attention_mask.to(dtype=torch.int64) if attention_mask is not None else None),
        }

    def requires_conditioning_dataset(self) -> bool:
        if self._is_edit_flavour():
            return True
        return super().requires_conditioning_dataset()

    def requires_conditioning_image_embeds(self) -> bool:
        if self._is_edit_v1_flavour():
            return True
        return super().requires_conditioning_image_embeds()

    def _get_conditioning_image_embedder(self):
        if not self._is_edit_v1_flavour():
            return None

        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG)
        processor = getattr(pipeline, "processor", None)
        if processor is None:
            raise ValueError("Qwen edit pipeline does not expose a processor for conditioning embeds.")
        text_encoder = getattr(pipeline, "text_encoder", None)
        dtype = getattr(text_encoder, "dtype", torch.float32)

        return self._EditV1ConditioningImageEmbedder(
            processor=processor,
            device=self.accelerator.device,
            dtype=dtype,
        )

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

        conditioning_pixels = batch.get("conditioning_pixel_values")
        if conditioning_pixels is None:
            logger.warning("Edit flavour batch is missing conditioning pixels; skipping edit conditioning.")
            return batch

        conditioning_images = batch.get("conditioning_image_embeds")
        pixel_values_for_prompt = None
        if isinstance(conditioning_images, dict) and "pixel_values" in conditioning_images:
            pixel_values_for_prompt = conditioning_images["pixel_values"]

        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG)
        pixel_dtype = getattr(pipeline.text_encoder, "dtype", torch.float32)
        device = self.accelerator.device

        if pixel_values_for_prompt is None:
            pixel_values_for_prompt = ((conditioning_pixels + 1.0) / 2.0).clamp_(0.0, 1.0)

        if pixel_values_for_prompt.dim() == 4:
            prompt_image = pixel_values_for_prompt.to(device=device, dtype=pixel_dtype)
        else:
            prompt_image = pixel_values_for_prompt
            if hasattr(prompt_image, "to"):
                prompt_image = prompt_image.to(device=device, dtype=pixel_dtype)

        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
            prompts,
            image=prompt_image,
            device=device,
            num_images_per_prompt=1,
        )

        batch["prompt_embeds"] = prompt_embeds.to(device=device, dtype=self.config.weight_dtype)
        batch["encoder_attention_mask"] = prompt_embeds_mask.to(device=device, dtype=torch.int64)

        vae_input = conditioning_pixels.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        if vae_input.dim() == 4:
            vae_input = vae_input.unsqueeze(2)

        self.vae.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        with torch.no_grad():
            encoded = self.vae.encode(vae_input).latent_dist.sample()

        if encoded.dim() == 5:
            encoded = encoded.squeeze(2)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1)
            .to(device=encoded.device, dtype=encoded.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1).to(
            device=encoded.device, dtype=encoded.dtype
        )
        control_latents = (encoded - latents_mean) * latents_std
        batch["edit_control_latents"] = control_latents

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
                processed_images.append(processed.to(device=device, dtype=pixel_dtype))

            prompt_embed, prompt_mask = pipeline.encode_prompt(
                [prompt],
                image=[processed_images],
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
        img_shapes = [(1, latent_height // 2, latent_width // 2)] * batch_size

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

        # Unpack the noise prediction back to original shape
        noise_pred = pipeline_class._unpack_latents(noise_pred, pixel_height, pixel_width, self.vae_scale_factor)

        # Remove the extra dimension that _unpack_latents adds
        if noise_pred.dim() == 5:
            noise_pred = noise_pred.squeeze(2)  # Remove the frame dimension

        return {"model_prediction": noise_pred}

    class _EditV1ConditioningImageEmbedder:
        def __init__(self, processor, device, dtype):
            self.processor = processor
            self.device = device
            self.dtype = dtype

        @torch.no_grad()
        def encode(self, images):
            processed = self.processor(images=images, return_tensors="pt")
            pixel_values = processed["pixel_values"].to(device=self.device, dtype=self.dtype)
            image_grid_thw = processed.get("image_grid_thw", None)

            embeds = []
            for idx in range(pixel_values.shape[0]):
                entry = {"pixel_values": pixel_values[idx]}
                if image_grid_thw is not None:
                    entry["image_grid_thw"] = image_grid_thw[idx]
                embeds.append(entry)
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
        ] * batch_size

        raw_timesteps = prepared_batch["timesteps"]
        if not torch.is_tensor(raw_timesteps):
            raw_timesteps = torch.tensor(raw_timesteps, device=self.accelerator.device, dtype=torch.float32)
        else:
            raw_timesteps = raw_timesteps.to(device=self.accelerator.device, dtype=torch.float32)
        timesteps = raw_timesteps.expand(batch_size) / 1000.0

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

        self.vae.to(device=self.accelerator.device, dtype=self.config.weight_dtype)

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
                    encoded = self.vae.encode(vae_input).latent_dist.sample()

                if encoded.dim() == 5:
                    encoded = encoded.squeeze(2)

                latents_mean = (
                    torch.tensor(self.vae.config.latents_mean)
                    .view(1, self.vae.config.z_dim, 1, 1)
                    .to(device=encoded.device, dtype=encoded.dtype)
                )
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1).to(
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
        sample_latents = sample.latent_dist.sample()
        if sample_latents.dim() == 5:
            sample_latents = sample_latents.squeeze(2)  # (B, C, 1, H, W) -> (B, C, H, W)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1)
            .to(sample_latents.device, sample_latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1).to(
            sample_latents.device, sample_latents.dtype
        )

        sample_latents = (sample_latents - latents_mean) * latents_std

        return sample_latents

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


from simpletuner.helpers.models.registry import ModelRegistry

ModelRegistry.register("qwen_image", QwenImage)
