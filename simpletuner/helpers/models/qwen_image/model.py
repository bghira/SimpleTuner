import logging
import os
import random

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLQwenImage, QwenImagePipeline
from diffusers.models.attention_processor import Attention
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
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

    # Default model flavor
    DEFAULT_MODEL_FLAVOUR = "v1.0"
    HUGGINGFACE_PATHS = {
        "v1.0": "Qwen/Qwen-Image",
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
        self._conditioning_image_embedder = None

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

    def requires_conditioning_image_embeds(self) -> bool:
        return True

    class _ConditioningImageEmbedder:
        def __init__(self, processor, vision_model, device, dtype):
            self.processor = processor
            self.vision_model = vision_model
            self.device = device
            self.dtype = dtype

            self.vision_model.eval()
            self.vision_model.to(device=self.device, dtype=self.dtype)
            for param in self.vision_model.parameters():
                param.requires_grad_(False)

        @torch.no_grad()
        def encode(self, images):
            processed = self.processor(images=images, return_tensors="pt")
            pixel_values = processed.get("pixel_values")
            if pixel_values is None:
                vision_inputs = processed.get("vision_inputs")
                pixel_values = None
                if isinstance(vision_inputs, dict):
                    pixel_values = vision_inputs.get("pixel_values")
                if pixel_values is None:
                    raise ValueError("Processor did not return 'pixel_values' for conditioning image encoding.")
            pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)
            outputs = self.vision_model(pixel_values=pixel_values, return_dict=True)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                hidden = outputs[0]
            hidden = hidden.to(device="cpu")
            return [hidden[i] for i in range(hidden.shape[0])]

    def _get_conditioning_image_embedder(self):
        if self._conditioning_image_embedder is not None:
            return self._conditioning_image_embedder

        if not self.text_encoders:
            self.load_text_encoder()
        if not self.text_encoders:
            raise ValueError("Qwen Image conditioning requires the text encoder to be loaded.")

        text_encoder = self.text_encoders[0]

        vision_model = None
        getter = getattr(text_encoder, "get_vision_tower", None)
        if callable(getter):
            try:
                vision_model = getter()
            except Exception:
                vision_model = None
        if vision_model is None:
            vision_model = getattr(text_encoder, "vision_model", None)
        if vision_model is None:
            vision_model = getattr(text_encoder, "vision_tower", None)
        if vision_model is None and hasattr(text_encoder, "model"):
            vision_model = getattr(text_encoder.model, "vision_tower", None)
        if vision_model is None:
            raise ValueError("Unable to locate a vision tower on the loaded Qwen text encoder for conditioning embeds.")

        processor = self._conditioning_processor
        if processor is None:
            pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG)
            processor = getattr(pipeline, "processor", None)
            if processor is None:
                processor_id = getattr(self.config, "conditioning_image_processor_name_or_path", None) or getattr(
                    self.config, "pretrained_model_name_or_path", None
                )
                if processor_id is None:
                    processor_id = "Qwen/Qwen-Image"
                processor = AutoProcessor.from_pretrained(processor_id)
            self._conditioning_processor = processor

        device = getattr(self.accelerator, "device", torch.device("cpu"))
        dtype = getattr(self.config, "weight_dtype", torch.float32)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, torch.float32)

        self._conditioning_image_embedder = self._ConditioningImageEmbedder(
            processor=processor,
            vision_model=vision_model,
            device=device,
            dtype=dtype,
        )
        return self._conditioning_image_embedder

    def model_predict(self, prepared_batch):
        """
        Perform a forward pass with the Qwen Image model.
        """
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
        timesteps = (
            torch.tensor(prepared_batch["timesteps"]).expand(batch_size).to(device=self.accelerator.device)
            / 1000.0  # Normalize to [0, 1]
        )

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
