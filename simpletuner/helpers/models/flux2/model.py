"""
FLUX.2 model implementation for SimpleTuner.

Based on Black Forest Labs FLUX.2-dev architecture with Mistral-3 text encoder.
"""

import logging
import os
import random
import sys
from typing import List, Optional

import torch
from einops import rearrange
from safetensors.torch import save_file
from torch import Tensor

from simpletuner.helpers.configuration.registry import (
    ConfigRegistry,
    ConfigRule,
    RuleType,
    ValidationResult,
    make_override_rule,
)
from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.flux2 import pack_latents, pack_text, unpack_latents
from simpletuner.helpers.models.flux2.autoencoder import AutoencoderKLFlux2
from simpletuner.helpers.models.flux2.pipeline import Flux2Pipeline
from simpletuner.helpers.models.flux2.transformer import Flux2Transformer2DModel
from simpletuner.helpers.training.multi_process import _get_rank

logger = logging.getLogger(__name__)
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")

# FLUX.2 constants
MISTRAL_PATH = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
SYSTEM_MESSAGE = """You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation."""
OUTPUT_LAYERS = [10, 20, 30]
MAX_SEQUENCE_LENGTH = 512

# Scheduler configuration for FLUX.2 flow matching
SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
}


class Flux2(ImageModelFoundation):
    """FLUX.2 model implementation for SimpleTuner."""

    NAME = "Flux.2"
    MODEL_DESCRIPTION = "FLUX.2-dev with Mistral-3 text encoder"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKLFlux2
    LATENT_CHANNEL_COUNT = 128  # 32 VAE channels × 4 (2×2 pixel shuffle) = 128 transformer channels
    VAE_SCALE_FACTOR = 16  # 8x spatial + 2x pixel shuffle

    # LoRA targets for FLUX.2 transformer (diffusers naming)
    DEFAULT_LORA_TARGET = [
        # Double stream attention
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        # "attn.add_q_proj",
        # "attn.add_k_proj",
        # "attn.add_v_proj",
        # "attn.to_add_out",
        # Double stream FF
        # "ff.linear_in",
        # "ff.linear_out",
        # "ff_context.linear_in",
        # "ff_context.linear_out",
        # Single stream (parallel attention + FF)
        "attn.to_qkv_mlp_proj",
        # "attn.to_out",
    ]
    DEFAULT_LYCORIS_TARGET = ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]

    MODEL_CLASS = Flux2Transformer2DModel
    MODEL_SUBFOLDER = "transformer"

    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: Flux2Pipeline,
    }

    DEFAULT_MODEL_FLAVOUR = "dev"
    HUGGINGFACE_PATHS = {
        "dev": "black-forest-labs/FLUX.2-dev",
    }
    MODEL_LICENSE = "other"

    # Single text encoder configuration (Mistral-3)
    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Mistral-Small-3.1-24B",
            "tokenizer": None,  # Uses AutoProcessor
            "model": None,  # Uses Mistral3ForConditionalGeneration
            "custom_loader": True,
        },
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mistral_model = None
        self._mistral_processor = None

    @staticmethod
    def _patchify_latents(latents: Tensor) -> Tensor:
        """
        Pixel-shuffle latents from (B, C, H, W) -> (B, 4C, H/2, W/2).
        Expected to be used to align 32-channel VAE latents with the 128-channel transformer input.
        """
        b, c, h, w = latents.shape
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"Latent spatial dims must be even to patchify, got {(h, w)}")
        latents = latents.view(b, c, h // 2, 2, w // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(b, c * 4, h // 2, w // 2)
        return latents

    def _normalize_latents(self, latents: Tensor) -> Tensor:
        """
        Normalize latents using VAE batch norm statistics to match pipeline behavior.
        """
        if self.vae is None or not hasattr(self.vae, "bn"):
            return latents
        bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps)
        return (latents - bn_mean) / bn_std

    @staticmethod
    def _latents_appear_normalized(latents: Tensor) -> bool:
        """Heuristic to avoid double-normalizing already normalized latents."""
        mean = latents.mean().abs().item()
        std = latents.std(unbiased=False).item()
        return mean < 0.05 and 0.8 < std < 1.25

    def load_model(self, move_to_device: bool = True):
        """Load the FLUX.2 transformer using diffusers from_pretrained."""
        dtype = self.config.weight_dtype
        model_path = self.config.pretrained_model_name_or_path

        logger.info("Loading FLUX.2 transformer...")

        transformer = Flux2Transformer2DModel.from_pretrained(
            model_path,
            subfolder=self.MODEL_SUBFOLDER,
            torch_dtype=dtype,
        )

        if move_to_device:
            transformer.to(self.accelerator.device, dtype=dtype)

        self.model = transformer
        logger.info("FLUX.2 transformer loaded successfully")
        return transformer

    def load_vae(self, move_to_device: bool = True):
        """Load the FLUX.2 custom VAE using diffusers from_pretrained."""
        dtype = self.config.weight_dtype
        model_path = self.config.pretrained_model_name_or_path

        logger.info("Loading FLUX.2 VAE...")

        # Use custom VAE path if provided, otherwise load from main model repo
        vae_path = self.config.pretrained_vae_model_name_or_path or model_path

        vae = AutoencoderKLFlux2.from_pretrained(
            vae_path,
            subfolder="vae" if vae_path == model_path else None,
            torch_dtype=dtype,
        )

        if move_to_device:
            vae.to(self.accelerator.device, dtype=dtype)
        vae.requires_grad_(False)
        vae.eval()

        self.vae = vae
        logger.info("FLUX.2 VAE loaded successfully")
        return vae

    def load_text_encoder(self, move_to_device: bool = True):
        """Load the Mistral-3 text encoder."""
        try:
            from transformers import AutoProcessor, Mistral3ForConditionalGeneration
        except ImportError:
            logger.error("Mistral3ForConditionalGeneration not found. " "Please install transformers>=4.45.0")
            sys.exit(1)

        dtype = self.config.weight_dtype
        text_encoder_precision = getattr(self.config, "text_encoder_1_precision", "no_change")
        should_quantize_text_encoder = text_encoder_precision not in (None, "no_change")
        quantize_via_cpu = getattr(self.config, "quantize_via", None) == "cpu"

        mistral_path = getattr(self.config, "pretrained_text_encoder_model_name_or_path", None) or MISTRAL_PATH
        mistral_revision = getattr(self.config, "text_encoder_revision", None) or getattr(self.config, "revision", None)

        logger.info(f"Loading Mistral-3 text encoder from {mistral_path}...")

        # Load processor
        processor_kwargs = {}
        if mistral_revision is not None:
            processor_kwargs["revision"] = mistral_revision
        self._mistral_processor = AutoProcessor.from_pretrained(mistral_path, **processor_kwargs)

        # Load model
        model_dtype = dtype if not should_quantize_text_encoder else torch.float32
        model_kwargs = {"torch_dtype": model_dtype}
        if mistral_revision is not None:
            model_kwargs["revision"] = mistral_revision
        self._mistral_model = Mistral3ForConditionalGeneration.from_pretrained(
            mistral_path,
            **model_kwargs,
        )
        if move_to_device:
            target_device = (
                torch.device("cpu") if quantize_via_cpu and should_quantize_text_encoder else self.accelerator.device
            )
            target_dtype = dtype if not should_quantize_text_encoder else model_dtype
            self._mistral_model.to(target_device, dtype=target_dtype)
        self._mistral_model.requires_grad_(False)
        self._mistral_model.eval()

        # Store in standard location for compatibility
        self.text_encoders = [self._mistral_model]
        self.tokenizers = [self._mistral_processor]

        logger.info("Mistral-3 text encoder loaded successfully")

    def tread_init(self):
        """Initialize TREAD router for efficient training."""
        from simpletuner.helpers.training.tread import TREADRouter

        if getattr(self.config, "tread_config", None) is None or self.config.tread_config is None:
            logger.error("TREAD training requires you to configure the routes")
            sys.exit(1)

        routes = self.config.tread_config.get("routes")
        if routes is None:
            logger.error("TREAD config requires 'routes' key")
            sys.exit(1)

        self.unwrap_model(self.model).set_router(
            TREADRouter(seed=self.config.seed, device=self.accelerator.device),
            routes,
        )
        logger.info(f"TREAD initialized with routes: {routes}")

    def _format_mistral_input(self, prompts: List[str]) -> List[List[dict]]:
        """Format prompts for Mistral chat template."""
        # Remove any [IMG] tokens to avoid issues
        cleaned = [p.replace("[IMG]", "") for p in prompts]

        return [
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_MESSAGE}],
                },
                {"role": "user", "content": [{"type": "text", "text": prompt}]},
            ]
            for prompt in cleaned
        ]

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode prompts using Mistral-3.

        Returns:
            prompt_embeds: (B, L, 15360) - stacked outputs from layers 10, 20, 30
            attention_mask: (B, L)
        """
        device = self.accelerator.device
        dtype = self._mistral_model.dtype

        if not isinstance(prompts, list):
            prompts = [prompts]

        # Format input messages
        messages_batch = self._format_mistral_input(prompts)

        # Tokenize
        inputs = self._mistral_processor.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward pass
        with torch.no_grad():
            output = self._mistral_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        # Stack outputs from layers 10, 20, 30
        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS], dim=1)
        prompt_embeds = rearrange(out, "b c l d -> b l (c d)")

        return prompt_embeds, attention_mask

    def _format_text_embedding(
        self,
        text_embedding,
    ) -> dict:
        """Format text embedding for storage/caching."""
        return {
            "prompt_embeds": text_embedding[0],
            "attention_mask": text_embedding[1],
        }

    def convert_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        """Convert cached embedding for pipeline use."""
        prompt_embeds = text_embedding["prompt_embeds"]
        attention_mask = text_embedding.get("attention_mask")

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        return {
            "prompt_embeds": prompt_embeds,
            "attention_mask": attention_mask,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        """Convert cached negative embedding for pipeline use."""
        # FLUX.2 doesn't use CFG in the traditional sense
        # Return empty dict if no guidance needed
        if self.config.validation_guidance is None or self.config.validation_guidance <= 1.0:
            return {}

        prompt_embeds = text_embedding["prompt_embeds"]
        attention_mask = text_embedding.get("attention_mask")

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        return {
            "negative_prompt_embeds": prompt_embeds,
            "negative_attention_mask": attention_mask,
        }

    def model_predict(self, prepared_batch: dict) -> dict:
        """
        Forward pass through FLUX.2 transformer.
        """
        # Patchify VAE latents on-the-fly if they haven't been pixel-shuffled yet (32 -> 128 channels)
        if prepared_batch["noisy_latents"].shape[1] == 32:
            prepared_batch["noisy_latents"] = self._patchify_latents(prepared_batch["noisy_latents"])
            if "latents" in prepared_batch and prepared_batch["latents"] is not None:
                prepared_batch["latents"] = self._patchify_latents(prepared_batch["latents"])
            if "noise" in prepared_batch and prepared_batch["noise"] is not None:
                prepared_batch["noise"] = self._patchify_latents(prepared_batch["noise"])

        batch_size = prepared_batch["latents"].shape[0]
        device = self.accelerator.device
        dtype = self.config.base_weight_dtype

        # Pack latents: (B, C, H, W) -> (B, S, C) with position IDs
        packed_latents, img_ids = pack_latents(prepared_batch["noisy_latents"])
        packed_latents = packed_latents.to(device=device, dtype=dtype)
        img_ids = img_ids.to(device=device)

        # Pack text embeddings with position IDs
        prompt_embeds = prepared_batch["prompt_embeds"].to(device=device, dtype=dtype)
        txt, txt_ids = pack_text(prompt_embeds)
        txt_ids = txt_ids.to(device=device)

        # Prepare timesteps (normalized to [0, 1])
        timesteps = prepared_batch["timesteps"]
        if isinstance(timesteps, (int, float)):
            timesteps = torch.tensor([timesteps], device=device, dtype=dtype)
        else:
            timesteps = torch.tensor(timesteps, device=device, dtype=dtype)
        timesteps = timesteps.expand(batch_size) / SCHEDULER_CONFIG["num_train_timesteps"]

        # Handle guidance
        if self.config.flux_guidance_mode == "constant":
            guidance_value = float(self.config.flux_guidance_value)
            guidance = torch.full((batch_size,), guidance_value, device=device, dtype=dtype)
        elif self.config.flux_guidance_mode == "random-range":
            guidance = torch.tensor(
                [
                    random.uniform(
                        self.config.flux_guidance_min,
                        self.config.flux_guidance_max,
                    )
                    for _ in range(batch_size)
                ],
                device=device,
                dtype=dtype,
            )
        else:
            guidance = torch.ones(batch_size, device=device, dtype=dtype)

        # Build force_keep_mask for TREAD if using mask/segmentation conditioning
        force_keep_mask = None
        if (
            getattr(self.config, "tread_config", None) is not None
            and "conditioning_pixel_values" in prepared_batch
            and prepared_batch["conditioning_pixel_values"] is not None
            and prepared_batch.get("conditioning_type") in ("mask", "segmentation")
        ):
            with torch.no_grad():
                h_tokens = prepared_batch["latents"].shape[2]
                w_tokens = prepared_batch["latents"].shape[3]
                mask_img = prepared_batch["conditioning_pixel_values"]
                # Fuse RGB to single channel, map to [0,1]
                mask_img = (mask_img.sum(1, keepdim=True) / 3 + 1) / 2
                # Downsample to token grid
                mask_lat = torch.nn.functional.interpolate(mask_img, size=(h_tokens, w_tokens), mode="area")
                force_keep_mask = mask_lat.flatten(2).squeeze(1) > 0.5  # (B, S)

        # Forward pass using diffusers interface
        # img_ids and txt_ids need to be 2D (S, 4) for the diffusers transformer
        output = self.model(
            hidden_states=packed_latents,
            encoder_hidden_states=txt,
            timestep=timesteps,
            img_ids=img_ids[0] if img_ids.ndim == 3 else img_ids,
            txt_ids=txt_ids[0] if txt_ids.ndim == 3 else txt_ids,
            guidance=guidance,
            return_dict=True,
            force_keep_mask=force_keep_mask,
        )

        # Extract sample from output
        model_pred = output.sample

        # Unpack: (B, S, C) -> (B, C, H, W)
        unpacked = unpack_latents(model_pred, img_ids)

        return {"model_prediction": unpacked}

    def encode_images(self, images: List[Tensor]) -> Tensor:
        """Encode images using FLUX.2 VAE."""
        device = self.vae.device
        dtype = self.vae.dtype

        # Stack and move to device
        if isinstance(images, list):
            images = torch.stack(images)
        images = images.to(device=device, dtype=dtype)

        # Encode
        with torch.no_grad():
            latents = self.vae.encode(images)

        return latents

    def decode_latents(self, latents: Tensor) -> Tensor:
        """Decode latents using FLUX.2 VAE."""
        with torch.no_grad():
            images = self.vae.decode(latents)
        return images

    def post_vae_encode_transform_sample(self, sample):
        """
        Ensure cached latents are patchified to match the transformer input expectations (128 channels).
        """
        if sample is None:
            return sample
        if hasattr(sample, "latent_dist"):
            sample = sample.latent_dist.sample()
        elif hasattr(sample, "sample"):
            sample = sample.sample
        if isinstance(sample, Tensor) and sample.dim() == 4:
            if sample.shape[1] == 32:
                sample = self._patchify_latents(sample)
            if sample.shape[1] == 128 and not self._latents_appear_normalized(sample):
                sample = self._normalize_latents(sample)
        return sample

    def get_loss_target(self, noise: Tensor, batch: dict) -> Tensor:
        """
        FLUX.2 uses flow matching: target is (noise - latents).
        """
        return (noise - batch["latents"]).detach()

    def check_user_config(self):
        """Validate FLUX.2 specific configuration."""
        super().check_user_config()

        # Override aspect bucket alignment for FLUX.2 (must be 16)
        if self.config.aspect_bucket_alignment != 16:
            logger.warning(
                f"FLUX.2 requires aspect_bucket_alignment=16, " f"overriding from {self.config.aspect_bucket_alignment}"
            )
            self.config.aspect_bucket_alignment = 16

        # Set tokenizer max length
        if self.config.tokenizer_max_length != MAX_SEQUENCE_LENGTH:
            logger.info(f"Setting tokenizer_max_length to {MAX_SEQUENCE_LENGTH} for FLUX.2")
            self.config.tokenizer_max_length = MAX_SEQUENCE_LENGTH

        # Validate guidance mode
        self.config.flux_guidance_mode = "constant"
        self.config.flux_guidance_value = 1.0

    @staticmethod
    def get_lora_target_modules(lora_target: str) -> List[str]:
        """Get LoRA target modules for FLUX.2 (diffusers naming)."""
        targets = {
            "all": [
                # Double stream attention
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_q_proj",
                "attn.add_k_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                # Double stream FF
                "ff.linear_in",
                "ff.linear_out",
                "ff_context.linear_in",
                "ff_context.linear_out",
                # Single stream (parallel attention + FF)
                "attn.to_qkv_mlp_proj",
                "attn.to_out",
            ],
            "attention": [
                "attn.to_q",
                "attn.to_k",
                "attn.to_v",
                "attn.to_out.0",
                "attn.add_q_proj",
                "attn.add_k_proj",
                "attn.add_v_proj",
                "attn.to_add_out",
                "attn.to_qkv_mlp_proj",
                "attn.to_out",
            ],
            "mlp": [
                "ff.linear_in",
                "ff.linear_out",
                "ff_context.linear_in",
                "ff_context.linear_out",
            ],
            "tiny": ["attn.to_q", "attn.to_k", "attn.to_v"],
        }
        return targets.get(lora_target, targets["all"])

    @classmethod
    def register_config_requirements(cls):
        """Register configuration rules for FLUX.2 model."""
        rules = [
            make_override_rule(
                field_name="aspect_bucket_alignment",
                value=16,
                message="FLUX.2 requires aspect bucket alignment of 16px",
                example="aspect_bucket_alignment: 16",
            ),
            ConfigRule(
                field_name="tokenizer_max_length",
                rule_type=RuleType.MAX,
                value=MAX_SEQUENCE_LENGTH,
                message=f"FLUX.2 supports a maximum of {MAX_SEQUENCE_LENGTH} tokens",
                example=f"tokenizer_max_length: {MAX_SEQUENCE_LENGTH}  # Maximum supported",
                error_level="warning",
            ),
            ConfigRule(
                field_name="base_model_precision",
                rule_type=RuleType.CHOICES,
                value=["int8-quanto", "fp8-torchao", "no_change", "int4-quanto", "nf4-torchao", "fp8-torchao-compile"],
                message="FLUX.2 supports limited precision options",
                example="base_model_precision: fp8-torchao",
                error_level="warning",
            ),
            ConfigRule(
                field_name="prediction_type",
                rule_type=RuleType.CUSTOM,
                value=None,
                message="FLUX.2 uses flow matching and does not support custom prediction types",
                error_level="warning",
            ),
        ]

        ConfigRegistry.register_rules("flux2", rules)
        ConfigRegistry.register_validator(
            "flux2",
            cls._validate_flux2_specific,
            """Validates FLUX.2-specific requirements:
- Warns about attention slicing on MPS devices
- Validates prediction_type compatibility
- Ensures proper aspect bucket alignment
- Checks tokenizer max length constraints""",
        )

    @staticmethod
    def _validate_flux2_specific(config: dict) -> List[ValidationResult]:
        """Custom validation logic for FLUX.2 models."""
        results = []

        # Check attention slicing on MPS
        if config.get("unet_attention_slice") and torch.backends.mps.is_available():
            results.append(
                ValidationResult(
                    passed=False,
                    field="unet_attention_slice",
                    message="Using attention slicing when training FLUX.2 on MPS can result in NaN errors",
                    level="warning",
                    suggestion="Disable attention slicing and reduce batch size instead to manage memory",
                )
            )

        # Check prediction type
        if config.get("prediction_type") is not None:
            results.append(
                ValidationResult(
                    passed=False,
                    field="prediction_type",
                    message="FLUX.2 does not support custom prediction types - it uses flow matching",
                    level="warning",
                    suggestion="Remove prediction_type from your configuration",
                )
            )

        return results


# Register the model
Flux2.register_config_requirements()
from simpletuner.helpers.models.registry import ModelRegistry

ModelRegistry.register("flux2", Flux2)
