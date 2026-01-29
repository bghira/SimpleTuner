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
from torch import Tensor

from simpletuner.helpers.acceleration import (
    AccelerationBackend,
    AccelerationPreset,
    get_bitsandbytes_presets,
    get_deepspeed_presets,
    get_quanto_presets,
    get_sdnq_presets,
    get_torchao_presets,
)
from simpletuner.helpers.configuration.registry import (
    ConfigRegistry,
    ConfigRule,
    RuleType,
    ValidationResult,
    make_override_rule,
)
from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.flux2 import build_flux2_conditioning_inputs, pack_latents, pack_text, unpack_latents
from simpletuner.helpers.models.flux2.autoencoder import AutoencoderKLFlux2
from simpletuner.helpers.models.flux2.pipeline import Flux2Pipeline
from simpletuner.helpers.models.flux2.transformer import Flux2Transformer2DModel
from simpletuner.helpers.models.tae.types import Flux2TAESpec
from simpletuner.helpers.musubi_block_swap import apply_musubi_pretrained_defaults

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

# FLUX.2-klein constants (uses Qwen3 text encoder with different layer config)
KLEIN_OUTPUT_LAYERS = (9, 18, 27)
KLEIN_FLAVOURS = {"klein-4b", "klein-9b"}

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
    """FLUX.2 model implementation for SimpleTuner.

    Supports:
    - dev: Full FLUX.2-dev with Mistral-3 text encoder (56 blocks, 12B params)
    - klein-9b: FLUX.2-klein 9B with Qwen3 text encoder (32 blocks)
    - klein-4b: FLUX.2-klein 4B with Qwen3 text encoder (25 blocks)

    Klein models do not have guidance embeddings (guidance_embeds=False).
    """

    SUPPORTS_MUON_CLIP = True

    NAME = "Flux.2"
    MODEL_DESCRIPTION = "FLUX.2 with Mistral-3 or Qwen3 text encoder"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTO_LORA_FORMAT_DETECTION = True
    NATIVE_COMFYUI_LORA_SUPPORT = True  # Flux2 has native ComfyUI LoRA support, no conversion needed
    AUTOENCODER_CLASS = AutoencoderKLFlux2
    LATENT_CHANNEL_COUNT = 128  # 32 VAE channels × 4 (2×2 pixel shuffle) = 128 transformer channels
    VAE_SCALE_FACTOR = 16  # 8x spatial + 2x pixel shuffle
    VALIDATION_USES_NEGATIVE_PROMPT = True  # Required for real CFG support
    VALIDATION_PREVIEW_SPEC = Flux2TAESpec()

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
    SLIDER_LORA_TARGET = [
        # Restrict to image/self-stream attention; avoid add_* context projections
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "attn.to_qkv_mlp_proj",
    ]
    DEFAULT_LYCORIS_TARGET = ["Flux2TransformerBlock", "Flux2SingleTransformerBlock"]

    MODEL_CLASS = Flux2Transformer2DModel
    MODEL_SUBFOLDER = "transformer"

    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: Flux2Pipeline,
        PipelineTypes.IMG2IMG: Flux2Pipeline,
    }

    DEFAULT_MODEL_FLAVOUR = "klein-9b"
    HUGGINGFACE_PATHS = {
        "dev": "black-forest-labs/FLUX.2-dev",
        "klein-4b": "black-forest-labs/FLUX.2-klein-base-4B",
        "klein-9b": "black-forest-labs/FLUX.2-klein-base-9B",
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

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        # FLUX.2 block counts vary by flavour:
        # - dev: 8 double + 48 single = 56 blocks
        # - klein-9b: 8 double + 24 single = 32 blocks
        # - klein-4b: 5 double + 20 single = 25 blocks
        # Leave at least 1 block on GPU
        if config is not None:
            flavour = getattr(config, "model_flavour", None)
            if flavour == "klein-4b":
                return 24  # 25 - 1
            elif flavour == "klein-9b":
                return 31  # 32 - 1
        return 55  # dev: 56 - 1

    @classmethod
    def get_acceleration_presets(cls) -> list[AccelerationPreset]:
        # Common settings for memory optimization presets
        _base_memory_config = {
            "base_model_precision": "no_change",
            "gradient_checkpointing": True,
        }

        return [
            # RamTorch presets - 3 levels (Light, Balanced, Aggressive)
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="light",
                name="RamTorch - Light",
                description="Streams double blocks only (8 of 56).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~15%",
                tradeoff_speed="Increases training time by ~10%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={
                    **_base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "transformer_blocks.*",
                },
            ),
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="balanced",
                name="RamTorch - Balanced",
                description="Streams double blocks + half of single blocks (32 of 56).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~45%",
                tradeoff_speed="Increases training time by ~30%",
                tradeoff_notes="Requires 128GB+ system RAM.",
                requires_min_system_ram_gb=128,
                config={
                    **_base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "transformer_blocks.*,single_transformer_blocks.0,single_transformer_blocks.1,single_transformer_blocks.2,single_transformer_blocks.3,single_transformer_blocks.4,single_transformer_blocks.5,single_transformer_blocks.6,single_transformer_blocks.7,single_transformer_blocks.8,single_transformer_blocks.9,single_transformer_blocks.10,single_transformer_blocks.11,single_transformer_blocks.12,single_transformer_blocks.13,single_transformer_blocks.14,single_transformer_blocks.15,single_transformer_blocks.16,single_transformer_blocks.17,single_transformer_blocks.18,single_transformer_blocks.19,single_transformer_blocks.20,single_transformer_blocks.21,single_transformer_blocks.22,single_transformer_blocks.23",
                },
            ),
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="aggressive",
                name="RamTorch - Aggressive",
                description="Streams all transformer blocks (56 of 56).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~85%",
                tradeoff_speed="Increases training time by ~75%",
                tradeoff_notes="Requires 128GB+ system RAM.",
                requires_min_system_ram_gb=128,
                config={
                    **_base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "transformer_blocks.*,single_transformer_blocks.*",
                },
            ),
            # Block Swap presets - 3 levels (Light, Balanced, Aggressive)
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="light",
                name="Block Swap - Light",
                description="Swaps 14 of 56 blocks (~25%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~22%",
                tradeoff_speed="Increases training time by ~15%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={**_base_memory_config, "musubi_blocks_to_swap": 14},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="balanced",
                name="Block Swap - Balanced",
                description="Swaps 28 of 56 blocks (~50%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~45%",
                tradeoff_speed="Increases training time by ~35%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={**_base_memory_config, "musubi_blocks_to_swap": 28},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="aggressive",
                name="Block Swap - Aggressive",
                description="Swaps 47 of 56 blocks (~84%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~75%",
                tradeoff_speed="Increases training time by ~65%",
                tradeoff_notes="Requires 128GB+ system RAM.",
                requires_min_system_ram_gb=128,
                config={**_base_memory_config, "musubi_blocks_to_swap": 47},
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mistral for dev
        self._mistral_model = None
        self._mistral_processor = None
        # Qwen3 for Klein
        self._qwen_model = None
        self._qwen_tokenizer = None

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
    def _unpatchify_latents(latents: Tensor) -> Tensor:
        """
        Reverse pixel-shuffle latents from (B, 4C, H/2, W/2) -> (B, C, H, W).
        Used to convert 128-channel transformer latents back to 32-channel VAE format.
        """
        b, c4, h2, w2 = latents.shape
        c = c4 // 4
        latents = latents.view(b, c, 2, 2, h2, w2)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(b, c, h2 * 2, w2 * 2)
        return latents

    def pre_latent_decode(self, latents: Tensor) -> Tensor:
        """
        Pre-process latents before passing to TAEF2 decoder.
        Unpatchifies from 128-channel to 32-channel format.
        """
        if latents.shape[1] == 128:
            return self._unpatchify_latents(latents)
        return latents

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

    def _is_klein_flavour(self) -> bool:
        """Check if current model flavour is a Klein variant."""
        flavour = getattr(self.config, "model_flavour", None)
        return flavour in KLEIN_FLAVOURS

    def load_text_encoder(self, move_to_device: bool = True):
        """Load the text encoder (Qwen3 for Klein, Mistral-3 for dev)."""
        if self._is_klein_flavour():
            return self._load_text_encoder_qwen3(move_to_device)
        return self._load_text_encoder_mistral(move_to_device)

    def _load_text_encoder_qwen3(self, move_to_device: bool = True):
        """Load the Qwen3 text encoder for Klein models."""
        try:
            from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM
        except ImportError:
            logger.error("Qwen3ForCausalLM not found. Please install transformers>=4.45.0")
            sys.exit(1)

        dtype = self.config.weight_dtype
        text_encoder_precision = getattr(self.config, "text_encoder_1_precision", "no_change")
        should_quantize_text_encoder = text_encoder_precision not in (None, "no_change")
        quantize_via_cpu = getattr(self.config, "quantize_via", None) == "cpu"

        # For Klein models, text encoder is bundled in the model repo under "text_encoder" subfolder
        # and tokenizer is in a separate "tokenizer" subfolder
        model_path = self.config.pretrained_model_name_or_path
        text_encoder_path = getattr(self.config, "pretrained_text_encoder_model_name_or_path", None)
        if text_encoder_path is None:
            text_encoder_path = model_path
            text_encoder_subfolder = "text_encoder"
            tokenizer_subfolder = "tokenizer"
        else:
            text_encoder_subfolder = None
            tokenizer_subfolder = None

        revision = getattr(self.config, "text_encoder_revision", None) or getattr(self.config, "revision", None)

        logger.info(f"Loading Qwen3 text encoder from {text_encoder_path}...")

        # Load tokenizer (from separate tokenizer subfolder in Klein models)
        tokenizer_kwargs = {"subfolder": tokenizer_subfolder} if tokenizer_subfolder else {}
        if revision is not None:
            tokenizer_kwargs["revision"] = revision
        self._qwen_tokenizer = Qwen2TokenizerFast.from_pretrained(text_encoder_path, **tokenizer_kwargs)

        # Load model
        model_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if text_encoder_subfolder:
            model_kwargs["subfolder"] = text_encoder_subfolder
        if revision is not None:
            model_kwargs["revision"] = revision
        self._qwen_model = Qwen3ForCausalLM.from_pretrained(text_encoder_path, **model_kwargs)

        if move_to_device and not self._ramtorch_text_encoders_requested():
            target_device = (
                torch.device("cpu") if quantize_via_cpu and should_quantize_text_encoder else self.accelerator.device
            )
            self._qwen_model.to(target_device, dtype=dtype)
        if self._ramtorch_text_encoders_requested():
            self._apply_ramtorch_layers(self._qwen_model, "text_encoder_1", percent=self._ramtorch_text_encoder_percent())
        self._qwen_model.requires_grad_(False)
        self._qwen_model.eval()

        # Store in standard location for compatibility
        self.text_encoders = [self._qwen_model]
        self.tokenizers = [self._qwen_tokenizer]

        logger.info("Qwen3 text encoder loaded successfully")

    def _load_text_encoder_mistral(self, move_to_device: bool = True):
        """Load the Mistral-3 text encoder."""
        try:
            from transformers import AutoProcessor, Mistral3ForConditionalGeneration
        except ImportError:
            logger.error("Mistral3ForConditionalGeneration not found. Please install transformers>=4.45.0")
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
        model_kwargs = {
            "torch_dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if mistral_revision is not None:
            model_kwargs["revision"] = mistral_revision
        self._mistral_model = Mistral3ForConditionalGeneration.from_pretrained(
            mistral_path,
            **model_kwargs,
        )
        if move_to_device and not self._ramtorch_text_encoders_requested():
            target_device = (
                torch.device("cpu") if quantize_via_cpu and should_quantize_text_encoder else self.accelerator.device
            )
            self._mistral_model.to(target_device, dtype=dtype)
        if self._ramtorch_text_encoders_requested():
            self._apply_ramtorch_layers(self._mistral_model, "text_encoder_1", percent=self._ramtorch_text_encoder_percent())
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
        Encode prompts using the appropriate text encoder.

        Returns:
            prompt_embeds: (B, L, D) - stacked outputs from hidden layers
            attention_mask: (B, L)
        """
        if self._is_klein_flavour():
            return self._encode_prompts_qwen3(prompts, is_negative_prompt)
        return self._encode_prompts_mistral(prompts, is_negative_prompt)

    def _get_text_encoder_layers(self) -> tuple:
        """Get the layer indices to extract from text encoder hidden states.

        Returns configured custom layers if set, otherwise model defaults.
        """
        custom_layers = getattr(self.config, "custom_text_encoder_intermediary_layers", None)
        if custom_layers is not None:
            # Parse JSON array if it's a string
            if isinstance(custom_layers, str):
                import json

                try:
                    custom_layers = json.loads(custom_layers)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid custom_text_encoder_intermediary_layers format: {custom_layers}, using defaults"
                    )
                    custom_layers = None
            if custom_layers is not None:
                return tuple(custom_layers)

        # Return model-specific defaults
        if self._is_klein_flavour():
            return KLEIN_OUTPUT_LAYERS
        return tuple(OUTPUT_LAYERS)

    def _encode_prompts_qwen3(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode prompts using Qwen3 for Klein models.

        Returns:
            prompt_embeds: (B, L, 7680 or 12288) - stacked outputs from configured layers
            attention_mask: (B, L)
        """
        device = self.accelerator.device

        if not isinstance(prompts, list):
            prompts = [prompts]

        all_input_ids = []
        all_attention_masks = []

        for single_prompt in prompts:
            # Format as chat messages (no system message for Klein)
            messages = [{"role": "user", "content": single_prompt}]
            text = self._qwen_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = self._qwen_tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=MAX_SEQUENCE_LENGTH,
            )
            all_input_ids.append(inputs["input_ids"])
            all_attention_masks.append(inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

        # Forward pass
        with torch.no_grad():
            output = self._qwen_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        # Stack outputs from configured layers
        output_layers = self._get_text_encoder_layers()
        out = torch.stack([output.hidden_states[k] for k in output_layers], dim=1)
        prompt_embeds = rearrange(out, "b c l d -> b l (c d)")

        return prompt_embeds, attention_mask

    def _encode_prompts_mistral(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode prompts using Mistral-3 for dev models.

        Returns:
            prompt_embeds: (B, L, 15360) - stacked outputs from configured layers
            attention_mask: (B, L)
        """
        device = self.accelerator.device

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

        # Stack outputs from configured layers
        output_layers = self._get_text_encoder_layers()
        out = torch.stack([output.hidden_states[k] for k in output_layers], dim=1)
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
        # Check if real CFG is needed (either via validation_guidance or validation_guidance_real)
        validation_guidance = getattr(self.config, "validation_guidance", None)
        validation_guidance_real = getattr(self.config, "validation_guidance_real", 1.0)
        needs_cfg = (validation_guidance is not None and validation_guidance > 1.0) or validation_guidance_real > 1.0
        if not needs_cfg:
            return {}

        prompt_embeds = text_embedding["prompt_embeds"]
        text_ids = text_embedding.get("text_ids")

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if text_ids is not None and text_ids.dim() == 2:
            text_ids = text_ids.unsqueeze(0)

        result = {
            "negative_prompt_embeds": prompt_embeds,
        }
        if text_ids is not None:
            result["negative_text_ids"] = text_ids
        return result

    def requires_conditioning_latents(self) -> bool:
        """
        FLUX.2 reference image conditioning uses latents (not pixels).
        This returns True so that when a conditioning dataset is configured,
        collate.py will collect VAE-encoded latents instead of pixel values.
        """
        # Always return True because FLUX.2 reference conditioning uses latents.
        # If no conditioning dataset is configured, this has no effect.
        # If controlnet is configured, base class already returns True.
        return True

    def supports_conditioning_dataset(self) -> bool:
        """
        FLUX.2 optionally supports reference image conditioning for dual T2I/I2I training.

        Unlike Flux Kontext which *requires* conditioning inputs, FLUX.2 can operate in
        either text-to-image mode (no conditioning) or image-to-image mode (with reference
        images). This allows the WebUI to show conditioning dataset options without
        making them mandatory.
        """
        return True

    def prepare_batch_conditions(self, batch: dict, state: dict):
        """
        Prepare conditioning inputs for FLUX.2 reference image conditioning.

        This builds packed conditioning latents with time-offset position IDs
        when conditioning_latents are present in the batch.
        """
        cond = batch.get("conditioning_latents")
        if cond is None:
            logger.debug("No conditioning latents found for FLUX.2")
            return super().prepare_batch_conditions(batch=batch, state=state)

        # Check sampling mode
        sampling_mode = state.get("args", {}).get("conditioning_multidataset_sampling", "random")

        if sampling_mode == "random" and isinstance(cond, list) and len(cond) >= 1:
            # Random mode should have selected just one conditioning set
            cond = cond[0]

        # Flatten nested lists that may result from check_latent_shapes returning
        # per-sample lists when conditioning images have different aspect ratios
        if isinstance(cond, list):
            flat_cond = []
            for item in cond:
                if isinstance(item, list):
                    flat_cond.extend(item)
                else:
                    flat_cond.append(item)
            cond = flat_cond if flat_cond else cond

        if isinstance(cond, list):
            logger.debug(f"FLUX.2 conditioning inputs shapes: {[d.shape for d in cond]} {cond[0].dtype}")
        else:
            logger.debug(f"FLUX.2 conditioning inputs shape: {cond.shape} {cond.dtype}")

        # Build FLUX.2 conditioning inputs with time-offset position IDs
        packed_cond, cond_ids = build_flux2_conditioning_inputs(
            cond if isinstance(cond, list) else [cond],
            dtype=self.config.weight_dtype,
            device=self.accelerator.device,
            latent_channels=self.LATENT_CHANNEL_COUNT,
        )
        logger.debug(f"FLUX.2 packed conditioning shape: {packed_cond.shape} {packed_cond.dtype}")

        batch["conditioning_packed_latents"] = packed_cond
        batch["conditioning_ids"] = cond_ids

        return super().prepare_batch_conditions(batch=batch, state=state)

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

        hidden_states_buffer = self._new_hidden_state_buffer()
        # Build force_keep_mask for TREAD if using mask/segmentation conditioning
        force_keep_mask = None
        if (
            getattr(self.config, "tread_config", None) is not None
            and "conditioning_pixel_values" in prepared_batch
            and prepared_batch["conditioning_pixel_values"] is not None
            and prepared_batch.get("loss_mask_type") in ("mask", "segmentation")
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

        # Pull optional reference image conditioning inputs
        cond_seq = prepared_batch.get("conditioning_packed_latents")
        cond_ids = prepared_batch.get("conditioning_ids")
        use_cond = cond_seq is not None
        logger.debug(f"FLUX.2 using conditioning: {use_cond}")

        # Concatenate conditioning with noisy latents if present
        if use_cond:
            lat_in = torch.cat([packed_latents, cond_seq], dim=1)
            id_in = torch.cat([img_ids, cond_ids], dim=1)
        else:
            lat_in = packed_latents
            id_in = img_ids

        # Forward pass using diffusers interface
        # img_ids and txt_ids need to be 2D (S, 4) for the diffusers transformer
        timestep_sign = prepared_batch.get("twinflow_time_sign") if getattr(self.config, "twinflow_enabled", False) else None
        output = self.model(
            hidden_states=lat_in,
            encoder_hidden_states=txt,
            timestep=timesteps,
            timestep_sign=timestep_sign,
            img_ids=id_in[0] if id_in.ndim == 3 else id_in,
            txt_ids=txt_ids[0] if txt_ids.ndim == 3 else txt_ids,
            guidance=guidance,
            return_dict=True,
            force_keep_mask=force_keep_mask,
            hidden_states_buffer=hidden_states_buffer,
        )

        # Extract sample from output
        model_pred = output.sample

        # Drop reference image tokens from output before unpacking
        if use_cond:
            scene_seq_len = packed_latents.shape[1]  # tokens that belong to the main image
            model_pred = model_pred[:, :scene_seq_len, :]  # (B, S_scene, C)

        # Unpack: (B, S, C) -> (B, C, H, W)
        unpacked = unpack_latents(model_pred, img_ids)

        return {"model_prediction": unpacked, "hidden_states_buffer": hidden_states_buffer}

    @torch.no_grad()
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
        Patchify (32 -> 128 channels) and normalize cached latents for transformer input.
        """
        if sample is None:
            return sample
        if hasattr(sample, "latent_dist"):
            # Use mode() (mean) not sample() to match upstream training
            sample = sample.latent_dist.mode()
        elif hasattr(sample, "sample"):
            sample = sample.sample
        if isinstance(sample, Tensor) and sample.dim() == 4:
            if sample.shape[1] == 32:
                sample = self._patchify_latents(sample)
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

        # Guidance is always 1.0 for training (different from inference default of 3.5)
        self.config.flux_guidance_mode = "constant"
        self.config.flux_guidance_value = 1.0

        # For Klein flavours (non-distilled), move validation_guidance to validation_guidance_real
        # The Flux2 pipeline only enables "true" CFG when validation_guidance_real is set
        # Only do this if validation_guidance_real is at its default (1.0) to avoid overwriting user intent
        flavour = getattr(self.config, "model_flavour", "") or ""
        if self._is_klein_flavour() and "distilled" not in flavour:
            validation_guidance = getattr(self.config, "validation_guidance", None)
            validation_guidance_real = getattr(self.config, "validation_guidance_real", 1.0)
            if validation_guidance is not None and validation_guidance_real == 1.0:
                logger.info(
                    f"Klein model detected: moving validation_guidance={validation_guidance} "
                    "to validation_guidance_real for true CFG support"
                )
                self.config.validation_guidance_real = validation_guidance
                self.config.validation_guidance = None

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

    def pretrained_load_args(self, pretrained_load_args: dict) -> dict:
        args = super().pretrained_load_args(pretrained_load_args)
        args = apply_musubi_pretrained_defaults(self.config, args)
        return args

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
