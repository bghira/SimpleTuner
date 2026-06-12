import logging
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from transformers import AutoTokenizer, T5GemmaModel

from simpletuner.helpers.acceleration import (
    AccelerationBackend,
    AccelerationPreset,
    get_bitsandbytes_presets,
    get_deepspeed_presets,
    get_quanto_presets,
    get_sdnq_presets,
    get_torchao_presets,
)
from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.musubi_block_swap import apply_musubi_pretrained_defaults
from simpletuner.helpers.models.zlab_i1.latent_utils import (
    normalize_flux2_latents,
    pixel_shuffle_2x,
    pixel_unshuffle_2x,
    unscale_flux2_latents,
)
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.models.zlab_i1.pipeline import ZlabI1Pipeline
from simpletuner.helpers.models.zlab_i1.transformer import ZlabI1Transformer2DModel

logger = logging.getLogger(__name__)


class ZLabI1(ImageModelFoundation):
    NAME = "zlab i1"
    MODEL_DESCRIPTION = "zlab-princeton i1 flow-matching transformer"
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKL
    AUTOENCODER_SCALING_FACTOR = 1.0
    LATENT_CHANNEL_COUNT = 32
    MODEL_CLASS = ZlabI1Transformer2DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: ZlabI1Pipeline,
    }
    DEFAULT_LORA_TARGET = [
        "qkv_image",
        "qkv_text",
        "proj_image",
        "proj_text",
        "w12",
        "w3",
        "linear",
    ]
    SLIDER_LORA_TARGET = DEFAULT_LORA_TARGET
    DEFAULT_MODEL_FLAVOUR = "3b"
    HUGGINGFACE_PATHS = {
        "3b": "bghira/zlab-i1-diffusers",
    }

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "T5Gemma 2B",
            "path": "google/t5gemma-2b-2b-ul2-it",
            "tokenizer": AutoTokenizer,
            "tokenizer_subfolder": "",
            "use_fast": False,
            "model": T5GemmaModel,
            "subfolder": "",
        },
    }

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        return 28

    @classmethod
    def get_acceleration_presets(cls) -> list[AccelerationPreset]:
        base_memory_config = {
            "base_model_precision": "no_change",
            "gradient_checkpointing": False,
        }
        return [
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="balanced",
                name="RamTorch - Balanced",
                description="Streams the output half of the transformer blocks from CPU RAM.",
                tab="basic",
                tradeoff_vram="Reduces transformer VRAM use.",
                tradeoff_speed="Increases training time.",
                tradeoff_notes="Requires substantial system RAM for the 3B checkpoint.",
                requires_min_system_ram_gb=64,
                config={
                    **base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "out_blocks.*,mid_block.*",
                },
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="conservative",
                name="Musubi Block Swap - Conservative",
                description="Streams the last 7 i1 transformer blocks from CPU during forward passes.",
                tab="basic",
                tradeoff_vram="Reduces transformer residency in VRAM.",
                tradeoff_speed="Adds CPU/GPU transfer overhead.",
                tradeoff_notes="Mutually exclusive with RamTorch and group offload.",
                requires_min_system_ram_gb=64,
                config={**base_memory_config, "musubi_blocks_to_swap": 7},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="balanced",
                name="Musubi Block Swap - Balanced",
                description="Streams the last 14 i1 transformer blocks from CPU during forward passes.",
                tab="basic",
                tradeoff_vram="Reduces transformer residency in VRAM more aggressively.",
                tradeoff_speed="Adds more CPU/GPU transfer overhead.",
                tradeoff_notes="Mutually exclusive with RamTorch and group offload.",
                requires_min_system_ram_gb=96,
                config={**base_memory_config, "musubi_blocks_to_swap": 14},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="aggressive",
                name="Musubi Block Swap - Aggressive",
                description="Streams the last 28 i1 transformer blocks from CPU during forward passes.",
                tab="advanced",
                tradeoff_vram="Minimizes resident transformer blocks.",
                tradeoff_speed="Largest transfer overhead.",
                tradeoff_notes="Use only when VRAM is the limiting factor.",
                requires_min_system_ram_gb=128,
                config={**base_memory_config, "musubi_blocks_to_swap": 28},
            ),
            *get_deepspeed_presets(base_memory_config),
            *get_sdnq_presets(base_memory_config),
            *get_torchao_presets(base_memory_config),
            *get_quanto_presets(base_memory_config),
            *get_bitsandbytes_presets(base_memory_config),
        ]

    def setup_model_flavour(self):
        super().setup_model_flavour()
        if getattr(self.config, "vae_path", None) in (None, "", "None"):
            self.config.vae_path = "black-forest-labs/FLUX.2-dev"
        if getattr(self.config, "pretrained_vae_model_name_or_path", None) in (None, "", "None"):
            self.config.pretrained_vae_model_name_or_path = "black-forest-labs/FLUX.2-dev"

    def pretrained_load_args(self, pretrained_load_args: dict) -> dict:
        args = super().pretrained_load_args(pretrained_load_args)
        return apply_musubi_pretrained_defaults(self.config, args)

    @staticmethod
    def _pixel_unshuffle_2x(latents: torch.Tensor) -> torch.Tensor:
        return pixel_unshuffle_2x(latents)

    @staticmethod
    def _pixel_shuffle_2x(latents: torch.Tensor) -> torch.Tensor:
        return pixel_shuffle_2x(latents)

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return normalize_flux2_latents(latents)

    def pre_latent_decode(self, latents: torch.Tensor) -> torch.Tensor:
        return unscale_flux2_latents(latents)

    def post_vae_encode_transform_sample(self, sample):
        if hasattr(sample, "latent_dist"):
            sample = sample.latent_dist.mode()
        elif hasattr(sample, "sample"):
            sample = sample.sample
        if not torch.is_tensor(sample):
            return sample
        if sample.dim() != 4 or sample.shape[1] != self.LATENT_CHANNEL_COUNT:
            raise ValueError(
                f"{self.NAME} expects {self.LATENT_CHANNEL_COUNT}-channel VAE latents, got {tuple(sample.shape)}."
            )
        return self._normalize_latents(sample)

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        tokenizer = self.tokenizers[0]
        text_encoder = self.text_encoders[0]
        encoder = getattr(text_encoder, "encoder", text_encoder)
        max_length = getattr(self.config, "tokenizer_max_length", None) or 256
        tokenized = tokenizer(
            prompts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        inputs = {key: value.to(self.accelerator.device) for key, value in tokenized.items()}
        with torch.no_grad():
            outputs = encoder(**inputs)
        hidden_states = outputs.last_hidden_state.float()
        attention_mask = inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        return {
            "prompt_embeds": hidden_states,
            "attention_mask": attention_mask.bool(),
        }

    def convert_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        return {
            "prompt_embeds": text_embedding["prompt_embeds"],
            "attention_mask": text_embedding["attention_mask"],
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        return {
            "negative_prompt_embeds": text_embedding["prompt_embeds"],
            "negative_attention_mask": text_embedding["attention_mask"],
        }

    def tread_init(self):
        from simpletuner.helpers.training.tread import TREADRouter

        tread_cfg = getattr(self.config, "tread_config", None)
        if not isinstance(tread_cfg, dict) or not tread_cfg or tread_cfg.get("routes") is None:
            logger.error("TREAD training requires you to configure the routes in the TREAD config")
            import sys

            sys.exit(1)

        self.unwrap_model(model=self.model).set_router(
            TREADRouter(
                seed=getattr(self.config, "seed", None) or 42,
                device=self.accelerator.device,
            ),
            tread_cfg["routes"],
        )
        logger.info("TREAD training is enabled for zlab i1")

    def supports_crepa_self_flow(self) -> bool:
        return True

    def _prepare_crepa_self_flow_batch(self, batch: dict, state: dict) -> dict:
        return self._prepare_image_crepa_self_flow_batch(batch=batch, state=state, patch_size=2)

    def _select_crepa_hidden_states(self, prepared_batch: dict, hidden_states_buffer: Optional[dict]):
        if hidden_states_buffer is None:
            return None
        capture_idx = prepared_batch.get("crepa_capture_block_index")
        if capture_idx is None:
            crepa = getattr(self, "crepa_regularizer", None)
            capture_idx = getattr(crepa, "block_index", None) if crepa is not None else None
        if capture_idx is None:
            return None
        return hidden_states_buffer.get(f"layer_{capture_idx}")

    def _build_tread_force_keep_mask(self, prepared_batch: dict, latents: torch.Tensor) -> Optional[torch.Tensor]:
        if (
            getattr(self.config, "tread_config", None) is None
            or "conditioning_pixel_values" not in prepared_batch
            or prepared_batch["conditioning_pixel_values"] is None
            or prepared_batch.get("loss_mask_type") not in ("mask", "segmentation")
        ):
            return None

        with torch.no_grad():
            mask_img = prepared_batch["conditioning_pixel_values"].to(device=latents.device, dtype=latents.dtype)
            if mask_img.ndim != 4:
                raise ValueError(f"conditioning_pixel_values must have shape [batch, channels, height, width], got {tuple(mask_img.shape)}.")
            if mask_img.shape[0] != latents.shape[0]:
                raise ValueError(
                    "conditioning_pixel_values batch size must match noisy_latents batch size, "
                    f"got {mask_img.shape[0]} and {latents.shape[0]}."
                )
            patch_size = self.model.patch_size
            h_tokens = latents.shape[2] // patch_size
            w_tokens = latents.shape[3] // patch_size
            if prepared_batch.get("loss_mask_type") == "segmentation":
                mask_img = torch.sum(mask_img, dim=1, keepdim=True) / mask_img.shape[1]
            else:
                mask_img = mask_img[:, :1]
            mask_img = mask_img / 2 + 0.5
            mask_lat = F.interpolate(mask_img, size=(h_tokens, w_tokens), mode="area")
            return mask_lat.flatten(2).squeeze(1) > 0.5

    def model_predict(self, prepared_batch, custom_timesteps: list = None):
        latents = prepared_batch["noisy_latents"]
        if latents.shape[1] != self.LATENT_CHANNEL_COUNT:
            raise ValueError(
                f"{self.NAME} requires {self.LATENT_CHANNEL_COUNT}-channel latents. "
                "Use the FLUX.2 VAE cache for this model family."
            )

        prompt_embeds = prepared_batch.get("prompt_embeds")
        if prompt_embeds is None:
            prompt_embeds = prepared_batch.get("encoder_hidden_states")
        if prompt_embeds is None:
            raise ValueError("prompt_embeds are required for i1 training.")
        if prompt_embeds.shape[0] != latents.shape[0]:
            raise ValueError(
                f"prompt_embeds batch size must match noisy_latents batch size, got {prompt_embeds.shape[0]} and {latents.shape[0]}."
            )

        attention_mask = prepared_batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = prepared_batch.get("attention_masks")
        if attention_mask is None:
            attention_mask = prepared_batch.get("encoder_attention_mask")
        if attention_mask is None:
            raise ValueError("attention_mask is required for i1 training.")
        if attention_mask.ndim == 3 and attention_mask.shape[1] == 1:
            attention_mask = attention_mask[:, 0]
        elif attention_mask.ndim != 2:
            raise ValueError(f"attention_mask must have shape [batch, tokens], got {tuple(attention_mask.shape)}.")
        if attention_mask.shape[0] != latents.shape[0]:
            raise ValueError(
                f"attention_mask batch size must match noisy_latents batch size, got {attention_mask.shape[0]} and {latents.shape[0]}."
            )

        timesteps = prepared_batch["timesteps"].to(device=self.accelerator.device, dtype=self.config.base_weight_dtype)
        if timesteps.shape[0] != latents.shape[0]:
            raise ValueError(f"timesteps batch size must match noisy_latents batch size, got {timesteps.shape[0]} and {latents.shape[0]}.")
        if timesteps.ndim > 1:
            timesteps = timesteps[:, 0]

        hidden_states_buffer = self._new_hidden_state_buffer()
        latents = latents.to(device=self.accelerator.device, dtype=self.config.base_weight_dtype)
        model_pred = self.model(
            latents,
            timesteps,
            prompt_embeds.to(device=self.accelerator.device, dtype=self.config.base_weight_dtype),
            attention_mask.to(device=self.accelerator.device),
            force_keep_mask=self._build_tread_force_keep_mask(prepared_batch, latents),
            hidden_states_buffer=hidden_states_buffer,
        )

        return {
            "model_prediction": model_pred.float(),
            "crepa_hidden_states": self._select_crepa_hidden_states(prepared_batch, hidden_states_buffer),
            "hidden_states_buffer": hidden_states_buffer,
        }

    def get_loss_target(self, noise: torch.Tensor, batch: dict) -> torch.Tensor:
        return (batch["latents"] - noise).detach()

    def get_prediction_target(self, prepared_batch: dict):
        if prepared_batch.get("target") is not None:
            return prepared_batch["target"]
        return (prepared_batch["latents"] - prepared_batch["noise"]).detach()

    def setup_training_noise_schedule(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        self.noise_schedule = FlowMatchEulerDiscreteScheduler.from_pretrained(
            "black-forest-labs/FLUX.2-dev",
            subfolder="scheduler",
            shift=self.config.flow_schedule_shift,
        )
        self.config.training_scheduler_timestep_spacing = self.noise_schedule.config.get("timestep_spacing")
        self.config.prediction_type = self.PREDICTION_TYPE.value
        self.config.rescale_betas_zero_snr = False
        return self.config, self.noise_schedule

    def check_user_config(self):
        super().check_user_config()
        if getattr(self.config, "validation_noise_scheduler", None) is None:
            self.config.validation_noise_scheduler = "flow_matching"


ModelRegistry.register("zlab_i1", ZLabI1)
