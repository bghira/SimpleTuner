import inspect
import logging
from typing import List, Optional

import torch
from diffusers import AutoencoderKL
from transformers import AutoModelForCausalLM, AutoTokenizer

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
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.models.tae.types import ImageTAESpec
from simpletuner.helpers.models.z_image.pipeline import ZImagePipeline
from simpletuner.helpers.models.z_image.transformer import ZImageTransformer2DModel

logger = logging.getLogger(__name__)


class ZImage(ImageModelFoundation):
    SUPPORTS_MUON_CLIP = True
    NAME = "Z-Image"
    MODEL_DESCRIPTION = "Z-Image flow-matching transformer"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 16
    VALIDATION_PREVIEW_SPEC = ImageTAESpec(repo_id="madebyollin/taef1")
    SLIDER_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]

    MODEL_CLASS = ZImageTransformer2DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: ZImagePipeline,
    }
    ASSISTANT_LORA_FLAVOURS = ["turbo", "turbo-ostris-v2"]
    ASSISTANT_LORA_PATH = "ostris/zimage_turbo_training_adapter"
    ASSISTANT_LORA_WEIGHT_NAME = "zimage_turbo_training_adapter_v1.safetensors"
    ASSISTANT_LORA_WEIGHT_NAMES = {
        "turbo": "zimage_turbo_training_adapter_v1.safetensors",
        "turbo-ostris-v2": "zimage_turbo_training_adapter_v2.safetensors",
    }

    TRANSFORMER_PATH_OVERRIDES = {
        "ostris-de-turbo": "ostris/Z-Image-De-Turbo",
    }

    # We do not bundle a default HF path; users must point at a released checkpoint.
    HUGGINGFACE_PATHS: dict = {
        "base": "TONGYI-MAI/Z-Image",
        "turbo": "TONGYI-MAI/Z-Image-Turbo",
        "turbo-ostris-v2": "TONGYI-MAI/Z-Image-Turbo",
        "ostris-de-turbo": "TONGYI-MAI/Z-Image-Turbo",
    }
    DEFAULT_MODEL_FLAVOUR = "turbo-ostris-v2"

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        # Z-Image has 30 transformer layers
        return 29

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
                description="Offloads half of transformer layers to CPU RAM.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~30%",
                tradeoff_speed="Increases training time by ~20%",
                tradeoff_notes="Requires 32GB+ system RAM.",
                requires_min_system_ram_gb=32,
                config={
                    **_base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "layers.15.*,layers.16.*,layers.17.*,layers.18.*,layers.19.*,layers.20.*,layers.21.*,layers.22.*,layers.23.*,layers.24.*,layers.25.*,layers.26.*,layers.27.*,layers.28.*,layers.29.*",
                },
            ),
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="balanced",
                name="RamTorch - Balanced",
                description="Offloads most transformer layers, keeping first 8 on GPU.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~45%",
                tradeoff_speed="Increases training time by ~35%",
                tradeoff_notes="Requires 48GB+ system RAM.",
                requires_min_system_ram_gb=48,
                config={
                    **_base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "layers.8.*,layers.9.*,layers.10.*,layers.11.*,layers.12.*,layers.13.*,layers.14.*,layers.15.*,layers.16.*,layers.17.*,layers.18.*,layers.19.*,layers.20.*,layers.21.*,layers.22.*,layers.23.*,layers.24.*,layers.25.*,layers.26.*,layers.27.*,layers.28.*,layers.29.*",
                },
            ),
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="aggressive",
                name="RamTorch - Aggressive",
                description="Offloads all transformer layers to CPU RAM.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~60%",
                tradeoff_speed="Increases training time by ~50%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={
                    **_base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "layers.*",
                },
            ),
            # Block Swap presets (Basic tab) - 3 levels for 6B model
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="conservative",
                name="Block Swap - Conservative",
                description="Swaps 10 of 30 blocks between GPU and CPU.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~25%",
                tradeoff_speed="Increases training time by ~15%",
                tradeoff_notes="Requires 32GB+ system RAM.",
                requires_min_system_ram_gb=32,
                config={**_base_memory_config, "musubi_blocks_to_swap": 10},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="balanced",
                name="Block Swap - Balanced",
                description="Swaps 15 of 30 blocks between GPU and CPU.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~40%",
                tradeoff_speed="Increases training time by ~25%",
                tradeoff_notes="Requires 48GB+ system RAM.",
                requires_min_system_ram_gb=48,
                config={**_base_memory_config, "musubi_blocks_to_swap": 15},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="aggressive",
                name="Block Swap - Aggressive",
                description="Swaps 22 of 30 blocks between GPU and CPU.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~55%",
                tradeoff_speed="Increases training time by ~40%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={**_base_memory_config, "musubi_blocks_to_swap": 22},
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
            "name": "Qwen3 4B",
            "tokenizer": AutoTokenizer,
            "tokenizer_subfolder": "tokenizer",
            "model": AutoModelForCausalLM,
            "subfolder": "text_encoder",
        },
    }

    def setup_model_flavour(self):
        super().setup_model_flavour()
        flavour = getattr(self.config, "model_flavour", None)
        override_map = getattr(self, "TRANSFORMER_PATH_OVERRIDES", {})
        if getattr(self.config, "pretrained_transformer_model_name_or_path", None) is None and flavour in override_map:
            self.config.pretrained_transformer_model_name_or_path = override_map[flavour]

    def tread_init(self):
        """Initialize the TREAD router when training with token routing enabled."""
        from simpletuner.helpers.training.tread import TREADRouter

        tread_cfg = getattr(self.config, "tread_config", None)
        if not isinstance(tread_cfg, dict) or tread_cfg == {} or tread_cfg.get("routes") is None:
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

        logger.info("TREAD training is enabled for Z-Image")

    def post_model_load_setup(self):
        super().post_model_load_setup()
        self._maybe_load_assistant_lora()

    def _assistant_lora_weight_for_flavour(self):
        weight_map = getattr(self, "ASSISTANT_LORA_WEIGHT_NAMES", None) or {}
        flavour = getattr(self.config, "model_flavour", None)
        if isinstance(weight_map, dict) and flavour in weight_map:
            return weight_map.get(flavour)
        return getattr(self, "ASSISTANT_LORA_WEIGHT_NAME", None)

    def _maybe_load_assistant_lora(self):
        if getattr(self.config, "disable_assistant_lora", False):
            return
        if not self.supports_assistant_lora(self.config):
            return
        if getattr(self.config, "model_type", "").lower() != "lora":
            return

        assistant_path = getattr(self.config, "assistant_lora_path", None) or self.ASSISTANT_LORA_PATH
        if not assistant_path:
            return

        from simpletuner.helpers.assistant_lora import load_assistant_adapter

        weight_name = getattr(self.config, "assistant_lora_weight_name", None) or self._assistant_lora_weight_for_flavour()
        loaded = load_assistant_adapter(
            transformer=self.unwrap_model(model=self.model),
            pipeline_cls=ZImagePipeline,
            lora_path=assistant_path,
            adapter_name=self.assistant_adapter_name,
            low_cpu_mem_usage=getattr(self.config, "low_cpu_mem_usage", False),
            weight_name=weight_name,
        )
        self.assistant_lora_loaded = loaded

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        text_encoder = self.text_encoders[0]
        tokenizer = self.tokenizers[0]
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Z-Image tokenizer must implement apply_chat_template.")

        processed = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            processed_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            processed.append(processed_prompt)

        max_length = getattr(self.config, "tokenizer_max_length", None) or 512
        tokenized = tokenizer(
            processed,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenized.input_ids.to(self.accelerator.device)
        attention_mask = tokenized.attention_mask.to(self.accelerator.device).bool()
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-2]
        return {
            "prompt_embeds": hidden_states,
            "attention_mask": attention_mask,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        attention_mask = text_embedding["attention_mask"]

        prompt_list: List[torch.Tensor] = []
        for embeds, mask in zip(prompt_embeds, attention_mask):
            flat_mask = mask.view(-1).bool()
            prompt_list.append(embeds[flat_mask])

        return {
            "prompt_embeds": prompt_list,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        attention_mask = text_embedding["attention_mask"]

        prompt_list: List[torch.Tensor] = []
        for embeds, mask in zip(prompt_embeds, attention_mask):
            flat_mask = mask.view(-1).bool()
            prompt_list.append(embeds[flat_mask])

        return {
            "negative_prompt_embeds": prompt_list,
        }

    def check_user_config(self):
        super().check_user_config()
        if (
            getattr(self.config, "model_type", "").lower() == "lora"
            and not getattr(self.config, "disable_assistant_lora", False)
            and self.supports_assistant_lora(self.config)
        ):
            if getattr(self.config, "assistant_lora_path", None) in (None, "", "None"):
                if self.ASSISTANT_LORA_PATH:
                    self.config.assistant_lora_path = self.ASSISTANT_LORA_PATH
                else:
                    raise ValueError(
                        "Z-Image turbo flavours require an assistant LoRA. Provide --assistant_lora_path pointing to the turbo assistant adapter."
                    )
            if getattr(self.config, "assistant_lora_weight_name", None) in (None, "", "None"):
                default_weight_name = self._assistant_lora_weight_for_flavour()
                if default_weight_name is not None:
                    self.config.assistant_lora_weight_name = default_weight_name

    def supports_crepa_self_flow(self) -> bool:
        return True

    def _prepare_crepa_self_flow_batch(self, batch: dict, state: dict) -> dict:
        transformer = self.unwrap_model(self.model) if getattr(self, "model", None) is not None else None
        if transformer is None or not hasattr(transformer, "config"):
            raise ValueError("Z-Image Self-Flow requires a loaded transformer config to determine patch size.")
        patch_sizes = getattr(transformer.config, "all_patch_size", (2,))
        patch_size = patch_sizes[0] if isinstance(patch_sizes, (tuple, list)) and patch_sizes else 2
        if not isinstance(patch_size, (int, float)):
            patch_size = 2
        return self._prepare_image_crepa_self_flow_batch(batch, state, patch_size=int(max(patch_size, 1)))

    def _prepare_model_predict_timesteps(
        self, raw_timesteps, batch_size: int, sequence_length: int | None = None
    ) -> torch.Tensor:
        if not torch.is_tensor(raw_timesteps):
            raw_timesteps = torch.tensor(raw_timesteps, device=self.accelerator.device, dtype=torch.float32)
        else:
            raw_timesteps = raw_timesteps.to(device=self.accelerator.device, dtype=torch.float32)

        if raw_timesteps.ndim == 0:
            timesteps = raw_timesteps.expand(batch_size)
        elif raw_timesteps.ndim == 1:
            if raw_timesteps.shape[0] == 1:
                timesteps = raw_timesteps.expand(batch_size)
            elif raw_timesteps.shape[0] == batch_size:
                timesteps = raw_timesteps
            else:
                raise ValueError(
                    f"Z-Image expected 1 timestep or {batch_size} per-batch timesteps, got {raw_timesteps.shape[0]}."
                )
        elif raw_timesteps.ndim == 2:
            if sequence_length is None:
                raise ValueError("Z-Image tokenwise timesteps require an explicit sequence_length.")
            if raw_timesteps.shape[1] != sequence_length:
                raise ValueError(
                    f"Z-Image expected tokenwise timesteps with sequence length {sequence_length}, got {raw_timesteps.shape[1]}."
                )
            if raw_timesteps.shape[0] == 1:
                timesteps = raw_timesteps.expand(batch_size, -1)
            elif raw_timesteps.shape[0] == batch_size:
                timesteps = raw_timesteps
            else:
                raise ValueError(
                    f"Z-Image expected tokenwise timesteps for batch size {batch_size}, got {raw_timesteps.shape[0]}."
                )
        else:
            raise ValueError(
                f"Z-Image expected a scalar, 1D batch tensor, or 2D tokenwise tensor, got shape {tuple(raw_timesteps.shape)}."
            )

        return (1000.0 - timesteps) / 1000.0

    def _latent_sequence_length(self, latent_tensor: torch.Tensor) -> int:
        if latent_tensor.dim() == 4:
            _, _, height, width = latent_tensor.shape
            frames = 1
        elif latent_tensor.dim() == 5:
            _, _, frames, height, width = latent_tensor.shape
        else:
            raise ValueError(f"Unexpected latent rank {latent_tensor.dim()} for Z-Image sequence length.")

        config = getattr(self.unwrap_model(self.model), "config", None)
        patch_sizes = getattr(config, "all_patch_size", (2,))
        frame_patch_sizes = getattr(config, "all_f_patch_size", (1,))
        patch_size = patch_sizes[0] if isinstance(patch_sizes, (tuple, list)) and patch_sizes else 2
        frame_patch_size = frame_patch_sizes[0] if isinstance(frame_patch_sizes, (tuple, list)) and frame_patch_sizes else 1
        if not isinstance(patch_size, (int, float)):
            patch_size = 2
        if not isinstance(frame_patch_size, (int, float)):
            frame_patch_size = 1
        patch_size = int(max(patch_size, 1))
        frame_patch_size = int(max(frame_patch_size, 1))
        return max((frames // frame_patch_size) * (height // patch_size) * (width // patch_size), 1)

    def _select_crepa_hidden_states(self, prepared_batch: dict, hidden_states_buffer):
        crepa = getattr(self, "crepa_regularizer", None)
        capture_layer = prepared_batch.get(
            "crepa_capture_block_index",
            getattr(crepa, "block_index", None),
        )
        if hidden_states_buffer is None or capture_layer is None:
            return None
        return hidden_states_buffer.get(f"layer_{int(capture_layer)}")

    def model_predict(self, prepared_batch, custom_timesteps: list = None):
        latents = prepared_batch["noisy_latents"]
        if latents.dim() == 4:
            latents = latents.unsqueeze(2)
        elif latents.dim() != 5:
            raise ValueError(f"Unexpected latent rank {latents.dim()} for Z-Image.")

        batch_size = latents.shape[0]
        prompt_embeds = prepared_batch["encoder_hidden_states"]
        attention_mask = prepared_batch.get("encoder_attention_mask")
        if attention_mask is None:
            raise ValueError("encoder_attention_mask is required for Z-Image training.")

        prompt_list: List[torch.Tensor] = []
        for idx in range(batch_size):
            mask = attention_mask[idx].view(-1).bool()
            prompt_list.append(prompt_embeds[idx][mask].to(device=self.accelerator.device, dtype=self.config.weight_dtype))

        latent_list = [sample.to(device=self.accelerator.device, dtype=self.config.weight_dtype) for sample in latents]

        normalized_t = self._prepare_model_predict_timesteps(
            prepared_batch["timesteps"],
            batch_size,
            sequence_length=self._latent_sequence_length(prepared_batch["noisy_latents"]),
        )

        hidden_states_buffer = self._new_hidden_state_buffer()
        call_kwargs = {}
        if "timestep_sign" in inspect.signature(self.model.__call__).parameters:
            call_kwargs["timestep_sign"] = prepared_batch.get("twinflow_time_sign")
        if hidden_states_buffer is not None:
            call_kwargs["hidden_states_buffer"] = hidden_states_buffer
        model_out_list = self.model(
            latent_list,
            normalized_t,
            prompt_list,
            **call_kwargs,
        )[0]

        noise_pred = torch.stack([out.float() for out in model_out_list], dim=0)
        if noise_pred.dim() == 5 and noise_pred.shape[2] == 1:
            noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred

        return {
            "model_prediction": noise_pred,
            "crepa_hidden_states": self._select_crepa_hidden_states(prepared_batch, hidden_states_buffer),
            "hidden_states_buffer": hidden_states_buffer,
        }


ModelRegistry.register("z_image", ZImage)
