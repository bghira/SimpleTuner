import inspect
import logging
from typing import List, Optional

import torch
from diffusers import AutoencoderKL
from transformers import AutoModelForCausalLM, AutoTokenizer

from simpletuner.helpers.acceleration import AccelerationBackend, AccelerationPreset
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
                    "ramtorch_target_modules": "layers.15,layers.16,layers.17,layers.18,layers.19,layers.20,layers.21,layers.22,layers.23,layers.24,layers.25,layers.26,layers.27,layers.28,layers.29",
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
                    "ramtorch_target_modules": "layers.8,layers.9,layers.10,layers.11,layers.12,layers.13,layers.14,layers.15,layers.16,layers.17,layers.18,layers.19,layers.20,layers.21,layers.22,layers.23,layers.24,layers.25,layers.26,layers.27,layers.28,layers.29",
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
            # DeepSpeed presets (Advanced tab)
            AccelerationPreset(
                backend=AccelerationBackend.DEEPSPEED_ZERO_1,
                level="zero1",
                name="DeepSpeed ZeRO-1",
                description="Optimizer state partitioning across GPUs.",
                tab="advanced",
                tradeoff_vram="Reduces optimizer VRAM by ~50%",
                tradeoff_speed="Minimal overhead",
                tradeoff_notes="Requires multi-GPU setup.",
                config={**_base_memory_config, "deepspeed_stage": 1},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.DEEPSPEED_ZERO_2,
                level="zero2",
                name="DeepSpeed ZeRO-2",
                description="Optimizer + gradient partitioning across GPUs.",
                tab="advanced",
                tradeoff_vram="Reduces optimizer + gradient VRAM by ~60%",
                tradeoff_speed="Slight communication overhead",
                tradeoff_notes="Requires multi-GPU setup with fast interconnect.",
                config={**_base_memory_config, "deepspeed_stage": 2},
            ),
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

    def pretrained_load_args(self, pretrained_load_args: dict) -> dict:
        args = super().pretrained_load_args(pretrained_load_args)
        # Default to full-weight loading to avoid meta tensors from diffusers low_cpu_mem_usage defaults.
        if "low_cpu_mem_usage" not in args:
            args["low_cpu_mem_usage"] = bool(getattr(self.config, "low_cpu_mem_usage", False))
        return args

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

        timesteps = prepared_batch["timesteps"]
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, device=self.accelerator.device, dtype=torch.float32)
        else:
            timesteps = timesteps.to(device=self.accelerator.device, dtype=torch.float32)
        normalized_t = (1000.0 - timesteps) / 1000.0

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

        return {"model_prediction": noise_pred, "hidden_states_buffer": hidden_states_buffer}


ModelRegistry.register("z-image", ZImage)
