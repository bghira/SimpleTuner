import logging
import os
import random
from typing import List, Optional

import torch
from diffusers import AutoencoderKL
from diffusers.models.attention_processor import Attention
from torch.nn import functional as F
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

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
    make_choice_rule,
    make_default_rule,
    make_override_rule,
)
from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.flux import build_kontext_inputs, pack_latents, prepare_latent_image_ids, unpack_latents
from simpletuner.helpers.models.flux.pipeline import FluxKontextPipeline, FluxPipeline
from simpletuner.helpers.models.flux.pipeline_controlnet import FluxControlNetPipeline, FluxControlPipeline
from simpletuner.helpers.models.flux.transformer import FluxTransformer2DModel
from simpletuner.helpers.models.tae.types import ImageTAESpec
from simpletuner.helpers.musubi_block_swap import apply_musubi_pretrained_defaults
from simpletuner.helpers.training import diffusers_overrides
from simpletuner.helpers.training.multi_process import _get_rank

logger = logging.getLogger(__name__)
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class Flux(ImageModelFoundation):
    SUPPORTS_MUON_CLIP = True
    NAME = "Flux.1"
    MODEL_DESCRIPTION = "High-quality image generation, 12B parameters"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTO_LORA_FORMAT_DETECTION = True
    # Flux Dev uses dynamic shifting (use_dynamic_shifting: true in scheduler config).
    # Schnell has use_dynamic_shifting: false, so the flag here just prevents static override.
    USES_DYNAMIC_SHIFT = True
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 16
    VALIDATION_PREVIEW_SPEC = ImageTAESpec(repo_id="madebyollin/taef1")
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0", "to_qkv"]
    SLIDER_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0", "to_qkv"]
    # Only training the Attention blocks by default.
    DEFAULT_LYCORIS_TARGET = ["Attention"]
    ASSISTANT_LORA_FLAVOURS = ["schnell"]
    ASSISTANT_LORA_PATH = "ostris/FLUX.1-schnell-training-adapter"
    ASSISTANT_LORA_WEIGHT_NAME = None

    MODEL_CLASS = FluxTransformer2DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: FluxPipeline,
        # PipelineTypes.IMG2IMG: None,
        PipelineTypes.CONTROLNET: FluxControlNetPipeline,
        PipelineTypes.CONTROL: FluxControlPipeline,
    }

    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "krea"
    HUGGINGFACE_PATHS = {
        "dev": "black-forest-labs/flux.1-dev",
        "krea": "black-forest-labs/flux.1-krea-dev",
        "schnell": "black-forest-labs/flux.1-schnell",
        "kontext": "black-forest-labs/flux.1-kontext-dev",
        "fluxbooru": "terminusresearch/fluxbooru-v0.3",
        "libreflux": "jimmycarter/LibreFlux-SimpleTuner",
    }
    MODEL_LICENSE = "other"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "CLIP-L/14",
            "tokenizer": CLIPTokenizer,
            "tokenizer_subfolder": "tokenizer",
            "model": CLIPTextModel,
        },
        "text_encoder_2": {
            "name": "T5 XXL v1.1",
            "tokenizer": T5TokenizerFast,
            "subfolder": "text_encoder_2",
            "tokenizer_subfolder": "tokenizer_2",
            "model": T5EncoderModel,
        },
    }

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        # Flux has 19 double blocks + 38 single blocks = 57 total
        # Leave at least 1 block on GPU
        return 56

    @classmethod
    def get_acceleration_presets(cls) -> list[AccelerationPreset]:
        # Common settings for memory optimization presets
        _base_memory_config = {
            "base_model_precision": "no_change",
            "gradient_checkpointing": True,
        }

        return [
            # Basic tab - RamTorch presets
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="basic",
                name="RamTorch - Basic",
                description="Streams double block weights from CPU RAM.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~35%",
                tradeoff_speed="Increases training time by ~25%",
                tradeoff_notes="Requires 64GB+ system RAM. CUDA/ROCm only.",
                requires_cuda=True,
                requires_min_system_ram_gb=64,
                config={
                    **_base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "transformer_blocks.*",
                },
            ),
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="aggressive",
                name="RamTorch - Aggressive",
                description="Streams all transformer block weights from CPU RAM.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~60%",
                tradeoff_speed="Increases training time by ~50%",
                tradeoff_notes="Requires 64GB+ system RAM. CUDA/ROCm only.",
                requires_cuda=True,
                requires_min_system_ram_gb=64,
                config={
                    **_base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "*",
                },
            ),
            # Basic tab - Block swap presets
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="light",
                name="Block Swap - Light",
                description="Swaps 14 of 57 blocks (~25%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~20%",
                tradeoff_speed="Increases training time by ~15%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={**_base_memory_config, "musubi_blocks_to_swap": 14},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="balanced",
                name="Block Swap - Balanced",
                description="Swaps 28 of 57 blocks (~50%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~45%",
                tradeoff_speed="Increases training time by ~30%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={**_base_memory_config, "musubi_blocks_to_swap": 28},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="aggressive",
                name="Block Swap - Aggressive",
                description="Swaps 42 of 57 blocks (~75%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~65%",
                tradeoff_speed="Increases training time by ~50%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={**_base_memory_config, "musubi_blocks_to_swap": 42},
            ),
            # DeepSpeed presets (multi-GPU only)
            *get_deepspeed_presets(_base_memory_config),
            # Advanced tab - Group Offload
            AccelerationPreset(
                backend=AccelerationBackend.GROUP_OFFLOAD,
                level="block",
                name="Group Offload - Block Level",
                description="Offloads module groups to CPU using diffusers hooks.",
                tab="advanced",
                tradeoff_vram="Substantial VRAM savings",
                tradeoff_speed="Significant overhead from CPU-GPU transfers",
                tradeoff_notes="Known stability issues. Mutually exclusive with RamTorch.",
                requires_min_system_ram_gb=64,
                config={
                    **_base_memory_config,
                    "enable_group_offload": True,
                    "group_offload_type": "block_level",
                },
            ),
            # SDNQ presets (works on AMD, Apple, NVIDIA)
            *get_sdnq_presets(_base_memory_config),
            # TorchAO presets (NVIDIA only)
            *get_torchao_presets(_base_memory_config),
            # Quanto presets (works on AMD, Apple, NVIDIA)
            *get_quanto_presets(_base_memory_config),
            # BitsAndBytes presets (NVIDIA only)
            *get_bitsandbytes_presets(_base_memory_config),
        ]

    def control_init(self):
        """
        Initialize Flux Control parameters.
        """
        if self.config.control and self.config.pretrained_transformer_model_name_or_path is None:
            with torch.no_grad():
                initial_input_channels = self.get_trained_component().config.in_channels
                # new linear layer for x_embedder
                new_linear = torch.nn.Linear(
                    self.get_trained_component().x_embedder.in_features * 2,
                    self.get_trained_component().x_embedder.out_features,
                    bias=self.get_trained_component().x_embedder.bias is not None,
                    dtype=self.get_trained_component().dtype,
                    device=self.get_trained_component().device,
                )
                new_linear.weight.zero_()
                new_linear.weight[:, :initial_input_channels].copy_(self.get_trained_component().x_embedder.weight)
                if self.get_trained_component().x_embedder.bias is not None:
                    new_linear.bias.copy_(self.get_trained_component().x_embedder.bias)
                self.get_trained_component().x_embedder = new_linear
                # new projection layer for pos_embed
                new_proj = torch.nn.Conv2d(
                    in_channels=self.get_trained_component().pos_embed.proj.in_channels * 2,
                    out_channels=self.get_trained_component().pos_embed.proj.out_channels,
                    kernel_size=self.get_trained_component().pos_embed.proj.kernel_size,
                    stride=self.get_trained_component().pos_embed.proj.stride,
                    bias=self.get_trained_component().pos_embed.proj.bias is not None,
                )
                new_proj.weight.zero_()
                new_proj.weight[:, :initial_input_channels].copy_(self.get_trained_component().pos_embed.proj.weight)
                if self.get_trained_component().pos_embed.proj.bias is not None:
                    new_proj.bias.copy_(self.get_trained_component().pos_embed.proj.bias)
                self.get_trained_component().pos_embed.proj = new_proj
                self.get_trained_component().register_to_config(
                    in_channels=initial_input_channels * 2,
                    out_channels=initial_input_channels,
                )

            assert torch.all(self.get_trained_component().x_embedder.weight[:, initial_input_channels:].data == 0)
            assert torch.all(self.get_trained_component().pos_embed.proj.weight[:, initial_input_channels:].data == 0)

    def controlnet_init(self):
        logger.info("Creating the controlnet..")
        from diffusers import FluxControlNetModel

        if self.config.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            self.controlnet = FluxControlNetModel.from_pretrained(self.config.controlnet_model_name_or_path)
        else:
            logger.info("Initializing controlnet weights from base model")
            self.controlnet = FluxControlNetModel.from_transformer(self.unwrap_model(self.model))
        self.controlnet.to(self.accelerator.device, self.config.weight_dtype)

    def tread_init(self):
        """
        Initialize the TREAD model training method.
        """
        from simpletuner.helpers.training.tread import TREADRouter

        if (
            getattr(self.config, "tread_config", None) is None
            or getattr(self.config, "tread_config", None) is {}
            or getattr(self.config, "tread_config", {}).get("routes", None) is None
        ):
            logger.error("TREAD training requires you to configure the routes in the TREAD config")
            import sys

            sys.exit(1)

        self.unwrap_model(model=self.model).set_router(
            TREADRouter(
                seed=getattr(self.config, "seed", None) or 42,
                device=self.accelerator.device,
            ),
            self.config.tread_config["routes"],
        )

        logger.info("TREAD training is enabled")

    def post_model_load_setup(self):
        super().post_model_load_setup()
        self._maybe_load_assistant_lora()

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

        loaded = load_assistant_adapter(
            transformer=self.unwrap_model(model=self.model),
            pipeline_cls=FluxPipeline,
            lora_path=assistant_path,
            adapter_name=self.assistant_adapter_name,
            low_cpu_mem_usage=getattr(self.config, "low_cpu_mem_usage", False),
            weight_name=getattr(self.config, "assistant_lora_weight_name", None),
        )
        self.assistant_lora_loaded = loaded

    def fuse_qkv_projections(self):
        if not self.config.fuse_qkv_projections or self._qkv_projections_fused:
            return

        try:
            from simpletuner.helpers.models.flux.attention import FluxFusedFlashAttnProcessor3

            attn_processor = FluxFusedFlashAttnProcessor3()
        except:
            from simpletuner.helpers.models.flux.attention import FluxFusedSDPAProcessor

            attn_processor = FluxFusedSDPAProcessor()

        if self.model is not None:
            logger.debug("Fusing QKV projections in the model..")
            for module in self.model.modules():
                if isinstance(module, Attention):
                    module.fuse_projections(fuse=True)
        else:
            logger.warning("Model does not support QKV projection fusing. Skipping.")

        self.unwrap_model(model=self.model).set_attn_processor(attn_processor)
        if self.controlnet is not None:
            logger.debug("Fusing QKV projections in the ControlNet..")
            for module in self.controlnet.modules():
                if isinstance(module, Attention):
                    module.fuse_projections(fuse=True)
            logger.debug("Setting ControlNet attention processor to FluxFusedFlashAttnProcessor3")
            self.unwrap_model(model=self.controlnet).set_attn_processor(attn_processor)
        elif self.config.controlnet:
            logger.warning("ControlNet does not support QKV projection fusing. Skipping.")
        self._qkv_projections_fused = True

    def unfuse_qkv_projections(self):
        """
        Unfuse QKV projections in the model and ControlNet if they were fused.
        """
        if not self.config.fuse_qkv_projections or not self._qkv_projections_fused:
            return
        self._qkv_projections_fused = False

        if self.model is not None:
            logger.debug("Temporarily unfusing QKV projections in the model..")
            for module in self.model.modules():
                if isinstance(module, Attention):
                    module.fuse_projections(fuse=False)
            if self.controlnet is not None:
                logger.debug("Tempoarily unfusing QKV projections in the ControlNet..")
                for module in self.controlnet.modules():
                    if isinstance(module, Attention):
                        module.fuse_projections(fuse=False)

    def requires_conditioning_latents(self) -> bool:
        # Flux ControlNet requires latent inputs instead of pixels.
        if self.config.controlnet or self.config.control:
            return True
        return False

    def requires_conditioning_validation_inputs(self) -> bool:
        # Whether this model / flavour requires conditioning inputs during validation.
        if self.config.controlnet or self.config.control:
            return True
        return False

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        """
        Models can optionally format the stored text embedding, eg. in a dict, or
        filter certain outputs from appearing in the file cache.

        self.config:
            text_embedding (torch.Tensor): The embed to adjust.

        Returns:
            torch.Tensor: The adjusted embed. By default, this method does nothing.
        """
        prompt_embeds, pooled_prompt_embeds, time_ids, masks = text_embedding

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds.squeeze(0),
            "time_ids": time_ids,
            "attention_masks": masks,
        }

    def update_pipeline_call_kwargs(self, kwargs: dict) -> dict:
        """
        Let the base class copy the dict unchanged, then patch in
        Kontext-specific keys if we're running a Kontext checkpoint.
        """
        if self.config.model_flavour == "kontext":
            # 1) rename the placeholder key coming from Validation.validate_prompt
            if "image" in kwargs and "conditioning_image" not in kwargs:
                kwargs["conditioning_image"] = kwargs.pop("image")

            # 2) if the caller didn’t specify a range, run the ref image
            #    for the *entire* denoise (default behaviour in training)
            if "cond_start_step" not in kwargs:
                kwargs["cond_start_step"] = 0
            if "cond_end_step" not in kwargs:
                # `num_inference_steps` is already inside kwargs
                kwargs["cond_end_step"] = kwargs.get("num_inference_steps", 28)

        return kwargs

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        # Only unsqueeze if it's missing the batch dimension
        prompt_embeds = text_embedding["prompt_embeds"]
        pooled_prompt_embeds = text_embedding["pooled_prompt_embeds"]
        attention_mask = text_embedding.get("attention_masks", None)

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:  # Shape: [seq, dim]
            prompt_embeds = prompt_embeds.unsqueeze(0)  # Shape: [1, seq, dim]
        if pooled_prompt_embeds.dim() == 1:  # Shape: [dim]
            pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0)  # Shape: [1, dim]
        if attention_mask is not None and attention_mask.dim() == 1:  # Shape: [seq]
            attention_mask = attention_mask.unsqueeze(0)  # Shape: [1, seq]

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "prompt_mask": (attention_mask if self.config.flux_attention_masked_training else None),
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        if self.config.validation_guidance_real is None or self.config.validation_guidance_real <= 1.0:
            # CFG is disabled, no negative prompts.
            return {}
        prompt_embeds = text_embedding["prompt_embeds"]
        pooled_prompt_embeds = text_embedding["pooled_prompt_embeds"]
        attention_mask = text_embedding.get("attention_masks", None)

        if prompt_embeds.dim() == 2:  # Shape: [seq, dim]
            prompt_embeds = prompt_embeds.unsqueeze(0)  # Shape: [1, seq, dim]
        if pooled_prompt_embeds.dim() == 1:  # Shape: [dim]
            pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0)  # Shape: [1, dim]
        if attention_mask is not None and attention_mask.dim() == 1:  # Shape: [seq]
            attention_mask = attention_mask.unsqueeze(0)  # Shape: [1, seq]
        return {
            "negative_prompt_embeds": prompt_embeds,
            "negative_pooled_prompt_embeds": pooled_prompt_embeds,
            "negative_mask": (attention_mask if self.config.flux_attention_masked_training else None),
            "guidance_scale_real": float(self.config.validation_guidance_real),
            "no_cfg_until_timestep": int(self.config.validation_no_cfg_until_timestep),
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode a prompt.

        Args:
            prompts: The list of prompts to encode.

        Returns:
            Text encoder output (raw)
        """
        pipeline = self.pipelines.get(PipelineTypes.TEXT2IMG)
        if pipeline is None:
            pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        prompt_embeds, pooled_prompt_embeds, time_ids, masks = pipeline.encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            device=self.accelerator.device,
            max_sequence_length=int(self.config.tokenizer_max_length),
        )
        if self.config.t5_padding == "zero":
            # we can zero the padding tokens if we're just going to mask them later anyway.
            prompt_embeds = prompt_embeds * masks.to(device=prompt_embeds.device).unsqueeze(-1).expand(prompt_embeds.shape)

        return prompt_embeds, pooled_prompt_embeds, time_ids, masks

    def prepare_batch_conditions(self, batch: dict, state: dict):
        cond = batch.get("conditioning_latents")
        if cond is None:
            logger.debug(f"No conditioning latents found :(")
            return super().prepare_batch_conditions(batch=batch, state=state)  # nothing to do
        # Check sampling mode
        sampling_mode = state.get("args", {}).get("conditioning_multidataset_sampling", "random")

        if sampling_mode == "random" and isinstance(cond, list) and len(cond) >= 1:
            # Random mode should have selected just one
            cond = cond[0]

        # This is only a thing we do if we're training kontext, but conditioning
        # images are also used for masked loss training, so if we hit an error
        # here for index, ignore it.
        if self.config.model_flavour == "kontext":
            if isinstance(cond, list):
                logger.debug(f"Inputs to kontext builder shapes: {[d.shape for d in cond]} {cond[0].dtype}")
            else:
                logger.debug(f"Inputs to kontext builder shapes: {cond.shape} {cond.dtype}")

            # Build Kontext inputs
            packed_cond, cond_ids = build_kontext_inputs(
                cond if isinstance(cond, list) else [cond],
                dtype=self.config.weight_dtype,
                device=self.accelerator.device,
                latent_channels=self.LATENT_CHANNEL_COUNT,
            )
            logger.debug(f"Now we have kontext shapes: {packed_cond.shape} {packed_cond.dtype}")

            batch["conditioning_packed_latents"] = packed_cond
            batch["conditioning_ids"] = cond_ids

        return super().prepare_batch_conditions(batch=batch, state=state)  # fixes ControlNet latents in super class.

    def model_predict(self, prepared_batch):
        # handle guidance
        hidden_states_buffer = self._new_hidden_state_buffer()
        packed_noisy_latents = pack_latents(
            prepared_batch["noisy_latents"],
            batch_size=prepared_batch["latents"].shape[0],
            num_channels_latents=prepared_batch["latents"].shape[1],
            height=prepared_batch["latents"].shape[2],
            width=prepared_batch["latents"].shape[3],
        ).to(
            dtype=self.config.base_weight_dtype,
            device=self.accelerator.device,
        )
        if self.config.flux_guidance_mode == "constant":
            guidance_scales = [float(self.config.flux_guidance_value)] * prepared_batch["latents"].shape[0]

        elif self.config.flux_guidance_mode == "random-range":
            # Generate a list of random values within the specified range for each latent
            guidance_scales = [
                random.uniform(
                    self.config.flux_guidance_min,
                    self.config.flux_guidance_max,
                )
                for _ in range(prepared_batch["latents"].shape[0])
            ]

        # Now `guidance` will have different values for each latent in `latents`.
        transformer_config = None
        if hasattr(self.get_trained_component(), "module"):
            transformer_config = self.get_trained_component().module.config
        elif hasattr(self.get_trained_component(), "config"):
            transformer_config = self.get_trained_component().config
        if transformer_config is not None and getattr(transformer_config, "guidance_embeds", False):
            guidance = torch.tensor(guidance_scales, device=self.accelerator.device)
        else:
            guidance = None
        img_ids = prepare_latent_image_ids(
            prepared_batch["latents"].shape[0],
            prepared_batch["latents"].shape[2],
            prepared_batch["latents"].shape[3],
            self.accelerator.device,
            self.config.weight_dtype,
        )
        prepared_batch["timesteps"] = (
            torch.tensor(prepared_batch["timesteps"])
            .expand(prepared_batch["noisy_latents"].shape[0])
            .to(device=self.accelerator.device, dtype=torch.float32)
            / self.noise_schedule.config.num_train_timesteps
        )

        text_ids = torch.zeros(
            prepared_batch["prompt_embeds"].shape[1],
            3,
        ).to(
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        logger.debug(
            "DTypes:"
            f"\n-> Text IDs shape: {text_ids.shape if hasattr(text_ids, 'shape') else None}, dtype: {text_ids.dtype if hasattr(text_ids, 'dtype') else None}"
            f"\n-> Image IDs shape: {img_ids.shape if hasattr(img_ids, 'shape') else None}, dtype: {img_ids.dtype if hasattr(img_ids, 'dtype') else None}"
            f"\n-> Timesteps shape: {prepared_batch['timesteps'].shape if hasattr(prepared_batch['timesteps'], 'shape') else None}, dtype: {prepared_batch['timesteps'].dtype if hasattr(prepared_batch['timesteps'], 'dtype') else None}"
            f"\n-> Guidance: {guidance}"
            f"\n-> Packed Noisy Latents shape: {packed_noisy_latents.shape if hasattr(packed_noisy_latents, 'shape') else None}, dtype: {packed_noisy_latents.dtype if hasattr(packed_noisy_latents, 'dtype') else None}"
            f"\n-> Conditioning Packed Latents shape: {prepared_batch['conditioning_packed_latents'].shape if 'conditioning_packed_latents' in prepared_batch else 'N/A'}, dtype: {prepared_batch['conditioning_packed_latents'].dtype if 'conditioning_packed_latents' in prepared_batch else 'N/A'}"
        )

        if img_ids.dim() == 2:  # (S, 3)  -> (1, S, 3) -> (B, S, 3)
            img_ids = img_ids.unsqueeze(0).expand(prepared_batch["latents"].shape[0], -1, -1)

        # pull optional kontext inputs
        cond_seq = prepared_batch.get("conditioning_packed_latents")
        cond_ids = prepared_batch.get("conditioning_ids")

        use_cond = cond_seq is not None
        logger.debug(f"Using conditioning: {use_cond}")
        lat_in = torch.cat([packed_noisy_latents, cond_seq], dim=1) if use_cond else packed_noisy_latents
        id_in = torch.cat([img_ids, cond_ids], dim=1) if use_cond else img_ids

        flux_transformer_kwargs = {
            "hidden_states": lat_in,
            # YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
            "timestep": prepared_batch["timesteps"],
            "guidance": guidance,
            "pooled_projections": prepared_batch["add_text_embeds"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            "encoder_hidden_states": prepared_batch["prompt_embeds"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            "txt_ids": text_ids.to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            "img_ids": id_in,
            "joint_attention_kwargs": None,
            "return_dict": False,
        }
        if hidden_states_buffer is not None:
            flux_transformer_kwargs["hidden_states_buffer"] = hidden_states_buffer
        if self.config.flux_attention_masked_training:
            attention_mask = prepared_batch["encoder_attention_mask"]
            if attention_mask is None:
                raise ValueError(
                    "No attention mask was discovered when attempting validation - this means you need to recreate your text embed cache."
                )
            # Squeeze out the extra dimension if present
            if attention_mask.dim() == 3 and attention_mask.size(1) == 1:
                attention_mask = attention_mask.squeeze(1)  # [B, 1, S] -> [B, S]
            flux_transformer_kwargs["attention_mask"] = attention_mask

        # For masking and segmentation training when combined with TREAD, avoid
        # dropping any tokens that are in the mask.
        if (
            getattr(self.config, "tread_config", None) is not None
            and self.config.tread_config is not None
            and "conditioning_pixel_values" in prepared_batch
            and prepared_batch["conditioning_pixel_values"] is not None
            and prepared_batch.get("loss_mask_type") in ("mask", "segmentation")
        ):
            with torch.no_grad():
                h_tokens = prepared_batch["latents"].shape[2] // 2  # H_latent // 2
                w_tokens = prepared_batch["latents"].shape[3] // 2  # W_latent // 2
                mask_img = prepared_batch["conditioning_pixel_values"]  # (B,3,Hc,Wc)
                # fuse RGB → single channel, map to [0,1]
                mask_img = (mask_img.sum(1, keepdim=True) / 3 + 1) / 2
                # down‑sample so each latent / image token corresponds to 1 pixel
                mask_lat = F.interpolate(mask_img, size=(h_tokens, w_tokens), mode="area")  # (B,1,32,32)
                force_keep = mask_lat.flatten(2).squeeze(1) > 0.5  # (B, S_img)
                flux_transformer_kwargs["force_keep_mask"] = force_keep

        model_pred = self.model(**flux_transformer_kwargs)[0]
        # Drop the reference-image tokens before unpacking
        if use_cond and self.config.model_flavour == "kontext":
            scene_seq_len = packed_noisy_latents.shape[1]  # tokens that belong to the main image
            model_pred = model_pred[:, :scene_seq_len, :]  # (B, S_scene, C*4)

        # Extract CREPA hidden states from buffer if enabled
        crepa_hidden = None
        crepa = getattr(self, "crepa_regularizer", None)
        if crepa and crepa.enabled and hidden_states_buffer is not None:
            layer_key = f"layer_{crepa.block_index}"
            crepa_hidden = hidden_states_buffer.get(layer_key)

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

    def controlnet_predict(self, prepared_batch: dict) -> dict:
        """
        Perform a forward pass with ControlNet for Flux model.

        Args:
            prepared_batch: Dictionary containing the batch data including conditioning_latents

        Returns:
            Dictionary containing the model prediction
        """
        # ControlNet conditioning - Flux uses latents instead of pixel values
        controlnet_cond = prepared_batch["conditioning_latents"].to(
            device=self.accelerator.device, dtype=self.config.weight_dtype
        )

        # Pack the conditioning latents (same as noisy latents)
        packed_controlnet_cond = pack_latents(
            controlnet_cond,
            batch_size=controlnet_cond.shape[0],
            num_channels_latents=controlnet_cond.shape[1],
            height=controlnet_cond.shape[2],
            width=controlnet_cond.shape[3],
        ).to(
            dtype=self.config.base_weight_dtype,
            device=self.accelerator.device,
        )

        # Pack noisy latents
        packed_noisy_latents = pack_latents(
            prepared_batch["noisy_latents"],
            batch_size=prepared_batch["latents"].shape[0],
            num_channels_latents=prepared_batch["latents"].shape[1],
            height=prepared_batch["latents"].shape[2],
            width=prepared_batch["latents"].shape[3],
        ).to(
            dtype=self.config.base_weight_dtype,
            device=self.accelerator.device,
        )

        # Handle guidance
        if self.config.flux_guidance_mode == "constant":
            guidance_scales = [float(self.config.flux_guidance_value)] * prepared_batch["latents"].shape[0]
        elif self.config.flux_guidance_mode == "random-range":
            guidance_scales = [
                random.uniform(
                    self.config.flux_guidance_min,
                    self.config.flux_guidance_max,
                )
                for _ in range(prepared_batch["latents"].shape[0])
            ]

        # Check if guidance embeds are enabled
        transformer_config = None
        if hasattr(self.get_trained_component(base_model=True), "module"):
            transformer_config = self.get_trained_component(base_model=True).module.config
        elif hasattr(self.get_trained_component(base_model=True), "config"):
            transformer_config = self.get_trained_component(base_model=True).config

        if transformer_config is not None and getattr(transformer_config, "guidance_embeds", False):
            guidance = torch.tensor(guidance_scales, device=self.accelerator.device)
        else:
            guidance = None

        # Prepare image IDs
        img_ids = prepare_latent_image_ids(
            prepared_batch["latents"].shape[0],
            prepared_batch["latents"].shape[2],
            prepared_batch["latents"].shape[3],
            self.accelerator.device,
            self.config.weight_dtype,
        )

        # Prepare timesteps
        prepared_batch["timesteps"] = (
            torch.tensor(prepared_batch["timesteps"])
            .expand(prepared_batch["noisy_latents"].shape[0])
            .to(device=self.accelerator.device, dtype=torch.float32)
            / self.noise_schedule.config.num_train_timesteps
        )

        # Prepare text IDs
        text_ids = torch.zeros(
            prepared_batch["prompt_embeds"].shape[1],
            3,
        ).to(
            device=self.accelerator.device,
            dtype=torch.float32,
        )

        # ControlNet forward pass
        controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
            hidden_states=packed_noisy_latents,
            controlnet_cond=packed_controlnet_cond,
            controlnet_mode=None,  # Set this if using ControlNet-Union
            conditioning_scale=1.0,  # You might want to make this configurable
            encoder_hidden_states=prepared_batch["prompt_embeds"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            pooled_projections=prepared_batch["add_text_embeds"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            timestep=prepared_batch["timesteps"],
            img_ids=img_ids,
            txt_ids=text_ids,
            guidance=guidance,
            joint_attention_kwargs=None,
            return_dict=False,
        )

        # Prepare kwargs for the main transformer
        flux_transformer_kwargs = {
            "hidden_states": packed_noisy_latents,
            "timestep": prepared_batch["timesteps"],
            "guidance": guidance,
            "pooled_projections": prepared_batch["add_text_embeds"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            "encoder_hidden_states": prepared_batch["prompt_embeds"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            "txt_ids": text_ids.to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            "img_ids": img_ids,
            "joint_attention_kwargs": None,
            "return_dict": False,
        }
        if prepared_batch.get("twinflow_time_sign") is not None:
            flux_transformer_kwargs["timestep_sign"] = prepared_batch["twinflow_time_sign"]

        # Add ControlNet outputs to kwargs
        if controlnet_block_samples is not None:
            flux_transformer_kwargs["controlnet_block_samples"] = [
                sample.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
                for sample in controlnet_block_samples
            ]

        if controlnet_single_block_samples is not None:
            flux_transformer_kwargs["controlnet_single_block_samples"] = [
                sample.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
                for sample in controlnet_single_block_samples
            ]

        # Add attention mask if using masked training
        if self.config.flux_attention_masked_training:
            flux_transformer_kwargs["attention_mask"] = prepared_batch["encoder_attention_mask"]
            if flux_transformer_kwargs["attention_mask"] is None:
                raise ValueError(
                    "No attention mask was discovered when attempting validation - "
                    "this means you need to recreate your text embed cache."
                )

        # Forward pass through the transformer with ControlNet residuals
        model_pred = self.get_trained_component(base_model=True)(**flux_transformer_kwargs)[0]

        # Unpack the latents back to original shape
        return {
            "model_prediction": unpack_latents(
                model_pred,
                height=prepared_batch["latents"].shape[2] * 8,
                width=prepared_batch["latents"].shape[3] * 8,
                vae_scale_factor=16,
            )
        }

    def check_user_config(self):
        """
        Checks self.config values against important issues. Optionally implemented in child class.
        """
        if self.config.unet_attention_slice:
            if torch.backends.mps.is_available():
                logger.warning(
                    f"Using attention slicing when training {self.NAME} on MPS can result in NaN errors on the first backward pass. If you run into issues, disable this option and reduce your batch size instead to reduce memory consumption."
                )
            if self.get_trained_component() is not None:
                self.get_trained_component().set_attention_slice("auto")

        # if self.config.base_model_precision == "fp8-quanto":
        #     raise ValueError(
        #         f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
        #     )
        if self.config.aspect_bucket_alignment != 64:
            logger.warning(
                f"{self.NAME} requires an alignment value of 64px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 64

        if self.config.prediction_type is not None:
            logger.warning(f"{self.NAME} does not support prediction type {self.config.prediction_type}.")

        if self.config.tokenizer_max_length is not None:
            logger.warning(f"-!- {self.NAME} supports a max length of 512 tokens, --tokenizer_max_length is ignored -!-")
        self.config.tokenizer_max_length = 512
        if self.config.model_flavour == "schnell":
            if not getattr(self.config, "disable_assistant_lora", False) and getattr(
                self.config, "assistant_lora_path", None
            ) in (None, "", "None"):
                self.config.assistant_lora_path = self.ASSISTANT_LORA_PATH
                if getattr(self.config, "assistant_lora_weight_name", None) in (None, "", "None"):
                    self.config.assistant_lora_weight_name = getattr(self, "ASSISTANT_LORA_WEIGHT_NAME", None)
            if not self.config.flux_fast_schedule and not self.config.i_know_what_i_am_doing:
                logger.error("Schnell requires --flux_fast_schedule (or --i_know_what_i_am_doing).")
                import sys

                sys.exit(1)
            self.config.tokenizer_max_length = 256

        if self.config.model_flavour == "dev":
            if self.config.validation_num_inference_steps > 28:
                logger.warning(
                    f"{self.NAME} {self.config.model_flavour} expects around 28 or fewer inference steps. Consider limiting --validation_num_inference_steps to 28."
                )
            if self.config.validation_num_inference_steps < 15:
                logger.warning(
                    f"{self.NAME} {self.config.model_flavour} expects around 15 or more inference steps. Consider increasing --validation_num_inference_steps to 15."
                )
        if self.config.model_flavour == "schnell" and self.config.validation_num_inference_steps > 4:
            logger.warning(
                "Flux Schnell requires fewer inference steps. Consider reducing --validation_num_inference_steps to 4."
            )
        if self.config.model_flavour == "kontext" and not isinstance(
            self.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG], FluxKontextPipeline
        ):
            self.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG] = FluxKontextPipeline

        if self.config.model_flavour == "libreflux":
            if self.config.validation_num_inference_steps < 28:
                logger.warning("LibreFlux requires at least 28 validation steps. Increasing value to 28.")
                self.config.validation_num_inference_steps = 28
            if self.config.validation_guidance_real <= 1.0:
                logger.warning("LibreFlux requires CFG at validation time. Enabling it.")
                self.config.validation_guidance_real = 6.0
            if not self.config.flux_attention_masked_training:
                logger.warning("LibreFlux requires attention masking. Enabling it.")
                self.config.flux_attention_masked_training = True
            if self.config.fuse_qkv_projections:
                logger.warning("LibreFlux does not support fused QKV projections. Disabling it.")
                self.config.fuse_qkv_projections = False
        if self.config.model_flavour == "fluxbooru":
            # FluxBooru requires some special settings, we'll just override them here.
            if self.config.validation_num_inference_steps < 28:
                logger.warning("FluxBooru requires at least 28 validation steps. Increasing value to 28.")
                self.config.validation_num_inference_steps = 28
            if self.config.validation_guidance_real <= 1.0:
                logger.warning("FluxBooru requires CFG at validation time. Enabling it.")
                self.config.validation_guidance_real = 6.0
            if self.config.flux_guidance_value != 3.5:
                logger.warning("FluxBooru requires a static guidance value of 3.5. Overriding --flux_guidance_value.")
                self.config.flux_guidance_value = 3.5
            if self.config.flux_attention_masked_training:
                logger.warning("FluxBooru does not support attention masking. Disabling it.")
                self.config.flux_attention_masked_training = False

    def conditioning_validation_dataset_type(self) -> bool:
        # Most conditioning inputs (ControlNet) etc require "conditioning" dataset, but Kontext requires "images".
        if self.config.model_flavour == "kontext":
            # Kontext wants the edited
            return "image"
        return "conditioning"

    def requires_conditioning_dataset(self) -> bool:
        if self.config.model_flavour == "kontext" or self.config.controlnet:
            # Any flavour of “Kontext” always expects an extra image stream
            return True
        return False

    def requires_conditioning_validation_inputs(self) -> bool:
        if self.config.model_flavour == "kontext" or self.config.controlnet:
            # Any flavour of “Kontext” always expects an extra image stream
            return True
        return False

    def requires_validation_edit_captions(self) -> bool:
        if self.config.model_flavour == "kontext" or self.config.controlnet:
            # Kontext models require edit captions to be present.
            return True
        return False

    def requires_conditioning_latents(self) -> bool:
        if self.config.model_flavour == "kontext" or self.config.controlnet:
            # Any flavour of “Kontext” needs latent inputs for its conditioning data.
            return True
        return super().requires_conditioning_latents()

    def get_lora_target_layers(self):
        # Some models, eg. Flux should override this with more complex config-driven logic.
        manual_targets = self._get_peft_lora_target_modules()
        if manual_targets:
            return manual_targets
        if self.config.lora_type.lower() == "standard" and getattr(self.config, "slider_lora_target", False):
            return getattr(self, "SLIDER_LORA_TARGET", None) or self.DEFAULT_SLIDER_LORA_TARGET
        if self.config.model_type == "lora" and (self.config.controlnet or self.config.control):
            if "control" not in self.config.flux_lora_target.lower():
                logger.warning(
                    "ControlNet or Control is enabled, but the LoRA target does not include 'control'. Overriding to controlnet."
                )
            self.config.flux_lora_target = "controlnet"
        if self.config.lora_type.lower() == "standard":
            if self.config.flux_lora_target == "all":
                # target_modules = mmdit layers here
                return [
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_qkv",
                    "add_qkv_proj",
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "to_out.0",
                    "to_add_out",
                ]
            elif self.config.flux_lora_target == "context":
                # i think these are the text input layers.
                return [
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "add_qkv_proj",
                    "to_add_out",
                ]
            elif self.config.flux_lora_target == "context+ffs":
                # i think these are the text input layers.
                return [
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "add_qkv_proj",
                    "to_add_out",
                    "ff_context.net.0.proj",
                    "ff_context.net.2",
                ]
            elif self.config.flux_lora_target == "all+ffs":
                return [
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_qkv",
                    "add_qkv_proj",
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "to_out.0",
                    "to_add_out",
                    "ff.net.0.proj",
                    "ff.net.2",
                    "ff_context.net.0.proj",
                    "ff_context.net.2",
                    "proj_mlp",
                    "proj_out",
                ]
            elif self.config.flux_lora_target == "controlnet":
                return [
                    "controlnet_x_embedder",
                    "controlnet_blocks.0",
                    "controlnet_blocks.1",
                    "controlnet_blocks.2",
                    "controlnet_blocks.3",
                    "controlnet_single_blocks.0",
                    "controlnet_single_blocks.1",
                    "controlnet_single_blocks.2",
                    "controlnet_single_blocks.3",
                    "controlnet_single_blocks.4",
                    "controlnet_single_blocks.5",
                    "controlnet_single_blocks.6",
                    "controlnet_single_blocks.7",
                    "controlnet_single_blocks.8",
                    "controlnet_single_blocks.9",
                ]
            elif self.config.flux_lora_target == "all+ffs+embedder":
                return [
                    "x_embedder",
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_qkv",
                    "add_qkv_proj",
                    "to_out.0",
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "to_add_out",
                    "ff.net.0.proj",
                    "ff.net.2",
                    "ff_context.net.0.proj",
                    "ff_context.net.2",
                    "proj_mlp",
                    "proj_out",
                ]
            elif self.config.flux_lora_target == "ai-toolkit":
                # from ostris' ai-toolkit, possibly required to continue finetuning one.
                return [
                    "to_q",
                    "to_k",
                    "to_qkv",
                    "add_qkv_proj",
                    "to_v",
                    "add_q_proj",
                    "add_k_proj",
                    "add_v_proj",
                    "to_out.0",
                    "to_add_out",
                    "ff.net.0.proj",
                    "ff.net.2",
                    "ff_context.net.0.proj",
                    "ff_context.net.2",
                    "norm.linear",
                    "norm1.linear",
                    "norm1_context.linear",
                    "proj_mlp",
                    "proj_out",
                ]
            elif self.config.flux_lora_target == "tiny":
                # From TheLastBen
                # https://www.reddit.com/r/StableDiffusion/comments/1f523bd/good_flux_loras_can_be_less_than_45mb_128_dim/
                return [
                    "single_transformer_blocks.7.proj_out",
                    "single_transformer_blocks.20.proj_out",
                ]
            elif self.config.flux_lora_target == "nano":
                # From TheLastBen
                # https://www.reddit.com/r/StableDiffusion/comments/1f523bd/good_flux_loras_can_be_less_than_45mb_128_dim/
                return [
                    "single_transformer_blocks.7.proj_out",
                ]

            return self.DEFAULT_LORA_TARGET
        elif self.config.lora_type.lower() == "lycoris":
            return self.DEFAULT_LYCORIS_TARGET
        else:
            raise NotImplementedError(f"Unknown LoRA target type {self.config.lora_type}.")

    def custom_model_card_schedule_info(self):
        output_args = []
        if self.config.flux_fast_schedule:
            output_args.append("flux_fast_schedule")
        if self.config.flow_schedule_auto_shift:
            output_args.append("flow_schedule_auto_shift")
        if self.config.flow_schedule_shift is not None:
            output_args.append(f"shift={self.config.flow_schedule_shift}")
        output_args.append(f"flux_guidance_mode={self.config.flux_guidance_mode}")
        if self.config.flux_guidance_value:
            output_args.append(f"flux_guidance_value={self.config.flux_guidance_value}")
        if self.config.flux_guidance_min:
            output_args.append(f"flux_guidance_min={self.config.flux_guidance_min}")
        if self.config.flux_guidance_mode == "random-range":
            output_args.append(f"flux_guidance_max={self.config.flux_guidance_max}")
            output_args.append(f"flux_guidance_min={self.config.flux_guidance_min}")
        if self.config.flow_use_beta_schedule:
            output_args.append(f"flow_beta_schedule_alpha={self.config.flow_beta_schedule_alpha}")
            output_args.append(f"flow_beta_schedule_beta={self.config.flow_beta_schedule_beta}")
        if self.config.flux_attention_masked_training:
            output_args.append("flux_attention_masked_training")
        if self.config.t5_padding != "unmodified":
            output_args.append(f"t5_padding={self.config.t5_padding}")
        if (
            self.config.model_type == "lora"
            and self.config.lora_type == "standard"
            and self.config.flux_lora_target is not None
        ):
            output_args.append(f"flux_lora_target={self.config.flux_lora_target}")
        output_str = f" (extra parameters={output_args})" if output_args else " (no special parameters set)"

        return output_str

    @classmethod
    def register_config_requirements(cls):
        """Register configuration rules for Flux model."""
        rules = [
            make_override_rule(
                field_name="aspect_bucket_alignment",
                value=64,
                message="Flux requires aspect bucket alignment of 64px",
                example="aspect_bucket_alignment: 64",
            ),
            ConfigRule(
                field_name="tokenizer_max_length",
                rule_type=RuleType.MAX,
                value=512,
                message="Flux supports a maximum of 512 tokens",
                example="tokenizer_max_length: 512  # Maximum supported",
                error_level="warning",
            ),
            ConfigRule(
                field_name="base_model_precision",
                rule_type=RuleType.CHOICES,
                value=["int8-quanto", "fp8-torchao", "no_change", "int4-quanto", "nf4-torchao", "fp8-torchao-compile"],
                message="Flux supports limited precision options",
                example="base_model_precision: fp8-torchao",
                error_level="warning",
            ),
            ConfigRule(
                field_name="prediction_type",
                rule_type=RuleType.CUSTOM,
                value=None,
                message="Flux uses flow matching and does not support custom prediction types",
                error_level="warning",
            ),
        ]

        ConfigRegistry.register_rules("flux", rules)
        ConfigRegistry.register_validator(
            "flux",
            cls._validate_flux_specific,
            """Validates Flux-specific requirements:
- Warns about attention slicing on MPS devices
- Validates prediction_type compatibility
- Ensures proper aspect bucket alignment
- Checks tokenizer max length constraints""",
        )

    def pretrained_load_args(self, pretrained_load_args: dict) -> dict:
        args = super().pretrained_load_args(pretrained_load_args)
        return apply_musubi_pretrained_defaults(self.config, args)

    @staticmethod
    def _validate_flux_specific(config: dict) -> List[ValidationResult]:
        """Custom validation logic for Flux models."""
        results = []

        # Check attention slicing on MPS
        if config.get("unet_attention_slice") and torch.backends.mps.is_available():
            results.append(
                ValidationResult(
                    passed=False,
                    field="unet_attention_slice",
                    message="Using attention slicing when training Flux on MPS can result in NaN errors on the first backward pass",
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
                    message="Flux does not support custom prediction types - it uses flow matching",
                    level="warning",
                    suggestion="Remove prediction_type from your configuration",
                )
            )

        return results


# Register Flux configuration requirements when module is imported
Flux.register_config_requirements()


from simpletuner.helpers.models.registry import ModelRegistry

ModelRegistry.register("flux", Flux)
