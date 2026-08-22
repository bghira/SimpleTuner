import logging
import os
import random
from dataclasses import fields, is_dataclass, replace
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, SD3ControlNetModel
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

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
from simpletuner.helpers.models.sd3.controlnet import StableDiffusion3ControlNetPipeline
from simpletuner.helpers.models.sd3.pipeline import StableDiffusion3Img2ImgPipeline, StableDiffusion3Pipeline
from simpletuner.helpers.models.sd3.transformer import SD3Transformer2DModel
from simpletuner.helpers.models.tae.types import ImageTAESpec
from simpletuner.helpers.training.explorative_modeling import repeat_batch_for_candidates

logger = logging.getLogger(__name__)
is_primary_process = True
if os.environ.get("RANK") is not None:
    if int(os.environ.get("RANK")) != 0:
        is_primary_process = False
logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO") if is_primary_process else "ERROR")


def _encode_sd3_prompt_with_t5(
    text_encoder,
    tokenizer,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
    zero_padding_tokens: bool = True,
    max_sequence_length: int = 77,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    attention_mask = text_inputs.attention_mask.to(device)

    if zero_padding_tokens:
        # for some reason, SAI's reference code doesn't bother to mask the prompt embeddings.
        # this can lead to a problem where the model fails to represent short and long prompts equally well.
        # additionally, the model learns the bias of the prompt embeds' noise.
        return prompt_embeds * attention_mask.unsqueeze(-1).expand(prompt_embeds.shape)
    else:
        return prompt_embeds


def _encode_sd3_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
    max_token_length: int = 77,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_token_length,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


class SD3(ImageModelFoundation):
    NAME = "Stable Diffusion 3.x"
    MODEL_DESCRIPTION = "Latest SD3 architecture with improved quality"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    COMFYUI_LORA_PRESERVE_COMPONENT_PREFIXES = {"transformer"}
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 16
    VALIDATION_PREVIEW_SPEC = ImageTAESpec(repo_id="madebyollin/taesd3")
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    SLIDER_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    # Only training the Attention blocks by default seems to help more with SD3.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = SD3Transformer2DModel
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: StableDiffusion3Pipeline,
        PipelineTypes.IMG2IMG: StableDiffusion3Img2ImgPipeline,
        PipelineTypes.CONTROLNET: StableDiffusion3ControlNetPipeline,
    }
    MODEL_SUBFOLDER = "transformer"
    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "medium"
    HUGGINGFACE_PATHS = {
        "medium": "stabilityai/stable-diffusion-3.5-medium",
        "large": "stabilityai/stable-diffusion-3.5-large",
    }
    MODEL_LICENSE = "other"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "CLIP-L/14",
            "tokenizer": CLIPTokenizer,
            "tokenizer_subfolder": "tokenizer",
            "model": CLIPTextModelWithProjection,
        },
        "text_encoder_2": {
            "name": "CLIP-G/14",
            "tokenizer": CLIPTokenizer,
            "subfolder": "text_encoder_2",
            "tokenizer_subfolder": "tokenizer_2",
            "model": CLIPTextModelWithProjection,
        },
        "text_encoder_3": {
            "name": "T5 XXL v1.1",
            "tokenizer": T5TokenizerFast,
            "subfolder": "text_encoder_3",
            "tokenizer_subfolder": "tokenizer_3",
            "model": T5EncoderModel,
        },
    }

    def supports_crepa_self_flow(self) -> bool:
        return True

    def _prepare_crepa_self_flow_batch(self, batch: dict, state: dict) -> dict:
        patch_size = getattr(getattr(self.unwrap_model(self.model), "config", None), "patch_size", 2)
        return self._prepare_image_crepa_self_flow_batch(batch, state, patch_size=patch_size)

    def _select_crepa_hidden_states(self, prepared_batch: dict, hidden_states_buffer):
        if hidden_states_buffer is None:
            return None
        crepa = getattr(self, "crepa_regularizer", None)
        block_idx = prepared_batch.get(
            "crepa_capture_block_index",
            getattr(crepa, "block_index", None),
        )
        if block_idx is None:
            return None
        return hidden_states_buffer.get(f"layer_{int(block_idx)}")

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        self._validate_xm_support()

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        # SD3 has 18 transformer blocks
        # Leave at least 1 block on GPU
        return 17

    @classmethod
    def get_acceleration_presets(cls) -> list[AccelerationPreset]:
        # Common settings for memory optimization presets
        _base_memory_config = {
            "base_model_precision": "no_change",
            "gradient_checkpointing": True,
        }

        return [
            # RamTorch presets - 3 levels
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="light",
                name="RamTorch - Light",
                description="Streams 6 of 18 transformer blocks from CPU RAM.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~30%",
                tradeoff_speed="Increases training time by ~20%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={
                    **_base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "transformer_blocks.0.*,transformer_blocks.1.*,transformer_blocks.2.*,transformer_blocks.3.*,transformer_blocks.4.*,transformer_blocks.5.*",
                },
            ),
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="balanced",
                name="RamTorch - Balanced",
                description="Streams 12 of 18 transformer blocks from CPU RAM.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~55%",
                tradeoff_speed="Increases training time by ~40%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={
                    **_base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "transformer_blocks.0.*,transformer_blocks.1.*,transformer_blocks.2.*,transformer_blocks.3.*,transformer_blocks.4.*,transformer_blocks.5.*,transformer_blocks.6.*,transformer_blocks.7.*,transformer_blocks.8.*,transformer_blocks.9.*,transformer_blocks.10.*,transformer_blocks.11.*",
                },
            ),
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="aggressive",
                name="RamTorch - Aggressive",
                description="Streams all transformer blocks from CPU RAM.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~85%",
                tradeoff_speed="Increases training time by ~70%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={
                    **_base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "transformer_blocks.*",
                },
            ),
            # Block Swap presets - 3 levels
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="light",
                name="Block Swap - Light",
                description="Swaps 5 of 18 blocks (~28%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~25%",
                tradeoff_speed="Increases training time by ~15%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={**_base_memory_config, "musubi_blocks_to_swap": 5},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="balanced",
                name="Block Swap - Balanced",
                description="Swaps 9 of 18 blocks (~50%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~45%",
                tradeoff_speed="Increases training time by ~35%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={**_base_memory_config, "musubi_blocks_to_swap": 9},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="aggressive",
                name="Block Swap - Aggressive",
                description="Swaps 15 of 18 blocks (~83%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~75%",
                tradeoff_speed="Increases training time by ~60%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={**_base_memory_config, "musubi_blocks_to_swap": 15},
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

    def controlnet_init(self):
        logger.info("Creating the SD3 controlnet..")

        if self.config.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            self.controlnet = SD3ControlNetModel.from_pretrained(self.config.controlnet_model_name_or_path)
        else:
            logger.info("Initializing controlnet weights from base model")
            # SD3ControlNetModel.from_transformer adds 1 extra conditioning channel by default
            # We set it to 0 because it's not really needed and increases complexity.
            num_extra_channels = 0
            self.controlnet = SD3ControlNetModel.from_transformer(
                self.unwrap_model(self.model),
                num_extra_conditioning_channels=num_extra_channels,
            )

        self.controlnet = self.controlnet.to(
            device=self.accelerator.device,
            dtype=(self.config.base_weight_dtype if hasattr(self.config, "base_weight_dtype") else self.config.weight_dtype),
        )
        # Log the expected input channels for debugging
        if hasattr(self.controlnet, "pos_embed_input") and hasattr(self.controlnet.pos_embed_input, "proj"):
            in_channels = self.controlnet.pos_embed_input.proj.in_channels
            logger.info(f"ControlNet expects {in_channels} input channels")

    def tread_init(self):
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

        logger.info("TREAD training is enabled for SD3")

    def requires_conditioning_latents(self) -> bool:
        """
        SD3 ControlNet uses latent inputs with optional extra conditioning channels.

        By default (sd3_controlnet_extra_conditioning_channels=0), it uses 16-channel latents.
        With extra channels, it expects latents + additional control signals.
        Beware, the pipeline doesn't seem to play well with the added channel.
        """
        if self.config.controlnet:
            return True  # SD3 uses latent inputs for controlnet
        return False

    def requires_conditioning_validation_inputs(self) -> bool:
        """
        Whether this model / flavour requires conditioning inputs during validation.
        """
        if self.config.controlnet:
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
        prompt_embeds, pooled_prompt_embeds = text_embedding

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds.squeeze(0),
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # Only unsqueeze if it's missing the batch dimension
        prompt_embeds = text_embedding["prompt_embeds"]
        pooled_prompt_embeds = text_embedding["pooled_prompt_embeds"]

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:  # Shape: [seq, dim]
            prompt_embeds = prompt_embeds.unsqueeze(0)  # Shape: [1, seq, dim]
        if pooled_prompt_embeds.dim() == 1:  # Shape: [dim]
            pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0)  # Shape: [1, dim]

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # Only unsqueeze if it's missing the batch dimension
        prompt_embeds = text_embedding["prompt_embeds"]
        pooled_prompt_embeds = text_embedding["pooled_prompt_embeds"]

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:  # Shape: [seq, dim]
            prompt_embeds = prompt_embeds.unsqueeze(0)  # Shape: [1, seq, dim]
        if pooled_prompt_embeds.dim() == 1:  # Shape: [dim]
            pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0)  # Shape: [1, dim]

        return {
            "negative_prompt_embeds": prompt_embeds,
            "negative_pooled_prompt_embeds": pooled_prompt_embeds,
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        num_images_per_prompt = 1

        clip_tokenizers = self.tokenizers[:2]
        clip_text_encoders = self.text_encoders[:2]

        clip_prompt_embeds_list = []
        clip_pooled_prompt_embeds_list = []
        for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
            prompt_embeds, pooled_prompt_embeds = _encode_sd3_prompt_with_clip(
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                prompt=prompts,
                device=self.accelerator.device,
                num_images_per_prompt=num_images_per_prompt,
            )
            clip_prompt_embeds_list.append(prompt_embeds)
            clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

        clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
        pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)
        zero_padding_tokens = True if self.config.t5_padding == "zero" else False
        t5_prompt_embed = _encode_sd3_prompt_with_t5(
            self.text_encoders[-1],
            self.tokenizers[-1],
            prompt=prompts,
            num_images_per_prompt=num_images_per_prompt,
            device=self.accelerator.device,
            zero_padding_tokens=zero_padding_tokens,
            max_sequence_length=self.config.tokenizer_max_length,
        )

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds,
            (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1]),
        )
        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

        return prompt_embeds, pooled_prompt_embeds

    def model_predict(self, prepared_batch):
        if self._xm_noise_candidates_enabled(prepared_batch):
            self._prepare_xm_noise_candidates(prepared_batch)
            model_output = self._model_predict_single(prepared_batch)
            model_output["xm_candidate_count"] = self.xm_config.candidate_count
            return model_output
        return self._model_predict_single(prepared_batch)

    def _repeat_xm_candidate_value(self, value, candidate_count: int, batch_size: int):
        if torch.is_tensor(value):
            if value.ndim == 0 or value.shape[0] != batch_size:
                return value
            return repeat_batch_for_candidates(value, candidate_count)
        if isinstance(value, list):
            if len(value) == batch_size:
                return value * candidate_count
            return [self._repeat_xm_candidate_value(item, candidate_count, batch_size) for item in value]
        if isinstance(value, tuple):
            if len(value) == batch_size:
                return tuple(list(value) * candidate_count)
            return tuple(self._repeat_xm_candidate_value(item, candidate_count, batch_size) for item in value)
        if isinstance(value, dict):
            return {key: self._repeat_xm_candidate_value(item, candidate_count, batch_size) for key, item in value.items()}
        if is_dataclass(value) and not isinstance(value, type):
            replacements = {
                field.name: self._repeat_xm_candidate_value(getattr(value, field.name), candidate_count, batch_size)
                for field in fields(value)
            }
            return replace(value, **replacements)
        return value

    def _prepare_xm_noise_candidates(self, prepared_batch: dict) -> dict:
        xm_config = self.xm_config
        candidate_count = int(xm_config.candidate_count)
        latents = prepared_batch.get("latents")
        timesteps = prepared_batch.get("timesteps")
        sigmas = prepared_batch.get("sigmas")
        if not torch.is_tensor(latents) or not torch.is_tensor(timesteps) or not torch.is_tensor(sigmas):
            raise ValueError("SD3 XM noise-candidate training requires latents, timesteps, and sigmas tensors.")
        if "noisy_latents" not in prepared_batch:
            raise ValueError("SD3 XM noise-candidate training requires prepared noisy_latents.")
        if prepared_batch.get("target") is not None:
            raise ValueError("SD3 XM noise-candidate training cannot be used with an explicit prepared target.")

        batch_size = latents.shape[0]
        expanded_batch = {
            key: self._repeat_xm_candidate_value(value, candidate_count, batch_size) for key, value in prepared_batch.items()
        }
        expanded_latents = expanded_batch["latents"]
        candidate_noise = torch.randn_like(expanded_latents)
        sigma_source = expanded_batch.get("mixflow_interpolation_sigmas")
        if sigma_source is None:
            sigma_source = expanded_batch["sigmas"]
        if not torch.is_tensor(sigma_source):
            raise ValueError("SD3 XM noise-candidate training requires tensor sigmas for interpolation.")
        flat_sigmas = sigma_source.reshape(sigma_source.shape[0], -1)
        if flat_sigmas.shape[1] > 1 and not torch.allclose(flat_sigmas, flat_sigmas[:, :1].expand_as(flat_sigmas)):
            raise ValueError("SD3 XM noise-candidate training requires per-sample scalar sigmas.")
        interpolation_grid = self._expand_sigma_values(sigma_source, expanded_latents)
        expanded_batch["noise"] = candidate_noise
        expanded_batch["input_noise"] = candidate_noise
        expanded_batch["noisy_latents"] = (
            1.0 - interpolation_grid
        ) * expanded_latents + interpolation_grid * candidate_noise
        expanded_batch["flow_target"] = self.get_flow_matching_target(
            expanded_batch,
            latents=expanded_latents,
            noise=candidate_noise,
            prefer_explicit_target=False,
        ).to(device=expanded_latents.device, dtype=expanded_latents.dtype)
        expanded_batch.pop("twinflow_tt", None)
        expanded_batch.pop("twinflow_rng_state", None)
        if self._twinflow_active():
            self._prepare_twinflow_metadata(expanded_batch)
        expanded_batch["xm_candidate_count"] = candidate_count
        expanded_batch["xm_original_batch_size"] = batch_size

        prepared_batch.clear()
        prepared_batch.update(expanded_batch)
        return prepared_batch

    def _model_predict_single(self, prepared_batch):
        hidden_states_buffer = self._new_hidden_state_buffer()
        timesteps = prepared_batch["timesteps"].to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        grounding_kwargs = self._build_grounding_position_net_kwargs(prepared_batch.get("grounding_batch"))
        transformer_kwargs = {
            "hidden_states": prepared_batch["noisy_latents"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            "timestep": timesteps,
            "timestep_sign": prepared_batch.get("twinflow_time_sign"),
            "encoder_hidden_states": prepared_batch["encoder_hidden_states"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            "pooled_projections": prepared_batch["add_text_embeds"].to(
                device=self.accelerator.device,
                dtype=self.config.weight_dtype,
            ),
            "return_dict": False,
            "hidden_states_buffer": hidden_states_buffer,
            "grounding_kwargs": grounding_kwargs,
        }
        self._apply_flowmap_r_timestep_kwargs(transformer_kwargs, prepared_batch)
        model_pred = self.model(**transformer_kwargs)[0]

        return {
            "model_prediction": model_pred,
            "crepa_hidden_states": self._select_crepa_hidden_states(prepared_batch, hidden_states_buffer),
            "hidden_states_buffer": hidden_states_buffer,
        }

    def prepare_controlnet_conditioning(self, conditioning_latents: torch.Tensor) -> torch.Tensor:
        """
        Prepare conditioning inputs for SD3 ControlNet.

        SD3 ControlNet can be configured with extra conditioning channels.
        We pray the user doesn't go this route, because it leads to pipeline complexity.

        Args:
            conditioning_latents: The conditioning latents from the dataloader

        Returns:
            Properly formatted conditioning tensor for the controlnet
        """
        # Check what the controlnet expects
        if hasattr(self.controlnet, "pos_embed_input") and hasattr(self.controlnet.pos_embed_input, "proj"):
            # Access the weight tensor shape to determine expected channels
            # Weight shape for Conv2d is [out_channels, in_channels, kernel_h, kernel_w]
            weight_shape = self.controlnet.pos_embed_input.proj.weight.shape
            expected_channels = weight_shape[1]  # in_channels is the second dimension
            actual_channels = conditioning_latents.shape[1]

            if expected_channels != actual_channels:
                if expected_channels == 17 and actual_channels == 16:
                    # extra channel required for 8b controlnet
                    batch_size, _, height, width = conditioning_latents.shape

                    extra_channel = torch.zeros(
                        batch_size,
                        1,
                        height,
                        width,
                        device=conditioning_latents.device,
                        dtype=conditioning_latents.dtype,
                    )

                    conditioning_latents = torch.cat([conditioning_latents, extra_channel], dim=1)

                elif expected_channels < actual_channels:
                    # ControlNet expects fewer channels, might need to select specific channels
                    logger.warning(
                        f"ControlNet expects {expected_channels} channels but got {actual_channels}. "
                        f"Using first {expected_channels} channels."
                    )
                    conditioning_latents = conditioning_latents[:, :expected_channels]

                else:
                    raise ValueError(
                        f"Channel mismatch: ControlNet expects {expected_channels} channels "
                        f"but received {actual_channels} channels. "
                        "Check your controlnet configuration or conditioning data."
                    )

        return conditioning_latents

    def controlnet_predict(self, prepared_batch: dict) -> dict:
        if self._xm_noise_candidates_enabled(prepared_batch):
            self._prepare_xm_noise_candidates(prepared_batch)
            model_output = self._controlnet_predict_single(prepared_batch)
            model_output["xm_candidate_count"] = self.xm_config.candidate_count
            return model_output
        return self._controlnet_predict_single(prepared_batch)

    def _controlnet_predict_single(self, prepared_batch: dict) -> dict:
        """
        Perform a forward pass with ControlNet for SD3 model.

        Args:
            prepared_batch: Dictionary containing the batch data including conditioning_latents

        Returns:
            Dictionary containing the model prediction
        """
        # Get and prepare the conditioning
        controlnet_cond = prepared_batch["conditioning_latents"].to(
            device=self.accelerator.device, dtype=self.config.weight_dtype
        )
        controlnet_cond = self.prepare_controlnet_conditioning(controlnet_cond)
        control_block_samples = self.controlnet(
            hidden_states=prepared_batch["noisy_latents"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            timestep=prepared_batch["timesteps"],
            encoder_hidden_states=prepared_batch["encoder_hidden_states"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            pooled_projections=prepared_batch["add_text_embeds"].to(
                device=self.accelerator.device,
                dtype=self.config.weight_dtype,
            ),
            joint_attention_kwargs=None,
            controlnet_cond=controlnet_cond,
            conditioning_scale=1.0,  # You might want to make this configurable
            return_dict=False,
        )[0]
        control_block_samples = [sample.to(dtype=self.config.base_weight_dtype) for sample in control_block_samples]
        model_pred = self.model(
            hidden_states=prepared_batch["noisy_latents"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            timestep=prepared_batch["timesteps"],
            encoder_hidden_states=prepared_batch["encoder_hidden_states"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            pooled_projections=prepared_batch["add_text_embeds"].to(
                device=self.accelerator.device,
                dtype=self.config.weight_dtype,
            ),
            timestep_sign=prepared_batch.get("twinflow_time_sign"),
            block_controlnet_hidden_states=control_block_samples,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        return {"model_prediction": model_pred}

    def loss_with_logs(self, prepared_batch: dict, model_output, apply_conditioning_mask: bool = True):
        candidate_count = model_output.get("xm_candidate_count") if isinstance(model_output, dict) else None
        if not candidate_count:
            return super().loss_with_logs(
                prepared_batch,
                model_output,
                apply_conditioning_mask=apply_conditioning_mask,
            )
        return self._xm_noise_loss_with_logs(
            prepared_batch,
            model_output,
            candidate_count=int(candidate_count),
            apply_conditioning_mask=apply_conditioning_mask,
        )

    def _xm_diffusion_loss_tensor(
        self, prepared_batch: dict, model_output: dict, apply_conditioning_mask: bool
    ) -> torch.Tensor:
        target = self.get_prediction_target(prepared_batch)
        if target is None:
            raise ValueError("Target is None. Cannot compute SD3 XM loss.")
        model_pred = model_output["model_prediction"]
        if model_pred.shape != target.shape:
            raise ValueError(
                f"SD3 XM loss expected prediction and target shapes to match, got {tuple(model_pred.shape)} and {tuple(target.shape)}."
            )

        loss_type = getattr(self.config, "loss_type", "l2")
        if loss_type in ["huber", "smooth_l1"]:
            timesteps = prepared_batch["timesteps"]
            if getattr(self.config, "huber_schedule", "constant") != "constant":
                losses = []
                for idx in range(model_pred.shape[0]):
                    huber_c = self.compute_scheduled_huber_c(timesteps[idx : idx + 1]).item()
                    losses.append(
                        self.conditional_loss(
                            model_pred[idx : idx + 1].float(),
                            target[idx : idx + 1].float(),
                            reduction="none",
                            loss_type=loss_type,
                            huber_c=huber_c,
                        )
                    )
                loss = torch.cat(losses, dim=0)
            else:
                loss = self.conditional_loss(
                    model_pred.float(),
                    target.float(),
                    reduction="none",
                    loss_type=loss_type,
                    huber_c=getattr(self.config, "huber_c", 0.1),
                )
        elif loss_type == "l2":
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        else:
            raise ValueError(f"SD3 XM noise-candidate training does not support loss_type={loss_type!r}.")

        loss_mask_type = prepared_batch.get("loss_mask_type")
        if not loss_mask_type:
            legacy_type = prepared_batch.get("conditioning_type")
            if legacy_type in ("mask", "segmentation"):
                loss_mask_type = legacy_type
        if loss_mask_type == "mask" and apply_conditioning_mask:
            mask_image = (
                prepared_batch["conditioning_pixel_values"].to(dtype=loss.dtype, device=loss.device)[:, 0].unsqueeze(1)
            )
            mask_image = torch.nn.functional.interpolate(mask_image, size=loss.shape[2:], mode="area")
            mask_image = mask_image / 2 + 0.5
            loss = loss * mask_image
        elif loss_mask_type == "segmentation" and apply_conditioning_mask:
            if random.random() < self.config.masked_loss_probability:
                mask_image = prepared_batch["conditioning_pixel_values"].to(dtype=loss.dtype, device=loss.device)
                mask_image = torch.sum(mask_image, dim=1, keepdim=True) / 3
                mask_image = torch.nn.functional.interpolate(mask_image, size=loss.shape[2:], mode="area")
                mask_image = mask_image / 2 + 0.5
                mask_image = (mask_image > 0).to(dtype=loss.dtype, device=loss.device)
                loss = loss * mask_image
        return loss

    def get_lora_target_layers(self):
        manual_targets = self._get_peft_lora_target_modules()
        if manual_targets:
            return manual_targets
        if getattr(self.config, "slider_lora_target", False) and self.config.lora_type.lower() == "standard":
            return getattr(self, "SLIDER_LORA_TARGET", None) or self.DEFAULT_SLIDER_LORA_TARGET
        # Override for ControlNet training if needed
        if self.config.model_type == "lora" and self.config.controlnet:
            # Comprehensive targeting including all layers
            targets = []

            # Controlnet blocks
            for i in range(12):
                targets.append(f"controlnet_blocks.{i}")

            # Position embeddings
            targets.extend(
                [
                    "pos_embed.proj",
                    "pos_embed_input.proj",
                ]
            )

            # Context and time embedders
            targets.append("context_embedder")
            targets.extend(
                [
                    "time_text_embed.timestep_embedder.linear_1",
                    "time_text_embed.timestep_embedder.linear_2",
                    "time_text_embed.text_embedder.linear_1",
                    "time_text_embed.text_embedder.linear_2",
                ]
            )

            # All attention layers in transformer blocks
            for i in range(12):
                # Main attention
                targets.extend(
                    [
                        f"transformer_blocks.{i}.attn.to_k",
                        f"transformer_blocks.{i}.attn.to_q",
                        f"transformer_blocks.{i}.attn.to_v",
                        f"transformer_blocks.{i}.attn.to_out.0",
                        f"transformer_blocks.{i}.attn.add_k_proj",
                        f"transformer_blocks.{i}.attn.add_q_proj",
                        f"transformer_blocks.{i}.attn.add_v_proj",
                        f"transformer_blocks.{i}.attn.to_add_out",
                    ]
                )
                # Cross attention
                targets.extend(
                    [
                        f"transformer_blocks.{i}.attn2.to_k",
                        f"transformer_blocks.{i}.attn2.to_q",
                        f"transformer_blocks.{i}.attn2.to_v",
                        f"transformer_blocks.{i}.attn2.to_out.0",
                    ]
                )
                # Feed-forward networks
                targets.extend(
                    [
                        f"transformer_blocks.{i}.ff.net.0.proj",
                        f"transformer_blocks.{i}.ff.net.2",
                        f"transformer_blocks.{i}.ff_context.net.0.proj",
                        f"transformer_blocks.{i}.ff_context.net.2",
                    ]
                )

            return targets

        # Default LoRA targets
        if self.config.lora_type.lower() == "standard":
            return self.DEFAULT_LORA_TARGET
        elif self.config.lora_type.lower() == "lycoris":
            return self.DEFAULT_LYCORIS_TARGET
        else:
            raise NotImplementedError(f"Unknown LoRA target type {self.config.lora_type}.")

    def check_user_config(self):
        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        t5_max_length = 154
        if self.config.tokenizer_max_length is None:
            self.config.tokenizer_max_length = t5_max_length
        if int(self.config.tokenizer_max_length) > t5_max_length:
            if not self.config.i_know_what_i_am_doing:
                logger.warning(f"Updating T5 XXL tokeniser max length to {t5_max_length} for {self.NAME}.")
                self.config.tokenizer_max_length = t5_max_length
            else:
                logger.warning(
                    f"-!- {self.NAME} supports a max length of {t5_max_length} tokens, but you have supplied `--i_know_what_i_am_doing`, so this limit will not be enforced. -!-"
                )
                logger.warning(
                    f"The model will begin to collapse after a short period of time, if the model you are continuing from has not been tuned beyond {t5_max_length} tokens."
                )
        # Disable custom VAEs.
        self.config.pretrained_vae_model_name_or_path = None
        if self.config.aspect_bucket_alignment != 64:
            logger.warning("MM-DiT requires an alignment value of 64px. Overriding the value of --aspect_bucket_alignment.")
            self.config.aspect_bucket_alignment = 64
        if self.config.sd3_t5_uncond_behaviour is None:
            self.config.sd3_t5_uncond_behaviour = self.config.sd3_clip_uncond_behaviour
        logger.info(
            f"{self.NAME} embeds for unconditional captions: t5={self.config.sd3_t5_uncond_behaviour}, clip={self.config.sd3_clip_uncond_behaviour}"
        )

        # ControlNet specific configuration
        if self.config.controlnet:
            self.config.sd3_controlnet_extra_conditioning_channels = 0

    def custom_model_card_schedule_info(self):
        output_args = []
        if self.config.flow_schedule_auto_shift:
            output_args.append("flow_schedule_auto_shift")
        if self.config.flow_schedule_shift is not None:
            output_args.append(f"shift={self.config.flow_schedule_shift}")
        if self.config.flow_use_beta_schedule:
            output_args.append(f"flow_beta_schedule_alpha={self.config.flow_beta_schedule_alpha}")
            output_args.append(f"flow_beta_schedule_beta={self.config.flow_beta_schedule_beta}")
        if self.config.flow_use_uniform_schedule:
            output_args.append(f"flow_use_uniform_schedule")
        output_str = f" (extra parameters={output_args})" if output_args else " (no special parameters set)"

        return output_str


from simpletuner.helpers.models.registry import ModelRegistry

ModelRegistry.register("sd3", SD3)
