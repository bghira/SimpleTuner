import logging
import os
import random
from typing import Optional

import torch
from diffusers.pipelines import LTXPipeline
from transformers import AutoTokenizer, T5EncoderModel

from simpletuner.helpers.acceleration import AccelerationBackend, AccelerationPreset
from simpletuner.helpers.models.common import ModelTypes, PipelineTypes, PredictionTypes, VideoModelFoundation
from simpletuner.helpers.models.ltxvideo import (
    apply_first_frame_protection,
    make_i2v_conditioning_mask,
    pack_ltx_latents,
    unpack_ltx_latents,
)
from simpletuner.helpers.models.ltxvideo.autoencoder import AutoencoderKLLTXVideo
from simpletuner.helpers.models.ltxvideo.transformer import LTXVideoTransformer3DModel
from simpletuner.helpers.musubi_block_swap import apply_musubi_pretrained_defaults
from simpletuner.helpers.training.multi_process import _get_rank

logger = logging.getLogger(__name__)
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class LTXVideo(VideoModelFoundation):
    NAME = "LTXVideo"
    MODEL_DESCRIPTION = "Video generation model with flow matching"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKLLTXVideo
    LATENT_CHANNEL_COUNT = 128
    DEFAULT_NOISE_SCHEDULER = "flow_matching"
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    # Only training the Attention blocks by default.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = LTXVideoTransformer3DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: LTXPipeline,
        # PipelineTypes.IMG2IMG: None,
        # PipelineTypes.CONTROLNET: None,
    }

    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "0.9.5"
    HUGGINGFACE_PATHS = {
        "0.9.5": "Lightricks/LTX-Video-0.9.5",
        "0.9.0": "Lightricks/LTX-Video",
    }
    MODEL_LICENSE = "apache-2.0"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "T5 XXL v1.1",
            "tokenizer": AutoTokenizer,
            "subfolder": "text_encoder",
            "tokenizer_subfolder": "tokenizer",
            "model": T5EncoderModel,
        },
    }

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        # LTXVideo has 28 transformer blocks
        # Leave at least 1 block on GPU
        return 27

    @classmethod
    def get_acceleration_presets(cls) -> list[AccelerationPreset]:
        return [
            # Basic tab - RamTorch options
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="basic",
                name="RamTorch - Basic",
                description="Streams half of transformer block weights from CPU RAM.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~30%",
                tradeoff_speed="Increases training time by ~20%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={
                    "ramtorch": True,
                    "ramtorch_target_modules": "transformer_blocks.0,transformer_blocks.1,transformer_blocks.2,transformer_blocks.3,transformer_blocks.4,transformer_blocks.5,transformer_blocks.6,transformer_blocks.7,transformer_blocks.8,transformer_blocks.9,transformer_blocks.10,transformer_blocks.11,transformer_blocks.12,transformer_blocks.13",
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
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={
                    "ramtorch": True,
                    "ramtorch_target_modules": "transformer_blocks.*",
                },
            ),
            # Basic tab - Block swap options
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="light",
                name="Block Swap - Light",
                description="Swaps 7 of 28 blocks (~25%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~20%",
                tradeoff_speed="Increases training time by ~15%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={"musubi_blocks_to_swap": 7},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="balanced",
                name="Block Swap - Balanced",
                description="Swaps 14 of 28 blocks (~50%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~45%",
                tradeoff_speed="Increases training time by ~30%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={"musubi_blocks_to_swap": 14},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="aggressive",
                name="Block Swap - Aggressive",
                description="Swaps 21 of 28 blocks (~75%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~65%",
                tradeoff_speed="Increases training time by ~55%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={"musubi_blocks_to_swap": 21},
            ),
            # Advanced tab - DeepSpeed options
            AccelerationPreset(
                backend=AccelerationBackend.DEEPSPEED_ZERO_1,
                level="zero1",
                name="DeepSpeed ZeRO Stage 1",
                description="Shards optimizer states across GPUs.",
                tab="advanced",
                tradeoff_vram="Reduces optimizer memory by 75% per GPU",
                tradeoff_speed="Minimal overhead",
                tradeoff_notes="Requires multi-GPU setup.",
                config={"deepspeed": "zero1"},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.DEEPSPEED_ZERO_2,
                level="zero2",
                name="DeepSpeed ZeRO Stage 2",
                description="Shards optimizer states and gradients across GPUs.",
                tab="advanced",
                tradeoff_vram="Reduces optimizer + gradient memory by 85% per GPU",
                tradeoff_speed="Moderate communication overhead",
                tradeoff_notes="Requires multi-GPU setup.",
                config={"deepspeed": "zero2"},
            ),
        ]

    def apply_i2v_augmentation(self, batch):
        num_frame_latents = batch["latents"].shape[2]
        if num_frame_latents > 1 and batch["is_i2v_data"] is True:
            # the theory is that if you have a single-frame latent, we expand it to num_frames and then do less destructive denoising.
            single_frame_latents = batch["latents"]
            if num_frame_latents > 1:
                # for an actual video though, we'll grab one frame using the worst syntax we can think of:
                single_frame_latents = batch["latents"][:, :, 0, :, :].unsqueeze(dim=2)
                logger.info(f"All latents shape: {batch['latents'].shape}")
                logger.info(f"Single frame latents shape: {single_frame_latents.shape}")
            batch["i2v_conditioning_mask"] = make_i2v_conditioning_mask(batch["latents"], protect_frame_index=0)
            batch["timesteps"], batch["noise"], new_sigmas = apply_first_frame_protection(
                batch["latents"],
                batch["timesteps"],
                batch["noise"],
                batch["i2v_conditioning_mask"],
                protect_first_frame=self.config.ltx_protect_first_frame,
                first_frame_probability=self.config.ltx_i2v_prob,
                partial_noise_fraction=self.config.ltx_partial_noise_fraction,
            )
            if new_sigmas is not None:
                batch["sigmas"] = new_sigmas
            logger.info(f"Applied mask {batch['i2v_conditioning_mask'].shape} to timestep {batch['timesteps'].shape}")

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        """
        When we're running the pipeline, we'll update the kwargs specifically for this model here.
        """
        # Wan video should max out around 81 frames for efficiency.
        pipeline_kwargs["num_frames"] = min(125, self.config.validation_num_video_frames or 125)
        # pipeline_kwargs["output_type"] = "pil"
        # replace embeds with prompt

        return pipeline_kwargs

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        """
        Models can optionally format the stored text embedding, eg. in a dict, or
        filter certain outputs from appearing in the file cache.

        self.config:
            text_embedding (torch.Tensor): The embed to adjust.

        Returns:
            torch.Tensor: The adjusted embed. By default, this method does nothing.
        """
        prompt_embeds, prompt_attention_mask, _, _ = text_embedding

        return {
            "prompt_embeds": prompt_embeds,
            "attention_masks": prompt_attention_mask,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # Only unsqueeze if it's missing the batch dimension
        prompt_embeds = text_embedding["prompt_embeds"]
        attention_masks = text_embedding["attention_masks"]

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:  # Shape: [seq, dim]
            prompt_embeds = prompt_embeds.unsqueeze(0)  # Shape: [1, seq, dim]
        if attention_masks.dim() == 1:  # Shape: [seq]
            attention_masks = attention_masks.unsqueeze(0)  # Shape: [1, seq]

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": attention_masks,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # Only unsqueeze if it's missing the batch dimension
        prompt_embeds = text_embedding["prompt_embeds"]
        attention_masks = text_embedding["attention_masks"]

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:  # Shape: [seq, dim]
            prompt_embeds = prompt_embeds.unsqueeze(0)  # Shape: [1, seq, dim]
        if attention_masks.dim() == 1:  # Shape: [seq]
            attention_masks = attention_masks.unsqueeze(0)  # Shape: [1, seq]

        return {
            "negative_prompt_embeds": prompt_embeds,
            "negative_prompt_attention_mask": attention_masks,
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
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = pipeline.encode_prompt(
            prompt=prompts,
            device=self.accelerator.device,
        )

        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    def model_predict(self, prepared_batch):
        if prepared_batch["noisy_latents"].shape[1] != 128:
            raise ValueError(
                "LTX Video requires a latent size of 128 channels. Ensure you are using the correct VAE cache path."
                f" Batch received: {prepared_batch}"
            )
        scale_value = 1
        height, width = (
            prepared_batch["noisy_latents"].shape[3] * scale_value,
            prepared_batch["noisy_latents"].shape[4] * scale_value,
        )
        logger.debug(f"Batch contents: {prepared_batch['noisy_latents'].shape} (h={height}, w={width})")
        # permute to (B, T, C, H, W)
        num_frames = prepared_batch["noisy_latents"].shape[2]

        if "conditioning_mask" in prepared_batch:
            conditioning_mask = pack_ltx_latents(prepared_batch["conditioning_mask"]).squeeze(-1)
        packed_noisy_latents = pack_ltx_latents(prepared_batch["noisy_latents"], 1, 1).to(self.config.weight_dtype)

        logger.debug(f"Packed batch shape: {packed_noisy_latents.shape}")
        logger.debug(
            "input dtypes:"
            f"\n -> noisy_latents: {prepared_batch['noisy_latents'].dtype}"
            f"\n -> encoder_hidden_states: {prepared_batch['encoder_hidden_states'].dtype}"
            f"\n -> timestep: {prepared_batch['timesteps'].dtype}"
        )
        # Copied from a-r-r-o-w's script.
        latent_frame_rate = self.config.framerate / 8
        spatial_compression_ratio = 32
        # [0.32, 32, 32]
        rope_interpolation_scale = [
            1 / latent_frame_rate,
            spatial_compression_ratio,
            spatial_compression_ratio,
        ]
        # rope_interpolation_scale = [1 / 25, 32, 32]

        hidden_states_buffer = self._new_hidden_state_buffer()
        capture_hidden = bool(getattr(self, "crepa_regularizer", None) and self.crepa_regularizer.wants_hidden_states())
        transformer_kwargs = {
            "encoder_hidden_states": prepared_batch["encoder_hidden_states"],
            "encoder_attention_mask": prepared_batch["encoder_attention_mask"],
            "timestep": prepared_batch["timesteps"],
            "timestep_sign": (
                prepared_batch.get("twinflow_time_sign") if getattr(self.config, "twinflow_enabled", False) else None
            ),
            "return_dict": False,
            "num_frames": num_frames,
            "rope_interpolation_scale": rope_interpolation_scale,
            "height": height,
            "width": width,
        }
        if capture_hidden:
            transformer_kwargs["output_hidden_states"] = True
            transformer_kwargs["hidden_state_layer"] = self.crepa_regularizer.block_index
        if hidden_states_buffer is not None:
            transformer_kwargs["hidden_states_buffer"] = hidden_states_buffer

        model_output = self.model(
            packed_noisy_latents,
            **transformer_kwargs,
        )
        if capture_hidden:
            if isinstance(model_output, tuple) and len(model_output) >= 2:
                model_pred, crepa_hidden = model_output[0], model_output[1]
            else:
                model_pred = model_output[0] if isinstance(model_output, tuple) else model_output
                crepa_hidden = None
            if crepa_hidden is None and not getattr(self.crepa_regularizer, "use_backbone_features", False):
                raise ValueError(
                    f"CREPA requested hidden states from layer {self.crepa_regularizer.block_index} "
                    "but none were returned. Check that crepa_block_index is within the model's block count."
                )
        else:
            model_pred = model_output[0] if isinstance(model_output, tuple) else model_output
            crepa_hidden = None
        logger.debug(f"Got to the end of prediction, {model_pred.shape}")
        # we need to unpack LTX video latents i think
        model_pred = unpack_ltx_latents(
            model_pred,
            num_frames=num_frames,
            patch_size=1,
            patch_size_t=1,
            height=height,
            width=width,
        )

        return {
            "model_prediction": model_pred,
            "crepa_hidden_states": crepa_hidden,
            "hidden_states_buffer": hidden_states_buffer,
        }

    def check_user_config(self):
        """
        Checks self.config values against important issues.
        """
        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        if self.config.aspect_bucket_alignment != 64:
            logger.warning(
                f"{self.NAME} requires an alignment value of 64px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 64

        if self.config.prediction_type is not None:
            logger.warning(f"{self.NAME} does not support prediction type {self.config.prediction_type}.")

        if self.config.tokenizer_max_length is not None:
            logger.warning(f"-!- {self.NAME} supports a max length of 226 tokens, --tokenizer_max_length is ignored -!-")
        self.config.tokenizer_max_length = 226
        if self.config.validation_num_inference_steps > 50:
            logger.warning(
                f"{self.NAME} {self.config.model_flavour} may be wasting compute with more than 50 steps. Consider reducing the value to save time."
            )
        if self.config.validation_num_inference_steps < 40:
            logger.warning(
                f"{self.NAME} {self.config.model_flavour or self.DEFAULT_MODEL_FLAVOUR} expects around 40 or more inference steps. Consider increasing --validation_num_inference_steps to 40."
            )
        if not self.config.validation_disable_unconditional:
            logger.info("Disabling unconditional validation to save on time.")
            self.config.validation_disable_unconditional = True

        if self.config.framerate is None:
            self.config.framerate = 25

        # self.config.vae_enable_tiling = True
        # self.config.vae_enable_slicing = True

    def pretrained_load_args(self, pretrained_load_args: dict) -> dict:
        args = super().pretrained_load_args(pretrained_load_args)
        return apply_musubi_pretrained_defaults(self.config, args)

    def custom_model_card_schedule_info(self):
        output_args = []
        if self.config.flow_schedule_auto_shift:
            output_args.append("flow_schedule_auto_shift")
        if self.config.flow_schedule_shift is not None:
            output_args.append(f"shift={self.config.flow_schedule_shift}")
        if self.config.flow_use_beta_schedule:
            output_args.append(f"flow_beta_schedule_alpha={self.config.flow_beta_schedule_alpha}")
            output_args.append(f"flow_beta_schedule_beta={self.config.flow_beta_schedule_beta}")
        if self.config.t5_padding != "unmodified":
            output_args.append(f"t5_padding={self.config.t5_padding}")
        output_str = f" (extra parameters={output_args})" if output_args else " (no special parameters set)"

        return output_str


from simpletuner.helpers.models.registry import ModelRegistry

ModelRegistry.register("ltxvideo", LTXVideo)
