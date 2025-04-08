import torch, os, logging
import random
from helpers.models.common import (
    VideoModelFoundation,
    PredictionTypes,
    PipelineTypes,
    ModelTypes,
)
from transformers import (
    T5TokenizerFast,
    T5EncoderModel,
)
from diffusers import AutoencoderKLLTXVideo
from diffusers import LTXVideoTransformer3DModel
from diffusers.pipelines import LTXPipeline
from helpers.training.multi_process import _get_rank
from helpers.models.ltxvideo import (
    pack_ltx_latents,
    unpack_ltx_latents,
    apply_first_frame_protection,
    make_i2v_conditioning_mask,
)

logger = logging.getLogger(__name__)
logger.setLevel(
    os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO") if _get_rank() == 0 else "ERROR"
)


class LTXVideo(VideoModelFoundation):
    NAME = "LTXVideo"
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
        "0.9.1": "Lightricks/LTX-Video-0.9.1",
        "0.9.0": "Lightricks/LTX-Video",
    }
    MODEL_LICENSE = "apache-2.0"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "T5 XXL v1.1",
            "tokenizer": T5TokenizerFast,
            "subfolder": "text_encoder",
            "tokenizer_subfolder": "tokenizer",
            "model": T5EncoderModel,
        },
    }

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        """
        When we're running the pipeline, we'll update the kwargs specifically for this model here.
        """
        # Wan video should max out around 81 frames for efficiency.
        pipeline_kwargs["num_frames"] = min(
            125, self.config.validation_num_video_frames or 125
        )
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
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        return {
            "prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            "prompt_attention_mask": text_embedding["attention_masks"].unsqueeze(0),
        }

    def convert_negative_text_embed_for_pipeline(
        self, text_embedding: torch.Tensor, prompt: str
    ) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        return {
            "negative_prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            "negative_prompt_attention_mask": text_embedding[
                "attention_masks"
            ].unsqueeze(0),
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode a prompt.

        Args:
            prompts: The list of prompts to encode.

        Returns:
            Text encoder output (raw)
        """
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = self.pipeline.encode_prompt(
            prompt=prompts,
            device=self.accelerator.device,
        )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def model_predict(self, prepared_batch):
        if prepared_batch["noisy_latents"].shape[1] != 128:
            raise ValueError(
                "LTX Video requires a latent size of 128 channels. Ensure you are using the correct VAE cache path."
                f" Shape received: {prepared_batch['noisy_latents'].shape}"
            )
        scale_value = 1
        height, width = (
            prepared_batch["noisy_latents"].shape[3] * scale_value,
            prepared_batch["noisy_latents"].shape[4] * scale_value,
        )
        logger.debug(
            f"Batch contents: {prepared_batch['noisy_latents'].shape} (h={height}, w={width})"
        )
        # permute to (B, T, C, H, W)
        num_frames = prepared_batch["noisy_latents"].shape[2]

        if "conditioning_mask" in prepared_batch:
            conditioning_mask = pack_ltx_latents(
                prepared_batch["conditioning_mask"]
            ).squeeze(-1)
        packed_noisy_latents = pack_ltx_latents(
            prepared_batch["noisy_latents"], 1, 1
        ).to(self.config.weight_dtype)

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

        model_pred = self.transformer(
            packed_noisy_latents,
            encoder_hidden_states=prepared_batch["encoder_hidden_states"],
            encoder_attention_mask=prepared_batch["encoder_attention_mask"],
            timestep=prepared_batch["timesteps"],
            return_dict=False,
            num_frames=num_frames,
            rope_interpolation_scale=rope_interpolation_scale,
            height=height,
            width=width,
        )[0]
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
        }

    def check_user_config(self):
        """
        Checks self.config values against important issues.
        """
        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        if self.config.aspect_bucket_alignment != 32:
            logger.warning(
                "{self.NAME} requires an alignment value of 32px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 32

        if self.config.prediction_type is not None:
            logger.warning(
                f"{self.NAME} does not support prediction type {self.config.prediction_type}."
            )

        if self.config.tokenizer_max_length is not None:
            logger.warning(
                f"-!- {self.NAME} supports a max length of 226 tokens, --tokenizer_max_length is ignored -!-"
            )
        self.config.tokenizer_max_length = 226
        if self.config.validation_num_inference_steps > 50:
            logger.warning(
                f"{self.NAME} {self.config.model_flavour} may be wasting compute with more than 50 steps. Consider reducing the value to save time."
            )
        if self.config.validation_num_inference_steps < 40:
            logger.warning(
                f"{self.NAME} {self.config.model_flavour} expects around 40 or more inference steps. Consider increasing --validation_num_inference_steps to 40."
            )
        if not self.config.validation_disable_unconditional:
            logger.info("Disabling unconditional validation to save on time.")
            self.config.validation_disable_unconditional = True

        if self.config.framerate is None:
            self.config.framerate = 25

        self.config.vae_enable_tiling = True
        self.config.vae_enable_slicing = True

    def custom_model_card_schedule_info(self):
        output_args = []
        if self.config.flow_schedule_auto_shift:
            output_args.append("flow_schedule_auto_shift")
        if self.config.flow_schedule_shift is not None:
            output_args.append(f"shift={self.config.flow_schedule_shift}")
        if self.config.flow_use_beta_schedule:
            output_args.append(
                f"flow_beta_schedule_alpha={self.config.flow_beta_schedule_alpha}"
            )
            output_args.append(
                f"flow_beta_schedule_beta={self.config.flow_beta_schedule_beta}"
            )
        if self.config.t5_padding != "unmodified":
            output_args.append(f"t5_padding={self.config.t5_padding}")
        output_str = (
            f" (extra parameters={output_args})"
            if output_args
            else " (no special parameters set)"
        )

        return output_str
