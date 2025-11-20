import logging
import os

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLWan
from diffusers.training_utils import compute_loss_weighting_for_sd3
from transformers import Gemma2Model, GemmaTokenizerFast

from simpletuner.helpers.models.common import (
    ModelTypes,
    PipelineTypes,
    PredictionTypes,
    VideoModelFoundation,
    get_model_config_path,
)
from simpletuner.helpers.models.sanavideo.pipeline import SanaVideoPipeline
from simpletuner.helpers.models.sanavideo.transformer import SanaVideoTransformer3DModel
from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger(__name__)

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class SanaVideo(VideoModelFoundation):
    NAME = "SanaVideo"
    MODEL_DESCRIPTION = "Sana Video generation model"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKLWan
    LATENT_CHANNEL_COUNT = 16

    DEFAULT_NOISE_SCHEDULER = "flow_matching"

    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    # Only training the Attention blocks by default.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = SanaVideoTransformer3DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: SanaVideoPipeline,
    }

    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "2b-480p"
    HUGGINGFACE_PATHS = {
        "2b-480p": "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
    }
    MODEL_LICENSE = "apache-2.0"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Gemma 2",
            "tokenizer": GemmaTokenizerFast,
            "subfolder": "text_encoder",
            "tokenizer_subfolder": "tokenizer",
            "model": Gemma2Model,
        },
    }

    COMPLEX_HUMAN_INSTRUCTION = [
        "Given a user prompt, generate an 'Enhanced prompt' that provides detailed visual descriptions suitable for video generation. Evaluate the level of detail in the user prompt:",
        "- If the prompt is simple, focus on adding specifics about colors, shapes, sizes, textures, motion, and temporal relationships to create vivid and dynamic scenes.",
        "- If the prompt is already detailed, refine and enhance the existing details slightly without overcomplicating.",
        "Here are examples of how to transform or refine prompts:",
        "- User Prompt: A cat sleeping -> Enhanced: A small, fluffy white cat slowly settling into a curled position, peacefully falling asleep on a warm sunny windowsill, with gentle sunlight filtering through surrounding pots of blooming red flowers.",
        "- User Prompt: A busy city street -> Enhanced: A bustling city street scene at dusk, featuring glowing street lamps gradually lighting up, a diverse crowd of people in colorful clothing walking past, and a double-decker bus smoothly passing by towering glass skyscrapers.",
        "Please generate only the enhanced description for the prompt below and avoid including any additional commentary or evaluations:",
        "User Prompt: ",
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.LATENT_CHANNEL_COUNT = self.config.vae_latent_channels or 16

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        """
        When we're running the pipeline, we'll update the kwargs specifically for this model here.
        """
        pipeline_kwargs["frames"] = min(81, self.config.validation_num_video_frames or 81)
        return pipeline_kwargs

    def _format_text_embedding(self, text_embedding: dict):
        """
        Models can optionally format the stored text embedding, eg. in a dict, or
        filter certain outputs from appearing in the file cache.
        """
        # Sana pipeline returns: prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask
        # We want to store them.
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = text_embedding

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": negative_prompt_attention_mask,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: dict, pipeline_type=None) -> dict:
        return {
            "prompt_embeds": text_embedding["prompt_embeds"],
            "prompt_attention_mask": text_embedding["prompt_attention_mask"],
            "negative_prompt_embeds": text_embedding.get("negative_prompt_embeds"),
            "negative_prompt_attention_mask": text_embedding.get("negative_prompt_attention_mask"),
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode a prompt.
        """
        # Handle Motion Score
        motion_score = getattr(self.config, "sana_motion_score", None)
        if motion_score is not None and not is_negative_prompt:
            # Append motion score to each prompt
            prompts = [f"{p} motion score: {motion_score}." for p in prompts]
            logger.debug(f"Appended motion score {motion_score} to prompts.")

        # Handle Complex Human Instruction (CHI)
        use_chi = getattr(self.config, "sana_complex_human_instruction", True)
        chi_prompt = self.COMPLEX_HUMAN_INSTRUCTION if use_chi else None

        # If is_negative_prompt, we might want to skip CHI?
        # Usually system prompts are for the positive prompt instructions.
        # The pipeline implementation applies CHI to the prompt passed to it.
        # If we pass negative prompts, we probably don't want CHI on them?
        # The pipeline's `encode_prompt` allows `complex_human_instruction` argument.
        # If we are encoding negative prompts, we should likely pass None for CHI.
        if is_negative_prompt:
            chi_prompt = None

        return self.pipelines[PipelineTypes.TEXT2IMG].encode_prompt(
            prompt=prompts,
            device=self.accelerator.device,
            do_classifier_free_guidance=False,  # We just want embeddings for now
            complex_human_instruction=chi_prompt,
        )

    def model_predict(self, prepared_batch):
        noisy_latents = prepared_batch["noisy_latents"]
        encoder_hidden_states = prepared_batch["encoder_hidden_states"]
        encoder_attention_mask = prepared_batch["encoder_attention_mask"]
        timesteps = prepared_batch["timesteps"]

        # noisy_latents shape: [B, C, F, H, W]
        # encoder_hidden_states: [B, Seq, Dim]

        model_pred = self.model(
            noisy_latents,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timesteps,
            return_dict=False,
        )[0]

        return {
            "model_prediction": model_pred,
        }

    def setup_training_noise_schedule(self):
        """
        Sana training uses the canonical flow-matching schedule and ignores user-provided shift overrides.
        """
        self.noise_schedule = FlowMatchEulerDiscreteScheduler.from_pretrained(
            get_model_config_path(self.config.model_family, self.config.pretrained_model_name_or_path),
            subfolder="scheduler",
        )
        # Lock Sana to the scheduler's built-in shift and distributions; ignore user overrides.
        self.config.flow_schedule_shift = None
        self.config.flow_schedule_auto_shift = False
        self.config.flow_use_beta_schedule = False
        self.config.flow_use_uniform_schedule = False
        return self.config, self.noise_schedule

    def sample_flow_sigmas(self, batch: dict, state: dict):
        """
        Sample sigmas uniformly from the scheduler tables (stateless) to mirror the reference Sana trainer.
        """
        bsz = batch["latents"].shape[0]
        num_train_timesteps = self.noise_schedule.config.num_train_timesteps
        u = torch.rand((bsz,), device=self.accelerator.device)
        indices = torch.clamp((u * num_train_timesteps).long(), max=num_train_timesteps - 1)
        scheduler_sigmas = self.noise_schedule.sigmas.to(device=self.accelerator.device, dtype=batch["latents"].dtype)
        scheduler_timesteps = self.noise_schedule.timesteps.to(device=self.accelerator.device)
        sigmas = scheduler_sigmas[indices]
        timesteps = scheduler_timesteps[indices]
        return sigmas, timesteps

    def loss(self, prepared_batch: dict, model_output, apply_conditioning_mask: bool = True):
        """
        Apply the flow-matching loss with SD3-style weighting for Sana.
        """
        if self.PREDICTION_TYPE is not PredictionTypes.FLOW_MATCHING:
            return super().loss(prepared_batch, model_output, apply_conditioning_mask)

        target = self.get_prediction_target(prepared_batch)
        model_pred = model_output["model_prediction"]
        if target is None:
            raise ValueError("Target is None. Cannot compute loss.")

        sigmas = prepared_batch.get("sigmas")
        weighting_scheme = getattr(self.config, "weighting_scheme", "none") or "none"
        if sigmas is None:
            weighting = torch.ones((model_pred.shape[0],), device=model_pred.device, dtype=model_pred.dtype)
        else:
            weighting = compute_loss_weighting_for_sd3(weighting_scheme, sigmas=sigmas.to(model_pred.device))
        weighting = weighting.view(weighting.shape[0], *([1] * (model_pred.dim() - 1)))

        loss = weighting * (model_pred.float() - target.float()) ** 2

        loss = loss.mean(dim=list(range(1, len(loss.shape)))).mean()
        return loss

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
        msg = f"SANA loaded flow matching logit-normal distribution scheduler{output_str}"
        logger.info(msg)
        return msg

    def check_user_config(self):
        """
        Checks self.config values against important issues.
        """
        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )

        # Check if sana_motion_score is set
        if not hasattr(self.config, "sana_motion_score"):
            logger.info(
                f"{self.NAME}: 'sana_motion_score' not found in config. Defaulting to None (no motion score appended)."
            )

        if not hasattr(self.config, "sana_complex_human_instruction"):
            logger.info(f"{self.NAME}: 'sana_complex_human_instruction' not found in config. Defaulting to True.")


from simpletuner.helpers.models.registry import ModelRegistry

ModelRegistry.register("sanavideo", SanaVideo)
