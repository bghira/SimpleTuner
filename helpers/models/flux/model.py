import torch, os, logging
import random
from helpers.models.common import (
    ImageModelFoundation,
    PredictionTypes,
    PipelineTypes,
    ModelTypes,
)
from transformers import (
    CLIPTokenizer,
    CLIPTextModel,
    T5TokenizerFast,
    T5EncoderModel,
)
from diffusers import AutoencoderKL
from helpers.models.flux.transformer import FluxTransformer2DModelWithMasking
from helpers.models.flux.pipeline import FluxPipeline
from helpers.training.multi_process import _get_rank
from helpers.models.flux import (
    prepare_latent_image_ids,
    pack_latents,
    unpack_latents,
)

logger = logging.getLogger(__name__)
logger.setLevel(
    os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO") if _get_rank() == 0 else "ERROR"
)


class Flux(ImageModelFoundation):
    NAME = "Flux.1"
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 16
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    # Only training the Attention blocks by default.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = FluxTransformer2DModelWithMasking
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: FluxPipeline,
        # PipelineTypes.IMG2IMG: None,
        # PipelineTypes.CONTROLNET: None,
    }

    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "dev"
    HUGGINGFACE_PATHS = {
        "dev": "black-forest-labs/flux.1-dev",
        "schnell": "black-forest-labs/flux.1-schnell",
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

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        return {
            "prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            "pooled_prompt_embeds": text_embedding["pooled_prompt_embeds"].unsqueeze(0),
            "prompt_mask": (
                text_embedding["attention_masks"].unsqueeze(0)
                if self.config.flux_attention_masked_training
                else None
            ),
        }

    def convert_negative_text_embed_for_pipeline(
        self, text_embedding: torch.Tensor, prompt: str
    ) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        if (
            self.config.validation_guidance_real is None
            or self.config.validation_guidance_real <= 1.0
        ):
            # CFG is disabled, no negative prompts.
            return {}
        return {
            "negative_prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            "negative_pooled_prompt_embeds": text_embedding[
                "pooled_prompt_embeds"
            ].unsqueeze(0),
            "negative_mask": (
                text_embedding["attention_masks"].unsqueeze(0)
                if self.config.flux_attention_masked_training
                else None
            ),
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
        prompt_embeds, pooled_prompt_embeds, time_ids, masks = self.pipelines[
            PipelineTypes.TEXT2IMG
        ].encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            device=self.accelerator.device,
            max_sequence_length=int(self.config.tokenizer_max_length),
        )
        if self.config.t5_padding == "zero":
            # we can zero the padding tokens if we're just going to mask them later anyway.
            prompt_embeds = prompt_embeds * masks.to(
                device=prompt_embeds.device
            ).unsqueeze(-1).expand(prompt_embeds.shape)

        return prompt_embeds, pooled_prompt_embeds, time_ids, masks

    def model_predict(self, prepared_batch):
        # handle guidance
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
            guidance_scales = [float(self.config.flux_guidance_value)] * prepared_batch[
                "latents"
            ].shape[0]

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
        if transformer_config is not None and getattr(
            transformer_config, "guidance_embeds", False
        ):
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
            .to(device=self.accelerator.device)
            / self.noise_schedule.config.num_train_timesteps
        )

        text_ids = torch.zeros(
            prepared_batch["prompt_embeds"].shape[1],
            3,
        ).to(
            device=self.accelerator.device,
            dtype=self.config.base_weight_dtype,
        )
        logger.debug(
            "DTypes:"
            f"\n-> Text IDs shape: {text_ids.shape if hasattr(text_ids, 'shape') else None}, dtype: {text_ids.dtype if hasattr(text_ids, 'dtype') else None}"
            f"\n-> Image IDs shape: {img_ids.shape if hasattr(img_ids, 'shape') else None}, dtype: {img_ids.dtype if hasattr(img_ids, 'dtype') else None}"
            f"\n-> Timesteps shape: {prepared_batch['timesteps'].shape if hasattr(prepared_batch['timesteps'], 'shape') else None}, dtype: {prepared_batch['timesteps'].dtype if hasattr(prepared_batch['timesteps'], 'dtype') else None}"
            f"\n-> Guidance: {guidance}"
            f"\n-> Packed Noisy Latents shape: {packed_noisy_latents.shape if hasattr(packed_noisy_latents, 'shape') else None}, dtype: {packed_noisy_latents.dtype if hasattr(packed_noisy_latents, 'dtype') else None}"
        )

        flux_transformer_kwargs = {
            "hidden_states": packed_noisy_latents,
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
            "img_ids": img_ids,
            "joint_attention_kwargs": None,
            "return_dict": False,
        }
        if self.config.flux_attention_masked_training:
            flux_transformer_kwargs["attention_mask"] = prepared_batch[
                "encoder_attention_mask"
            ]
            if flux_transformer_kwargs["attention_mask"] is None:
                raise ValueError(
                    "No attention mask was discovered when attempting validation - this means you need to recreate your text embed cache."
                )

        model_pred = self.get_trained_component()(**flux_transformer_kwargs)[0]

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
                    "Using attention slicing when training {self.NAME} on MPS can result in NaN errors on the first backward pass. If you run into issues, disable this option and reduce your batch size instead to reduce memory consumption."
                )
            if self.get_trained_component() is not None:
                self.get_trained_component().set_attention_slice("auto")

        # if self.config.base_model_precision == "fp8-quanto":
        #     raise ValueError(
        #         f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
        #     )
        if self.config.aspect_bucket_alignment != 64:
            logger.warning(
                "{self.NAME} requires an alignment value of 64px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 64

        if self.config.prediction_type is not None:
            logger.warning(
                f"{self.NAME} does not support prediction type {self.config.prediction_type}."
            )

        if self.config.tokenizer_max_length is not None:
            logger.warning(
                f"-!- {self.NAME} supports a max length of 512 tokens, --tokenizer_max_length is ignored -!-"
            )
        self.config.tokenizer_max_length = 512
        if self.config.model_flavour == "schnell":
            if (
                not self.config.flux_fast_schedule
                and not self.config.i_know_what_i_am_doing
            ):
                logger.error(
                    "Schnell requires --flux_fast_schedule (or --i_know_what_i_am_doing)."
                )
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
        if (
            self.config.model_flavour == "schnell"
            and self.config.validation_num_inference_steps > 4
        ):
            logger.warning(
                "Flux Schnell requires fewer inference steps. Consider reducing --validation_num_inference_steps to 4."
            )

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
            output_args.append(
                f"flow_beta_schedule_alpha={self.config.flow_beta_schedule_alpha}"
            )
            output_args.append(
                f"flow_beta_schedule_beta={self.config.flow_beta_schedule_beta}"
            )
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
        output_str = (
            f" (extra parameters={output_args})"
            if output_args
            else " (no special parameters set)"
        )

        return output_str
