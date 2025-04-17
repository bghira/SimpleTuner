import torch, os, logging
from helpers.models.common import (
    ImageModelFoundation,
    PredictionTypes,
    PipelineTypes,
    ModelTypes,
)
from transformers import T5TokenizerFast, T5EncoderModel
from helpers.models.pixart.pipeline import (
    PixArtSigmaPipeline,
)
from diffusers import AutoencoderKL, PixArtTransformer2DModel

logger = logging.getLogger(__name__)
is_primary_process = True
if os.environ.get("RANK") is not None:
    if int(os.environ.get("RANK")) != 0:
        is_primary_process = False
logger.setLevel(
    os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO") if is_primary_process else "ERROR"
)


class PixartSigma(ImageModelFoundation):
    NAME = "PixArt Sigma"
    PREDICTION_TYPE = PredictionTypes.EPSILON
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 4
    DEFAULT_NOISE_SCHEDULER = "ddim"
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    # Only training the Attention blocks by default seems to help more with SD3.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = PixArtTransformer2DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: PixArtSigmaPipeline,
        PipelineTypes.IMG2IMG: PixArtSigmaPipeline,
        # PipelineTypes.CONTROLNET: None,
    }

    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "900M-1024-v0.6"
    HUGGINGFACE_PATHS = {
        "900M-1024-v0.6": "terminusresearch/pixart-900m-1024-ft-v0.6",
        "900M-1024-v0.7-stage1": "terminusresearch/pixart-900m-1024-ft-v0.7-stage1",
        "900M-1024-v0.7-stage2": "terminusresearch/pixart-900m-1024-ft-v0.7-stage2",
        "600M-512": "PixArt-alpha/PixArt-Sigma-XL-2-512-MS",
        "600M-1024": "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        "600M-2048": "PixArt-alpha/PixArt-Sigma-XL-2-2K-MS",
    }
    MODEL_LICENSE = "openrail++"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "T5 XXL v1.1",
            "tokenizer": T5TokenizerFast,
            "subfolder": "text_encoder",
            "tokenizer_subfolder": "tokenizer",
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
        prompt_embeds, prompt_attention_mask = text_embedding

        return {
            "prompt_embeds": prompt_embeds,
            "attention_mask": prompt_attention_mask.squeeze(0),
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['prompt_attention_mask'].shape}")
        return {
            "prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            "prompt_attention_mask": text_embedding["attention_mask"].unsqueeze(0),
        }

    def convert_negative_text_embed_for_pipeline(
        self, text_embedding: torch.Tensor, prompt: str
    ) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['prompt_attention_mask'].shape}")
        return {
            "negative_prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            "negative_prompt_attention_mask": text_embedding[
                "attention_mask"
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
        prompt_embeds, prompt_attention_mask, _, _ = self.pipelines[
            PipelineTypes.TEXT2IMG
        ].encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            device=self.accelerator.device,
            max_sequence_length=min(self.config.tokenizer_max_length, 300),
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            clean_caption=True,
        )
        if self.config.t5_padding == "zero":
            # we can zero the padding tokens if we're just going to mask them later anyway.
            prompt_embeds = prompt_embeds * prompt_attention_mask.to(
                device=prompt_embeds.device
            ).unsqueeze(-1).expand(prompt_embeds.shape)

        return prompt_embeds, prompt_attention_mask

    def model_predict(self, prepared_batch):
        logger.debug(
            "Input shapes:"
            f"\n{prepared_batch['noisy_latents'].shape}"
            f"\n{prepared_batch['timesteps'].shape}"
            f"\n{prepared_batch['encoder_hidden_states'].shape}"
            f"\n{prepared_batch['encoder_attention_mask'].shape}"
        )
        if prepared_batch["noisy_latents"].shape[1] != self.LATENT_CHANNEL_COUNT:
            raise ValueError(
                f"{self.NAME} requires a latent size of {self.LATENT_CHANNEL_COUNT} channels. Ensure you are using the correct VAE cache path."
            )

        return {
            "model_prediction": self.model(
                prepared_batch["noisy_latents"].to(
                    device=self.accelerator.device,
                    dtype=self.config.base_weight_dtype,
                ),
                encoder_hidden_states=prepared_batch["encoder_hidden_states"].to(
                    device=self.accelerator.device, dtype=self.config.base_weight_dtype
                ),
                timestep=prepared_batch["timesteps"],
                encoder_attention_mask=prepared_batch["encoder_attention_mask"].to(
                    device=self.accelerator.device,
                    dtype=self.config.base_weight_dtype,
                ),
                return_dict=False,
            )[0].chunk(2, dim=1)[0]
        }

    def post_model_load_setup(self):
        """
        We'll check the current model config to ensure we're loading a base or refiner model.
        """
        if (
            "stage1" in self.config.model_flavour
            or "stage1" in self.config.pretrained_model_name_or_path
        ):
            logger.info(
                f"{self.NAME} stage1 eDiffi model is detected, enabling special training configuration settings."
            )
            self.config.refiner_training = True
            self.config.refiner_training_invert_schedule = True
        elif (
            "stage2" in self.config.model_flavour
            or "stage2" in self.config.pretrained_model_name_or_path
        ):
            logger.info(
                f"{self.NAME} stage2 eDiffi model is detected, enabling special training configuration settings."
            )
            self.config.refiner_training = True

    def check_user_config(self):
        """
        Checks self.config values against important issues. Optionally implemented in child class.
        """
        if self.config.unet_attention_slice:
            if torch.backends.mps.is_available():
                logger.warning(
                    "Using attention slicing when training {self.NAME} on MPS can result in NaN errors on the first backward pass. If you run into issues, disable this option and reduce your batch size instead to reduce memory consumption."
                )
            if self.model.get_trained_component() is not None:
                self.model.get_trained_component().set_attention_slice("auto")

        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        self.config.tokenizer_max_length = 300
        if self.config.tokenizer_max_length is not None:
            logger.warning(
                f"-!- {self.NAME} supports a max length of 300 tokens, --tokenizer_max_length is ignored -!-"
            )
        if self.config.aspect_bucket_alignment != 64:
            logger.warning(
                "{self.NAME} requires an alignment value of 64px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 64

        if (
            self.config.max_grad_norm is None
            or float(self.config.max_grad_norm) > 0.01
            and not self.config.i_know_what_i_am_doing
        ):
            logger.warning(
                f"PixArt Sigma requires --max_grad_norm=0.01. Overriding value. Set this value manually to disable this warning, or pass --i_know_what_i_am_doing to ignore it."
            )
            self.config.max_grad_norm = 0.01

        if self.config.prediction_type is not None:
            logger.info(
                f"Setting {self.NAME} prediction type: {self.config.prediction_type}"
            )
            self.PREDICTION_TYPE = PredictionTypes.from_str(self.config.prediction_type)
            if self.config.validation_noise_scheduler is None:
                self.config.validation_noise_scheduler = self.DEFAULT_NOISE_SCHEDULER

    def custom_model_card_schedule_info(self):
        output_args = []
        if self.config.snr_gamma:
            output_args.append(f"snr_gamma={self.config.snr_gamma}")
        if self.config.use_soft_min_snr:
            output_args.append(f"use_soft_min_snr")
            if self.config.soft_min_snr_sigma_data:
                output_args.append(
                    f"soft_min_snr_sigma_data={self.config.soft_min_snr_sigma_data}"
                )
        if self.config.rescale_betas_zero_snr:
            output_args.append(f"rescale_betas_zero_snr")
        if self.config.offset_noise:
            output_args.append(f"offset_noise")
            output_args.append(f"noise_offset={self.config.noise_offset}")
            output_args.append(
                f"noise_offset_probability={self.config.noise_offset_probability}"
            )
        output_args.append(
            f"training_scheduler_timestep_spacing={self.config.training_scheduler_timestep_spacing}"
        )
        output_args.append(
            f"inference_scheduler_timestep_spacing={self.config.inference_scheduler_timestep_spacing}"
        )
        output_str = (
            f" (extra parameters={output_args})"
            if output_args
            else " (no special parameters set)"
        )

        return output_str
