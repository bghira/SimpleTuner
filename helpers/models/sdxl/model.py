import torch, os, logging
from helpers.models.common import (
    ImageModelFoundation,
    PredictionTypes,
    PipelineTypes,
    ModelTypes,
)
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPTextModel
from helpers.models.sdxl.pipeline import (
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
)
from diffusers import AutoencoderKL, UNet2DConditionModel
from helpers.models.sdxl.controlnet import ControlNetModel
from helpers.training.multi_process import _get_rank

logger = logging.getLogger(__name__)
from helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class SDXL(ImageModelFoundation):
    NAME = "Stable Diffusion XL"
    PREDICTION_TYPE = PredictionTypes.EPSILON
    MODEL_TYPE = ModelTypes.UNET
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 4
    DEFAULT_NOISE_SCHEDULER = "ddim"
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    # Only training the Attention blocks by default seems to help more with SD3.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = UNet2DConditionModel
    MODEL_SUBFOLDER = "unet"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: StableDiffusionXLPipeline,
        PipelineTypes.IMG2IMG: StableDiffusionXLImg2ImgPipeline,
        PipelineTypes.CONTROLNET: StableDiffusionXLControlNetPipeline,
    }

    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "base-1.0"
    HUGGINGFACE_PATHS = {
        "base-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
        "refiner-1.0": "stabilityai/stable-diffusion-xl-refiner-1.0",
        "base-0.9": "stabilityai/stable-diffusion-xl-base-0.9",
        "refiner-0.9": "stabilityai/stable-diffusion-xl-refiner-0.9",
    }
    MODEL_LICENSE = "creativeml-openrail-m"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "CLIP-L/14",
            "tokenizer": CLIPTokenizer,
            "tokenizer_subfolder": "tokenizer",
            "model": CLIPTextModel,
        },
        "text_encoder_2": {
            "name": "CLIP-G/14",
            "tokenizer": CLIPTokenizer,
            "subfolder": "text_encoder_2",
            "tokenizer_subfolder": "tokenizer_2",
            "model": CLIPTextModelWithProjection,
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
        prompt_embeds, pooled_prompt_embeds = text_embedding

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds.squeeze(0),
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        return {
            "prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            "pooled_prompt_embeds": text_embedding["pooled_prompt_embeds"].unsqueeze(0),
        }

    def convert_negative_text_embed_for_pipeline(
        self, text_embedding: torch.Tensor, prompt: str
    ) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        return {
            "negative_prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            "negative_pooled_prompt_embeds": text_embedding[
                "pooled_prompt_embeds"
            ].unsqueeze(0),
        }

    # Adapted from pipelines.StableDiffusionXLPipeline.encode_sdxl_prompt
    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode a prompt.

        Args:
            prompts: The list of prompts to encode.

        Returns:
            Text encoder output (raw)
        """
        prompt_embeds_list = []
        pooled_prompt_embeds_list = []
        emitted_warning = False
        try:
            for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
                if tokenizer is None or text_encoder is None:
                    # Refiner only has one text encoder and tokenizer
                    continue
                if type(prompts) is not str and type(prompts) is not list:
                    prompts = str(prompts)
                max_seq_len = 77
                text_inputs = tokenizer(
                    prompts,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_seq_len,
                )
                untruncated_ids = tokenizer(
                    prompts,
                    padding="longest",
                    return_tensors="pt",
                    max_length=max_seq_len,
                ).input_ids

                if untruncated_ids.shape[
                    -1
                ] > tokenizer.model_max_length and not torch.equal(
                    text_inputs.input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.batch_decode(
                        untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                    )
                    if not emitted_warning:
                        # Only print this once. It's a bit spammy otherwise.
                        emitted_warning = True
                        logger.warning(
                            f"The following part of your input was truncated because CLIP can only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}"
                        )
                prompt_embeds_output = text_encoder(
                    text_inputs.input_ids.to(self.accelerator.device),
                    output_hidden_states=True,
                )
                # We are always interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds_output[0]
                prompt_embeds = prompt_embeds_output.hidden_states[-2]
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)

                # Clear out anything we moved to the text encoder device
                text_inputs.input_ids.to("cpu")
                del prompt_embeds_output
                del text_inputs

                prompt_embeds_list.append(prompt_embeds)
                pooled_prompt_embeds_list.append(pooled_prompt_embeds)
        except Exception as e:
            import traceback

            logger.error(
                f"Failed to encode prompt: {prompts}\n-> error: {e}\n-> traceback: {traceback.format_exc()}"
            )
            raise e

        # pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=-1)
        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    def controlnet_init(self):
        logger.info("Creating the controlnet..")
        if self.config.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            self.controlnet = ControlNetModel.from_pretrained(
                self.config.controlnet_model_name_or_path
            )
        else:
            logger.info("Initializing controlnet weights from base model")
            self.controlnet = ControlNetModel.from_unet(self.unwrap_model(self.model))
        self.controlnet.to(self.accelerator.device, self.config.weight_dtype)

    def controlnet_predict(self, prepared_batch: dict) -> dict:
        # ControlNet conditioning.
        controlnet_image = prepared_batch["conditioning_pixel_values"].to(
            device=self.accelerator.device, dtype=self.config.weight_dtype
        )
        logger.debug(f"Image shape: {controlnet_image.shape}")
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            prepared_batch["noisy_latents"].to(
                device=self.accelerator.device, dtype=self.config.base_weight_dtype
            ),
            prepared_batch["timesteps"],
            encoder_hidden_states=prepared_batch["encoder_hidden_states"].to(
                device=self.accelerator.device, dtype=self.config.base_weight_dtype
            ),
            added_cond_kwargs=prepared_batch["added_cond_kwargs"],
            controlnet_cond=controlnet_image,
            return_dict=False,
        )

        return {
            "model_prediction": self.model(
                prepared_batch["noisy_latents"].to(
                    device=self.accelerator.device,
                    dtype=self.config.base_weight_dtype,
                ),
                prepared_batch["timesteps"].to(self.accelerator.device),
                encoder_hidden_states=prepared_batch["encoder_hidden_states"].to(
                    device=self.accelerator.device,
                    dtype=self.config.base_weight_dtype,
                ),
                added_cond_kwargs=prepared_batch["added_cond_kwargs"],
                down_block_additional_residuals=[
                    sample.to(
                        device=self.accelerator.device, dtype=self.config.weight_dtype
                    )
                    for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(
                    device=self.accelerator.device, dtype=self.config.weight_dtype
                ),
                return_dict=False,
            )[0]
        }

    def model_predict(self, prepared_batch):
        logger.debug(
            "Input shapes:"
            f"\n{prepared_batch['noisy_latents'].shape}"
            f"\n{prepared_batch['timesteps'].shape}"
            f"\n{prepared_batch['encoder_hidden_states'].shape}"
            f"\n{prepared_batch['add_text_embeds'].shape}"
            f"\n{prepared_batch['added_cond_kwargs']['text_embeds'].shape}"
        )
        return {
            "model_prediction": self.model(
                prepared_batch["noisy_latents"].to(
                    device=self.accelerator.device,
                    dtype=self.config.base_weight_dtype,
                ),
                prepared_batch["timesteps"],
                prepared_batch["encoder_hidden_states"].to(
                    device=self.accelerator.device,
                    dtype=self.config.base_weight_dtype,
                ),
                prepared_batch["add_text_embeds"].to(
                    device=self.accelerator.device,
                    dtype=self.config.weight_dtype,
                ),
                added_cond_kwargs=prepared_batch["added_cond_kwargs"],
                return_dict=False,
            )[0]
        }

    def post_model_load_setup(self):
        """
        We'll check the current model config to ensure we're loading a base or refiner model.
        """
        if self.model.config.cross_attention_dim == 1280:
            logger.info(
                f"{self.NAME} Refiner model is detected, enabling refiner training configuration settings."
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
        if self.config.tokenizer_max_length is not None:
            logger.warning(
                f"-!- {self.NAME} supports a max length of 77 tokens, --tokenizer_max_length is ignored -!-"
            )
        if self.config.aspect_bucket_alignment != 64:
            logger.warning(
                "{self.NAME} requires an alignment value of 64px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 64

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
