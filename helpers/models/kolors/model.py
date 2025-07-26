import torch, os, logging
from helpers.models.common import (
    ImageModelFoundation,
    PredictionTypes,
    PipelineTypes,
    ModelTypes,
)
from diffusers.pipelines.kolors.text_encoder import ChatGLMModel
from diffusers.pipelines.kolors.tokenizer import ChatGLMTokenizer
from helpers.models.kolors.pipeline import KolorsPipeline, KolorsImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel

logger = logging.getLogger(__name__)
from helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class Kolors(ImageModelFoundation):
    NAME = "Kwai Kolors"
    PREDICTION_TYPE = PredictionTypes.EPSILON
    MODEL_TYPE = ModelTypes.UNET
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 4
    DEFAULT_NOISE_SCHEDULER = "euler"
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    # Only training the Attention blocks by default seems to help more with SD3.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = UNet2DConditionModel
    MODEL_SUBFOLDER = "unet"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: KolorsPipeline,
        PipelineTypes.IMG2IMG: KolorsImg2ImgPipeline,
    }

    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "1.0"
    HUGGINGFACE_PATHS = {
        "1.0": "terminusresearch/kwai-kolors-1.0",
    }
    MODEL_LICENSE = "apache-2.0"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "ChatGLM-6B",
            "model": ChatGLMModel,
            "tokenizer_subfolder": "tokenizer",
            "tokenizer": ChatGLMTokenizer,
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
                            f"The following part of your input was truncated because ChatGLM can only handle sequences up to {tokenizer.model_max_length} tokens: {removed_text}"
                        )
                # we pass the attention mask into the text encoder. it transforms the embeds but does not attend to them.
                # unfortunately, kolors does not return the attention mask for later use by the U-net to avoid attending to the padding tokens.
                prompt_embeds_output = text_encoder(
                    input_ids=text_inputs["input_ids"].to(self.accelerator.device),
                    attention_mask=text_inputs["attention_mask"].to(
                        self.accelerator.device
                    ),
                    position_ids=text_inputs["position_ids"],
                    output_hidden_states=True,
                )
                # the ChatGLM encoder output is hereby mangled in fancy ways for Kolors to be useful.
                prompt_embeds = (
                    prompt_embeds_output.hidden_states[-2].permute(1, 0, 2).clone()
                )
                # [max_sequence_length, batch, hidden_size] -> [batch, hidden_size]
                pooled_prompt_embeds = prompt_embeds_output.hidden_states[-1][
                    -1, :, :
                ].clone()
                bs_embed, seq_len, _ = prompt_embeds.shape
                prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)

                # Clear out anything we moved to the text encoder device
                text_inputs.input_ids.to("cpu")
                del prompt_embeds_output
                del text_inputs

                prompt_embeds_list.append(prompt_embeds)
        except Exception as e:
            import traceback

            logger.error(
                f"Failed to encode prompt: {prompts}\n-> error: {e}\n-> traceback: {traceback.format_exc()}"
            )
            raise e
        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        return prompt_embeds, pooled_prompt_embeds

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

        output_str = (
            f" (extra parameters={output_args})"
            if output_args
            else " (no special parameters set)"
        )

        return output_str
