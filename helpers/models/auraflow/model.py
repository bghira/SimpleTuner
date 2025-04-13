import torch, os, logging
from helpers.models.common import (
    ImageModelFoundation,
    PredictionTypes,
    PipelineTypes,
    ModelTypes,
)
from transformers import (
    LlamaTokenizerFast,
    UMT5EncoderModel,
)
from helpers.models.auraflow.transformer import AuraFlowTransformer2DModel
from helpers.models.auraflow.pipeline import AuraFlowPipeline
from diffusers import AutoencoderKL

logger = logging.getLogger(__name__)
is_primary_process = True
if os.environ.get("RANK") is not None:
    if int(os.environ.get("RANK")) != 0:
        is_primary_process = False
logger.setLevel(
    os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO") if is_primary_process else "ERROR"
)

class Auraflow(ImageModelFoundation):
    NAME = "Auraflow"
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 4
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    # Only training the Attention blocks by default seems to help more since this model is relatively unstable.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = AuraFlowTransformer2DModel
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: AuraFlowPipeline,
        PipelineTypes.IMG2IMG: AuraFlowPipeline,
    }
    MODEL_SUBFOLDER = "transformer"
    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "v0.3"
    HUGGINGFACE_PATHS = {
        "v0.3": "terminusresearch/auraflow-v0.3",
    }
    MODEL_LICENSE = "apache-2.0"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Pile T5",
            "tokenizer": LlamaTokenizerFast,
            "subfolder": "text_encoder",
            "tokenizer_subfolder": "tokenizer",
            "model": UMT5EncoderModel,
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
            "prompt_attention_mask": prompt_attention_mask,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        return {
            "prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            "prompt_attention_mask": text_embedding["prompt_attention_mask"].unsqueeze(0),
        }

    def convert_negative_text_embed_for_pipeline(
        self, text_embedding: torch.Tensor, prompt: str
    ) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        return {
            "negative_prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            "negative_prompt_attention_mask": text_embedding["prompt_attention_mask"].unsqueeze(0),
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode a prompt for a model.

        Args:
            prompts: The list of prompts to encode.

        Returns:
            Text encoder output (raw)
        """
        prompt_embeds, prompt_attention_mask, _, _ = self.pipelines[PipelineTypes.TEXT2IMG].encode_prompt(
            prompt=prompts,
            negative_prompt=None,
            do_classifier_free_guidance=False,
            num_images_per_prompt=1,
            device=self.accelerator.device,
            max_sequence_length=self.config.tokenizer_max_length,
        )

        return prompt_embeds, prompt_attention_mask

    def model_predict(self, prepared_batch):
        logger.debug(
            "Input shapes:"
            f"\n{prepared_batch['noisy_latents'].shape}"
            f"\n{prepared_batch['timesteps'].shape}"
            f"\n{prepared_batch['encoder_hidden_states'].shape}"
        )
        batch, channels, height, width = prepared_batch["noisy_latents"].shape
        if channels != self.LATENT_CHANNEL_COUNT:
            raise ValueError(
                f"Input latent channels must be {self.LATENT_CHANNEL_COUNT}, but got {prepared_batch['noisy_latents'].shape[1]}."
            )
        if height % self.model.config.patch_size != 0 or width % self.model.config.patch_size != 0:
            raise ValueError(
                f"Input latent height and width must be divisible by patch_size ({self.model.config.patch_size})."
                f" Got height={height}, width={width}."
            )
        
        model_output = self.model(
            prepared_batch["noisy_latents"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            encoder_hidden_states=prepared_batch["encoder_hidden_states"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            timestep=prepared_batch["timesteps"],
            return_dict=False,
        )[0]
        
        # unpatchify model_output
        height = height // self.model.config.patch_size
        width = width // self.model.config.patch_size

        model_output = model_output.reshape(
            shape=(
                model_output.shape[0],
                height,
                width,
                self.model.config.patch_size,
                self.model.config.patch_size,
                self.model.config.out_channels,
            )
        )
        model_output = torch.einsum("nhwpqc->nchpwq", model_output)
        model_output = model_output.reshape(
            shape=(
                model_output.shape[0],
                self.model.config.out_channels,
                height * self.model.config.patch_size,
                width * self.model.config.patch_size,
            )
        )

        return {
            "model_prediction": model_output
        }

    def check_user_config(self):
        """
        Checks self.config values against important issues. Optionally implemented in child class.
        """
        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        t5_max_length = 120
        if self.config.tokenizer_max_length is None or self.config.tokenizer_max_length == 0:
            logger.warning(
                f"Setting T5 XXL tokeniser max length to {t5_max_length} for {self.NAME}."
            )
            self.config.tokenizer_max_length = t5_max_length
        if int(self.config.tokenizer_max_length) > t5_max_length:
            if not self.config.i_know_what_i_am_doing:
                logger.warning(
                    f"Overriding T5 XXL tokeniser max length to {t5_max_length} for {self.NAME} because `--i_know_what_i_am_doing` has not been set."
                )
                self.config.tokenizer_max_length = t5_max_length
            else:
                logger.warning(
                    f"-!- {self.NAME} supports a max length of {t5_max_length} tokens, but you have supplied `--i_know_what_i_am_doing`, so this limit will not be enforced. -!-"
                )
                logger.warning(
                    f"The model will begin to collapse after a short period of time, if the model you are continuing from has not been tuned beyond {t5_max_length} tokens."
                )
        if self.config.aspect_bucket_alignment != 64:
            logger.warning(
                "MM-DiT requires an alignment value of 64px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 64


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
        if self.config.flow_use_uniform_schedule:
            output_args.append(f"flow_use_uniform_schedule")
        output_str = (
            f" (extra parameters={output_args})"
            if output_args
            else " (no special parameters set)"
        )

        return output_str
