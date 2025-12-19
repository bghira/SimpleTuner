import logging
import os

import torch
from diffusers import AutoencoderKL
from transformers import LlamaTokenizerFast, UMT5EncoderModel

from simpletuner.helpers.models.auraflow.pipeline import AuraFlowPipeline
from simpletuner.helpers.models.auraflow.pipeline_controlnet import AuraFlowControlNetModel, AuraFlowControlNetPipeline
from simpletuner.helpers.models.auraflow.transformer import AuraFlowTransformer2DModel
from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.tae.types import ImageTAESpec

logger = logging.getLogger(__name__)
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class Auraflow(ImageModelFoundation):
    NAME = "Auraflow"
    MODEL_DESCRIPTION = "Open-source flow-based image generation"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 4
    VALIDATION_PREVIEW_SPEC = ImageTAESpec(repo_id="madebyollin/taesdxl")
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = [
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
        "to_qkv",
    ]
    SLIDER_LORA_TARGET = [
        # Exclude add_* (text) projections to stay on the visual stream.
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "to_qkv",
    ]
    # Only training the Attention blocks by default seems to help more since this model is relatively unstable.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = AuraFlowTransformer2DModel
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: AuraFlowPipeline,
        PipelineTypes.IMG2IMG: AuraFlowPipeline,
        PipelineTypes.CONTROLNET: AuraFlowControlNetPipeline,
    }
    MODEL_SUBFOLDER = "transformer"
    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "v0.3"
    HUGGINGFACE_PATHS = {
        "pony": "purplesmartai/pony-v7-base",
        "v0.3": "terminusresearch/auraflow-v0.3",
        "v0.2": "fal/AuraFlow-v0.2",
        "v0.1": "fal/AuraFlow",
    }
    MODEL_LICENSE = "apache-2.0"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Pile T5",
            "tokenizer": LlamaTokenizerFast,
            "subfolder": "text_encoder",
            "tokenizer_subfolder": "tokenizer",
            "model": UMT5EncoderModel,
            "path": "terminusresearch/auraflow-v0.3",
        },
    }

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        prompt_embeds, prompt_attention_mask = text_embedding

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # Only unsqueeze if it's missing the batch dimension
        prompt_embeds = text_embedding["prompt_embeds"]
        prompt_attention_mask = text_embedding["prompt_attention_mask"]

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:  # Shape: [seq, dim]
            prompt_embeds = prompt_embeds.unsqueeze(0)  # Shape: [1, seq, dim]
        if prompt_attention_mask.dim() == 1:  # Shape: [seq]
            prompt_attention_mask = prompt_attention_mask.unsqueeze(0)  # Shape: [1, seq]

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # Only unsqueeze if it's missing the batch dimension
        negative_prompt_embeds = text_embedding["prompt_embeds"]
        negative_prompt_attention_mask = text_embedding["prompt_attention_mask"]

        # Add batch dimension if missing
        if negative_prompt_embeds.dim() == 2:  # Shape: [seq, dim]
            negative_prompt_embeds = negative_prompt_embeds.unsqueeze(0)  # Shape: [1, seq, dim]
        if negative_prompt_attention_mask.dim() == 1:  # Shape: [seq]
            negative_prompt_attention_mask = negative_prompt_attention_mask.unsqueeze(0)  # Shape: [1, seq]

        return {
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": negative_prompt_attention_mask,
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        pipeline = self.pipelines.get(PipelineTypes.TEXT2IMG)
        if pipeline is None:
            pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        prompt_embeds, prompt_attention_mask, _, _ = pipeline.encode_prompt(
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
        hidden_states_buffer = self._new_hidden_state_buffer()
        batch, channels, height, width = prepared_batch["noisy_latents"].shape
        if channels != self.LATENT_CHANNEL_COUNT:
            raise ValueError(
                f"Input latent channels must be {self.LATENT_CHANNEL_COUNT}, but got {prepared_batch['noisy_latents'].shape[1]}."
            )
        if height % self.unwrap_model().config.patch_size != 0 or width % self.unwrap_model().config.patch_size != 0:
            raise ValueError(
                f"Input latent height and width must be divisible by patch_size ({self.unwrap_model().config.patch_size})."
                f" Got height={height}, width={width}."
            )
        timesteps = (
            prepared_batch["timesteps"].to(device=self.accelerator.device, dtype=torch.float32) / 1000.0
        )  # normalize to [0, 1]

        timestep_sign = prepared_batch.get("twinflow_time_sign") if getattr(self.config, "twinflow_enabled", False) else None
        model_output = self.model(
            prepared_batch["noisy_latents"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            encoder_hidden_states=prepared_batch["encoder_hidden_states"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            timestep=timesteps,
            timestep_sign=timestep_sign,
            return_dict=True,
            hidden_states_buffer=hidden_states_buffer,
        ).sample

        return {"model_prediction": model_output, "hidden_states_buffer": hidden_states_buffer}

    def check_user_config(self):
        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        t5_max_length = 120
        if self.config.model_flavour == "pony":
            t5_max_length = 768
        if self.config.tokenizer_max_length is None or self.config.tokenizer_max_length == 0:
            logger.warning(f"Setting T5 XXL tokeniser max length to {t5_max_length} for {self.NAME}.")
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
            logger.warning("MM-DiT requires an alignment value of 64px. Overriding the value of --aspect_bucket_alignment.")
            self.config.aspect_bucket_alignment = 64

    def control_init(self):
        """modify model in-place to support control conditioning"""
        if self.config.control and self.config.pretrained_transformer_model_name_or_path is None:
            with torch.no_grad():
                initial_input_channels = self.get_trained_component().config.in_channels

                # Modify x_embedder to accept concatenated input (original + control)
                if hasattr(self.get_trained_component(), "x_embedder"):
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

                # Modify pos_embed projection to accept concatenated input
                if hasattr(self.get_trained_component(), "pos_embed"):
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

                # Update config to reflect new input channels
                self.get_trained_component().register_to_config(
                    in_channels=initial_input_channels * 2,
                    out_channels=initial_input_channels,
                )

    def controlnet_init(self):
        logger.info("Creating the AuraFlow controlnet...")
        if self.config.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            self.controlnet = AuraFlowControlNetModel.from_pretrained(self.config.controlnet_model_name_or_path)
        else:
            logger.info("Initializing controlnet weights from base model")
            # Initialize controlnet from the transformer
            self.controlnet = AuraFlowControlNetModel.from_transformer(self.unwrap_model(self.model))

        self.controlnet.to(self.accelerator.device, self.config.weight_dtype)

        if self.config.controlnet:
            self.controlnet.train()

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

        logger.info("TREAD training is enabled for AuraFlow")

    def requires_conditioning_latents(self) -> bool:
        # auraflow controlnet uses latents not pixels
        if self.config.controlnet or self.config.control:
            return True
        return False

    def requires_conditioning_validation_inputs(self) -> bool:
        if self.config.controlnet or self.config.control:
            return True
        return False

    def controlnet_predict(self, prepared_batch: dict) -> dict:
        # controlnet uses latents
        controlnet_cond = prepared_batch.get("conditioning_latents")

        if controlnet_cond is None:
            raise ValueError("conditioning_latents must be provided for ControlNet training")

        controlnet_cond = controlnet_cond.to(device=self.accelerator.device, dtype=self.config.weight_dtype)

        if controlnet_cond.shape[1] != self.LATENT_CHANNEL_COUNT:
            raise ValueError(
                f"ControlNet conditioning latents must have {self.LATENT_CHANNEL_COUNT} channels. "
                f"Got {controlnet_cond.shape[1]} channels."
            )

        conditioning_scale = getattr(self.config, "controlnet_conditioning_scale", 1.0)

        timesteps = (
            prepared_batch["timesteps"].to(device=self.accelerator.device, dtype=torch.float32) / 1000.0
        )  # normalize to [0, 1]

        timesteps = timesteps.expand(prepared_batch["noisy_latents"].shape[0])

        controlnet_kwargs = {
            "hidden_states": prepared_batch["noisy_latents"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            "controlnet_cond": controlnet_cond,
            "encoder_hidden_states": prepared_batch["prompt_embeds"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            "timestep": timesteps,
            "conditioning_scale": conditioning_scale,
            "return_dict": False,
        }

        block_controlnet_hidden_states = self.controlnet(**controlnet_kwargs)[0]

        transformer_kwargs = {
            "hidden_states": prepared_batch["noisy_latents"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            "encoder_hidden_states": prepared_batch["prompt_embeds"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            "timestep": timesteps,
            "block_controlnet_hidden_states": block_controlnet_hidden_states,
            "return_dict": False,
        }

        model_pred = self.get_trained_component(base_model=True)(**transformer_kwargs)[0]

        return {"model_prediction": model_pred}

    def get_lora_target_layers(self):
        manual_targets = self._get_peft_lora_target_modules()
        if manual_targets:
            return manual_targets
        if getattr(self.config, "slider_lora_target", False) and self.config.lora_type.lower() == "standard":
            return getattr(self, "SLIDER_LORA_TARGET", None) or self.DEFAULT_SLIDER_LORA_TARGET
        if self.config.model_type == "lora" and (self.config.controlnet or self.config.control):
            controlnet_block_modules = [f"controlnet_blocks.{i}" for i in range(36)]
            return controlnet_block_modules

        if self.config.lora_type.lower() == "standard":
            return self.DEFAULT_LORA_TARGET
        elif self.config.lora_type.lower() == "lycoris":
            return self.DEFAULT_LYCORIS_TARGET
        else:
            raise NotImplementedError(f"Unknown LoRA target type {self.config.lora_type}.")

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

        if self.config.controlnet:
            output_args.append("controlnet_enabled")
            if hasattr(self.config, "controlnet_conditioning_scale"):
                output_args.append(f"controlnet_scale={self.config.controlnet_conditioning_scale}")

        if self.config.control:
            output_args.append("control_enabled")

        output_str = f" (extra parameters={output_args})" if output_args else " (no special parameters set)"

        return output_str


from simpletuner.helpers.models.registry import ModelRegistry

ModelRegistry.register("auraflow", Auraflow)
