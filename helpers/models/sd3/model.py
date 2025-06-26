import torch, os, logging
from helpers.models.common import (
    ImageModelFoundation,
    PredictionTypes,
    PipelineTypes,
    ModelTypes,
)
from diffusers import AutoencoderKL, SD3ControlNetModel
from transformers import (
    T5TokenizerFast,
    T5EncoderModel,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
)
from helpers.models.sd3.transformer import SD3Transformer2DModel
from helpers.models.sd3.pipeline import (
    StableDiffusion3Pipeline,
    StableDiffusion3Img2ImgPipeline,
)
from helpers.models.sd3.controlnet import StableDiffusion3ControlNetPipeline
from diffusers import AutoencoderKL, SD3ControlNetModel

logger = logging.getLogger(__name__)
is_primary_process = True
if os.environ.get("RANK") is not None:
    if int(os.environ.get("RANK")) != 0:
        is_primary_process = False
logger.setLevel(
    os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO") if is_primary_process else "ERROR"
)


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
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 16
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
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

    def controlnet_init(self):
        """
        Initialize SD3 ControlNet model.
        """
        logger.info("Creating the SD3 controlnet..")

        if self.config.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            self.controlnet = SD3ControlNetModel.from_pretrained(
                self.config.controlnet_model_name_or_path
            )
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
            dtype=(
                self.config.base_weight_dtype
                if hasattr(self.config, "base_weight_dtype")
                else self.config.weight_dtype
            ),
        )
        # Log the expected input channels for debugging
        if hasattr(self.controlnet, "pos_embed_input") and hasattr(
            self.controlnet.pos_embed_input, "proj"
        ):
            in_channels = self.controlnet.pos_embed_input.proj.in_channels
            logger.info(f"ControlNet expects {in_channels} input channels")

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

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode a prompt for an SD3 model.

        Args:
            prompts: The list of prompts to encode.

        Returns:
            Text encoder output (raw)
        """
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
        logger.debug(
            "Input shapes:"
            f"\n{prepared_batch['noisy_latents'].shape}"
            f"\n{prepared_batch['timesteps'].shape}"
            f"\n{prepared_batch['encoder_hidden_states'].shape}"
            f"\n{prepared_batch['add_text_embeds'].shape}"
        )
        return {
            "model_prediction": self.model(
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
                return_dict=False,
            )[0]
        }

    def prepare_controlnet_conditioning(
        self, conditioning_latents: torch.Tensor
    ) -> torch.Tensor:
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
        if hasattr(self.controlnet, "pos_embed_input") and hasattr(
            self.controlnet.pos_embed_input, "proj"
        ):
            # Access the weight tensor shape to determine expected channels
            # Weight shape for Conv2d is [out_channels, in_channels, kernel_h, kernel_w]
            weight_shape = self.controlnet.pos_embed_input.proj.weight.shape
            expected_channels = weight_shape[1]  # in_channels is the second dimension
            actual_channels = conditioning_latents.shape[1]

            if expected_channels != actual_channels:
                if expected_channels == 17 and actual_channels == 16:
                    # SD3 ControlNet was initialized with 1 extra conditioning channel
                    # Add a zero channel or a specific control signal
                    batch_size, _, height, width = conditioning_latents.shape

                    # You can customize this to add specific control information
                    # For example: depth maps, edge maps, segmentation masks, etc.
                    extra_channel = torch.zeros(
                        batch_size,
                        1,
                        height,
                        width,
                        device=conditioning_latents.device,
                        dtype=conditioning_latents.dtype,
                    )

                    # If you have specific control data, you can add it here:
                    # extra_channel = your_control_data.unsqueeze(1)  # shape: [batch, 1, H, W]

                    conditioning_latents = torch.cat(
                        [conditioning_latents, extra_channel], dim=1
                    )
                    logger.debug(
                        f"Added extra conditioning channel, new shape: {conditioning_latents.shape}"
                    )

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
        control_block_samples = [
            sample.to(dtype=self.config.base_weight_dtype)
            for sample in control_block_samples
        ]
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
            block_controlnet_hidden_states=control_block_samples,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        return {"model_prediction": model_pred}

    def get_lora_target_layers(self):
        """
        Get the target layers for LoRA training based on configuration.
        """
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
            raise NotImplementedError(
                f"Unknown LoRA target type {self.config.lora_type}."
            )

    def check_user_config(self):
        """
        Checks self.config values against important issues. Optionally implemented in child class.
        """
        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        t5_max_length = 154
        if self.config.tokenizer_max_length is None:
            self.config.tokenizer_max_length = t5_max_length
        if int(self.config.tokenizer_max_length) > t5_max_length:
            if not self.config.i_know_what_i_am_doing:
                logger.warning(
                    f"Updating T5 XXL tokeniser max length to {t5_max_length} for {self.NAME}."
                )
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
            logger.warning(
                "MM-DiT requires an alignment value of 64px. Overriding the value of --aspect_bucket_alignment."
            )
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
