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
from helpers.models.flux.transformer import FluxTransformer2DModel
from helpers.models.flux.pipeline import FluxPipeline, FluxKontextPipeline
from helpers.models.flux.pipeline_controlnet import (
    FluxControlNetPipeline,
    FluxControlPipeline,
)

from helpers.training.multi_process import _get_rank
from helpers.models.flux import (
    prepare_latent_image_ids,
    pack_latents,
    unpack_latents,
    build_kontext_inputs,
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

    MODEL_CLASS = FluxTransformer2DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: FluxPipeline,
        # PipelineTypes.IMG2IMG: None,
        PipelineTypes.CONTROLNET: FluxControlNetPipeline,
        PipelineTypes.CONTROL: FluxControlPipeline,
    }

    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "dev"
    HUGGINGFACE_PATHS = {
        "dev": "black-forest-labs/flux.1-dev",
        "schnell": "black-forest-labs/flux.1-schnell",
        "kontext": "black-forest-labs/flux.1-kontext-dev",
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

    def control_init(self):
        """
        Initialize Flux Control parameters.
        """
        if (
            self.config.control
            and self.config.pretrained_transformer_model_name_or_path is None
        ):
            with torch.no_grad():
                initial_input_channels = self.get_trained_component().config.in_channels
                # new linear layer for x_embedder
                new_linear = torch.nn.Linear(
                    self.get_trained_component().x_embedder.in_features * 2,
                    self.get_trained_component().x_embedder.out_features,
                    bias=self.get_trained_component().x_embedder.bias is not None,
                    dtype=self.get_trained_component().dtype,
                    device=self.get_trained_component().device,
                )
                new_linear.weight.zero_()
                new_linear.weight[:, :initial_input_channels].copy_(
                    self.get_trained_component().x_embedder.weight
                )
                if self.get_trained_component().x_embedder.bias is not None:
                    new_linear.bias.copy_(self.get_trained_component().x_embedder.bias)
                self.get_trained_component().x_embedder = new_linear
                # new projection layer for pos_embed
                new_proj = torch.nn.Conv2d(
                    in_channels=self.get_trained_component().pos_embed.proj.in_channels
                    * 2,
                    out_channels=self.get_trained_component().pos_embed.proj.out_channels,
                    kernel_size=self.get_trained_component().pos_embed.proj.kernel_size,
                    stride=self.get_trained_component().pos_embed.proj.stride,
                    bias=self.get_trained_component().pos_embed.proj.bias is not None,
                )
                new_proj.weight.zero_()
                new_proj.weight[:, :initial_input_channels].copy_(
                    self.get_trained_component().pos_embed.proj.weight
                )
                if self.get_trained_component().pos_embed.proj.bias is not None:
                    new_proj.bias.copy_(
                        self.get_trained_component().pos_embed.proj.bias
                    )
                self.get_trained_component().pos_embed.proj = new_proj
                self.get_trained_component().register_to_config(
                    in_channels=initial_input_channels * 2,
                    out_channels=initial_input_channels,
                )

            assert torch.all(
                self.get_trained_component()
                .x_embedder.weight[:, initial_input_channels:]
                .data
                == 0
            )
            assert torch.all(
                self.get_trained_component()
                .pos_embed.proj.weight[:, initial_input_channels:]
                .data
                == 0
            )

    def controlnet_init(self):
        logger.info("Creating the controlnet..")
        from diffusers import FluxControlNetModel

        if self.config.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            self.controlnet = FluxControlNetModel.from_pretrained(
                self.config.controlnet_model_name_or_path
            )
        else:
            logger.info("Initializing controlnet weights from base model")
            self.controlnet = FluxControlNetModel.from_transformer(
                self.unwrap_model(self.model)
            )
        self.controlnet.to(self.accelerator.device, self.config.weight_dtype)

    def requires_conditioning_latents(self) -> bool:
        # Flux ControlNet requires latent inputs instead of pixels.
        if self.config.controlnet or self.config.control:
            return True
        return False

    def requires_conditioning_validation_inputs(self) -> bool:
        # Whether this model / flavour requires conditioning inputs during validation.
        if self.config.controlnet or self.config.control:
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
        prompt_embeds, pooled_prompt_embeds, time_ids, masks = text_embedding

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds.squeeze(0),
            "time_ids": time_ids,
            "attention_masks": masks,
        }

    def update_pipeline_call_kwargs(self, kwargs: dict) -> dict:
        """
        Let the base class copy the dict unchanged, then patch in
        Kontext-specific keys if we're running a Kontext checkpoint.
        """
        if self.config.model_flavour == "kontext":
            # 1) rename the placeholder key coming from Validation.validate_prompt
            if "image" in kwargs and "conditioning_image" not in kwargs:
                kwargs["conditioning_image"] = kwargs.pop("image")

            # 2) if the caller didn’t specify a range, run the ref image
            #    for the *entire* denoise (default behaviour in training)
            if "cond_start_step" not in kwargs:
                kwargs["cond_start_step"] = 0
            if "cond_end_step" not in kwargs:
                # `num_inference_steps` is already inside kwargs
                kwargs["cond_end_step"] = kwargs.get("num_inference_steps", 28)

        return kwargs

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        # Only unsqueeze if it's missing the batch dimension
        attention_mask = text_embedding.get("attention_masks", None)
        if attention_mask.dim() == 1:  # Shape: [512]
            attention_mask = attention_mask.unsqueeze(0)  # Shape: [1, 512]
        return {
            "prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            "pooled_prompt_embeds": text_embedding["pooled_prompt_embeds"].unsqueeze(0),
            "prompt_mask": (
                attention_mask if self.config.flux_attention_masked_training else None
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
        # Only unsqueeze if it's missing the batch dimension
        attention_mask = text_embedding.get("attention_masks", None)
        if attention_mask.dim() == 1:  # Shape: [512]
            attention_mask = attention_mask.unsqueeze(0)  # Shape: [1, 512]
        return {
            "negative_prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            "negative_pooled_prompt_embeds": text_embedding[
                "pooled_prompt_embeds"
            ].unsqueeze(0),
            "negative_mask": (
                attention_mask if self.config.flux_attention_masked_training else None
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

    def prepare_batch_conditions(self, batch: dict, state: dict):
        """
        If collate gave us `conditioning_latents`, turn them into packed
        sequence + ids that model_predict expects.
        """
        cond = batch.get("conditioning_latents")
        if cond is None:
            logger.debug(f"No conditioning latents found :(")
            return batch  # nothing to do

        packed_cond, cond_ids = build_kontext_inputs(
            cond,
            dtype=self.config.weight_dtype,
            device=self.accelerator.device,
            latent_channels=self.LATENT_CHANNEL_COUNT,
        )

        batch["conditioning_packed_latents"] = packed_cond
        batch["conditioning_ids"] = cond_ids

        return batch

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

        if img_ids.dim() == 2:  # (S, 3)  -> (1, S, 3) -> (B, S, 3)
            img_ids = img_ids.unsqueeze(0).expand(
                prepared_batch["latents"].shape[0], -1, -1
            )

        # pull optional kontext inputs
        cond_seq = prepared_batch.get("conditioning_packed_latents")
        cond_ids = prepared_batch.get("conditioning_ids")

        use_cond = cond_seq is not None
        logger.debug(f"Using conditioning: {use_cond}")
        lat_in = (
            torch.cat([packed_noisy_latents, cond_seq], dim=1)
            if use_cond
            else packed_noisy_latents
        )
        id_in = torch.cat([img_ids, cond_ids], dim=1) if use_cond else img_ids

        flux_transformer_kwargs = {
            "hidden_states": lat_in,
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
            "img_ids": id_in,
            "joint_attention_kwargs": None,
            "return_dict": False,
        }
        if self.config.flux_attention_masked_training:
            attention_mask = prepared_batch["encoder_attention_mask"]
            if attention_mask is None:
                raise ValueError(
                    "No attention mask was discovered when attempting validation - this means you need to recreate your text embed cache."
                )
            # Squeeze out the extra dimension if present
            if attention_mask.dim() == 3 and attention_mask.size(1) == 1:
                attention_mask = attention_mask.squeeze(1)  # [B, 1, S] -> [B, S]
            flux_transformer_kwargs["attention_mask"] = attention_mask
        model_pred = self.get_trained_component()(**flux_transformer_kwargs)[0]
        # Drop the reference-image tokens before unpacking
        if use_cond and self.config.model_flavour == "kontext":
            scene_seq_len = packed_noisy_latents.shape[
                1
            ]  # tokens that belong to the main image
            model_pred = model_pred[:, :scene_seq_len, :]  # (B, S_scene, C*4)

        return {
            "model_prediction": unpack_latents(
                model_pred,
                height=prepared_batch["latents"].shape[2] * 8,
                width=prepared_batch["latents"].shape[3] * 8,
                vae_scale_factor=16,
            )
        }

    def controlnet_predict(self, prepared_batch: dict) -> dict:
        """
        Perform a forward pass with ControlNet for Flux model.

        Args:
            prepared_batch: Dictionary containing the batch data including conditioning_latents

        Returns:
            Dictionary containing the model prediction
        """
        # ControlNet conditioning - Flux uses latents instead of pixel values
        controlnet_cond = prepared_batch["conditioning_latents"].to(
            device=self.accelerator.device, dtype=self.config.weight_dtype
        )

        # Pack the conditioning latents (same as noisy latents)
        packed_controlnet_cond = pack_latents(
            controlnet_cond,
            batch_size=controlnet_cond.shape[0],
            num_channels_latents=controlnet_cond.shape[1],
            height=controlnet_cond.shape[2],
            width=controlnet_cond.shape[3],
        ).to(
            dtype=self.config.base_weight_dtype,
            device=self.accelerator.device,
        )

        # Pack noisy latents
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

        # Handle guidance
        if self.config.flux_guidance_mode == "constant":
            guidance_scales = [float(self.config.flux_guidance_value)] * prepared_batch[
                "latents"
            ].shape[0]
        elif self.config.flux_guidance_mode == "random-range":
            guidance_scales = [
                random.uniform(
                    self.config.flux_guidance_min,
                    self.config.flux_guidance_max,
                )
                for _ in range(prepared_batch["latents"].shape[0])
            ]

        # Check if guidance embeds are enabled
        transformer_config = None
        if hasattr(self.get_trained_component(base_model=True), "module"):
            transformer_config = self.get_trained_component(
                base_model=True
            ).module.config
        elif hasattr(self.get_trained_component(base_model=True), "config"):
            transformer_config = self.get_trained_component(base_model=True).config

        if transformer_config is not None and getattr(
            transformer_config, "guidance_embeds", False
        ):
            guidance = torch.tensor(guidance_scales, device=self.accelerator.device)
        else:
            guidance = None

        # Prepare image IDs
        img_ids = prepare_latent_image_ids(
            prepared_batch["latents"].shape[0],
            prepared_batch["latents"].shape[2],
            prepared_batch["latents"].shape[3],
            self.accelerator.device,
            self.config.weight_dtype,
        )

        # Prepare timesteps
        prepared_batch["timesteps"] = (
            torch.tensor(prepared_batch["timesteps"])
            .expand(prepared_batch["noisy_latents"].shape[0])
            .to(device=self.accelerator.device)
            / self.noise_schedule.config.num_train_timesteps
        )

        # Prepare text IDs
        text_ids = torch.zeros(
            prepared_batch["prompt_embeds"].shape[1],
            3,
        ).to(
            device=self.accelerator.device,
            dtype=self.config.base_weight_dtype,
        )

        # ControlNet forward pass
        controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
            hidden_states=packed_noisy_latents,
            controlnet_cond=packed_controlnet_cond,
            controlnet_mode=None,  # Set this if using ControlNet-Union
            conditioning_scale=1.0,  # You might want to make this configurable
            encoder_hidden_states=prepared_batch["prompt_embeds"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            pooled_projections=prepared_batch["add_text_embeds"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            timestep=prepared_batch["timesteps"],
            img_ids=img_ids,
            txt_ids=text_ids,
            guidance=guidance,
            joint_attention_kwargs=None,
            return_dict=False,
        )

        # Prepare kwargs for the main transformer
        flux_transformer_kwargs = {
            "hidden_states": packed_noisy_latents,
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

        # Add ControlNet outputs to kwargs
        if controlnet_block_samples is not None:
            flux_transformer_kwargs["controlnet_block_samples"] = [
                sample.to(
                    device=self.accelerator.device, dtype=self.config.weight_dtype
                )
                for sample in controlnet_block_samples
            ]

        if controlnet_single_block_samples is not None:
            flux_transformer_kwargs["controlnet_single_block_samples"] = [
                sample.to(
                    device=self.accelerator.device, dtype=self.config.weight_dtype
                )
                for sample in controlnet_single_block_samples
            ]

        # Add attention mask if using masked training
        if self.config.flux_attention_masked_training:
            flux_transformer_kwargs["attention_mask"] = prepared_batch[
                "encoder_attention_mask"
            ]
            if flux_transformer_kwargs["attention_mask"] is None:
                raise ValueError(
                    "No attention mask was discovered when attempting validation - "
                    "this means you need to recreate your text embed cache."
                )

        # Forward pass through the transformer with ControlNet residuals
        model_pred = self.get_trained_component(base_model=True)(
            **flux_transformer_kwargs
        )[0]

        # Unpack the latents back to original shape
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
        if self.config.model_flavour == "kontext" and not isinstance(
            self.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG], FluxKontextPipeline
        ):
            self.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG] = FluxKontextPipeline

    def conditioning_validation_dataset_type(self) -> bool:
        # Most conditioning inputs (ControlNet) etc require "conditioning" dataset, but Kontext requires "images".
        if self.config.model_flavour == "kontext":
            # Kontext wants the edited
            return "image"
        return "conditioning"

    def requires_conditioning_dataset(self) -> bool:
        if self.config.model_flavour == "kontext" or self.config.controlnet:
            # Any flavour of “Kontext” always expects an extra image stream
            return True
        return False

    def requires_conditioning_validation_inputs(self) -> bool:
        if self.config.model_flavour == "kontext" or self.config.controlnet:
            # Any flavour of “Kontext” always expects an extra image stream
            return True
        return False

    def requires_validation_edit_captions(self) -> bool:
        if self.config.model_flavour == "kontext" or self.config.controlnet:
            # Kontext models require edit captions to be present.
            return True
        return False

    def requires_conditioning_latents(self) -> bool:
        if self.config.model_flavour == "kontext" or self.config.controlnet:
            # Any flavour of “Kontext” needs latent inputs for its conditioning data.
            return True
        return super().requires_conditioning_latents()

    def get_lora_target_layers(self):
        # Some models, eg. Flux should override this with more complex config-driven logic.
        if self.config.model_type == "lora" and (
            self.config.controlnet or self.config.control
        ):
            if "control" not in self.config.flux_lora_target.lower():
                logger.warning(
                    "ControlNet or Control is enabled, but the LoRA target does not include 'control'. Overriding to controlnet."
                )
            self.config.flux_lora_target = "controlnet"
        if self.config.lora_type.lower() == "standard":
            if self.config.flux_lora_target == "all":
                # target_modules = mmdit layers here
                return [
                    "to_k",
                    "to_q",
                    "to_v",
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "to_out.0",
                    "to_add_out",
                ]
            elif self.config.flux_lora_target == "context":
                # i think these are the text input layers.
                return [
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "to_add_out",
                ]
            elif self.config.flux_lora_target == "context+ffs":
                # i think these are the text input layers.
                return [
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "to_add_out",
                    "ff_context.net.0.proj",
                    "ff_context.net.2",
                ]
            elif self.config.flux_lora_target == "all+ffs":
                return [
                    "to_k",
                    "to_q",
                    "to_v",
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "to_out.0",
                    "to_add_out",
                    "ff.net.0.proj",
                    "ff.net.2",
                    "ff_context.net.0.proj",
                    "ff_context.net.2",
                    "proj_mlp",
                    "proj_out",
                ]
            elif self.config.flux_lora_target == "controlnet":
                return [
                    "controlnet_x_embedder",
                    "controlnet_blocks.0",
                    "controlnet_blocks.1",
                    "controlnet_blocks.2",
                    "controlnet_blocks.3",
                    "controlnet_single_blocks.0",
                    "controlnet_single_blocks.1",
                    "controlnet_single_blocks.2",
                    "controlnet_single_blocks.3",
                    "controlnet_single_blocks.4",
                    "controlnet_single_blocks.5",
                    "controlnet_single_blocks.6",
                    "controlnet_single_blocks.7",
                    "controlnet_single_blocks.8",
                    "controlnet_single_blocks.9",
                ]
            elif self.config.flux_lora_target == "all+ffs+embedder":
                return [
                    "x_embedder",
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_out.0",
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "to_add_out",
                    "ff.net.0.proj",
                    "ff.net.2",
                    "ff_context.net.0.proj",
                    "ff_context.net.2",
                    "proj_mlp",
                    "proj_out",
                ]
            elif self.config.flux_lora_target == "ai-toolkit":
                # from ostris' ai-toolkit, possibly required to continue finetuning one.
                return [
                    "to_q",
                    "to_k",
                    "to_v",
                    "add_q_proj",
                    "add_k_proj",
                    "add_v_proj",
                    "to_out.0",
                    "to_add_out",
                    "ff.net.0.proj",
                    "ff.net.2",
                    "ff_context.net.0.proj",
                    "ff_context.net.2",
                    "norm.linear",
                    "norm1.linear",
                    "norm1_context.linear",
                    "proj_mlp",
                    "proj_out",
                ]
            elif self.config.flux_lora_target == "tiny":
                # From TheLastBen
                # https://www.reddit.com/r/StableDiffusion/comments/1f523bd/good_flux_loras_can_be_less_than_45mb_128_dim/
                return [
                    "single_transformer_blocks.7.proj_out",
                    "single_transformer_blocks.20.proj_out",
                ]
            elif self.config.flux_lora_target == "nano":
                # From TheLastBen
                # https://www.reddit.com/r/StableDiffusion/comments/1f523bd/good_flux_loras_can_be_less_than_45mb_128_dim/
                return [
                    "single_transformer_blocks.7.proj_out",
                ]

            return self.DEFAULT_LORA_TARGET
        elif self.config.lora_type.lower() == "lycoris":
            return self.DEFAULT_LYCORIS_TARGET
        else:
            raise NotImplementedError(
                f"Unknown LoRA target type {self.config.lora_type}."
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
