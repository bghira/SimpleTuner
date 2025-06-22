import torch, os, logging, einops, inspect
from helpers.training.wrappers import (
    gather_dict_of_tensors_shapes,
    move_dict_of_tensors_to_device,
)
from helpers.models.common import (
    ImageModelFoundation,
    PredictionTypes,
    PipelineTypes,
    ModelTypes,
)
from transformers import (
    T5EncoderModel,
    AutoTokenizer,
    LlamaForCausalLM,
    CLIPTokenizer,
    CLIPTextModelWithProjection,
)

HiDreamImageTransformer2DModel = None
HiDreamImagePipeline: object = None
HiDreamControlNetPipeline: object = None
from helpers.models.hidream.transformer import (
    HiDreamImageTransformer2DModel,
    get_load_balancing_loss,
    clear_load_balancing_loss,
)
from helpers.models.hidream.pipeline import HiDreamImagePipeline
from helpers.models.hidream.controlnet import HiDreamControlNetPipeline
from diffusers import AutoencoderKL

logger = logging.getLogger(__name__)
is_primary_process = True
if os.environ.get("RANK") is not None:
    if int(os.environ.get("RANK")) != 0:
        is_primary_process = False
logger.setLevel(
    os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO") if is_primary_process else "ERROR"
)


class HiDream(ImageModelFoundation):
    NAME = "HiDream"
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 16
    DEFAULT_NOISE_SCHEDULER = "flow_unipc"
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    # Only training the Attention blocks by default seems to help more with HiDream.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = HiDreamImageTransformer2DModel
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: HiDreamImagePipeline,
        # PipelineTypes.IMG2IMG: None,
        PipelineTypes.CONTROLNET: HiDreamControlNetPipeline,
    }
    MODEL_SUBFOLDER = "transformer"
    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "full"
    HUGGINGFACE_PATHS = {
        "dev": "HiDream-ai/HiDream-I1-Dev",
        "full": "HiDream-ai/HiDream-I1-Full",
        "fast": "HiDream-ai/HiDream-I1-Fast",
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
            "tokenizer": AutoTokenizer,
            "subfolder": "text_encoder_3",
            "tokenizer_subfolder": "tokenizer_3",
            "model": T5EncoderModel,
        },
        "text_encoder_4": {
            "name": "Llama",
            "tokenizer": AutoTokenizer,
            "subfolder": "text_encoder_4",
            "tokenizer_subfolder": "tokenizer_4",
            "model": LlamaForCausalLM,
            "path": "terminusresearch/HiDream-I1-Llama-3.1-8B-Instruct",
            # "required_quantisation_level": "int4_weight_only",
        },
    }

    def controlnet_init(self):
        """Initialize HiDream ControlNet"""
        logger.info("Creating the HiDream controlnet..")
        from helpers.models.hidream.controlnet import HiDreamControlNetModel

        if self.config.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            self.controlnet = HiDreamControlNetModel.from_pretrained(
                self.config.controlnet_model_name_or_path
            )
        else:
            logger.info("Initializing controlnet weights from base model")
            cn_custom_config = self.config.controlnet_custom_config or {}
            self.controlnet = HiDreamControlNetModel.from_transformer(
                self.unwrap_model(self.model), **cn_custom_config
            )
        self.controlnet.train()

    def requires_conditioning_latents(self) -> bool:
        """HiDream ControlNet requires latent inputs instead of pixels."""
        if self.config.controlnet:
            return True
        return False

    def requires_conditioning_validation_inputs(self) -> bool:
        """Whether this model requires conditioning inputs during validation."""
        if self.config.controlnet:
            return True
        return False

    def pretrained_load_args(self, pretrained_load_args: dict) -> dict:
        if (
            self.config.hidream_load_balancing_loss_weight is not None
            and self.config.hidream_load_balancing_loss_weight > 0
        ):
            pretrained_load_args["aux_loss_alpha"] = (
                self.config.hidream_load_balancing_loss_weight
            )

        return pretrained_load_args

    def _load_pipeline(
        self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True
    ):
        """
        Loads the pipeline class for the model.
        """
        active_pipelines = getattr(self, "pipelines", {})
        if pipeline_type in active_pipelines:
            setattr(
                active_pipelines[pipeline_type],
                self.MODEL_TYPE.value,
                self.unwrap_model(model=self.model),
            )
            if self.config.controlnet:
                setattr(
                    active_pipelines[pipeline_type],
                    "controlnet",
                    self.unwrap_model(self.get_trained_component()),
                )
            return active_pipelines[pipeline_type]
        pipeline_kwargs = {
            "pretrained_model_name_or_path": self._model_config_path(),
        }
        if not hasattr(self, "PIPELINE_CLASSES"):
            raise NotImplementedError("Pipeline class not defined.")
        if pipeline_type not in self.PIPELINE_CLASSES:
            raise NotImplementedError(
                f"Pipeline type {pipeline_type} not defined in {self.__class__.__name__}."
            )
        pipeline_class = self.PIPELINE_CLASSES[pipeline_type]
        if not hasattr(pipeline_class, "from_pretrained"):
            raise NotImplementedError(
                f"Pipeline type {pipeline_type} class {pipeline_class} does not have from_pretrained method."
            )
        signature = inspect.signature(pipeline_class.from_pretrained)
        if "watermarker" in signature.parameters:
            pipeline_kwargs["watermarker"] = None
        if "watermark" in signature.parameters:
            pipeline_kwargs["watermark"] = None
        if load_base_model:
            pipeline_kwargs[self.MODEL_TYPE.value] = self.unwrap_model(model=self.model)
        else:
            pipeline_kwargs[self.MODEL_TYPE.value] = None

        if getattr(self, "vae", None) is not None:
            pipeline_kwargs["vae"] = self.unwrap_model(self.vae)
        elif getattr(self, "AUTOENCODER_CLASS", None) is not None:
            pipeline_kwargs["vae"] = self.get_vae()

        text_encoder_idx = 0
        for (
            text_encoder_attr,
            text_encoder_config,
        ) in self.TEXT_ENCODER_CONFIGURATION.items():
            tokenizer_attr = text_encoder_attr.replace("text_encoder", "tokenizer")
            if (
                self.text_encoders is not None
                and len(self.text_encoders) >= text_encoder_idx
            ):
                pipeline_kwargs[text_encoder_attr] = self.unwrap_model(
                    self.text_encoders[text_encoder_idx]
                )
                pipeline_kwargs[tokenizer_attr] = self.tokenizers[text_encoder_idx]
            else:
                pipeline_kwargs[text_encoder_attr] = None
                pipeline_kwargs[tokenizer_attr] = None
            text_encoder_idx += 1

        self.load_text_tokenizer()
        pipeline_kwargs["tokenizer_4"] = self.tokenizers[3]

        if self.config.controlnet:
            pipeline_kwargs["controlnet"] = self.unwrap_model(
                self.get_trained_component()
            )

        logger.debug(
            f"Initialising {pipeline_class.__name__} with components: {pipeline_kwargs}"
        )
        self.pipelines[pipeline_type] = pipeline_class.from_pretrained(
            **pipeline_kwargs,
        )

        return self.pipelines[pipeline_type]

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        """
        Models can optionally format the stored text embedding, eg. in a dict, or
        filter certain outputs from appearing in the file cache.

        self.config:
            text_embedding (torch.Tensor): The embed to adjust.

        Returns:
            torch.Tensor: The adjusted embed. By default, this method does nothing.
        """
        t5_embeds, llama_embeds, pooled_prompt_embeds = text_embedding

        return {
            "t5_prompt_embeds": t5_embeds.squeeze(0).detach().clone().to("cpu"),
            "llama_prompt_embeds": llama_embeds.squeeze(0).detach().clone().to("cpu"),
            "pooled_prompt_embeds": pooled_prompt_embeds.squeeze(0)
            .detach()
            .clone()
            .to("cpu"),
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        return {
            "t5_prompt_embeds": text_embedding["t5_prompt_embeds"]
            .unsqueeze(0)
            .to(self.accelerator.device),
            "llama_prompt_embeds": text_embedding["llama_prompt_embeds"]
            .unsqueeze(0)
            .to(self.accelerator.device),
            "pooled_prompt_embeds": text_embedding["pooled_prompt_embeds"]
            .unsqueeze(0)
            .to(self.accelerator.device),
        }

    def convert_negative_text_embed_for_pipeline(
        self, text_embedding: torch.Tensor, prompt: str
    ) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        return {
            "negative_t5_prompt_embeds": text_embedding["t5_prompt_embeds"]
            .unsqueeze(0)
            .to(self.accelerator.device),
            "negative_llama_prompt_embeds": text_embedding["llama_prompt_embeds"]
            .unsqueeze(0)
            .to(self.accelerator.device),
            "negative_pooled_prompt_embeds": text_embedding["pooled_prompt_embeds"]
            .unsqueeze(0)
            .to(self.accelerator.device),
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode prompts for an HiDream model into a single tensor.

        Args:
            prompts: The list of prompts to encode.

        Returns:
            Text encoder output (raw)
        """
        t5_embeds, llama_embeds, pooled_prompt_embeds = self.pipelines[
            PipelineTypes.TEXT2IMG
        ]._encode_prompt(
            prompt=prompts,
            prompt_2=prompts,
            prompt_3=prompts,
            prompt_4=prompts,
            device=self.accelerator.device,
            dtype=self.config.base_weight_dtype,
            num_images_per_prompt=1,
            max_sequence_length=self.config.tokenizer_max_length or 128,
        )
        # verbose declarations are simply for clarity.
        return t5_embeds, llama_embeds, pooled_prompt_embeds

    def collate_prompt_embeds(self, text_encoder_output: dict) -> dict:
        return move_dict_of_tensors_to_device(
            {
                "t5_prompt_embeds": torch.stack(
                    [e["t5_prompt_embeds"] for e in text_encoder_output], dim=0
                ),
                "llama_prompt_embeds": torch.stack(
                    [e["llama_prompt_embeds"] for e in text_encoder_output], dim=0
                ),
                "pooled_prompt_embeds": torch.stack(
                    [e["pooled_prompt_embeds"] for e in text_encoder_output], dim=0
                ),
            },
            self.accelerator.device,
        )

    def model_predict(self, prepared_batch):
        """
        Process a batch through the transformer model.

        Args:
            prepared_batch: Dictionary containing input tensors and embeddings

        Returns:
            Dictionary containing model predictions
        """
        logger.debug(f"Prompt embeds: {prepared_batch['text_encoder_output']}")
        logger.debug(
            "Input shapes:"
            f"\n{prepared_batch['noisy_latents'].shape}"
            f"\n{prepared_batch['timesteps'].shape}"
            f"\nT5: {prepared_batch['text_encoder_output']['t5_prompt_embeds'].shape if hasattr(prepared_batch['text_encoder_output']['t5_prompt_embeds'], 'shape') else [x.shape for x in prepared_batch['text_encoder_output']['t5_prompt_embeds']]}"
            f"\nLlama: {prepared_batch['text_encoder_output']['llama_prompt_embeds'].shape if hasattr(prepared_batch['text_encoder_output']['llama_prompt_embeds'], 'shape') else [x.shape for x in prepared_batch['text_encoder_output']['llama_prompt_embeds']]}"
            f"\nCLIP L + G: {prepared_batch['text_encoder_output']['pooled_prompt_embeds'].shape}"
        )

        # Handle non-square images
        if (
            prepared_batch["noisy_latents"].shape[-2]
            != prepared_batch["noisy_latents"].shape[-1]
        ):
            B, C, H, W = prepared_batch["noisy_latents"].shape
            pH, pW = (
                H // self.unwrap_model(model=self.model).config.patch_size,
                W // self.unwrap_model(model=self.model).config.patch_size,
            )

            img_sizes = torch.tensor([pH, pW], dtype=torch.int64).reshape(-1)
            img_ids = torch.zeros(pH, pW, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW)[None, :]
            img_ids = img_ids.reshape(pH * pW, -1)
            img_ids_pad = torch.zeros(self.unwrap_model(model=self.model).max_seq, 3)
            img_ids_pad[: pH * pW, :] = img_ids

            img_sizes = img_sizes.unsqueeze(0).to(
                prepared_batch["noisy_latents"].device
            )
            img_ids = img_ids_pad.unsqueeze(0).to(
                prepared_batch["noisy_latents"].device
            )
            img_sizes = img_sizes.repeat(B, 1)
            img_ids = img_ids.repeat(B, 1, 1)
        else:
            img_sizes = img_ids = None

        latent_model_input = prepared_batch["noisy_latents"]
        if latent_model_input.shape[-2] != latent_model_input.shape[-1]:
            B, C, H, W = latent_model_input.shape
            patch_size = self.unwrap_model(model=self.model).config.patch_size
            pH, pW = H // patch_size, W // patch_size
            out = torch.zeros(
                (
                    B,
                    C,
                    self.unwrap_model(model=self.model).max_seq,
                    patch_size * patch_size,
                ),
                dtype=latent_model_input.dtype,
                device=latent_model_input.device,
            )
            latent_model_input = einops.rearrange(
                latent_model_input,
                "B C (H p1) (W p2) -> B C (H W) (p1 p2)",
                p1=patch_size,
                p2=patch_size,
            )
            out[:, :, 0 : pH * pW] = latent_model_input
            latent_model_input = out

        # Call the forward method with the updated parameter names
        return {
            "model_prediction": self.model(
                hidden_states=latent_model_input.to(
                    device=self.accelerator.device,
                    dtype=self.config.base_weight_dtype,
                ),
                timesteps=prepared_batch["timesteps"],
                t5_hidden_states=prepared_batch["text_encoder_output"][
                    "t5_prompt_embeds"
                ],
                llama_hidden_states=prepared_batch["text_encoder_output"][
                    "llama_prompt_embeds"
                ],
                pooled_embeds=prepared_batch["text_encoder_output"][
                    "pooled_prompt_embeds"
                ],
                img_sizes=img_sizes,
                img_ids=img_ids,
                return_dict=False,
            )[0]
            * -1  # the model is trained with inverted velocity :(
        }

    def controlnet_predict(self, prepared_batch: dict) -> dict:
        """
        Perform a forward pass with ControlNet for HiDream model.

        Args:
            prepared_batch: Dictionary containing the batch data including conditioning_latents

        Returns:
            Dictionary containing the model prediction
        """
        # ControlNet conditioning - HiDream uses latents instead of pixel values
        controlnet_cond = prepared_batch["conditioning_latents"].to(
            device=self.accelerator.device, dtype=self.config.weight_dtype
        )

        # Handle non-square images for noisy latents
        if (
            prepared_batch["noisy_latents"].shape[-2]
            != prepared_batch["noisy_latents"].shape[-1]
        ):
            B, C, H, W = prepared_batch["noisy_latents"].shape
            pH, pW = (
                H // self.unwrap_model(model=self.model).config.patch_size,
                W // self.unwrap_model(model=self.model).config.patch_size,
            )

            img_sizes = torch.tensor([pH, pW], dtype=torch.int64).reshape(-1)
            img_ids = torch.zeros(pH, pW, 3)
            img_ids[..., 1] = img_ids[..., 1] + torch.arange(pH)[:, None]
            img_ids[..., 2] = img_ids[..., 2] + torch.arange(pW)[None, :]
            img_ids = img_ids.reshape(pH * pW, -1)
            img_ids_pad = torch.zeros(self.unwrap_model(model=self.model).max_seq, 3)
            img_ids_pad[: pH * pW, :] = img_ids

            img_sizes = img_sizes.unsqueeze(0).to(
                prepared_batch["noisy_latents"].device
            )
            img_ids = img_ids_pad.unsqueeze(0).to(
                prepared_batch["noisy_latents"].device
            )
            img_sizes = img_sizes.repeat(B, 1)
            img_ids = img_ids.repeat(B, 1, 1)
        else:
            img_sizes = img_ids = None

        # Prepare latent model input (for non-square handling)
        latent_model_input = prepared_batch["noisy_latents"]
        if latent_model_input.shape[-2] != latent_model_input.shape[-1]:
            B, C, H, W = latent_model_input.shape
            patch_size = self.unwrap_model(model=self.model).config.patch_size
            pH, pW = H // patch_size, W // patch_size
            out = torch.zeros(
                (
                    B,
                    C,
                    self.unwrap_model(model=self.model).max_seq,
                    patch_size * patch_size,
                ),
                dtype=latent_model_input.dtype,
                device=latent_model_input.device,
            )
            latent_model_input = einops.rearrange(
                latent_model_input,
                "B C (H p1) (W p2) -> B C (H W) (p1 p2)",
                p1=patch_size,
                p2=patch_size,
            )
            out[:, :, 0 : pH * pW] = latent_model_input
            latent_model_input = out

        # ControlNet forward pass - let the controlnet handle its own preprocessing
        # Pass the raw latents without any special preprocessing
        controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
            hidden_states=prepared_batch["noisy_latents"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            controlnet_cond=controlnet_cond.to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            timesteps=prepared_batch["timesteps"],
            t5_hidden_states=prepared_batch["text_encoder_output"]["t5_prompt_embeds"],
            llama_hidden_states=prepared_batch["text_encoder_output"][
                "llama_prompt_embeds"
            ],
            pooled_embeds=prepared_batch["text_encoder_output"]["pooled_prompt_embeds"],
            img_sizes=img_sizes,
            img_ids=img_ids,
            conditioning_scale=1.0,  # You might want to make this configurable
            return_dict=False,
        )

        # Prepare kwargs for the main transformer using the preprocessed latent_model_input
        hidream_transformer_kwargs = {
            "hidden_states": latent_model_input.to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            "timesteps": prepared_batch["timesteps"],
            "t5_hidden_states": prepared_batch["text_encoder_output"][
                "t5_prompt_embeds"
            ],
            "llama_hidden_states": prepared_batch["text_encoder_output"][
                "llama_prompt_embeds"
            ],
            "pooled_embeds": prepared_batch["text_encoder_output"][
                "pooled_prompt_embeds"
            ],
            "img_sizes": img_sizes,
            "img_ids": img_ids,
            "return_dict": False,
        }

        # Add ControlNet outputs to kwargs
        if controlnet_block_samples is not None:
            hidream_transformer_kwargs["controlnet_block_samples"] = [
                sample.to(
                    device=self.accelerator.device, dtype=self.config.weight_dtype
                )
                for sample in controlnet_block_samples
            ]

        if controlnet_single_block_samples is not None:
            hidream_transformer_kwargs["controlnet_single_block_samples"] = [
                sample.to(
                    device=self.accelerator.device, dtype=self.config.weight_dtype
                )
                for sample in controlnet_single_block_samples
            ]

        # Forward pass through the transformer with ControlNet residuals
        model_pred = self.get_trained_component(base_model=True)(
            **hidream_transformer_kwargs
        )[0]

        return {
            "model_prediction": model_pred
            * -1  # the model is trained with inverted velocity :(
        }

    def get_lora_target_layers(self):
        targets = [
            "controlnet_x_embedder",
        ]
        for i in range(len(self.unwrap_model().controlnet_blocks)):
            targets.extend(
                [
                    f"controlnet_blocks.{i}",
                    f"controlnet_single_blocks.{i}",
                ]
            )
        return targets

    def check_user_config(self):
        """
        Checks self.config values against important issues. Optionally implemented in child class.
        """
        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        t5_max_length = 128
        if (
            self.config.tokenizer_max_length is None
            or self.config.tokenizer_max_length == 0
        ):
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
        if self.config.flow_use_uniform_schedule:
            output_args.append(f"flow_use_uniform_schedule")
        output_str = (
            f" (extra parameters={output_args})"
            if output_args
            else " (no special parameters set)"
        )

        return output_str

    def auxiliary_loss(self, model_output, prepared_batch: dict, loss: torch.Tensor):
        aux_losses = get_load_balancing_loss()
        aux_log_info = {}

        if aux_losses:
            # Extract and accumulate the actual loss values (first element of each tuple)
            accumulated_aux_loss = torch.sum(
                torch.stack([aux_tuple[0] for aux_tuple in aux_losses])
            )

            # For logging purposes - gather these regardless of whether we add to main loss
            aux_log_info = {
                "total": accumulated_aux_loss.item(),
                "count": len(aux_losses),
                "mean": accumulated_aux_loss.item() / max(1, len(aux_losses)),
                # Extract statistics about expert utilization (the third element)
                "expert_usage_min": min(
                    [torch.min(aux_tuple[2]).item() for aux_tuple in aux_losses],
                    default=0,
                ),
                "expert_usage_max": max(
                    [torch.max(aux_tuple[2]).item() for aux_tuple in aux_losses],
                    default=0,
                ),
                "expert_usage_mean": sum(
                    [torch.mean(aux_tuple[2]).item() for aux_tuple in aux_losses]
                )
                / max(1, len(aux_losses)),
            }

            # Only add to the main loss if configured to do so
            if self.config.hidream_use_load_balancing_loss:
                total_loss = (
                    loss
                    + accumulated_aux_loss
                    * self.config.hidream_load_balancing_loss_weight
                )
            else:
                total_loss = loss
        else:
            total_loss = loss
            aux_log_info = {"total": 0.0, "count": 0}

        # Always clear the global list after processing to prevent memory buildup
        clear_load_balancing_loss()

        return total_loss, aux_log_info
