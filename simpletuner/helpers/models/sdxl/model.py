import copy
import logging
import os
from typing import Any, Callable, Dict

import torch
from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

from simpletuner.helpers.acceleration import (
    AccelerationBackend,
    AccelerationPreset,
    get_bitsandbytes_presets,
    get_deepspeed_presets,
    get_quanto_presets,
    get_sdnq_presets,
    get_torchao_presets,
)
from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.sdxl.controlnet import ControlNetModel
from simpletuner.helpers.models.sdxl.pipeline import (
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from simpletuner.helpers.models.tae.types import ImageTAESpec
from simpletuner.helpers.models.unet_flowmap import FlowMapUNet2DConditionModel as UNet2DConditionModel
from simpletuner.helpers.training.multi_process import _get_rank

logger = logging.getLogger(__name__)
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class SDXL(ImageModelFoundation):
    NAME = "Stable Diffusion XL"
    MODEL_DESCRIPTION = "SDXL 1.0 - high quality 1024px images"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.EPSILON
    MODEL_TYPE = ModelTypes.UNET
    ATTENTION_KWARG_NAME = "cross_attention_kwargs"
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 4
    VALIDATION_PREVIEW_SPEC = ImageTAESpec(repo_id="madebyollin/taesdxl")
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
    SUPPORTS_MULTISTAGE_VALIDATION = True
    SDXL_VALIDATION_PIPELINE_MODES = {"trained-stage", "full-pipeline"}
    SDXL_VALIDATION_PIPELINE_KEYS = {
        "stage1": "sdxl_validation_stage1",
        "stage2": "sdxl_validation_stage2",
    }

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
        # Only unsqueeze if it's missing the batch dimension
        prompt_embeds = text_embedding["prompt_embeds"]
        pooled_prompt_embeds = text_embedding["pooled_prompt_embeds"]

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:  # Shape: [seq, dim]
            prompt_embeds = prompt_embeds.unsqueeze(0)  # Shape: [1, seq, dim]
        if pooled_prompt_embeds.dim() == 1:  # Shape: [dim]
            pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0)  # Shape: [1, dim]

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # Only unsqueeze if it's missing the batch dimension
        prompt_embeds = text_embedding["prompt_embeds"]
        pooled_prompt_embeds = text_embedding["pooled_prompt_embeds"]

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:  # Shape: [seq, dim]
            prompt_embeds = prompt_embeds.unsqueeze(0)  # Shape: [1, seq, dim]
        if pooled_prompt_embeds.dim() == 1:  # Shape: [dim]
            pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0)  # Shape: [1, dim]

        return {
            "negative_prompt_embeds": prompt_embeds,
            "negative_pooled_prompt_embeds": pooled_prompt_embeds,
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

                if untruncated_ids.shape[-1] > tokenizer.model_max_length and not torch.equal(
                    text_inputs.input_ids, untruncated_ids
                ):
                    removed_text = tokenizer.decode(untruncated_ids[:, tokenizer.model_max_length - 1 : -1])
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

            logger.error(f"Failed to encode prompt: {prompts}\n-> error: {e}\n-> traceback: {traceback.format_exc()}")
            raise e

        # pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=-1)
        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    def controlnet_init(self):
        logger.info("Creating the controlnet..")
        if self.config.controlnet_model_name_or_path:
            logger.info("Loading existing controlnet weights")
            self.controlnet = ControlNetModel.from_pretrained(self.config.controlnet_model_name_or_path)
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
            prepared_batch["noisy_latents"].to(device=self.accelerator.device, dtype=self.config.base_weight_dtype),
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
                    sample.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
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

        # Check if U-REPA is enabled and we need to capture mid-block hidden states
        urepa = getattr(self, "urepa_regularizer", None)
        capture_mid_block = urepa is not None and urepa.enabled

        cross_attention_kwargs = self._build_gligen_cross_attention_kwargs(prepared_batch.get("grounding_batch"))

        urepa_hidden = None
        if capture_mid_block:
            from simpletuner.helpers.utils.hidden_state_buffer import UNetMidBlockCapture

            unwrapped_model = self.unwrap_model(self.model)
            with UNetMidBlockCapture(unwrapped_model) as capture:
                model_pred = self.model(
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
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                    **self._get_flowmap_r_timestep_forward_kwargs(prepared_batch),
                )[0]
                urepa_hidden = capture.get_captured()
        else:
            model_pred = self.model(
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
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
                **self._get_flowmap_r_timestep_forward_kwargs(prepared_batch),
            )[0]

        return {
            "model_prediction": model_pred,
            "hidden_states_buffer": None,
            "urepa_hidden_states": urepa_hidden,
        }

    def post_model_load_setup(self):
        """
        We'll check the current model config to ensure we're loading a base or refiner model.
        """
        if self.model.config.cross_attention_dim == 1280:
            logger.info(f"{self.NAME} Refiner model is detected, enabling refiner training configuration settings.")
            self.config.refiner_training = True

    def _sdxl_validation_mode(self) -> str:
        mode = getattr(self.config, "sdxl_validation_pipeline_mode", "trained-stage") or "trained-stage"
        mode = str(mode).strip().lower()
        if mode not in self.SDXL_VALIDATION_PIPELINE_MODES:
            raise ValueError(
                "sdxl_validation_pipeline_mode must be one of: " f"{', '.join(sorted(self.SDXL_VALIDATION_PIPELINE_MODES))}"
            )
        return mode

    def supports_multistage_validation(self) -> bool:
        if self._sdxl_validation_mode() != "full-pipeline":
            return False
        if getattr(self.config, "validation_using_datasets", False):
            return False
        if getattr(self.config, "controlnet", False) or getattr(self.config, "control", False):
            return False
        return True

    def validation_adapter_stage_aliases(self) -> Dict[str, set[str]]:
        return {
            "stage1": {"stage1", "stage_1", "1", "one", "base"},
            "stage2": {"stage2", "stage_2", "2", "two", "refiner"},
        }

    def _sdxl_current_stage(self) -> int:
        model_flavour = str(getattr(self.config, "model_flavour", "") or "").lower()
        model_path = str(getattr(self.config, "pretrained_model_name_or_path", "") or "").lower()
        if "refiner" in model_flavour or "refiner" in model_path:
            return 2
        if getattr(self.config, "refiner_training", False) and not getattr(
            self.config, "refiner_training_invert_schedule", False
        ):
            return 2
        return 1

    def _sdxl_version_suffix(self) -> str:
        model_flavour = str(getattr(self.config, "model_flavour", "") or "")
        model_path = str(getattr(self.config, "pretrained_model_name_or_path", "") or "")
        if "0.9" in model_flavour or "0.9" in model_path:
            return "0.9"
        return "1.0"

    def _sdxl_validation_stage_model(self, stage: int) -> str:
        if stage == 1:
            configured = getattr(self.config, "sdxl_validation_stage1_model", None)
            default = self.HUGGINGFACE_PATHS[f"base-{self._sdxl_version_suffix()}"]
        elif stage == 2:
            configured = getattr(self.config, "sdxl_validation_stage2_model", None)
            default = self.HUGGINGFACE_PATHS[f"refiner-{self._sdxl_version_suffix()}"]
        else:
            raise ValueError(f"Unsupported SDXL validation stage: {stage}")
        return str(configured or default)

    def _sdxl_from_pretrained_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "torch_dtype": self.config.weight_dtype,
        }
        vae = self.get_vae()
        if vae is not None:
            kwargs["vae"] = vae
        for attr in ("revision", "cache_dir", "local_files_only", "variant"):
            value = getattr(self.config, attr, None)
            if value is not None:
                kwargs[attr] = value
        return kwargs

    def _sdxl_clone_validation_scheduler(self):
        scheduler = getattr(getattr(self, "pipeline", None), "scheduler", None)
        if scheduler is None:
            return None
        try:
            return scheduler.__class__.from_config(scheduler.config)
        except Exception:
            return copy.deepcopy(scheduler)

    def _get_sdxl_validation_pipeline(self, cache_key: str, pipeline_cls, repo_id: str):
        if cache_key in self.pipelines:
            return self.pipelines[cache_key]

        logger.info("Loading SDXL validation pipeline '%s' from %s.", cache_key, repo_id)
        pipeline = pipeline_cls.from_pretrained(repo_id, **self._sdxl_from_pretrained_kwargs())
        for attr in ("watermarker", "watermark"):
            if hasattr(pipeline, attr):
                setattr(pipeline, attr, None)
        scheduler = self._sdxl_clone_validation_scheduler()
        if scheduler is not None:
            pipeline.scheduler = scheduler
        pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        self.pipelines[cache_key] = pipeline
        return pipeline

    def _sdxl_stage1_pipeline(self):
        if self._sdxl_current_stage() == 1:
            return self.pipeline
        repo_id = self._sdxl_validation_stage_model(1)
        return self._get_sdxl_validation_pipeline(
            self.SDXL_VALIDATION_PIPELINE_KEYS["stage1"],
            StableDiffusionXLPipeline,
            repo_id,
        )

    def _sdxl_stage2_pipeline(self):
        if self._sdxl_current_stage() == 2:
            pipeline = self.get_pipeline(PipelineTypes.IMG2IMG, load_base_model=False)
            scheduler = self._sdxl_clone_validation_scheduler()
            if scheduler is not None:
                pipeline.scheduler = scheduler
            pipeline.to(self.accelerator.device)
            pipeline.set_progress_bar_config(disable=True)
            return pipeline
        repo_id = self._sdxl_validation_stage_model(2)
        return self._get_sdxl_validation_pipeline(
            self.SDXL_VALIDATION_PIPELINE_KEYS["stage2"],
            StableDiffusionXLImg2ImgPipeline,
            repo_id,
        )

    def get_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True):
        if (
            not load_base_model
            and pipeline_type is PipelineTypes.TEXT2IMG
            and self.supports_multistage_validation()
            and self._sdxl_current_stage() == 2
        ):
            pipeline_type = PipelineTypes.IMG2IMG
        return super().get_pipeline(pipeline_type=pipeline_type, load_base_model=load_base_model)

    def unload_validation_models(self) -> None:
        super().unload_validation_models()
        for key in (*self.SDXL_VALIDATION_PIPELINE_KEYS.values(), PipelineTypes.IMG2IMG):
            if key in self.pipelines:
                del self.pipelines[key]

    def _sdxl_split_schedule_boundary(self) -> float:
        strength = float(getattr(self.config, "refiner_training_strength", 0.2) or 0.0)
        if strength <= 0.0 or strength >= 1.0:
            raise ValueError(
                "refiner_training_strength must be greater than 0 and less than 1 for SDXL full-pipeline validation."
            )
        return 1.0 - strength

    def _sdxl_prompt_kwargs(self, pipeline_kwargs: Dict[str, Any], *, use_embeds: bool) -> Dict[str, Any]:
        if use_embeds and "prompt_embeds" in pipeline_kwargs:
            keys = (
                "prompt_embeds",
                "negative_prompt_embeds",
                "pooled_prompt_embeds",
                "negative_pooled_prompt_embeds",
            )
            return {key: pipeline_kwargs[key] for key in keys if key in pipeline_kwargs}
        return {
            "prompt": pipeline_kwargs.get("_validation_prompt_text") or pipeline_kwargs.get("prompt") or "",
            "negative_prompt": (
                pipeline_kwargs.get("_validation_negative_prompt_text") or pipeline_kwargs.get("negative_prompt") or ""
            ),
        }

    def _sdxl_copy_optional_kwargs(self, pipeline_kwargs: Dict[str, Any], keys: tuple[str, ...]) -> Dict[str, Any]:
        return {key: pipeline_kwargs[key] for key in keys if key in pipeline_kwargs}

    def _sdxl_result_count(self, images) -> int:
        if isinstance(images, torch.Tensor) and images.ndim > 0:
            return int(images.shape[0])
        try:
            return len(images)
        except TypeError:
            return 1

    def run_multistage_validation(
        self,
        pipeline_kwargs: Dict[str, Any],
        pipeline_call: Callable[[Any, Dict[str, Any]], Any],
    ) -> Any:
        stage1 = self._sdxl_stage1_pipeline()
        stage2 = self._sdxl_stage2_pipeline()
        trained_stage = self._sdxl_current_stage()
        split_boundary = self._sdxl_split_schedule_boundary()
        common_optional = (
            "generator",
            "guidance_rescale",
            "callback",
            "callback_steps",
            "callback_on_step_end",
            "callback_on_step_end_tensor_inputs",
        )

        stage1_kwargs = {
            **self._sdxl_prompt_kwargs(pipeline_kwargs, use_embeds=trained_stage == 1),
            **self._sdxl_copy_optional_kwargs(pipeline_kwargs, common_optional),
            "num_images_per_prompt": pipeline_kwargs.get("num_images_per_prompt", 1),
            "num_inference_steps": int(pipeline_kwargs["num_inference_steps"]),
            "guidance_scale": float(pipeline_kwargs.get("guidance_scale", 7.5)),
            "denoising_end": split_boundary,
            "output_type": "latent",
            "width": pipeline_kwargs.get("width"),
            "height": pipeline_kwargs.get("height"),
        }
        logger.info("Running SDXL validation stage 1 to %.2f of the schedule.", split_boundary)
        stage1_result = pipeline_call(stage1, stage1_kwargs, target_stage="stage1")
        stage1_images = stage1_result.images
        use_stage2_embeds = trained_stage == 2
        stage2_prompt_kwargs = self._sdxl_prompt_kwargs(pipeline_kwargs, use_embeds=use_stage2_embeds)
        if not use_stage2_embeds:
            image_count = self._sdxl_result_count(stage1_images)
            for key in ("prompt", "negative_prompt"):
                if isinstance(stage2_prompt_kwargs.get(key), str):
                    stage2_prompt_kwargs[key] = [stage2_prompt_kwargs[key]] * image_count

        stage2_kwargs = {
            **stage2_prompt_kwargs,
            **self._sdxl_copy_optional_kwargs(pipeline_kwargs, common_optional),
            "image": stage1_images,
            "num_images_per_prompt": pipeline_kwargs.get("num_images_per_prompt", 1) if use_stage2_embeds else 1,
            "num_inference_steps": int(pipeline_kwargs["num_inference_steps"]),
            "guidance_scale": float(pipeline_kwargs.get("guidance_scale", 7.5)),
            "denoising_start": split_boundary,
            "output_type": "pil",
        }
        logger.info("Running SDXL validation stage 2 from %.2f of the schedule.", split_boundary)
        return pipeline_call(stage2, stage2_kwargs, target_stage="stage2")

    def check_user_config(self):
        """
        Checks self.config values against important issues. Optionally implemented in child class.
        """
        if self.config.unet_attention_slice:
            if torch.backends.mps.is_available():
                logger.warning(
                    f"Using attention slicing when training {self.NAME} on MPS can result in NaN errors on the first backward pass. If you run into issues, disable this option and reduce your batch size instead to reduce memory consumption."
                )
            if self.model.get_trained_component() is not None:
                self.model.get_trained_component().set_attention_slice("auto")

        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        if self.config.tokenizer_max_length is not None:
            logger.warning(f"-!- {self.NAME} supports a max length of 77 tokens, --tokenizer_max_length is ignored -!-")
        if self.config.aspect_bucket_alignment != 64:
            logger.warning(
                f"{self.NAME} requires an alignment value of 64px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 64

        if self.config.prediction_type is not None:
            logger.info(f"Setting {self.NAME} prediction type: {self.config.prediction_type}")
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
                output_args.append(f"soft_min_snr_sigma_data={self.config.soft_min_snr_sigma_data}")
        if self.config.rescale_betas_zero_snr:
            output_args.append(f"rescale_betas_zero_snr")
        if self.config.offset_noise:
            output_args.append(f"offset_noise")
            output_args.append(f"noise_offset={self.config.noise_offset}")
            output_args.append(f"noise_offset_probability={self.config.noise_offset_probability}")
        output_args.append(f"training_scheduler_timestep_spacing={self.config.training_scheduler_timestep_spacing}")
        output_args.append(f"inference_scheduler_timestep_spacing={self.config.inference_scheduler_timestep_spacing}")
        output_str = f" (extra parameters={output_args})" if output_args else " (no special parameters set)"

        return output_str

    @classmethod
    def get_acceleration_presets(cls) -> list[AccelerationPreset]:
        # Common settings for memory optimization presets
        _base_memory_config = {
            "base_model_precision": "no_change",
            "gradient_checkpointing": True,
        }

        return [
            # Basic tab - Gradient checkpointing
            AccelerationPreset(
                backend=AccelerationBackend.GRADIENT_CHECKPOINTING,
                level="basic",
                name="Gradient Checkpointing",
                description="Trade compute for memory by recomputing activations during backward pass.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~30%",
                tradeoff_speed="Increases training time by ~20%",
                tradeoff_notes="Recommended for most users.",
                config=_base_memory_config,
            ),
            # DeepSpeed presets (multi-GPU only)
            *get_deepspeed_presets(_base_memory_config),
            # SDNQ presets (works on AMD, Apple, NVIDIA)
            *get_sdnq_presets(_base_memory_config),
            # TorchAO presets (NVIDIA only)
            *get_torchao_presets(_base_memory_config),
            # Quanto presets (works on AMD, Apple, NVIDIA)
            *get_quanto_presets(_base_memory_config),
            # BitsAndBytes presets (NVIDIA only)
            *get_bitsandbytes_presets(_base_memory_config),
        ]


from simpletuner.helpers.models.registry import ModelRegistry

ModelRegistry.register("sdxl", SDXL)
