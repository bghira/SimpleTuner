import logging
import os
from types import SimpleNamespace
from typing import Any, Callable, Dict

import torch
from diffusers import AutoencoderKL, StableDiffusionUpscalePipeline
from diffusers.pipelines import IFPipeline, IFSuperResolutionPipeline
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from peft import set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from transformers import AutoTokenizer, T5EncoderModel

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
from simpletuner.helpers.models.unet_flowmap import FlowMapUNet2DConditionModel as UNet2DConditionModel

logger = logging.getLogger(__name__)
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class DeepFloydIF(ImageModelFoundation):
    NAME = "DeepFloyd IF"
    MODEL_DESCRIPTION = "Pixel-space diffusion model with T5 text encoder"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.EPSILON
    MODEL_TYPE = ModelTypes.UNET
    # DeepFloyd-IF is a pixel space model.
    AUTOENCODER_CLASS = None
    LATENT_CHANNEL_COUNT = None
    DEFAULT_NOISE_SCHEDULER = "ddpm"
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    SLIDER_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    # Only training the Attention blocks by default seems to be the most stable..
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = UNet2DConditionModel
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: IFPipeline,
        PipelineTypes.IMG2IMG: IFSuperResolutionPipeline,
    }
    MODEL_SUBFOLDER = "unet"
    REQUIRES_FLAVOUR = True
    HUGGINGFACE_PATHS = {
        # stage one, text-to-image models
        "i-medium-400m": "DeepFloyd/IF-I-M-v1.0",
        "i-large-900m": "DeepFloyd/IF-I-L-v1.0",
        "i-xlarge-4.3b": "DeepFloyd/IF-I-XL-v1.0",
        # stage two, super-resolution models
        "ii-medium-450m": "DeepFloyd/IF-II-M-v1.0",
        "ii-large-1.2b": "DeepFloyd/IF-II-L-v1.0",
    }
    MODEL_LICENSE = "deepfloyd-if-license"
    DEFAULT_STAGE1_MODEL = "DeepFloyd/IF-I-XL-v1.0"
    DEFAULT_STAGE2_MODEL = "DeepFloyd/IF-II-M-v1.0"
    DEFAULT_STAGE3_MODEL = "stabilityai/stable-diffusion-x4-upscaler"
    SUPPORTS_MULTISTAGE_VALIDATION = True
    DEEPFLOYD_VALIDATION_PIPELINE_MODES = {"auto", "trained-stage", "full-pipeline"}
    DEEPFLOYD_VALIDATION_STAGE3_MODES = {"none", "sd-x4-upscaler"}

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "T5 XXL v1.1",
            "tokenizer": AutoTokenizer,
            "subfolder": "text_encoder",
            "tokenizer_subfolder": "tokenizer",
            "model": T5EncoderModel,
        },
    }

    def validation_image_input_edge_length(self):
        # If a model requires a specific input edge length (HiDream E1 -> 768px, DeepFloyd stage2 -> 64px)
        if self.config.model_flavour.startswith("ii-"):
            return 64
        return None

    def _deepfloyd_current_stage(self) -> int:
        return 2 if str(getattr(self.config, "model_flavour", "")).startswith("ii-") else 1

    def _deepfloyd_validation_mode(self) -> str:
        mode = getattr(self.config, "deepfloyd_validation_pipeline_mode", "auto") or "auto"
        mode = str(mode).strip().lower()
        if mode not in self.DEEPFLOYD_VALIDATION_PIPELINE_MODES:
            raise ValueError(
                "deepfloyd_validation_pipeline_mode must be one of: "
                f"{', '.join(sorted(self.DEEPFLOYD_VALIDATION_PIPELINE_MODES))}"
            )
        if mode == "auto" and getattr(self.config, "validation_using_datasets", False):
            return "trained-stage"
        return "full-pipeline" if mode == "auto" else mode

    def supports_multistage_validation(self) -> bool:
        return self._deepfloyd_validation_mode() == "full-pipeline"

    def validation_adapter_stage_aliases(self) -> Dict[str, set[str]]:
        return {
            "stage1": {"stage1", "stage_1", "1", "one", "i", "stage_i"},
            "stage2": {"stage2", "stage_2", "2", "two", "ii", "stage_ii"},
            "stage3": {"stage3", "stage_3", "3", "three", "iii", "stage_iii"},
        }

    def _deepfloyd_validation_stage3_mode(self) -> str:
        mode = getattr(self.config, "deepfloyd_validation_stage3_mode", "none") or "none"
        mode = str(mode).strip().lower()
        if mode not in self.DEEPFLOYD_VALIDATION_STAGE3_MODES:
            raise ValueError(
                "deepfloyd_validation_stage3_mode must be one of: "
                f"{', '.join(sorted(self.DEEPFLOYD_VALIDATION_STAGE3_MODES))}"
            )
        return mode

    def _deepfloyd_validation_repo(self, config_name: str, default_repo: str) -> str:
        repo = getattr(self.config, config_name, None) or default_repo
        return str(repo)

    def _deepfloyd_from_pretrained_kwargs(self, *, load_text_encoder: bool) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "torch_dtype": self.config.weight_dtype,
        }
        if not load_text_encoder:
            kwargs["text_encoder"] = None
            kwargs["tokenizer"] = None
        for attr in ("revision", "cache_dir", "local_files_only"):
            value = getattr(self.config, attr, None)
            if value is not None:
                kwargs[attr] = value
        variant = getattr(self.config, "variant", None)
        if variant is not None:
            kwargs["variant"] = variant
        return kwargs

    def _get_deepfloyd_validation_pipeline(
        self,
        cache_key: str,
        pipeline_cls,
        repo_id: str,
        *,
        load_text_encoder: bool = False,
    ):
        if cache_key in self.pipelines:
            return self.pipelines[cache_key]

        load_kwargs = self._deepfloyd_from_pretrained_kwargs(load_text_encoder=load_text_encoder)
        signature = pipeline_cls.from_pretrained
        logger.info("Loading DeepFloyd validation pipeline '%s' from %s.", cache_key, repo_id)
        pipeline = signature(repo_id, **load_kwargs)
        for attr in ("safety_checker", "watermarker", "feature_extractor"):
            if hasattr(pipeline, attr):
                setattr(pipeline, attr, None)
        pipeline.to(self.accelerator.device)
        pipeline.set_progress_bar_config(disable=True)
        self.pipelines[cache_key] = pipeline
        return pipeline

    def _deepfloyd_stage1_pipeline(self):
        if self._deepfloyd_current_stage() == 1:
            return self.pipeline
        repo_id = self._deepfloyd_validation_repo(
            "deepfloyd_validation_stage1_model",
            self.DEFAULT_STAGE1_MODEL,
        )
        return self._get_deepfloyd_validation_pipeline(
            "deepfloyd_validation_stage1",
            IFPipeline,
            repo_id,
        )

    def _deepfloyd_stage2_pipeline(self):
        if self._deepfloyd_current_stage() == 2:
            return self.pipeline
        repo_id = self._deepfloyd_validation_repo(
            "deepfloyd_validation_stage2_model",
            self.DEFAULT_STAGE2_MODEL,
        )
        return self._get_deepfloyd_validation_pipeline(
            "deepfloyd_validation_stage2",
            IFSuperResolutionPipeline,
            repo_id,
        )

    def _deepfloyd_stage3_pipeline(self):
        repo_id = self._deepfloyd_validation_repo(
            "deepfloyd_validation_stage3_model",
            self.DEFAULT_STAGE3_MODEL,
        )
        return self._get_deepfloyd_validation_pipeline(
            "deepfloyd_validation_stage3",
            StableDiffusionUpscalePipeline,
            repo_id,
            load_text_encoder=True,
        )

    def unload_validation_models(self) -> None:
        for key in (
            "deepfloyd_validation_stage1",
            "deepfloyd_validation_stage2",
            "deepfloyd_validation_stage3",
        ):
            if key in self.pipelines:
                del self.pipelines[key]

    def _deepfloyd_stage1_resolution(self, requested_width: int, requested_height: int) -> tuple[int, int]:
        stage3_scale = 4 if self._deepfloyd_validation_stage3_mode() == "sd-x4-upscaler" else 1
        stage2_width = max(64, int(requested_width / stage3_scale))
        stage2_height = max(64, int(requested_height / stage3_scale))
        width = max(64, int(stage2_width / 4))
        height = max(64, int(stage2_height / 4))
        width = max(8, (width // 8) * 8)
        height = max(8, (height // 8) * 8)
        return width, height

    def _deepfloyd_stage_steps(self, stage: int, default: int) -> int:
        value = getattr(self.config, f"deepfloyd_validation_stage{stage}_num_inference_steps", None)
        if value is None:
            value = default
        return max(1, int(value))

    def _deepfloyd_stage_guidance(self, stage: int, default: float) -> float:
        value = getattr(self.config, f"deepfloyd_validation_stage{stage}_guidance", None)
        if value is None:
            value = default
        return float(value)

    def run_multistage_validation(
        self,
        pipeline_kwargs: Dict[str, Any],
        pipeline_call: Callable[[Any, Dict[str, Any]], Any],
    ) -> Any:
        stage1 = self._deepfloyd_stage1_pipeline()
        stage2 = self._deepfloyd_stage2_pipeline()
        width = int(pipeline_kwargs.get("width", 256))
        height = int(pipeline_kwargs.get("height", 256))
        stage1_width, stage1_height = self._deepfloyd_stage1_resolution(width, height)
        stage2_width = stage1_width * 4
        stage2_height = stage1_height * 4

        stage1_kwargs = {
            "prompt_embeds": pipeline_kwargs["prompt_embeds"],
            "negative_prompt_embeds": pipeline_kwargs.get("negative_prompt_embeds"),
            "num_inference_steps": self._deepfloyd_stage_steps(1, min(int(pipeline_kwargs["num_inference_steps"]), 30)),
            "generator": pipeline_kwargs.get("generator"),
            "guidance_scale": self._deepfloyd_stage_guidance(1, float(pipeline_kwargs.get("guidance_scale", 7.0))),
            "output_type": "pt",
            "width": stage1_width,
            "height": stage1_height,
            "num_images_per_prompt": pipeline_kwargs.get("num_images_per_prompt", 1),
        }
        logger.info("Running DeepFloyd validation stage I at %sx%s.", stage1_width, stage1_height)
        stage1_result = pipeline_call(stage1, stage1_kwargs, target_stage="stage1")
        stage1_images = stage1_result.images

        stage2_kwargs = {
            "image": stage1_images,
            "prompt_embeds": pipeline_kwargs["prompt_embeds"],
            "negative_prompt_embeds": pipeline_kwargs.get("negative_prompt_embeds"),
            "num_inference_steps": self._deepfloyd_stage_steps(2, int(pipeline_kwargs["num_inference_steps"])),
            "generator": pipeline_kwargs.get("generator"),
            "guidance_scale": self._deepfloyd_stage_guidance(2, float(pipeline_kwargs.get("guidance_scale", 4.0))),
            "output_type": "pil",
            "width": stage2_width,
            "height": stage2_height,
            "num_images_per_prompt": pipeline_kwargs.get("num_images_per_prompt", 1),
        }
        logger.info("Running DeepFloyd validation stage II at %sx%s.", stage2_width, stage2_height)
        stage2_result = pipeline_call(stage2, stage2_kwargs, target_stage="stage2")

        if self._deepfloyd_validation_stage3_mode() != "sd-x4-upscaler":
            return stage2_result

        stage3 = self._deepfloyd_stage3_pipeline()
        prompt = pipeline_kwargs.get("_validation_prompt_text") or pipeline_kwargs.get("prompt") or ""
        negative_prompt = (
            pipeline_kwargs.get("_validation_negative_prompt_text") or pipeline_kwargs.get("negative_prompt") or ""
        )
        stage2_images = stage2_result.images
        if not isinstance(stage2_images, list):
            stage2_images = [stage2_images]
        stage3_kwargs = {
            "prompt": [prompt] * len(stage2_images),
            "negative_prompt": [negative_prompt] * len(stage2_images),
            "image": stage2_images,
            "noise_level": int(getattr(self.config, "deepfloyd_validation_stage3_noise_level", 100) or 100),
            "guidance_scale": self._deepfloyd_stage_guidance(3, float(pipeline_kwargs.get("guidance_scale", 4.0))),
        }
        logger.info("Running DeepFloyd validation stage III with Stable Diffusion x4 upscaler.")
        stage3_result = pipeline_call(stage3, stage3_kwargs, target_stage="stage3")
        if hasattr(stage3_result, "images"):
            return stage3_result
        return SimpleNamespace(images=stage3_result)

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        """
        Models can optionally format the stored text embedding, eg. in a dict, or
        filter certain outputs from appearing in the file cache.

        self.config:
            text_embedding (torch.Tensor): The embed to adjust.

        Returns:
            torch.Tensor: The adjusted embed. By default, this method does nothing.
        """
        return {"prompt_embeds": text_embedding}

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # Only unsqueeze if it's missing the batch dimension
        prompt_embeds = text_embedding["prompt_embeds"]

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:  # Shape: [seq, dim]
            prompt_embeds = prompt_embeds.unsqueeze(0)  # Shape: [1, seq, dim]

        return {
            "prompt_embeds": prompt_embeds,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # Only unsqueeze if it's missing the batch dimension
        prompt_embeds = text_embedding["prompt_embeds"]

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:  # Shape: [seq, dim]
            prompt_embeds = prompt_embeds.unsqueeze(0)  # Shape: [1, seq, dim]

        return {
            "negative_prompt_embeds": prompt_embeds,
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode a prompt for a DeepFloyd model.

        Args:
            prompts: The list of prompts to encode.

        Returns:
            Text encoder output (raw)
        """

        pipeline = self.pipelines.get(PipelineTypes.TEXT2IMG)
        if pipeline is None:
            pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        positive_embed, negative_embed = pipeline.encode_prompt(
            prompt=prompts,
            do_classifier_free_guidance=False,
            device=self.accelerator.device,
        )

        return positive_embed

    def model_predict(self, prepared_batch):
        logger.debug(
            "Input shapes:"
            f"\n{prepared_batch['noisy_latents'].shape}"
            f"\n{prepared_batch['timesteps'].shape}"
            f"\n{prepared_batch['encoder_hidden_states'].shape}"
        )
        prediction_kwargs = {
            "timestep": prepared_batch["timesteps"],
            "encoder_hidden_states": prepared_batch["encoder_hidden_states"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
        }
        if self.config.model_flavour.startswith("ii-"):
            # expand noisy latents by doubling dim
            logger.info(f"Pre-expansion shape: {prepared_batch['noisy_latents'].shape}")
            prepared_batch["noisy_latents"] = torch.cat(
                [prepared_batch["noisy_latents"], prepared_batch["noisy_latents"]],
                dim=1,
            )
            logger.info(f"Post--expansion shape: {prepared_batch['noisy_latents'].shape}")
            prediction_kwargs["class_labels"] = prepared_batch["timesteps"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            )
        self._apply_flowmap_r_timestep_kwargs(prediction_kwargs, prepared_batch)
        model_pred = self.model(
            prepared_batch["noisy_latents"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            return_dict=False,
            **prediction_kwargs,
        )[0]
        # chunk prediction and discard learnt variance
        model_pred = model_pred.chunk(2, dim=1)[0]
        return {
            "model_prediction": model_pred,
        }

    def check_user_config(self):
        """
        Checks self.config values against important issues. Optionally implemented in child class.
        """
        self._deepfloyd_validation_mode()
        self._deepfloyd_validation_stage3_mode()
        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                "DeepFloyd does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        t5_max_length = 77
        if self.config.tokenizer_max_length is None or int(self.config.tokenizer_max_length) > t5_max_length:
            if not self.config.i_know_what_i_am_doing:
                logger.warning(f"Updating T5 XXL tokeniser max length to {t5_max_length} for DeepFloyd.")
                self.config.tokenizer_max_length = t5_max_length
            else:
                logger.warning(
                    f"-!- DeepFloyd supports a max length of {t5_max_length} tokens, but you have supplied `--i_know_what_i_am_doing`, so this limit will not be enforced. -!-"
                )
                logger.warning(
                    f"The model will begin to collapse after a short period of time, if the model you are continuing from has not been tuned beyond {t5_max_length} tokens."
                )
        # Disable custom VAEs for DeepFloyd.
        self.config.pretrained_vae_model_name_or_path = None
        self.config.vae_path = None

        if self.config.aspect_bucket_alignment != 32:
            logger.warning(
                "DeepFloyd requires an alignment value of 32px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 32

        # Stage II needs different default pipeline.
        if self.config.model_flavour.startswith("ii-"):
            self.DEFAULT_PIPELINE_TYPE = PipelineTypes.IMG2IMG

    def custom_model_card_schedule_info(self):
        output_args = []
        # TODO: Implement scheduler info for DeepFloyd.
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

ModelRegistry.register("deepfloyd", DeepFloydIF)
