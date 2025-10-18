import logging
import os
import random
from typing import Dict, Optional

import torch
from diffusers import AutoencoderKLWan
from torchvision import transforms
from transformers import T5TokenizerFast, UMT5EncoderModel

from simpletuner.helpers.models.common import ModelTypes, PipelineTypes, PredictionTypes, VideoModelFoundation, VideoToTensor
from simpletuner.helpers.models.wan.pipeline import WanPipeline
from simpletuner.helpers.models.wan.transformer import WanTransformer3DModel

logger = logging.getLogger(__name__)
from torch.nn import functional as F

from simpletuner.helpers.training.multi_process import should_log
from simpletuner.helpers.training.tread import TREADRouter

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class Wan(VideoModelFoundation):
    NAME = "Wan"
    MODEL_DESCRIPTION = "Video generation model (text-to-video)"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKLWan
    LATENT_CHANNEL_COUNT = 16
    DEFAULT_NOISE_SCHEDULER = "unipc"
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    # Only training the Attention blocks by default.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = WanTransformer3DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: WanPipeline,
        # PipelineTypes.IMG2IMG: None,
        # PipelineTypes.CONTROLNET: None,
    }

    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "t2v-480p-1.3b-2.1"
    HUGGINGFACE_PATHS = {
        "t2v-480p-1.3b-2.1": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        "t2v-480p-14b-2.1": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "i2v-14b-2.2-high": "Wan-AI/Wan2.2-I2V-14B-Diffusers",
        "i2v-14b-2.2-low": "Wan-AI/Wan2.2-I2V-14B-Diffusers",
        # "i2v-480p-14b-2.1": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        # "i2v-720p-14b-2.1": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    }
    MODEL_LICENSE = "apache-2.0"

    WAN_STAGE_OVERRIDES: Dict[str, Dict[str, object]] = {
        "i2v-14b-2.2-high": {
            "trained_stage": "high",
            "stage_subfolder": "high_noise_model",
            "other_stage_subfolder": "low_noise_model",
            "flow_shift": 5.0,
            "sample_steps": 40,
            "boundary_ratio": 0.90,
            "guidance": {"high": 3.5, "low": 3.5},
        },
        "i2v-14b-2.2-low": {
            "trained_stage": "low",
            "stage_subfolder": "low_noise_model",
            "other_stage_subfolder": "high_noise_model",
            "flow_shift": 5.0,
            "sample_steps": 40,
            "boundary_ratio": 0.90,
            "guidance": {"high": 3.5, "low": 3.5},
        },
    }

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "UMT5",
            "tokenizer": T5TokenizerFast,
            "subfolder": "text_encoder",
            "tokenizer_subfolder": "tokenizer",
            "model": UMT5EncoderModel,
        },
    }

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        self._wan_cached_stage_modules: Dict[str, WanTransformer3DModel] = {}

    def requires_conditioning_image_embeds(self) -> bool:
        return self._wan_stage_info() is not None

    def setup_model_flavour(self):
        super().setup_model_flavour()
        stage_info = self._wan_stage_info()
        if stage_info is None:
            return

        if getattr(self.config, "pretrained_transformer_model_name_or_path", None) is None:
            self.config.pretrained_transformer_model_name_or_path = self.config.pretrained_model_name_or_path
        self.config.pretrained_transformer_subfolder = stage_info["stage_subfolder"]

        self.config.wan_trained_stage = stage_info["trained_stage"]
        self.config.wan_stage_main_subfolder = stage_info["stage_subfolder"]
        self.config.wan_stage_other_subfolder = stage_info["other_stage_subfolder"]
        self.config.wan_boundary_ratio = stage_info["boundary_ratio"]

        self.config.flow_schedule_shift = stage_info["flow_shift"]
        self.config.validation_num_inference_steps = stage_info["sample_steps"]
        self.config.validation_guidance = stage_info["guidance"][stage_info["trained_stage"]]

        if not hasattr(self.config, "wan_validation_load_other_stage"):
            self.config.wan_validation_load_other_stage = False

    def _wan_stage_info(self) -> Optional[Dict[str, object]]:
        flavour = getattr(self.config, "model_flavour", None)
        return self.WAN_STAGE_OVERRIDES.get(flavour)

    def _should_load_other_stage(self) -> bool:
        stage_info = self._wan_stage_info()
        if stage_info is None:
            return False
        return bool(getattr(self.config, "wan_validation_load_other_stage", False))

    def _get_or_load_wan_stage_module(self, subfolder: str) -> WanTransformer3DModel:
        if subfolder in self._wan_cached_stage_modules:
            return self._wan_cached_stage_modules[subfolder]

        logger.info("Loading Wan stage weights for validation from subfolder '%s'.", subfolder)
        stage = self.MODEL_CLASS.from_pretrained(
            self.config.pretrained_model_name_or_path,
            subfolder=subfolder,
            torch_dtype=self.config.weight_dtype,
            use_safetensors=True,
        )
        stage.requires_grad_(False)
        stage.to(self.accelerator.device, dtype=self.config.weight_dtype)
        stage.eval()
        self._wan_cached_stage_modules[subfolder] = stage
        return stage

    def unload_model(self):
        super().unload_model()
        self._wan_cached_stage_modules.clear()

    def get_group_offload_components(self, pipeline):
        components = dict(super().get_group_offload_components(pipeline))
        transformer_2 = getattr(pipeline, "transformer_2", None)
        if transformer_2 is not None:
            components.setdefault("transformer_2", transformer_2)
        return components

    def get_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True):
        pipeline = super().get_pipeline(pipeline_type, load_base_model)
        stage_info = self._wan_stage_info()
        if stage_info is None:
            return pipeline

        load_other = self._should_load_other_stage()
        trained_stage = stage_info["trained_stage"]
        other_subfolder = stage_info["other_stage_subfolder"]

        if trained_stage == "low":
            if load_other:
                pipeline.transformer_2 = pipeline.transformer
                pipeline.transformer = self._get_or_load_wan_stage_module(other_subfolder)
            else:
                pipeline.transformer_2 = None
        else:  # trained_stage == "high"
            if load_other:
                pipeline.transformer_2 = self._get_or_load_wan_stage_module(other_subfolder)
            else:
                pipeline.transformer_2 = None

        if load_other:
            pipeline.config.boundary_ratio = stage_info["boundary_ratio"]
        else:
            pipeline.config.boundary_ratio = None

        return pipeline

    def tread_init(self):
        """
        Initialize the TREAD model training method for Wan.
        """

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

        logger.info("TREAD training is enabled for Wan")

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        """
        When we're running the pipeline, we'll update the kwargs specifically for this model here.
        """
        # Wan video should max out around 81 frames for efficiency.
        pipeline_kwargs["num_frames"] = min(81, self.config.validation_num_video_frames or 81)
        pipeline_kwargs["output_type"] = "pil"

        stage_info = self._wan_stage_info()
        if stage_info is not None:
            trained_stage = stage_info["trained_stage"]
            pipeline_kwargs["num_inference_steps"] = stage_info["sample_steps"]
            pipeline_kwargs["guidance_scale"] = stage_info["guidance"][trained_stage]
            if self._should_load_other_stage():
                other_stage = "low" if trained_stage == "high" else "high"
                pipeline_kwargs["guidance_scale_2"] = stage_info["guidance"][other_stage]
            else:
                pipeline_kwargs.pop("guidance_scale_2", None)
        else:
            pipeline_kwargs.pop("guidance_scale_2", None)

        return pipeline_kwargs

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        """
        Models can optionally format the stored text embedding, eg. in a dict, or
        filter certain outputs from appearing in the file cache.

        self.config:
            text_embedding (torch.Tensor): The embed to adjust.

        Returns:
            torch.Tensor: The adjusted embed. By default, this method does nothing.
        """
        prompt_embeds, masks = text_embedding

        return {
            "prompt_embeds": prompt_embeds,
            "attention_masks": masks,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        return {
            "prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            # "attention_mask": (
            #     text_embedding["attention_masks"].unsqueeze(0)
            #     if self.config.flux_attention_masked_training
            #     else None
            # ),
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor, prompt: str) -> dict:
        # logger.info(f"Converting embeds with shapes: {text_embedding['prompt_embeds'].shape} {text_embedding['pooled_prompt_embeds'].shape}")
        return {
            "negative_prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
            # "negative_mask": (
            #     text_embedding["attention_masks"].unsqueeze(0)
            #     if self.config.flux_attention_masked_training
            #     else None
            # ),
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode a prompt.

        Args:
            prompts: The list of prompts to encode.

        Returns:
            Text encoder output (raw)
        """
        prompt_embeds, masks = self.pipelines[PipelineTypes.TEXT2IMG].encode_prompt(
            prompt=prompts,
            device=self.accelerator.device,
        )
        if self.config.t5_padding == "zero":
            # we can zero the padding tokens if we're just going to mask them later anyway.
            prompt_embeds = prompt_embeds * masks.to(device=prompt_embeds.device).unsqueeze(-1).expand(prompt_embeds.shape)

        return prompt_embeds, masks

    def model_predict(self, prepared_batch):
        """
        Modify the existing model_predict to support TREAD with masked training.
        """
        wan_transformer_kwargs = {
            "hidden_states": prepared_batch["noisy_latents"].to(self.config.weight_dtype),
            "encoder_hidden_states": prepared_batch["encoder_hidden_states"].to(self.config.weight_dtype),
            "timestep": prepared_batch["timesteps"],
            "return_dict": False,
        }

        if prepared_batch.get("conditioning_image_embeds") is not None:
            wan_transformer_kwargs["encoder_hidden_states_image"] = prepared_batch["conditioning_image_embeds"].to(
                self.config.weight_dtype
            )

        # For masking with TREAD, avoid dropping any tokens that are in the mask
        if (
            getattr(self.config, "tread_config", None) is not None
            and self.config.tread_config is not None
            and "conditioning_pixel_values" in prepared_batch
            and prepared_batch["conditioning_pixel_values"] is not None
            and prepared_batch.get("conditioning_type") in ("mask", "segmentation")
        ):
            with torch.no_grad():
                # For video: B, C, T, H, W
                b, c, t, h, w = prepared_batch["latents"].shape
                # Wan uses patch_size (1, 2, 2), so token dimensions are:
                t_tokens = t // 1  # temporal patches
                h_tokens = h // 2  # height patches
                w_tokens = w // 2  # width patches

                mask_vid = prepared_batch["conditioning_pixel_values"]  # (B,C,T,H,W) for video
                # fuse channels → single channel, map to [0,1]
                mask_vid = (mask_vid.mean(1, keepdim=True) + 1) / 2
                # downsample to match token dimensions
                mask_tok = F.interpolate(
                    mask_vid,
                    size=(t_tokens, h_tokens, w_tokens),
                    mode="trilinear",
                    align_corners=False,
                )  # (B,1,t_tok,h_tok,w_tok)
                # Flatten in the same order as patch_embedding
                # After conv3d: (B, D, T', H', W')
                # After flatten(2): (B, D, T'*H'*W') with order T->H->W
                # After transpose(1,2): (B, T'*H'*W', D)
                # So we flatten the mask with the same T->H->W order
                force_keep = mask_tok.squeeze(1).flatten(1) > 0.5  # (B, S_vid)
                wan_transformer_kwargs["force_keep_mask"] = force_keep

        model_pred = self.model(**wan_transformer_kwargs)[0]

        return {
            "model_prediction": model_pred,
        }

    def check_user_config(self):
        """
        Checks self.config values against important issues.
        """
        stage_info = self._wan_stage_info()
        if stage_info is not None:
            trained_stage = stage_info["trained_stage"]
            self.config.validation_guidance = stage_info["guidance"][trained_stage]
            if hasattr(self.config, "validation_guidance_skip_layers"):
                self.config.validation_guidance_skip_layers = None
            if hasattr(self.config, "validation_guidance_skip_layers_start"):
                self.config.validation_guidance_skip_layers_start = None
            if hasattr(self.config, "validation_guidance_skip_layers_stop"):
                self.config.validation_guidance_skip_layers_stop = None
            if hasattr(self.config, "validation_guidance_skip_scale"):
                self.config.validation_guidance_skip_scale = None

        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        if self.config.aspect_bucket_alignment != 32:
            logger.warning(
                f"{self.NAME} requires an alignment value of 32px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 32

        if self.config.prediction_type is not None:
            logger.warning(f"{self.NAME} does not support prediction type {self.config.prediction_type}.")

        if self.config.tokenizer_max_length is not None:
            logger.warning(f"-!- {self.NAME} supports a max length of 512 tokens, --tokenizer_max_length is ignored -!-")
        self.config.tokenizer_max_length = 512
        if self.config.validation_num_inference_steps > 50:
            logger.warning(
                f"{self.NAME} {self.config.model_flavour} may be wasting compute with more than 50 steps. Consider reducing the value to save time."
            )
        if self.config.validation_num_inference_steps < 40:
            logger.warning(
                f"{self.NAME} {self.config.model_flavour} expects around 40 or more inference steps. Consider increasing --validation_num_inference_steps to 40."
            )
        if not self.config.validation_disable_unconditional:
            logger.info("Disabling unconditional validation to save on time.")
            self.config.validation_disable_unconditional = True

        if self.config.framerate is None:
            self.config.framerate = 15

        self.config.vae_enable_tiling = True
        self.config.vae_enable_slicing = True

    def custom_model_card_schedule_info(self):
        output_args = []
        if self.config.flow_schedule_auto_shift:
            output_args.append("flow_schedule_auto_shift")
        if self.config.flow_schedule_shift is not None:
            output_args.append(f"shift={self.config.flow_schedule_shift}")
        if self.config.flow_use_beta_schedule:
            output_args.append(f"flow_beta_schedule_alpha={self.config.flow_beta_schedule_alpha}")
            output_args.append(f"flow_beta_schedule_beta={self.config.flow_beta_schedule_beta}")
        if self.config.t5_padding != "unmodified":
            output_args.append(f"t5_padding={self.config.t5_padding}")
        output_str = f" (extra parameters={output_args})" if output_args else " (no special parameters set)"

        return output_str

    def get_transforms(self, dataset_type: str = "image"):
        return transforms.Compose(
            [
                VideoToTensor() if dataset_type == "video" else transforms.ToTensor(),
                # Normalize [0,1] input to [-1,1] range using (input - 0.5) / 0.5
                transforms.Normalize([0.5], [0.5]),
            ]
        )


from simpletuner.helpers.models.registry import ModelRegistry

ModelRegistry.register("wan", Wan)
