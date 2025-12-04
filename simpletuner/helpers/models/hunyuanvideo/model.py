# HunyuanVideo 1.5 integration (AGPL-3.0-or-later), SimpleTuner © 2025

import logging
import os
from typing import Dict, Optional

import loguru
import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.models.common import ModelTypes, PipelineTypes, PredictionTypes, VideoModelFoundation
from simpletuner.helpers.models.hunyuanvideo.autoencoder import AutoencoderKLConv3D
from simpletuner.helpers.models.hunyuanvideo.commons import PIPELINE_CONFIGS, TRANSFORMER_VERSION_TO_SR_VERSION
from simpletuner.helpers.models.hunyuanvideo.pipeline import HunyuanVideo_1_5_Pipeline
from simpletuner.helpers.models.hunyuanvideo.scheduling_flow_match_discrete import FlowMatchDiscreteScheduler
from simpletuner.helpers.models.hunyuanvideo.text_encoders import PROMPT_TEMPLATE, TextEncoder
from simpletuner.helpers.models.hunyuanvideo.transformer import HunyuanVideo_1_5_DiffusionTransformer
from simpletuner.helpers.models.hunyuanvideo.utils.multitask_utils import merge_tensor_by_mask
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.training.multi_process import should_log
from simpletuner.helpers.training.tread import TREADRouter

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class HunyuanVideo(VideoModelFoundation):
    """
    Hunyuan Video 1.5 (8.3B) text-to-video and image-to-video transformer.
    """

    NAME = "HunyuanVideo"
    MODEL_DESCRIPTION = "Text-to-video / image-to-video flow-matching transformer (8.3B)"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKLConv3D
    LATENT_CHANNEL_COUNT = 4
    DEFAULT_NOISE_SCHEDULER = "flow_match_discrete"
    MODEL_CLASS = HunyuanVideo_1_5_DiffusionTransformer
    MODEL_SUBFOLDER = None  # Direct repos load from root
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: HunyuanVideo_1_5_Pipeline,
        PipelineTypes.IMG2VIDEO: HunyuanVideo_1_5_Pipeline,
    }
    DEFAULT_PIPELINE_TYPE = PipelineTypes.TEXT2IMG
    DEFAULT_MODEL_FLAVOUR = "t2v-480p"
    HUGGINGFACE_PATHS: Dict[str, str] = {
        "t2v-480p": "DiffusersVersionsOfModels/HunyuanVideo-1.5-480p_t2v",
        "t2v-720p": "DiffusersVersionsOfModels/HunyuanVideo-1.5-720p_t2v",
        "i2v-480p": "DiffusersVersionsOfModels/HunyuanVideo-1.5-480p_i2v",
        "i2v-720p": "DiffusersVersionsOfModels/HunyuanVideo-1.5-720p_i2v",
    }
    MODEL_LICENSE = "agpl-3.0"

    # Component repositories - direct loading without subfolders
    VAE_REPO = "DiffusersVersionsOfModels/HunyuanVideo-1.5-vae"
    GLYPH_BYT5_REPO = "DiffusersVersionsOfModels/Glyph-ByT5"
    TEXT_ENCODER_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
    VISION_ENCODER_REPO = "black-forest-labs/FLUX.1-Redux-dev"

    # Transformer version mapping for pipeline configs
    TRANSFORMER_VERSIONS: Dict[str, str] = {
        "t2v-480p": "480p_t2v",
        "t2v-720p": "720p_t2v",
        "i2v-480p": "480p_i2v",
        "i2v-720p": "720p_i2v",
    }
    STRICT_I2V_FLAVOURS = ("i2v-480p", "i2v-720p")

    # Only required to satisfy encode_text_batch checks; loading is handled manually.
    TEXT_ENCODER_CONFIGURATION = {"text_encoder": {"name": "Hunyuan LLM"}}

    def __init__(self, config: dict, accelerator):
        super().__init__(config, accelerator)
        self._transformer_version = self._resolve_transformer_version()
        self._sr_version = TRANSFORMER_VERSION_TO_SR_VERSION.get(self._transformer_version)
        # Direct repos load from root - no subfolder needed
        self.config.pretrained_transformer_subfolder = None
        if getattr(self.config, "flow_schedule_shift", None) is None:
            default_cfg = PIPELINE_CONFIGS.get(self._transformer_version, {})
            self.config.flow_schedule_shift = default_cfg.get("flow_shift", 7.0)
        if getattr(self.config, "validation_guidance", None) is None:
            default_cfg = PIPELINE_CONFIGS.get(self._transformer_version, {})
            self.config.validation_guidance = default_cfg.get("guidance_scale", 6.0)

    def _resolve_transformer_version(self) -> str:
        flavour = getattr(self.config, "model_flavour", self.DEFAULT_MODEL_FLAVOUR) or self.DEFAULT_MODEL_FLAVOUR
        if flavour not in self.TRANSFORMER_VERSIONS:
            raise ValueError(
                f"Unsupported HunyuanVideo flavour '{flavour}'. Expected one of {list(self.TRANSFORMER_VERSIONS)}"
            )
        return self.TRANSFORMER_VERSIONS[flavour]

    def setup_training_noise_schedule(self):
        self.noise_schedule = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=self.config.flow_schedule_shift,
        )
        return self.config, self.noise_schedule

    def load_text_encoder(self, move_to_device: bool = True):
        """
        Load the custom TextEncoder (llm) stack from Qwen repo.
        """
        device = self.accelerator.device if move_to_device else torch.device("cpu")
        override_path = getattr(self.config, "hunyuan_text_encoder_path", None)

        # Use override path if provided and exists, otherwise use TEXT_ENCODER_REPO
        if override_path and os.path.exists(override_path):
            text_encoder_path = override_path
        else:
            text_encoder_path = self.TEXT_ENCODER_REPO

        logger.info(f"Loading HunyuanVideo text encoder from {text_encoder_path}")
        text_encoder = TextEncoder(
            text_encoder_type="llm",
            tokenizer_type="llm",
            text_encoder_path=text_encoder_path,
            max_length=1000,
            text_encoder_precision="fp16",
            prompt_template=PROMPT_TEMPLATE["li-dit-encode-image-json"],
            prompt_template_video=PROMPT_TEMPLATE["li-dit-encode-video-json"],
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
            logger=loguru.logger,
            device=device,
        )

        self.text_encoders = [text_encoder]
        self.text_encoder_1 = text_encoder
        self.text_encoder_2 = None

        self.tokenizers = [text_encoder.tokenizer]
        self.tokenizer_1 = text_encoder.tokenizer

    def load_text_tokenizer(self):
        """
        Tokenization is handled by the custom TextEncoder stack.
        """
        return

    def load_vae(self, move_to_device: bool = True):
        """
        Load VAE from the dedicated HunyuanVideo VAE repo.
        """
        from transformers.utils import ContextManagers

        from simpletuner.helpers.models.common import deepspeed_zero_init_disabled_context_manager

        logger.info(f"Loading {self.AUTOENCODER_CLASS.__name__} from {self.VAE_REPO}")
        self.vae = None
        self.config.vae_kwargs = {
            "pretrained_model_name_or_path": self.VAE_REPO,
            "subfolder": None,  # Direct repo, no subfolder
            "revision": self.config.revision,
            "force_upcast": False,
            "variant": self.config.variant,
        }
        if getattr(self.config, "vae_enable_patch_conv", False):
            logger.info("Enabling VAE patch-based convolution for HunyuanVideo VAE.")
            self.config.vae_kwargs["enable_patch_conv"] = True
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            self.vae = self.AUTOENCODER_CLASS.from_pretrained(**self.config.vae_kwargs)
        if self.vae is None:
            raise ValueError(f"Could not load VAE from {self.VAE_REPO}.")
        if self.config.vae_enable_tiling and hasattr(self.vae, "enable_tiling"):
            logger.info("Enabling VAE tiling.")
            self.vae.enable_tiling()
        if move_to_device and self.vae.device != self.accelerator.device:
            _vae_dtype = torch.bfloat16
            if hasattr(self.config, "vae_dtype") and self.config.vae_dtype == "fp32":
                _vae_dtype = torch.float32
            self.vae.to(self.accelerator.device, dtype=_vae_dtype)
        self.AUTOENCODER_SCALING_FACTOR = getattr(self.vae.config, "scaling_factor", 1.0)

    def get_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True):
        """
        Return the HunyuanVideo inference pipeline with pre-loaded components.

        Unlike the base class implementation which uses from_pretrained(),
        we instantiate the pipeline directly since our transformer repos
        don't contain model_index.json.
        """
        active_pipelines = getattr(self, "pipelines", {})
        if pipeline_type in active_pipelines:
            pipeline = active_pipelines[pipeline_type]
            if load_base_model and self.model is not None:
                pipeline.transformer = self.unwrap_model(self.model)
            return pipeline

        device = self.accelerator.device
        flow_shift = getattr(self.config, "flow_schedule_shift", 7.0)
        guidance_scale = getattr(self.config, "validation_guidance", 6.0)

        # Create scheduler for inference
        scheduler = FlowMatchDiscreteScheduler(
            num_train_timesteps=1000,
            shift=flow_shift,
        )

        # Load ByT5 glyph encoder for the pipeline
        byt5_kwargs, prompt_format = HunyuanVideo_1_5_Pipeline._load_byt5(
            cached_folder=None,
            glyph_byT5_v2=True,
            byt5_max_length=256,
            device=device,
        )
        byt5_model = byt5_kwargs.get("byt5_model") if byt5_kwargs else None
        byt5_tokenizer = byt5_kwargs.get("byt5_tokenizer") if byt5_kwargs else None

        # Instantiate the pipeline directly with pre-loaded components
        pipeline = HunyuanVideo_1_5_Pipeline(
            vae=self.unwrap_model(self.vae) if self.vae is not None else None,
            text_encoder=self.text_encoder_1,
            transformer=self.unwrap_model(self.model) if load_base_model and self.model is not None else None,
            scheduler=scheduler,
            text_encoder_2=getattr(self, "text_encoder_2", None),
            flow_shift=flow_shift,
            guidance_scale=guidance_scale,
            glyph_byT5_v2=True,
            byt5_model=byt5_model,
            byt5_tokenizer=byt5_tokenizer,
            byt5_max_length=256,
            prompt_format=prompt_format,
            execution_device=device,
            vision_encoder=None,
            enable_offloading=False,
        )

        if not hasattr(self, "pipelines"):
            self.pipelines = {}
        self.pipelines[pipeline_type] = pipeline
        return pipeline

    def _prepare_cond_latents(self, cond_latents: Optional[torch.Tensor], latents: torch.Tensor, task_type: str):
        if cond_latents is not None:
            if cond_latents.dim() == 4:
                cond_latents = cond_latents.unsqueeze(2)
            cond_latents = cond_latents.to(device=latents.device, dtype=latents.dtype)
        else:
            cond_latents = torch.zeros_like(latents, device=latents.device, dtype=latents.dtype)

        if task_type == "i2v":
            cond_latents = cond_latents.repeat(1, 1, latents.shape[2], 1, 1)
            cond_latents[:, :, 1:, :, :] = 0.0

        mask_zeros = torch.zeros(
            latents.shape[0],
            1,
            latents.shape[2],
            latents.shape[3],
            latents.shape[4],
            device=latents.device,
            dtype=latents.dtype,
        )
        mask_ones = torch.ones_like(mask_zeros)
        mask = torch.zeros(latents.shape[2], device=latents.device, dtype=latents.dtype)
        if task_type == "i2v":
            mask[0] = 1.0
        multitask_mask = mask
        mask_concat = merge_tensor_by_mask(mask_zeros, mask_ones, mask=multitask_mask, dim=2)
        return torch.concat([cond_latents, mask_concat], dim=1)

    def _format_text_embedding(self, text_embedding: dict):
        prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask = text_embedding
        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "negative_prompt_embeds": negative_prompt_embeds,
            "negative_prompt_attention_mask": negative_prompt_attention_mask,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        return {
            "prompt_embeds": text_embedding.get("prompt_embeds"),
            "prompt_mask": text_embedding.get("prompt_attention_mask"),
            "negative_prompt_embeds": text_embedding.get("negative_prompt_embeds"),
            "negative_prompt_mask": text_embedding.get("negative_prompt_attention_mask"),
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        negative_embeds = text_embedding.get("negative_prompt_embeds") or text_embedding.get("prompt_embeds")
        negative_attention_mask = (
            text_embedding.get("negative_prompt_attention_mask")
            or text_embedding.get("prompt_attention_mask")
            or text_embedding.get("attention_mask")
        )
        if negative_embeds is None:
            raise ValueError("Negative prompt embeddings are missing for HunyuanVideo.")
        if negative_embeds.dim() == 2:
            negative_embeds = negative_embeds.unsqueeze(0)
        if negative_attention_mask is not None and negative_attention_mask.dim() == 1:
            negative_attention_mask = negative_attention_mask.unsqueeze(0)
        return {
            "negative_prompt_embeds": negative_embeds,
            "negative_prompt_mask": negative_attention_mask,
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        negative = prompts if is_negative_prompt else ["" for _ in prompts]
        prompt_input = "" if is_negative_prompt else prompts
        return pipeline.encode_prompt(
            prompt=prompt_input,
            device=self.accelerator.device,
            num_videos_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt=negative,
            data_type="video",
        )

    def _prepare_force_keep_mask(self, latents: torch.Tensor, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        transformer = self.unwrap_model(model=self.model)
        patch_size = transformer.config.patch_size if hasattr(transformer, "config") else [1, 2, 2]
        if mask.dim() == 5:
            b, c, t, h, w = mask.shape
            t_tokens = t // patch_size[0]
            h_tokens = h // patch_size[1]
            w_tokens = w // patch_size[2]
            mask = (mask.mean(1, keepdim=True) + 1) / 2
            mask = F.interpolate(
                mask,
                size=(t_tokens, h_tokens, w_tokens),
                mode="trilinear",
                align_corners=False,
            )
            mask = mask.squeeze(1).flatten(1) > 0.5
        elif mask.dim() > 2:
            mask = mask.view(mask.shape[0], -1).to(dtype=torch.bool)
        return mask

    def model_predict(self, prepared_batch):
        latents = prepared_batch["noisy_latents"].to(self.config.weight_dtype)
        if latents.dim() != 5:
            raise ValueError(f"Expected 5D video latents, got shape {latents.shape}")
        bs, _, frames, _, _ = latents.shape

        encoder_hidden_states = prepared_batch["encoder_hidden_states"].to(self.config.weight_dtype)
        attention_mask = prepared_batch.get("encoder_attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=latents.device)
        else:
            attention_mask = prepared_batch.get("attention_mask")

        prompt_embeds_2 = prepared_batch.get("encoder_hidden_states_2")
        if prompt_embeds_2 is not None:
            prompt_embeds_2 = prompt_embeds_2.to(self.config.weight_dtype)
        attention_mask_2 = prepared_batch.get("encoder_attention_mask_2")
        if attention_mask_2 is not None:
            attention_mask_2 = attention_mask_2.to(device=latents.device)

        timesteps = prepared_batch["timesteps"]
        task_type = "i2v" if prepared_batch.get("is_i2v_data", False) else "t2v"

        cond_latents = prepared_batch.get("conditioning_latents")
        if task_type == "i2v" and cond_latents is None:
            raise ValueError("HunyuanVideo i2v training requires conditioning_latents in the batch.")
        cond_latents = self._prepare_cond_latents(cond_latents, latents, task_type)
        latent_model_input = torch.cat([latents, cond_latents], dim=1)

        force_keep_mask = prepared_batch.get("force_keep_mask")
        if force_keep_mask is None and prepared_batch.get("conditioning_pixel_values") is not None:
            force_keep_mask = self._prepare_force_keep_mask(latents, prepared_batch["conditioning_pixel_values"])

        transformer_kwargs = {
            "hidden_states": latent_model_input,
            "timestep": timesteps,
            "text_states": encoder_hidden_states,
            "text_states_2": prompt_embeds_2,
            "encoder_attention_mask": attention_mask,
            "vision_states": prepared_batch.get("vision_states"),
            "mask_type": task_type,
            "force_keep_mask": force_keep_mask,
            "return_dict": False,
        }
        if attention_mask_2 is not None:
            transformer_kwargs["attention_kwargs"] = {"encoder_attention_mask_2": attention_mask_2}

        model_pred = self.model(**transformer_kwargs)[0]
        return {
            "model_prediction": model_pred,
        }

    def tread_init(self):
        """
        Initialize TREAD routing for HunyuanVideo.
        """
        tread_cfg = getattr(self.config, "tread_config", None)
        if not isinstance(tread_cfg, dict) or tread_cfg == {} or tread_cfg.get("routes") is None:
            raise ValueError("TREAD training requires a non-empty tread_config with routes.")

        self.unwrap_model(model=self.model).set_router(
            TREADRouter(
                seed=getattr(self.config, "seed", None) or 42,
                device=self.accelerator.device,
            ),
            tread_cfg["routes"],
        )

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        """
        Populate pipeline kwargs for validation runs.
        """
        default_resolution = getattr(self.config, "validation_resolution", None)
        aspect_ratio = pipeline_kwargs.get("aspect_ratio")
        if aspect_ratio is None:
            if isinstance(default_resolution, str) and "x" in default_resolution:
                width, height = default_resolution.lower().split("x")
                aspect_ratio = f"{width}:{height}"
            else:
                aspect_ratio = "1280:720"
        pipeline_kwargs["aspect_ratio"] = aspect_ratio
        pipeline_kwargs.setdefault("video_length", getattr(self.config, "validation_num_video_frames", 25) or 25)
        pipeline_kwargs.setdefault("num_inference_steps", getattr(self.config, "validation_num_inference_steps", 30) or 30)
        pipeline_kwargs.setdefault("guidance_scale", getattr(self.config, "validation_guidance", 6.0))
        pipeline_kwargs.setdefault("flow_shift", getattr(self.config, "flow_schedule_shift", 7.0))
        pipeline_kwargs.setdefault("enable_sr", True)
        return pipeline_kwargs

    def get_transforms(self, dataset_type: str = "image"):
        if dataset_type == DatasetType.VIDEO.value or dataset_type == "video":
            return super().get_transforms(dataset_type="video")
        return super().get_transforms(dataset_type=dataset_type)


ModelRegistry.register("hunyuanvideo", HunyuanVideo)
