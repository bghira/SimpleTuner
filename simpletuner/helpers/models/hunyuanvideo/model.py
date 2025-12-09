# HunyuanVideo 1.5 integration (AGPL-3.0-or-later), SimpleTuner © 2025

import logging
import os
from typing import Dict, Optional

import torch
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from transformers import (
    ByT5Tokenizer,
    Qwen2_5_VLTextModel,
    Qwen2Tokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
)

from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.models.common import ModelTypes, PipelineTypes, PredictionTypes, VideoModelFoundation
from simpletuner.helpers.models.hunyuanvideo.autoencoder_hv15 import AutoencoderKLHunyuanVideo as AutoencoderKLHunyuanVideo15
from simpletuner.helpers.models.hunyuanvideo.commons import PIPELINE_CONFIGS, TRANSFORMER_VERSION_TO_SR_VERSION
from simpletuner.helpers.models.hunyuanvideo.pipeline import HunyuanVideo15Pipeline
from simpletuner.helpers.models.hunyuanvideo.pipeline_i2v import HunyuanVideo15ImageToVideoPipeline
from simpletuner.helpers.models.hunyuanvideo.transformer import HunyuanVideo15Transformer3DModel
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.training.multi_process import should_log

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
    AUTOENCODER_CLASS = AutoencoderKLHunyuanVideo15
    LATENT_CHANNEL_COUNT = 32
    DEFAULT_NOISE_SCHEDULER = "flow_match_euler"
    MODEL_CLASS = HunyuanVideo15Transformer3DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: HunyuanVideo15Pipeline,
        PipelineTypes.IMG2VIDEO: HunyuanVideo15ImageToVideoPipeline,
    }
    DEFAULT_PIPELINE_TYPE = PipelineTypes.TEXT2IMG
    DEFAULT_MODEL_FLAVOUR = "t2v-480p"
    HUGGINGFACE_PATHS: Dict[str, str] = {
        "t2v-480p": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "t2v-720p": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        "t2v-480p-distilled": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled",
        "i2v-480p": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
        "i2v-720p": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v",
        "i2v-480p-distilled": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_distilled",
        "i2v-720p-distilled": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled",
    }
    MODEL_LICENSE = "agpl-3.0"

    # Component repositories - direct loading without subfolders
    VAE_REPO = "DiffusersVersionsOfModels/HunyuanVideo-1.5-vae"
    GLYPH_BYT5_REPO = "DiffusersVersionsOfModels/Glyph-ByT5"
    TEXT_ENCODER_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
    VISION_ENCODER_REPO = "google/siglip-so400m-patch14-384"

    # Transformer version mapping for pipeline configs
    TRANSFORMER_VERSIONS: Dict[str, str] = {
        "t2v-480p": "480p_t2v",
        "t2v-720p": "720p_t2v",
        "t2v-480p-distilled": "480p_t2v",
        "i2v-480p": "480p_i2v",
        "i2v-720p": "720p_i2v",
        "i2v-480p-distilled": "480p_i2v",
        "i2v-720p-distilled": "720p_i2v",
    }
    STRICT_I2V_FLAVOURS = ("i2v-480p", "i2v-720p", "i2v-480p-distilled", "i2v-720p-distilled")

    # Only required to satisfy encode_text_batch checks; loading is handled manually.
    TEXT_ENCODER_CONFIGURATION = {"text_encoder": {"name": "Hunyuan LLM"}}
    DEFAULT_LORA_TARGET = [
        "img_attn_q",
        "img_attn_k",
        "img_attn_v",
        "img_attn_proj",
        "txt_attn_q",
        "txt_attn_k",
        "txt_attn_v",
        "txt_attn_proj",
        "linear1_q",
        "linear1_k",
        "linear1_v",
        "linear1_mlp",
    ]

    def __init__(self, config: dict, accelerator):
        super().__init__(config, accelerator)
        self._transformer_version = self._resolve_transformer_version()
        self._sr_version = TRANSFORMER_VERSION_TO_SR_VERSION.get(self._transformer_version)
        if getattr(self.config, "pretrained_transformer_subfolder", None) is None:
            # Default to the standard diffusers transformer folder layout.
            self.config.pretrained_transformer_subfolder = self.MODEL_SUBFOLDER
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

    def _is_i2v_like_flavour(self) -> bool:
        flavour = getattr(self.config, "model_flavour", None) or self.DEFAULT_MODEL_FLAVOUR
        if not flavour:
            return False
        try:
            return str(flavour).strip().lower().startswith("i2v")
        except Exception:
            return False

    def requires_conditioning_dataset(self) -> bool:
        return self._is_i2v_like_flavour() or super().requires_conditioning_dataset()

    def requires_conditioning_latents(self) -> bool:
        return self._is_i2v_like_flavour()

    def setup_training_noise_schedule(self):
        self.noise_schedule = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=self.config.flow_schedule_shift,
        )
        return self.config, self.noise_schedule

    def load_text_encoder(self, move_to_device: bool = True):
        """
        Load the Qwen2.5 VL text encoder and ByT5 glyph encoder.
        """
        device = self.accelerator.device if move_to_device else torch.device("cpu")
        qwen_path = getattr(self.config, "hunyuan_text_encoder_path", None) or self.TEXT_ENCODER_REPO

        logger.info(f"Loading HunyuanVideo text encoder from {qwen_path}")
        tokenizer = Qwen2Tokenizer.from_pretrained(qwen_path)
        text_encoder = Qwen2_5_VLTextModel.from_pretrained(qwen_path, torch_dtype=torch.bfloat16)
        text_encoder.requires_grad_(False)
        if move_to_device:
            text_encoder = text_encoder.to(device)

        logger.info(f"Loading Glyph ByT5 encoder from {self.GLYPH_BYT5_REPO}")
        byt5_tokenizer = ByT5Tokenizer.from_pretrained(self.GLYPH_BYT5_REPO)
        byt5_model = T5EncoderModel.from_pretrained(self.GLYPH_BYT5_REPO, torch_dtype=torch.bfloat16)
        byt5_model.requires_grad_(False)
        if move_to_device:
            byt5_model = byt5_model.to(device)

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_encoder_2 = byt5_model
        self.tokenizer_2 = byt5_tokenizer

        # Maintain attributes expected by the training stack.
        self.text_encoders = [text_encoder]
        self.text_encoder_1 = text_encoder
        self.tokenizers = [tokenizer]
        self.tokenizer_1 = tokenizer
        self._image_encoder = None
        self._image_processor = None

    def _load_image_encoder(self, move_to_device: bool = True):
        if not hasattr(self, "_image_encoder"):
            self._image_encoder = None
            self._image_processor = None
        if self._image_encoder is not None and self._image_processor is not None:
            return self._image_encoder, self._image_processor

        device = self.accelerator.device if move_to_device else torch.device("cpu")
        repo = getattr(self.config, "hunyuan_image_encoder_path", None) or self.VISION_ENCODER_REPO
        logger.info(f"Loading SigLIP vision encoder from {repo}")
        feature_extractor = SiglipImageProcessor.from_pretrained(repo)
        image_encoder = SiglipVisionModel.from_pretrained(repo, torch_dtype=torch.bfloat16)
        image_encoder.requires_grad_(False)
        if move_to_device:
            image_encoder = image_encoder.to(device)

        self._image_encoder = image_encoder
        self._image_processor = feature_extractor
        return image_encoder, feature_extractor

    def load_text_tokenizer(self):
        """
        Tokenizers are loaded alongside the text encoders.
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
            "enable_temporal_roll": getattr(self.config, "vae_enable_temporal_roll", False),
        }
        if getattr(self.config, "vae_enable_patch_conv", False):
            logger.info("Enabling VAE patch-based convolution for HunyuanVideo VAE.")
            self.config.vae_kwargs["enable_patch_conv"] = True
        if getattr(self.config, "vae_enable_temporal_roll", False):
            logger.info("Enabling temporal rolling for HunyuanVideo VAE to reduce VRAM.")
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

        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=flow_shift)
        guider = ClassifierFreeGuidance(guidance_scale=guidance_scale)

        transformer = self.unwrap_model(self.model) if load_base_model and self.model is not None else None
        vae = self.unwrap_model(self.vae) if self.vae is not None else None

        pipeline_kwargs = {
            "text_encoder": self.text_encoder,
            "tokenizer": self.tokenizer,
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder_2": self.text_encoder_2,
            "tokenizer_2": self.tokenizer_2,
            "guider": guider,
        }

        if pipeline_type == PipelineTypes.IMG2VIDEO:
            image_encoder, feature_extractor = self._load_image_encoder(move_to_device=True)
            pipeline = HunyuanVideo15ImageToVideoPipeline(
                image_encoder=image_encoder,
                feature_extractor=feature_extractor,
                **pipeline_kwargs,
            )
        else:
            pipeline = HunyuanVideo15Pipeline(**pipeline_kwargs)

        pipeline.to(device)

        if not hasattr(self, "pipelines"):
            self.pipelines = {}
        self.pipelines[pipeline_type] = pipeline
        return pipeline

    def _prepare_cond_latents(self, cond_latents: Optional[torch.Tensor], latents: torch.Tensor, task_type: str):
        batch, _, frames, height, width = latents.shape
        dtype = latents.dtype
        device = latents.device

        if cond_latents is None:
            cond = torch.zeros(batch, self.LATENT_CHANNEL_COUNT, frames, height, width, device=device, dtype=dtype)
        else:
            if cond_latents.dim() == 4:
                cond_latents = cond_latents.unsqueeze(2)
            cond = cond_latents.to(device=device, dtype=dtype).repeat(1, 1, frames, 1, 1)
            if task_type == "i2v":
                cond[:, :, 1:, :, :] = 0.0

        mask = torch.zeros(batch, 1, frames, height, width, device=device, dtype=dtype)
        if task_type == "i2v":
            mask[:, :, 0, :, :] = 1.0
        return cond, mask

    def _format_text_embedding(self, text_embedding: dict):
        prompt_embeds, prompt_attention_mask, prompt_embeds_2, prompt_attention_mask_2 = text_embedding
        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "prompt_embeds_2": prompt_embeds_2,
            "prompt_attention_mask_2": prompt_attention_mask_2,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        return {
            "prompt_embeds": text_embedding.get("prompt_embeds"),
            "prompt_embeds_mask": text_embedding.get("prompt_attention_mask"),
            "prompt_embeds_2": text_embedding.get("prompt_embeds_2"),
            "prompt_embeds_mask_2": text_embedding.get("prompt_attention_mask_2"),
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        return {
            "negative_prompt_embeds": text_embedding.get("prompt_embeds"),
            "negative_prompt_embeds_mask": text_embedding.get("prompt_attention_mask"),
            "negative_prompt_embeds_2": text_embedding.get("prompt_embeds_2"),
            "negative_prompt_embeds_mask_2": text_embedding.get("prompt_attention_mask_2"),
        }

    def collate_prompt_embeds(self, text_encoder_output: dict) -> dict:
        """
        Stack prompt embeddings and masks for the current batch.
        """

        def _collate(field):
            tensors = [entry[field] for entry in text_encoder_output if field in entry]
            if not tensors:
                return None
            first = tensors[0]
            if first.dim() == 3:
                return torch.cat(tensors, dim=0)
            return torch.stack(tensors, dim=0)

        collated = {}
        if text_encoder_output and isinstance(text_encoder_output, list):
            prompt_embeds = _collate("prompt_embeds")
            if prompt_embeds is not None:
                collated["prompt_embeds"] = prompt_embeds
            prompt_masks = _collate("prompt_attention_mask")
            if prompt_masks is not None:
                collated["attention_masks"] = prompt_masks
            prompt_embeds_2 = _collate("prompt_embeds_2")
            if prompt_embeds_2 is not None:
                collated["prompt_embeds_2"] = prompt_embeds_2
            prompt_masks_2 = _collate("prompt_attention_mask_2")
            if prompt_masks_2 is not None:
                collated["attention_masks_2"] = prompt_masks_2
        return collated

    def prepare_batch_conditions(self, batch: dict, state: dict) -> dict:
        batch = super().prepare_batch_conditions(batch=batch, state=state)
        text_output = batch.get("text_encoder_output") or {}

        prompt_embeds = batch.get("prompt_embeds") or text_output.get("prompt_embeds")
        if prompt_embeds is not None:
            batch["encoder_hidden_states"] = prompt_embeds

        attention_masks = batch.get("attention_masks") or text_output.get("attention_masks")
        if attention_masks is not None:
            batch["encoder_attention_mask"] = attention_masks

        prompt_embeds_2 = batch.get("prompt_embeds_2") or text_output.get("prompt_embeds_2")
        if prompt_embeds_2 is not None:
            batch["encoder_hidden_states_2"] = prompt_embeds_2

        attention_masks_2 = batch.get("attention_masks_2") or text_output.get("attention_masks_2")
        if attention_masks_2 is not None:
            batch["encoder_attention_mask_2"] = attention_masks_2
        return batch

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        prompt_list = prompts if isinstance(prompts, list) else [prompts]
        prompt_embeds, prompt_attention_mask, prompt_embeds_2, prompt_attention_mask_2 = pipeline.encode_prompt(
            prompt=prompt_list,
            device=self.accelerator.device,
            batch_size=len(prompt_list),
            num_videos_per_prompt=1,
        )
        return prompt_embeds, prompt_attention_mask, prompt_embeds_2, prompt_attention_mask_2

    def model_predict(self, prepared_batch):
        latents = prepared_batch["noisy_latents"].to(self.config.weight_dtype)
        if latents.dim() != 5:
            raise ValueError(f"Expected 5D video latents, got shape {latents.shape}")

        encoder_hidden_states = prepared_batch["encoder_hidden_states"].to(self.config.weight_dtype)
        encoder_attention_mask = prepared_batch.get("encoder_attention_mask")
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(device=latents.device, dtype=torch.bool)

        encoder_hidden_states_2 = prepared_batch.get("encoder_hidden_states_2")
        if encoder_hidden_states_2 is not None:
            encoder_hidden_states_2 = encoder_hidden_states_2.to(self.config.weight_dtype)
        else:
            encoder_hidden_states_2 = torch.zeros(
                encoder_hidden_states.shape[0],
                1,
                self.model.config.hidden_size,
                device=latents.device,
                dtype=self.config.weight_dtype,
            )

        encoder_attention_mask_2 = prepared_batch.get("encoder_attention_mask_2")
        if encoder_attention_mask_2 is not None:
            encoder_attention_mask_2 = encoder_attention_mask_2.to(device=latents.device, dtype=torch.bool)
        else:
            encoder_attention_mask_2 = torch.zeros(
                encoder_hidden_states_2.shape[0],
                encoder_hidden_states_2.shape[1],
                device=latents.device,
                dtype=torch.bool,
            )

        timesteps = prepared_batch["timesteps"]
        wants_i2v_batch = bool(prepared_batch.get("is_i2v_data", False))
        is_i2v_model = self._is_i2v_like_flavour()

        if wants_i2v_batch and not is_i2v_model and should_log() and not getattr(self, "_warned_spurious_i2v_batch", False):
            logger.warning(
                "Received an i2v-labelled batch for a t2v flavour; ignoring the flag and continuing with t2v training."
            )
            self._warned_spurious_i2v_batch = True

        if is_i2v_model and prepared_batch.get("conditioning_latents") is None:
            raise ValueError("HunyuanVideo i2v training requires conditioning_latents in the batch.")

        task_type = "i2v" if is_i2v_model else "t2v"
        cond_latents, cond_mask = self._prepare_cond_latents(prepared_batch.get("conditioning_latents"), latents, task_type)
        latent_model_input = torch.cat([latents, cond_latents, cond_mask], dim=1)

        batch_size = latents.shape[0]
        model_config = getattr(self.model, "config", None)
        vision_tokens = getattr(self.config, "vision_num_semantic_tokens", 729)
        if model_config is not None:
            vision_tokens = getattr(model_config, "vision_num_semantic_tokens", vision_tokens)
        vision_dim = getattr(self.config, "vision_states_dim", 1152)
        if model_config is not None:
            vision_dim = getattr(model_config, "image_embed_dim", vision_dim)
        image_embeds = prepared_batch.get("vision_states")
        if image_embeds is None:
            image_embeds = torch.zeros(
                batch_size,
                vision_tokens,
                vision_dim,
                device=latents.device,
                dtype=latents.dtype,
            )

        model_pred = self.model(
            hidden_states=latent_model_input,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states_2=encoder_hidden_states_2,
            encoder_attention_mask_2=encoder_attention_mask_2,
            image_embeds=image_embeds,
            return_dict=False,
        )[0]
        return {
            "model_prediction": model_pred,
        }

    def tread_init(self):
        raise NotImplementedError("TREAD routing is not supported for the diffusers HunyuanVideo 1.5 transformer.")

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
