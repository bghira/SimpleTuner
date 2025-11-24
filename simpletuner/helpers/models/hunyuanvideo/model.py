# HunyuanVideo 1.5 integration (AGPL-3.0-or-later), SimpleTuner © 2025

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import loguru
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download

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
    MODEL_SUBFOLDER = "transformer/480p_t2v"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: HunyuanVideo_1_5_Pipeline,
        PipelineTypes.IMG2VIDEO: HunyuanVideo_1_5_Pipeline,
    }
    DEFAULT_PIPELINE_TYPE = PipelineTypes.TEXT2IMG
    DEFAULT_MODEL_FLAVOUR = "t2v-480p"
    HUGGINGFACE_PATHS: Dict[str, str] = {
        "t2v-480p": "tencent/HunyuanVideo-1.5",
        "t2v-720p": "tencent/HunyuanVideo-1.5",
        "i2v-480p": "tencent/HunyuanVideo-1.5",
        "i2v-720p": "tencent/HunyuanVideo-1.5",
    }
    MODEL_LICENSE = "agpl-3.0"

    TRANSFORMER_SUBFOLDERS: Dict[str, str] = {
        "t2v-480p": "transformer/480p_t2v",
        "t2v-720p": "transformer/720p_t2v",
        "i2v-480p": "transformer/480p_i2v",
        "i2v-720p": "transformer/720p_i2v",
    }
    UPSAMPLER_SUBFOLDERS: Dict[str, str] = {
        "t2v-480p": "upsampler/720p_sr_distilled",
        "t2v-720p": "upsampler/1080p_sr_distilled",
        "i2v-480p": "upsampler/720p_sr_distilled",
        "i2v-720p": "upsampler/1080p_sr_distilled",
    }
    STRICT_I2V_FLAVOURS = ("i2v-480p", "i2v-720p")

    # Only required to satisfy encode_text_batch checks; loading is handled manually.
    TEXT_ENCODER_CONFIGURATION = {"text_encoder": {"name": "Hunyuan LLM"}}

    def __init__(self, config: dict, accelerator):
        super().__init__(config, accelerator)
        self._transformer_version = self._resolve_transformer_version()
        self._sr_version = TRANSFORMER_VERSION_TO_SR_VERSION.get(self._transformer_version)
        if not getattr(self.config, "pretrained_transformer_subfolder", None):
            self.config.pretrained_transformer_subfolder = self.TRANSFORMER_SUBFOLDERS.get(
                self.config.model_flavour, self.MODEL_SUBFOLDER
            )
        if getattr(self.config, "flow_schedule_shift", None) is None:
            default_cfg = PIPELINE_CONFIGS.get(self._transformer_version, {})
            self.config.flow_schedule_shift = default_cfg.get("flow_shift", 7.0)
        if getattr(self.config, "validation_guidance", None) is None:
            default_cfg = PIPELINE_CONFIGS.get(self._transformer_version, {})
            self.config.validation_guidance = default_cfg.get("guidance_scale", 6.0)

    def _resolve_transformer_version(self) -> str:
        flavour = getattr(self.config, "model_flavour", self.DEFAULT_MODEL_FLAVOUR) or self.DEFAULT_MODEL_FLAVOUR
        if flavour not in self.TRANSFORMER_SUBFOLDERS:
            raise ValueError(
                f"Unsupported HunyuanVideo flavour '{flavour}'. Expected one of {list(self.TRANSFORMER_SUBFOLDERS)}"
            )
        return self.TRANSFORMER_SUBFOLDERS[flavour].split("/", maxsplit=1)[-1]

    def setup_training_noise_schedule(self):
        self.noise_schedule = FlowMatchDiscreteScheduler(
            shift=self.config.flow_schedule_shift,
            reverse=True,
            solver="euler",
        )
        return self.config, self.noise_schedule

    def _find_cached_snapshot(self, repo_id: str) -> Optional[str]:
        """
        Locate an existing Hugging Face snapshot on disk for the given repo.
        """
        try:
            repo_cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{repo_id.replace('/', '--')}"
            ref_main = repo_cache / "refs" / "main"
            if ref_main.exists():
                snapshot_hash = ref_main.read_text().strip()
                snap_dir = repo_cache / "snapshots" / snapshot_hash
                if snap_dir.exists():
                    return str(snap_dir)
        except Exception:
            return None
        return None

    def _resolve_model_root(
        self,
        pretrained_model_path: str,
        allow_patterns: Optional[List[str]] = None,
        required_subdir: Optional[str] = None,
    ) -> Optional[str]:
        """
        Ensure the checkpoint assets exist locally; download requested parts when given an HF repo id.
        """
        candidate_paths: List[str] = []
        if pretrained_model_path and os.path.isdir(pretrained_model_path):
            candidate_paths.append(pretrained_model_path)
        cached = self._find_cached_snapshot(pretrained_model_path)
        if cached:
            candidate_paths.append(cached)

        def _has_required(path: str) -> bool:
            return required_subdir is None or os.path.exists(os.path.join(path, required_subdir))

        for path in candidate_paths:
            if _has_required(path):
                return path

        try:
            cache_dir = snapshot_download(
                repo_id=pretrained_model_path,
                allow_patterns=allow_patterns
                or [
                    "text_encoder/*",
                    "text_encoder/llm/*",
                    "vision_encoder/*",
                    "vision_encoder/siglip/*",
                ],
            )
            if _has_required(cache_dir):
                return cache_dir
        except Exception as exc:
            logger.warning(f"Failed to resolve model root for {pretrained_model_path}: {exc}")
            pass

        return None

    def load_text_encoder(self, move_to_device: bool = True):
        """
        Use the upstream helper to load the custom TextEncoder (llm) stack.
        """
        device = self.accelerator.device if move_to_device else torch.device("cpu")
        override_text = getattr(self.config, "hunyuan_text_encoder_path", None)
        override_text_repo = getattr(self.config, "hunyuan_text_encoder_repo", None)
        override_text_subpath = getattr(self.config, "hunyuan_text_encoder_subpath", None)

        subpath = override_text_subpath or "text_encoder/llm"
        fallback_repo = override_text_repo or "Qwen/Qwen2.5-VL-7B-Instruct"
        base_model_path = getattr(self.config, "pretrained_model_name_or_path", None)

        text_encoder_path = None
        # 1. Try override path if provided
        if override_text and os.path.exists(override_text):
            text_encoder_path = override_text

        # 2. Try base model path with subpath
        if not text_encoder_path and base_model_path:
            resolved_base = self._resolve_model_root(
                override_text or base_model_path,
                allow_patterns=["text_encoder/*", "text_encoder/llm/*"],
                required_subdir=subpath,
            )
            if resolved_base:
                check_path = os.path.join(resolved_base, subpath)
                if os.path.exists(check_path):
                    text_encoder_path = check_path

        # 3. Try fallback repo
        if not text_encoder_path and fallback_repo:
            # For Qwen fallback, we expect files at ROOT, unless user overrode subpath
            fallback_subpath = subpath
            if fallback_repo == "Qwen/Qwen2.5-VL-7B-Instruct" and not override_text_subpath:
                fallback_subpath = None

            resolved_fallback = self._resolve_model_root(
                fallback_repo, allow_patterns=None, required_subdir=fallback_subpath
            )

            if resolved_fallback:
                if fallback_subpath:
                    check_path = os.path.join(resolved_fallback, fallback_subpath)
                    if os.path.exists(check_path):
                        text_encoder_path = check_path
                else:
                    text_encoder_path = resolved_fallback

        if text_encoder_path is None:
            msg = (
                f"Required assets ({subpath}) not found under {base_model_path} or {fallback_repo}. "
                "Set HUNYUANVIDEO_TEXT_ENCODER_PATH or HUNYUANVIDEO_TEXT_ENCODER_REPO to a downloaded text encoder "
                "or follow checkpoints-download.md to fetch the dependencies."
            )
            loguru.logger.error(msg)
            raise FileNotFoundError(msg)

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

    def load_text_tokenizer(self):
        """
        Tokenization is handled by the custom TextEncoder stack.
        """
        return

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
        if (
            getattr(self.config, "tread_config", None) is None
            or getattr(self.config, "tread_config", None) is {}
            or getattr(self.config, "tread_config", {}).get("routes", None) is None
        ):
            raise ValueError("TREAD training requires a non-empty tread_config with routes.")

        self.unwrap_model(model=self.model).set_router(
            TREADRouter(
                seed=getattr(self.config, "seed", None) or 42,
                device=self.accelerator.device,
            ),
            self.config.tread_config["routes"],
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
