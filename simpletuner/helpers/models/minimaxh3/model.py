import logging
import os
from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.video_processor import VideoProcessor
from PIL import Image
from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from simpletuner.helpers.acceleration import (
    AccelerationBackend,
    AccelerationPreset,
    get_bitsandbytes_presets,
    get_deepspeed_presets,
    get_quanto_presets,
    get_sdnq_presets,
    get_torchao_presets,
)
from simpletuner.helpers.models.common import (
    ModelTypes,
    PipelineTypes,
    PredictionTypes,
    TextEmbedCacheKey,
    VideoModelFoundation,
)
from simpletuner.helpers.models.minimaxh3.autoencoder import AutoencoderKLMiniMaxH3
from simpletuner.helpers.models.minimaxh3.autoencoder_audio import AutoencoderKLMiniMaxH3Audio
from simpletuner.helpers.models.minimaxh3.encoders import MINIMAX_H3_DEFAULT_MAX_TEXT_LENGTH, MiniMaxH3TextEncoderStep
from simpletuner.helpers.models.minimaxh3.modular_blocks_minimax_h3 import MiniMaxH3Blocks, MiniMaxH3Ref2VABlocks
from simpletuner.helpers.models.minimaxh3.packing import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_FPS,
    MINIMAX_H3_KEYFRAME_NOISE_AUG,
    MINIMAX_H3_PIXEL_MEAN,
    MINIMAX_H3_PIXEL_STD,
    MINIMAX_H3_TEXT_TAG,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timestep_intervals,
    build_row_timesteps,
    patchify_video_latents,
    unpatchify_video_tokens,
)
from simpletuner.helpers.models.minimaxh3.pipeline import MiniMaxH3Pipeline
from simpletuner.helpers.models.minimaxh3.pipeline_ref import MiniMaxH3Ref2VAPipeline
from simpletuner.helpers.models.minimaxh3.scheduler import MiniMaxH3Scheduler
from simpletuner.helpers.models.minimaxh3.sparse_attention import initialize_minimax_h3_flex_attention
from simpletuner.helpers.models.minimaxh3.transformer import MiniMaxH3Transformer3DModel
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.musubi_block_swap import apply_musubi_pretrained_defaults
from simpletuner.helpers.training.flow_match import fix_flow_match_euler_schedule_bounds
from simpletuner.helpers.training.lora_format import (
    PEFTLoRAFormat,
    collect_lora_ranks,
    detect_state_dict_format,
    normalize_lora_format,
)
from simpletuner.helpers.training.multi_process import should_log
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


MINIMAX_H3_BASE_REPO = "MiniMaxAI/MiniMax-H3"
MINIMAX_H3_SINGLE_FILE_SUFFIXES = (".safetensors", ".sft")
MINIMAX_H3_TARGET_MODES = ("auto", "video", "av")
MINIMAX_H3_TARGET_MODE_KEYS = ("minimax_h3_target_mode", "h3_target_mode")
MINIMAX_H3_VAE_TILE_SIZE = 256
MINIMAX_H3_VAE_TILE_OVERLAP = 64


def _register_minimax_h3_diffusers_components() -> None:
    """Expose bundled H3 classes for diffusers modular-pipeline model indexes."""
    import diffusers

    for name, cls in (
        ("AutoencoderKLMiniMaxH3", AutoencoderKLMiniMaxH3),
        ("AutoencoderKLMiniMaxH3Audio", AutoencoderKLMiniMaxH3Audio),
        ("MiniMaxH3Scheduler", MiniMaxH3Scheduler),
        ("MiniMaxH3Transformer3DModel", MiniMaxH3Transformer3DModel),
    ):
        if not hasattr(diffusers, name):
            setattr(diffusers, name, cls)


def _is_single_file_path(path: Any) -> bool:
    return isinstance(path, str) and path.lower().split("?", 1)[0].endswith(MINIMAX_H3_SINGLE_FILE_SUFFIXES)


class MiniMaxH3(VideoModelFoundation):
    SUPPORTS_MUON_CLIP = True
    AUTO_LORA_FORMAT_DETECTION = True
    NAME = "MiniMax H3"
    MODEL_DESCRIPTION = "Joint audio-video flow-matching transformer"
    ENABLED_IN_WIZARD = True
    VALIDATION_USES_NEGATIVE_PROMPT = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    ATTENTION_KWARG_NAME = "attention_kwargs"

    AUTOENCODER_CLASS = AutoencoderKLMiniMaxH3
    AUDIO_AUTOENCODER_CLASS = AutoencoderKLMiniMaxH3Audio
    LATENT_CHANNEL_COUNT = 24
    DEFAULT_NOISE_SCHEDULER = "flow_matching"
    VALIDATION_SCHEDULER_NAME = "MiniMaxH3Scheduler"

    MODEL_CLASS = MiniMaxH3Transformer3DModel
    MODEL_SUBFOLDER = "transformer"
    DEFAULT_PIPELINE_TYPE = PipelineTypes.TEXT2IMG
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: MiniMaxH3Pipeline,
        PipelineTypes.IMG2VIDEO: MiniMaxH3Pipeline,
        PipelineTypes.IMG2IMG: MiniMaxH3Pipeline,
    }

    DEFAULT_MODEL_FLAVOUR = "fl2va"
    HUGGINGFACE_PATHS: Dict[str, str] = {
        "fl2va": MINIMAX_H3_BASE_REPO,
        "ref2va": MINIMAX_H3_BASE_REPO,
        "convrot-int8": MINIMAX_H3_BASE_REPO,
        "convrot-int4": MINIMAX_H3_BASE_REPO,
        "nvfp4": MINIMAX_H3_BASE_REPO,
        "fp8-e4m3fn": MINIMAX_H3_BASE_REPO,
    }
    TRANSFORMER_PATH_OVERRIDES: Dict[str, str] = {
        "convrot-int8": (
            "https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot/resolve/main/"
            "MiniMax_H3_FL2VA_pruned_int8_convrot.safetensors"
        ),
        "convrot-int4": (
            "https://huggingface.co/Abiray/Minimax-H3-nvfp4-INT4-INT8-Convrot/resolve/main/"
            "MiniMax_H3_FL2VA_pruned_int4_convrot.safetensors"
        ),
        "nvfp4": ("https://huggingface.co/rockerBOO/minimax-h3-nvfp4/resolve/main/" "minimax_h3_fl2va_nvfp4.safetensors"),
        "fp8-e4m3fn": (
            "https://huggingface.co/rzgar/minimax_h3_fl2va_fp8_e4m3fn/resolve/main/"
            "minimax_h3_fl2va_fp8_e4m3fn.safetensors"
        ),
    }
    VAE_PATH_OVERRIDES: Dict[str, str] = {
        "convrot-int8": (
            "https://huggingface.co/Kijai/MiniMax-H3-experimental/resolve/main/"
            "minimax_h3_video_vae_int8_convrot.safetensors"
        ),
    }
    MODEL_LICENSE = "other"
    MODEL_LICENSE_NAME = "minimax-h3-community-license-agreement"
    MODEL_LICENSE_LINK = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"

    DEFAULT_LORA_TARGET = ["to_q", "to_k", "to_v", "to_out.0"]
    DEFAULT_LYCORIS_TARGET = ["MiniMaxH3Attention", "FeedForward"]

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Qwen3-VL",
            "tokenizer": Qwen2TokenizerFast,
            "tokenizer_subfolder": "tokenizer",
            "model": Qwen3VLForConditionalGeneration,
            "subfolder": "text_encoder",
        },
    }
    PROCESSOR_CLASS = Qwen3VLProcessor
    PROCESSOR_SUBFOLDER = "processor"

    def __init__(self, config: dict, accelerator):
        super().__init__(config, accelerator)
        sparse_mode = str(getattr(self.config, "minimax_h3_sparse_attention", "disabled") or "disabled").lower()
        if sparse_mode not in {"disabled", "none", "full", "dense"}:
            initialize_minimax_h3_flex_attention()
        self.audio_vae = None
        self.processor = None
        self._warned_missing_audio = False
        self._warned_audio_disabled = False
        self._warned_image_audio_disabled = False

    def post_model_load_setup(self):
        super().post_model_load_setup()
        transformer = self.unwrap_model(self.model)
        sparse_config = transformer.configure_h3_sparse_attention(
            mode=getattr(self.config, "minimax_h3_sparse_attention", "disabled") or "disabled",
            block_shape=getattr(self.config, "minimax_h3_sparse_block_shape", "1,8,16") or "1,8,16",
            video_kv_fraction=getattr(self.config, "minimax_h3_sparse_video_kv_fraction", 0.5),
            share_across_heads=getattr(self.config, "minimax_h3_sparse_share_heads", False),
            start_layer=getattr(self.config, "minimax_h3_sparse_start_layer", 0),
        )
        if sparse_config.enabled:
            logger.warning(
                "Enabled experimental MiniMax-H3 %s sparse attention: block_shape=%s, video_kv_fraction=%.3f, "
                "share_across_heads=%s, start_layer=%d. MiniMax has not released its exact production routing "
                "configuration yet.",
                sparse_config.mode,
                sparse_config.block_shape,
                sparse_config.video_kv_fraction,
                sparse_config.share_across_heads,
                sparse_config.start_layer,
            )

    def supports_crepa_self_flow(self) -> bool:
        return True

    def _prepare_crepa_self_flow_batch(self, batch: dict, state: dict) -> dict:
        return self._prepare_video_crepa_self_flow_batch(batch=batch, state=state)

    def flow_matching_target_direction(self) -> float:
        return -1.0

    def _configured_anyflow(self) -> bool:
        method = str(getattr(self.config, "distillation_method", "") or "").strip().lower()
        configured = getattr(self.config, "distillation_config", None)
        configured = configured if isinstance(configured, dict) else {}
        if method == "anyflow":
            return True
        if method != "h3_drift":
            return False
        h3_config = configured.get("h3_drift", configured)
        return str(h3_config.get("inner_distillation_method", "") or "").strip().lower() == "anyflow"

    def _anyflow_distillation_config(self) -> dict:
        method = str(getattr(self.config, "distillation_method", "") or "").strip().lower()
        configured = getattr(self.config, "distillation_config", None)
        configured = configured if isinstance(configured, dict) else {}
        if method == "h3_drift":
            h3_config = configured.get("h3_drift", configured)
            if not isinstance(h3_config, dict):
                return {}
            anyflow_config = h3_config.get("inner_distillation_config", {})
        else:
            anyflow_config = configured.get("anyflow", configured)
        return anyflow_config if isinstance(anyflow_config, dict) else {}

    def _apply_h3_anyflow_guidance_defaults(self) -> None:
        if not self._configured_anyflow():
            return

        configured = getattr(self.config, "distillation_config", None)
        if not isinstance(configured, dict):
            configured = {}
            self.config.distillation_config = configured

        method = str(getattr(self.config, "distillation_method", "") or "").strip().lower()
        if method == "h3_drift":
            h3_config = configured.get("h3_drift", configured)
            if not isinstance(h3_config, dict):
                raise ValueError("MiniMax-H3 h3_drift distillation config must be a mapping.")
            anyflow_config = h3_config.get("inner_distillation_config")
            if anyflow_config is None:
                anyflow_config = {}
                h3_config["inner_distillation_config"] = anyflow_config
        else:
            anyflow_config = configured.get("anyflow", configured)

        if not isinstance(anyflow_config, dict):
            raise ValueError("MiniMax-H3 AnyFlow distillation config must be a mapping.")
        anyflow_config.setdefault("fuse_guidance_scale", 1.0)
        anyflow_config.setdefault("real_score_guidance_scale", 0.0)

    def _get_additional_lora_targets(self) -> list[str]:
        targets = super()._get_additional_lora_targets()
        if not self._configured_anyflow():
            return targets

        anyflow_config = self._anyflow_distillation_config()
        train_time_embedder = bool(anyflow_config.get("train_time_embedder", True))
        train_delta_embedder = bool(anyflow_config.get("train_delta_embedder", True))
        targets.extend(["ff.net.0.proj", "ff.net.2"])
        if bool(anyflow_config.get("lora_target_adaln", False)):
            targets.append("adaln_proj.linear")
        transformer = self.unwrap_model(self.model) if getattr(self, "model", None) is not None else None
        if transformer is not None and getattr(transformer, "time_embedder", None) is not None:
            if train_time_embedder:
                targets.extend(["time_embedder.linear_1", "time_embedder.linear_2"])
            if train_delta_embedder:
                targets.extend(["delta_time_embedder.linear_1", "delta_time_embedder.linear_2"])
        return list(dict.fromkeys(targets))

    def get_lora_save_layers(self):
        if not self._configured_anyflow():
            return super().get_lora_save_layers()
        if not bool(self._anyflow_distillation_config().get("train_delta_embedder", True)):
            return super().get_lora_save_layers()
        transformer = self.unwrap_model(self.model) if getattr(self, "model", None) is not None else None
        if transformer is not None and getattr(transformer, "delta_adaln_embedder", None) is not None:
            return ["delta_adaln_embedder"]
        return super().get_lora_save_layers()

    def _assert_anyflow_endpoint_parameters_trainable(self) -> None:
        if not self._configured_anyflow():
            return
        if not bool(self._anyflow_distillation_config().get("train_delta_embedder", True)):
            return
        transformer = self.unwrap_model(self.model)
        trainable = [
            name
            for name, parameter in transformer.named_parameters()
            if parameter.requires_grad and ("delta_adaln_embedder" in name or "delta_time_embedder" in name)
        ]
        if not trainable:
            raise RuntimeError(
                "MiniMax-H3 AnyFlow requested train_delta_embedder=true, but the PEFT adapter has no trainable "
                "delta timestep parameters. The student cannot learn large (t, r) interval conditioning."
            )
        logger.info("MiniMax-H3 AnyFlow endpoint conditioning is trainable through: %s", ", ".join(trainable))

    def add_lora_adapter(self):
        result = super().add_lora_adapter()
        self._assert_anyflow_endpoint_parameters_trainable()
        return result

    @classmethod
    def adjust_video_frames(cls, num_frames: int) -> int:
        if num_frames < 1:
            raise ValueError(f"`num_frames` must be positive, got {num_frames}.")
        if num_frames == 1:
            return 1
        while num_frames % 17 != 5:
            num_frames += 1
        return num_frames

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        return 49

    @classmethod
    def get_acceleration_presets(cls) -> list[AccelerationPreset]:
        base_config = {
            "base_model_precision": "no_change",
            "gradient_checkpointing": True,
        }
        return [
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="balanced",
                name="RamTorch - Balanced",
                description="Streams roughly half of the H3 transformer blocks from CPU RAM.",
                tab="basic",
                tradeoff_vram="Substantial VRAM savings for the 33B transformer.",
                tradeoff_speed="Increases training time from CPU-GPU transfers.",
                tradeoff_notes="Requires high system RAM.",
                requires_min_system_ram_gb=256,
                config={
                    **base_config,
                    "ramtorch": True,
                    "ramtorch_disable_extensions": False,
                    "ramtorch_target_modules": ",".join(f"transformer_blocks.{idx}.*" for idx in range(25)),
                    "ramtorch_text_encoder": True,
                },
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="light",
                name="Block Swap - Light",
                description="Swaps 12 of 50 transformer blocks between GPU and CPU.",
                tab="basic",
                tradeoff_vram="Reduces VRAM with moderate transfer overhead.",
                tradeoff_speed="Increases training time from CPU-GPU transfers.",
                tradeoff_notes="Requires high system RAM.",
                requires_min_system_ram_gb=192,
                config={**base_config, "musubi_blocks_to_swap": 12},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="balanced",
                name="Block Swap - Balanced",
                description="Swaps 25 of 50 transformer blocks between GPU and CPU.",
                tab="basic",
                tradeoff_vram="Large VRAM reduction with higher transfer overhead.",
                tradeoff_speed="Increases training time from CPU-GPU transfers.",
                tradeoff_notes="Requires high system RAM.",
                requires_min_system_ram_gb=256,
                config={**base_config, "musubi_blocks_to_swap": 25},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="aggressive",
                name="Block Swap - Aggressive",
                description="Swaps 37 of 50 transformer blocks between GPU and CPU.",
                tab="basic",
                tradeoff_vram="Maximum block-swap VRAM reduction.",
                tradeoff_speed="Largest transfer overhead.",
                tradeoff_notes="Requires high system RAM.",
                requires_min_system_ram_gb=256,
                config={**base_config, "musubi_blocks_to_swap": 37},
            ),
            *get_deepspeed_presets(base_config),
            *get_sdnq_presets(base_config),
            *get_torchao_presets(base_config),
            *get_quanto_presets(base_config),
            *get_bitsandbytes_presets(base_config),
        ]

    def setup_model_flavour(self):
        explicit_vae_path = getattr(self.config, "pretrained_vae_model_name_or_path", None)
        super().setup_model_flavour()
        flavour = getattr(self.config, "model_flavour", None)
        override_map = getattr(self, "TRANSFORMER_PATH_OVERRIDES", {})
        if getattr(self.config, "pretrained_transformer_model_name_or_path", None) is None and flavour in override_map:
            self.config.pretrained_transformer_model_name_or_path = override_map[flavour]
            self.config.pretrained_transformer_subfolder = None
        vae_override_map = getattr(self, "VAE_PATH_OVERRIDES", {})
        if explicit_vae_path is None and flavour in vae_override_map:
            vae_override = vae_override_map[flavour]
            self.config.pretrained_vae_model_name_or_path = vae_override
            if getattr(self.config, "vae_path", None) in (
                None,
                self.config.pretrained_model_name_or_path,
            ):
                self.config.vae_path = vae_override
        if flavour == "ref2va":
            self.config.pretrained_transformer_subfolder = "transformer_ref"
            self.PIPELINE_CLASSES = {
                PipelineTypes.TEXT2IMG: MiniMaxH3Ref2VAPipeline,
                PipelineTypes.IMG2VIDEO: MiniMaxH3Ref2VAPipeline,
                PipelineTypes.IMG2IMG: MiniMaxH3Ref2VAPipeline,
            }
        self._apply_h3_schedule_defaults()

    def check_user_config(self):
        super().check_user_config()
        if getattr(self.config, "framerate", None) is None:
            self.config.framerate = MINIMAX_H3_FPS
        self._apply_h3_anyflow_guidance_defaults()
        self._apply_h3_schedule_defaults()
        self._force_video_vae_reference_settings()

    def _apply_h3_schedule_defaults(self):
        video_shift = getattr(self.config, "flow_schedule_shift", None)
        if video_shift is None:
            self.config.flow_schedule_shift = 12.0
        else:
            try:
                video_shift_float = float(video_shift)
            except (TypeError, ValueError):
                video_shift_float = None
            if video_shift_float is not None and abs(video_shift_float - 3.0) <= 1e-9:
                logger.warning(
                    "MiniMax-H3 uses video flow_schedule_shift=12.0. "
                    "Overriding inherited global default flow_schedule_shift=3.0."
                )
                self.config.flow_schedule_shift = 12.0
        if getattr(self.config, "audio_flow_schedule_shift", None) is None:
            self.config.audio_flow_schedule_shift = 3.0

    def _force_video_vae_reference_settings(self):
        if getattr(self.config, "vae_enable_tiling", None) is not True:
            if getattr(self.config, "vae_enable_tiling", None) is False:
                logger.warning(
                    "MiniMax-H3 requires VAE tiling for stable video VAE output; overriding vae_enable_tiling=true."
                )
            self.config.vae_enable_tiling = True
        if getattr(self.config, "vae_enable_temporal_roll", None) is not True:
            if getattr(self.config, "vae_enable_temporal_roll", None) is False:
                logger.warning(
                    "MiniMax-H3 requires temporal VAE chunking for stable video VAE output; "
                    "overriding vae_enable_temporal_roll=true."
                )
            self.config.vae_enable_temporal_roll = True

    def _model_config_path(self):
        model_path = getattr(self.config, "pretrained_model_name_or_path", None)
        transformer_path = getattr(self.config, "pretrained_transformer_model_name_or_path", None)
        if _is_single_file_path(model_path) or _is_single_file_path(transformer_path):
            return MINIMAX_H3_BASE_REPO
        return super()._model_config_path()

    def setup_training_noise_schedule(self):
        shift = float(getattr(self.config, "flow_schedule_shift", 12.0) or 12.0)
        self.noise_schedule = fix_flow_match_euler_schedule_bounds(
            FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=shift)
        )
        self.audio_noise_schedule = MiniMaxH3Scheduler(
            shift=float(getattr(self.config, "audio_flow_schedule_shift", 3.0) or 3.0)
        )
        return self.config, self.noise_schedule

    def sample_flow_sigmas(self, batch: dict, state: dict) -> tuple[torch.Tensor, torch.Tensor]:
        sigmas, _timesteps = super().sample_flow_sigmas(batch=batch, state=state)
        return sigmas, self.flow_matching_timesteps_from_sigmas(sigmas)

    def flow_matching_timesteps_from_sigmas(
        self,
        sigmas: torch.Tensor,
        *,
        reference_timesteps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del reference_timesteps
        return 1.0 - sigmas

    def pretrained_load_args(self, pretrained_load_args: dict) -> dict:
        args = super().pretrained_load_args(pretrained_load_args)
        return apply_musubi_pretrained_defaults(self.config, args)

    def _resolve_component_path(self, explicit_path: str | None = None) -> str:
        path = explicit_path or self.config.pretrained_model_name_or_path
        if _is_single_file_path(path):
            return self._model_config_path()
        return path

    def _resolve_vae_dtype(self):
        vae_dtype = getattr(self.config, "vae_dtype", None)
        if vae_dtype == "bf16":
            return torch.bfloat16
        if vae_dtype == "fp16":
            return torch.float16
        if vae_dtype == "fp32":
            return torch.float32
        return self.config.weight_dtype

    def load_vae(self, move_to_device: bool = True):
        self._force_video_vae_reference_settings()
        if self.vae is None:
            explicit_vae_path = getattr(self.config, "pretrained_vae_model_name_or_path", None)
            if _is_single_file_path(explicit_vae_path):
                self.vae = self.AUTOENCODER_CLASS.from_single_file(
                    explicit_vae_path,
                    torch_dtype=self._resolve_vae_dtype(),
                    revision=self.config.revision,
                )
            else:
                vae_path = self._resolve_component_path(explicit_vae_path)
                self.vae = self.AUTOENCODER_CLASS.from_pretrained(
                    vae_path,
                    subfolder="vae",
                    torch_dtype=self._resolve_vae_dtype(),
                    revision=self.config.revision,
                    variant=self.config.variant,
                    use_safetensors=True,
                )
            self.vae.requires_grad_(False)
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling(
                tile_sample_min_height=MINIMAX_H3_VAE_TILE_SIZE,
                tile_sample_min_width=MINIMAX_H3_VAE_TILE_SIZE,
                tile_sample_min_overlap_height=MINIMAX_H3_VAE_TILE_OVERLAP,
                tile_sample_min_overlap_width=MINIMAX_H3_VAE_TILE_OVERLAP,
            )
        else:
            logger.warning("MiniMax-H3 VAE tiling is required but this VAE does not expose enable_tiling().")
        if getattr(self.config, "vae_enable_slicing", False):
            self.vae.enable_slicing()
        elif hasattr(self.vae, "disable_slicing"):
            self.vae.disable_slicing()
        if hasattr(self.vae, "enable_temporal_chunking"):
            self.vae.enable_temporal_chunking()
        else:
            logger.warning(
                "MiniMax-H3 temporal VAE chunking is required but this VAE does not expose enable_temporal_chunking()."
            )
        if move_to_device and self.vae.device != self.accelerator.device:
            self.vae.to(self.accelerator.device, dtype=self._resolve_vae_dtype())
        self.post_vae_load_setup()
        self._load_audio_vae(move_to_device=move_to_device)

    def _load_audio_vae(self, move_to_device: bool = True):
        if self.audio_vae is not None:
            return
        audio_vae_path = self._resolve_component_path(getattr(self.config, "pretrained_audio_vae_model_name_or_path", None))
        self.audio_vae = self.AUDIO_AUTOENCODER_CLASS.from_pretrained(
            audio_vae_path,
            subfolder="audio_vae",
            torch_dtype=self._resolve_vae_dtype(),
            revision=self.config.revision,
            variant=self.config.variant,
            use_safetensors=True,
        )
        self.audio_vae.requires_grad_(False)
        if move_to_device:
            self.audio_vae.to(self.accelerator.device, dtype=self._resolve_vae_dtype())

    def unload_vae(self):
        for pipeline in getattr(self, "pipelines", {}).values():
            if hasattr(pipeline, "update_components"):
                pipeline.update_components(vae=None, audio_vae=None)
            else:
                setattr(pipeline, "vae", None)
                setattr(pipeline, "audio_vae", None)
        super().unload_vae()
        if self.audio_vae is not None:
            if hasattr(self.audio_vae, "to"):
                self.audio_vae.to("meta")
            self.audio_vae = None

    def _load_processor_for_pipeline(self):
        if self.processor is not None:
            return self.processor
        processor_path = self._resolve_qwen_processor_path(self._model_config_path())
        processor_subfolder = self._resolve_qwen_processor_subfolder(self.PROCESSOR_SUBFOLDER)
        processor_kwargs = {
            "pretrained_model_name_or_path": processor_path,
            "subfolder": processor_subfolder,
            "revision": getattr(self.config, "revision", None),
        }
        if getattr(self.config, "local_files_only", False):
            processor_kwargs["local_files_only"] = True
        self.processor = self.PROCESSOR_CLASS.from_pretrained(**processor_kwargs)
        return self.processor

    def _h3_lora_component_name(self) -> str:
        return "transformer_ref" if getattr(self.config, "model_flavour", None) == "ref2va" else "transformer"

    def _h3_transformer_uses_gate_first_swiglu(self) -> bool:
        transformer = self.unwrap_model(self.model) if getattr(self, "model", None) is not None else None
        return bool(getattr(getattr(transformer, "config", None), "swiglu_gate_first", False))

    def _convert_lora_state_dict_to_comfyui(
        self,
        weights: dict,
        *,
        adapter_metadata: Optional[dict] = None,
        component_adapter_metadata: Optional[dict] = None,
    ) -> dict:
        from simpletuner.helpers.models.minimaxh3.modular_pipeline import _convert_minimax_h3_diffusers_lora_to_comfyui

        return _convert_minimax_h3_diffusers_lora_to_comfyui(
            weights,
            adapter_metadata=adapter_metadata,
            source_swiglu_gate_first=self._h3_transformer_uses_gate_first_swiglu(),
        )

    def _convert_lora_state_dict_from_comfyui(
        self,
        weights: dict,
        *,
        target_prefix: str,
    ) -> tuple[dict, dict]:
        from simpletuner.helpers.models.minimaxh3.modular_pipeline import _convert_minimax_h3_comfy_lora_to_diffusers

        return _convert_minimax_h3_comfy_lora_to_diffusers(
            weights,
            target_prefix=target_prefix,
            target_swiglu_gate_first=self._h3_transformer_uses_gate_first_swiglu(),
        )

    def _prepare_plain_h3_lora_swiglu_layout(self, state_dict: dict, metadata: Optional[dict]) -> dict:
        from simpletuner.helpers.models.minimaxh3.modular_pipeline import (
            _convert_minimax_h3_diffusers_swiglu_lora_layout,
            _minimax_h3_swiglu_gate_first_from_metadata,
        )

        source_gate_first = _minimax_h3_swiglu_gate_first_from_metadata(
            metadata,
            target_prefix=self._h3_lora_component_name(),
        )
        if source_gate_first is None:
            return state_dict
        return _convert_minimax_h3_diffusers_swiglu_lora_layout(
            state_dict,
            source_gate_first=source_gate_first,
            target_gate_first=self._h3_transformer_uses_gate_first_swiglu(),
        )

    def _lora_state_dict_load_kwargs(self) -> dict:
        return {"return_lora_metadata": True}

    def _prepare_loaded_lora_state_dict(self, state_dict: dict, metadata: Optional[dict] = None) -> dict:
        from simpletuner.helpers.models.minimaxh3.modular_pipeline import _is_minimax_h3_native_lora_state_dict

        if _is_minimax_h3_native_lora_state_dict(state_dict):
            return state_dict
        return self._prepare_plain_h3_lora_swiglu_layout(state_dict, metadata)

    def _prepare_init_lora_state_dict(self, state_dict: dict, metadata: Optional[dict] = None) -> dict:
        from simpletuner.helpers.models.minimaxh3.modular_pipeline import (
            _convert_minimax_h3_comfy_lora_to_diffusers,
            _is_minimax_h3_native_lora_state_dict,
        )

        lora_format = normalize_lora_format(getattr(self.config, "lora_format", None))
        detected_format = detect_state_dict_format(state_dict)
        if lora_format == PEFTLoRAFormat.DIFFUSERS and (
            detected_format == PEFTLoRAFormat.COMFYUI or _is_minimax_h3_native_lora_state_dict(state_dict)
        ):
            lora_format = PEFTLoRAFormat.COMFYUI
        if lora_format != PEFTLoRAFormat.COMFYUI:
            return self._prepare_plain_h3_lora_swiglu_layout(state_dict, metadata)
        converted, network_alphas = _convert_minimax_h3_comfy_lora_to_diffusers(
            state_dict,
            target_prefix=self._h3_lora_component_name(),
            target_swiglu_gate_first=self._h3_transformer_uses_gate_first_swiglu(),
        )
        prepared = {}
        for key, value in converted.items():
            if key.endswith(".lora.down.weight"):
                key = f"{key[: -len('.lora.down.weight')]}.lora_A.weight"
            elif key.endswith(".lora.up.weight"):
                key = f"{key[: -len('.lora.up.weight')]}.lora_B.weight"
            prepared[key] = value
        for key, alpha in network_alphas.items():
            prepared[key] = torch.tensor(alpha, dtype=torch.float32)
        return prepared

    def save_lora_weights(self, *args, **kwargs):
        from simpletuner.helpers.models.minimaxh3.modular_pipeline import (
            MINIMAX_H3_FLOWMAP_DELTATIME_METADATA_KEY,
            MINIMAX_H3_FLOWMAP_GATE_METADATA_KEY,
            MINIMAX_H3_SWIGLU_GATE_FIRST_METADATA_KEY,
        )

        metadata_key = f"{self.MODEL_SUBFOLDER}_lora_adapter_metadata"
        adapter_metadata = dict(kwargs.get(metadata_key) or {})
        lora_format = normalize_lora_format(getattr(self.config, "lora_format", None))
        adapter_metadata[MINIMAX_H3_SWIGLU_GATE_FIRST_METADATA_KEY] = (
            lora_format == PEFTLoRAFormat.COMFYUI or self._h3_transformer_uses_gate_first_swiglu()
        )
        transformer = self.unwrap_model(self.model) if getattr(self, "model", None) is not None else None
        deltatime_type = getattr(transformer, "flowmap_deltatime_type", None)
        if deltatime_type is not None:
            gate = getattr(transformer, "flowmap_delta_emb_gate", None)
            if torch.is_tensor(gate):
                adapter_metadata[MINIMAX_H3_FLOWMAP_GATE_METADATA_KEY] = float(gate.detach().float().cpu().item())
            adapter_metadata[MINIMAX_H3_FLOWMAP_DELTATIME_METADATA_KEY] = str(deltatime_type)
        kwargs[metadata_key] = adapter_metadata
        return super().save_lora_weights(*args, **kwargs)

    def get_lora_target_layers(self):
        manual_targets = self._get_peft_lora_target_modules()
        if manual_targets:
            return manual_targets
        if str(getattr(self.config, "lora_type", "standard")).lower() == "standard":
            init_lora_state_dict = self._load_init_lora_state_dict()
            if init_lora_state_dict:
                ranks = collect_lora_ranks(
                    init_lora_state_dict,
                    prefix_to_strip=f"{self._h3_lora_component_name()}.",
                )
                if ranks:
                    return sorted(ranks)
        return super().get_lora_target_layers()

    def load_text_tokenizer(self):
        super().load_text_tokenizer()
        if self.processor is None:
            self.processor = self._load_processor_for_pipeline()

    def _text_encoder_components(self):
        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder(move_to_device=True)
        if self.tokenizers is None or len(self.tokenizers) == 0:
            self.load_text_tokenizer()
        processor = self._load_processor_for_pipeline()
        transformer = (
            self.unwrap_model(self.model) if self.model is not None else SimpleNamespace(dtype=self.config.weight_dtype)
        )
        return SimpleNamespace(
            text_encoder=self.text_encoders[0],
            tokenizer=self.tokenizers[0],
            processor=processor,
            transformer=transformer,
            transformer_ref=transformer,
            _execution_device=self.accelerator.device,
        )

    def text_embed_cache_key(self) -> TextEmbedCacheKey:
        return TextEmbedCacheKey.DATASET_AND_FILENAME

    def requires_text_embed_image_context(self) -> bool:
        return True

    def should_precompute_dropout_caption(self) -> bool:
        return False

    def use_text_cache_dropout_sentinel(self) -> bool:
        return False

    def uses_image_context_dropout_caption_cache(self) -> bool:
        return True

    def requires_special_scheduler_setup(self) -> bool:
        return True

    def text_embed_cache_key_value(self, *, prompt: str, default_key: str, metadata: dict) -> str:
        del metadata
        if prompt == "":
            return f"{default_key}:__caption_dropout__"
        return default_key

    def text_embed_cache_metadata_for_filepath(
        self,
        *,
        init_backend: dict,
        image_path: str,
        prompt: str,
        data_backend_id: str | None,
        dataset_relative_path: str | None,
    ) -> dict:
        del init_backend, image_path, prompt
        conditioning_datasets = StateTracker.get_conditioning_datasets(data_backend_id)
        if not conditioning_datasets:
            return {}

        image_paths = []
        data_backend_ids = []
        for conditioning_backend in conditioning_datasets:
            conditioning_backend_id = conditioning_backend.get("id")
            conditioning_config = conditioning_backend.get("config", {})
            conditioning_root = conditioning_config.get("instance_data_dir")
            if not conditioning_backend_id or not conditioning_root or not dataset_relative_path:
                continue
            conditioning_path = os.path.join(conditioning_root, dataset_relative_path)
            if (conditioning_config.get("conditioning_config") or {}).get("type") == "i2v_first_frame":
                conditioning_path = os.path.splitext(conditioning_path)[0] + ".png"
            image_paths.append(conditioning_path)
            data_backend_ids.append(conditioning_backend_id)

        if not image_paths:
            return {}
        return {
            "image_paths": image_paths,
            "data_backend_ids": data_backend_ids,
            "image_path": image_paths[0],
            "data_backend_id": data_backend_ids[0],
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        encoded = []
        components = self._text_encoder_components()
        max_text_length = getattr(self.config, "tokenizer_max_length", None)
        if max_text_length is None:
            max_text_length = MINIMAX_H3_DEFAULT_MAX_TEXT_LENGTH
        prompt_contexts = getattr(self, "_current_prompt_contexts", None)
        if self.requires_text_embed_image_context():
            if not prompt_contexts or len(prompt_contexts) != len(prompts):
                prompt_contexts = [{} for _ in prompts]
                prompt_images = [None] * len(prompts)
            else:
                prompt_images = self._prepare_prompt_image_batch(prompt_contexts, len(prompts))
        else:
            prompt_images = [None] * len(prompts)
        if prompt_contexts is None:
            prompt_contexts = [{} for _ in prompts]
        prepared_prompts = []
        null_instructions = []
        for prompt, context in zip(prompts, prompt_contexts):
            null_instruction = False
            if is_negative_prompt and isinstance(context, dict) and str(prompt).strip() == "":
                positive_prompt = context.get("positive_prompt")
                if positive_prompt is not None:
                    prompt = str(positive_prompt)
                    null_instruction = True
            if prompt == "" and not null_instruction:
                prompt = " "
            prepared_prompts.append(prompt)
            null_instructions.append(null_instruction)

        if len(prepared_prompts) == 1:
            prompt_embeds, text_token_tags = MiniMaxH3TextEncoderStep.encode_prompt(
                components,
                prepared_prompts[0],
                images=prompt_images[0],
                device=self.accelerator.device,
                dtype=self.config.weight_dtype,
                null_instruction=null_instructions[0],
                max_length=max_text_length,
            )
            encoded = [{"prompt_embeds": prompt_embeds, "text_token_tags": text_token_tags}]
        else:
            batch_outputs = MiniMaxH3TextEncoderStep.encode_prompt_batch(
                components,
                prepared_prompts,
                image_batches=prompt_images,
                device=self.accelerator.device,
                dtype=self.config.weight_dtype,
                null_instructions=null_instructions,
                max_length=max_text_length,
            )
            encoded = [
                {"prompt_embeds": prompt_embeds, "text_token_tags": text_token_tags}
                for prompt_embeds, text_token_tags in batch_outputs
            ]
        return self.collate_prompt_embeds(encoded)

    def _prepare_prompt_image_batch(
        self,
        prompt_contexts: list[dict],
        batch_size: int,
    ) -> list[list[Image.Image] | None]:
        if not prompt_contexts or len(prompt_contexts) != batch_size:
            raise ValueError("MiniMax-H3 text encoding requires one context record per caption.")
        image_batch = []
        for index, context in enumerate(prompt_contexts):
            image = self._extract_prompt_image_from_context(context)
            if image is None:
                if not self._prompt_context_declares_image(context):
                    image_batch.append(None)
                    continue
                raise ValueError(f"Failed to resolve MiniMax-H3 text conditioning image for caption index {index}.")
            image_batch.append([image])
        return image_batch

    @staticmethod
    def _prompt_context_declares_image(context: dict) -> bool:
        if not isinstance(context, dict):
            return False
        return any(
            context.get(key) is not None
            for key in (
                "conditioning_pixel_values",
                "image_paths",
            )
        )

    def _extract_prompt_image_from_context(self, context: dict) -> Image.Image | None:
        if not isinstance(context, dict):
            return None
        direct_image = context.get("conditioning_pixel_values")
        if direct_image is not None:
            return self._coerce_prompt_image(direct_image)
        image_paths = context.get("image_paths")
        data_backend_ids = context.get("data_backend_ids")
        if isinstance(image_paths, (list, tuple)) and image_paths:
            image_path = image_paths[0]
            if isinstance(data_backend_ids, (list, tuple)) and data_backend_ids:
                data_backend_id = data_backend_ids[0]
            else:
                data_backend_id = context.get("data_backend_id")
        else:
            # `image_path` alone identifies the target sample in ordinary T2V
            # backends. Only plural paths or direct pixels denote reference context.
            return None
        if not image_path or not data_backend_id:
            return None
        backend_entry = StateTracker.get_data_backend(data_backend_id)
        if backend_entry is None:
            return None
        data_backend = backend_entry.get("data_backend")
        if data_backend is None:
            return None
        return self._coerce_prompt_image(data_backend.read_image(image_path))

    @staticmethod
    def _coerce_prompt_image(image) -> Image.Image | None:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            array = image[0] if image.ndim == 4 else image
            if array.ndim == 3 and array.shape[0] in (1, 3):
                array = np.transpose(array, (1, 2, 0))
            if array.ndim == 3 and array.shape[2] == 4:
                array = array[:, :, :3]
            if array.dtype != np.uint8:
                array = array.astype(np.float32)
                if array.max() <= 1.0 and array.min() >= 0.0:
                    array = array * 255.0
                elif array.min() < 0.0:
                    array = (array + 1.0) * 127.5
                array = np.clip(array, 0.0, 255.0).round().astype(np.uint8)
            return Image.fromarray(array).convert("RGB")
        if torch.is_tensor(image):
            tensor = image.detach().float().cpu()
            if tensor.dim() == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            if tensor.dim() != 3:
                return None
            if tensor.shape[0] in (1, 3):
                tensor = tensor.permute(1, 2, 0)
            if tensor.shape[-1] == 4:
                tensor = tensor[..., :3]
            if tensor.max().item() <= 1.0 and tensor.min().item() >= 0.0:
                tensor = tensor * 255.0
            elif tensor.min().item() < 0.0:
                tensor = (tensor + 1.0) * 127.5
            array = tensor.clamp(0.0, 255.0).round().to(torch.uint8).numpy()
            if array.shape[-1] == 1:
                array = np.repeat(array, 3, axis=-1)
            return Image.fromarray(array).convert("RGB")
        return None

    def convert_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        return {
            "prompt_embeds": text_embedding["prompt_embeds"],
            "text_token_tags": text_embedding.get("text_token_tags"),
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        guidance_scale = float(getattr(self.config, "validation_guidance_real", 1.0) or 1.0)
        if guidance_scale == 1.0:
            return {}
        result = {
            "negative_prompt_embeds": text_embedding["prompt_embeds"],
            "negative_text_token_tags": text_embedding.get("text_token_tags"),
            "guidance_scale_real": guidance_scale,
        }
        no_cfg_until = getattr(self.config, "validation_no_cfg_until_timestep", None)
        if isinstance(no_cfg_until, int):
            result["no_cfg_until_timestep"] = no_cfg_until
        return result

    def collate_prompt_embeds(self, text_encoder_output: list[dict]) -> dict:
        if not text_encoder_output:
            return {}
        embeds = [item["prompt_embeds"] for item in text_encoder_output]
        tags = [item["text_token_tags"] for item in text_encoder_output]
        max_seq_len = max(embed.shape[-2] for embed in embeds)
        padded_embeds = []
        padded_tags = []
        for embed, tag in zip(embeds, tags):
            if embed.dim() == 2:
                embed = embed.unsqueeze(0)
            if tag.dim() == 2 and tag.shape[0] == 1:
                tag = tag.squeeze(0)
            if tag.dim() != 1:
                raise ValueError(f"MiniMax-H3 text_token_tags must be 1-D per sample, got {tuple(tag.shape)}.")
            if embed.shape[1] != tag.shape[0]:
                raise ValueError(
                    f"MiniMax-H3 prompt embeds length {embed.shape[1]} does not match tag length {tag.shape[0]}."
                )
            if embed.shape[1] < max_seq_len:
                pad_len = max_seq_len - embed.shape[1]
                embed = torch.cat(
                    [embed, embed.new_zeros(embed.shape[0], pad_len, embed.shape[2])],
                    dim=1,
                )
                tag = torch.cat([tag, tag.new_full((pad_len,), -1)], dim=0)
            padded_embeds.append(embed)
            padded_tags.append(tag.unsqueeze(0))
        return {
            "prompt_embeds": torch.cat(padded_embeds, dim=0),
            "text_token_tags": torch.cat(padded_tags, dim=0),
        }

    def slice_text_embedding_for_cache(self, text_encoder_output: dict, batch_index: int, batch_size: int) -> dict | None:
        tags = text_encoder_output.get("text_token_tags")
        embeds = text_encoder_output.get("prompt_embeds")
        if not isinstance(tags, torch.Tensor) or not isinstance(embeds, torch.Tensor):
            return None
        if tags.ndim != 2 or embeds.ndim != 3 or tags.shape[0] != batch_size or embeds.shape[0] != batch_size:
            raise ValueError(
                "MiniMax-H3 batched cache output must contain prompt_embeds and text_token_tags with matching batches."
            )
        true_length = int((tags[batch_index] != -1).sum().item())
        return {
            "prompt_embeds": embeds[batch_index : batch_index + 1, :true_length].clone().contiguous(),
            "text_token_tags": tags[batch_index : batch_index + 1, :true_length].clone().contiguous(),
        }

    def pre_vae_encode_transform_sample(self, sample):
        if torch.is_tensor(sample) and sample.ndim == 5 and sample.shape[1] == 3:
            mean = torch.tensor(MINIMAX_H3_PIXEL_MEAN, device=sample.device, dtype=sample.dtype).view(1, 3, 1, 1, 1)
            std = torch.tensor(MINIMAX_H3_PIXEL_STD, device=sample.device, dtype=sample.dtype).view(1, 3, 1, 1, 1)
            sample = (sample + 1.0) * 0.5
            return (sample - mean) / std
        return sample

    @staticmethod
    def _cache_batch_is_i2v_first_frame(metadata_entries: Optional[list]) -> bool:
        if not metadata_entries:
            return False
        matches = [
            isinstance(entry, dict)
            and isinstance(entry.get("metadata"), dict)
            and bool(entry["metadata"].get("training_sample_path"))
            for entry in metadata_entries
        ]
        if any(matches) and not all(matches):
            raise ValueError(
                "MiniMax-H3 first-frame conditioning VAE cache batches cannot mix generated keyframes and videos."
            )
        return all(matches)

    def encode_cache_batch(self, vae, samples, metadata_entries: Optional[list] = None):
        if isinstance(vae, AutoencoderKLMiniMaxH3):
            if self._cache_batch_is_i2v_first_frame(metadata_entries):
                moments = vae._encode_clip(samples)
                posterior = DiagonalGaussianDistribution(moments)
                return posterior.mode()
            return vae.encode(samples, return_dict=True).latent_dist.mode()
        if isinstance(vae, AutoencoderKLMiniMaxH3Audio):
            if samples.ndim != 3 or samples.shape[1] != MINIMAX_H3_AUDIO_CHANNELS:
                raise ValueError(
                    "MiniMax-H3 audio VAE caching expects stereo waveform tensors with shape "
                    f"`[batch, 2, samples]`, got {tuple(samples.shape)}."
                )
            batch_size, channels, sample_count = samples.shape
            flattened = samples.reshape(batch_size * channels, 1, sample_count)
            output = vae.encode(flattened, return_dict=True)
            latents = output.latent_dist.mode()
            return latents.reshape(batch_size, channels, latents.shape[1], latents.shape[2])
        return super().encode_cache_batch(vae, samples, metadata_entries=metadata_entries)

    def scale_vae_latents_for_cache(self, latents, vae):
        if not torch.is_tensor(latents):
            return latents
        if isinstance(vae, AutoencoderKLMiniMaxH3):
            mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, -1, 1, 1, 1)
            return (latents - mean) / std
        if isinstance(vae, AutoencoderKLMiniMaxH3Audio):
            mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, 1, -1, 1)
            std = torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(1, 1, -1, 1)
            return (latents - mean) / std
        return latents

    def supports_audio_inputs(self) -> bool:
        return True

    def uses_audio_latents(self) -> bool:
        return True

    @staticmethod
    def _normalise_h3_target_mode(value: Any, *, source: str = "MiniMax-H3 target mode") -> str:
        if value is None or value == "":
            return "auto"
        mode = str(value).strip().lower()
        if mode not in MINIMAX_H3_TARGET_MODES:
            raise ValueError(f"{source} must be one of {', '.join(MINIMAX_H3_TARGET_MODES)}, got {value!r}.")
        return mode

    def _configured_h3_target_mode(self) -> str:
        value = getattr(self.config, "minimax_h3_target_mode", None)
        if value is None:
            value = getattr(self.config, "h3_target_mode", None)
        return self._normalise_h3_target_mode(value, source="--minimax_h3_target_mode")

    def _target_mode_from_backend_config(self, data_backend_id: Optional[str]) -> Optional[str]:
        if not data_backend_id:
            return None
        backend_config = StateTracker.get_data_backend_config(data_backend_id) or {}
        for key in MINIMAX_H3_TARGET_MODE_KEYS:
            if key in backend_config:
                return self._normalise_h3_target_mode(backend_config.get(key), source=f"{data_backend_id}.{key}")
        if backend_config.get("dataset_type") == "audio":
            source_dataset_id = backend_config.get("source_dataset_id")
            if source_dataset_id:
                source_config = StateTracker.get_data_backend_config(source_dataset_id) or {}
                for key in MINIMAX_H3_TARGET_MODE_KEYS:
                    if key in source_config:
                        return self._normalise_h3_target_mode(source_config.get(key), source=f"{source_dataset_id}.{key}")
        return None

    def _h3_target_mode_for_data_backend(self, data_backend_id: Optional[str] = None) -> str:
        mode = self._target_mode_from_backend_config(data_backend_id) or self._configured_h3_target_mode()
        if mode == "auto":
            return "video"
        return mode

    @staticmethod
    def _is_h3_image_latent_batch(batch: dict) -> bool:
        latents = batch.get("latents")
        return torch.is_tensor(latents) and latents.ndim == 5 and int(latents.shape[2]) == 1

    def _h3_target_mode_for_training_batch(self, batch: dict) -> str:
        h3_target_mode = self._h3_target_mode_for_data_backend(batch.get("data_backend_id"))
        if h3_target_mode != "av" or not self._is_h3_image_latent_batch(batch):
            return h3_target_mode

        audio_latents = batch.get("audio_latent_batch")
        if isinstance(audio_latents, dict):
            audio_latents = audio_latents.get("latents")
        if torch.is_tensor(audio_latents) and not self._warned_image_audio_disabled:
            logger.info(
                "MiniMax-H3 target mode is av, but this batch has one video latent frame; "
                "ignoring cached audio latents for image-mode training."
            )
            self._warned_image_audio_disabled = True
        return "video"

    def uses_audio_latents_for_data_backend(self, data_backend_id: Optional[str] = None) -> bool:
        return self._h3_target_mode_for_data_backend(data_backend_id) == "av"

    def supports_conditioning_dataset(self) -> bool:
        return True

    def _is_i2v_like_flavour(self) -> bool:
        return True

    def requires_conditioning_dataset(self) -> bool:
        return False

    def requires_conditioning_latents(self) -> bool:
        return True

    def uses_validation_negative_prompt(self) -> bool:
        return float(getattr(self.config, "validation_guidance_real", 1.0) or 1.0) != 1.0

    def validation_negative_prompt_requires_prompt_context(self) -> bool:
        return True

    def should_precompute_validation_negative_prompt(self) -> bool:
        return False

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        pipeline_kwargs.setdefault("minimax_h3_target_mode", self._configured_h3_target_mode())
        guidance_scale_real = pipeline_kwargs.pop("guidance_scale_real", None)
        if guidance_scale_real is None:
            guidance_scale_real = getattr(self.config, "validation_guidance_real", None)
        if guidance_scale_real is not None and float(guidance_scale_real) != 1.0:
            pipeline_kwargs["guidance_scale"] = float(guidance_scale_real)
            if isinstance(getattr(self.config, "validation_no_cfg_until_timestep", None), int):
                pipeline_kwargs.setdefault(
                    "no_cfg_until_timestep",
                    self.config.validation_no_cfg_until_timestep,
                )
        return pipeline_kwargs

    def _extract_sigmas_1d(self, sigmas: torch.Tensor) -> torch.Tensor:
        if sigmas.ndim == 1:
            return sigmas
        return sigmas.view(sigmas.shape[0], -1)[:, 0]

    @staticmethod
    def _shift_sigmas_between_schedules(
        sigmas: torch.Tensor,
        from_shift: float,
        to_shift: float,
    ) -> torch.Tensor:
        base = sigmas / (from_shift + sigmas * (1.0 - from_shift))
        return to_shift * base / (1.0 + (to_shift - 1.0) * base)

    @staticmethod
    def _video_frames_from_latent_frames(num_latent_frames: int) -> int:
        if num_latent_frames == 1:
            return 1
        if num_latent_frames < 2 or (num_latent_frames - 2) % 5:
            raise ValueError(
                "MiniMax-H3 video latent frames must be 1 or of the form `5 * n + 2`, " f"got {num_latent_frames}."
            )
        return ((num_latent_frames - 2) // 5) * 17 + 5

    def _expected_audio_latents(self, video_latents: torch.Tensor) -> int:
        video_frames = self._video_frames_from_latent_frames(int(video_latents.shape[2]))
        return audio_latent_num_frames(video_frames)

    def _build_empty_audio_latents(self, video_latents: torch.Tensor, device: torch.device, dtype: torch.dtype):
        audio_len = self._expected_audio_latents(video_latents)
        audio_channels = self._audio_latent_channels()
        return torch.zeros(
            video_latents.shape[0],
            MINIMAX_H3_AUDIO_CHANNELS,
            audio_channels,
            audio_len,
            device=device,
            dtype=dtype,
        )

    def prepare_batch_conditions(self, batch: dict, state: dict) -> dict:
        batch = super().prepare_batch_conditions(batch=batch, state=state)
        target_device = self.accelerator.device
        target_dtype = self.config.weight_dtype
        h3_target_mode = self._h3_target_mode_for_training_batch(batch)
        batch["minimax_h3_target_mode"] = h3_target_mode
        conditioning_latents = batch.get("conditioning_latents")
        if torch.is_tensor(conditioning_latents) and batch.get("h3_conditioning_noise") is None:
            # Distillation evaluates the same prepared batch multiple times. Keep keyframe augmentation identical
            # across the student, online teacher, and adapter-disabled H3 drift reference passes.
            batch["h3_conditioning_noise"] = torch.randn_like(conditioning_latents)
        if h3_target_mode == "video":
            audio_disabled_for_image = (
                self._is_h3_image_latent_batch(batch)
                and self._h3_target_mode_for_data_backend(batch.get("data_backend_id")) == "av"
            )
            if (
                torch.is_tensor(batch.get("audio_latent_batch"))
                and not audio_disabled_for_image
                and not self._warned_audio_disabled
            ):
                logger.info("MiniMax-H3 target mode is video; ignoring cached audio latents for this backend.")
                self._warned_audio_disabled = True
            for key in (
                "audio_latent_batch",
                "audio_latents",
                "audio_latent_mask",
                "audio_noise",
                "audio_sigmas",
                "audio_timesteps",
                "audio_noisy_latents",
            ):
                batch.pop(key, None)
            return batch

        audio_latents = batch.get("audio_latent_batch")
        audio_mask = batch.get("audio_latent_mask")
        if isinstance(audio_latents, dict):
            audio_latents = audio_latents.get("latents")
        if audio_latents is None:
            audio_latents = self._build_empty_audio_latents(batch["latents"], target_device, torch.float32)
            audio_mask = torch.zeros(audio_latents.shape[0], device=target_device, dtype=torch.float32)
            if not self._warned_missing_audio:
                logger.warning("MiniMax-H3 received no cached audio latents; using zero audio rows and masking audio loss.")
                self._warned_missing_audio = True
        elif not torch.is_tensor(audio_latents):
            raise ValueError(f"Expected MiniMax-H3 audio latents to be a tensor, got {type(audio_latents)}.")
        else:
            audio_latents = audio_latents.to(device=target_device, dtype=torch.float32)

        audio_channels = self._audio_latent_channels()
        if audio_latents.ndim == 4 and audio_latents.shape[1] == MINIMAX_H3_AUDIO_CHANNELS:
            if audio_latents.shape[2] != audio_channels and audio_latents.shape[3] == audio_channels:
                audio_latents = audio_latents.transpose(2, 3).contiguous()
            if audio_latents.shape[2] != audio_channels:
                raise ValueError(
                    f"MiniMax-H3 audio latents must have shape `[batch, 2, {audio_channels}, audio_latents]`, "
                    f"got {tuple(audio_latents.shape)}."
                )
        else:
            raise ValueError(
                f"MiniMax-H3 audio latents must have shape `[batch, 2, {audio_channels}, audio_latents]`, "
                f"got {tuple(audio_latents.shape)}."
            )
        expected_audio_latents = self._expected_audio_latents(batch["latents"])
        if audio_latents.shape[-1] != expected_audio_latents:
            raise ValueError(
                f"MiniMax-H3 audio latent length {audio_latents.shape[-1]} does not match the video duration "
                f"({expected_audio_latents}). Rebuild the audio VAE cache for this dataset."
            )

        if audio_mask is None:
            audio_mask = torch.ones(audio_latents.shape[0], device=target_device, dtype=torch.float32)
        else:
            audio_mask = audio_mask.to(device=target_device, dtype=torch.float32)

        audio_noise = torch.randn_like(audio_latents)
        audio_input_noise = audio_noise
        if self.config.input_perturbation != 0 and (
            not getattr(self.config, "input_perturbation_steps", None)
            or state.get("global_step", 0) < self.config.input_perturbation_steps
        ):
            input_perturbation = self.config.input_perturbation
            if getattr(self.config, "input_perturbation_steps", None):
                input_perturbation *= 1.0 - (state.get("global_step", 0) / self.config.input_perturbation_steps)
            audio_input_noise = audio_noise + input_perturbation * torch.randn_like(audio_latents)

        audio_sigmas = batch.get("audio_sigmas")
        if audio_sigmas is None:
            sigma_1d = self._extract_sigmas_1d(batch["sigmas"]).to(device=target_device, dtype=torch.float32)
            video_shift = float(getattr(self.config, "flow_schedule_shift", 12.0) or 12.0)
            audio_shift = float(getattr(self.config, "audio_flow_schedule_shift", 3.0) or 3.0)
            audio_sigma_1d = self._shift_sigmas_between_schedules(sigma_1d, video_shift, audio_shift)
            audio_sigmas = audio_sigma_1d.view(audio_sigma_1d.shape[0], 1, 1, 1)
        else:
            audio_sigmas = audio_sigmas.to(device=target_device, dtype=torch.float32)
            if audio_sigmas.ndim == 1:
                audio_sigmas = audio_sigmas.view(audio_sigmas.shape[0], 1, 1, 1)
        audio_noisy = (1 - audio_sigmas) * audio_latents + audio_sigmas * audio_input_noise

        batch["audio_latents"] = audio_latents
        batch["audio_latent_mask"] = audio_mask
        batch["audio_noise"] = audio_noise
        batch["audio_sigmas"] = audio_sigmas
        batch["audio_timesteps"] = (1.0 - audio_sigmas.view(audio_sigmas.shape[0], -1)[:, 0]).to(
            device=target_device, dtype=torch.float32
        )
        batch["audio_noisy_latents"] = audio_noisy.to(device=target_device, dtype=target_dtype)
        return batch

    def _resolve_text_token_tags(
        self,
        prepared_batch: dict,
        text_seq_len: int,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tags = prepared_batch.get("text_token_tags")
        if tags is None:
            text_output = prepared_batch.get("text_encoder_output")
            if isinstance(text_output, dict):
                tags = text_output.get("text_token_tags")
        if tags is None:
            return torch.full((text_seq_len,), MINIMAX_H3_TEXT_TAG, dtype=torch.long, device=device), None
        tags = tags.to(device=device, dtype=torch.long)
        if tags.ndim == 1:
            if tags.shape[0] != text_seq_len:
                raise ValueError(f"MiniMax-H3 text_token_tags length {tags.shape[0]} != prompt length {text_seq_len}.")
            return tags, None
        if tags.ndim == 2:
            if tags.shape[0] != batch_size:
                raise ValueError(f"MiniMax-H3 text_token_tags batch {tags.shape[0]} != latent batch {batch_size}.")
            if tags.shape[1] != text_seq_len:
                raise ValueError(f"MiniMax-H3 text_token_tags length {tags.shape[1]} != prompt length {text_seq_len}.")
            live_mask = tags >= 0
            text_lengths = live_mask.sum(dim=1)
            expected_live_mask = torch.arange(text_seq_len, device=device).unsqueeze(0) < text_lengths.unsqueeze(1)
            if not bool(torch.equal(live_mask, expected_live_mask)):
                raise ValueError("MiniMax-H3 batched text padding must be trailing.")
            layout_tags = tags.max(dim=0).values
            if bool((layout_tags < 0).any()):
                raise ValueError("MiniMax-H3 batched text layout contains a column that is padding for every sample.")
            if bool((live_mask & (tags != layout_tags.unsqueeze(0))).any()):
                raise ValueError("MiniMax-H3 batched training requires matching non-padding text modality layouts.")
            return layout_tags, None if bool(live_mask.all()) else live_mask
        raise ValueError(f"MiniMax-H3 text_token_tags must be 1-D or 2-D, got {tuple(tags.shape)}.")

    @staticmethod
    def _batch_timestep_values(value: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
        value = torch.as_tensor(value, dtype=torch.float32)
        if value.ndim == 0:
            return value.expand(batch_size)
        if value.shape[0] != batch_size:
            raise ValueError(f"MiniMax-H3 {name} timestep batch {value.shape[0]} != latent batch {batch_size}.")
        return value.reshape(batch_size, -1)[:, 0]

    @staticmethod
    def _batched_row_timesteps(
        layout,
        video_timesteps: torch.Tensor,
        audio_timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = []
        for video_timestep, audio_timestep in zip(video_timesteps, audio_timesteps):
            values, indices = build_row_timesteps(
                layout,
                video_timestep=float(video_timestep.item()),
                audio_timestep=float(audio_timestep.item()),
                condition_video_timestep=MINIMAX_H3_KEYFRAME_NOISE_AUG,
                condition_audio_timestep=MINIMAX_H3_KEYFRAME_NOISE_AUG,
            )
            rows.append(values[indices])
        row_timesteps = torch.stack(rows)
        values, indices = torch.unique(row_timesteps, sorted=True, return_inverse=True)
        indices = indices.view_as(row_timesteps)
        return (values, indices[0]) if row_timesteps.shape[0] == 1 else (values, indices)

    @staticmethod
    def _batched_row_timestep_intervals(
        layout,
        video_timesteps: torch.Tensor,
        audio_timesteps: torch.Tensor,
        video_r_timesteps: torch.Tensor,
        audio_r_timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = []
        for values in zip(video_timesteps, audio_timesteps, video_r_timesteps, audio_r_timesteps):
            video_timestep, audio_timestep, video_r_timestep, audio_r_timestep = values
            timesteps, r_timesteps, indices = build_row_timestep_intervals(
                layout,
                video_timestep=float(video_timestep.item()),
                audio_timestep=float(audio_timestep.item()),
                condition_video_timestep=MINIMAX_H3_KEYFRAME_NOISE_AUG,
                condition_audio_timestep=MINIMAX_H3_KEYFRAME_NOISE_AUG,
                video_r_timestep=float(video_r_timestep.item()),
                audio_r_timestep=float(audio_r_timestep.item()),
            )
            rows.append(torch.stack((timesteps[indices], r_timesteps[indices]), dim=-1))
        row_pairs = torch.stack(rows)
        pairs, indices = torch.unique(row_pairs.view(-1, 2), dim=0, sorted=True, return_inverse=True)
        indices = indices.view(row_pairs.shape[:-1])
        if row_pairs.shape[0] == 1:
            indices = indices[0]
        return pairs[:, 0], pairs[:, 1], indices

    def _audio_latent_channels(self) -> int:
        transformer = getattr(self, "model", None)
        if transformer is not None:
            transformer = self.unwrap_model(transformer)
            channels = getattr(getattr(transformer, "config", None), "audio_in_channels", None)
            if channels is not None:
                return int(channels)
        audio_vae = getattr(self, "audio_vae", None)
        channels = getattr(getattr(audio_vae, "config", None), "latent_channels", None)
        return int(channels or 32)

    def _pack_audio_latents(self, audio_latents: torch.Tensor) -> torch.Tensor:
        audio_channels = self._audio_latent_channels()
        if (
            audio_latents.ndim != 4
            or audio_latents.shape[1] != MINIMAX_H3_AUDIO_CHANNELS
            or audio_latents.shape[2] != audio_channels
        ):
            raise ValueError(
                f"MiniMax-H3 audio latents must have shape `[batch, 2, {audio_channels}, audio_latents]`, "
                f"got {tuple(audio_latents.shape)}."
            )
        return audio_latents.permute(0, 1, 3, 2).reshape(audio_latents.shape[0], -1, audio_latents.shape[2])

    def _unpack_audio_prediction(self, audio_rows: torch.Tensor, num_audio_latents: int) -> torch.Tensor:
        return audio_rows.reshape(
            audio_rows.shape[0],
            MINIMAX_H3_AUDIO_CHANNELS,
            num_audio_latents,
            audio_rows.shape[-1],
        ).permute(0, 1, 3, 2)

    def _scalar_timestep(self, value: torch.Tensor, name: str) -> float:
        if not torch.is_tensor(value):
            return float(value)
        flat = value.detach().float().view(value.shape[0], -1)[:, 0] if value.ndim > 1 else value.detach().float().flatten()
        if flat.numel() > 1 and not torch.allclose(flat, flat[0].expand_as(flat)):
            raise ValueError(
                f"MiniMax-H3 packed training currently requires one shared {name} timestep per batch. "
                "Use train_batch_size=1 or disable segmented timestep sampling."
            )
        return float(flat[0].item())

    def model_predict(self, prepared_batch):
        noisy_latents = prepared_batch["noisy_latents"].to(self.accelerator.device, dtype=self.config.weight_dtype)
        audio_noisy = prepared_batch.get("audio_noisy_latents")
        h3_target_mode = prepared_batch.get(
            "minimax_h3_target_mode",
            "av" if torch.is_tensor(audio_noisy) else "video",
        )
        h3_target_mode = self._normalise_h3_target_mode(h3_target_mode, source="prepared_batch.minimax_h3_target_mode")
        if h3_target_mode == "auto":
            h3_target_mode = "video"
        use_audio = h3_target_mode == "av" and torch.is_tensor(audio_noisy)
        encoder_hidden_states = prepared_batch["encoder_hidden_states"].to(
            self.accelerator.device, dtype=self.config.weight_dtype
        )

        if noisy_latents.ndim != 5 or noisy_latents.shape[1] != self.LATENT_CHANNEL_COUNT:
            raise ValueError(
                "MiniMax-H3 requires normalized 24-channel video latents shaped `[batch, 24, frames, height, width]`, "
                f"got {tuple(noisy_latents.shape)}."
            )

        transformer = self.unwrap_model(self.model)
        patch_size = tuple(getattr(transformer.config, "patch_size", (1, 2, 2)))
        patch_product = int(patch_size[0] * patch_size[1] * patch_size[2])
        batch_size, channels, latent_frames, latent_height, latent_width = noisy_latents.shape
        text_seq_len = encoder_hidden_states.shape[1]
        use_audio = use_audio and latent_frames > 1
        if use_audio:
            audio_noisy = audio_noisy.to(self.accelerator.device, dtype=self.config.weight_dtype)
        text_token_tags, text_valid_mask = self._resolve_text_token_tags(
            prepared_batch,
            text_seq_len,
            batch_size,
            self.accelerator.device,
        )
        packed_target_video = patchify_video_latents(noisy_latents, patch_size).view(
            batch_size, -1, channels * patch_product
        )
        condition_latents = prepared_batch.get("conditioning_latents")
        keyframe_anchors: tuple[str, ...] = ()
        force_keep_mask = None
        if condition_latents is not None:
            if isinstance(condition_latents, list):
                raise ValueError("MiniMax-H3 FL2VA training expects one conditioning latent tensor, not a list.")
            condition_latents = condition_latents.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
            if condition_latents.ndim != 5 or condition_latents.shape[1] != channels or condition_latents.shape[2] != 1:
                raise ValueError(
                    "MiniMax-H3 FL2VA currently supports one encoded keyframe shaped `[batch, 24, 1, h, w]`, "
                    f"got {tuple(condition_latents.shape)}."
                )
            if condition_latents.shape[3:] != noisy_latents.shape[3:]:
                raise ValueError(
                    "MiniMax-H3 conditioning latents must match target latent height/width. "
                    f"Got {tuple(condition_latents.shape[3:])} vs {tuple(noisy_latents.shape[3:])}."
                )
            condition_noise = prepared_batch.get("h3_conditioning_noise")
            if condition_noise is None:
                condition_noise = torch.randn_like(condition_latents)
            else:
                condition_noise = condition_noise.to(device=condition_latents.device, dtype=condition_latents.dtype)
            condition_noisy = (
                MINIMAX_H3_KEYFRAME_NOISE_AUG * condition_latents + (1.0 - MINIMAX_H3_KEYFRAME_NOISE_AUG) * condition_noise
            )
            packed_condition_video = patchify_video_latents(condition_noisy, patch_size).view(
                batch_size, -1, channels * patch_product
            )
            packed_video = torch.cat([packed_condition_video, packed_target_video], dim=1)
            keyframe_anchors = ("first",)
        else:
            packed_video = packed_target_video

        if use_audio:
            num_audio_latents = audio_noisy.shape[-1]
            packed_audio = self._pack_audio_latents(audio_noisy)
        else:
            num_audio_latents = 0
            packed_audio = encoder_hidden_states.new_empty((batch_size, 0, self._audio_latent_channels()))
        layout = build_packed_sequence(
            text_token_tags=text_token_tags.detach().cpu(),
            num_latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            num_audio_latents=num_audio_latents,
            patch_size=patch_size,
            keyframe_anchors=keyframe_anchors,
        )
        video_timesteps = self._batch_timestep_values(prepared_batch["timesteps"], batch_size, "video")
        if use_audio:
            audio_timesteps = self._batch_timestep_values(
                prepared_batch.get("audio_timesteps", prepared_batch["timesteps"]),
                batch_size,
                "audio",
            )
        else:
            audio_timesteps = video_timesteps
        flowmap_r_timesteps = prepared_batch.get(self.FLOWMAP_R_TIMESTEP_BATCH_KEY)
        r_timestep = None
        if flowmap_r_timesteps is not None:
            video_r_timesteps = self._batch_timestep_values(flowmap_r_timesteps, batch_size, "video r")
            if use_audio:
                video_r_sigma = 1.0 - video_r_timesteps
                audio_r_sigma = self._shift_sigmas_between_schedules(
                    video_r_sigma,
                    float(getattr(self.config, "flow_schedule_shift", 12.0) or 12.0),
                    float(getattr(self.config, "audio_flow_schedule_shift", 3.0) or 3.0),
                )
                audio_r_timesteps = 1.0 - audio_r_sigma
            else:
                audio_r_timesteps = video_r_timesteps
            timestep, r_timestep, timestep_indices = self._batched_row_timestep_intervals(
                layout,
                video_timesteps,
                audio_timesteps,
                video_r_timesteps,
                audio_r_timesteps,
            )
        else:
            timestep, timestep_indices = self._batched_row_timesteps(
                layout,
                video_timesteps,
                audio_timesteps,
            )

        token_tags = layout.token_tags.to(self.accelerator.device)
        position_ids = layout.position_ids.to(self.accelerator.device)
        video_indices = layout.video_indices.to(self.accelerator.device)
        audio_indices = layout.audio_indices.to(self.accelerator.device)
        text_indices = layout.text_indices.to(self.accelerator.device)
        packed_valid_mask = None
        if text_valid_mask is not None:
            packed_valid_mask = torch.ones(
                (batch_size, layout.sequence_length),
                device=self.accelerator.device,
                dtype=torch.bool,
            )
            packed_valid_mask[:, text_indices] = text_valid_mask
            text_lengths = text_valid_mask.sum(dim=1)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1).clone()
            position_ids[:, text_seq_len:, 0] += (text_lengths - text_seq_len).to(position_ids.dtype).unsqueeze(1)
        timestep = timestep.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        if r_timestep is not None:
            r_timestep = r_timestep.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        timestep_indices = timestep_indices.to(self.accelerator.device)
        if layout.num_condition_video_rows:
            force_keep_mask = torch.zeros(layout.sequence_length, device=self.accelerator.device, dtype=torch.bool)
            force_keep_mask[video_indices[: layout.num_condition_video_rows]] = True

        hidden_states_buffer = self._new_hidden_state_buffer()
        crepa = getattr(self, "crepa_regularizer", None)
        capture_block_index = prepared_batch.get(
            "crepa_capture_block_index",
            getattr(crepa, "block_index", None),
        )
        capture_hidden = bool(crepa and crepa.wants_hidden_states() and capture_block_index is not None)
        video_hidden_shape = (
            latent_frames // patch_size[0],
            latent_height // patch_size[1],
            latent_width // patch_size[2],
        )

        transformer_kwargs = {
            "hidden_states": packed_video,
            "audio_hidden_states": packed_audio,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "timestep_indices": timestep_indices,
            "token_tags": token_tags,
            "position_ids": position_ids,
            "video_indices": video_indices,
            "audio_indices": audio_indices,
            "text_indices": text_indices,
            "packed_valid_mask": packed_valid_mask,
            "attention_kwargs": {},
            "skip_layers": prepared_batch.get("skip_layers"),
            "force_keep_mask": force_keep_mask,
            "hidden_states_buffer": hidden_states_buffer,
            "output_hidden_states": capture_hidden,
            "hidden_state_layer": capture_block_index,
            "video_hidden_shape": video_hidden_shape,
            "num_condition_video_rows": layout.num_condition_video_rows,
            "num_condition_audio_rows": layout.num_condition_audio_rows,
            "minimax_h3_reference_mode": getattr(self.config, "minimax_h3_reference_mode", "vanilla") or "vanilla",
            "return_dict": True,
        }
        if getattr(self.config, "twinflow_enabled", False):
            transformer_kwargs["timestep_sign"] = prepared_batch.get("twinflow_time_sign")
        if r_timestep is not None:
            transformer_kwargs[self.FLOWMAP_R_TIMESTEP_KWARG] = r_timestep
        else:
            self._apply_flowmap_r_timestep_kwargs(transformer_kwargs, prepared_batch)
        output = self.model(**transformer_kwargs)
        video_rows = output.sample[:, layout.num_condition_video_rows :, :]
        video_pred = unpatchify_video_tokens(
            video_rows,
            num_latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            channels=channels,
            patch_size=patch_size,
        )
        audio_pred = self._unpack_audio_prediction(output.audio_sample, num_audio_latents) if use_audio else None
        return {
            "model_prediction": video_pred,
            "audio_prediction": audio_pred,
            "crepa_hidden_states": output.crepa_hidden_states,
            "hidden_states_buffer": hidden_states_buffer,
        }

    def get_prediction_target(self, prepared_batch: dict):
        if prepared_batch.get("target") is not None:
            return prepared_batch["target"]
        return self.get_flow_matching_target(prepared_batch, prefer_explicit_target=False)

    def loss(self, prepared_batch: dict, model_output, apply_conditioning_mask: bool = True):
        total_loss, _, _, _ = self._compute_av_loss(
            prepared_batch=prepared_batch,
            model_output=model_output,
            apply_conditioning_mask=apply_conditioning_mask,
        )
        return total_loss

    def loss_with_logs(self, prepared_batch: dict, model_output, apply_conditioning_mask: bool = True):
        total_loss, video_loss, audio_loss, audio_weight = self._compute_av_loss(
            prepared_batch=prepared_batch,
            model_output=model_output,
            apply_conditioning_mask=apply_conditioning_mask,
        )
        logs = {"video_loss": video_loss.detach().item()}
        if audio_loss is not None:
            logs["audio_loss"] = audio_loss.detach().item()
            if audio_weight != 1.0:
                logs["audio_loss_weighted"] = (audio_loss * audio_weight).detach().item()
        return total_loss, logs

    def _compute_av_loss(self, prepared_batch: dict, model_output, apply_conditioning_mask: bool = True):
        video_loss = super().loss(
            prepared_batch,
            model_output,
            apply_conditioning_mask=apply_conditioning_mask,
        )
        if os.environ.get("SIMPLETUNER_MINIMAXH3_LOSS_DEBUG", "0") == "1":
            video_pred = model_output.get("model_prediction")
            latents = prepared_batch.get("latents")
            noise = prepared_batch.get("noise")
            if torch.is_tensor(video_pred) and torch.is_tensor(latents) and torch.is_tensor(noise):
                with torch.no_grad():
                    current_target = latents - noise
                    flipped_target = noise - latents
                    current_loss = (video_pred.detach().float() - current_target.float()).pow(2).mean()
                    flipped_loss = (video_pred.detach().float() - flipped_target.float()).pow(2).mean()
                    zero_loss = current_target.float().pow(2).mean()
                    pred_flat = video_pred.detach().float().flatten()
                    target_flat = current_target.float().flatten()
                    pred_target_dot = torch.dot(pred_flat, target_flat)
                    cosine = pred_target_dot / (pred_flat.norm() * target_flat.norm()).clamp_min(1e-12)
                    sigmas = prepared_batch.get("sigmas")
                    timesteps = prepared_batch.get("timesteps")
                    sigma_mean = sigmas.detach().float().mean().item() if torch.is_tensor(sigmas) else float("nan")
                    timestep_mean = timesteps.detach().float().mean().item() if torch.is_tensor(timesteps) else float("nan")
                    conditioning_latents = prepared_batch.get("conditioning_latents")
                    conditioning_shape = "none"
                    conditioning_mean = float("nan")
                    conditioning_std = float("nan")
                    if isinstance(conditioning_latents, list):
                        conditioning_shape = f"list[{len(conditioning_latents)}]"
                        first_conditioning = conditioning_latents[0] if conditioning_latents else None
                        if torch.is_tensor(first_conditioning):
                            conditioning_mean = first_conditioning.detach().float().mean().item()
                            conditioning_std = first_conditioning.detach().float().std().item()
                    elif torch.is_tensor(conditioning_latents):
                        conditioning_shape = str(tuple(conditioning_latents.shape))
                        conditioning_mean = conditioning_latents.detach().float().mean().item()
                        conditioning_std = conditioning_latents.detach().float().std().item()
                    logger.info(
                        "MiniMax-H3 loss debug: current_target_mse=%.6f flipped_target_mse=%.6f "
                        "zero_target_mse=%.6f pred_target_cosine=%.6f pred_target_dot=%.6f "
                        "pred_mean=%.6f pred_std=%.6f target_mean=%.6f target_std=%.6f "
                        "latents_mean=%.6f latents_std=%.6f noise_std=%.6f sigma_mean=%.6f "
                        "timestep_mean=%.6f conditioning_shape=%s conditioning_mean=%.6f "
                        "conditioning_std=%.6f conditioning_type=%s",
                        current_loss.item(),
                        flipped_loss.item(),
                        zero_loss.item(),
                        cosine.item(),
                        pred_target_dot.item(),
                        video_pred.detach().float().mean().item(),
                        video_pred.detach().float().std().item(),
                        current_target.float().mean().item(),
                        current_target.float().std().item(),
                        latents.detach().float().mean().item(),
                        latents.detach().float().std().item(),
                        noise.detach().float().std().item(),
                        sigma_mean,
                        timestep_mean,
                        conditioning_shape,
                        conditioning_mean,
                        conditioning_std,
                        prepared_batch.get("conditioning_latents_type"),
                    )
        audio_pred = model_output.get("audio_prediction")
        if audio_pred is None:
            return video_loss, video_loss, None, 0.0
        audio_target = prepared_batch.get("audio_target")
        if audio_target is not None:
            if not torch.is_tensor(audio_target):
                raise ValueError(f"MiniMax-H3 audio_target must be a tensor, got {type(audio_target)}.")
            audio_target = audio_target.to(device=audio_pred.device, dtype=audio_pred.dtype)
            if audio_target.shape != audio_pred.shape:
                raise ValueError(
                    f"MiniMax-H3 audio_target shape {tuple(audio_target.shape)} does not match "
                    f"audio_prediction shape {tuple(audio_pred.shape)}."
                )
            audio_target = audio_target.detach()
        else:
            audio_latents = prepared_batch.get("audio_latents")
            audio_noise = prepared_batch.get("audio_noise")
            if audio_latents is None or audio_noise is None:
                return video_loss, video_loss, None, 0.0
            audio_target = audio_latents - audio_noise
        weight = float(getattr(self.config, "audio_loss_weight", 1.0) or 1.0)
        if weight == 0.0:
            return video_loss, video_loss, None, weight
        audio_mask = prepared_batch.get("audio_latent_mask")
        if audio_mask is not None:
            if torch.all(audio_mask == 0):
                return video_loss, video_loss, None, weight
            mask = audio_mask.view(audio_mask.shape[0], *([1] * (audio_pred.ndim - 1)))
            audio_pred = torch.where(mask > 0, audio_pred, torch.zeros_like(audio_pred))
            audio_target = torch.where(mask > 0, audio_target, torch.zeros_like(audio_target))
        audio_loss = (audio_pred.float() - audio_target.float()) ** 2
        audio_loss = audio_loss.mean()
        return video_loss + audio_loss * weight, video_loss, audio_loss, weight

    def tread_init(self):
        from simpletuner.helpers.training.tread import TREADRouter

        tread_cfg = getattr(self.config, "tread_config", None)
        if not isinstance(tread_cfg, dict) or tread_cfg == {} or tread_cfg.get("routes") is None:
            logger.error("TREAD training requires you to configure the routes in the TREAD config")
            import sys

            sys.exit(1)

        self.unwrap_model(model=self.model).set_router(
            TREADRouter(
                seed=getattr(self.config, "seed", None) or 42,
                device=self.accelerator.device,
            ),
            tread_cfg["routes"],
        )
        logger.info("TREAD training is enabled for MiniMax-H3")

    def get_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True):
        _register_minimax_h3_diffusers_components()
        # Validation constructs the pipeline with load_base_model=False after preprocessing
        # has unloaded the VAEs. Decoders are still required by every non-latent pipeline call.
        vae = self.get_vae()
        self._load_audio_vae(move_to_device=True)
        if pipeline_type in self.pipelines:
            pipeline = self.pipelines[pipeline_type]
            component_name = "transformer_ref" if getattr(self.config, "model_flavour", None) == "ref2va" else "transformer"
            transformer = self.unwrap_model(self.model)
            if hasattr(pipeline, "update_components"):
                pipeline.update_components(
                    **{
                        component_name: transformer,
                        "vae": vae,
                        "audio_vae": self.audio_vae,
                    }
                )
            else:
                setattr(pipeline, component_name, transformer)
                setattr(pipeline, "vae", vae)
                setattr(pipeline, "audio_vae", self.audio_vae)
            return pipeline
        if pipeline_type not in self.PIPELINE_CLASSES:
            raise NotImplementedError(f"Pipeline type {pipeline_type} not defined in {self.__class__.__name__}.")
        if load_base_model:
            if self.model is None:
                self.load_model(move_to_device=True)
            if self.text_encoders is None:
                self.load_text_encoder(move_to_device=True)
            if self.tokenizers is None:
                self.load_text_tokenizer()
        processor = self._load_processor_for_pipeline()
        transformer = self.unwrap_model(self.model) if self.model is not None else None
        is_ref = getattr(self.config, "model_flavour", None) == "ref2va"
        pipeline_class = self.PIPELINE_CLASSES[pipeline_type]
        blocks = MiniMaxH3Ref2VABlocks() if is_ref else MiniMaxH3Blocks()
        component_kwargs = {
            "vae": vae,
            "audio_vae": self.audio_vae,
            "scheduler": MiniMaxH3Scheduler(shift=float(getattr(self.config, "flow_schedule_shift", 12.0) or 12.0)),
            "audio_scheduler": MiniMaxH3Scheduler(
                shift=float(getattr(self.config, "audio_flow_schedule_shift", 3.0) or 3.0)
            ),
            "text_encoder": self.text_encoders[0] if self.text_encoders else None,
            "tokenizer": self.tokenizers[0] if self.tokenizers else None,
            "processor": processor,
            "video_processor": VideoProcessor(
                do_resize=False,
                do_normalize=False,
                vae_scale_factor=16,
                vae_latent_channels=self.LATENT_CHANNEL_COUNT,
            ),
        }
        component_kwargs["transformer_ref" if is_ref else "transformer"] = transformer
        pipeline = pipeline_class(
            blocks=blocks,
            pretrained_model_name_or_path=self._model_config_path(),
        )
        pipeline.update_components(
            **{name: component for name, component in component_kwargs.items() if component is not None}
        )
        self.pipelines[pipeline_type] = pipeline
        return pipeline


ModelRegistry.register("minimaxh3", MiniMaxH3)
