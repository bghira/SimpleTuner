import contextlib
import copy
import inspect
import json
import logging
import math
import os
import random
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline

    from simpletuner.helpers.acceleration import AccelerationPreset

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.distributions import Beta
from torchvision import transforms

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    FSDP_AVAILABLE = True
except ImportError:
    FSDP_AVAILABLE = False

from simpletuner.diff2flow import DiffusionToFlowBridge
from simpletuner.helpers.assistant_lora import build_adapter_stack, set_adapter_stack
from simpletuner.helpers.models.foundation_mixins import (
    AudioTransformMixin,
    PipelineSupportMixin,
    VaeLatentScalingMixin,
    VideoTransformMixin,
)
from simpletuner.helpers.models.tae import load_tae_decoder
from simpletuner.helpers.scheduled_sampling import build_rollout_schedule
from simpletuner.helpers.training.adapter import load_lora_weights
from simpletuner.helpers.training.crepa import CrepaRegularizer
from simpletuner.helpers.training.custom_schedule import (
    apply_flow_schedule_shift,
    generate_timestep_weights,
    segmented_timestep_selection,
)
from simpletuner.helpers.training.deepspeed import deepspeed_zero_init_disabled_context_manager, prepare_model_for_deepspeed
from simpletuner.helpers.training.layersync import LayerSyncRegularizer
from simpletuner.helpers.training.lora_format import (
    PEFTLoRAFormat,
    convert_comfyui_to_diffusers,
    convert_diffusers_to_comfyui,
    detect_state_dict_format,
    normalize_lora_format,
)
from simpletuner.helpers.training.min_snr_gamma import compute_snr
from simpletuner.helpers.training.multi_process import _get_rank
from simpletuner.helpers.training.quantisation import (
    PIPELINE_ONLY_PRESETS,
    PIPELINE_QUANTIZATION_PRESETS,
    build_gguf_quantization_config,
    get_pipeline_quantization_builder,
)
from simpletuner.helpers.training.state_tracker import StateTracker
from simpletuner.helpers.training.wrappers import unwrap_model
from simpletuner.helpers.utils import ramtorch as ramtorch_utils
from simpletuner.helpers.utils.hidden_state_buffer import HiddenStateBuffer
from simpletuner.helpers.utils.offloading import enable_group_offload_on_components

logger = logging.getLogger(__name__)
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


flow_matching_model_families = [
    "flux",
    "flux2",
    "sana",
    "ltxvideo",
    "ltxvideo2",
    "wan",
    "wan_s2v",
    "sd3",
    "chroma",
    "hunyuanvideo",
    "longcat_image",
    "longcat_video",
    "auraflow",
    "qwen_image",
    "z_image",
    "z_image_omni",
]
upstream_config_sources = {
    "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
    "kolors": "terminusresearch/kwai-kolors-1.0",
    "sd3": "stabilityai/stable-diffusion-3.5-large",
    "sana": "terminusresearch/sana-1.6b-1024px",
    "flux": "black-forest-labs/flux.1-dev",
    "flux2": "black-forest-labs/flux.2-dev",
    "chroma": "lodestones/Chroma1-Base",
    "sd1x": "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "sd2x": "stabilityai/stable-diffusion-v2-1",
    "ltxvideo": "Lightricks/LTX-Video",
    "ltxvideo2": "Lightricks/LTX-2",
    "wan": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "hunyuanvideo": "tencent/HunyuanVideo-1.5",
}


def get_model_config_path(model_family: str, model_path: str):
    if model_path is not None and model_path.endswith((".safetensors", ".gguf")):
        if model_family in upstream_config_sources:
            return upstream_config_sources[model_family]
        else:
            raise ValueError(
                "Cannot find noise schedule config for single-file checkpoint in architecture {}".format(model_family)
            )

    return model_path


def get_hf_cache_repo_path(model_path: str) -> Optional[str]:
    """
    Resolve a model path to its HuggingFace cache repository path.

    For Hub models (org/model-name), finds the corresponding cache directory.
    For local paths, returns None (nothing to delete).
    For single-file checkpoints (.safetensors, .gguf), returns None.

    Returns the repo-level cache path (e.g., ~/.cache/huggingface/hub/models--org--name)
    which contains blobs, refs, and snapshots directories.
    """
    if model_path is None:
        return None

    # Skip single-file checkpoints
    if isinstance(model_path, str) and model_path.endswith((".safetensors", ".gguf")):
        return None

    # Skip local paths
    if isinstance(model_path, str) and os.path.exists(model_path):
        # Check if it's already a local directory (not in HF cache)
        from huggingface_hub.constants import HF_HUB_CACHE

        hf_cache = os.environ.get("HF_HUB_CACHE", HF_HUB_CACHE)
        if not model_path.startswith(hf_cache):
            return None

    # Try to find the repo in the HF cache
    try:
        from huggingface_hub import scan_cache_dir
        from huggingface_hub.constants import HF_HUB_CACHE

        hf_cache = os.environ.get("HF_HUB_CACHE", HF_HUB_CACHE)
        cache_info = scan_cache_dir(hf_cache)

        # Normalize model_path to repo_id format
        repo_id = model_path.strip("/")

        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                return str(repo.repo_path)

    except Exception as e:
        logger.debug(f"Could not resolve HF cache path for {model_path}: {e}")

    return None


def delete_model_from_cache(
    component_key: str,
    accelerator,
    force: bool = False,
) -> bool:
    """
    Delete a model component's cached files from the HuggingFace cache.

    This function is gated to only run on local-rank 0 processes to avoid
    race conditions on shared/network storage. Deletion failures are silently
    ignored to handle cases where multiple nodes share the same network mount.

    Args:
        component_key: The StateTracker key for the component (e.g., "transformer", "vae")
        accelerator: The accelerator instance for checking local_rank
        force: If True, delete even without checking delete_model_after_load config

    Returns:
        True if deletion was attempted (regardless of success), False if skipped
    """
    import shutil

    from simpletuner.helpers.training.state_tracker import StateTracker

    # Only local-rank 0 should delete
    if not getattr(accelerator, "is_local_main_process", True):
        return False

    args = StateTracker.get_args()
    if not force and not getattr(args, "delete_model_after_load", False):
        return False

    repo_path = StateTracker.get_model_snapshot_path(component_key)
    if not repo_path:
        logger.debug(f"No cached path found for {component_key}, skipping deletion")
        return False

    # Skip pending placeholders (model path, not actual cache path)
    if repo_path.startswith("__pending__:"):
        logger.debug(f"Path for {component_key} is pending placeholder, skipping")
        StateTracker.clear_model_snapshot_path(component_key)
        return False

    logger.info(f"Deleting cached model files for {component_key}: {repo_path}")

    try:
        if os.path.isdir(repo_path):
            shutil.rmtree(repo_path)
            logger.info(f"Successfully deleted cache for {component_key}")
            StateTracker.clear_model_snapshot_path(component_key)
            return True
        elif os.path.isfile(repo_path):
            os.remove(repo_path)
            logger.info(f"Successfully deleted cache file for {component_key}")
            StateTracker.clear_model_snapshot_path(component_key)
            return True
        else:
            # Path doesn't exist (already deleted or never existed)
            logger.debug(f"Cache path does not exist for {component_key}: {repo_path}")
            StateTracker.clear_model_snapshot_path(component_key)
            return False
    except Exception as e:
        # Silently ignore deletion errors (race conditions on shared storage)
        logger.debug(f"Could not delete cache for {component_key}: {e}")
        StateTracker.clear_model_snapshot_path(component_key)
        return False


def delete_all_model_caches(accelerator) -> int:
    """
    Delete all registered model caches after all models have been loaded.

    This is called once after VAE, text encoders, and transformer/unet are all loaded
    and the text encoders have been unloaded (their paths cleared). This ensures
    that shared repos (like unified pipelines) are only deleted after all components
    that use them have loaded.

    Args:
        accelerator: The accelerator instance for checking local_rank

    Returns:
        Number of components whose caches were deleted
    """
    from simpletuner.helpers.training.state_tracker import StateTracker

    args = StateTracker.get_args()
    delete_enabled = getattr(args, "delete_model_after_load", False)
    logger.info(f"delete_all_model_caches called, delete_model_after_load={delete_enabled}")

    if not delete_enabled:
        return 0

    # Only local-rank 0 should delete
    if not getattr(accelerator, "is_local_main_process", True):
        logger.info("delete_all_model_caches: not local main process, skipping")
        return 0

    # Log all registered paths
    all_paths = StateTracker.get_all_model_snapshot_paths()
    logger.info(f"delete_all_model_caches: registered paths = {all_paths}")

    deleted_count = 0

    # Delete remaining component caches (VAE may be kept if validation is enabled)
    for component_key in ["vae", "transformer", "unet"]:
        # Check if VAE should be kept for validation
        if component_key == "vae":
            validation_enabled = (
                getattr(args, "validation_prompts", None)
                or getattr(args, "validation_prompt", None)
                or getattr(args, "validation_image_prompts", None)
            )
            if validation_enabled:
                logger.info("Skipping VAE cache deletion because validation is enabled")
                StateTracker.clear_model_snapshot_path("vae")
                continue

        if delete_model_from_cache(component_key, accelerator):
            deleted_count += 1

    logger.info(f"delete_all_model_caches: deleted {deleted_count} caches")
    return deleted_count


class PipelineTypes(Enum):
    IMG2IMG = "img2img"
    TEXT2IMG = "text2img"
    TEXT2AUDIO = "text2audio"
    IMG2VIDEO = "img2video"
    CONTROLNET = "controlnet"
    CONTROL = "control"


class PredictionTypes(Enum):
    EPSILON = "epsilon"
    SAMPLE = "sample"
    V_PREDICTION = "v_prediction"
    FLOW_MATCHING = "flow_matching"
    AUTOREGRESSIVE_NEXT_TOKEN = "autoregressive_next_token"

    @staticmethod
    def from_str(label):
        if label in ("eps", "epsilon"):
            return PredictionTypes.EPSILON
        elif label in ("vpred", "v_prediction", "v-prediction"):
            return PredictionTypes.V_PREDICTION
        elif label in ("sample", "x_prediction", "x-prediction"):
            return PredictionTypes.SAMPLE
        elif label in ("flow", "flow_matching", "flow-matching"):
            return PredictionTypes.FLOW_MATCHING
        elif label in ("ar", "autoregressive", "autoregressive_next_token"):
            return PredictionTypes.AUTOREGRESSIVE_NEXT_TOKEN
        else:
            raise NotImplementedError


class ModelTypes(Enum):
    UNET = "unet"
    TRANSFORMER = "transformer"
    VAE = "vae"
    TEXT_ENCODER = "text_encoder"


class TextEmbedCacheKey(Enum):
    CAPTION = "caption"
    FILENAME = "filename"
    DATASET_AND_FILENAME = "dataset_and_filename"


class PipelineConditioningImageEmbedder:
    """Wraps a Diffusers pipeline to expose a simple conditioning image encode interface."""

    def __init__(self, pipeline, image_encoder, image_processor, device=None, weight_dtype=None):
        if image_encoder is None or image_processor is None:
            raise ValueError("PipelineConditioningImageEmbedder requires both an image encoder and image processor.")
        self.pipeline = pipeline
        self.image_encoder = image_encoder
        self.image_processor = image_processor
        self.device = device if device is not None else torch.device("cpu")
        if isinstance(weight_dtype, str):
            weight_dtype = getattr(torch, weight_dtype, None)
        self.weight_dtype = weight_dtype

        if self.weight_dtype is not None:
            self.image_encoder.to(self.device, dtype=self.weight_dtype)
        else:
            self.image_encoder.to(self.device)
        self.image_encoder.eval()

    def encode(self, images):
        inputs = self.image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.image_encoder(**inputs, output_hidden_states=True)
        embeddings = None
        hidden_states = getattr(outputs, "hidden_states", None)
        if isinstance(hidden_states, (list, tuple)) and len(hidden_states) > 1:
            embeddings = hidden_states[-2]
        elif getattr(outputs, "last_hidden_state", None) is not None:
            embeddings = outputs.last_hidden_state
        elif torch.is_tensor(outputs):
            embeddings = outputs
        if embeddings is None:
            raise ValueError("Image encoder did not return hidden states suitable for conditioning embeds.")
        if self.weight_dtype is not None:
            embeddings = embeddings.to(self.weight_dtype)
        return embeddings


class ModelFoundation(ABC):
    """
    Base class that contains all the universal logic:
      - Noise schedule, prediction target (epsilon, sample, v_prediction, flow-matching)
      - Batch preparation (moving to device, sampling noise, etc.)
      - Loss calculation (including optional SNR weighting)
    """

    MODEL_LICENSE = "other"
    CONTROLNET_LORA_STATE_DICT_PREFIX = "controlnet"
    MAXIMUM_CANVAS_SIZE = None
    SUPPORTS_LORA = None
    SUPPORTS_CONTROLNET = None
    STRICT_I2V_FLAVOURS = tuple()
    STRICT_I2V_FOR_ALL_FLAVOURS = False
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    ASSISTANT_LORA_PATH = None
    ASSISTANT_LORA_FLAVOURS = None
    DEFAULT_CONTROLNET_LORA_TARGET = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
        "proj_in",
        "proj_out",
        "conv",
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "linear_1",
        "linear_2",
        "time_emb_proj",
        "controlnet_cond_embedding.conv_in",
        "controlnet_cond_embedding.blocks.0",
        "controlnet_cond_embedding.blocks.1",
        "controlnet_cond_embedding.blocks.2",
        "controlnet_cond_embedding.blocks.3",
        "controlnet_cond_embedding.blocks.4",
        "controlnet_cond_embedding.blocks.5",
        "controlnet_cond_embedding.conv_out",
        "controlnet_down_blocks.0",
        "controlnet_down_blocks.1",
        "controlnet_down_blocks.2",
        "controlnet_down_blocks.3",
        "controlnet_down_blocks.4",
        "controlnet_down_blocks.5",
        "controlnet_down_blocks.6",
        "controlnet_down_blocks.7",
        "controlnet_down_blocks.8",
        "controlnet_mid_block",
    ]
    DEFAULT_SLIDER_LORA_TARGET = [
        "attn1.to_q",
        "attn1.to_k",
        "attn1.to_v",
        "attn1.to_out.0",
        "attn1.to_qkv",
        "to_qkv",
        "proj_in",
        "proj_out",
        "conv_in",
        "conv_out",
        "time_embedding.linear_1",
        "time_embedding.linear_2",
    ]
    SLIDER_LORA_TARGET = DEFAULT_SLIDER_LORA_TARGET
    DEFAULT_LYCORIS_TARGET = ["Attention", "FeedForward"]
    VALIDATION_USES_NEGATIVE_PROMPT = False
    AUTO_LORA_FORMAT_DETECTION = False
    SUPPORTS_MUON_CLIP = False
    DEFAULT_AUDIO_CHANNELS = 1
    DEFAULT_LORA_EXCLUDE_TARGETS = None  # regex, not list

    # Acceleration backend support - models declare what they DON'T support
    UNSUPPORTED_BACKENDS: set = set()  # Empty = supports all backends

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        """Return the maximum number of blocks that can be swapped for Musubi block swap.

        Override in subclasses that support block swapping to return the model's
        maximum swappable block count. Returns None if unknown or varies by variant.
        """
        return None

    @classmethod
    def get_acceleration_presets(cls) -> list["AccelerationPreset"]:
        """Return model-specific acceleration presets.

        Each model defines its own presets with model-appropriate levels,
        target modules, block counts, and tradeoff descriptions.
        Override in subclasses to provide model-specific presets.
        """
        return []

    def __init__(self, config: dict, accelerator):
        self.config = config
        self.accelerator = accelerator
        self.noise_schedule = None
        self.pipelines = {}
        self._qkv_projections_fused = False
        self._validation_preview_decoder = None
        self._validation_preview_decoder_failed = False
        self.setup_model_flavour()
        self.setup_training_noise_schedule()
        self.diff2flow_bridge = None
        self.setup_diff2flow_bridge()
        self._maybe_enable_reflexflow_default()
        self.assistant_adapter_name = "assistant"
        self.assistant_lora_loaded = False
        self.pipeline_quantization_active = False
        self._twinflow_prediction_warning = False
        self._twinflow_diffusion_bridge = False
        self._twinflow_store_rng = False
        self._twinflow_allow_student_teacher = False
        self._twinflow_requires_ema = False
        self.layersync_regularizer: Optional[LayerSyncRegularizer] = None
        self._validate_twinflow_config()

    def pack_text_embeddings_for_cache(self, embeddings):
        """
        Optional hook for models to compress or adjust text embeds before caching.
        Defaults to no-op.
        """
        return embeddings

    def enable_muon_clip_logging(self) -> None:
        """Override in subclasses to wire attention logit publishers when MuonClip is in use."""
        return

    def unpack_text_embeddings_from_cache(self, embeddings):
        """
        Optional hook for models to restore cached text embeds to training format.
        Defaults to no-op.
        """
        return embeddings

    def load_validation_models(self, pipeline=None, pipeline_type=None) -> None:
        """
        Optional hook for models to lazily load validation-only components.

        This is a no-op by default.
        """
        return

    def validation_audio_sample_rate(self) -> Optional[int]:
        """
        Optional hook for models that emit audio during validation.
        """
        return None

    def extract_validation_audio(self, pipeline_result: Any, expected_count: Optional[int] = None) -> Optional[list[Any]]:
        """
        Extract audio outputs from a pipeline result, normalising to a list of samples.
        """
        audio = None
        if hasattr(pipeline_result, "audio"):
            audio = pipeline_result.audio
        elif hasattr(pipeline_result, "audios"):
            audio = pipeline_result.audios

        if audio is None:
            return None

        if isinstance(audio, (list, tuple)):
            items = []
            for item in audio:
                if torch.is_tensor(item):
                    items.append(item.detach().cpu())
                else:
                    items.append(item)
        elif torch.is_tensor(audio) or isinstance(audio, np.ndarray):
            if torch.is_tensor(audio):
                audio = audio.detach().cpu()
            if getattr(audio, "ndim", 0) >= 3:
                items = [audio[idx] for idx in range(audio.shape[0])]
            else:
                items = [audio]
        else:
            logger.warning("Unsupported validation audio payload type %s", type(audio))
            return None

        if expected_count is not None and len(items) != expected_count:
            raise ValueError(f"Validation audio count ({len(items)}) does not match validation samples ({expected_count}).")

        return items

    @classmethod
    def supports_assistant_lora(cls, config=None) -> bool:
        """
        Indicates whether the model family can leverage a fixed assistant LoRA.
        When a config is provided, optionally gate support by flavour.
        """
        if config is not None and getattr(config, "disable_assistant_lora", False):
            return False
        has_default = bool(getattr(cls, "ASSISTANT_LORA_PATH", None))
        flavour_whitelist = getattr(cls, "ASSISTANT_LORA_FLAVOURS", None)
        if config is None or flavour_whitelist is None:
            return has_default or bool(flavour_whitelist)

        try:
            flavour = getattr(config, "model_flavour", None)
        except Exception:
            flavour = None
        if not flavour_whitelist:
            return has_default
        return flavour in flavour_whitelist

    def get_assistant_lora(self):
        """
        Returns the PEFT-enabled model component when an assistant adapter is present.
        """
        if not getattr(self, "assistant_lora_loaded", False):
            return None
        try:
            return self.get_trained_component(unwrap_model=False)
        except Exception:
            return None

    def configure_assistant_lora_for_training(self):
        """
        Activate the assistant adapter for training (frozen) alongside the trainable adapter.
        """
        if getattr(self.config, "disable_assistant_lora", False):
            return
        if not getattr(self, "assistant_lora_loaded", False):
            return
        trained_component = self.get_trained_component(unwrap_model=False)
        if trained_component is None:
            return

        from simpletuner.helpers.assistant_lora import set_adapter_stack

        assistant_weight = getattr(self.config, "assistant_lora_strength", 1.0)
        try:
            assistant_weight = float(assistant_weight)
        except Exception:
            assistant_weight = 1.0

        target_component = self.unwrap_model(trained_component)
        peft_config = getattr(target_component, "peft_config", {}) or {}
        has_default_adapter = isinstance(peft_config, dict) and "default" in peft_config
        lora_type = str(getattr(self.config, "lora_type", "standard")).lower()
        require_default = lora_type == "standard"
        include_default = has_default_adapter or require_default
        adapter_names, weight_arg, freeze_names = build_adapter_stack(
            peft_config=peft_config,
            assistant_adapter_name=self.assistant_adapter_name,
            assistant_weight=assistant_weight if assistant_weight != 0 else None,
            include_default=include_default,
            require_default=require_default,
        )

        if not adapter_names:
            return

        def _weight_for(idx: int) -> object:
            if isinstance(weight_arg, list):
                return weight_arg[idx]
            return weight_arg

        weight_summary = ", ".join(f"{name}={_weight_for(idx)}" for idx, name in enumerate(adapter_names))
        logger.info(f"Configuring assistant LoRA for training with weights: {weight_summary}")
        components = [
            target_component,
            trained_component if trained_component is not target_component else None,
        ]
        pipeline_component = getattr(getattr(self, "pipeline", None), self.MODEL_TYPE.value, None)
        if pipeline_component is not None:
            components.append(pipeline_component)
        seen: set[int] = set()
        for component in components:
            if component is None:
                continue
            if id(component) in seen:
                continue
            seen.add(id(component))
            set_adapter_stack(component, adapter_names, weights=weight_arg, freeze_names=freeze_names)

    def configure_assistant_lora_for_inference(self):
        """
        Configure the assistant adapter for validation/inference (typically disabled).
        """
        if getattr(self.config, "disable_assistant_lora", False):
            return
        if not getattr(self, "assistant_lora_loaded", False):
            return
        trained_component = self.get_trained_component(unwrap_model=False)
        if trained_component is None:
            return

        try:
            inference_weight = float(getattr(self.config, "assistant_lora_inference_strength", 0.0))
        except Exception:
            inference_weight = 0.0

        target_component = self.unwrap_model(trained_component)
        peft_config = getattr(target_component, "peft_config", {}) or {}
        has_default_adapter = isinstance(peft_config, dict) and "default" in peft_config
        lora_type = str(getattr(self.config, "lora_type", "standard")).lower()
        require_default = lora_type == "standard"
        include_default = has_default_adapter or require_default

        if inference_weight == 0:
            adapter_names, weight_arg, freeze_names = build_adapter_stack(
                peft_config=peft_config,
                assistant_adapter_name=self.assistant_adapter_name,
                assistant_weight=None,
                include_default=include_default,
                require_default=require_default,
            )
            if not adapter_names:
                return
            logger.info("Configuring assistant LoRA for inference with weights: default=1.0")
            components = [
                target_component,
                trained_component if trained_component is not target_component else None,
            ]
            pipeline_component = getattr(getattr(self, "pipeline", None), self.MODEL_TYPE.value, None)
            if pipeline_component is not None:
                components.append(pipeline_component)
            seen: set[int] = set()
            for component in components:
                if component is None:
                    continue
                if id(component) in seen:
                    continue
                seen.add(id(component))
                set_adapter_stack(component, adapter_names, weights=weight_arg, freeze_names=freeze_names)
            return

        adapter_names, weight_arg, freeze_names = build_adapter_stack(
            peft_config=peft_config,
            assistant_adapter_name=self.assistant_adapter_name,
            assistant_weight=inference_weight,
            include_default=include_default,
            require_default=require_default,
        )

        if not adapter_names:
            return

        def _weight_for(idx: int) -> object:
            if isinstance(weight_arg, list):
                return weight_arg[idx]
            return weight_arg

        weight_summary = ", ".join(f"{name}={_weight_for(idx)}" for idx, name in enumerate(adapter_names))
        logger.info(f"Configuring assistant LoRA for inference with weights: {weight_summary}")
        components = [
            target_component,
            trained_component if trained_component is not target_component else None,
        ]
        pipeline_component = getattr(getattr(self, "pipeline", None), self.MODEL_TYPE.value, None)
        if pipeline_component is not None:
            components.append(pipeline_component)
        seen: set[int] = set()
        for component in components:
            if component is None:
                continue
            if id(component) in seen:
                continue
            seen.add(id(component))
            set_adapter_stack(component, adapter_names, weights=weight_arg, freeze_names=freeze_names)

    @classmethod
    def supports_lora(cls) -> bool:
        """
        Indicates whether this model family supports LoRA fine-tuning.
        Subclasses may override. Defaults to False unless explicitly enabled.
        """
        if cls.SUPPORTS_LORA is not None:
            return bool(cls.SUPPORTS_LORA)
        return False

    @classmethod
    def supports_controlnet(cls) -> bool:
        """
        Indicates whether this model family supports ControlNet training.
        Subclasses may override. Defaults to False unless explicitly enabled.
        """
        if cls.SUPPORTS_CONTROLNET is not None:
            return bool(cls.SUPPORTS_CONTROLNET)
        return False

    @classmethod
    def supports_chunked_feed_forward(cls) -> bool:
        """
        Indicates whether feed-forward chunking can be enabled for this model family.
        Subclasses should override if chunking is supported.
        """
        return False

    # -------------------------------------------------------------------------
    # Data signal infrastructure
    # -------------------------------------------------------------------------
    # These flags are set by the factory after data backend configuration to
    # inform the model about what data types are present. Models can use these
    # signals to adjust behavior (e.g., LoRA target modules).

    _data_has_images: bool = False
    _data_has_video: bool = False
    _data_has_audio: bool = False

    def configure_data_signals(
        self,
        has_images: bool = False,
        has_video: bool = False,
        has_audio: bool = False,
    ) -> None:
        """
        Configure data type signals from the dataloader.

        Called by the factory after data backend configuration to inform the
        model about what data types are present in the training data. Models
        can override this to take action based on data types, or use the stored
        flags in other methods like get_lora_target_layers().

        Args:
            has_images: Whether image data is present
            has_video: Whether video data is present
            has_audio: Whether audio data is present
        """
        self._data_has_images = has_images
        self._data_has_video = has_video
        self._data_has_audio = has_audio

    def _get_additional_lora_targets(self) -> list[str]:
        """
        Return additional LoRA target modules based on data signals.

        Models can override this to add extra targets when certain data types
        are present. For example, a model with audio support might add audio
        projection layers when audio data is detected.

        Returns:
            List of additional module name patterns to include in LoRA targets.
            Empty list by default.
        """
        return []

    # -------------------------------------------------------------------------
    # LoRA/PEFT helpers (shared across model families)
    # -------------------------------------------------------------------------
    def get_lora_save_layers(self):
        return None

    def _get_peft_lora_target_modules(self):
        if str(getattr(self.config, "lora_type", "standard")).lower() != "standard":
            return None

        raw_targets = getattr(self.config, "peft_lora_target_modules", None)
        if raw_targets in (None, "", "None"):
            return None

        if not isinstance(raw_targets, (list, tuple)):
            raise ValueError(
                "peft_lora_target_modules must be a list of module name strings. " f"Received {type(raw_targets)}."
            )

        normalized = []
        for entry in raw_targets:
            if entry in (None, "", "None"):
                continue
            if not isinstance(entry, str):
                raise ValueError("peft_lora_target_modules entries must be strings. " f"Received {type(entry)}.")
            candidate = entry.strip()
            if candidate:
                normalized.append(candidate)

        return normalized or None

    def get_lora_target_layers(self):
        manual_targets = self._get_peft_lora_target_modules()
        if manual_targets:
            return manual_targets

        lora_type = getattr(self.config, "lora_type", "standard")
        base_targets = None
        if lora_type.lower() == "standard":
            if getattr(self.config, "slider_lora_target", False):
                slider_target = getattr(self, "SLIDER_LORA_TARGET", None) or getattr(
                    self, "DEFAULT_SLIDER_LORA_TARGET", None
                )
                if slider_target:
                    base_targets = slider_target
            if base_targets is None and getattr(self.config, "controlnet", False):
                base_targets = self.DEFAULT_CONTROLNET_LORA_TARGET
            if base_targets is None:
                base_targets = self.DEFAULT_LORA_TARGET
        elif lora_type.lower() == "lycoris":
            base_targets = self.DEFAULT_LYCORIS_TARGET
        else:
            raise NotImplementedError(f"Unknown LoRA target type {lora_type}.")

        additional_targets = self._get_additional_lora_targets()
        if not additional_targets:
            return base_targets

        combined = list(base_targets) if base_targets else []
        for target in additional_targets:
            if target not in combined:
                combined.append(target)
        return combined

    def add_lora_adapter(self):
        from peft import LoraConfig

        target_modules = self.get_lora_target_layers()
        save_modules = self.get_lora_save_layers()
        addkeys, misskeys = [], []

        if getattr(self.config, "controlnet", False) and getattr(self, "MODEL_TYPE", None) == ModelTypes.UNET:
            logger.warning(
                "ControlNet with UNet requires Conv2d layer support. "
                "Using LyCORIS (LoHa) adapter instead of standard LoRA."
            )
            from peft import LoHaConfig

            self.lora_config = LoHaConfig(
                r=self.config.lora_rank,
                alpha=(self.config.lora_alpha if self.config.lora_alpha is not None else self.config.lora_rank),
                rank_dropout=self.config.lora_dropout,
                module_dropout=0.0,
                use_effective_conv2d=True,
                target_modules=target_modules,
                exclude_modules=self.DEFAULT_LORA_EXCLUDE_TARGETS,
                modules_to_save=save_modules,
            )
        else:
            lora_config_cls = LoraConfig
            lora_config_kwargs = {}
            if getattr(self.config, "peft_lora_mode", None) is not None:
                if self.config.peft_lora_mode.lower() == "singlora":
                    from peft_singlora import SingLoRAConfig, setup_singlora

                    lora_config_cls = SingLoRAConfig
                    lora_config_kwargs = {
                        "ramp_up_steps": self.config.singlora_ramp_up_steps or 100,
                    }

                    logger.info("Enabling SingLoRA for LoRA training.")
                    setup_singlora()
            self.lora_config = lora_config_cls(
                r=self.config.lora_rank,
                lora_alpha=(self.config.lora_alpha if self.config.lora_alpha is not None else self.config.lora_rank),
                lora_dropout=self.config.lora_dropout,
                init_lora_weights=self.config.lora_initialisation_style,
                target_modules=target_modules,
                modules_to_save=save_modules,
                exclude_modules=self.DEFAULT_LORA_EXCLUDE_TARGETS,
                use_dora=getattr(self.config, "use_dora", False),
                **lora_config_kwargs,
            )

        if self._ramtorch_enabled():
            ramtorch_utils.register_lora_custom_module(self.lora_config)

        if getattr(self.config, "controlnet", False):
            self.controlnet.add_adapter(self.lora_config)
        else:
            self.model.add_adapter(self.lora_config)

        if getattr(self.config, "init_lora", None):
            use_dora = getattr(self.config, "use_dora", False) if isinstance(self.lora_config, LoraConfig) else False
            addkeys, misskeys = load_lora_weights(
                {self.MODEL_TYPE.value: (self.controlnet if getattr(self.config, "controlnet", False) else self.model)},
                self.config.init_lora,
                use_dora=use_dora,
            )

        return addkeys, misskeys

    def enable_chunked_feed_forward(self, *, chunk_size: Optional[int] = None, chunk_dim: int = 0) -> None:
        """
        Activate feed-forward chunking on the underlying model.
        Base implementation is a no-op and should be overridden by subclasses that support chunking.
        """
        return

    @classmethod
    def strict_i2v_flavours(cls):
        """
        Return flavour identifiers that require strict image-to-video validation inputs.
        """
        flavours = getattr(cls, "STRICT_I2V_FLAVOURS", tuple())
        if not flavours:
            return []
        if isinstance(flavours, (list, tuple, set, frozenset)):
            result = []
            for entry in flavours:
                try:
                    value = str(entry).strip()
                except Exception:
                    continue
                if value:
                    result.append(value)
            return list(dict.fromkeys(result))
        if isinstance(flavours, str):
            trimmed = flavours.strip()
            return [trimmed] if trimmed else []
        return []

    @classmethod
    def is_strict_i2v_flavour(cls, flavour) -> bool:
        """
        Determine whether the provided flavour requires strict image-to-video validation inputs.
        """
        if getattr(cls, "STRICT_I2V_FOR_ALL_FLAVOURS", False):
            return True
        if not flavour:
            return False
        try:
            candidate = str(flavour).strip().lower()
        except Exception:
            return False
        if not candidate:
            return False
        for entry in cls.strict_i2v_flavours():
            if entry.strip().lower() == candidate:
                return True
        return False

    def log_model_devices(self):
        """
        Log the devices of the model components.
        """
        if hasattr(self, "model") and self.model is not None:
            logger.debug(f"Model device: {self.model.device}")
        if hasattr(self, "vae") and self.vae is not None:
            logger.debug(f"VAE device: {self.vae.device}")
        if hasattr(self, "text_encoders") and self.text_encoders is not None:
            for i, text_encoder in enumerate(self.text_encoders):
                if text_encoder is None:
                    continue
                logger.debug(f"Text encoder {i} device: {text_encoder.device}")

    def setup_model_flavour(self):
        """
        Sets up the model flavour based on the config.
        This is used to determine the model path if none was provided.
        """
        if getattr(self, "REQUIRES_FLAVOUR", False):
            if getattr(self.config, "model_flavour", None) is None:
                raise ValueError(
                    f"{str(self.__class__)} models require model_flavour to be provided."
                    f" Possible values: {self.HUGGINGFACE_PATHS.keys()}"
                )
        if self.config.pretrained_model_name_or_path is None:
            if self.config.model_flavour is None:
                default_flavour = getattr(self, "DEFAULT_MODEL_FLAVOUR", None)
                if default_flavour is None and len(self.HUGGINGFACE_PATHS) > 0:
                    raise ValueError(
                        f"The current model family {self.config.model_family} requires a model_flavour to be provided. Options: {self.HUGGINGFACE_PATHS.keys()}"
                    )
                elif default_flavour is not None:
                    self.config.model_flavour = default_flavour
            if self.config.model_flavour is not None:
                if self.config.model_flavour not in self.HUGGINGFACE_PATHS:
                    raise ValueError(
                        f"Model flavour {self.config.model_flavour} not found in {self.HUGGINGFACE_PATHS.keys()}"
                    )
                self.config.pretrained_model_name_or_path = self.HUGGINGFACE_PATHS.get(self.config.model_flavour)
            else:
                raise ValueError(f"Model flavour {self.config.model_flavour} not found in {self.HUGGINGFACE_PATHS.keys()}")
        if self.config.pretrained_vae_model_name_or_path is None:
            self.config.pretrained_vae_model_name_or_path = self.config.pretrained_model_name_or_path
        if self.config.vae_path is None:
            self.config.vae_path = self.config.pretrained_model_name_or_path

    @abstractmethod
    def model_predict(self, prepared_batch, custom_timesteps: list = None):
        """
        Run a forward pass on the model.
        Must be implemented by the subclass.
        """
        raise NotImplementedError("model_predict must be implemented in the child class.")

    # -------------------------------------------------------------------------
    # Conditioning Capability Methods
    # -------------------------------------------------------------------------
    # These methods define how a model handles conditioning inputs (reference
    # images, control signals, etc.). The WebUI and training pipeline use these
    # to determine what UI elements to show and how to process data.
    #
    # There are two categories:
    #   - "requires_*" methods: Model CANNOT function without this capability.
    #     Training will fail if the requirement is not met.
    #   - "supports_*" methods: Model CAN use this capability but doesn't require
    #     it. The WebUI will show the option, but it's optional.
    #
    # Example model patterns:
    #   - Flux Kontext: requires_conditioning_dataset() = True (always needs refs)
    #   - Flux2: supports_conditioning_dataset() = True (optional dual T2I/I2I)
    #   - LTXVideo2: supports_conditioning_dataset() = True (optional I2V)
    #   - SD with ControlNet: requires_conditioning_dataset() = True (via config)
    # -------------------------------------------------------------------------

    def requires_conditioning_dataset(self) -> bool:
        """
        Returns True when the model REQUIRES a conditioning dataset to train.

        Override this to return True when:
        - The model architecture inherently needs conditioning inputs (e.g., edit models)
        - A config option like controlnet/control is enabled

        When True, the dataloader will fail if no conditioning dataset is configured.
        The WebUI will mark conditioning as mandatory.
        """
        if self.config.controlnet or self.config.control:
            return True
        return False

    def supports_conditioning_dataset(self) -> bool:
        """
        Returns True when the model SUPPORTS optional conditioning datasets.

        Override this to return True when:
        - The model can operate in both text-to-image AND image-to-image modes
        - Conditioning is useful but not mandatory (e.g., Flux2, LTXVideo2)

        When True and requires_conditioning_dataset() is False, the WebUI will
        show conditioning options without making them mandatory. This enables
        dual T2I/I2I training workflows.
        """
        return False

    def text_embed_cache_key(self) -> TextEmbedCacheKey:
        """
        Controls how prompt embeddings are keyed inside the cache. Most models can
        key by caption text; edit models can override to use filenames.
        """
        return TextEmbedCacheKey.CAPTION

    def requires_text_embed_image_context(self) -> bool:
        """
        Returns True when encode_text_batch must be supplied with per-prompt image
        context (e.g., reference pixels). Defaults to False.
        """
        return False

    def requires_conditioning_latents(self) -> bool:
        """
        Returns True when conditioning inputs should be VAE-encoded latents
        instead of raw pixel values.

        Override to True when:
        - The model processes conditioning through the latent space (e.g., Flux, Flux2)
        - ControlNet-style conditioning uses latent inputs

        When True, collate.py will collect VAE-encoded latents for conditioning
        instead of pixel tensors.
        """
        return False

    def requires_conditioning_image_embeds(self) -> bool:
        """
        Returns True when conditioning requires pre-computed image embeddings
        (e.g., from CLIP or similar vision encoder).

        Override to True for models that use image embeddings as conditioning
        signals rather than raw pixels or latents.
        """
        return False

    def supports_audio_inputs(self) -> bool:
        return False

    def uses_audio_latents(self) -> bool:
        return False

    def uses_audio_tokens(self) -> bool:
        """
        Override to True for autoregressive audio models that consume discrete token sequences
        instead of VAE latents.
        """
        return False

    def uses_text_embeddings_cache(self) -> bool:
        """
        Override to False for models that do not use text encoder embeddings.
        """
        return bool(getattr(self, "TEXT_ENCODER_CONFIGURATION", None))

    def uses_noise_schedule(self) -> bool:
        """
        Override to False for autoregressive models that do not use diffusion timesteps/sigmas.
        """
        return self.PREDICTION_TYPE is not PredictionTypes.AUTOREGRESSIVE_NEXT_TOKEN

    def get_vae_for_dataset_type(self, dataset_type: str):
        return self.get_vae()

    def conditioning_image_embeds_use_reference_dataset(self) -> bool:
        """
        Override to True when conditioning image embeds should be generated from the reference datasets
        instead of the primary training dataset.
        """
        return False

    def requires_validation_edit_captions(self) -> bool:
        """
        Some edit / in-painting models want the *reference* image plus the
        *edited* caption.  Override to return True when that is the case.
        """
        return False

    def requires_conditioning_validation_inputs(self) -> bool:
        """
        Returns True when validation requires conditioning inputs (images/latents).

        Override to True when:
        - The model needs reference images to generate meaningful validation outputs
        - Validation without conditioning would produce unusable results

        When True, the validation system will load images from configured
        validation datasets or use eval_dataset_id to source inputs.
        """
        return False

    def conditioning_validation_dataset_type(self) -> str:
        """
        Returns the dataset type to use for conditioning during validation.

        Common values:
        - "conditioning": Use datasets marked as conditioning type (default)
        - "image": Use standard image datasets (e.g., for edit models like Kontext)

        Override this when the model expects a specific dataset type for its
        conditioning inputs during validation.
        """
        return "conditioning"

    def validation_image_input_edge_length(self):
        # If a model requires a specific input edge length (HiDream E1 -> 768px, DeepFloyd stage2 -> 64px)
        return None

    def control_init(self):
        """
        Initialize the channelwise Control model.
        This is distinct from ControlNet.
        This is a stub and should be implemented in subclasses.
        """
        raise NotImplementedError("control_init must be implemented in the child class.")

    def controlnet_init(self):
        """
        Initialize the controlnet model.
        This is a stub and should be implemented in subclasses.
        """
        raise NotImplementedError("controlnet_init must be implemented in the child class.")

    def controlnet_predict(self, prepared_batch, custom_timesteps: list = None):
        """
        Run a forward pass on the model.
        Must be implemented by the subclass.
        """
        raise NotImplementedError("model_predict must be implemented in the child class.")

    def tread_init(self):
        """
        Initialize the TREAD model training method.
        This is a stub and should be implemented in subclasses.
        """
        raise NotImplementedError("tread_init must be implemented in the child class.")

    @abstractmethod
    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encodes a batch of text using the text encoder.
        Must be implemented by the subclass.
        """
        raise NotImplementedError("_encode_prompts must be implemented in the child class.")

    @abstractmethod
    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        """
        Converts the text embedding to the format expected by the pipeline.
        This is a stub and should be implemented in subclasses.
        """
        raise NotImplementedError("convert_text_embed_for_pipeline must be implemented in the child class.")

    @abstractmethod
    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        """
        Converts the text embedding to the format expected by the pipeline for negative prompt inputs.
        This is a stub and should be implemented in subclasses.
        """
        raise NotImplementedError("convert_text_embed_for_pipeline must be implemented in the child class.")

    def collate_prompt_embeds(self, text_encoder_output: dict) -> dict:
        """
        Optional stub method for client classes to do their own text embed collation/stacking.

        Returns a dictionary. If the dictionary is empty, it is ignored and usual collate occurs.
        """
        return {}

    def collate_audio_tokens(self, examples: list[dict]) -> dict:
        """
        Optional hook for autoregressive audio models to build token batches.

        Must return a dict containing at least:
        - tokens: LongTensor [batch, seq_len, num_codebooks + 1]
        - tokens_mask: BoolTensor [batch, seq_len, num_codebooks + 1]
        - audio_frame_mask: BoolTensor [batch, seq_len] where True marks audio frames
        """
        raise NotImplementedError("collate_audio_tokens must be implemented by the child class.")

    @classmethod
    def get_flavour_choices(cls):
        """
        Returns the available model flavours for this model.
        """
        return list(cls.HUGGINGFACE_PATHS.keys())

    def get_transforms(self, dataset_type: str = "image"):
        """
        Returns nothing, but subclasses can implement different torchvision transforms as needed.

        dataset_type is passed in for models that support transforming videos or images etc.
        """
        if dataset_type in ["video"]:
            raise ValueError(f"{dataset_type} transforms are not supported by {self.NAME}.")
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def _apply_alpha_scaling(self, module, alpha_map: dict, adapter_name: str = "default"):
        """
        Apply per-layer alpha values to loaded PEFT adapters when supplied by the checkpoint.
        """
        if module is None or not alpha_map:
            return

        try:
            from peft.tuners.tuners_utils import BaseTunerLayer
        except Exception:
            return

        for name, layer in module.named_modules():
            if not isinstance(layer, BaseTunerLayer):
                continue
            if adapter_name not in getattr(layer, "lora_alpha", {}):
                continue
            if name not in alpha_map:
                continue

            try:
                alpha_value = alpha_map[name]
                if torch.is_tensor(alpha_value):
                    alpha_value = alpha_value.detach().float().cpu().item()
                alpha_value = float(alpha_value)
            except Exception:
                continue

            layer.lora_alpha[adapter_name] = alpha_value
            if adapter_name in getattr(layer, "r", {}) and layer.r[adapter_name]:
                try:
                    scaling = (
                        alpha_value / math.sqrt(layer.r[adapter_name])
                        if getattr(layer, "use_rslora", False)
                        else alpha_value / layer.r[adapter_name]
                    )
                    layer.scaling[adapter_name] = scaling
                except Exception:
                    continue

    def load_lora_weights(self, models, input_dir):
        """
        Generalized LoRA loading method.
        1) Pop models from the 'models' list, detect which is main (denoiser) vs. text encoders.
        2) Pull the relevant LoRA keys out of the pipeline's lora_state_dict() output by prefix.
        3) Convert & load them into the unwrapped PyTorch modules with set_peft_model_state_dict().
        4) Optionally handle text_encoder_x using the diffusers _set_state_dict_into_text_encoder() helper.
        """
        denoiser = None
        text_encoder_one_ = None
        text_encoder_two_ = None

        while len(models) > 0:
            model = models.pop()
            unwrapped_model = self.unwrap_model(model)

            if isinstance(unwrapped_model, type(self.unwrap_model(self.model))):
                denoiser = model
            elif isinstance(unwrapped_model, type(self.unwrap_model(self.controlnet))):
                denoiser = model
            # If your text_encoders exist:
            elif (
                getattr(self, "text_encoders", None)
                and len(self.text_encoders) > 0
                and isinstance(unwrapped_model, type(self.unwrap_model(self.text_encoders[0])))
            ):
                text_encoder_one_ = model
            elif (
                getattr(self, "text_encoders", None)
                and len(self.text_encoders) > 1
                and isinstance(unwrapped_model, type(self.unwrap_model(self.text_encoders[1])))
            ):
                text_encoder_two_ = model
            else:
                raise ValueError(
                    f"Unexpected model type in load_lora_weights: {model.__class__}\n"
                    f"Unwrapped: {unwrapped_model.__class__}\n"
                    f"Expected main model type {type(self.unwrap_model(self.model))}"
                )

        pipeline_cls = self.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG]
        lora_state = pipeline_cls.lora_state_dict(input_dir)
        network_alphas = None

        if isinstance(lora_state, tuple):
            if len(lora_state) >= 2:
                lora_state_dict, maybe_network_alphas = lora_state[:2]
                if isinstance(maybe_network_alphas, dict) or maybe_network_alphas is None:
                    network_alphas = maybe_network_alphas
                else:
                    lora_state_dict = lora_state[0]
            else:
                lora_state_dict = lora_state[0]
        else:
            lora_state_dict = lora_state

        if isinstance(lora_state_dict, tuple) and len(lora_state_dict) == 2 and lora_state_dict[1] is None:
            logger.debug("Overriding ControlNet LoRA state dict with correct structure")
            lora_state_dict = lora_state_dict[0]

        key_to_replace = self.CONTROLNET_LORA_STATE_DICT_PREFIX if self.config.controlnet else self.MODEL_TYPE.value
        prefix = f"{key_to_replace}."

        if not isinstance(lora_state_dict, dict):
            raise ValueError("LoRA checkpoint did not return a state dictionary.")

        def _normalise_alpha_map(alpha_dict: dict) -> dict:
            if not alpha_dict:
                return {}
            results = {}
            prefixes = [prefix, "text_encoder.", "text_encoder_2."]
            for alpha_key, alpha_val in alpha_dict.items():
                key_name = alpha_key[:-6] if alpha_key.endswith(".alpha") else alpha_key
                try:
                    alpha_float = alpha_val.detach().float().cpu().item() if torch.is_tensor(alpha_val) else float(alpha_val)
                except Exception:
                    continue
                results[key_name] = alpha_float
                for pfx in prefixes:
                    if key_name.startswith(pfx):
                        results[key_name.replace(pfx, "", 1)] = alpha_float
            return results

        config_format = normalize_lora_format(getattr(self.config, "lora_format", None))
        active_format = config_format
        if getattr(self, "AUTO_LORA_FORMAT_DETECTION", False) and config_format == PEFTLoRAFormat.DIFFUSERS:
            detected_format = detect_state_dict_format(lora_state_dict)
            if detected_format == PEFTLoRAFormat.COMFYUI:
                logger.info("Detected ComfyUI-formatted LoRA checkpoint, converting to Diffusers/PEFT format.")
                active_format = PEFTLoRAFormat.COMFYUI
        alpha_map = {}
        if active_format == PEFTLoRAFormat.COMFYUI:
            lora_state_dict, comfy_alphas = convert_comfyui_to_diffusers(lora_state_dict, target_prefix=key_to_replace)
            alpha_map.update(_normalise_alpha_map(comfy_alphas))
        if network_alphas:
            alpha_map.update(_normalise_alpha_map(network_alphas))

        denoiser_sd = {}
        for k, v in lora_state_dict.items():
            if k.startswith(prefix):
                new_key = k.replace(prefix, "")
                denoiser_sd[new_key] = v

        from diffusers.utils import convert_unet_state_dict_to_peft

        denoiser_sd = convert_unet_state_dict_to_peft(denoiser_sd)

        from peft.utils import set_peft_model_state_dict

        incompatible_keys = set_peft_model_state_dict(denoiser, denoiser_sd, adapter_name="default")
        if alpha_map:
            self._apply_alpha_scaling(self.unwrap_model(denoiser), alpha_map)
            if text_encoder_one_ is not None:
                self._apply_alpha_scaling(self.unwrap_model(text_encoder_one_), alpha_map)
            if text_encoder_two_ is not None:
                self._apply_alpha_scaling(self.unwrap_model(text_encoder_two_), alpha_map)

        if incompatible_keys is not None:
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(f"LoRA loading found unexpected keys not in the denoiser model: {unexpected_keys}")

        if getattr(self.config, "train_text_encoder", False):
            from diffusers.training_utils import _set_state_dict_into_text_encoder

            # For text_encoder_1, the prefix in your pipeline's state dict is usually "text_encoder."
            # For text_encoder_2, it might be "text_encoder_2."
            # We'll do them in separate calls:

            if text_encoder_one_ is not None:
                _set_state_dict_into_text_encoder(
                    lora_state_dict,
                    prefix="text_encoder.",  # Must match how your pipeline outputs these
                    text_encoder=text_encoder_one_,
                )

            if text_encoder_two_ is not None:
                _set_state_dict_into_text_encoder(
                    lora_state_dict,
                    prefix="text_encoder_2.",  # Must match how your pipeline organizes these
                    text_encoder=text_encoder_two_,
                )

        logger.info("Finished loading LoRA weights successfully.")

    def save_lora_weights(self, *args, **kwargs):
        """
        Proxy to the pipeline save_lora_weights, ensuring save_directory is passed explicitly.
        """
        user_save_function = kwargs.pop("save_function", None)
        if args:
            save_directory, *remaining = args
        else:
            save_directory = kwargs.pop("save_directory", None)
            remaining = []
        if save_directory is None:
            raise ValueError("save_directory is required to save LoRA weights.")

        adapter_metadata = {}
        for key, value in kwargs.items():
            if key.endswith("lora_adapter_metadata") and isinstance(value, dict):
                adapter_metadata.update(value)

        lora_format = normalize_lora_format(getattr(self.config, "lora_format", None))
        if lora_format == PEFTLoRAFormat.COMFYUI and getattr(self, "NATIVE_COMFYUI_LORA_SUPPORT", False):
            logger.info("Skipping ComfyUI LoRA conversion - model has native ComfyUI support")
            lora_format = PEFTLoRAFormat.DIFFUSERS  # Treat as diffusers format (no-op)
        if lora_format == PEFTLoRAFormat.COMFYUI:
            import safetensors.torch
            from diffusers.loaders.lora_base import LORA_ADAPTER_METADATA_KEY

            comfy_preserve_prefix_families = {
                "flux",
                "flux2",
                "lumina2",
                "z_image",
                "sd3",
                "auraflow",
                "pixart_sigma",
                "qwen_image",
                "hidream",
            }
            preserve_component_prefixes = (
                {"transformer"} if self.config.model_family in comfy_preserve_prefix_families else None
            )

            # Use model-specific converters where available
            model_family = self.config.model_family

            def comfyui_save_function(weights, filename):
                metadata = {"format": "pt"}
                if adapter_metadata:
                    try:
                        metadata[LORA_ADAPTER_METADATA_KEY] = json.dumps(adapter_metadata, indent=2, sort_keys=True)
                    except Exception:
                        pass

                if model_family == "flux2":
                    from simpletuner.helpers.models.flux2.pipeline import _convert_diffusers_flux2_lora_to_comfyui

                    converted = _convert_diffusers_flux2_lora_to_comfyui(weights, adapter_metadata=adapter_metadata)
                else:
                    converted = convert_diffusers_to_comfyui(
                        weights,
                        adapter_metadata=adapter_metadata,
                        preserve_component_prefixes=preserve_component_prefixes,
                    )
                safetensors.torch.save_file(converted, filename, metadata=metadata)

            kwargs["save_function"] = comfyui_save_function
            kwargs["safe_serialization"] = True
        elif user_save_function is not None:
            kwargs["save_function"] = user_save_function

        self.PIPELINE_CLASSES[
            (PipelineTypes.TEXT2IMG if not self.config.controlnet else PipelineTypes.CONTROLNET)
        ].save_lora_weights(save_directory=save_directory, *remaining, **kwargs)

    def pre_ema_creation(self):
        """
        A hook that can be overridden in the subclass to perform actions before EMA creation.
        """
        self.fuse_qkv_projections()

    def post_ema_creation(self):
        """
        A hook that can be overridden in the subclass to perform actions after EMA creation.
        """
        pass

    def check_user_config(self):
        """
        Checks self.config values against important issues. Optionally implemented in child class.
        """
        pass

    def _model_config_path(self):
        return get_model_config_path(
            model_family=self.config.model_family,
            model_path=self.config.pretrained_model_name_or_path,
        )

    def unwrap_model(self, model=None):
        if self.config.controlnet and model is None:
            if self.controlnet is None:
                return None
            return unwrap_model(self.accelerator, self.controlnet)
        if self.model is None:
            return None
        return unwrap_model(self.accelerator, model or self.model)

    @staticmethod
    def _module_has_meta_tensors(module: Optional[torch.nn.Module]) -> bool:
        if module is None:
            return False
        try:
            for tensor in module.parameters(recurse=True):
                if tensor is not None and getattr(tensor, "device", None) is not None and tensor.device.type == "meta":
                    return True
            for tensor in module.buffers(recurse=True):
                if tensor is not None and getattr(tensor, "device", None) is not None and tensor.device.type == "meta":
                    return True
        except Exception:
            logger.debug("Meta tensor detection failed for %s", module.__class__.__name__, exc_info=True)
        return False

    @staticmethod
    def _sample_meta_tensor_names(module: Optional[torch.nn.Module], limit: int = 3) -> list[str]:
        names: list[str] = []
        if module is None:
            return names

        def _collect(iterator):
            for name, tensor in iterator:
                if tensor is None or getattr(tensor, "device", None) is None:
                    continue
                if tensor.device.type == "meta":
                    names.append(name)
                    if len(names) >= limit:
                        return True
            return False

        try:
            if _collect(module.named_parameters(recurse=True)):
                return names
            _collect(module.named_buffers(recurse=True))
        except Exception:
            logger.debug("Meta tensor name sampling failed for %s", module.__class__.__name__, exc_info=True)
        return names

    def move_extra_models(self, target_device):
        """
        Move any extra models in the child class.

        This is a stub and can be optionally implemented in subclasses.
        """
        pass

    def move_models(self, target_device):
        """
        Moves the model to the target device.
        """
        target_device_obj = torch.device(target_device) if isinstance(target_device, str) else target_device
        accelerator_device = torch.device(self.accelerator.device) if hasattr(self.accelerator, "device") else None
        base_precision = str(getattr(self.config, "base_model_precision", "") or "").lower()
        torchao_quantized = "torchao" in base_precision
        should_configure_offload = (
            self.group_offload_requested()
            and accelerator_device is not None
            and isinstance(target_device_obj, torch.device)
            and target_device_obj == accelerator_device
        )
        skip_moving_trained_component = any(
            [
                (should_configure_offload and self.group_offload_configured),
                self.config.musubi_blocks_to_swap or 0 > 0,
                self.config.quantize_via == "pipeline",
                self.config.ramtorch,
                torchao_quantized,
            ]
        )

        if self.model is not None and not skip_moving_trained_component:
            model_ref = self.unwrap_model(model=self.model)
            if getattr(model_ref, "device", None) != "meta":
                model_ref.to(target_device)
        elif self.model is not None and torchao_quantized:
            logger.info(
                "Skipping model.to(%s) for TorchAO-quantized base model to avoid weight swap errors.",
                target_device,
            )
        if self.controlnet is not None and not skip_moving_trained_component:
            self.unwrap_model(model=self.controlnet).to(target_device)
        if self.vae is not None and self.vae.device != "meta":
            self.vae.to(target_device)
        if self.text_encoders is not None and not self._ramtorch_text_encoders_requested():
            for text_encoder in self.text_encoders:
                if text_encoder is None or getattr(text_encoder, "device", None) == "meta":
                    continue
                text_encoder.to(target_device)
        self.move_extra_models(target_device)

        if should_configure_offload:
            self.configure_group_offload()

    def get_validation_preview_spec(self):
        """
        Return the Tiny AutoEncoder specification for this model family, if any.
        """
        return getattr(self, "VALIDATION_PREVIEW_SPEC", None)

    def supports_validation_preview(self) -> bool:
        return self.get_validation_preview_spec() is not None

    def get_validation_preview_decoder(self):
        """
        Lazily instantiate the Tiny AutoEncoder used for validation previews.
        """
        if self._validation_preview_decoder_failed:
            return None
        spec = self.get_validation_preview_spec()
        if spec is None:
            return None
        if self._validation_preview_decoder is None:
            try:
                dtype = getattr(self.config, "weight_dtype", torch.float16)
                decoder = load_tae_decoder(
                    spec,
                    device=getattr(self.accelerator, "device", None),
                    dtype=dtype,
                )
                self._validation_preview_decoder = decoder
            except Exception as exc:  # pragma: no cover - hardware / network dependent
                logger.warning("Failed to load Tiny AutoEncoder decoder: %s", exc)
                self._validation_preview_decoder_failed = True
                return None
        return self._validation_preview_decoder

    def _vae_scaling_factor(self):
        vae = self.get_vae()
        if vae is None:
            return 1.0
        scaling_factor = getattr(vae.config, "scaling_factor", None)
        if scaling_factor is None:
            scaling_factor = getattr(self, "AUTOENCODER_SCALING_FACTOR", 1.0)
        return float(scaling_factor or 1.0)

    def denormalize_latents_for_preview(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Convert latents from the diffusion space back into decoder space prior to Tiny AE decode.
        """
        vae = self.get_vae()
        if vae is None:
            return latents
        scaling_factor = self._vae_scaling_factor()
        latents_mean = getattr(vae.config, "latents_mean", None)
        latents_std = getattr(vae.config, "latents_std", None)
        if latents.ndim == 4:
            view_shape = (1, latents.shape[1], 1, 1)
        elif latents.ndim == 5:
            view_shape = (1, latents.shape[1], 1, 1, 1)
        else:
            raise ValueError(f"Unsupported number of dimensions for latents: {latents.ndim}. Expected 4 or 5.")

        if latents_mean is not None and latents_std is not None:
            mean = torch.tensor(latents_mean, device=latents.device, dtype=latents.dtype).view(view_shape)
            std = torch.tensor(latents_std, device=latents.device, dtype=latents.dtype).view(view_shape)
            latents = latents * std / scaling_factor + mean
        else:
            latents = latents / scaling_factor
        return latents

    def pre_latent_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Pre-process latents before passing to any decoder (VAE or TAE).

        Override for model-specific shape transformations (e.g., unpacking,
        adding frame dimensions).

        Args:
            latents: The latents tensor to transform

        Returns:
            The transformed latents tensor
        """
        return latents

    # Backward compatibility alias
    pre_validation_preview_decode = pre_latent_decode

    def decode_latents_to_pixels(
        self,
        latents: torch.Tensor,
        *,
        use_tae: bool = False,
    ) -> torch.Tensor:
        """
        Decode latents to pixel space using VAE or TAE.

        Handles all normalization/denormalization internally - callers provide
        latents in training format and receive pixels in consistent format.

        Args:
            latents: Latents in training/diffusion space
            use_tae: If True, use TinyAutoEncoder (faster, lower quality)

        Returns:
            Decoded pixels in (B, T, C, H, W) format, [0, 1] range.
            For image models, T=1.
        """
        latents = latents.detach()

        if use_tae:
            decoder = self.get_validation_preview_decoder()
            if decoder is None:
                raise ValueError(f"{self.NAME} does not support TAE decoding")
            # TAE expects normalized latents (training space) - no denorm
            preprocessed = self.pre_latent_decode(latents)
            preprocessed = preprocessed.to(device=decoder.device, dtype=decoder.dtype)
            decoded = decoder.decode(preprocessed)
            # TAE outputs [0, 1] already
        else:
            vae = self.get_vae()
            if vae is None:
                raise ValueError(f"{self.NAME} does not have a VAE available")
            vae_dtype = next(vae.parameters()).dtype
            # VAE expects denormalized latents
            denormed = self.denormalize_latents_for_preview(latents)
            preprocessed = self.pre_latent_decode(denormed)
            preprocessed = preprocessed.to(device=self.accelerator.device, dtype=vae_dtype)
            with torch.no_grad():
                decoded = vae.decode(preprocessed).sample
            # VAE outputs [-1, 1], normalize to [0, 1]
            decoded = (decoded.clamp(-1, 1) + 1) / 2

        # Ensure consistent output format: (B, T, C, H, W)
        return self._ensure_video_format(decoded)

    def _ensure_video_format(self, decoded: torch.Tensor) -> torch.Tensor:
        """Normalize decoded output to (B, T, C, H, W) format in [0, 1] range."""
        if decoded.ndim == 4:
            # (B, C, H, W) -> (B, 1, C, H, W)
            return decoded.unsqueeze(1)
        if decoded.ndim == 5:
            # Check if (B, C, T, H, W) and convert to (B, T, C, H, W)
            # Heuristic: channel dim is typically 3 or 4, and the temporal
            # dimension is not larger than the spatial dimensions (H, W).
            if decoded.shape[1] in (3, 4) and decoded.shape[2] <= decoded.shape[3] and decoded.shape[2] <= decoded.shape[4]:
                return decoded.permute(0, 2, 1, 3, 4)
        return decoded

    def get_vae(self):
        """
        Returns the VAE model.
        """
        if not getattr(self, "AUTOENCODER_CLASS", None):
            return
        if not hasattr(self, "vae") or self.vae is None or getattr(self.vae, "device", None) == "meta":
            self.load_vae()
        return self.vae

    def load_vae(self, move_to_device: bool = True):
        from transformers.utils import ContextManagers

        if not getattr(self, "AUTOENCODER_CLASS", None):
            return

        logger.info(f"Loading {self.AUTOENCODER_CLASS.__name__} from {self.config.vae_path}")

        # Register VAE cache path for potential deletion
        if getattr(self.config, "delete_model_after_load", False):
            from simpletuner.helpers.training.state_tracker import StateTracker

            cache_repo_path = get_hf_cache_repo_path(self.config.vae_path)
            logger.info(
                f"load_vae: delete_model_after_load=True, vae_path={self.config.vae_path}, cache_repo_path={cache_repo_path}"
            )
            if cache_repo_path:
                StateTracker.set_model_snapshot_path("vae", cache_repo_path)
                logger.info(f"load_vae: registered VAE cache path: {cache_repo_path}")

        self.vae = None
        self.config.vae_kwargs = {
            "pretrained_model_name_or_path": get_model_config_path(self.config.model_family, self.config.vae_path),
            "subfolder": "vae",
            "revision": self.config.revision,
            "force_upcast": False,
            "variant": self.config.variant,
        }
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            try:
                self.vae = self.AUTOENCODER_CLASS.from_pretrained(**self.config.vae_kwargs)
            except Exception as e:
                self.config.vae_kwargs["subfolder"] = None
                self.vae = self.AUTOENCODER_CLASS.from_pretrained(**self.config.vae_kwargs)
        if self.vae is None:
            raise ValueError("Could not load VAE. Please check the model path and ensure the VAE is compatible.")
        if self.config.vae_enable_tiling:
            if hasattr(self.vae, "enable_tiling"):
                logger.info("Enabling VAE tiling.")
                self.vae.enable_tiling()
            else:
                logger.warning(f"VAE tiling is enabled, but not yet supported by {self.config.model_family}.")
        if self.config.vae_enable_slicing:
            if hasattr(self.vae, "enable_slicing"):
                logger.info("Enabling VAE slicing.")
                self.vae.enable_slicing()
            else:
                logger.warning(f"VAE slicing is enabled, but not yet supported by {self.config.model_family}.")
        if getattr(self.config, "crepa_drop_vae_encoder", False):
            logger.info("CREPA decode-only mode enabled; dropping VAE encoder/quant_conv to save memory.")
            if hasattr(self.vae, "encoder"):
                self.vae.encoder = None
            if hasattr(self.vae, "quant_conv"):
                self.vae.quant_conv = None
        if self._ramtorch_vae_requested():
            mid_block = getattr(self.vae, "mid_block", None)
            if mid_block is None:
                logger.debug("RamTorch VAE requested but no VAE mid_block was found; skipping RamTorch conversion.")
            else:
                self._apply_ramtorch_layers(mid_block, "vae.mid_block")
        if move_to_device and self.vae.device != self.accelerator.device:
            _vae_dtype = torch.bfloat16
            if hasattr(self.config, "vae_dtype"):
                # Let's use a case-switch for convenience: bf16, fp16, fp32, none/default
                if self.config.vae_dtype == "bf16":
                    _vae_dtype = torch.bfloat16
                elif self.config.vae_dtype == "fp16":
                    raise ValueError("fp16 is not supported for SDXL's VAE. Please use bf16 or fp32.")
                elif self.config.vae_dtype == "fp32":
                    _vae_dtype = torch.float32
                elif self.config.vae_dtype == "none" or self.config.vae_dtype == "default":
                    _vae_dtype = torch.bfloat16
            logger.info(
                f"Moving {self.AUTOENCODER_CLASS.__name__} to accelerator, converting from {self.vae.dtype} to {_vae_dtype}"
            )
            self.vae.to(self.accelerator.device, dtype=_vae_dtype)
        self.AUTOENCODER_SCALING_FACTOR = getattr(self.vae.config, "scaling_factor", 1.0)

        self.post_vae_load_setup()

    def post_vae_load_setup(self):
        """
        Post VAE load setup.

        This is a stub and can be optionally implemented in subclasses for eg. updating configuration settings
        based on the loaded VAE weights. SDXL uses this to update the user config to reflect refiner training.

        """
        pass

    def pre_vae_encode_transform_sample(self, sample):
        """
        Pre-encode transform for the sample before passing it to the VAE.
        This is a stub and can be optionally implemented in subclasses.
        """
        return sample

    @torch.no_grad()
    def encode_with_vae(self, vae, samples):
        """
        Hook for models to customize VAE encoding behaviour (e.g. applying flavour-specific patches).
        By default this simply forwards to the provided VAE.
        """
        return vae.encode(samples)

    def encode_cache_batch(self, vae, samples, metadata_entries: Optional[list] = None):
        """
        Hook invoked by VAECache so models can consume per-sample metadata (e.g. lyrics)
        while performing VAE encoding. Default implementation ignores metadata.
        """
        return self.encode_with_vae(vae, samples)

    def post_vae_encode_transform_sample(self, sample):
        """
        Post-encode transform for the sample after passing it to the VAE.
        This is a stub and can be optionally implemented in subclasses.
        """
        return sample

    def scale_vae_latents_for_cache(self, latents, vae):
        """
        Optional hook for caching flows to apply model-specific VAE scaling.
        """
        return latents

    def unload_vae(self):
        if self.vae is not None:
            if hasattr(self.vae, "to"):
                self.vae.to("meta")
            self.vae = None

    @staticmethod
    def _is_bitsandbytes_model(model) -> bool:
        if model is None:
            return False
        if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
            return True
        try:
            for param in model.parameters():
                param_type = type(param).__name__
                if "Params4bit" in param_type or "Int8Params" in param_type:
                    return True
        except Exception:
            return False
        return False

    @staticmethod
    def _is_gemma_component(component_cls) -> bool:
        if component_cls is None:
            return False
        component_name = getattr(component_cls, "__name__", "")
        return "gemma" in component_name.lower()

    def _resolve_text_encoder_path(self, text_encoder_config: dict) -> str:
        text_encoder_path = get_model_config_path(self.config.model_family, self.config.pretrained_model_name_or_path)
        config_path = text_encoder_config.get("path", None)
        if config_path is not None:
            text_encoder_path = config_path
        gemma_path = getattr(self.config, "pretrained_gemma_model_name_or_path", None)
        if gemma_path and self._is_gemma_component(text_encoder_config.get("model")):
            text_encoder_path = gemma_path
        return text_encoder_path

    def load_text_tokenizer(self):
        if self.TEXT_ENCODER_CONFIGURATION is None or len(self.TEXT_ENCODER_CONFIGURATION) == 0:
            return
        self.tokenizers = []
        tokenizer_kwargs = {
            "subfolder": "tokenizer",
            "revision": self.config.revision,
            "use_fast": False,
        }
        tokenizer_idx = 0
        for attr_name, text_encoder_config in self.TEXT_ENCODER_CONFIGURATION.items():
            tokenizer_idx += 1
            tokenizer_cls = text_encoder_config.get("tokenizer")
            tokenizer_kwargs["subfolder"] = text_encoder_config.get("tokenizer_subfolder", "tokenizer")
            tokenizer_kwargs["use_fast"] = text_encoder_config.get("use_fast", False)
            tokenizer_kwargs["pretrained_model_name_or_path"] = self._resolve_text_encoder_path(text_encoder_config)
            logger.info(f"Loading tokenizer {tokenizer_idx}: {tokenizer_cls.__name__} with args: {tokenizer_kwargs}")
            tokenizer = tokenizer_cls.from_pretrained(**tokenizer_kwargs)
            self.tokenizers.append(tokenizer)
            setattr(self, f"tokenizer_{tokenizer_idx}", tokenizer)

    def load_text_encoder(self, move_to_device: bool = True):
        from transformers.utils import ContextManagers

        self.text_encoders = []
        if self.TEXT_ENCODER_CONFIGURATION is None or len(self.TEXT_ENCODER_CONFIGURATION) == 0:
            return
        self.load_text_tokenizer()

        # Track if we should store cache paths for text encoder deletion
        should_track_for_deletion = getattr(self.config, "delete_model_after_load", False)

        text_encoder_idx = 0
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            for (
                attr_name,
                text_encoder_config,
            ) in self.TEXT_ENCODER_CONFIGURATION.items():
                text_encoder_idx += 1
                # Prefer indexed precision flags (text_encoder_1_precision, etc) to match config schema.
                precision_attr = f"text_encoder_{text_encoder_idx}_precision"
                text_encoder_precision = getattr(
                    self.config, precision_attr, getattr(self.config, f"{attr_name}_precision", None)
                )
                quantize_via_pipeline = getattr(self.config, "quantize_via", None) == "pipeline"
                pipeline_preset = (
                    text_encoder_precision
                    if (
                        text_encoder_precision in PIPELINE_ONLY_PRESETS
                        or (quantize_via_pipeline and text_encoder_precision in PIPELINE_QUANTIZATION_PRESETS)
                    )
                    else None
                )
                # load_tes returns a variant and three text encoders
                signature = inspect.signature(text_encoder_config["model"])
                extra_kwargs = {}
                if "torch_dtype" in signature.parameters:
                    extra_kwargs["torch_dtype"] = self.config.weight_dtype
                logger.info(f"Loading {text_encoder_config.get('name')} text encoder")
                text_encoder_path = self._resolve_text_encoder_path(text_encoder_config)

                # Register text encoder cache path for potential deletion
                if should_track_for_deletion:
                    from simpletuner.helpers.training.state_tracker import StateTracker

                    component_key = f"text_encoder_{text_encoder_idx}"
                    cache_repo_path = get_hf_cache_repo_path(text_encoder_path)
                    if cache_repo_path:
                        StateTracker.set_model_snapshot_path(component_key, cache_repo_path)

                requires_quant = text_encoder_config.get("required_quantisation_level", None)
                quantization_config = None
                if (
                    getattr(self.config, "quantize_via", None) == "pipeline"
                    or pipeline_preset is not None
                    or getattr(self.config, "quantization_config", None) is not None
                ):
                    quantization_config = self._build_component_quantization_config(
                        component_keys=[attr_name, f"text_encoder_{text_encoder_idx}", "text_encoder"],
                        preset=pipeline_preset,
                    )
                if requires_quant is not None and requires_quant == "int4_weight_only":
                    from torchao.quantization import Int4WeightOnlyConfig
                    from transformers import TorchAoConfig

                    extra_kwargs["device_map"] = "auto"
                    quant_config = Int4WeightOnlyConfig(group_size=128)
                    quantization_config = quantization_config or TorchAoConfig(quant_type=quant_config)
                if quantization_config is not None:
                    extra_kwargs["quantization_config"] = quantization_config
                    self.pipeline_quantization_active = True
                    setattr(self.config, "pipeline_quantization", True)

                # When ramtorch is enabled for text encoders, load on CPU first.
                # Ramtorch will convert Linear layers to CPU-bouncing, then move
                # non-Linear layers (Embedding, LayerNorm, etc.) to GPU.
                if self._ramtorch_text_encoders_requested() and "device_map" not in extra_kwargs:
                    extra_kwargs["device_map"] = "cpu"
                    # Force dtype - the signature check may fail for some models
                    if "torch_dtype" not in extra_kwargs:
                        extra_kwargs["torch_dtype"] = self.config.weight_dtype

                text_encoder_kwargs = {
                    "pretrained_model_name_or_path": text_encoder_path,
                    "variant": self.config.variant,
                    "revision": self.config.revision,
                    "subfolder": text_encoder_config.get("subfolder", "text_encoder") or "",
                    **extra_kwargs,
                }
                logger.debug(f"Text encoder {text_encoder_idx} load args: {text_encoder_kwargs}")
                text_encoder = text_encoder_config["model"].from_pretrained(**text_encoder_kwargs)
                is_bnb_quantized = self._is_bitsandbytes_model(text_encoder)
                if is_bnb_quantized:
                    logger.info("Detected bitsandbytes-quantized text encoder; skipping device/dtype move.")
                if text_encoder.__class__.__name__ in [
                    "UMT5EncoderModel",
                    "T5EncoderModel",
                ]:
                    pass

                if self._ramtorch_text_encoders_requested():
                    # Use full ramtorch for text encoders - all layer types stream from CPU
                    self._apply_ramtorch_layers(
                        text_encoder,
                        f"text_encoder_{text_encoder_idx}",
                        full_ramtorch=True,
                        percent=self._ramtorch_text_encoder_percent(),
                    )

                if (
                    move_to_device
                    and text_encoder_precision in ["no_change", None]
                    and quantization_config is None
                    and not self._ramtorch_text_encoders_requested()
                    and not is_bnb_quantized
                ):
                    text_encoder.to(
                        self.accelerator.device,
                        dtype=self.config.weight_dtype,
                    )
                if hasattr(text_encoder, "eval"):
                    text_encoder.eval()
                # Disable gradients immediately - text encoders are only used for inference
                # during embed caching, and keeping requires_grad=True would cause PyTorch
                # to retain computation graphs in CUDA memory.
                text_encoder.requires_grad_(False)
                setattr(self, f"text_encoder_{text_encoder_idx}", text_encoder)
                self.text_encoders.append(text_encoder)

    def get_text_encoder(self, index: int):
        if self.text_encoders is not None:
            return self.text_encoders[index] if index in self.text_encoders else None

    def unload_text_encoder(self):
        if self.text_encoders is not None:
            for idx, text_encoder in enumerate(self.text_encoders):
                if text_encoder is None:
                    continue
                if hasattr(text_encoder, "to"):
                    is_bnb_quantized = self._is_bitsandbytes_model(text_encoder)
                    # Log memory before move
                    if torch.cuda.is_available():
                        mem_before = torch.cuda.memory_allocated() / (1024**3)
                        logger.info("Text encoder %s: memory before unload: %.2f GB", idx + 1, mem_before)

                        # Count tensors on CUDA before move
                        cuda_params = sum(1 for p in text_encoder.parameters() if p.device.type == "cuda")
                        cuda_buffers = sum(1 for b in text_encoder.buffers() if b.device.type == "cuda")
                        logger.info(
                            "Text encoder %s: %d params and %d buffers on CUDA before move",
                            idx + 1,
                            cuda_params,
                            cuda_buffers,
                        )

                    if is_bnb_quantized:
                        logger.info(
                            "Text encoder %s is bitsandbytes-quantized; skipping meta/CPU move.",
                            idx + 1,
                        )
                    else:
                        # Always move off accelerator; fall back to CPU if meta tensors aren't supported.
                        try:
                            text_encoder.to("meta")
                            logger.info("Text encoder %s successfully moved to meta device", idx + 1)
                        except Exception as exc:
                            logger.warning(
                                "Text encoder %s could not be moved to meta, moving to CPU instead: %s",
                                idx + 1,
                                exc,
                            )
                            exc_message = str(exc)
                            if "Params4bit" in exc_message or "Int8Params" in exc_message:
                                logger.warning(
                                    "Text encoder %s appears to be bitsandbytes-quantized; skipping CPU move.",
                                    idx + 1,
                                )
                            else:
                                has_meta_tensors = any(p.is_meta for p in text_encoder.parameters()) or any(
                                    b.is_meta for b in text_encoder.buffers()
                                )
                                if has_meta_tensors and hasattr(text_encoder, "to_empty"):
                                    text_encoder.to_empty(device="cpu")
                                else:
                                    text_encoder.to("cpu")

                    # Log memory after move
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        mem_after = torch.cuda.memory_allocated() / (1024**3)
                        logger.info(
                            "Text encoder %s: memory after unload: %.2f GB (freed %.2f GB)",
                            idx + 1,
                            mem_after,
                            mem_before - mem_after,
                        )

                        # Check if any tensors are still on CUDA
                        cuda_params_after = sum(1 for p in text_encoder.parameters() if p.device.type == "cuda")
                        cuda_buffers_after = sum(1 for b in text_encoder.buffers() if b.device.type == "cuda")
                        if cuda_params_after > 0 or cuda_buffers_after > 0:
                            logger.warning(
                                "Text encoder %s still has %d params and %d buffers on CUDA after move!",
                                idx + 1,
                                cuda_params_after,
                                cuda_buffers_after,
                            )

                setattr(self, f"text_encoder_{idx + 1}", None)
            self.text_encoders = None
        if self.tokenizers is not None:
            self.tokenizers = None

    def unload(self):
        """
        Comprehensively unload all model components to free GPU memory.
        This moves all models to the 'meta' device which releases GPU memory.
        """
        logger.info("Unloading all model components...")
        self._group_offload_configured = False

        # Unload VAE
        self.unload_vae()

        # Unload text encoders
        self.unload_text_encoder()

        # Unload main model (transformer/unet)
        if hasattr(self, "model") and self.model is not None:
            if hasattr(self.model, "to"):
                self.model.to("meta")
            self.model = None

        # Unload controlnet if present
        if hasattr(self, "controlnet") and self.controlnet is not None:
            if hasattr(self.controlnet, "to"):
                self.controlnet.to("meta")
            self.controlnet = None

        # Clear any cached pipelines
        if hasattr(self, "pipelines") and self.pipelines:
            self.pipelines.clear()

        # Reclaim memory
        from simpletuner.helpers.caching.memory import reclaim_memory

        reclaim_memory()

        logger.info("Model components unloaded successfully.")

    def pretrained_load_args(self, pretrained_load_args: dict) -> dict:
        """
        Augment `from_pretrained` kwargs before loading the base model.

        This is commonly used by child classes, but we also handle shared feature flags here when safe.
        """
        if getattr(self.config, "twinflow_enabled", False):
            model_cls = getattr(self, "MODEL_CLASS", None)
            if model_cls is not None:
                try:
                    import inspect

                    signature = inspect.signature(model_cls.__init__)
                    if "enable_time_sign_embed" in signature.parameters:
                        pretrained_load_args.setdefault("enable_time_sign_embed", True)
                except (TypeError, ValueError):
                    # Some callables may not have introspectable signatures (e.g., C-extensions).
                    pass
        return pretrained_load_args

    def _extract_quantization_entry(self, raw_config, component_keys: list[str]):
        if raw_config is None:
            return None
        try:
            from diffusers import PipelineQuantizationConfig

            if isinstance(raw_config, PipelineQuantizationConfig):
                quant_mapping = getattr(raw_config, "quant_mapping", None) or {}
                if isinstance(quant_mapping, Mapping):
                    for key in component_keys:
                        if key in quant_mapping:
                            return quant_mapping[key]
                    if "default" in quant_mapping:
                        return quant_mapping["default"]
                return raw_config
        except Exception:
            # Avoid failing when diffusers is unavailable or the config type differs.
            pass

        if isinstance(raw_config, Mapping):
            for key in component_keys:
                if key in raw_config:
                    return raw_config[key]
            if "default" in raw_config:
                return raw_config["default"]

            return None

        return raw_config

    def _build_component_quantization_config(self, component_keys: list[str], preset: Optional[str] = None):
        raw_config = getattr(self.config, "quantization_config", None)
        if raw_config is None:
            raw_config = getattr(self.config, "pipeline_quantization_config", None)
        entry = self._extract_quantization_entry(raw_config, component_keys)
        preset_candidate = preset
        if isinstance(entry, str):
            preset_candidate = entry

        builder = get_pipeline_quantization_builder(preset_candidate)
        if builder is None and isinstance(entry, Mapping):
            entry_keys = set(entry.keys())
            if any(key.startswith("bnb_") or key in {"load_in_4bit", "load_in_8bit"} for key in entry_keys):
                builder = get_pipeline_quantization_builder("nf4-bnb")
            elif "weights_dtype" in entry_keys or "weights" in entry_keys:
                builder = get_pipeline_quantization_builder("int8-quanto")
            elif "quant_type" in entry_keys or "quant_type_kwargs" in entry_keys or "modules_to_not_convert" in entry_keys:
                builder = get_pipeline_quantization_builder("int4-torchao")
        # If we have a known preset builder, prefer it and treat mapping entries as overrides.
        if builder is not None and (entry is None or isinstance(entry, (Mapping, str))):
            overrides = entry if isinstance(entry, Mapping) else None
            component_type = "transformers" if any("text_encoder" in key for key in component_keys) else "diffusers"
            quantization_config = builder(getattr(self.config, "weight_dtype", None), overrides, component_type)
            self._register_pipeline_quantization_config(component_keys, quantization_config)
            return quantization_config

        if entry is not None:
            try:
                from diffusers import PipelineQuantizationConfig
            except Exception:
                PipelineQuantizationConfig = None
            if PipelineQuantizationConfig is not None and isinstance(entry, PipelineQuantizationConfig):
                component_type = "transformers" if any("text_encoder" in key for key in component_keys) else "diffusers"
                quantization_config = self._resolve_pipeline_backend_config(entry, component_keys, component_type)
                self._register_pipeline_quantization_config(component_keys, quantization_config)
                return quantization_config
        if entry is not None:
            self._register_pipeline_quantization_config(component_keys, entry)
        return entry

    def _resolve_pipeline_backend_config(self, pipeline_config, component_keys: list[str], component_type: str):
        quant_backend = getattr(pipeline_config, "quant_backend", None)
        if not quant_backend:
            return None
        components_to_quantize = getattr(pipeline_config, "components_to_quantize", None)
        if components_to_quantize:
            if not any(key in components_to_quantize for key in component_keys):
                return None

        quant_kwargs = getattr(pipeline_config, "quant_kwargs", None) or {}
        if component_type == "transformers":
            try:
                from transformers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING as auto_mapping
            except ImportError as exc:
                raise ImportError(
                    "Pipeline quantization for text encoders requires transformers with quantization config support."
                ) from exc
        else:
            try:
                from diffusers.quantizers.auto import AUTO_QUANTIZATION_CONFIG_MAPPING as auto_mapping
            except ImportError as exc:
                raise ImportError(
                    "Pipeline quantization for diffusion components requires diffusers quantization config support."
                ) from exc

        config_cls = auto_mapping.get(quant_backend)
        if config_cls is None:
            raise ValueError(f"Unknown pipeline quantization backend: {quant_backend}")
        return config_cls(**quant_kwargs)

    def _register_pipeline_quantization_config(self, component_keys: list[str], quantization_config) -> None:
        if (
            quantization_config is None
            or isinstance(quantization_config, Mapping)
            or getattr(self.config, "quantize_via", None) != "pipeline"
        ):
            return
        try:
            from diffusers import PipelineQuantizationConfig
        except ImportError as exc:
            raise ImportError("Pipeline quantization requires diffusers with PipelineQuantizationConfig support.") from exc

        if isinstance(quantization_config, PipelineQuantizationConfig):
            setattr(self.config, "pipeline_quantization_config", quantization_config)
            return

        pipeline_config = getattr(self.config, "pipeline_quantization_config", None)
        if pipeline_config is None:
            pipeline_config = PipelineQuantizationConfig(quant_mapping={key: quantization_config for key in component_keys})
            setattr(self.config, "pipeline_quantization_config", pipeline_config)
            return

        quant_mapping = getattr(pipeline_config, "quant_mapping", None)
        if quant_mapping is None:
            quant_mapping = {}
            pipeline_config.quant_mapping = quant_mapping
        for key in component_keys:
            quant_mapping.setdefault(key, quantization_config)

    def load_model(self, move_to_device: bool = True):
        self._group_offload_configured = False
        pretrained_load_args = {
            "revision": self.config.revision,
            "variant": self.config.variant,
            "torch_dtype": self.config.weight_dtype,
            "use_safetensors": True,
        }
        loader_fn = self.MODEL_CLASS.from_pretrained
        model_path = (
            self.config.pretrained_transformer_model_name_or_path
            if self.MODEL_TYPE is ModelTypes.TRANSFORMER
            else self.config.pretrained_unet_model_name_or_path
        ) or self.config.pretrained_model_name_or_path

        # Register transformer/unet cache path for potential deletion
        if getattr(self.config, "delete_model_after_load", False):
            from simpletuner.helpers.training.state_tracker import StateTracker

            component_key = self.MODEL_TYPE.value  # "transformer" or "unet"
            cache_repo_path = get_hf_cache_repo_path(model_path)
            logger.info(
                f"load_model: delete_model_after_load=True, model_path={model_path}, cache_repo_path={cache_repo_path}"
            )
            if cache_repo_path:
                StateTracker.set_model_snapshot_path(component_key, cache_repo_path)
                logger.info(f"load_model: registered {component_key} cache path: {cache_repo_path}")

        is_safetensors = isinstance(model_path, str) and model_path.endswith(".safetensors")
        is_gguf = isinstance(model_path, str) and model_path.endswith(".gguf")
        if isinstance(self.config.pretrained_model_name_or_path, str) and self.config.pretrained_model_name_or_path.endswith(
            (".safetensors", ".gguf")
        ):
            self.config.pretrained_model_name_or_path = get_model_config_path(self.config.model_family, model_path)
        if is_safetensors or is_gguf:
            loader_fn = self.MODEL_CLASS.from_single_file
        if is_gguf:
            pretrained_load_args["use_safetensors"] = False
        if is_gguf:
            setattr(self.config, "pipeline_quantization", True)
            setattr(self.config, "pipeline_quantization_base", True)
        quantize_via_pipeline = getattr(self.config, "quantize_via", None) == "pipeline"
        if quantize_via_pipeline:
            setattr(self.config, "pipeline_quantization", True)

        pipeline_only_precision = self.config.base_model_precision in PIPELINE_ONLY_PRESETS
        pipeline_capable_precision = self.config.base_model_precision in PIPELINE_QUANTIZATION_PRESETS
        base_pipeline_preset = (
            self.config.base_model_precision
            if (pipeline_only_precision or (quantize_via_pipeline and pipeline_capable_precision))
            else None
        )
        if base_pipeline_preset is not None:
            setattr(self.config, "pipeline_quantization", True)
            setattr(self.config, "pipeline_quantization_base", True)
        quantization_config = None
        if is_gguf:
            quantization_config = build_gguf_quantization_config(model_path)
        elif (
            getattr(self.config, "quantize_via", None) == "pipeline"
            or base_pipeline_preset is not None
            or getattr(self.config, "quantization_config", None) is not None
        ):
            quantization_config = self._build_component_quantization_config(
                component_keys=[key for key in [self.MODEL_TYPE.value, "model", self.MODEL_SUBFOLDER] if key],
                preset=base_pipeline_preset,
            )

        if quantization_config is not None:
            pretrained_load_args["quantization_config"] = quantization_config
            self.pipeline_quantization_active = True
            setattr(self.config, "pipeline_quantization", True)
            setattr(self.config, "pipeline_quantization_base", True)
        elif is_gguf:
            # Failed to build a quantization_config but a GGUF checkpoint was requested.
            raise RuntimeError(
                f"GGUF checkpoint {model_path} requires GGUFQuantizationConfig support but no quantization_config could be constructed."
            )

        pretrained_load_args = self.pretrained_load_args(pretrained_load_args)
        model_subfolder = self.MODEL_SUBFOLDER
        if self.MODEL_TYPE is ModelTypes.TRANSFORMER and self.config.pretrained_transformer_model_name_or_path == model_path:
            # we're using a custom transformer, let's check its subfolder
            if str(self.config.pretrained_transformer_subfolder).lower() == "none":
                model_subfolder = None
            elif str(self.config.pretrained_unet_model_name_or_path).lower() is None:
                model_subfolder = self.MODEL_SUBFOLDER
            else:
                model_subfolder = self.config.pretrained_transformer_subfolder
        elif self.MODEL_TYPE is ModelTypes.UNET and self.config.pretrained_unet_model_name_or_path == model_path:
            # we're using a custom transformer, let's check its subfolder
            if str(self.config.pretrained_unet_model_name_or_path).lower() == "none":
                model_subfolder = None
            elif str(self.config.pretrained_unet_model_name_or_path).lower() is None:
                model_subfolder = self.MODEL_SUBFOLDER
            else:
                model_subfolder = self.config.pretrained_unet_subfolder

        from transformers.utils import ContextManagers

        logger.info(f"Loading diffusion model from {model_path}")

        def _load_model(load_kwargs: dict):
            with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
                return loader_fn(
                    model_path,
                    subfolder=model_subfolder,
                    **load_kwargs,
                )

        load_kwargs = dict(pretrained_load_args)
        self.model = _load_model(load_kwargs)
        unwrapped_model = self.unwrap_model(model=self.model)

        if self._module_has_meta_tensors(unwrapped_model):
            if load_kwargs.get("low_cpu_mem_usage", True):
                sample_meta = ", ".join(self._sample_meta_tensor_names(unwrapped_model)) or "(unknown tensors)"
                logger.warning(
                    "Detected meta tensors after loading %s (e.g. %s); disabling low_cpu_mem_usage and retrying.",
                    model_path,
                    sample_meta,
                )
                load_kwargs["low_cpu_mem_usage"] = False
                self.model = _load_model(load_kwargs)
                unwrapped_model = self.unwrap_model(model=self.model)
            if self._module_has_meta_tensors(unwrapped_model):
                raise RuntimeError(
                    f"{self.__class__.__name__} failed to materialize weights for {model_path}. "
                    "All model parameters remain on the meta device after reload."
                )
        if self._ramtorch_enabled() and self.model is not None:
            self._apply_ramtorch_layers(
                self.model,
                self.MODEL_TYPE.value,
                percent=self._ramtorch_transformer_percent(),
                full_ramtorch=True,  # Convert all layer types including RMSNorm
            )
        if move_to_device and self.model is not None and not self._ramtorch_enabled():
            # Skip device move when ramtorch is enabled - ramtorch keeps weights on CPU
            self.model.to(self.accelerator.device, dtype=self.config.weight_dtype)

        self.configure_chunked_feed_forward()

        # Set gradient checkpointing backend
        checkpoint_backend = getattr(self.config, "gradient_checkpointing_backend", "torch")
        if checkpoint_backend == "unsloth" and not torch.cuda.is_available():
            logger.warning("Unsloth gradient checkpointing backend requires CUDA, falling back to torch")
            checkpoint_backend = "torch"

        from simpletuner.helpers.training.gradient_checkpointing_interval import set_checkpoint_backend

        set_checkpoint_backend(checkpoint_backend)
        if checkpoint_backend == "unsloth":
            logger.info("Using Unsloth-style gradient checkpointing (CPU offload)")

        if (
            self.config.gradient_checkpointing_interval is not None
            and self.config.gradient_checkpointing_interval > 1
            and self.MODEL_TYPE is ModelTypes.UNET
        ):
            logger.warning(
                "Using experimental gradient checkpointing monkeypatch for a checkpoint interval of {}".format(
                    self.config.gradient_checkpointing_interval
                )
            )
            # monkey-patch gradient checkpointing for nth call intervals - easier than modifying diffusers blocks
            from simpletuner.helpers.training.gradient_checkpointing_interval import set_checkpoint_interval

            set_checkpoint_interval(int(self.config.gradient_checkpointing_interval))

        if self.config.gradient_checkpointing_interval is not None and self.config.gradient_checkpointing_interval > 1:
            if self.model is not None and hasattr(self.model, "set_gradient_checkpointing_interval"):
                logger.info("Setting gradient checkpointing interval..")
                self.unwrap_model(model=self.model).set_gradient_checkpointing_interval(
                    int(self.config.gradient_checkpointing_interval)
                )

        # Set gradient checkpointing backend on model if supported
        if self.model is not None and hasattr(self.model, "set_gradient_checkpointing_backend"):
            self.unwrap_model(model=self.model).set_gradient_checkpointing_backend(checkpoint_backend)

        self.fuse_qkv_projections()
        self.post_model_load_setup()

    def post_model_load_setup(self):
        """
        Post model load setup.

        This is a stub and can be optionally implemented in subclasses for eg. updating configuration settings
        based on the loaded model weights. SDXL uses this to update the user config to reflect refiner training.

        """
        self._init_layersync_regularizer()

    def fuse_qkv_projections(self):
        if self.config.fuse_qkv_projections:
            logger.warning(
                f"{self.__class__.__name__} does not support fused QKV projection yet, please open a feature request on the issue tracker."
            )

    def unfuse_qkv_projections(self):
        """
        Unfuse QKV projections before critical operations like saving.
        This is a no-op by default, but subclasses can override to implement
        proper unfusing when using fused QKV projections.

        Should be called before:
        - Saving LoRA weights
        - Saving full model checkpoints
        - Any operation that expects separate Q, K, V projections
        """
        pass

    def set_prepared_model(self, model, base_model: bool = False):
        if self.config.controlnet and not base_model:
            self.controlnet = model
        else:
            self.model = model

    def freeze_components(self):
        if self.vae is not None:
            self.vae.requires_grad_(False)
        if self.text_encoders is not None and len(self.text_encoders) > 0:
            for text_encoder in self.text_encoders:
                text_encoder.requires_grad_(False)
        if "lora" in self.config.model_type:
            if self.model is not None:
                self.model.requires_grad_(False)
        if self.config.controlnet and self.controlnet is not None:
            self.controlnet.train()

    def get_lyrics_embedder_modules(self, unwrap: bool = True) -> list[tuple[str, torch.nn.Module]]:
        """Return a list of (name, module) tuples for any lyrics embedder components."""
        return []

    def get_lyrics_embedder_parameters(
        self, *, require_grad_only: bool = True, unwrap: bool = True
    ) -> list[torch.nn.Parameter]:
        """Return parameters belonging to the lyrics embedder, if present."""
        params: list[torch.nn.Parameter] = []
        for _, module in self.get_lyrics_embedder_modules(unwrap=unwrap):
            if module is None:
                continue
            for param in module.parameters():
                if require_grad_only and not param.requires_grad:
                    continue
                params.append(param)
        return params

    def enable_lyrics_embedder_training(self) -> list[str]:
        """
        Mark lyrics embedder parameters as trainable and return the module names enabled.
        """
        enabled: list[str] = []
        for name, module in self.get_lyrics_embedder_modules(unwrap=False):
            if module is None:
                continue
            module.train()
            module.requires_grad_(True)
            enabled.append(name)
        return enabled

    def get_lyrics_embedder_state_dict(self) -> OrderedDict:
        """
        Return a flattened state dict for lyrics embedder components, if any exist.
        """
        state = OrderedDict()
        for name, module in self.get_lyrics_embedder_modules(unwrap=True):
            if module is None:
                continue
            for key, tensor in module.state_dict().items():
                state[f"{name}.{key}"] = tensor.detach().cpu()
        return state

    def load_lyrics_embedder_state_dict(self, state_dict: dict) -> list[str]:
        """
        Load the provided state dict into any available lyrics embedder components.
        """
        if not state_dict:
            return []

        grouped: dict[str, dict] = {}
        for key, tensor in state_dict.items():
            if not isinstance(key, str) or "." not in key:
                continue
            prefix, remainder = key.split(".", 1)
            grouped.setdefault(prefix, {})[remainder] = tensor

        loaded: list[str] = []
        for name, module in self.get_lyrics_embedder_modules(unwrap=False):
            module_state = grouped.get(name)
            if not module_state:
                continue
            module.load_state_dict(module_state, strict=False)
            loaded.append(name)
        return loaded

    def uses_shared_modules(self):
        return False

    def get_trained_component(self, base_model: bool = False, unwrap_model: bool = True):
        if unwrap_model:
            return self.unwrap_model(model=self.model if base_model else None)
        return self.controlnet if self.config.controlnet and not base_model else self.model

    def _load_processor_for_pipeline(self):
        """
        Hook for subclasses to load or return any processor modules required by their pipelines.
        """
        return getattr(self, "processor", None)

    def _load_text_processor_for_pipeline(self):
        """
        Hook for subclasses to load or return any text_processor modules required by their pipelines.
        """
        return getattr(self, "text_processor", None)

    def _load_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True):
        """
        Loads the pipeline class for the model.
        """
        active_pipelines = getattr(self, "pipelines", {})
        if pipeline_type in active_pipelines:
            pipeline_instance = active_pipelines[pipeline_type]
            setattr(
                pipeline_instance,
                self.MODEL_TYPE.value,
                self.unwrap_model(model=self.model),
            )
            if self.config.controlnet:
                setattr(pipeline_instance, "controlnet", self.unwrap_model(model=self.controlnet))
            return pipeline_instance

        pipeline_kwargs = {
            "pretrained_model_name_or_path": self._model_config_path(),
        }
        if not hasattr(self, "PIPELINE_CLASSES"):
            raise NotImplementedError("Pipeline class not defined.")
        if pipeline_type not in self.PIPELINE_CLASSES:
            raise NotImplementedError(f"Pipeline type {pipeline_type} not defined in {self.__class__.__name__}.")
        pipeline_class = self.PIPELINE_CLASSES[pipeline_type]
        if not hasattr(pipeline_class, "from_pretrained"):
            raise NotImplementedError(f"Pipeline class {pipeline_class} does not have from_pretrained method.")
        signature = inspect.signature(pipeline_class.from_pretrained)
        if "watermarker" in signature.parameters:
            pipeline_kwargs["watermarker"] = None
        if "watermark" in signature.parameters:
            pipeline_kwargs["watermark"] = None
        pipeline_kwargs[self.MODEL_TYPE.value] = self.unwrap_model(model=self.model)

        if getattr(self, "vae", None) is not None:
            pipeline_kwargs["vae"] = self.unwrap_model(self.vae)
        elif getattr(self, "AUTOENCODER_CLASS", None) is not None:
            pipeline_kwargs["vae"] = self.get_vae()

        text_encoder_idx = 0
        pipeline_init_signature = inspect.signature(pipeline_class.__init__)

        for (
            text_encoder_attr,
            text_encoder_config,
        ) in self.TEXT_ENCODER_CONFIGURATION.items():
            tokenizer_attr = text_encoder_attr.replace("text_encoder", "tokenizer")
            if self.text_encoders is not None and len(self.text_encoders) >= text_encoder_idx:
                pipeline_kwargs[text_encoder_attr] = self.unwrap_model(self.text_encoders[text_encoder_idx])

                # Only add tokenizer if the pipeline expects it and we have it
                if tokenizer_attr in pipeline_init_signature.parameters:
                    if self.tokenizers is not None and len(self.tokenizers) >= text_encoder_idx:
                        logger.info(f"Adding {tokenizer_attr}")
                        pipeline_kwargs[tokenizer_attr] = self.tokenizers[text_encoder_idx]
                    else:
                        pipeline_kwargs[tokenizer_attr] = None
            else:
                pipeline_kwargs[text_encoder_attr] = None
                if tokenizer_attr in pipeline_init_signature.parameters:
                    pipeline_kwargs[tokenizer_attr] = None

            text_encoder_idx += 1

        if "processor" in pipeline_init_signature.parameters and "processor" not in pipeline_kwargs:
            processor = self._load_processor_for_pipeline()
            if processor is None:
                raise ValueError(f"{pipeline_class.__name__} requires a processor but none was provided or could be loaded.")
            pipeline_kwargs["processor"] = processor

        if "text_processor" in pipeline_init_signature.parameters and "text_processor" not in pipeline_kwargs:
            text_processor = self._load_text_processor_for_pipeline()
            if text_processor is None:
                raise ValueError(
                    f"{pipeline_class.__name__} requires a text_processor but none was provided or could be loaded."
                )
            pipeline_kwargs["text_processor"] = text_processor

        if self.config.controlnet and pipeline_type is PipelineTypes.CONTROLNET:
            pipeline_kwargs["controlnet"] = self.controlnet

        optional_components = getattr(pipeline_class, "_optional_components", [])
        require_conditioning_components = bool(self.requires_conditioning_image_embeds())
        if (
            "image_encoder" in optional_components
            and "image_encoder" not in pipeline_kwargs
            and getattr(self, "config", None) is not None
        ):
            repo_id = (
                getattr(
                    self.config,
                    "image_encoder_pretrained_model_name_or_path",
                    None,
                )
                or self._model_config_path()
            )
            processor_repo_id = (
                getattr(
                    self.config,
                    "image_processor_pretrained_model_name_or_path",
                    None,
                )
                or repo_id
            )
            explicit_encoder_source = getattr(self.config, "image_encoder_pretrained_model_name_or_path", None)
            explicit_processor_source = getattr(self.config, "image_processor_pretrained_model_name_or_path", None)
            image_encoder = None
            image_processor = None
            try:
                from transformers import CLIPImageProcessor, CLIPVisionModel  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency guard
                raise ValueError(
                    "Model requires conditioning image embeds but transformers is unavailable "
                    "to load the image encoder components."
                ) from exc

            def _dedupe_subfolders(values):
                seen = set()
                result = []
                for value in values:
                    if not value or value in seen:
                        continue
                    seen.add(value)
                    result.append(value)
                return result

            encoder_subfolders = []
            config_encoder_subfolder = getattr(self.config, "image_encoder_subfolder", None)
            if isinstance(config_encoder_subfolder, (list, tuple, set)):
                encoder_subfolders.extend(config_encoder_subfolder)
            elif config_encoder_subfolder:
                encoder_subfolders.append(config_encoder_subfolder)
            encoder_subfolders.extend(("image_encoder", "vision_encoder"))
            encoder_subfolders = _dedupe_subfolders(encoder_subfolders)

            loader_errors: list[tuple[str, Exception]] = []
            encoder_revision = getattr(self.config, "image_encoder_revision", getattr(self.config, "revision", None))
            for subfolder in encoder_subfolders:
                try:
                    image_encoder = CLIPVisionModel.from_pretrained(
                        repo_id,
                        subfolder=subfolder,
                        use_safetensors=True,
                        revision=encoder_revision,
                    )
                    break
                except Exception as exc:  # pragma: no cover - defensive
                    loader_errors.append((subfolder, exc))
            if image_encoder is None:
                loader_error_text = (
                    ", ".join(f"{repo_id}/{subfolder}: {error}" for subfolder, error in loader_errors)
                    if loader_errors
                    else "no matching subfolders were found."
                )
                message = (
                    "Unable to automatically load image encoder required for conditioning embeddings from "
                    f"'{repo_id}'. Attempts failed with: {loader_error_text}"
                )
                if explicit_encoder_source:
                    raise ValueError(message) from (loader_errors[-1][1] if loader_errors else None)
                log_fn = logger.warning if require_conditioning_components else logger.debug
                log_fn(
                    "%s Set `image_encoder_pretrained_model_name_or_path` (and optionally "
                    "`image_encoder_subfolder`) in your config to provide the weights manually.",
                    message,
                )
            else:
                pipeline_kwargs["image_encoder"] = image_encoder

            processor_errors: list[tuple[str, Exception]] = []
            processor_subfolders = []
            config_processor_subfolder = getattr(self.config, "image_processor_subfolder", None)
            if isinstance(config_processor_subfolder, (list, tuple, set)):
                processor_subfolders.extend(config_processor_subfolder)
            elif config_processor_subfolder:
                processor_subfolders.append(config_processor_subfolder)
            processor_subfolders.extend(("image_processor", "feature_extractor"))
            processor_subfolders = _dedupe_subfolders(processor_subfolders)
            processor_revision = getattr(self.config, "image_processor_revision", getattr(self.config, "revision", None))
            for subfolder in processor_subfolders:
                try:
                    image_processor = CLIPImageProcessor.from_pretrained(
                        processor_repo_id,
                        subfolder=subfolder,
                        revision=processor_revision,
                    )
                    break
                except Exception as exc:  # pragma: no cover - defensive
                    processor_errors.append((subfolder, exc))
            if image_processor is None:
                processor_error_text = (
                    ", ".join(f"{processor_repo_id}/{subfolder}: {error}" for subfolder, error in processor_errors)
                    if processor_errors
                    else "no matching subfolders were found."
                )
                message = (
                    "Unable to automatically load image processor required for conditioning embeddings from "
                    f"'{processor_repo_id}'. Attempts failed with: {processor_error_text}"
                )
                if explicit_processor_source:
                    raise ValueError(message) from (processor_errors[-1][1] if processor_errors else None)
                log_fn = logger.warning if require_conditioning_components else logger.debug
                log_fn(
                    "%s Set `image_processor_pretrained_model_name_or_path` (and optionally "
                    "`image_processor_subfolder`) in your config to provide the processor configuration.",
                    message,
                )
            else:
                pipeline_kwargs["image_processor"] = image_processor

        base_scheduler = getattr(self, "noise_schedule", None)
        if "scheduler" not in pipeline_kwargs and base_scheduler is not None:
            try:
                pipeline_kwargs["scheduler"] = base_scheduler.__class__.from_config(base_scheduler.config)
            except Exception:
                pipeline_kwargs["scheduler"] = copy.deepcopy(base_scheduler)

        logger.debug(f"Initialising {pipeline_class.__name__} with components: {pipeline_kwargs}")
        if load_base_model:
            try:
                pipeline_instance = pipeline_class.from_pretrained(**pipeline_kwargs)
            except (OSError, EnvironmentError, ValueError) as exc:
                alt_repo = getattr(self.config, "pretrained_model_name_or_path", None)
                current_repo = pipeline_kwargs.get("pretrained_model_name_or_path")
                if alt_repo and isinstance(alt_repo, str) and alt_repo != current_repo:
                    logger.warning(
                        "Pipeline load failed from resolved config path '%s' (%s). Retrying with repository id '%s'.",
                        current_repo,
                        exc,
                        alt_repo,
                    )
                    alt_kwargs = dict(pipeline_kwargs)
                    alt_kwargs["pretrained_model_name_or_path"] = alt_repo
                    pipeline_instance = pipeline_class.from_pretrained(**alt_kwargs)
                else:
                    raise
        else:
            init_kwargs = {
                key: value
                for key, value in pipeline_kwargs.items()
                if key not in ("pretrained_model_name_or_path", "watermarker", "watermark")
            }
            pipeline_instance = pipeline_class(**init_kwargs)
        self.pipelines[pipeline_type] = pipeline_instance

        return pipeline_instance

    def get_conditioning_image_embedder(self):
        """Return an adapter capable of encoding conditioning images, or None if unavailable."""
        if not self.requires_conditioning_image_embeds():
            return None

        return self._get_conditioning_image_embedder()

    def _get_conditioning_image_embedder(self):
        """Subclass hook for providing conditioning image embedder (default: unsupported)."""
        return None

    def group_offload_requested(self) -> bool:
        return bool(getattr(self.config, "enable_group_offload", False))

    def _ramtorch_enabled(self) -> bool:
        return bool(getattr(self.config, "ramtorch", False))

    def _ramtorch_targets(self) -> Optional[list[str]]:
        targets = getattr(self.config, "ramtorch_target_modules", None)
        if targets is None:
            return None
        if isinstance(targets, (list, tuple)):
            normalized = [str(entry).strip() for entry in targets if str(entry).strip()]
        else:
            normalized = [segment.strip() for segment in str(targets).split(",") if segment.strip()]
        return normalized or None

    def _ramtorch_device(self):
        return getattr(self.accelerator, "device", torch.device("cuda"))

    def _ramtorch_targets_for_component(self, override: Optional[list[str]] = None) -> Optional[list[str]]:
        if override is not None:
            return override
        return self._ramtorch_targets()

    def _ramtorch_transformer_percent(self) -> Optional[float]:
        """Get the percentage of transformer Linear layers to offload (0-100)."""
        percent = getattr(self.config, "ramtorch_transformer_percent", None)
        if percent is None:
            return None
        return float(percent) if percent < 100 else None

    def _ramtorch_text_encoder_percent(self) -> Optional[float]:
        """Get the percentage of text encoder Linear layers to offload (0-100)."""
        percent = getattr(self.config, "ramtorch_text_encoder_percent", None)
        if percent is None:
            return None
        return float(percent) if percent < 100 else None

    def _apply_ramtorch_layers(
        self,
        module,
        component_label: str,
        *,
        target_patterns: Optional[list[str]] = None,
        full_ramtorch: bool = False,
        percent: Optional[float] = None,
    ) -> int:
        """
        Apply RamTorch to a module's layers.

        Args:
            module: The module to apply RamTorch to.
            component_label: Label for logging.
            target_patterns: Optional patterns to filter which Linear layers to convert.
            full_ramtorch: If True, convert all supported layer types (Linear, Embedding,
                          Conv, LayerNorm) to bouncing versions. If False, only Linear.
            percent: Optional percentage (0-100) of eligible Linear layers to replace.
        """
        if module is None or not self._ramtorch_enabled():
            return 0

        try:
            if full_ramtorch:
                # Replace all supported layer types with bouncing versions
                counts = ramtorch_utils.replace_all_layers_with_ramtorch(
                    module,
                    device=self._ramtorch_device(),
                    include_linear=True,
                    include_embedding=True,
                    include_conv=True,
                    include_layernorm=True,
                    percent=percent,
                )
                total = counts.get("linear", 0) + counts.get("other", 0)
                if total:
                    logger.info(
                        "Applied full RamTorch to %s: %d Linear, %d other layers.",
                        component_label,
                        counts.get("linear", 0),
                        counts.get("other", 0),
                    )
                # Log diagnostic info about ramtorch conversion
                ramtorch_params = 0
                ramtorch_elements = 0
                non_ramtorch_params = 0
                non_ramtorch_elements = 0
                for name, param in module.named_parameters():
                    if getattr(param, "is_ramtorch", False):
                        ramtorch_params += 1
                        ramtorch_elements += param.numel()
                    else:
                        non_ramtorch_params += 1
                        non_ramtorch_elements += param.numel()
                        if non_ramtorch_params <= 10:
                            logger.debug(
                                "Non-ramtorch param: %s (%d elements, device=%s)", name, param.numel(), param.device
                            )
                logger.info(
                    "RamTorch conversion stats for %s: %d ramtorch params (%.2f GB), %d non-ramtorch params (%.2f GB)",
                    component_label,
                    ramtorch_params,
                    ramtorch_elements * 2 / 1e9,  # bf16 = 2 bytes
                    non_ramtorch_params,
                    non_ramtorch_elements * 2 / 1e9,
                )

                # Move any remaining non-ramtorch modules to GPU (e.g., custom LayerNorm classes)
                # and buffers (e.g., position_ids)
                moved = ramtorch_utils.move_embeddings_to_device(module, self._ramtorch_device())
                if moved:
                    logger.info(
                        "Moved %s remaining non-RamTorch params to %s for %s (%.2f GB).",
                        moved,
                        self._ramtorch_device(),
                        component_label,
                        non_ramtorch_elements * 2 / 1e9,
                    )
                return total
            else:
                # Only replace Linear layers, move other layers to GPU
                replaced = ramtorch_utils.replace_linear_layers_with_ramtorch(
                    module,
                    device=self._ramtorch_device(),
                    target_patterns=self._ramtorch_targets_for_component(target_patterns),
                    name_prefix=component_label,
                    percent=percent,
                )
                if replaced:
                    logger.info("Applied RamTorch to %s Linear layers on %s.", replaced, component_label)

                # Move non-ramtorch layers to GPU so they can process GPU activations
                moved = ramtorch_utils.move_embeddings_to_device(module, self._ramtorch_device())
                if moved:
                    logger.debug(
                        "Moved %s non-RamTorch layers to %s for %s.", moved, self._ramtorch_device(), component_label
                    )

                return replaced

        except Exception as exc:
            raise RuntimeError(f"Failed to apply RamTorch to {component_label}: {exc}") from exc

    def _ramtorch_text_encoders_requested(self) -> bool:
        return self._ramtorch_enabled() and bool(getattr(self.config, "ramtorch_text_encoder", False))

    def _ramtorch_vae_requested(self) -> bool:
        return self._ramtorch_enabled() and bool(getattr(self.config, "ramtorch_vae", False))

    def _ramtorch_controlnet_requested(self) -> bool:
        return self._ramtorch_enabled() and bool(getattr(self.config, "ramtorch_controlnet", False))

    def apply_ramtorch_to_controlnet(self) -> int:
        if not self._ramtorch_controlnet_requested():
            return 0
        controlnet = getattr(self, "controlnet", None)
        if controlnet is None:
            logger.debug("RamTorch ControlNet requested but no controlnet module is initialised.")
            return 0
        return self._apply_ramtorch_layers(controlnet, "controlnet")

    @property
    def group_offload_configured(self) -> bool:
        return getattr(self, "_group_offload_configured", False)

    def get_group_offload_modules(self) -> Dict[str, torch.nn.Module]:
        modules: Dict[str, torch.nn.Module] = {}
        # Transformer (always included for transformer-based models)
        if self.MODEL_TYPE is ModelTypes.TRANSFORMER and getattr(self, "model", None) is not None:
            unwrapped_model = self.unwrap_model(self.model)
            if isinstance(unwrapped_model, torch.nn.Module):
                modules["transformer"] = unwrapped_model

        # Text encoders (optional, controlled by --group_offload_text_encoder)
        if getattr(self.config, "group_offload_text_encoder", False):
            text_encoders = getattr(self, "text_encoders", None)
            if text_encoders is not None:
                for i, te in enumerate(text_encoders):
                    if te is not None:
                        unwrapped_te = self.unwrap_model(te)
                        if isinstance(unwrapped_te, torch.nn.Module):
                            modules[f"text_encoder_{i}"] = unwrapped_te

        # VAE (optional, controlled by --group_offload_vae)
        if getattr(self.config, "group_offload_vae", False):
            vae = getattr(self, "vae", None)
            if vae is not None:
                unwrapped_vae = self.unwrap_model(vae)
                if isinstance(unwrapped_vae, torch.nn.Module):
                    modules["vae"] = unwrapped_vae

        return modules

    def _resolve_group_offload_device(self) -> torch.device:
        if hasattr(self.accelerator, "device"):
            return torch.device(self.accelerator.device)
        return torch.device("cpu")

    def _resolve_group_offload_disk_path(self):
        raw_path = getattr(self.config, "group_offload_to_disk_path", None)
        if not raw_path:
            return None
        expanded = os.path.expanduser(raw_path)
        return expanded

    def configure_group_offload(self) -> None:
        if self.group_offload_configured or not self.group_offload_requested():
            return

        if self._ramtorch_enabled():
            raise ValueError("Group offload cannot be used together with RamTorch (--ramtorch).")

        if self.MODEL_TYPE is not ModelTypes.TRANSFORMER:
            raise ValueError("Group offload is only supported for transformer-based models.")

        modules = self.get_group_offload_modules()
        if not modules:
            raise ValueError(
                "Group offload requested but no transformer module is available. Ensure the model has been loaded."
            )

        if not torch.cuda.is_available():
            raise ValueError(
                "Group offload requires a CUDA device. Disable --enable_group_offload or select a CUDA accelerator."
            )

        device = self._resolve_group_offload_device()
        if device.type != "cuda":
            device = torch.device(getattr(self.accelerator, "device", "cuda") if self.accelerator is not None else "cuda")

        use_stream = bool(getattr(self.config, "group_offload_use_stream", False))
        if use_stream and bool(getattr(self.config, "gradient_checkpointing", False)):
            logger.warning(
                "Disabling group offload streams because gradient checkpointing replays layers during backward, "
                "which breaks diffusers' group offload prefetch order and leads to CPU/CUDA mismatches. "
                "Re-run without --gradient_checkpointing if you need streamed group offload."
            )
            use_stream = False
            setattr(self.config, "group_offload_use_stream", False)

        offload_type = getattr(self.config, "group_offload_type", "block_level")
        blocks_per_group = getattr(self.config, "group_offload_blocks_per_group", 1)

        try:
            enable_group_offload_on_components(
                modules,
                device=device,
                offload_type=offload_type,
                number_blocks_per_group=blocks_per_group,
                use_stream=use_stream,
                offload_to_disk_path=self._resolve_group_offload_disk_path(),
            )
        except Exception as error:
            raise RuntimeError(f"Failed to configure group offloading: {error}") from error

        logger.info("Group offloading enabled for %s.", ", ".join(modules.keys()))
        self._group_offload_configured = True

    def chunked_feed_forward_requested(self) -> bool:
        return bool(getattr(self.config, "enable_chunked_feed_forward", False))

    def configure_chunked_feed_forward(self) -> None:
        if not self.chunked_feed_forward_requested():
            return

        if not self.supports_chunked_feed_forward():
            raise ValueError(
                f"{self.__class__.__name__} does not support feed-forward chunking. Disable "
                "`--enable_chunked_feed_forward` for this model family."
            )

        if self.model is None:
            raise RuntimeError("Model must be loaded before enabling feed-forward chunking.")

        raw_chunk_size = getattr(self.config, "feed_forward_chunk_size", None)
        chunk_size_value: Optional[int]
        if raw_chunk_size in ("", None):
            chunk_size_value = None
        else:
            try:
                chunk_size_value = int(raw_chunk_size)
            except (TypeError, ValueError) as error:
                raise ValueError("`feed_forward_chunk_size` must be a positive integer when provided.") from error
            if chunk_size_value <= 0:
                chunk_size_value = None

        chunk_dim_value = getattr(self.config, "feed_forward_chunk_dim", None)
        try:
            chunk_dim_value = int(chunk_dim_value) if chunk_dim_value is not None else None
        except (TypeError, ValueError):
            chunk_dim_value = None

        try:
            self.enable_chunked_feed_forward(chunk_size=chunk_size_value, chunk_dim=chunk_dim_value)
        except Exception as error:  # pragma: no cover - safety net
            raise RuntimeError(f"Failed to enable feed-forward chunking: {error}") from error

        logger.info(
            "Feed-forward chunking enabled for %s (%s).",
            self.__class__.__name__,
            (
                f"chunk_size={'auto' if chunk_size_value is None else chunk_size_value}, "
                f"chunk_dim={'auto' if chunk_dim_value is None else chunk_dim_value}"
            ),
        )

    def get_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True) -> "DiffusionPipeline":
        possibly_cached_pipeline = self._load_pipeline(pipeline_type, load_base_model)
        if self.model is not None and getattr(possibly_cached_pipeline, self.MODEL_TYPE.value, None) is None:
            # if the transformer or unet aren't in the cached pipeline, we'll add it.
            # For FSDP models, we should NOT unwrap them - FSDP implements __call__ transparently
            # and unwrapping breaks validation with reshard_after_forward=True
            if FSDP_AVAILABLE and isinstance(self.model, FSDP):
                model_for_pipeline = self.model  # Keep FSDP wrapper
            else:
                model_for_pipeline = self.unwrap_model(model=self.model)

            setattr(
                possibly_cached_pipeline,
                self.MODEL_TYPE.value,
                model_for_pipeline,
            )
        # attach the vae to the cached pipeline.
        setattr(possibly_cached_pipeline, "vae", self.get_vae())
        if self.text_encoders is not None:
            for (
                text_encoder_attr,
                text_encoder_config,
            ) in self.TEXT_ENCODER_CONFIGURATION.items():
                if getattr(possibly_cached_pipeline, text_encoder_attr, None) is None:
                    text_encoder_attr_number = 1
                    if "encoder_" in text_encoder_attr:
                        # support multi-encoder model pipelines
                        text_encoder_attr_number = text_encoder_attr.split("_")[-1]
                    setattr(
                        possibly_cached_pipeline,
                        text_encoder_attr,
                        self.text_encoders[int(text_encoder_attr_number) - 1],
                    )
        if self.config.controlnet:
            if getattr(possibly_cached_pipeline, "controlnet", None) is None:
                setattr(possibly_cached_pipeline, "controlnet", self.controlnet)

        return possibly_cached_pipeline

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        """
        When we're running the pipeline, we'll update the kwargs specifically for this model here.
        """

        return pipeline_kwargs

    def setup_training_noise_schedule(self):
        """
        Loads the noise schedule from the config.

        It's important to note, this is the *training* schedule, not inference.
        """
        if not self.uses_noise_schedule():
            self.noise_schedule = None
            return self.config, None
        flow_matching = False
        if self.PREDICTION_TYPE is PredictionTypes.FLOW_MATCHING:
            from diffusers import FlowMatchEulerDiscreteScheduler

            self.noise_schedule = FlowMatchEulerDiscreteScheduler.from_pretrained(
                get_model_config_path(self.config.model_family, self.config.pretrained_model_name_or_path),
                subfolder="scheduler",
                shift=self.config.flow_schedule_shift,
            )
            flow_matching = True
        elif self.PREDICTION_TYPE in [
            PredictionTypes.EPSILON,
            PredictionTypes.V_PREDICTION,
            PredictionTypes.SAMPLE,
        ]:
            from diffusers import DDPMScheduler

            self.noise_schedule = DDPMScheduler.from_pretrained(
                get_model_config_path(self.config.model_family, self.config.pretrained_model_name_or_path),
                subfolder="scheduler",
                rescale_betas_zero_snr=self.config.rescale_betas_zero_snr,
                timestep_spacing=self.config.training_scheduler_timestep_spacing,
            )
            if self.config.prediction_type is None:
                self.config.prediction_type = self.noise_schedule.config.prediction_type
        else:
            raise NotImplementedError(f"Unknown prediction type {self.PREDICTION_TYPE}.")

        return self.config, self.noise_schedule

    def setup_diff2flow_bridge(self):
        """
        Optionally attach a diffusion-to-flow adapter for epsilon/v-prediction models.
        """
        self.config.diff2flow_enabled = getattr(self.config, "diff2flow_enabled", False)
        self.config.diff2flow_loss = getattr(self.config, "diff2flow_loss", False)
        if self.PREDICTION_TYPE not in [PredictionTypes.EPSILON, PredictionTypes.V_PREDICTION]:
            return
        if not self.config.diff2flow_enabled:
            return
        alphas_cumprod = getattr(self.noise_schedule, "alphas_cumprod", None)
        if alphas_cumprod is None or not torch.is_tensor(alphas_cumprod):
            logger.warning("Diff2Flow requested but scheduler lacks alphas_cumprod; disabling.")
            self.config.diff2flow_enabled = False
            return
        self.diff2flow_bridge = DiffusionToFlowBridge(alphas_cumprod=alphas_cumprod)
        self.diff2flow_bridge.to(device=self.accelerator.device, dtype=self.config.weight_dtype)

    def get_prediction_target(self, prepared_batch: dict):
        """
        Returns the target used in the loss function.
        Depending on the noise schedule prediction type or flow-matching settings,
        the target is computed differently.
        """
        if prepared_batch.get("target") is not None:
            # Parent-student training
            target = prepared_batch["target"]
        elif self.PREDICTION_TYPE is PredictionTypes.FLOW_MATCHING:
            target = prepared_batch["noise"] - prepared_batch["latents"]
        elif self.PREDICTION_TYPE is PredictionTypes.EPSILON:
            target = prepared_batch["noise"]
        elif self.PREDICTION_TYPE is PredictionTypes.V_PREDICTION:
            target = self.noise_schedule.get_velocity(
                prepared_batch["latents"],
                prepared_batch["noise"],
                prepared_batch["timesteps"],
            )
        elif self.PREDICTION_TYPE is PredictionTypes.SAMPLE:
            target = prepared_batch["latents"]
        else:
            raise ValueError(f"Unknown prediction type {self.PREDICTION_TYPE}.")
        return target

    def get_flow_target(self, prepared_batch: dict):
        flow_target = prepared_batch.get("flow_target")
        if flow_target is not None:
            return flow_target
        if self.diff2flow_bridge is None:
            return None
        return self.diff2flow_bridge.flow_target(prepared_batch["latents"], prepared_batch["noise"])

    def prepare_batch_conditions(self, batch: dict, state: dict) -> dict:
        # it's a list, but most models will expect it to be a length-1 list containing a tensor, which is what they actually want
        if isinstance(batch.get("conditioning_pixel_values"), list) and len(batch["conditioning_pixel_values"]) > 0:
            batch["conditioning_pixel_values"] = batch["conditioning_pixel_values"][0]
        if isinstance(batch.get("conditioning_latents"), list) and len(batch["conditioning_latents"]) > 0:
            batch["conditioning_latents"] = batch["conditioning_latents"][0]
        conditioning_embeds = batch.get("conditioning_image_embeds")
        if isinstance(conditioning_embeds, list) and len(conditioning_embeds) > 0:
            if not isinstance(conditioning_embeds[0], dict):
                batch["conditioning_image_embeds"] = conditioning_embeds[0]
        return batch

    def _get_patch_size_for_dynamic_shift(self, noise_scheduler):
        component = None
        try:
            component = self.get_trained_component()
        except Exception:
            component = None

        if component is not None:
            patch_size = getattr(getattr(component, "config", None), "patch_size", None)
            if patch_size is not None:
                return patch_size

        scheduler_config = getattr(noise_scheduler, "config", None)
        if scheduler_config is not None:
            patch_size = getattr(scheduler_config, "patch_size", None)
            if patch_size is not None:
                return patch_size

        return getattr(self.config, "patch_size", None)

    def calculate_dynamic_shift_mu(self, noise_scheduler, latents: torch.Tensor | None):
        """
        Compute resolution-dependent shift value for schedulers that support dynamic shifting.
        """
        if latents is None:
            return None

        scheduler_config = getattr(noise_scheduler, "config", None)
        if scheduler_config is None:
            return None

        required_fields = [
            "base_image_seq_len",
            "max_image_seq_len",
            "base_shift",
            "max_shift",
        ]
        missing_fields = [field for field in required_fields if getattr(scheduler_config, field, None) is None]
        if missing_fields:
            raise ValueError(
                f"Cannot compute dynamic timestep shift; scheduler is missing config values: {', '.join(missing_fields)}"
            )

        patch_size = self._get_patch_size_for_dynamic_shift(noise_scheduler)
        if patch_size is None or patch_size <= 0:
            raise ValueError("Cannot compute dynamic timestep shift because no valid `patch_size` was found.")

        height, width = latents.shape[-2:]
        image_seq_len = (int(height) // int(patch_size)) * (int(width) // int(patch_size))
        from simpletuner.helpers.models.sd3.pipeline import calculate_shift

        return calculate_shift(
            image_seq_len,
            scheduler_config.base_image_seq_len,
            scheduler_config.max_image_seq_len,
            scheduler_config.base_shift,
            scheduler_config.max_shift,
        )

    def _normalize_flow_custom_timesteps(self, raw_value) -> Optional[torch.Tensor]:
        """
        Parse user-specified custom flow timesteps/sigmas into a 1D tensor on the current device.
        Accepts comma-separated strings, JSON-style lists, or tensor/array inputs.
        """
        if raw_value in (None, "", "None"):
            return None

        candidate = raw_value
        if isinstance(candidate, str):
            stripped = candidate.strip()
            if stripped == "":
                return None
            try:
                candidate = json.loads(stripped)
            except Exception:
                segments = [seg for seg in stripped.replace(";", ",").split(",") if seg.strip()]
                try:
                    candidate = [float(seg.strip()) for seg in segments]
                except Exception:
                    return None

        if isinstance(candidate, np.ndarray):
            candidate = candidate.tolist()

        try:
            tensor = torch.as_tensor(candidate, device=self.accelerator.device, dtype=torch.float32).flatten()
        except Exception:
            return None

        if tensor.numel() == 0:
            return None

        finite_mask = torch.isfinite(tensor)
        if not torch.all(finite_mask):
            tensor = tensor[finite_mask]
        if tensor.numel() == 0:
            return None

        return tensor

    def sample_flow_sigmas(self, batch: dict, state: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample flow-matching sigmas/timesteps for the current batch.

        Subclasses can override to implement model-specific sampling strategies.
        """
        bsz = batch["latents"].shape[0]
        custom_timesteps = self._normalize_flow_custom_timesteps(getattr(self.config, "flow_custom_timesteps", None))
        if custom_timesteps is not None:
            # Interpret values <=1.0 as sigmas, otherwise as timesteps in [0, 1000].
            if torch.max(custom_timesteps) <= 1.0:
                base_sigmas = custom_timesteps.clamp(0.0, 1.0)
                base_timesteps = base_sigmas * 1000.0
            else:
                base_timesteps = custom_timesteps.clamp(0.0, 1000.0)
                base_sigmas = (base_timesteps / 1000.0).clamp(0.0, 1.0)

            if base_timesteps.numel() == 1:
                sigmas = base_sigmas.expand(bsz)
                timesteps = base_timesteps.expand(bsz)
            else:
                indices = torch.randint(0, base_timesteps.numel(), (bsz,), device=self.accelerator.device)
                sigmas = base_sigmas[indices]
                timesteps = base_timesteps[indices]
            return sigmas, timesteps

        if not self.config.flux_fast_schedule and not any(
            [
                self.config.flow_use_beta_schedule,
                self.config.flow_use_uniform_schedule,
            ]
        ):
            sigmas = torch.sigmoid(self.config.flow_sigmoid_scale * torch.randn((bsz,), device=self.accelerator.device))
            sigmas = apply_flow_schedule_shift(self.config, self.noise_schedule, sigmas, batch["noise"])
        elif self.config.flow_use_uniform_schedule:
            sigmas = torch.rand((bsz,), device=self.accelerator.device)
            sigmas = apply_flow_schedule_shift(self.config, self.noise_schedule, sigmas, batch["noise"])
        elif self.config.flow_use_beta_schedule:
            alpha = self.config.flow_beta_schedule_alpha
            beta = self.config.flow_beta_schedule_beta
            beta_dist = Beta(alpha, beta)
            sigmas = beta_dist.sample((bsz,)).to(device=self.accelerator.device)
            sigmas = apply_flow_schedule_shift(self.config, self.noise_schedule, sigmas, batch["noise"])
        else:
            available_sigmas = [1.0] * 7 + [0.75, 0.5, 0.25]
            sigmas = torch.tensor(
                random.choices(available_sigmas, k=bsz),
                device=self.accelerator.device,
            )
        timesteps = sigmas * 1000.0
        return sigmas, timesteps

    def _validate_twinflow_config(self) -> None:
        """
        Validate TwinFlow configuration and record common flags.
        """
        # Mirror reference TwinFlow: always capture/restore RNG around teacher passes.
        self._twinflow_store_rng = True
        # Default TwinFlow flag off unless explicitly enabled; allow model configs to opt in via attribute.
        if not hasattr(self.config, "twinflow_enabled"):
            setattr(self.config, "twinflow_enabled", False)
        self._twinflow_allow_student_teacher = bool(getattr(self.config, "twinflow_allow_no_ema_teacher", False))
        self._twinflow_requires_ema = bool(getattr(self.config, "twinflow_require_ema", True))
        self._twinflow_diffusion_bridge = False

        if not getattr(self.config, "twinflow_enabled", False):
            return

        prediction_is_flow = self.PREDICTION_TYPE is PredictionTypes.FLOW_MATCHING
        allow_diff2flow = bool(getattr(self.config, "twinflow_allow_diff2flow", False))
        uses_diff2flow = bool(getattr(self.config, "diff2flow_enabled", False))
        if not prediction_is_flow:
            if not (allow_diff2flow and uses_diff2flow):
                raise ValueError(
                    "TwinFlow requires flow-matching prediction_type. Enable diff2flow_enabled and "
                    "twinflow_allow_diff2flow to bridge epsilon/v_prediction models explicitly."
                )
            self._twinflow_diffusion_bridge = True

        if (
            self._twinflow_requires_ema
            and not getattr(self.config, "use_ema", False)
            and not self._twinflow_allow_student_teacher
        ):
            raise ValueError(
                "TwinFlow requires an EMA teacher; enable use_ema or set twinflow_allow_no_ema_teacher "
                "to fall back to student weights explicitly."
            )

    def _init_layersync_regularizer(self):
        if not getattr(self.config, "layersync_enabled", False):
            self.layersync_regularizer = None
            return
        self.layersync_regularizer = LayerSyncRegularizer(self.config)

    def _needs_hidden_state_buffer(self) -> bool:
        layersync = getattr(self, "layersync_regularizer", None)
        ls_needed = bool(layersync and layersync.wants_hidden_states())
        crepa = getattr(self, "crepa_regularizer", None)
        crepa_buffer = bool(crepa and crepa.enabled and getattr(crepa, "use_backbone_features", False))
        return ls_needed or crepa_buffer

    def _new_hidden_state_buffer(self) -> Optional[HiddenStateBuffer]:
        return HiddenStateBuffer() if self._needs_hidden_state_buffer() else None

    def _apply_layersync_regularizer(
        self, loss: torch.Tensor, aux_logs: Optional[dict], hidden_states_buffer: Optional[dict]
    ) -> tuple[torch.Tensor, Optional[dict]]:
        layersync = getattr(self, "layersync_regularizer", None)
        if layersync and layersync.wants_hidden_states():
            ls_loss, ls_logs = layersync.compute_loss(hidden_states_buffer)
            if ls_loss is not None:
                loss = loss + ls_loss
            if ls_logs:
                aux_logs = (aux_logs or {}) | ls_logs
        return loss, aux_logs

    def _maybe_enable_reflexflow_default(self) -> bool:
        """
        Enable ReflexFlow automatically when scheduled sampling is active on flow-matching models
        and the user did not explicitly set the flag.
        """
        try:
            offset_value = getattr(self.config, "scheduled_sampling_max_step_offset", 0)
            max_offset = float(offset_value or 0)
        except Exception:
            return False

        if max_offset <= 0:
            return False

        if getattr(self.config, "scheduled_sampling_reflexflow", None) is not None:
            return False

        if self.PREDICTION_TYPE is not PredictionTypes.FLOW_MATCHING:
            return False

        setattr(self.config, "scheduled_sampling_reflexflow", True)
        return True

    def _twinflow_active(self) -> bool:
        """
        Guard TwinFlow/RCGM auxiliary losses to flow-matching models with an explicit opt-in flag.
        """
        if not getattr(self.config, "twinflow_enabled", False):
            return False
        if self.PREDICTION_TYPE is not PredictionTypes.FLOW_MATCHING and not self._twinflow_diffusion_bridge:
            raise ValueError(
                "TwinFlow is only supported for flow-matching models. Enable diff2flow bridging explicitly "
                "to run with epsilon/v_prediction targets."
            )
        return True

    def _twinflow_settings(self) -> dict:
        """
        Collect TwinFlow hyperparameters with sensible defaults.
        Following the original TwinFlow paper, all loss components are enabled by default.
        """
        return {
            "estimate_order": max(1, int(getattr(self.config, "twinflow_estimate_order", 2) or 2)),
            "enhanced_ratio": float(getattr(self.config, "twinflow_enhanced_ratio", 0.5) or 0.5),
            "target_step_count": max(1, int(getattr(self.config, "twinflow_target_step_count", 1) or 1)),
            "delta_t": float(getattr(self.config, "twinflow_delta_t", 0.01) or 0.01),
            "clamp_target": float(getattr(self.config, "twinflow_target_clamp", 1.0) or 1.0),
            "require_ema": bool(getattr(self.config, "twinflow_require_ema", True)),
            "use_rng_state": True,
            # Adversarial branch settings (L_adv + L_rectify)
            "adversarial_enabled": bool(getattr(self.config, "twinflow_adversarial_enabled", False)),
            "adversarial_weight": float(getattr(self.config, "twinflow_adversarial_weight", 1.0) or 1.0),
            "rectify_weight": float(getattr(self.config, "twinflow_rectify_weight", 1.0) or 1.0),
        }

    @staticmethod
    def _twinflow_restore_rng_state(rng_state: Optional[Mapping[str, torch.Tensor]]) -> None:
        """
        Restore RNG state captured during batch preparation.
        """
        if rng_state is None:
            return
        try:
            cpu_state = rng_state.get("cpu")
            if cpu_state is not None:
                torch.random.set_rng_state(cpu_state)
            cuda_state = rng_state.get("cuda")
            if cuda_state is not None and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(cuda_state)
                except Exception:
                    torch.cuda.set_rng_state(cuda_state)
        except Exception:
            logger.debug("Unable to restore RNG state for TwinFlow teacher run.", exc_info=True)

    @staticmethod
    def _twinflow_sample_tt(sigmas: torch.Tensor) -> torch.Tensor:
        """
        Sample a secondary time tt for TwinFlow such that tt < t.
        """
        tt = sigmas - torch.rand_like(sigmas) * sigmas
        eps = torch.finfo(sigmas.dtype).eps if torch.is_floating_point(sigmas) else 1e-4
        # Use torch.minimum/maximum to handle tensor bounds (clamp doesn't mix scalar and tensor)
        tt = torch.maximum(tt, torch.zeros_like(tt))
        tt = torch.minimum(tt, sigmas - eps)
        return tt

    def _twinflow_generate_fake_samples(
        self,
        prepared_batch: dict,
        settings: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate fake samples via one-step generation (t=1, tt=0).

        Following the TwinFlow reference implementation, we run the model with
        t=1 (pure noise) and tt=0 (predict to clean) to generate fake samples.

        Returns:
            (x_fake, z): The generated fake sample and the noise used.
        """
        latents = prepared_batch["latents"]
        z = torch.randn_like(latents)

        ones = torch.ones(z.shape[0], device=z.device, dtype=z.dtype)
        zeros = torch.zeros_like(ones)

        with torch.no_grad():
            F_fake = self._twinflow_forward(
                prepared_batch=prepared_batch,
                noisy_latents=z,
                sigmas=ones,
                tt=zeros,
                use_grad=False,
            )

        # Integrate one step: x_fake = z - F_fake (from t=1 to t=0)
        x_fake = z - F_fake
        return x_fake.detach(), z

    def _twinflow_compute_adversarial_loss(
        self,
        prepared_batch: dict,
        x_fake: torch.Tensor,
        z: torch.Tensor,
        settings: dict,
    ) -> torch.Tensor:
        """
        Compute L_adv: train fake trajectory with negative time.

        From the TwinFlow paper:
            x_t_fake = t*z + (1-t)*x_fake
            target_fake = z - x_fake (velocity from fake clean to noise)
            Forward with -t (negative time signals fake trajectory)
            L_adv = MSE(F_pred, target_fake)
        """
        bsz = x_fake.shape[0]
        # Sample time for fake trajectory
        t = torch.rand(bsz, device=x_fake.device, dtype=x_fake.dtype).clamp(min=0.01, max=0.99)
        t_b = self._twinflow_match_time_shape(t, x_fake)

        # Construct fake trajectory interpolation
        x_t_fake = t_b * z + (1 - t_b) * x_fake

        # Target velocity for fake trajectory
        target_fake = z - x_fake

        # Forward with NEGATIVE time (sign embedding handles distinction)
        neg_t = -t
        F_pred = self._twinflow_forward(
            prepared_batch=prepared_batch,
            noisy_latents=x_t_fake,
            sigmas=neg_t,
            tt=None,
            use_grad=True,
        )

        return F.mse_loss(F_pred.float(), target_fake.float(), reduction="mean")

    def _twinflow_compute_rectify_loss(
        self,
        prepared_batch: dict,
        base_pred: torch.Tensor,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        settings: dict,
    ) -> torch.Tensor:
        """
        Compute L_rectify: align real/fake trajectory predictions.

        From the TwinFlow paper:
            F_grad = F(x_t, -t) - F(x_t, t)
            L_rectify = MSE(F(x_t, t), sg(F(x_t, t) - F_grad))

        This aligns the real trajectory predictions with corrections from the
        fake trajectory, enabling distribution matching without a discriminator.
        """
        # Get prediction at negative time (same x_t, opposite trajectory)
        with torch.no_grad():
            F_neg = self._twinflow_forward(
                prepared_batch=prepared_batch,
                noisy_latents=noisy_latents,
                sigmas=-sigmas,
                tt=None,
                use_grad=False,
            )

        # F_grad = F(x_t, -t) - F(x_t, t)
        F_grad = F_neg - base_pred.detach()

        # Target with stop gradient: sg(F(x_t, t) - F_grad)
        rectify_target = (base_pred.detach() - F_grad).detach()

        return F.mse_loss(base_pred.float(), rectify_target.float(), reduction="mean")

    def _prepare_twinflow_metadata(self, batch: dict) -> None:
        """
        Prepare TwinFlow-specific batch entries (tt and optional RNG state).
        """
        sigmas = batch.get("sigmas")
        if sigmas is None:
            return

        tt = batch.get("twinflow_tt")
        if tt is None:
            tt = self._twinflow_sample_tt(sigmas)
        else:
            eps = torch.finfo(sigmas.dtype).eps if torch.is_floating_point(sigmas) else 1e-4
            tt = torch.clamp(tt, min=0.0, max=sigmas - eps)
        batch["twinflow_tt"] = tt

        if self._twinflow_store_rng:
            try:
                rng_state = {"cpu": torch.random.get_rng_state()}
                if torch.cuda.is_available():
                    rng_state["cuda"] = torch.cuda.get_rng_state_all()
                batch["twinflow_rng_state"] = rng_state
            except Exception:
                logger.debug("Unable to snapshot RNG state for TwinFlow dual passes.", exc_info=True)

    @staticmethod
    def _twinflow_match_time_shape(time_tensor: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """
        Broadcast a per-sample time tensor to match latent or prediction shapes.
        """
        if time_tensor is None:
            return None
        while time_tensor.dim() < like.dim():
            time_tensor = time_tensor.unsqueeze(-1)
        return time_tensor

    @staticmethod
    def _twinflow_diffusion_xt(
        bridge,
        latents: torch.Tensor,
        noise: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct diffusion-style x_t for a given flow sigma using the bridge's alphas.
        """
        timesteps = bridge.sigma_to_timesteps(sigma.view(sigma.shape[0], -1)[:, 0])
        sqrt_alpha = bridge._extract(bridge.sqrt_alphas_cumprod, timesteps, latents.shape)
        sqrt_one_minus = bridge._extract(bridge.sqrt_one_minus_alphas_cumprod, timesteps, latents.shape)
        return sqrt_alpha * latents + sqrt_one_minus * noise

    @contextlib.contextmanager
    def _twinflow_teacher_context(
        self,
        prepared_batch: dict,
        settings: Mapping[str, object],
        rng_state: Optional[Mapping[str, torch.Tensor]],
    ):
        """
        Swap EMA weights in for teacher predictions when available.
        """
        ema_model = getattr(self, "ema_model", None)
        has_ema = ema_model is not None and getattr(self.config, "use_ema", False)
        require_ema = bool(settings.get("require_ema", True))
        teacher_parameters = None

        if require_ema and not has_ema and not self._twinflow_allow_student_teacher:
            raise ValueError("TwinFlow requires EMA teacher weights; enable use_ema or allow student fallback explicitly.")

        if has_ema:
            teacher_component = self.get_trained_component(unwrap_model=False)
            if teacher_component is None:
                raise ValueError("TwinFlow teacher unavailable because no trainable component was found.")
            teacher_parameters = list(teacher_component.parameters())
            ema_model.store(teacher_parameters)
            ema_model.copy_to(teacher_parameters)

        try:

            def _teacher_forward(
                noisy_latents: torch.Tensor,
                sigma: torch.Tensor,
                tt: Optional[torch.Tensor] = None,
                batch: Optional[dict] = None,
                use_diff2flow_bridge: bool = False,
            ) -> torch.Tensor:
                if rng_state is not None:
                    self._twinflow_restore_rng_state(rng_state)
                target_batch = batch if batch is not None else prepared_batch
                return self._twinflow_forward(
                    prepared_batch=target_batch,
                    noisy_latents=noisy_latents,
                    sigmas=sigma,
                    tt=tt,
                    use_grad=False,
                    use_diff2flow_bridge=use_diff2flow_bridge,
                )

            yield _teacher_forward
        finally:
            if has_ema and teacher_parameters is not None:
                try:
                    ema_model.restore(teacher_parameters)
                except Exception:
                    logger.exception("Failed to restore student weights after TwinFlow teacher swap.")

    def _twinflow_cfg_batches(self, prepared_batch: dict) -> Optional[list[dict]]:
        """
        Build conditional/unconditional batch copies when CFG target enhancement is requested.
        """
        negative = prepared_batch.get("negative_prompt_embeds")
        if negative is None:
            negative = prepared_batch.get("uncond_prompt_embeds")
        positive = prepared_batch.get("prompt_embeds")
        if positive is None:
            positive = prepared_batch.get("encoder_hidden_states")
        if negative is None or positive is None:
            return None
        cond_batch = dict(prepared_batch)
        uncond_batch = dict(prepared_batch)
        if "prompt_embeds" in cond_batch:
            cond_batch["prompt_embeds"] = positive
            uncond_batch["prompt_embeds"] = negative
        if "encoder_hidden_states" in cond_batch:
            cond_batch["encoder_hidden_states"] = positive
            uncond_batch["encoder_hidden_states"] = negative
        return [cond_batch, uncond_batch]

    def _twinflow_enhance_target(
        self,
        prepared_batch: dict,
        teacher_forward,
        target: torch.Tensor,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        tt: torch.Tensor,
        settings: Mapping[str, object],
        use_diff2flow_bridge: bool = False,
    ) -> torch.Tensor:
        """
        Optionally refine the target using CFG teacher predictions.

        For diff2flow bridge mode, set use_diff2flow_bridge=True so timesteps are
        derived from the current sigma, keeping UNet conditioning in sync.
        """
        ratio = float(settings.get("enhanced_ratio", 0.0) or 0.0)
        if ratio <= 0.0:
            return target

        cfg_batches = self._twinflow_cfg_batches(prepared_batch)
        if not cfg_batches or len(cfg_batches) < 2:
            return target

        try:
            teacher_cond = teacher_forward(
                noisy_latents=noisy_latents,
                sigma=sigmas,
                tt=tt,
                batch=cfg_batches[0],
                use_diff2flow_bridge=use_diff2flow_bridge,
            )
            teacher_uncond = teacher_forward(
                noisy_latents=noisy_latents,
                sigma=sigmas,
                tt=tt,
                batch=cfg_batches[1],
                use_diff2flow_bridge=use_diff2flow_bridge,
            )
        except Exception:
            logger.debug("TwinFlow CFG target enhancement failed; using base target.", exc_info=True)
            return target
        return target + ratio * (teacher_cond - teacher_uncond)

    def _twinflow_forward(
        self,
        prepared_batch: dict,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
        tt: Optional[torch.Tensor] = None,
        use_grad: bool = True,
        use_diff2flow_bridge: bool = False,
    ) -> torch.Tensor:
        """
        Run a forward pass with custom noisy latents and sigma/tt values.

        For diff2flow bridge mode (use_diff2flow_bridge=True), timesteps are derived
        from the current sigma using the bridge's sigma_to_timesteps(), ensuring the
        UNet receives timestep conditioning that matches the noise level of noisy_latents.
        The prediction is then converted to flow.
        """
        twin_batch = dict(prepared_batch)
        twin_batch["noisy_latents"] = noisy_latents
        # Split sigma into magnitude and sign for models that can consume signed time.
        sigma_flat = sigmas.view(sigmas.shape[0], -1)[:, 0]
        sigma_sign = torch.sign(sigma_flat)
        sigma_sign = torch.where(sigma_sign == 0, torch.ones_like(sigma_sign), sigma_sign)
        sigma_abs = sigmas.abs()
        twin_batch["sigmas"] = sigma_abs
        if tt is not None:
            twin_batch["twinflow_tt"] = tt
        twin_batch["twinflow_time_sign"] = sigma_sign

        # Derive timesteps from sigma
        if use_diff2flow_bridge and self.diff2flow_bridge is not None:
            # Convert current flow sigma to matching diffusion timestep
            diffusion_timesteps = self.diff2flow_bridge.sigma_to_timesteps(sigma_abs.view(sigma_abs.shape[0], -1)[:, 0])
            twin_batch["timesteps"] = diffusion_timesteps.to(dtype=sigmas.dtype)
        else:
            # Native flow mode: derive timesteps from |sigma| (scale to [0, 1000]).
            # Negative-time semantics are carried by twinflow_time_sign.
            twin_batch["timesteps"] = sigma_abs.view(sigmas.shape[0], -1)[:, 0] * 1000.0
            diffusion_timesteps = None

        if use_grad:
            pred = self.model_predict(prepared_batch=twin_batch)["model_prediction"]
        else:
            with torch.no_grad():
                pred = self.model_predict(prepared_batch=twin_batch)["model_prediction"]

        # Convert prediction to flow if using diff2flow bridge
        if use_diff2flow_bridge and self.diff2flow_bridge is not None and diffusion_timesteps is not None:
            pred = self.diff2flow_bridge.prediction_to_flow(
                pred.float(),
                noisy_latents.float(),
                diffusion_timesteps,
                prediction_type=self.PREDICTION_TYPE.value,
            )

        return pred

    @staticmethod
    def _twinflow_reconstruct_states(x_t: torch.Tensor, sigma: torch.Tensor, flow_pred: torch.Tensor):
        """
        Recover x_hat and z_hat from the predicted flow under linear interpolation x_t = t*z + (1-t)*x.
        """
        sigma_b = ModelFoundation._twinflow_match_time_shape(sigma, x_t)
        gamma = 1 - sigma_b
        x_hat = x_t - sigma_b * flow_pred
        z_hat = x_t + gamma * flow_pred
        return x_hat, z_hat

    def _twinflow_rcgm_target(
        self,
        prepared_batch: dict,
        base_pred: torch.Tensor,
        target: torch.Tensor,
        noisy_latents: torch.Tensor,
        sigma: torch.Tensor,
        tt: torch.Tensor,
        settings: dict,
        teacher_forward,
        use_diff2flow_bridge: bool = False,
        rcgm_latents: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Approximate the recursive consistency target (RCGM) used by TwinFlow.

        For diff2flow bridge mode, set use_diff2flow_bridge=True so each teacher
        call derives timesteps from the current sigma (t_prev), keeping UNet
        conditioning in sync as x_t evolves through the integration loop.
        """
        delta = settings.get("delta_t", 0.01)
        sigma_b = self._twinflow_match_time_shape(sigma, base_pred)
        tt_b = self._twinflow_match_time_shape(tt, base_pred)
        clamp_val = settings.get("clamp_target", 1.0)
        steps = max(1, int(settings.get("estimate_order", 1)))
        t_anchor = torch.maximum(tt_b, sigma_b - delta)

        x_t = rcgm_latents if rcgm_latents is not None else noisy_latents
        pred_accum = torch.zeros_like(base_pred)
        t_prev = sigma_b

        time_schedule = []
        if steps == 1:
            time_schedule.append(tt_b)
        else:
            for i in range(steps - 1):
                frac = float(i + 1) / float(steps)
                time_schedule.append(t_anchor * frac + sigma_b * (1 - frac))
            time_schedule.append(tt_b)

        for t_next in time_schedule:
            # Teacher runs on diffusion-style noisy latents; flow RC-GM operates on flow-style x_t.
            teacher_input = x_t
            if use_diff2flow_bridge and self.diff2flow_bridge is not None and latents is not None and noise is not None:
                teacher_input = self._twinflow_diffusion_xt(self.diff2flow_bridge, latents, noise, t_prev)
            # Pass t_prev as sigma so the bridge can derive matching diffusion timesteps
            F_c = teacher_forward(teacher_input, t_prev, tt=t_next, use_diff2flow_bridge=use_diff2flow_bridge)
            x_hat, z_hat = self._twinflow_reconstruct_states(x_t, t_prev, F_c)
            x_t = t_next * z_hat + (1 - t_next) * x_hat
            pred_accum = pred_accum + F_c * (t_prev - t_next)
            t_prev = t_next

        # Detach base_pred to avoid self-referential gradients in the target.
        # Reference: TwinFlow MNIST uses F_th_t.data (equivalent to detach).
        base_pred_detached = base_pred.detach()
        rcgm_raw = base_pred_detached - pred_accum - target
        rcgm = base_pred_detached - rcgm_raw.clamp(min=-clamp_val, max=clamp_val)
        return rcgm

    def prepare_batch(self, batch: dict, state: dict) -> dict:
        """
        Moves the batch to the proper device/dtype,
        samples noise, timesteps and, if applicable, flow-matching sigmas.
        This code is mostly common across models, but if you'd like to override certain pieces, use prepare_batch_conditions.

        Args:
            batch (dict): The batch to prepare.
            state (dict): The training state.
        Returns:
            dict: The prepared batch.
        """
        if not batch:
            return batch

        target_device_kwargs = {
            "device": self.accelerator.device,
            "dtype": self.config.weight_dtype,
        }

        logger.debug(f"Preparing batch: {batch.keys()}")
        # Ensure the encoder hidden states are on device
        if batch["prompt_embeds"] is not None and hasattr(batch["prompt_embeds"], "to"):
            batch["encoder_hidden_states"] = batch["prompt_embeds"].to(**target_device_kwargs)

        # Process additional conditioning if provided
        pooled_embeds = batch.get("add_text_embeds")
        time_ids = batch.get("batch_time_ids")
        batch["added_cond_kwargs"] = {}
        if pooled_embeds is not None and hasattr(pooled_embeds, "to"):
            batch["added_cond_kwargs"]["text_embeds"] = pooled_embeds.to(**target_device_kwargs)
        if time_ids is not None and hasattr(time_ids, "to"):
            batch["added_cond_kwargs"]["time_ids"] = time_ids.to(**target_device_kwargs)

        # Process latents (assumed to be in 'latent_batch')
        latents = batch.get("latent_batch")
        if not hasattr(latents, "to"):
            raise ValueError("Received invalid value for latents.")
        batch["latents"] = latents.to(**target_device_kwargs)

        encoder_attention_mask = batch.get("encoder_attention_mask")
        if encoder_attention_mask is not None and hasattr(encoder_attention_mask, "to"):
            batch["encoder_attention_mask"] = encoder_attention_mask.to(**target_device_kwargs)

        conditioning_image_embeds = batch.get("conditioning_image_embeds")
        if isinstance(conditioning_image_embeds, list) and conditioning_image_embeds:
            first_entry = conditioning_image_embeds[0]
            if isinstance(first_entry, dict):
                stacked: dict[str, torch.Tensor | list] = {}
                for key in first_entry.keys():
                    values = [entry[key] for entry in conditioning_image_embeds]
                    if values and torch.is_tensor(values[0]):
                        stacked[key] = torch.stack(values, dim=0).to(**target_device_kwargs)
                    else:
                        stacked[key] = values
                batch["conditioning_image_embeds"] = stacked
            elif hasattr(first_entry, "to"):
                batch["conditioning_image_embeds"] = torch.stack(conditioning_image_embeds, dim=0).to(**target_device_kwargs)
        elif isinstance(conditioning_image_embeds, dict):
            batch["conditioning_image_embeds"] = {
                key: value.to(**target_device_kwargs) if hasattr(value, "to") else value
                for key, value in conditioning_image_embeds.items()
            }
        elif conditioning_image_embeds is not None and hasattr(conditioning_image_embeds, "to"):
            batch["conditioning_image_embeds"] = conditioning_image_embeds.to(**target_device_kwargs)

        # Sample noise
        noise = torch.randn_like(batch["latents"])
        bsz = batch["latents"].shape[0]
        # If not flow matching, possibly apply an offset to noise
        if not self.config.flow_matching and self.config.offset_noise:
            if self.config.noise_offset_probability == 1.0 or random.random() < self.config.noise_offset_probability:
                noise = noise + self.config.noise_offset * torch.randn(
                    latents.shape[0],
                    latents.shape[1],
                    1,
                    1,
                    device=latents.device,
                )
        batch["noise"] = noise

        if getattr(self.config, "diff2flow_enabled", False):
            batch["flow_target"] = (batch["noise"] - batch["latents"]).to(**target_device_kwargs)

        # Possibly add input perturbation to input noise only
        if self.config.input_perturbation != 0 and (
            not getattr(self.config, "input_perturbation_steps", None)
            or state["global_step"] < self.config.input_perturbation_steps
        ):
            input_perturbation = self.config.input_perturbation
            if getattr(self.config, "input_perturbation_steps", None):
                input_perturbation *= 1.0 - (state["global_step"] / self.config.input_perturbation_steps)
            batch["input_noise"] = noise + input_perturbation * torch.randn_like(batch["latents"])
        else:
            batch["input_noise"] = noise

        if self.PREDICTION_TYPE is PredictionTypes.FLOW_MATCHING:
            batch["sigmas"], batch["timesteps"] = self.sample_flow_sigmas(batch=batch, state=state)
            self.expand_sigmas(batch)
            batch["noisy_latents"] = (1 - batch["sigmas"]) * batch["latents"] + batch["sigmas"] * batch["input_noise"]
            if self._twinflow_active():
                self._prepare_twinflow_metadata(batch)
        else:
            weights = generate_timestep_weights(self.config, self.noise_schedule.config.num_train_timesteps).to(
                self.accelerator.device
            )
            if bsz > 1 and not self.config.disable_segmented_timestep_sampling:
                batch["timesteps"] = segmented_timestep_selection(
                    actual_num_timesteps=self.noise_schedule.config.num_train_timesteps,
                    bsz=bsz,
                    weights=weights,
                    config=self.config,
                    use_refiner_range=False,
                ).to(self.accelerator.device)
            else:
                batch["timesteps"] = torch.multinomial(weights, bsz, replacement=True).long()
            batch["noisy_latents"] = self.noise_schedule.add_noise(
                batch["latents"].float(),
                batch["input_noise"].float(),
                batch["timesteps"],
            ).to(device=self.accelerator.device, dtype=self.config.weight_dtype)
            if self._twinflow_active() and self._twinflow_diffusion_bridge:
                if self.diff2flow_bridge is None:
                    raise ValueError("TwinFlow diff2flow bridge requested but unavailable.")
                sigmas = self.diff2flow_bridge.timesteps_to_sigma(
                    batch["timesteps"].to(device=self.accelerator.device).long(),
                    broadcast_shape=batch["noisy_latents"].shape,
                ).to(dtype=self.config.weight_dtype)
                batch["sigmas"] = sigmas
                self._prepare_twinflow_metadata(batch)

        self._maybe_enable_reflexflow_default()
        if getattr(self.config, "scheduled_sampling_max_step_offset", 0) > 0:
            effective_prob = float(getattr(self.config, "scheduled_sampling_probability", 0.0) or 0.0)
            prob_start = float(getattr(self.config, "scheduled_sampling_prob_start", effective_prob) or effective_prob)
            prob_end = float(getattr(self.config, "scheduled_sampling_prob_end", effective_prob) or effective_prob)
            ramp_steps = int(getattr(self.config, "scheduled_sampling_ramp_steps", 0) or 0)
            ramp_shape = getattr(self.config, "scheduled_sampling_ramp_shape", "linear") or "linear"
            ramp_start = int(getattr(self.config, "scheduled_sampling_start_step", 0) or 0)
            global_step = int(state.get("global_step", 0) or 0)
            if ramp_steps > 0 and global_step >= ramp_start:
                progress = min(1.0, max(0.0, (global_step - ramp_start) / max(ramp_steps, 1)))
                if ramp_shape == "cosine":
                    progress = 0.5 - 0.5 * math.cos(math.pi * progress)
                effective_prob = prob_start + (prob_end - prob_start) * progress
            else:
                effective_prob = prob_start if ramp_steps > 0 else effective_prob
            batch["scheduled_sampling_plan"] = build_rollout_schedule(
                num_train_timesteps=self.noise_schedule.config.num_train_timesteps,
                batch_size=bsz,
                max_step_offset=getattr(self.config, "scheduled_sampling_max_step_offset", 0),
                device=self.accelerator.device,
                base_timesteps=batch["timesteps"],
                strategy=getattr(self.config, "scheduled_sampling_strategy", "uniform"),
                apply_probability=effective_prob,
            )

        batch = self.prepare_batch_conditions(batch=batch, state=state)

        return batch

    @torch.no_grad()
    def encode_text_batch(
        self,
        text_batch: list,
        is_negative_prompt: bool = False,
        prompt_contexts: Optional[List[dict]] = None,
    ):
        """
        Encodes a batch of text using the text encoder.
        """
        if not self.TEXT_ENCODER_CONFIGURATION:
            raise ValueError("No text encoder configuration found.")
        previous_context = getattr(self, "_current_prompt_contexts", None)
        self._current_prompt_contexts = prompt_contexts
        try:
            encoded_text = self._encode_prompts(text_batch, is_negative_prompt)
            return self._format_text_embedding(encoded_text)
        finally:
            self._current_prompt_contexts = previous_context

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        """
        Models can optionally format the stored text embedding, eg. in a dict, or
        filter certain outputs from appearing in the file cache.

        Args:
            text_embedding (torch.Tensor): The embed to adjust.

        Returns:
            torch.Tensor: The adjusted embed. By default, this method does nothing.
        """
        return text_embedding

    @classmethod
    def caption_field_preferences(cls, dataset_type: Optional[str] = None) -> list[str]:
        """
        Preferred caption-related fields (by name) to use when harvesting captions from metadata backends.
        Models can override to request lyrics/tags or other domain-specific fields.
        """
        return []

    def requires_validation_i2v_samples(self) -> bool:
        """
        Override for models that need to pair validation videos with their conditioning images.
        """
        return False

    def should_precompute_validation_negative_prompt(self) -> bool:
        """
        Whether to pre-encode negative prompts during validation setup.
        Override for models that need per-sample negative prompt encoding (e.g., with reference images).
        """
        return True

    @torch.no_grad()
    def encode_validation_negative_prompt(self, negative_prompt: str, positive_prompt_embeds: dict = None):
        """
        Encode the negative prompt for validation.

        Args:
            negative_prompt: The negative prompt text to encode
            positive_prompt_embeds: Optional positive prompt embeddings to use as template for zeros

        Returns:
            Dictionary of encoded negative prompt embeddings
        """
        return self._encode_prompts([negative_prompt], is_negative_prompt=True)

    @torch.no_grad()
    def encode_dropout_caption(self, positive_prompt_embeds: dict = None):
        """
        Encode a null/empty prompt for caption dropout. Models with custom behaviour can override.
        """
        encoded_text = self._encode_prompts([""], is_negative_prompt=False)
        return self._format_text_embedding(encoded_text)

    def conditional_loss(
        self,
        model_pred: torch.Tensor,
        target: torch.Tensor,
        reduction: str = "mean",
        loss_type: str = "l2",
        huber_c: float = 0.1,
    ):
        """
        Compute loss with support for L2, Huber, and Smooth L1.

        Args:
            model_pred: Model predictions
            target: Target values
            reduction: Reduction type ('mean' or 'sum')
            loss_type: Type of loss ('l2', 'huber', 'smooth_l1')
            huber_c: Huber loss parameter
        """
        if loss_type == "l2":
            loss = F.mse_loss(model_pred, target, reduction=reduction)
        elif loss_type == "huber":
            loss = 2 * huber_c * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
            if reduction == "mean":
                loss = torch.mean(loss)
            elif reduction == "sum":
                loss = torch.sum(loss)
        elif loss_type == "smooth_l1":
            loss = 2 * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
            if reduction == "mean":
                loss = torch.mean(loss)
            elif reduction == "sum":
                loss = torch.sum(loss)
        else:
            raise NotImplementedError(f"Unsupported Loss Type {loss_type}")
        return loss

    def compute_scheduled_huber_c(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Compute the scheduled huber_c parameter based on timesteps.

        Args:
            timesteps: Current timesteps in the diffusion process

        Returns:
            Scheduled huber_c values
        """
        if not hasattr(self.config, "loss_type"):
            return torch.tensor(0.1)  # Default value

        if self.config.loss_type not in ["huber", "smooth_l1"]:
            return torch.tensor(0.1)  # Not used for other loss types

        huber_schedule = getattr(self.config, "huber_schedule", "constant")
        base_huber_c = getattr(self.config, "huber_c", 0.1)

        if huber_schedule == "constant":
            return torch.tensor(base_huber_c)

        elif huber_schedule == "exponential":
            # Exponential decay based on timestep
            num_train_timesteps = self.noise_schedule.config.num_train_timesteps
            alpha = -math.log(base_huber_c) / num_train_timesteps

            # Handle batch of timesteps
            # Vectorized computation of huber_c_values using PyTorch
            huber_c_values = torch.exp(-alpha * timesteps)

            return huber_c_values.to(timesteps.device)

        elif huber_schedule == "snr":
            # SNR-based scheduling
            snr = compute_snr(timesteps, self.noise_schedule)
            sigmas = (
                (1.0 - self.noise_schedule.alphas_cumprod[timesteps]) / self.noise_schedule.alphas_cumprod[timesteps]
            ) ** 0.5
            huber_c = (1 - base_huber_c) / (1 + sigmas) ** 2 + base_huber_c
            return huber_c

        else:
            raise NotImplementedError(f"Unknown Huber loss schedule {huber_schedule}")

    def loss(self, prepared_batch: dict, model_output, apply_conditioning_mask: bool = True):
        """
        Computes the loss between the model prediction and the target.
        Optionally applies SNR weighting and a conditioning mask.
        """
        target = self.get_prediction_target(prepared_batch)
        model_pred = model_output["model_prediction"]
        extra_sample_loss = None
        if target is None:
            raise ValueError("Target is None. Cannot compute loss.")

        # Get loss type from config (default to l2 for backward compatibility)
        loss_type = getattr(self.config, "loss_type", "l2")

        use_diff2flow_loss = (
            getattr(self.config, "diff2flow_loss", False)
            and getattr(self.config, "diff2flow_enabled", False)
            and self.diff2flow_bridge is not None
            and self.PREDICTION_TYPE in [PredictionTypes.EPSILON, PredictionTypes.V_PREDICTION]
        )

        if use_diff2flow_loss:
            flow_pred = self.diff2flow_bridge.prediction_to_flow(
                model_pred.float(),
                prepared_batch["noisy_latents"].float(),
                prepared_batch["timesteps"],
                prediction_type=self.PREDICTION_TYPE.value,
            )
            flow_target = self.get_flow_target(prepared_batch)
            if flow_target is None:
                raise ValueError("Flow target is None while diff2flow_loss is enabled.")
            loss = F.mse_loss(flow_pred.float(), flow_target.float(), reduction="none")
        elif self.PREDICTION_TYPE == PredictionTypes.FLOW_MATCHING:
            # Flow matching always uses L2 loss
            loss = (model_pred.float() - target.float()) ** 2
            if getattr(self.config, "scheduled_sampling_reflexflow", False):
                clean_pred = prepared_batch.get("_reflexflow_clean_pred")
                biased_pred = prepared_batch.get("_reflexflow_biased_pred")
                beta2 = getattr(self.config, "scheduled_sampling_reflexflow_beta2", 1.0)
                beta2 = 1.0 if beta2 is None else float(beta2)
                if clean_pred is not None and biased_pred is not None:
                    # Weight toward components that vanish in the rollout (clean > biased).
                    exposure = (clean_pred - biased_pred).detach()
                    norm_dims = tuple(range(1, exposure.dim()))
                    exposure_norm = exposure.abs().sum(dim=norm_dims, keepdim=True).clamp_min(1e-6)
                    alpha = float(getattr(self.config, "scheduled_sampling_reflexflow_alpha", 1.0) or 0.0)
                    if alpha != 0.0:
                        weight = 1.0 + alpha * exposure / exposure_norm
                        loss = loss * weight
                if beta2 != 1.0:
                    loss = loss * beta2

                adr_scale = float(getattr(self.config, "scheduled_sampling_reflexflow_beta1", 10.0) or 0.0)
                if adr_scale != 0.0:
                    biased_latents = prepared_batch.get("noisy_latents")
                    clean_latents = prepared_batch.get("latents")
                    if biased_latents is not None and clean_latents is not None:
                        # Align with the flow-matching vector field (clean -> noise).
                        target_vec = biased_latents - clean_latents
                        flat_target = target_vec.reshape(target_vec.shape[0], -1)
                        flat_pred = model_pred.reshape(model_pred.shape[0], -1)
                        target_norm = torch.norm(flat_target, dim=1, keepdim=True).clamp_min(1e-6)
                        pred_norm = torch.norm(flat_pred, dim=1, keepdim=True).clamp_min(1e-6)
                        target_dir = flat_target / target_norm
                        pred_dir = flat_pred / pred_norm
                        adr = (pred_dir - target_dir).pow(2).sum(dim=1)
                        extra_sample_loss = adr_scale * adr
        elif self.PREDICTION_TYPE in [
            PredictionTypes.EPSILON,
            PredictionTypes.V_PREDICTION,
        ]:
            # Check if we're using Huber or smooth L1 loss
            if loss_type in ["huber", "smooth_l1"]:
                # Get timesteps for the batch
                timesteps = prepared_batch["timesteps"]

                # For scheduled huber, we compute per-sample then average
                if getattr(self.config, "huber_schedule", "constant") != "constant":
                    batch_size = model_pred.shape[0]
                    losses = []

                    for i in range(batch_size):
                        # Get scheduled huber_c for this timestep
                        huber_c = self.compute_scheduled_huber_c(timesteps[i : i + 1]).item()

                        # Compute loss for this sample
                        sample_loss = self.conditional_loss(
                            model_pred[i : i + 1].float(),
                            target[i : i + 1].float(),
                            reduction="none",
                            loss_type=loss_type,
                            huber_c=huber_c,
                        )
                        losses.append(sample_loss)

                    loss = torch.cat(losses, dim=0)
                else:
                    # Constant huber_c - can be computed all at once
                    huber_c = getattr(self.config, "huber_c", 0.1)
                    loss = self.conditional_loss(
                        model_pred.float(),
                        target.float(),
                        reduction="none",
                        loss_type=loss_type,
                        huber_c=huber_c,
                    )

                # Apply SNR weighting if configured (for Huber/smooth L1)
                if self.config.snr_gamma is not None and self.config.snr_gamma > 0:
                    snr = compute_snr(prepared_batch["timesteps"], self.noise_schedule)
                    snr_divisor = snr
                    if self.noise_schedule.config.prediction_type == PredictionTypes.V_PREDICTION.value:
                        snr_divisor = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [
                                snr,
                                self.config.snr_gamma * torch.ones_like(prepared_batch["timesteps"]),
                            ],
                            dim=1,
                        ).min(dim=1)[0]
                        / snr_divisor
                    )
                    mse_loss_weights = mse_loss_weights.view(-1, 1, 1, 1)
                    loss = loss * mse_loss_weights

            else:
                if self.config.snr_gamma is None or self.config.snr_gamma == 0:
                    loss = self.config.snr_weight * F.mse_loss(model_pred.float(), target.float(), reduction="none")
                else:
                    snr = compute_snr(prepared_batch["timesteps"], self.noise_schedule)
                    snr_divisor = snr
                    if self.noise_schedule.config.prediction_type == PredictionTypes.V_PREDICTION.value:
                        snr_divisor = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [
                                snr,
                                self.config.snr_gamma * torch.ones_like(prepared_batch["timesteps"]),
                            ],
                            dim=1,
                        ).min(dim=1)[0]
                        / snr_divisor
                    )
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    mse_loss_weights = mse_loss_weights.view(-1, 1, 1, 1)
                    loss = loss * mse_loss_weights
        else:
            raise NotImplementedError(f"Loss calculation not implemented for prediction type {self.PREDICTION_TYPE}.")

        # Apply conditioning mask if needed
        loss_mask_type = prepared_batch.get("loss_mask_type")
        # Backwards compatibility: fall back to conditioning_type if loss_mask_type not set
        if not loss_mask_type:
            legacy_type = prepared_batch.get("conditioning_type")
            if legacy_type in ("mask", "segmentation"):
                loss_mask_type = legacy_type
        if loss_mask_type == "mask" and apply_conditioning_mask:
            logger.debug("Applying conditioning mask to loss.")
            mask_image = (
                prepared_batch["conditioning_pixel_values"].to(dtype=loss.dtype, device=loss.device)[:, 0].unsqueeze(1)
            )
            mask_image = torch.nn.functional.interpolate(mask_image, size=loss.shape[2:], mode="area")
            mask_image = mask_image / 2 + 0.5
            loss = loss * mask_image
        elif loss_mask_type == "segmentation" and apply_conditioning_mask:
            if random.random() < self.config.masked_loss_probability:
                mask_image = prepared_batch["conditioning_pixel_values"].to(dtype=loss.dtype, device=loss.device)
                mask_image = torch.sum(mask_image, dim=1, keepdim=True) / 3
                mask_image = torch.nn.functional.interpolate(mask_image, size=loss.shape[2:], mode="area")
                mask_image = mask_image / 2 + 0.5
                mask_image = (mask_image > 0).to(dtype=loss.dtype, device=loss.device)
                loss = loss * mask_image

        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        if extra_sample_loss is not None:
            loss = loss + extra_sample_loss.to(device=loss.device, dtype=loss.dtype)
        loss = loss.mean()
        return loss

    def loss_with_logs(self, prepared_batch: dict, model_output, apply_conditioning_mask: bool = True):
        """
        Computes loss and optional per-batch metrics for logging.
        """
        return self.loss(prepared_batch, model_output, apply_conditioning_mask=apply_conditioning_mask), None

    def auxiliary_loss(
        self,
        model_output,
        prepared_batch: dict,
        loss: torch.Tensor,
        *,
        apply_layersync: bool = True,
        clear_hidden_state_buffer: bool = True,
    ):
        """
        Computes auxiliary losses (TwinFlow + optional LayerSync).
        """
        aux_logs = None
        hidden_states_buffer = model_output.get("hidden_states_buffer")

        try:
            if self._twinflow_active():
                try:
                    twin_loss, twin_logs = self._compute_twinflow_losses(
                        prepared_batch=prepared_batch,
                        base_pred=model_output.get("model_prediction"),
                    )
                    loss = loss + twin_loss
                    if twin_logs:
                        aux_logs = twin_logs
                except Exception:
                    logger.exception("TwinFlow auxiliary loss failed; stopping because TwinFlow is explicitly enabled.")
                    raise

            if apply_layersync:
                loss, aux_logs = self._apply_layersync_regularizer(loss, aux_logs, hidden_states_buffer)
        finally:
            if clear_hidden_state_buffer and hidden_states_buffer is not None:
                hidden_states_buffer.clear()

        return loss, aux_logs

    def _compute_twinflow_losses(self, prepared_batch: dict, base_pred: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Calculate TwinFlow/RCGM auxiliary losses using existing batch fields.
        """
        settings = self._twinflow_settings()
        latents = prepared_batch.get("latents")
        noise = prepared_batch.get("noise")
        sigmas = prepared_batch.get("sigmas")
        tt = prepared_batch.get("twinflow_tt")
        noisy_latents = prepared_batch.get("noisy_latents")
        rng_state = prepared_batch.get("twinflow_rng_state") if settings.get("use_rng_state") else None

        if base_pred is None:
            raise ValueError("TwinFlow requires a model_prediction tensor.")
        if latents is None or noise is None or noisy_latents is None:
            raise ValueError("TwinFlow requires latents, noise, sigmas, and noisy_latents in the prepared batch.")

        # For diff2flow bridge mode, derive flow-equivalent sigma from the scheduler's alpha_cumprod.
        # Teacher/forward calls will dynamically convert sigma back to timesteps as x_t evolves.
        use_diff2flow_bridge = False
        if sigmas is None:
            if self._twinflow_diffusion_bridge:
                timesteps = prepared_batch.get("timesteps")
                if timesteps is None:
                    raise ValueError("TwinFlow bridging requires timesteps for sigma computation.")
                if self.diff2flow_bridge is None:
                    raise ValueError("TwinFlow diff2flow bridge requested but unavailable.")
                use_diff2flow_bridge = True
                # Use proper flow-equivalent sigma derived from alpha_cumprod
                sigmas = self.diff2flow_bridge.timesteps_to_sigma(
                    timesteps.to(device=self.accelerator.device).long(),
                    broadcast_shape=noisy_latents.shape,
                ).to(dtype=self.config.weight_dtype)
            else:
                raise ValueError("TwinFlow requires sigmas in the prepared batch.")

        sigmas = sigmas.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        tt = tt if tt is not None else self._twinflow_sample_tt(sigmas)
        sigmas = self._twinflow_match_time_shape(sigmas, noisy_latents)
        tt = self._twinflow_match_time_shape(tt, sigmas)
        eps = torch.finfo(sigmas.dtype).eps if torch.is_floating_point(sigmas) else 1e-4
        tt = torch.minimum(tt, sigmas - eps)
        noisy_latents = noisy_latents.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        # For diffusion-bridged models, RC-GM operates in flow space; build a flow-consistent x_t trajectory.
        rcgm_latents = noisy_latents
        if self._twinflow_diffusion_bridge:
            rcgm_latents = (1 - sigmas) * latents + sigmas * noise
        prepared_batch["twinflow_tt"] = tt

        target = (noise - latents).to(device=self.accelerator.device, dtype=self.config.weight_dtype)

        if self._twinflow_diffusion_bridge:
            if self.diff2flow_bridge is None:
                raise ValueError("TwinFlow diff2flow bridge requested but unavailable.")
            # Convert student's base prediction to flow (prediction only, not latents)
            base_pred = self.diff2flow_bridge.prediction_to_flow(
                base_pred.float(),
                prepared_batch["noisy_latents"].float(),
                prepared_batch["timesteps"],
                prediction_type=self.PREDICTION_TYPE.value,
            )
            # Keep original diffusion noisy_latents - do NOT rewrite with flow interpolation.
            # The UNet was trained on diffusion-style x_t, so we must preserve it.
            # Teacher passes will derive timesteps dynamically from sigma via the bridge.

        with self._twinflow_teacher_context(prepared_batch, settings, rng_state) as teacher_forward:
            target = self._twinflow_enhance_target(
                prepared_batch=prepared_batch,
                teacher_forward=teacher_forward,
                target=target,
                noisy_latents=noisy_latents,
                sigmas=sigmas,
                tt=tt,
                settings=settings,
                use_diff2flow_bridge=use_diff2flow_bridge,
            )
            rcgm_target = self._twinflow_rcgm_target(
                prepared_batch=prepared_batch,
                base_pred=base_pred,
                target=target,
                noisy_latents=noisy_latents,
                sigma=sigmas,
                tt=tt,
                settings=settings,
                teacher_forward=teacher_forward,
                use_diff2flow_bridge=use_diff2flow_bridge,
                rcgm_latents=rcgm_latents,
                latents=latents,
                noise=noise,
            )

        twin_losses: list[torch.Tensor] = []
        log_payload: dict[str, float] = {}

        # 1. RCGM Base Loss (L_base) - always enabled
        # This is the core consistency loss that enables few-step generation
        loss_base = F.mse_loss(base_pred.float(), rcgm_target.float(), reduction="mean")
        twin_losses.append(loss_base)
        log_payload["twinflow_base"] = loss_base.detach().float().mean().item()

        # 2. Real Velocity Loss - helps learn the real velocity field well
        loss_real = F.mse_loss(base_pred.float(), target.float(), reduction="mean")
        twin_losses.append(loss_real)
        log_payload["twinflow_realvel"] = loss_real.detach().float().mean().item()

        # 3. Adversarial Loss (L_adv) + Rectification Loss (L_rectify)
        # These use negative time to train a "fake" trajectory, enabling distribution
        # matching without an external discriminator. The sign embedding infrastructure
        # in the transformers distinguishes fake (negative time) from real (positive time).
        if settings.get("adversarial_enabled", False):
            # Generate fake samples via one-step generation
            x_fake, z_fake = self._twinflow_generate_fake_samples(
                prepared_batch=prepared_batch,
                settings=settings,
            )

            # L_adv: fake velocity loss with negative time
            loss_adv = self._twinflow_compute_adversarial_loss(
                prepared_batch=prepared_batch,
                x_fake=x_fake,
                z=z_fake,
                settings=settings,
            )
            adv_weight = settings.get("adversarial_weight", 1.0)
            twin_losses.append(adv_weight * loss_adv)
            log_payload["twinflow_adv"] = loss_adv.detach().float().mean().item()

            # L_rectify: distribution matching via velocity alignment
            # Use the flat sigmas (not broadcasted) for the rectify loss
            sigmas_flat = sigmas.view(sigmas.shape[0], -1)[:, 0]
            loss_rectify = self._twinflow_compute_rectify_loss(
                prepared_batch=prepared_batch,
                base_pred=base_pred,
                noisy_latents=noisy_latents,
                sigmas=sigmas_flat,
                settings=settings,
            )
            rectify_weight = settings.get("rectify_weight", 1.0)
            twin_losses.append(rectify_weight * loss_rectify)
            log_payload["twinflow_rectify"] = loss_rectify.detach().float().mean().item()

        total_twin_loss = torch.stack(twin_losses).sum()
        return total_twin_loss, log_payload


class ImageModelFoundation(PipelineSupportMixin, VaeLatentScalingMixin, ModelFoundation):
    """
    Implements logic common to image-based diffusion models.
    Handles typical VAE, text encoder loading and a UNet forward pass.
    """

    SUPPORTS_TEXT_ENCODER_TRAINING = False
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    DEFAULT_CONTROLNET_LORA_TARGET = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
        "proj_in",
        "proj_out",
        "conv",
        "conv1",
        "conv2",
        "conv_in",
        "conv_shortcut",
        "linear_1",
        "linear_2",
        "time_emb_proj",
        "controlnet_cond_embedding.conv_in",
        "controlnet_cond_embedding.blocks.0",
        "controlnet_cond_embedding.blocks.1",
        "controlnet_cond_embedding.blocks.2",
        "controlnet_cond_embedding.blocks.3",
        "controlnet_cond_embedding.blocks.4",
        "controlnet_cond_embedding.blocks.5",
        "controlnet_cond_embedding.conv_out",
        "controlnet_down_blocks.0",
        "controlnet_down_blocks.1",
        "controlnet_down_blocks.2",
        "controlnet_down_blocks.3",
        "controlnet_down_blocks.4",
        "controlnet_down_blocks.5",
        "controlnet_down_blocks.6",
        "controlnet_down_blocks.7",
        "controlnet_down_blocks.8",
        "controlnet_mid_block",
    ]
    SHARED_MODULE_PREFIXES = None
    DEFAULT_LYCORIS_TARGET = ["Attention", "FeedForward"]
    DEFAULT_PIPELINE_TYPE = PipelineTypes.TEXT2IMG
    VALIDATION_USES_NEGATIVE_PROMPT = True

    def __init__(self, config: dict, accelerator):
        super().__init__(config, accelerator)
        self.has_vae = True
        self.has_text_encoder = True
        self.vae = None
        self.model = None
        self.controlnet = None
        self.text_encoders = None
        self.tokenizers = None
        self._group_offload_configured = False

    def expand_sigmas(self, batch: dict) -> dict:
        batch["sigmas"] = batch["sigmas"].view(-1, 1, 1, 1)

        return batch

    def custom_model_card_schedule_info(self):
        """
        Override this in your subclass to add model-specific info.

        See SD3 or Flux classes for an example.
        """
        return []

    def custom_model_card_code_example(self, repo_id: str = None) -> str:
        """
        Override this to provide custom code examples for model cards.
        Returns None by default to use the standard template.
        """
        return None


class VideoModelFoundation(VideoTransformMixin, ImageModelFoundation):
    """
    Base class for video models. Provides default 5D handling and optional
    text encoder instantiation. The actual text encoder classes and their
    attributes can be stored in a hardcoded dict if needed. This base class
    does not do it by default.
    """

    def __init__(self, config, accelerator):
        """
        :param config: The training configuration object/dict.
        """
        super().__init__(config, accelerator)
        self.config = config
        self.crepa_regularizer: Optional[CrepaRegularizer] = None

    def expand_sigmas(self, batch):
        if len(batch["latents"].shape) == 5:
            logger.debug(
                f"Latents shape vs sigmas, timesteps: {batch['latents'].shape}, {batch['sigmas'].shape}, {batch['timesteps'].shape}"
            )
            batch["sigmas"] = batch["sigmas"].reshape(batch["latents"].shape[0], 1, 1, 1, 1)

    def post_model_load_setup(self):
        super().post_model_load_setup()
        self._init_crepa_regularizer()

    def _init_crepa_regularizer(self):
        if not getattr(self.config, "crepa_enabled", False):
            self.crepa_regularizer = None
            return

        hidden_size = self._infer_crepa_hidden_size()
        if hidden_size is None:
            raise ValueError("CREPA enabled but unable to infer transformer hidden size.")

        max_train_steps = int(getattr(self.config, "max_train_steps", 0) or 0)
        self.crepa_regularizer = CrepaRegularizer(
            self.config,
            self.accelerator,
            hidden_size,
            model_foundation=self,
            max_train_steps=max_train_steps,
        )
        model_component = self.get_trained_component(unwrap_model=False)
        if model_component is None:
            raise ValueError("CREPA requires an attached diffusion model to register its projector.")
        self.crepa_regularizer.attach_to_model(model_component)

    def _infer_crepa_hidden_size(self) -> Optional[int]:
        model = getattr(self, "model", None)
        if model is None:
            return None
        unwrapped = self.unwrap_model(model=model)
        config = getattr(unwrapped, "config", None)
        if config is None:
            return None
        # Primary: num_attention_heads * attention_head_dim (most DiT models)
        heads = getattr(config, "num_attention_heads", None)
        head_dim = getattr(config, "attention_head_dim", None)
        if heads is not None and head_dim is not None:
            return int(heads * head_dim)
        # Fallback: model_dim (Kandinsky5)
        model_dim = getattr(config, "model_dim", None)
        if model_dim is not None:
            return int(model_dim)
        # Fallback: hidden_size (some models expose this directly)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
        return None

    def auxiliary_loss(self, model_output, prepared_batch: dict, loss: torch.Tensor):
        hidden_states_buffer = model_output.get("hidden_states_buffer")

        # Run base auxiliary losses (TwinFlow) but keep the buffer intact and defer LayerSync.
        loss, aux_logs = super().auxiliary_loss(
            model_output=model_output,
            prepared_batch=prepared_batch,
            loss=loss,
            apply_layersync=False,
            clear_hidden_state_buffer=False,
        )

        crepa = getattr(self, "crepa_regularizer", None)
        if crepa and crepa.enabled:
            crepa_hidden = model_output.get("crepa_hidden_states")
            crepa_frame_features = model_output.get("crepa_frame_features")
            if getattr(crepa, "use_backbone_features", False):
                if hidden_states_buffer is None:
                    raise ValueError("CREPA backbone feature mode requested but no hidden state buffer was provided.")
                if crepa_hidden is None:
                    crepa_hidden = hidden_states_buffer.get(f"layer_{crepa.block_index}")
                teacher_idx = crepa.teacher_block_index if crepa.teacher_block_index is not None else crepa.block_index
                if crepa_frame_features is None and teacher_idx is not None:
                    crepa_frame_features = hidden_states_buffer.get(f"layer_{teacher_idx}")
                if crepa_hidden is None:
                    raise ValueError(f"CREPA requested hidden states from layer {crepa.block_index} but none were stored.")
                if crepa_frame_features is None:
                    raise ValueError(
                        f"CREPA backbone feature mode could not find layer_{teacher_idx} in the hidden state buffer."
                    )

            crepa_loss, crepa_logs = crepa.compute_loss(
                hidden_states=crepa_hidden,
                latents=prepared_batch.get("latents"),
                vae=self.get_vae(),
                frame_features=crepa_frame_features,
                step=StateTracker.get_global_step(),
            )
            if crepa_loss is not None:
                loss = loss + crepa_loss
            if crepa_logs:
                aux_logs = (aux_logs or {}) | crepa_logs

        # Apply LayerSync (if requested) after CREPA so both can share the buffer.
        loss, aux_logs = self._apply_layersync_regularizer(loss, aux_logs, hidden_states_buffer)
        if hidden_states_buffer is not None:
            hidden_states_buffer.clear()

        return loss, aux_logs


class AudioModelFoundation(AudioTransformMixin, ModelFoundation):
    """
    Base class for audio-first models. Provides minimal audio transform helpers
    and ensures autoencoders that return auxiliary metadata (e.g. sample lengths)
    are wrapped in a cache-friendly structure.
    """

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        self.text_encoders = None
        self.tokenizers = None

    def expand_sigmas(self, batch: dict) -> dict:
        """
        Broadcast sampled sigmas to match the latent dimensionality expected by audio
        transformers (batch, channels, height, width).
        """
        sigmas = batch.get("sigmas")
        latents = batch.get("latents")
        if sigmas is None or latents is None:
            return batch
        view_shape = [sigmas.shape[0]] + [1] * (latents.ndim - 1)
        batch["sigmas"] = sigmas.view(*view_shape)
        return batch
