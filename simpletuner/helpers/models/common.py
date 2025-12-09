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
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from diffusers import DiffusionPipeline

import numpy as np
import torch
import torch.nn as nn
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
from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.models.tae import load_tae_decoder
from simpletuner.helpers.scheduled_sampling import build_rollout_schedule
from simpletuner.helpers.training.adapter import load_lora_weights
from simpletuner.helpers.training.custom_schedule import (
    apply_flow_schedule_shift,
    generate_timestep_weights,
    segmented_timestep_selection,
)
from simpletuner.helpers.training.deepspeed import deepspeed_zero_init_disabled_context_manager, prepare_model_for_deepspeed
from simpletuner.helpers.training.lora_format import (
    PEFTLoRAFormat,
    convert_comfyui_to_diffusers,
    convert_diffusers_to_comfyui,
    detect_state_dict_format,
    normalize_lora_format,
)
from simpletuner.helpers.training.min_snr_gamma import compute_snr
from simpletuner.helpers.training.multi_process import _get_rank
from simpletuner.helpers.training.wrappers import unwrap_model
from simpletuner.helpers.utils import ramtorch as ramtorch_utils
from simpletuner.helpers.utils.offloading import enable_group_offload_on_components

logger = logging.getLogger(__name__)
from simpletuner.helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


flow_matching_model_families = ["flux", "sana", "ltxvideo", "wan", "sd3", "chroma", "hunyuanvideo", "longcat_image"]
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
    "wan": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    "hunyuanvideo": "tencent/HunyuanVideo-1.5",
}


def get_model_config_path(model_family: str, model_path: str):
    if model_path is not None and model_path.endswith(".safetensors"):
        if model_family in upstream_config_sources:
            return upstream_config_sources[model_family]
        else:
            raise ValueError(
                "Cannot find noise schedule config for .safetensors file in architecture {}".format(model_family)
            )

    return model_path


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
        self.assistant_adapter_name = "assistant"
        self.assistant_lora_loaded = False
        self._pipeline_component_placeholders: dict[str, nn.Module] = {}

    def pack_text_embeddings_for_cache(self, embeddings):
        """
        Optional hook for models to compress or adjust text embeds before caching.
        Defaults to no-op.
        """
        return embeddings

    def enable_muon_clip_logging(self) -> None:
        """Override in subclasses to wire attention logit publishers when MuonClip is in use."""
        return

    def _get_pipeline_component_placeholder(self, component_name: str) -> nn.Module:
        placeholder = self._pipeline_component_placeholders.get(component_name)
        if placeholder is not None:
            return placeholder

        class _PipelineComponentPlaceholder(nn.Module):
            def __init__(self, name: str):
                super().__init__()
                self._name = name

            def forward(self, *args, **kwargs):  # pragma: no cover - safety net
                raise RuntimeError(f"Pipeline component '{self._name}' is unavailable in this context.")

        placeholder = _PipelineComponentPlaceholder(component_name)
        self._pipeline_component_placeholders[component_name] = placeholder
        return placeholder

    def unpack_text_embeddings_from_cache(self, embeddings):
        """
        Optional hook for models to restore cached text embeds to training format.
        Defaults to no-op.
        """
        return embeddings

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
    # LoRA/PEFT helpers (shared across model families)
    # -------------------------------------------------------------------------
    def get_lora_save_layers(self):
        return None

    def get_lora_target_layers(self):
        lora_type = getattr(self.config, "lora_type", "standard")
        if lora_type.lower() == "standard":
            if getattr(self.config, "slider_lora_target", False):
                slider_target = getattr(self, "SLIDER_LORA_TARGET", None) or getattr(
                    self, "DEFAULT_SLIDER_LORA_TARGET", None
                )
                if slider_target:
                    return slider_target
            if getattr(self.config, "controlnet", False):
                return self.DEFAULT_CONTROLNET_LORA_TARGET
            return self.DEFAULT_LORA_TARGET
        if lora_type.lower() == "lycoris":
            return self.DEFAULT_LYCORIS_TARGET
        raise NotImplementedError(f"Unknown LoRA target type {lora_type}.")

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
                use_dora=getattr(self.config, "use_dora", False),
                **lora_config_kwargs,
            )

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

    def requires_conditioning_dataset(self) -> bool:
        if self.config.controlnet or self.config.control:
            return True
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
        return False

    def requires_conditioning_image_embeds(self) -> bool:
        return False

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
        return False

    def conditioning_validation_dataset_type(self) -> bool:
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
        if lora_format == PEFTLoRAFormat.COMFYUI:
            import safetensors.torch
            from diffusers.loaders.lora_base import LORA_ADAPTER_METADATA_KEY

            def comfyui_save_function(weights, filename):
                metadata = {"format": "pt"}
                if adapter_metadata:
                    try:
                        metadata[LORA_ADAPTER_METADATA_KEY] = json.dumps(adapter_metadata, indent=2, sort_keys=True)
                    except Exception:
                        pass
                converted = convert_diffusers_to_comfyui(weights, adapter_metadata=adapter_metadata)
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
        should_configure_offload = (
            self.group_offload_requested()
            and accelerator_device is not None
            and isinstance(target_device_obj, torch.device)
            and target_device_obj == accelerator_device
        )
        skip_moving_trained_component = should_configure_offload and self.group_offload_configured

        if self.model is not None and not skip_moving_trained_component:
            self.unwrap_model(model=self.model).to(target_device)
        if self.controlnet is not None and not skip_moving_trained_component:
            self.unwrap_model(model=self.controlnet).to(target_device)
        if self.vae is not None and self.vae.device != "meta":
            self.vae.to(target_device)
        if self.text_encoders is not None:
            for text_encoder in self.text_encoders:
                if text_encoder.device != "meta":
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

    def pre_validation_preview_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Pre-process latents before passing to validation preview decoder.

        This is a hook for models that need to transform latents before decode.
        For example, models using video decoders may need to add a frame dimension.

        Args:
            latents: The latents tensor to transform

        Returns:
            The transformed latents tensor
        """
        return latents

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
            tokenizer_kwargs["pretrained_model_name_or_path"] = get_model_config_path(
                self.config.model_family, self.config.pretrained_model_name_or_path
            )
            if text_encoder_config.get("path", None) is not None:
                tokenizer_kwargs["pretrained_model_name_or_path"] = text_encoder_config.get("path")
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
                # load_tes returns a variant and three text encoders
                signature = inspect.signature(text_encoder_config["model"])
                extra_kwargs = {}
                if "torch_dtype" in signature.parameters:
                    extra_kwargs["torch_dtype"] = self.config.weight_dtype
                logger.info(f"Loading {text_encoder_config.get('name')} text encoder")
                text_encoder_path = get_model_config_path(
                    self.config.model_family, self.config.pretrained_model_name_or_path
                )
                if text_encoder_config.get("path", None) is not None:
                    text_encoder_path = text_encoder_config.get("path")
                requires_quant = text_encoder_config.get("required_quantisation_level", None)
                if requires_quant is not None and requires_quant == "int4_weight_only":
                    from torchao.quantization import Int4WeightOnlyConfig
                    from transformers import TorchAoConfig

                    extra_kwargs["device_map"] = "auto"
                    quant_config = Int4WeightOnlyConfig(group_size=128)
                    extra_kwargs["quantization_config"] = TorchAoConfig(quant_type=quant_config)

                text_encoder = text_encoder_config["model"].from_pretrained(
                    text_encoder_path,
                    variant=self.config.variant,
                    revision=self.config.revision,
                    subfolder=text_encoder_config.get("subfolder", "text_encoder") or "",
                    **extra_kwargs,
                )
                if text_encoder.__class__.__name__ in [
                    "UMT5EncoderModel",
                    "T5EncoderModel",
                ]:
                    pass

                if self._ramtorch_text_encoders_requested():
                    self._apply_ramtorch_layers(text_encoder, f"text_encoder_{text_encoder_idx}")

                if move_to_device and text_encoder_precision in ["no_change", None]:
                    text_encoder.to(
                        self.accelerator.device,
                        dtype=self.config.weight_dtype,
                    )
                if hasattr(text_encoder, "eval"):
                    text_encoder.eval()
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
                    # Always move off accelerator; fall back to CPU if meta tensors aren't supported.
                    try:
                        text_encoder.to("meta")
                    except Exception as exc:
                        logger.debug("Text encoder %s could not be moved to meta, moving to CPU instead: %s", idx + 1, exc)
                        text_encoder.to("cpu")
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
        A stub method for child classes to augment pretrained class load arguments with.
        """
        return pretrained_load_args

    def load_model(self, move_to_device: bool = True):
        self._group_offload_configured = False
        pretrained_load_args = {
            "revision": self.config.revision,
            "variant": self.config.variant,
            "torch_dtype": self.config.weight_dtype,
            "use_safetensors": True,
        }
        if "nf4-bnb" == self.config.base_model_precision:
            from diffusers import BitsAndBytesConfig

            pretrained_load_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self.config.weight_dtype,
            )
        loader_fn = self.MODEL_CLASS.from_pretrained
        model_path = (
            self.config.pretrained_transformer_model_name_or_path
            if self.MODEL_TYPE is ModelTypes.TRANSFORMER
            else self.config.pretrained_unet_model_name_or_path
        ) or self.config.pretrained_model_name_or_path
        if self.config.pretrained_model_name_or_path.endswith(".safetensors"):
            self.config.pretrained_model_name_or_path = get_model_config_path(self.config.model_family, model_path)
        if model_path.endswith(".safetensors"):
            loader_fn = self.MODEL_CLASS.from_single_file
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

        logger.info(f"Loading diffusion model from {model_path}")
        self.model = loader_fn(
            model_path,
            subfolder=model_subfolder,
            **pretrained_load_args,
        )
        if self._ramtorch_enabled() and self.model is not None:
            self._apply_ramtorch_layers(self.model, self.MODEL_TYPE.value)
        if move_to_device and self.model is not None:
            self.model.to(self.accelerator.device, dtype=self.config.weight_dtype)

        self.configure_chunked_feed_forward()

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
        self.fuse_qkv_projections()
        self.post_model_load_setup()

    def post_model_load_setup(self):
        """
        Post model load setup.

        This is a stub and can be optionally implemented in subclasses for eg. updating configuration settings
        based on the loaded model weights. SDXL uses this to update the user config to reflect refiner training.

        """
        pass

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
        if load_base_model:
            pipeline_kwargs[self.MODEL_TYPE.value] = self.unwrap_model(model=self.model)
        else:
            pipeline_kwargs[self.MODEL_TYPE.value] = self._get_pipeline_component_placeholder(self.MODEL_TYPE.value)

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

    def _apply_ramtorch_layers(self, module, component_label: str, *, target_patterns: Optional[list[str]] = None) -> int:
        if module is None or not self._ramtorch_enabled():
            return 0

        try:
            replaced = ramtorch_utils.replace_linear_layers_with_ramtorch(
                module,
                device=self._ramtorch_device(),
                target_patterns=self._ramtorch_targets_for_component(target_patterns),
                name_prefix=component_label,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to apply RamTorch to {component_label}: {exc}") from exc

        if replaced:
            logger.info("Applied RamTorch to %s Linear layers on %s.", replaced, component_label)
        else:
            logger.debug("RamTorch enabled for %s but no Linear layers matched the configured targets.", component_label)

        return replaced

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

    def sample_flow_sigmas(self, batch: dict, state: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample flow-matching sigmas/timesteps for the current batch.

        Subclasses can override to implement model-specific sampling strategies.
        """
        bsz = batch["latents"].shape[0]
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
        conditioning_type = prepared_batch.get("conditioning_type")
        if conditioning_type == "mask" and apply_conditioning_mask:
            logger.debug("Applying conditioning mask to loss.")
            mask_image = (
                prepared_batch["conditioning_pixel_values"].to(dtype=loss.dtype, device=loss.device)[:, 0].unsqueeze(1)
            )
            mask_image = torch.nn.functional.interpolate(mask_image, size=loss.shape[2:], mode="area")
            mask_image = mask_image / 2 + 0.5
            loss = loss * mask_image
        elif conditioning_type == "segmentation" and apply_conditioning_mask:
            if random.random() < self.config.masked_loss_probability:
                mask_image = prepared_batch["conditioning_pixel_values"].to(dtype=loss.dtype, device=loss.device)
                mask_image = torch.sum(mask_image, dim=1, keepdim=True) / 3
                mask_image = torch.nn.functional.interpolate(mask_image, size=loss.shape[2:], mode="area")
                mask_image = mask_image / 2 + 0.5
                mask_image = (mask_image > 0).to(dtype=loss.dtype, device=loss.device)
                loss = loss * mask_image

        loss = loss.mean(dim=list(range(1, len(loss.shape)))).mean()
        return loss

    def auxiliary_loss(self, model_output, prepared_batch: dict, loss: torch.Tensor):
        """
        Computes an auxiliary loss if needed.
        This is a stub and can be optionally implemented in subclasses.
        """
        return loss, None


class ImageModelFoundation(ModelFoundation):
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

    def scale_vae_latents_for_cache(self, latents, vae):
        if vae is None or not hasattr(vae, "config") or latents is None:
            return latents
        shift_factor = getattr(vae.config, "shift_factor", None)
        scaling_factor = getattr(self, "AUTOENCODER_SCALING_FACTOR", getattr(vae.config, "scaling_factor", 1.0))
        if shift_factor is not None:
            return (latents - shift_factor) * scaling_factor
        if isinstance(latents, torch.Tensor) and hasattr(vae.config, "scaling_factor"):
            scaled = latents * scaling_factor
            logger.debug("Latents shape after scaling: %s", scaled.shape)
            return scaled
        return latents

    @classmethod
    def _iter_pipeline_classes(cls):
        pipelines = getattr(cls, "PIPELINE_CLASSES", {})
        if not isinstance(pipelines, dict):
            return []
        pipeline_classes = []
        for pipeline_cls in pipelines.values():
            if inspect.isclass(pipeline_cls):
                pipeline_classes.append(pipeline_cls)
        return pipeline_classes

    @staticmethod
    def _pipeline_has_lora_loader(pipeline_cls) -> bool:
        if not inspect.isclass(pipeline_cls):
            return False
        for base in inspect.getmro(pipeline_cls):
            if base.__name__.endswith("LoraLoaderMixin"):
                return True
        return False

    @classmethod
    def supports_lora(cls) -> bool:
        if cls.SUPPORTS_LORA is not None:
            return bool(cls.SUPPORTS_LORA)

        for pipeline_cls in cls._iter_pipeline_classes():
            if cls._pipeline_has_lora_loader(pipeline_cls):
                return True
        return False

    @classmethod
    def supports_controlnet(cls) -> bool:
        if cls.SUPPORTS_CONTROLNET is not None:
            return bool(cls.SUPPORTS_CONTROLNET)

        pipelines = getattr(cls, "PIPELINE_CLASSES", {})
        if not isinstance(pipelines, dict):
            return False

        for pipeline_type, pipeline_cls in pipelines.items():
            if pipeline_cls is None:
                continue
            if isinstance(pipeline_type, PipelineTypes):
                if pipeline_type in (PipelineTypes.CONTROLNET, PipelineTypes.CONTROL):
                    return True
            elif isinstance(pipeline_type, str) and pipeline_type.lower() in {"controlnet", "control"}:
                return True
        return False

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


class VideoToTensor:
    def __call__(self, video):
        """
        Converts a video (numpy array of shape (num_frames, height, width, channels))
        to a tensor of shape (num_frames, channels, height, width) by applying the
        standard ToTensor conversion to each frame.
        """
        if isinstance(video, np.ndarray):
            frames = []
            for frame in video:
                # Convert frame to PIL Image if not already.
                if not isinstance(frame, Image.Image):
                    frame = Image.fromarray(frame)
                frame_tensor = transforms.functional.to_tensor(frame)
                frames.append(frame_tensor)
            return torch.stack(frames)
        elif isinstance(video, list):
            frames = []
            for frame in video:
                if not isinstance(frame, Image.Image):
                    frame = Image.fromarray(frame)
                frames.append(transforms.functional.to_tensor(frame))
            return torch.stack(frames)
        else:
            raise TypeError("Input video must be a numpy array or a list of frames.")

    def __repr__(self):
        return self.__class__.__name__ + "()"


class VideoModelFoundation(ImageModelFoundation):
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

    def get_transforms(self, dataset_type: str = "image"):
        return transforms.Compose(
            [
                VideoToTensor() if dataset_type == "video" else transforms.ToTensor(),
            ]
        )

    def expand_sigmas(self, batch):
        if len(batch["latents"].shape) == 5:
            logger.debug(
                f"Latents shape vs sigmas, timesteps: {batch['latents'].shape}, {batch['sigmas'].shape}, {batch['timesteps'].shape}"
            )
            batch["sigmas"] = batch["sigmas"].reshape(batch["latents"].shape[0], 1, 1, 1, 1)

    def apply_i2v_augmentation(self, batch):
        pass

    def prepare_5d_inputs(self, tensor):
        """
        Example method to handle default 5D shape. The typical shape might be:
        (batch_size, frames, channels, height, width).

        You can reshape or permute as needed for the underlying model.
        """
        return tensor


class AudioModelFoundation(ModelFoundation):
    """
    Base class for audio-first models. Provides minimal audio transform helpers
    and ensures autoencoders that return auxiliary metadata (e.g. sample lengths)
    are wrapped in a cache-friendly structure.
    """

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)

    def get_transforms(self, dataset_type: str = "image"):
        if dataset_type == DatasetType.AUDIO.value or dataset_type == "audio":

            def _audio_transform(sample):
                waveform = sample
                if isinstance(sample, dict):
                    waveform = sample.get("waveform")
                if waveform is None:
                    raise ValueError("Audio transform expected a waveform tensor in the sample payload.")
                if isinstance(waveform, np.ndarray):
                    waveform = torch.from_numpy(waveform)
                if not torch.is_tensor(waveform):
                    raise ValueError(f"Unsupported audio payload type: {type(waveform)}")
                waveform = waveform.detach().clone()
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0)
                return waveform

            return _audio_transform
        return super().get_transforms(dataset_type=dataset_type)

    @torch.no_grad()
    def encode_with_vae(self, vae, samples):
        """
        Music-focused autoencoders often return both latents and accompanying
        sequence lengths. Wrap both into a dict so downstream caches retain the metadata.
        """
        if samples is None:
            raise ValueError("Audio VAE received no samples to encode.")
        audio = samples
        if not torch.is_tensor(audio):
            raise ValueError(f"Audio encoder expected a Tensor input, received {type(audio)}.")
        if audio.ndim == 2:
            audio = audio.unsqueeze(0)
        audio = audio.to(device=self.accelerator.device, dtype=torch.float32)
        result = vae.encode(audio)
        latent_lengths = None
        latents = result
        if isinstance(result, tuple):
            latents, *extras = result
            latent_lengths = extras[0] if extras else None
        elif isinstance(result, dict):
            latents = result.get("latents")
            latent_lengths = result.get("latent_lengths")
        payload = {"latents": latents}
        if latent_lengths is not None:
            payload["latent_lengths"] = latent_lengths
        return payload

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
