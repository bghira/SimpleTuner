# Copyright 2026 The MiniMax Team and The HuggingFace Team.
# Modifications for SimpleTuner are distributed under the AGPL-3.0-or-later.

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.guiders import ClassifierFreeGuidance
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HFValidationError, LocalEntryNotFoundError, RepositoryNotFoundError
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, Qwen3ForCausalLM

from simpletuner.helpers.acceleration import (
    AccelerationBackend,
    AccelerationPreset,
    get_bitsandbytes_presets,
    get_deepspeed_presets,
    get_quanto_presets,
    get_sdnq_presets,
    get_torchao_presets,
)
from simpletuner.helpers.configuration.registry import ConfigRegistry, ConfigRule, RuleType, ValidationResult
from simpletuner.helpers.models.common import (
    AudioModelFoundation,
    ModelTypes,
    PipelineTypes,
    PredictionTypes,
    TextEmbedCacheKey,
)
from simpletuner.helpers.models.minimaxmusic.condition_encoder import MiniMaxMusic3ConditionEncoder
from simpletuner.helpers.models.minimaxmusic.encoders import (
    _AR_CFG_SCALE,
    _AR_CFG_TOP_K,
    _AUDIO_CFG_TOKEN_ID,
    _AUDIO_CODE_OFFSET,
    _AUDIO_END_TOKEN_ID,
    _MAX_AUDIO_FRAMES,
    _MAX_PROMPT_TOKENS,
    _SEMANTIC_VOCAB_SIZE,
    _clean_caption,
    _embed_audio_frame,
    _generate_depth_codes,
    _normalize_lyrics,
    _sample_top_k,
)
from simpletuner.helpers.models.minimaxmusic.modular_blocks import MiniMaxMusic3Blocks
from simpletuner.helpers.models.minimaxmusic.modular_pipeline import MiniMaxMusic3ModularPipeline
from simpletuner.helpers.models.minimaxmusic.rvq_depth_decoder import MiniMaxMusic3RVQDepthDecoder
from simpletuner.helpers.models.minimaxmusic.transformer import MiniMaxMusic3Transformer1DModel
from simpletuner.helpers.models.minimaxmusic.vocoder import MiniMaxMusic3DAV, MiniMaxMusic3Vocoder
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.musubi_block_swap import apply_musubi_pretrained_defaults
from simpletuner.helpers.training.lora_format import (
    PEFTLoRAFormat,
    collect_lora_ranks,
    detect_state_dict_format,
    normalize_lora_format,
)

logger = logging.getLogger(__name__)


class MiniMaxMusic(AudioModelFoundation):
    NAME = "MiniMax Music 3"
    MODEL_DESCRIPTION = "Lyrics- and caption-conditioned music flow transformer"
    ENABLED_IN_WIZARD = True
    MODEL_TYPE = ModelTypes.TRANSFORMER
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_CLASS = MiniMaxMusic3Transformer1DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2AUDIO: MiniMaxMusic3ModularPipeline,
        PipelineTypes.TEXT2IMG: MiniMaxMusic3ModularPipeline,
    }
    DEFAULT_PIPELINE_TYPE = PipelineTypes.TEXT2AUDIO
    AUTOENCODER_CLASS = MiniMaxMusic3DAV
    LATENT_CHANNEL_COUNT = 128
    DEFAULT_NOISE_SCHEDULER = "flow_matching"
    DEFAULT_MODEL_FLAVOUR = "music3"
    HUGGINGFACE_PATHS = {
        "music3": "MiniMaxAI/MiniMax-Music3",
    }
    TEXT_ENCODER_CONFIGURATION = {
        "language_model": {
            "name": "Qwen3 AR language model",
            "tokenizer": AutoTokenizer,
            "model": Qwen3ForCausalLM,
            "subfolder": "language_model",
            "tokenizer_subfolder": "tokenizer",
        }
    }
    MODEL_LICENSE = "apache-2.0"
    SUPPORTS_LYRICS_EMBEDDER_TRAINING = False
    VALIDATION_USES_NEGATIVE_PROMPT = False
    AUTO_LORA_FORMAT_DETECTION = True
    DEFAULT_LORA_TARGET = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "ff_in",
        "ff_out",
        "proj_in",
        "proj_out",
    ]
    DEFAULT_LYCORIS_TARGET = ["MiniMaxMusic3Attention", "MiniMaxMusic3TransformerBlock"]

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        self.condition_encoder: Optional[MiniMaxMusic3ConditionEncoder] = None
        self.rvq_depth_decoder: Optional[MiniMaxMusic3RVQDepthDecoder] = None
        self.language_model: Optional[Qwen3ForCausalLM] = None
        self.guider: Optional[ClassifierFreeGuidance] = None
        self.vae = None

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        return 35

    @classmethod
    def get_acceleration_presets(cls) -> list[AccelerationPreset]:
        base_memory_config = {
            "base_model_precision": "no_change",
            "gradient_checkpointing": True,
        }
        return [
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="basic",
                name="RamTorch - Basic",
                description="Offloads half of transformer layers to CPU RAM.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by roughly 30%.",
                tradeoff_speed="Increases training time.",
                tradeoff_notes="Requires enough system RAM for streamed transformer weights.",
                requires_min_system_ram_gb=32,
                config={
                    **base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": ",".join(f"transformer_blocks.{idx}.*" for idx in range(18, 36)),
                },
            ),
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="balanced",
                name="RamTorch - Balanced",
                description="Offloads most transformer layers, keeping the first quarter on GPU.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by roughly 45%.",
                tradeoff_speed="Increases training time.",
                tradeoff_notes="Requires enough system RAM for streamed transformer weights.",
                requires_min_system_ram_gb=48,
                config={
                    **base_memory_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": ",".join(f"transformer_blocks.{idx}.*" for idx in range(9, 36)),
                },
            ),
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="aggressive",
                name="RamTorch - Aggressive",
                description="Offloads all transformer layers to CPU RAM.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by roughly 60%.",
                tradeoff_speed="Increases training time substantially.",
                tradeoff_notes="Requires enough system RAM for all streamed transformer weights.",
                requires_min_system_ram_gb=64,
                config={**base_memory_config, "ramtorch": True, "ramtorch_target_modules": "transformer_blocks.*"},
            ),
            *get_deepspeed_presets(base_memory_config),
            *get_sdnq_presets(base_memory_config),
            *get_torchao_presets(base_memory_config),
            *get_quanto_presets(base_memory_config),
            *get_bitsandbytes_presets(base_memory_config),
        ]

    @classmethod
    def caption_field_preferences(cls, dataset_type: Optional[str] = None) -> list[str]:
        if dataset_type and str(dataset_type).lower() == "audio":
            return ["prompt", "lyrics", "tags"]
        return []

    @staticmethod
    def _prompt_context_from_audio_metadata(sample_metadata: dict, prompt: str | None = None) -> dict:
        metadata = {}
        if prompt is not None:
            metadata["prompt"] = str(prompt)
        if not isinstance(sample_metadata, dict):
            return metadata
        for key in (
            "lyrics",
            "audio_duration",
            "duration",
            "duration_seconds",
            "bucket_duration_seconds",
            "truncated_duration_seconds",
            "original_duration_seconds",
        ):
            value = sample_metadata.get(key)
            if value not in (None, ""):
                metadata[key] = value
        return metadata

    @classmethod
    def register_config_requirements(cls):
        rules = [
            ConfigRule(
                field_name="dataset_type",
                rule_type=RuleType.CUSTOM,
                value=None,
                message="MiniMax Music 3 expects audio datasets; VAECache encodes raw audio through the DAV autoencoder.",
                error_level="warning",
            ),
        ]
        ConfigRegistry.register_rules("minimaxmusic", rules)
        ConfigRegistry.register_validator(
            "minimaxmusic",
            cls._validate_audio_dataset_usage,
            "Validates MiniMax Music 3 audio dataset requirements.",
        )

    @staticmethod
    def _validate_audio_dataset_usage(config: dict) -> List[ValidationResult]:
        dataset_type = (config or {}).get("dataset_type")
        if dataset_type and str(dataset_type).lower() != "audio":
            return [
                ValidationResult(
                    passed=False,
                    field="dataset_type",
                    message="MiniMax Music 3 requires audio datasets for training.",
                    level="warning",
                    suggestion="Set dataset_type: audio so VAECache can encode raw audio into MiniMax Music 3 DAV latents.",
                )
            ]
        return []

    def supports_crepa_self_flow(self) -> bool:
        return True

    @classmethod
    def supports_audio_only_training(cls) -> bool:
        return True

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

    def text_embed_cache_key_value(self, *, prompt: str, default_key: str, metadata: dict) -> str:
        del metadata
        if prompt == "":
            return f"{default_key}:__caption_dropout__"
        return default_key

    def text_embed_cache_metadata_for_sample(
        self,
        *,
        example: dict,
        latent: Optional[torch.Tensor],
        prompt: str,
        data_backend_id: Optional[str],
        dataset_relative_path: Optional[str],
    ) -> dict:
        del latent, data_backend_id, dataset_relative_path
        sample_metadata = {}
        if isinstance(example, dict):
            embedded_metadata = example.get("image_metadata")
            if isinstance(embedded_metadata, dict):
                sample_metadata.update(embedded_metadata)
            sample_metadata.update(example)
        else:
            embedded_metadata = getattr(example, "image_metadata", None)
            if isinstance(embedded_metadata, dict):
                sample_metadata.update(embedded_metadata)

        return self._prompt_context_from_audio_metadata(sample_metadata, str(prompt))

    def text_embed_cache_metadata_for_filepath(
        self,
        *,
        init_backend: dict,
        image_path: str,
        prompt: str,
        data_backend_id: str | None,
        dataset_relative_path: str | None,
    ) -> dict:
        del data_backend_id, dataset_relative_path
        metadata_backend = init_backend.get("metadata_backend") if isinstance(init_backend, dict) else None
        if metadata_backend is None:
            return {"prompt": str(prompt)}
        sample_metadata = metadata_backend.get_metadata_by_filepath(image_path) or {}
        return self._prompt_context_from_audio_metadata(sample_metadata, str(prompt))

    def flow_matching_target_direction(self) -> float:
        return -1.0

    def setup_training_noise_schedule(self):
        self.noise_schedule = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1,
            shift=float(getattr(self.config, "flow_schedule_shift", 1.0) or 1.0),
            invert_sigmas=True,
        )
        return self.config, self.noise_schedule

    def pretrained_load_args(self, pretrained_load_args: dict) -> dict:
        pretrained_load_args = super().pretrained_load_args(pretrained_load_args)
        if self._configured_anyflow():
            anyflow_config = self._anyflow_distillation_config()
            pretrained_load_args.setdefault("deltatime_type", anyflow_config.get("deltatime_type", "r"))
            pretrained_load_args.setdefault("gate_value", float(anyflow_config.get("gate_value", 0.25)))
        return apply_musubi_pretrained_defaults(self.config, pretrained_load_args)

    def post_model_load_setup(self):
        super().post_model_load_setup()
        if self._configured_anyflow():
            transformer = self.unwrap_model(self.model)
            anyflow_config = self._anyflow_distillation_config()
            if getattr(transformer, "flowmap_deltatime_type", None) is None:
                transformer.enable_flowmap_time_conditioning(
                    gate_value=float(anyflow_config.get("gate_value", 0.25)),
                    deltatime_type=anyflow_config.get("deltatime_type", "r"),
                )

    def _configured_anyflow(self) -> bool:
        return str(getattr(self.config, "distillation_method", "") or "").strip().lower() == "anyflow"

    def _anyflow_distillation_config(self) -> dict:
        configured = getattr(self.config, "distillation_config", None)
        configured = configured if isinstance(configured, dict) else {}
        anyflow_config = configured.get("anyflow", configured)
        return anyflow_config if isinstance(anyflow_config, dict) else {}

    def _music_lora_component_name(self) -> str:
        return "transformer"

    def _music_transformer_uses_gate_first_swiglu(self) -> bool:
        transformer = self.unwrap_model(self.model) if getattr(self, "model", None) is not None else None
        return bool(getattr(getattr(transformer, "config", None), "swiglu_gate_first", False))

    def _convert_lora_state_dict_to_comfyui(
        self,
        weights: dict,
        *,
        adapter_metadata: Optional[dict] = None,
        component_adapter_metadata: Optional[dict] = None,
    ) -> dict:
        del component_adapter_metadata
        from simpletuner.helpers.models.minimaxmusic.modular_pipeline import _convert_minimax_music_diffusers_lora_to_comfyui

        return _convert_minimax_music_diffusers_lora_to_comfyui(
            weights,
            adapter_metadata=adapter_metadata,
            source_swiglu_gate_first=self._music_transformer_uses_gate_first_swiglu(),
        )

    def _convert_lora_state_dict_from_comfyui(
        self,
        weights: dict,
        *,
        target_prefix: str,
    ) -> tuple[dict, dict]:
        from simpletuner.helpers.models.minimaxmusic.modular_pipeline import _convert_minimax_music_comfy_lora_to_diffusers

        return _convert_minimax_music_comfy_lora_to_diffusers(
            weights,
            target_prefix=target_prefix,
            target_swiglu_gate_first=self._music_transformer_uses_gate_first_swiglu(),
        )

    def _prepare_plain_music_lora_swiglu_layout(self, state_dict: dict, metadata: Optional[dict]) -> dict:
        from simpletuner.helpers.models.minimaxmusic.modular_pipeline import (
            _convert_minimax_music_swiglu_lora_layout,
            _minimax_music_swiglu_gate_first_from_metadata,
        )

        source_gate_first = _minimax_music_swiglu_gate_first_from_metadata(
            metadata,
            target_prefix=self._music_lora_component_name(),
        )
        if source_gate_first is None:
            return state_dict
        return _convert_minimax_music_swiglu_lora_layout(
            state_dict,
            source_gate_first=source_gate_first,
            target_gate_first=self._music_transformer_uses_gate_first_swiglu(),
        )

    def _lora_state_dict_load_kwargs(self) -> dict:
        return {"return_lora_metadata": True}

    def _prepare_loaded_lora_state_dict(self, state_dict: dict, metadata: Optional[dict] = None) -> dict:
        from simpletuner.helpers.models.minimaxmusic.modular_pipeline import (
            _convert_minimax_music_comfy_lora_to_diffusers,
            _is_minimax_music_native_lora_state_dict,
        )

        if _is_minimax_music_native_lora_state_dict(state_dict):
            lora_format = normalize_lora_format(getattr(self.config, "lora_format", None))
            if lora_format == PEFTLoRAFormat.COMFYUI or detect_state_dict_format(state_dict) == PEFTLoRAFormat.COMFYUI:
                return state_dict
            converted, _network_alphas = _convert_minimax_music_comfy_lora_to_diffusers(
                state_dict,
                target_prefix=self._music_lora_component_name(),
                target_swiglu_gate_first=self._music_transformer_uses_gate_first_swiglu(),
            )
            return converted
        return self._prepare_plain_music_lora_swiglu_layout(state_dict, metadata)

    def _prepare_init_lora_state_dict(self, state_dict: dict, metadata: Optional[dict] = None) -> dict:
        from simpletuner.helpers.models.minimaxmusic.modular_pipeline import (
            _convert_minimax_music_comfy_lora_to_diffusers,
            _is_minimax_music_native_lora_state_dict,
        )

        lora_format = normalize_lora_format(getattr(self.config, "lora_format", None))
        detected_format = detect_state_dict_format(state_dict)
        if lora_format == PEFTLoRAFormat.DIFFUSERS and (
            detected_format == PEFTLoRAFormat.COMFYUI or _is_minimax_music_native_lora_state_dict(state_dict)
        ):
            lora_format = PEFTLoRAFormat.COMFYUI
        if lora_format != PEFTLoRAFormat.COMFYUI:
            return self._prepare_plain_music_lora_swiglu_layout(state_dict, metadata)
        converted, network_alphas = _convert_minimax_music_comfy_lora_to_diffusers(
            state_dict,
            target_prefix=self._music_lora_component_name(),
            target_swiglu_gate_first=self._music_transformer_uses_gate_first_swiglu(),
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
        from simpletuner.helpers.models.minimaxmusic.modular_pipeline import (
            MINIMAX_MUSIC_FLOWMAP_DELTATIME_METADATA_KEY,
            MINIMAX_MUSIC_FLOWMAP_GATE_METADATA_KEY,
            MINIMAX_MUSIC_SWIGLU_GATE_FIRST_METADATA_KEY,
        )

        metadata_key = f"{self.MODEL_SUBFOLDER}_lora_adapter_metadata"
        adapter_metadata = dict(kwargs.get(metadata_key) or {})
        lora_format = normalize_lora_format(getattr(self.config, "lora_format", None))
        adapter_metadata[MINIMAX_MUSIC_SWIGLU_GATE_FIRST_METADATA_KEY] = (
            False if lora_format == PEFTLoRAFormat.COMFYUI else self._music_transformer_uses_gate_first_swiglu()
        )
        transformer = self.unwrap_model(self.model) if getattr(self, "model", None) is not None else None
        deltatime_type = getattr(transformer, "flowmap_deltatime_type", None)
        if deltatime_type is not None:
            gate = getattr(transformer, "flowmap_delta_emb_gate", None)
            if torch.is_tensor(gate):
                adapter_metadata[MINIMAX_MUSIC_FLOWMAP_GATE_METADATA_KEY] = float(gate.detach().float().cpu().item())
            adapter_metadata[MINIMAX_MUSIC_FLOWMAP_DELTATIME_METADATA_KEY] = str(deltatime_type)
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
                    prefix_to_strip=f"{self._music_lora_component_name()}.",
                )
                if ranks:
                    return sorted(ranks)
        if not self._configured_anyflow() or str(getattr(self.config, "lora_type", "standard")).lower() != "standard":
            return super().get_lora_target_layers()

        targets = [
            "to_q",
            "to_k",
            "to_v",
            "to_out.0",
            "ff_in",
            "ff_out",
            "proj_in",
            "proj_out",
        ]
        anyflow_config = self._anyflow_distillation_config()
        if bool(anyflow_config.get("train_time_embedder", True)):
            targets.extend(["time_embed.linear_1", "time_embed.linear_2"])
        if bool(anyflow_config.get("train_delta_embedder", True)):
            targets.extend(["delta_time_embed.linear_1", "delta_time_embed.linear_2"])
        return targets

    def _assert_anyflow_endpoint_parameters_trainable(self) -> None:
        if not self._configured_anyflow():
            return
        if not bool(self._anyflow_distillation_config().get("train_delta_embedder", True)):
            return
        transformer = self.unwrap_model(self.model)
        trainable = [
            name
            for name, parameter in transformer.named_parameters()
            if parameter.requires_grad and "delta_time_embed" in name
        ]
        if not trainable:
            raise RuntimeError(
                "MiniMax Music 3 AnyFlow requested train_delta_embedder=true, but the PEFT adapter has no trainable "
                "delta timestep parameters."
            )
        logger.info("MiniMax Music 3 AnyFlow endpoint conditioning is trainable through: %s", ", ".join(trainable))

    def add_lora_adapter(self):
        result = super().add_lora_adapter()
        self._assert_anyflow_endpoint_parameters_trainable()
        return result

    def validation_audio_sample_rate(self) -> Optional[int]:
        vocoder = self.vae
        if vocoder is not None and getattr(vocoder, "config", None) is not None:
            return int(getattr(vocoder.config, "sampling_rate", 44100))
        return 44100

    def _checkpoint_path(self) -> str:
        return self.config.pretrained_model_name_or_path or self.HUGGINGFACE_PATHS[self.DEFAULT_MODEL_FLAVOUR]

    def _vae_checkpoint_path(self) -> str:
        return self.config.pretrained_vae_model_name_or_path or self._checkpoint_path()

    def _has_diffusers_audio_vae(self, checkpoint_path: str) -> bool:
        if os.path.isfile(checkpoint_path):
            return False
        if os.path.isdir(checkpoint_path):
            checkpoint = Path(checkpoint_path)
            if (checkpoint / "config.json").is_file():
                return True
            return (checkpoint / "audio_vae" / "config.json").is_file()
        try:
            hf_hub_download(
                checkpoint_path,
                "audio_vae/config.json",
                revision=getattr(self.config, "revision", None),
                repo_type="model",
            )
            return True
        except (EntryNotFoundError, LocalEntryNotFoundError, RepositoryNotFoundError, HFValidationError):
            return False

    def _load_diffusers_audio_vae(self, checkpoint_path: str) -> MiniMaxMusic3DAV:
        if os.path.isdir(checkpoint_path) and os.path.isfile(os.path.join(checkpoint_path, "config.json")):
            return MiniMaxMusic3DAV.from_pretrained(checkpoint_path, torch_dtype=torch.float32)
        return MiniMaxMusic3DAV.from_pretrained(checkpoint_path, subfolder="audio_vae", torch_dtype=torch.float32)

    def _resolve_dav_checkpoint(self) -> Optional[str]:
        checkpoint_path = self._vae_checkpoint_path()
        if os.path.isfile(checkpoint_path):
            return checkpoint_path
        if os.path.isdir(checkpoint_path):
            local_dav_path = os.path.join(checkpoint_path, "dav.pth")
            return local_dav_path if os.path.isfile(local_dav_path) else None
        try:
            return hf_hub_download(
                checkpoint_path,
                "dav.pth",
                revision=getattr(self.config, "revision", None),
                repo_type="model",
            )
        except (EntryNotFoundError, LocalEntryNotFoundError, RepositoryNotFoundError, HFValidationError):
            return None

    def load_vae(self, move_to_device: bool = True):
        if self.vae is None:
            checkpoint_path = self._vae_checkpoint_path()
            if self._has_diffusers_audio_vae(checkpoint_path):
                self.vae = self._load_diffusers_audio_vae(checkpoint_path)
            elif (dav_checkpoint := self._resolve_dav_checkpoint()) is not None:
                self.vae = MiniMaxMusic3DAV.from_original_dav(dav_checkpoint)
            else:
                self.vae = MiniMaxMusic3Vocoder.from_pretrained(
                    checkpoint_path,
                    subfolder="vocoder",
                    torch_dtype=self.config.weight_dtype,
                )
            self.vae.requires_grad_(False)
        if move_to_device and self.vae is not None:
            vae_dtype = torch.float32 if isinstance(self.vae, MiniMaxMusic3DAV) else self.config.weight_dtype
            self.vae.to(self.accelerator.device, dtype=vae_dtype)
        return self.vae

    def encode_cache_batch(self, vae, samples, metadata_entries: Optional[list] = None):
        del metadata_entries
        if not hasattr(vae, "encode"):
            raise RuntimeError(
                "MiniMax Music 3 VAE caching requires the original dav.pth checkpoint with the audio encoder. "
                "Use SimpleTuner/MiniMax-Music-3-Encoder, MiniMaxAI/MiniMax-Music3, or set "
                "pretrained_vae_model_name_or_path to a local path containing dav.pth."
            )
        samples = samples.to(device=self.accelerator.device, dtype=torch.float32)
        latents = vae.encode(samples)
        if not isinstance(latents, torch.Tensor):
            raise TypeError("MiniMax Music 3 DAV encode() must return a tensor.")
        return latents.to(dtype=self.config.weight_dtype)

    def load_text_tokenizer(self):
        if self.tokenizers is not None:
            return
        tokenizer = AutoTokenizer.from_pretrained(
            self._checkpoint_path(),
            subfolder="tokenizer",
            revision=getattr(self.config, "revision", None),
            trust_remote_code=True,
        )
        self.tokenizers = [tokenizer]
        self.tokenizer_1 = tokenizer

    def load_text_encoder(self, move_to_device: bool = True):
        self.load_text_tokenizer()
        if self.language_model is not None and self.rvq_depth_decoder is not None and self.condition_encoder is not None:
            return
        base_path = self._checkpoint_path()
        language_model = Qwen3ForCausalLM.from_pretrained(
            base_path,
            subfolder="language_model",
            revision=getattr(self.config, "revision", None),
            torch_dtype=self.config.weight_dtype,
            trust_remote_code=True,
        )
        rvq_depth_decoder = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
            base_path,
            subfolder="rvq_depth_decoder",
            torch_dtype=self.config.weight_dtype,
        )
        condition_encoder = self.load_condition_encoder(move_to_device=False)
        for component in (language_model, rvq_depth_decoder, condition_encoder):
            component.eval()
            component.requires_grad_(False)
        if move_to_device and not self._ramtorch_text_encoders_requested():
            language_model.to(self.accelerator.device, dtype=self.config.weight_dtype)
            rvq_depth_decoder.to(self.accelerator.device, dtype=self.config.weight_dtype)
            condition_encoder.to(self.accelerator.device, dtype=self.config.weight_dtype)
        if self._ramtorch_text_encoders_requested():
            self._apply_ramtorch_layers(language_model, "text_encoder_1", percent=self._ramtorch_text_encoder_percent())
            rvq_depth_decoder.to(self.accelerator.device, dtype=self.config.weight_dtype)
            condition_encoder.to(self.accelerator.device, dtype=self.config.weight_dtype)
        self.language_model = language_model
        self.rvq_depth_decoder = rvq_depth_decoder
        self.condition_encoder = condition_encoder
        self.text_encoders = [language_model]
        self.text_encoder_1 = language_model

    def load_condition_encoder(self, move_to_device: bool = True):
        if self.condition_encoder is None:
            self.condition_encoder = MiniMaxMusic3ConditionEncoder.from_pretrained(
                self._checkpoint_path(),
                subfolder="condition_encoder",
                revision=getattr(self.config, "revision", None),
                torch_dtype=self.config.weight_dtype,
            )
            self.condition_encoder.eval()
            self.condition_encoder.requires_grad_(False)
        if move_to_device:
            self.condition_encoder.to(self.accelerator.device, dtype=self.config.weight_dtype)
        return self.condition_encoder

    def unload_text_encoder(self):
        super().unload_text_encoder()
        self.language_model = None
        self.rvq_depth_decoder = None
        self.condition_encoder = None

    def _audio_duration_for_context(self, context: dict) -> float:
        for key in (
            "audio_duration",
            "duration",
            "duration_seconds",
            "bucket_duration_seconds",
            "truncated_duration_seconds",
            "original_duration_seconds",
        ):
            value = context.get(key) if isinstance(context, dict) else None
            if value is not None:
                return max(float(value), 0.04)
        return max(float(getattr(self.config, "validation_audio_duration", 60.0) or 60.0), 0.04)

    def _lyrics_for_context(self, context: dict, prompt: str) -> str:
        if isinstance(context, dict):
            lyrics = context.get("lyrics")
            if lyrics:
                return str(lyrics)
        configured = getattr(self.config, "validation_lyrics", None)
        if configured:
            return str(configured)
        return str(prompt)

    @torch.no_grad()
    def _encode_single_prompt(self, prompt: str, context: dict) -> Dict[str, torch.Tensor]:
        self.load_text_encoder(move_to_device=True)
        tokenizer = self.tokenizers[0]
        language_model = self.language_model
        rvq_depth_decoder = self.rvq_depth_decoder
        if language_model is None or rvq_depth_decoder is None:
            raise ValueError("MiniMax Music 3 text components are not loaded.")

        prompt_text = (
            f"<|im_start|><|caption_start|>{_clean_caption(str(prompt))}<|caption_end|>"
            f"<|lyrics_start|>{_normalize_lyrics(self._lyrics_for_context(context, str(prompt)))}<|lyrics_end|>"
            f"<|im_end|><|audio_start|>"
        )
        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"]
        if input_ids.shape[1] > _MAX_PROMPT_TOKENS:
            raise ValueError(
                f"The assembled MiniMax Music 3 prompt has {input_ids.shape[1]} tokens; "
                f"the maximum is {_MAX_PROMPT_TOKENS}."
            )
        unconditional_ids = input_ids.clone()
        unconditional_ids[:, 1:-2] = _AUDIO_CFG_TOKEN_ID
        text_ids = torch.cat((input_ids, unconditional_ids), dim=0).to(self.accelerator.device)

        max_frames = min(int(self._audio_duration_for_context(context) * self._frame_rate()), _MAX_AUDIO_FRAMES)
        text_embeds = language_model.model.embed_tokens(text_ids)
        output = language_model.model(inputs_embeds=text_embeds, use_cache=True)
        past_key_values = output.past_key_values
        last_hidden = output.last_hidden_state[:, -1]

        vocab_mask = torch.ones(language_model.config.vocab_size, dtype=torch.bool, device=text_ids.device)
        vocab_mask[_AUDIO_CODE_OFFSET : _AUDIO_CODE_OFFSET + _SEMANTIC_VOCAB_SIZE] = False
        vocab_mask[_AUDIO_END_TOKEN_ID] = False

        frame_hiddens = []
        generator = None
        for frame_index in range(max_frames + 1):
            logits = language_model.lm_head(last_hidden).float()
            logits = logits.masked_fill(vocab_mask, -float("inf"))
            conditional, unconditional = logits[0:1], logits[1:2]
            guided = unconditional + (conditional - unconditional) * _AR_CFG_SCALE
            threshold = torch.topk(conditional, _AR_CFG_TOP_K, dim=-1).values[..., -1, None]
            guided = guided.masked_fill(conditional < threshold, -float("inf"))
            guided = guided.masked_fill(vocab_mask.unsqueeze(0), -float("inf"))
            sampled = _sample_top_k(guided, generator)
            if int(sampled.item()) == _AUDIO_END_TOKEN_ID:
                break

            semantic_code = sampled - _AUDIO_CODE_OFFSET
            frame_codes, depth_hidden = _generate_depth_codes(
                self._component_proxy(),
                last_hidden,
                semantic_code.repeat(2),
                generator,
            )
            if frame_index > 0:
                frame_hiddens.append(torch.cat((last_hidden[:1], depth_hidden), dim=-1))
                if len(frame_hiddens) >= max_frames:
                    break
            feedback = _embed_audio_frame(self._component_proxy(), frame_codes)
            output = language_model.model(inputs_embeds=feedback, past_key_values=past_key_values, use_cache=True)
            past_key_values = output.past_key_values
            last_hidden = output.last_hidden_state[:, -1]

        if not frame_hiddens:
            raise ValueError("MiniMax Music 3 generated zero conditioning frames for the prompt.")
        return {"prompt_embeds": torch.stack(frame_hiddens, dim=1).to(dtype=self.config.weight_dtype)}

    def _component_proxy(self):
        class _Components:
            pass

        components = _Components()
        components.language_model = self.language_model
        components.rvq_depth_decoder = self.rvq_depth_decoder
        components.num_codebooks = int(self.rvq_depth_decoder.config.num_codebooks)
        components.audio_vocab_size = int(self.rvq_depth_decoder.config.audio_vocab_size)
        return components

    def _frame_rate(self) -> float:
        if self.condition_encoder is not None:
            cfg = self.condition_encoder.config
            return float(cfg.input_sampling_rate) / float(cfg.input_hop_length)
        return 25.0

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False) -> Dict[str, torch.Tensor]:
        if is_negative_prompt:
            raise ValueError("MiniMax Music 3 validation/training does not use negative prompt embeddings.")
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]
        contexts = getattr(self, "_current_prompt_contexts", None) or [{} for _ in prompts]
        if len(contexts) != len(prompts):
            raise ValueError("MiniMax Music 3 text encoding requires one metadata context per prompt.")
        encoded = [self._encode_single_prompt(str(prompt), context or {}) for prompt, context in zip(prompts, contexts)]
        return self.collate_prompt_embeds(encoded)

    def _format_text_embedding(self, text_embedding):
        return text_embedding

    def collate_prompt_embeds(self, text_encoder_output: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not text_encoder_output:
            return {}

        def _norm(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.ndim == 3 and tensor.shape[0] == 1:
                return tensor.squeeze(0)
            return tensor

        embeds = [_norm(item["prompt_embeds"]) for item in text_encoder_output]
        lengths = torch.tensor([item.shape[0] for item in embeds], dtype=torch.long)
        padded = pad_sequence(embeds, batch_first=True, padding_value=0)
        return {"prompt_embeds": padded, "attention_masks": lengths}

    def slice_text_embedding_for_cache(self, text_encoder_output: dict, batch_index: int, batch_size: int) -> dict | None:
        embeds = text_encoder_output.get("prompt_embeds")
        lengths = text_encoder_output.get("attention_masks")
        if embeds is None:
            return None
        sample = embeds[batch_index : batch_index + 1]
        if lengths is not None:
            length = int(lengths[batch_index].item())
            sample = sample[:, :length]
        return {"prompt_embeds": sample.clone().contiguous()}

    def convert_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        return {"frame_hiddens": text_embedding["prompt_embeds"]}

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        return {}

    def _condition_from_frame_hiddens(self, frame_hiddens: torch.Tensor, latent_length: int) -> torch.Tensor:
        if self.condition_encoder is None:
            self.load_condition_encoder(move_to_device=True)
        condition_encoder = self.condition_encoder
        if condition_encoder is None:
            raise ValueError("MiniMax Music 3 condition encoder is not loaded.")
        condition = condition_encoder(frame_hiddens.to(device=self.accelerator.device, dtype=self.config.weight_dtype))
        return self._match_condition_length(condition, latent_length)

    def _match_condition_length(self, condition: torch.Tensor, latent_length: int) -> torch.Tensor:
        if condition.shape[1] == latent_length:
            return condition
        condition = condition.transpose(1, 2)
        condition = F.interpolate(condition, size=latent_length, mode="nearest")
        return condition.transpose(1, 2)

    def _resolve_condition(self, batch: dict, latents: torch.Tensor) -> torch.Tensor:
        latent_length = int(latents.shape[-1])
        candidate = batch.get("encoder_hidden_states")
        if candidate is None:
            candidate = batch.get("prompt_embeds")
        if candidate is None:
            candidate = batch.get("frame_hiddens")
        if candidate is None:
            raise ValueError(
                "MiniMax Music 3 training requires cached frame hidden states or latent-aligned condition embeddings."
            )
        if isinstance(candidate, dict):
            dict_candidate = candidate.get("prompt_embeds")
            if dict_candidate is None:
                dict_candidate = candidate.get("frame_hiddens")
            candidate = dict_candidate
        if not torch.is_tensor(candidate):
            raise ValueError(f"MiniMax Music 3 conditioning must be a tensor, got {type(candidate)}.")
        candidate = candidate.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        transformer = self.unwrap_model(self.model) if self.model is not None else None
        condition_dim = int(getattr(getattr(transformer, "config", None), "condition_dim", candidate.shape[-1]))
        if candidate.shape[-1] == condition_dim:
            return self._match_condition_length(candidate, latent_length)
        return self._condition_from_frame_hiddens(candidate, latent_length)

    def _latent_channel_count(self) -> int:
        transformer = self.unwrap_model(self.model) if self.model is not None else None
        channels = getattr(getattr(transformer, "config", None), "in_channels", None)
        return int(channels) if channels is not None else self.LATENT_CHANNEL_COUNT

    def sample_flow_sigmas(self, batch: dict, state: dict) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = batch["latents"].shape[0]
        device = self.accelerator.device
        dtype = getattr(self.config, "weight_dtype", torch.float32)
        mean = float(getattr(self.config, "logit_mean", 0.0) or 0.0)
        std = float(getattr(self.config, "logit_std", 1.0) or 1.0)
        u = torch.normal(mean=mean, std=std, size=(bsz,), device=device, dtype=torch.float32).sigmoid()
        data_timesteps = u.to(dtype=dtype)
        noise_sigmas = (1.0 - data_timesteps).clamp(0.0, 1.0)
        return noise_sigmas, data_timesteps

    def flow_matching_timesteps_from_sigmas(
        self, sigmas: torch.Tensor, reference_timesteps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        del reference_timesteps
        return (1.0 - sigmas).clamp(0.0, 1.0)

    def prepare_batch(self, batch: dict, state: dict) -> dict:
        if not batch:
            return batch
        latent_batch = batch.get("latent_batch")
        if latent_batch is None:
            latent_batch = batch.get("audio_latent_batch")
        if latent_batch is None:
            raise ValueError(
                "MiniMax Music 3 training requires cached Flow-VAE latents in latent_batch or audio_latent_batch."
            )
        device = self.accelerator.device
        dtype = self.config.weight_dtype
        latents = latent_batch.to(device=device, dtype=dtype)
        if latents.ndim != 3:
            raise ValueError(
                "MiniMax Music 3 Flow-VAE latents must be shaped `[batch, channels, latent_length]`, "
                f"got {tuple(latents.shape)}."
            )
        latent_channels = self._latent_channel_count()
        if latents.shape[1] != latent_channels and latents.shape[-1] == latent_channels:
            latents = latents.transpose(1, 2).contiguous()
        if latents.shape[1] != latent_channels:
            raise ValueError(
                f"MiniMax Music 3 Flow-VAE latents must have {latent_channels} channels, got {latents.shape[1]}."
            )
        batch["latents"] = latents
        batch["encoder_hidden_states"] = self._resolve_condition(batch, latents)

        noise = torch.randn_like(latents)
        batch["noise"] = noise
        input_noise = noise
        input_perturbation_value = float(getattr(self.config, "input_perturbation", 0.0) or 0.0)
        if input_perturbation_value != 0 and (
            not getattr(self.config, "input_perturbation_steps", None)
            or state["global_step"] < self.config.input_perturbation_steps
        ):
            input_perturbation = input_perturbation_value
            if getattr(self.config, "input_perturbation_steps", None):
                input_perturbation *= 1.0 - (state["global_step"] / self.config.input_perturbation_steps)
            input_noise = noise + input_perturbation * torch.randn_like(latents)
        batch["input_noise"] = input_noise

        batch["sigmas"], batch["timesteps"] = self.sample_flow_sigmas(batch=batch, state=state)
        crepa = getattr(self, "crepa_regularizer", None)
        if crepa and crepa.enabled and getattr(crepa, "use_self_flow_features", False):
            batch = self._prepare_crepa_self_flow_batch(batch=batch, state=state)
        else:
            self.expand_sigmas(batch)
            batch["noisy_latents"] = batch["sigmas"] * input_noise + (1.0 - batch["sigmas"]) * latents
            if self._twinflow_active():
                self._prepare_twinflow_metadata(batch)
        return batch

    def _prepare_crepa_self_flow_batch(self, batch: dict, state: dict) -> dict:
        latents = batch["latents"]
        input_noise = batch["input_noise"]
        base_sigmas = batch["sigmas"].to(device=latents.device, dtype=latents.dtype)
        alt_sigmas, _alt_timesteps = self.sample_flow_sigmas(batch=batch, state=state)
        alt_sigmas = alt_sigmas.to(device=latents.device, dtype=latents.dtype)

        mask_ratio = float(getattr(self.config, "crepa_self_flow_mask_ratio", 0.1) or 0.0)
        token_mask = torch.rand(latents.shape[0], latents.shape[-1], device=latents.device, dtype=latents.dtype) < mask_ratio
        base_sigma_tokens = base_sigmas.view(-1, 1)
        alt_sigma_tokens = alt_sigmas.view(-1, 1)
        student_sigmas = torch.where(token_mask, alt_sigma_tokens, base_sigma_tokens)
        teacher_sigmas = torch.minimum(base_sigmas, alt_sigmas)

        batch["sigmas"] = student_sigmas[:, None, :].to(dtype=latents.dtype)
        batch["timesteps"] = (1.0 - student_sigmas).to(dtype=latents.dtype)
        batch["noisy_latents"] = batch["sigmas"] * input_noise + (1.0 - batch["sigmas"]) * latents
        batch["crepa_teacher_sigmas"] = teacher_sigmas.view(-1, 1, 1).to(dtype=latents.dtype)
        batch["crepa_teacher_timesteps"] = (1.0 - teacher_sigmas).to(dtype=latents.dtype)
        batch["crepa_teacher_noisy_latents"] = (
            batch["crepa_teacher_sigmas"] * input_noise + (1.0 - batch["crepa_teacher_sigmas"]) * latents
        )
        batch["crepa_self_flow_mask"] = token_mask
        return batch

    def _timesteps_for_transformer(self, prepared_batch: dict, noisy_latents: torch.Tensor) -> torch.Tensor:
        timesteps = prepared_batch["timesteps"].to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        if timesteps.numel() > 0 and torch.max(timesteps.detach().abs()) > 1.0:
            sigmas = prepared_batch.get("sigmas")
            if sigmas is None:
                timesteps = (timesteps / 1000.0).clamp(0.0, 1.0)
            else:
                sigma_values = sigmas.to(device=noisy_latents.device, dtype=noisy_latents.dtype).abs()
                flat_sigmas = sigma_values.reshape(sigma_values.shape[0], -1)
                if timesteps.ndim > 1 and flat_sigmas.shape[1] == timesteps.reshape(timesteps.shape[0], -1).shape[1]:
                    timesteps = 1.0 - flat_sigmas.reshape_as(timesteps)
                elif timesteps.ndim > 1:
                    timesteps = (timesteps / 1000.0).clamp(0.0, 1.0)
                else:
                    timesteps = 1.0 - flat_sigmas[:, 0]
        return timesteps

    def _flowmap_r_for_transformer(self, prepared_batch: dict, timestep: torch.Tensor) -> Optional[torch.Tensor]:
        r_timestep = prepared_batch.get(self.FLOWMAP_R_TIMESTEP_BATCH_KEY)
        if r_timestep is None:
            return None
        # The AnyFlow distiller converts r through flow_matching_timesteps_from_sigmas,
        # so values arrive in the transformer's [0, 1] flow-time domain already.
        return r_timestep.to(device=timestep.device, dtype=timestep.dtype)

    def _select_crepa_hidden_states(self, prepared_batch: dict, hidden_states_buffer):
        if hidden_states_buffer is None:
            return None
        crepa = getattr(self, "crepa_regularizer", None)
        block_idx = prepared_batch.get(
            "crepa_capture_block_index",
            getattr(crepa, "block_index", None),
        )
        if block_idx is None:
            return None
        return hidden_states_buffer.get(f"layer_{int(block_idx)}")

    def model_predict(self, prepared_batch: dict) -> Dict[str, object]:
        transformer = self.get_trained_component()
        if transformer is None:
            raise ValueError("MiniMax Music 3 transformer has not been loaded before model_predict was invoked.")
        noisy_latents = prepared_batch["noisy_latents"].to(self.accelerator.device, dtype=self.config.weight_dtype)
        encoder_hidden_states = prepared_batch["encoder_hidden_states"].to(
            self.accelerator.device, dtype=self.config.weight_dtype
        )
        timestep = self._timesteps_for_transformer(prepared_batch, noisy_latents)
        r_timestep = self._flowmap_r_for_transformer(prepared_batch, timestep)
        hidden_states_buffer = self._new_hidden_state_buffer()
        crepa = getattr(self, "crepa_regularizer", None)
        capture_block_index = prepared_batch.get(
            "crepa_capture_block_index",
            getattr(crepa, "block_index", None),
        )
        capture_hidden = bool(crepa and crepa.wants_hidden_states() and capture_block_index is not None)

        output = transformer(
            hidden_states=noisy_latents,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            timestep_sign=(
                prepared_batch.get("twinflow_time_sign") if getattr(self.config, "twinflow_enabled", False) else None
            ),
            r_timestep=r_timestep,
            skip_layers=prepared_batch.get("skip_layers"),
            hidden_states_buffer=hidden_states_buffer,
            output_hidden_states=capture_hidden,
            hidden_state_layer=capture_block_index,
            return_dict=True,
        )
        crepa_hidden_states = getattr(output, "hidden_states", None)
        if crepa_hidden_states is None:
            crepa_hidden_states = self._select_crepa_hidden_states(prepared_batch, hidden_states_buffer)
        return {
            "model_prediction": output.sample,
            "crepa_hidden_states": crepa_hidden_states,
            "hidden_states_buffer": hidden_states_buffer,
        }

    def get_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2AUDIO, load_base_model: bool = True):
        if isinstance(pipeline_type, str):
            pipeline_type = PipelineTypes(pipeline_type)
        if pipeline_type not in self.PIPELINE_CLASSES:
            raise NotImplementedError(f"Pipeline type {pipeline_type} not defined in {self.__class__.__name__}.")
        cached_pipeline = self.pipelines.get(pipeline_type)
        transformer = self.unwrap_model(self.model) if self.model is not None else None
        if cached_pipeline is None:
            cached_pipeline = MiniMaxMusic3ModularPipeline(MiniMaxMusic3Blocks())
            self.pipelines[pipeline_type] = cached_pipeline
        if (
            self.tokenizers is None
            or self.language_model is None
            or self.rvq_depth_decoder is None
            or self.condition_encoder is None
        ):
            self.load_text_encoder(move_to_device=True)
        if self.vae is None:
            self.load_vae(move_to_device=True)
        if self.guider is None:
            self.guider = ClassifierFreeGuidance(guidance_scale=1.7)
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1,
            shift=float(getattr(self.config, "flow_schedule_shift", 1.0) or 1.0),
            invert_sigmas=True,
        )
        cached_pipeline.update_components(
            tokenizer=self.tokenizers[0],
            language_model=self.language_model,
            rvq_depth_decoder=self.rvq_depth_decoder,
            condition_encoder=self.condition_encoder,
            transformer=transformer,
            scheduler=scheduler,
            guider=self.guider,
            vocoder=self.vae,
        )
        return cached_pipeline

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        validation_prompt = pipeline_kwargs.get("_validation_prompt_text")
        if pipeline_kwargs.get("prompt") is None and validation_prompt:
            pipeline_kwargs["prompt"] = validation_prompt
        lyrics = pipeline_kwargs.get("lyrics")
        if not isinstance(lyrics, str) or not lyrics.strip():
            configured_lyrics = getattr(self.config, "validation_lyrics", None)
            if configured_lyrics and str(configured_lyrics).strip():
                pipeline_kwargs["lyrics"] = str(configured_lyrics)
            elif validation_prompt:
                pipeline_kwargs["lyrics"] = str(validation_prompt)
        for cached_embed_key in ("prompt_embeds", "attention_masks"):
            pipeline_kwargs.pop(cached_embed_key, None)
        return pipeline_kwargs


ModelRegistry.register("minimaxmusic", MiniMaxMusic)
