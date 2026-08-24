# Copyright 2026 The MiniMax Team and The HuggingFace Team.
# Modifications for SimpleTuner are distributed under the AGPL-3.0-or-later.

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.guiders import ClassifierFreeGuidance
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, HFValidationError, LocalEntryNotFoundError, RepositoryNotFoundError
from torch import nn
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
from simpletuner.helpers.training.explorative_modeling import (
    blockwise_cross_entropy,
    route_usage_histogram,
    select_min_candidate_loss,
    select_winning_candidates,
)
from simpletuner.helpers.training.lora_format import (
    PEFTLoRAFormat,
    collect_lora_ranks,
    detect_state_dict_format,
    normalize_lora_format,
)

logger = logging.getLogger(__name__)

DEFAULT_RVQ_ENCODER_MODEL = "SimpleTuner/open-rvq-encoder-minimax-music3-169m-v4"
DEFAULT_RVQ_ENCODER_SUBFOLDER = "final"
DEFAULT_LM_AUDIO_VAE_MODEL = "SimpleTuner/MiniMax-Music-3-Encoder"


class MiniMaxMusicRVQCacheEncoder(nn.Module):
    """Cache-time encoder that turns waveforms into MiniMax Music RVQ code tensors."""

    def __init__(self, *, audio_vae: MiniMaxMusic3DAV, rvq_encoder: nn.Module):
        super().__init__()
        self.audio_vae = audio_vae
        self.rvq_encoder = rvq_encoder
        self.config = getattr(rvq_encoder, "config", SimpleNamespace())

    @property
    def device(self) -> torch.device:
        try:
            return next(self.rvq_encoder.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def to(self, device=None, dtype=None, **kwargs):
        del dtype
        self.audio_vae.to(device=device, dtype=torch.float32, **kwargs)
        self.rvq_encoder.to(device=device, dtype=torch.float32, **kwargs)
        return self

    def requires_grad_(self, requires_grad: bool = False):
        self.audio_vae.requires_grad_(requires_grad)
        self.rvq_encoder.requires_grad_(requires_grad)
        return self

    def eval(self):
        super().eval()
        self.audio_vae.eval()
        self.rvq_encoder.eval()
        return self

    @staticmethod
    def _legacy_frame_latent_starts(n_frames: int) -> list[int]:
        latent_rate_num = 441
        latent_rate_den = 128
        chunk_frames = 200
        chunk_hop_frames = 100
        stitched_hop_latents = 345
        non_first_chunk_owned_from = 25
        num_windows = max(1, (n_frames - 1) // chunk_hop_frames)
        starts = []
        for frame_index in range(n_frames + 1):
            chunk_index = min(
                max((frame_index - non_first_chunk_owned_from) // chunk_hop_frames, 0),
                num_windows - 1,
            )
            local_frame = frame_index - chunk_index * chunk_hop_frames
            current_chunk_frames = min(chunk_frames, n_frames - chunk_index * chunk_hop_frames)
            chunk_latents = current_chunk_frames * latent_rate_num // latent_rate_den
            local_latent = (local_frame * chunk_latents + current_chunk_frames - 1) // current_chunk_frames
            starts.append(chunk_index * stitched_hop_latents + local_latent)
        return starts

    @staticmethod
    def _build_window_pool(bounds: list[int], device: torch.device) -> tuple[torch.Tensor, int, int]:
        if len(bounds) < 2:
            raise ValueError("At least two frame/latent boundaries are required.")
        origin = int(bounds[0])
        local_bounds = [int(value) - origin for value in bounds]
        latent_count = local_bounds[-1]
        if latent_count <= 0:
            raise ValueError(f"Invalid MiniMax Music RVQ latent bounds: {bounds[:4]}...{bounds[-4:]}.")
        pool = torch.zeros((len(local_bounds) - 1, latent_count), dtype=torch.float32, device=device)
        for frame_index, (start, end) in enumerate(zip(local_bounds[:-1], local_bounds[1:])):
            if end <= start:
                raise ValueError(f"MiniMax Music RVQ frame {frame_index} has invalid latent span [{start}, {end}).")
            pool[frame_index, start:end] = 1.0 / float(end - start)
        return pool, origin, int(bounds[-1])

    def _max_position_embeddings(self) -> int:
        configured = getattr(getattr(self.rvq_encoder, "config", None), "max_position_embeddings", None)
        if configured is not None:
            return int(configured)
        position = getattr(self.rvq_encoder, "position", None)
        if torch.is_tensor(position) and position.ndim >= 2:
            return int(position.shape[1])
        raise ValueError("MiniMax Music RVQ encoder does not expose max_position_embeddings.")

    @classmethod
    def _frame_count_for_cache(
        cls,
        *,
        latent_frames: int,
        sample_frames: int,
        sample_rate: int,
        frame_rate: float,
    ) -> int:
        duration_frames = max(1, int(round(float(sample_frames) / float(sample_rate) * float(frame_rate))))
        frame_count = min(duration_frames, _MAX_AUDIO_FRAMES)
        while frame_count > 1 and cls._legacy_frame_latent_starts(frame_count)[-1] > latent_frames:
            frame_count -= 1
        return frame_count

    @staticmethod
    def _resample_audio_if_needed(
        waveform: torch.Tensor,
        *,
        source_rate: Optional[int],
        target_rate: int,
    ) -> torch.Tensor:
        if source_rate is None or int(source_rate) == int(target_rate):
            return waveform
        import torchaudio

        return torchaudio.functional.resample(waveform, int(source_rate), int(target_rate))

    @torch.no_grad()
    def encode_audio_codes(
        self,
        samples: torch.Tensor,
        *,
        sample_rates: Optional[list[Optional[int]]] = None,
        device: Optional[torch.device] = None,
        frame_rate: float = 25.0,
    ) -> torch.Tensor:
        if samples.ndim != 3:
            raise ValueError(
                f"MiniMax Music LM RVQ cache expects audio [batch, channels, samples], got {tuple(samples.shape)}."
            )

        device = device or self.device
        target_rate = int(getattr(self.audio_vae.config, "sampling_rate", 44100) or 44100)
        sample_rates = sample_rates or [None for _ in range(samples.shape[0])]
        code_rows = []
        for index in range(samples.shape[0]):
            source_rate = sample_rates[index] if index < len(sample_rates) else None
            waveform = samples[index : index + 1].to(device=device, dtype=self.dtype)
            waveform = self._resample_audio_if_needed(
                waveform.squeeze(0),
                source_rate=int(source_rate) if source_rate else None,
                target_rate=target_rate,
            ).unsqueeze(0)
            latents = self.audio_vae.encode(waveform)
            if not isinstance(latents, torch.Tensor):
                raise TypeError("MiniMax Music LM RVQ cache DAV encode() must return a tensor.")
            latent_rows = latents[0].transpose(0, 1).contiguous()
            latent_frames = int(latent_rows.shape[0])
            frame_count = self._frame_count_for_cache(
                latent_frames=latent_frames,
                sample_frames=int(waveform.shape[-1]),
                sample_rate=target_rate,
                frame_rate=frame_rate,
            )
            bounds = self._legacy_frame_latent_starts(frame_count)
            max_window = self._max_position_embeddings()
            code_windows = []
            for frame_start in range(0, frame_count, max_window):
                frame_end = min(frame_start + max_window, frame_count)
                pool, latent_start, latent_end = self._build_window_pool(
                    bounds[frame_start : frame_end + 1],
                    device=latent_rows.device,
                )
                window_latents = latent_rows[latent_start:latent_end].unsqueeze(0).to(dtype=torch.float32)
                logits = self.rvq_encoder(window_latents, pool.unsqueeze(0))
                if not isinstance(logits, (list, tuple)) or len(logits) == 0:
                    raise TypeError("MiniMax Music RVQ encoder must return one logits tensor per codebook.")
                code_windows.append(torch.stack([book_logits.argmax(dim=-1).squeeze(0) for book_logits in logits], dim=-1))
            codes = torch.cat(code_windows, dim=0)
            code_rows.append(codes.to(device="cpu", dtype=torch.long))
        if len(code_rows) == 1:
            return code_rows[0].unsqueeze(0)
        return pad_sequence(code_rows, batch_first=True, padding_value=0)


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
    _train_language_model = False
    XM_ROUTE_EMBEDDING_MODULE_NAME = "xm_route_embeddings"

    def __init__(self, config, accelerator):
        user_pretrained_vae_model_name_or_path = getattr(config, "pretrained_vae_model_name_or_path", None)
        super().__init__(config, accelerator)
        self._user_pretrained_vae_model_name_or_path = user_pretrained_vae_model_name_or_path
        self.condition_encoder: Optional[MiniMaxMusic3ConditionEncoder] = None
        self.rvq_depth_decoder: Optional[MiniMaxMusic3RVQDepthDecoder] = None
        self.language_model: Optional[Qwen3ForCausalLM] = None
        self.lm_rvq_cache_encoder: Optional[MiniMaxMusicRVQCacheEncoder] = None
        self.guider: Optional[ClassifierFreeGuidance] = None
        self.vae = None
        train_component = str(getattr(config, "minimax_music_train_component", "transformer") or "transformer")
        self._train_language_model = train_component == "language_model"
        if self._train_language_model:
            self.PREDICTION_TYPE = PredictionTypes.AUTOREGRESSIVE_NEXT_TOKEN
            self.MODEL_CLASS = Qwen3ForCausalLM
            self.MODEL_SUBFOLDER = "language_model"
            self.AUTOENCODER_CLASS = MiniMaxMusicRVQCacheEncoder
            self.TEXT_ENCODER_CONFIGURATION = {}
            self.DEFAULT_LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    def custom_model_card_training_mode_info(self, args) -> str:
        train_component = str(getattr(args, "minimax_music_train_component", "transformer") or "transformer")
        component_labels = {
            "language_model": "language_model (global LM / RVQ planner)",
            "transformer": "transformer (DiT/audio denoiser)",
        }
        lines = [f"- MiniMax Music train component: `{component_labels.get(train_component, train_component)}`"]
        lm_max_frames = getattr(args, "minimax_music_lm_max_frames", None)
        if lm_max_frames:
            lines.append(f"- MiniMax Music LM max frames: `{lm_max_frames}`")
            lines.append(
                "- MiniMax Music LM window mode: " f"`{getattr(args, 'minimax_music_lm_window_mode', 'prefix') or 'prefix'}`"
            )
        return "\n".join(lines)

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
            "audio_crop_start_seconds",
            "audio_crop_end_seconds",
            "audio_crop_duration_seconds",
            "audio_crop_is_terminal",
            "audio_tokens",
            "audio_tokens_path",
            "data_backend_id",
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
        if self._lm_xm_route_enabled():
            self._ensure_lm_xm_route_embeddings()
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
        return "language_model" if self._train_language_model else "transformer"

    def check_user_config(self):
        super().check_user_config()
        if getattr(self.xm_config, "enabled", False):
            if self._train_language_model:
                if self.xm_config.training_target != "route":
                    raise ValueError(
                        "MiniMax Music 3 language model training supports XM only with xm_training_target=route."
                    )
            elif self.xm_config.training_target == "route":
                raise ValueError(
                    "MiniMax Music 3 XM route training is implemented only for "
                    "--minimax_music_train_component=language_model."
                )
        if self._train_language_model:
            window_mode = self._lm_window_mode()
            max_frames = int(getattr(self.config, "minimax_music_lm_max_frames", 0) or 0)
            if window_mode in {"random", "continuation"} and max_frames <= 0:
                raise ValueError(
                    f"--minimax_music_lm_window_mode={window_mode} requires "
                    "--minimax_music_lm_max_frames to be greater than 0."
                )
            if normalize_lora_format(getattr(self.config, "lora_format", None)) == PEFTLoRAFormat.COMFYUI:
                raise ValueError(
                    "--lora_format comfyui applies to the music transformer; use the default diffusers format "
                    "when training the language model."
                )
            if str(getattr(self.config, "model_type", "lora")).lower() == "lora" and (
                str(getattr(self.config, "lora_type", "standard")).lower() != "standard"
            ):
                raise ValueError("MiniMax Music 3 language model training supports standard PEFT LoRA only.")
            if not getattr(self.config, "validation_disable", False):
                logger.warning(
                    "MiniMax Music 3 language model training does not support in-trainer validation audio yet; "
                    "disabling validation. Render audio from saved checkpoints instead."
                )
                self.config.validation_disable = True

    def enable_gradient_checkpointing(self):
        if self._train_language_model and self.model is not None:
            self.unwrap_model(self.model).gradient_checkpointing_enable()

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
        if self._train_language_model:
            from safetensors.torch import save_file

            save_directory = args[0] if args else kwargs.get("save_directory")
            if save_directory is None:
                raise ValueError("save_directory is required to save LoRA weights.")
            os.makedirs(save_directory, exist_ok=True)
            language_model = self.unwrap_model(self.model)
            if not hasattr(language_model, "get_adapter_state_dict"):
                raise NotImplementedError("MiniMax Music 3 language model LoRA saving requires a PEFT adapter.")
            adapter_state = {
                f"language_model.{key}": value.detach().cpu()
                for key, value in language_model.get_adapter_state_dict().items()
                if (
                    "lora_" in key
                    or ".modules_to_save." in key
                    or self.XM_ROUTE_EMBEDDING_MODULE_NAME in key
                    or "nextlat_predictor" in key
                )
            }
            save_file(adapter_state, os.path.join(save_directory, "pytorch_lora_weights.safetensors"))
            return None
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

    def get_lora_save_layers(self):
        save_layers = list(super().get_lora_save_layers() or [])
        if self._lm_xm_route_enabled():
            save_layers.append(self.XM_ROUTE_EMBEDDING_MODULE_NAME)
        return list(dict.fromkeys(save_layers)) or None

    def add_lora_adapter(self):
        if self._lm_xm_route_enabled():
            self._ensure_lm_xm_route_embeddings()
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

    def uses_audio_latents(self) -> bool:
        return not self._train_language_model

    def uses_audio_tokens(self) -> bool:
        return self._train_language_model

    def uses_text_embeddings_cache(self) -> bool:
        return not self._train_language_model

    def get_vae_for_dataset_type(self, dataset_type: str):
        if self._train_language_model and str(dataset_type).lower() == "audio":
            return self.load_lm_rvq_cache_encoder(move_to_device=True)
        return super().get_vae_for_dataset_type(dataset_type)

    def load_model(self, move_to_device: bool = True):
        if not self._train_language_model:
            return super().load_model(move_to_device=move_to_device)
        self.load_text_tokenizer()
        base_path = self._checkpoint_path()
        self.model = Qwen3ForCausalLM.from_pretrained(
            base_path,
            subfolder="language_model",
            revision=getattr(self.config, "revision", None),
            torch_dtype=self.config.weight_dtype,
            trust_remote_code=True,
        )
        self.rvq_depth_decoder = MiniMaxMusic3RVQDepthDecoder.from_pretrained(
            base_path,
            subfolder="rvq_depth_decoder",
            torch_dtype=self.config.weight_dtype,
        )
        self.rvq_depth_decoder.eval()
        self.rvq_depth_decoder.requires_grad_(False)
        if move_to_device:
            self.model.to(self.accelerator.device)
            self.rvq_depth_decoder.to(self.accelerator.device, dtype=self.config.weight_dtype)
        self.apply_gradient_checkpointing_settings()
        self.post_model_load_setup()
        return self.model

    def load_vae(self, move_to_device: bool = True):
        if self._train_language_model:
            self.vae = self.load_lm_rvq_cache_encoder(move_to_device=move_to_device)
            return self.vae
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

    def _lm_rvq_encoder_path(self) -> str:
        return getattr(self.config, "minimax_music_rvq_encoder_model_name_or_path", None) or DEFAULT_RVQ_ENCODER_MODEL

    def _lm_rvq_encoder_subfolder(self) -> str:
        return (getattr(self.config, "minimax_music_rvq_encoder_subfolder", None) or DEFAULT_RVQ_ENCODER_SUBFOLDER).strip(
            "/"
        )

    def _lm_rvq_encoder_revision(self) -> Optional[str]:
        return getattr(self.config, "minimax_music_rvq_encoder_revision", None) or getattr(self.config, "revision", None)

    def _resolve_lm_rvq_file(self, filename: str) -> str:
        root = self._lm_rvq_encoder_path()
        subfolder = self._lm_rvq_encoder_subfolder()
        if os.path.isdir(root):
            candidates = []
            if subfolder:
                candidates.append(os.path.join(root, subfolder, filename))
            candidates.append(os.path.join(root, filename))
            for candidate in candidates:
                if os.path.isfile(candidate):
                    return candidate
            raise FileNotFoundError(f"MiniMax Music RVQ encoder file not found under {root}: {filename}")
        repo_filename = f"{subfolder}/{filename}" if subfolder else filename
        return hf_hub_download(
            root,
            repo_filename,
            revision=self._lm_rvq_encoder_revision(),
            repo_type="model",
        )

    def _lm_audio_vae_path(self) -> str:
        return (
            getattr(self.config, "minimax_music_rvq_vae_model_name_or_path", None)
            or self._user_pretrained_vae_model_name_or_path
            or DEFAULT_LM_AUDIO_VAE_MODEL
        )

    def _load_lm_audio_vae(self) -> MiniMaxMusic3DAV:
        checkpoint_path = self._lm_audio_vae_path()
        if self._has_diffusers_audio_vae(checkpoint_path):
            return self._load_diffusers_audio_vae(checkpoint_path)
        if os.path.isfile(checkpoint_path):
            return MiniMaxMusic3DAV.from_original_dav(checkpoint_path)
        if os.path.isdir(checkpoint_path):
            dav_path = os.path.join(checkpoint_path, "dav.pth")
            if os.path.isfile(dav_path):
                return MiniMaxMusic3DAV.from_original_dav(dav_path)
        dav_checkpoint = None
        try:
            dav_checkpoint = hf_hub_download(
                checkpoint_path,
                "dav.pth",
                revision=getattr(self.config, "revision", None),
                repo_type="model",
            )
        except (EntryNotFoundError, LocalEntryNotFoundError, RepositoryNotFoundError, HFValidationError):
            dav_checkpoint = None
        if dav_checkpoint is not None:
            return MiniMaxMusic3DAV.from_original_dav(dav_checkpoint)
        return self._load_diffusers_audio_vae(checkpoint_path)

    def load_lm_rvq_cache_encoder(self, move_to_device: bool = True) -> MiniMaxMusicRVQCacheEncoder:
        if not self._train_language_model:
            raise ValueError("MiniMax Music RVQ cache encoder is only used for language_model training.")
        if self.lm_rvq_cache_encoder is None:
            from safetensors.torch import load_file as load_safetensors_file

            from scripts.train_minimax_music_rvq_encoder import (
                MiniMaxMusicRVQEncoder,
                RVQEncoderConfig,
                _load_mup_package,
                _validate_mup_shape_metadata,
            )

            config_path = self._resolve_lm_rvq_file("rvq_encoder_config.json")
            config_values = json.loads(Path(config_path).read_text(encoding="utf-8"))
            config_values["codebook_vocab_sizes"] = tuple(int(value) for value in config_values["codebook_vocab_sizes"])
            config_values["conv_dilations"] = tuple(int(value) for value in config_values["conv_dilations"])
            rvq_config = RVQEncoderConfig(**config_values)
            rvq_encoder = MiniMaxMusicRVQEncoder(rvq_config)
            if rvq_config.mup:
                base_shapes_path = self._resolve_lm_rvq_file("mup_base_shapes.bsh")
                self._resolve_lm_rvq_file("mup_base_shapes.bsh.meta.json")
                _validate_mup_shape_metadata(base_shapes_path, rvq_encoder)
                _load_mup_package().set_base_shapes(rvq_encoder, base_shapes_path, rescale_params=False)
            state_path = self._resolve_lm_rvq_file("rvq_encoder.safetensors")
            rvq_encoder.load_state_dict(load_safetensors_file(state_path, device="cpu"), strict=True)
            rvq_encoder.eval()
            rvq_encoder.requires_grad_(False)

            audio_vae = self._load_lm_audio_vae()
            audio_vae.eval()
            audio_vae.requires_grad_(False)
            self.lm_rvq_cache_encoder = MiniMaxMusicRVQCacheEncoder(audio_vae=audio_vae, rvq_encoder=rvq_encoder)
        if move_to_device:
            self.lm_rvq_cache_encoder.to(self.accelerator.device)
        return self.lm_rvq_cache_encoder

    def encode_cache_batch(self, vae, samples, metadata_entries: Optional[list] = None):
        if self._train_language_model:
            return self._encode_lm_rvq_cache_batch(vae, samples, metadata_entries=metadata_entries)
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

    def _encode_lm_rvq_cache_batch(
        self,
        cache_encoder,
        samples: torch.Tensor,
        metadata_entries: Optional[list] = None,
    ) -> torch.Tensor | dict:
        if not isinstance(cache_encoder, MiniMaxMusicRVQCacheEncoder):
            raise TypeError(
                "MiniMax Music LM VAE cache expects MiniMaxMusicRVQCacheEncoder; " f"received {type(cache_encoder)}."
            )

        entries = metadata_entries or [{} for _ in range(samples.shape[0])]
        sample_rates = []
        boundary_metadata = []
        for index in range(samples.shape[0]):
            entry = entries[index] if index < len(entries) else {}
            metadata = entry.get("metadata", {}) if isinstance(entry, dict) else {}
            backend_id = entry.get("data_backend_id") if isinstance(entry, dict) else None
            backend_audio_config = {}
            if backend_id:
                from simpletuner.helpers.training.state_tracker import StateTracker

                backend_audio_config = (StateTracker.get_data_backend_config(backend_id) or {}).get("audio", {})
            source_rate = (
                metadata.get("sample_rate") or metadata.get("sampling_rate") or backend_audio_config.get("sample_rate")
            )
            sample_rates.append(int(source_rate) if source_rate else None)
            boundary_metadata.append(
                self._lm_audio_boundary_metadata(
                    metadata,
                    source_rate=source_rate,
                    sample_count=samples[index].shape[-1],
                )
            )
        codes = cache_encoder.encode_audio_codes(
            samples,
            sample_rates=sample_rates,
            device=self.accelerator.device,
            frame_rate=self._frame_rate(),
        )
        return {"latents": codes, "metadata": boundary_metadata}

    @staticmethod
    def _metadata_float(metadata: dict, *keys: str) -> Optional[float]:
        for key in keys:
            value = metadata.get(key)
            if value in (None, ""):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _metadata_bool(metadata: dict, key: str) -> Optional[bool]:
        value = metadata.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes"}:
                return True
            if normalized in {"0", "false", "no"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return None

    def _lm_audio_boundary_metadata(
        self,
        metadata: dict,
        *,
        source_rate: Any = None,
        sample_count: Optional[int] = None,
    ) -> dict:
        if not isinstance(metadata, dict):
            metadata = {}
        crop_start_seconds = self._metadata_float(metadata, "audio_crop_start_seconds")
        crop_end_seconds = self._metadata_float(metadata, "audio_crop_end_seconds")
        crop_duration_seconds = self._metadata_float(
            metadata,
            "audio_crop_duration_seconds",
            "truncated_duration_seconds",
            "duration_seconds",
        )
        original_duration_seconds = self._metadata_float(
            metadata,
            "original_duration_seconds",
            "audio_original_duration_seconds",
        )
        if crop_start_seconds is None:
            crop_start_seconds = 0.0
        if crop_duration_seconds is None and sample_count is not None:
            try:
                sample_rate = float(source_rate or metadata.get("sample_rate") or metadata.get("sampling_rate"))
                if sample_rate > 0:
                    crop_duration_seconds = float(sample_count) / sample_rate
            except (TypeError, ValueError):
                pass
        if crop_end_seconds is None and crop_duration_seconds is not None:
            crop_end_seconds = crop_start_seconds + crop_duration_seconds
        if original_duration_seconds is None:
            original_duration_seconds = crop_end_seconds or crop_duration_seconds
        terminal = self._metadata_bool(metadata, "audio_crop_is_terminal")
        if terminal is None:
            if original_duration_seconds is not None and crop_end_seconds is not None:
                terminal = crop_end_seconds >= original_duration_seconds - (0.5 / self._frame_rate())
            else:
                terminal = True
        return {
            "audio_crop_start_seconds": float(crop_start_seconds or 0.0),
            "audio_crop_end_seconds": float(crop_end_seconds) if crop_end_seconds is not None else None,
            "audio_crop_duration_seconds": float(crop_duration_seconds) if crop_duration_seconds is not None else None,
            "original_duration_seconds": float(original_duration_seconds) if original_duration_seconds is not None else None,
            "audio_crop_is_terminal": bool(terminal),
        }

    def _lm_boundary_frame_context(self, metadata: dict, cached_frame_count: int) -> tuple[int, int, bool]:
        frame_rate = self._frame_rate()
        crop_start_seconds = self._metadata_float(metadata, "audio_crop_start_seconds") or 0.0
        source_start_frame = max(0, int(round(crop_start_seconds * frame_rate)))

        original_duration_seconds = self._metadata_float(metadata, "original_duration_seconds")
        crop_end_seconds = self._metadata_float(metadata, "audio_crop_end_seconds")
        if original_duration_seconds is not None:
            source_total_frames = int(round(original_duration_seconds * frame_rate))
        elif crop_end_seconds is not None:
            source_total_frames = int(round(crop_end_seconds * frame_rate))
        else:
            source_total_frames = source_start_frame + cached_frame_count
        source_total_frames = max(source_total_frames, source_start_frame + cached_frame_count)

        crop_is_terminal = self._metadata_bool(metadata, "audio_crop_is_terminal")
        if crop_is_terminal is None:
            if original_duration_seconds is not None and crop_end_seconds is not None:
                crop_is_terminal = crop_end_seconds >= original_duration_seconds - (0.5 / frame_rate)
            else:
                crop_is_terminal = True
        return source_start_frame, source_total_frames, bool(crop_is_terminal)

    def _lm_window_mode(self) -> str:
        mode = str(getattr(self.config, "minimax_music_lm_window_mode", "prefix") or "prefix").strip().lower()
        if mode not in {"prefix", "random", "continuation"}:
            raise ValueError("MiniMax Music 3 LM window mode must be one of: prefix, random, continuation.")
        return mode

    def _lm_prompt_text(
        self,
        caption: str,
        lyrics: str,
        *,
        window_start_frame: int = 0,
        window_frame_count: Optional[int] = None,
        total_frame_count: Optional[int] = None,
        window_mode: str = "prefix",
    ) -> str:
        window_text = ""
        if window_frame_count is not None:
            frame_rate = self._frame_rate()
            start_seconds = float(window_start_frame) / frame_rate
            end_seconds = float(window_start_frame + window_frame_count) / frame_rate
            total_seconds = (
                float(total_frame_count) / frame_rate
                if total_frame_count is not None and total_frame_count > 0
                else end_seconds
            )
            window_text = (
                f"<|window_start|>{start_seconds:.2f}s<|window_end|>{end_seconds:.2f}s"
                f"<|track_duration|>{total_seconds:.2f}s<|window_kind|>audio_excerpt"
            )
        return (
            f"<|im_start|><|caption_start|>{_clean_caption(str(caption))}<|caption_end|>"
            f"{window_text}<|lyrics_start|>{_normalize_lyrics(str(lyrics))}<|lyrics_end|>"
            f"<|im_end|><|audio_start|>"
        )

    def _lm_slice_audio_codes(
        self,
        codes: torch.Tensor,
        max_frames: int,
        window_mode: str,
    ) -> tuple[torch.Tensor, int, int, bool]:
        original_frames = int(codes.shape[0])
        available_frames = min(original_frames, _MAX_AUDIO_FRAMES)
        frame_limit = max_frames if 0 < max_frames < available_frames else available_frames
        start_frame = 0
        loss_start_frame = 0
        if frame_limit < available_frames and window_mode in {"random", "continuation"}:
            max_start = available_frames - frame_limit
            start_frame = int(torch.randint(0, max_start + 1, (1,)).item())
        if window_mode == "continuation":
            end_frame = start_frame + frame_limit
            sliced_codes = codes[:end_frame]
            loss_start_frame = start_frame
            truncated = end_frame < original_frames
        else:
            sliced_codes = codes[start_frame : start_frame + frame_limit]
            truncated = frame_limit < original_frames
        return sliced_codes, start_frame, loss_start_frame, truncated

    def _lm_load_audio_codes(self, example: dict) -> tuple[torch.Tensor, dict]:
        codes = example.get("audio_tokens")
        cache_metadata = {}
        if codes is None:
            token_path = example.get("audio_tokens_path")
            if token_path:
                resolved = str(token_path)
                if not os.path.isabs(resolved):
                    backend_id = example.get("data_backend_id")
                    from simpletuner.helpers.training.state_tracker import StateTracker

                    backend_cfg = StateTracker.get_data_backend_config(backend_id) if backend_id else {}
                    dataset_root = backend_cfg.get("instance_data_dir") if backend_cfg else None
                    if dataset_root:
                        resolved = os.path.join(dataset_root, resolved)
                if not os.path.exists(resolved):
                    raise FileNotFoundError(f"MiniMax Music 3 audio token file not found: {resolved}")
                codes = torch.load(resolved, map_location="cpu", weights_only=True)
            else:
                backend_id = example.get("data_backend_id")
                filepath = example.get("image_path") or example.get("filepath")
                if not backend_id or not filepath:
                    raise ValueError(
                        "MiniMax Music 3 language model training requires audio_tokens, audio_tokens_path, "
                        "or an audio dataset sample with data_backend_id and image_path so the RVQ VAE cache can "
                        "provide raw per-codebook codes."
                    )
                from simpletuner.helpers.training.state_tracker import StateTracker

                codes = StateTracker.get_vaecache(id=backend_id).retrieve_from_cache(filepath)
        if isinstance(codes, dict):
            metadata = codes.get("metadata")
            if isinstance(metadata, dict):
                cache_metadata = metadata
            for key in ("codes", "audio_tokens", "latents"):
                if codes.get(key) is not None:
                    codes = codes[key]
                    break
            else:
                raise ValueError("MiniMax Music 3 cached audio entry does not contain codes, audio_tokens, or latents.")
        if not isinstance(codes, torch.Tensor):
            codes = torch.as_tensor(codes)
        codes = codes.to(dtype=torch.long)
        num_codebooks = int(self.rvq_depth_decoder.config.num_codebooks)
        if codes.ndim != 2 or codes.shape[1] != num_codebooks:
            raise ValueError(
                f"MiniMax Music 3 audio codes must be shaped [frames, {num_codebooks}], got {tuple(codes.shape)}."
            )
        audio_vocab = int(self.rvq_depth_decoder.config.audio_vocab_size)
        if int(codes[:, 0].max()) >= _SEMANTIC_VOCAB_SIZE or int(codes[:, 1:].max()) >= audio_vocab:
            raise ValueError(
                "MiniMax Music 3 audio codes must be raw per-codebook indices (semantic < "
                f"{_SEMANTIC_VOCAB_SIZE}, residual < {audio_vocab}). Re-export them without vocabulary offsets."
            )
        boundary_source = dict(cache_metadata)
        for key in (
            "audio_crop_start_seconds",
            "audio_crop_end_seconds",
            "audio_crop_duration_seconds",
            "audio_crop_is_terminal",
            "original_duration_seconds",
            "audio_original_duration_seconds",
        ):
            if example.get(key) not in (None, ""):
                boundary_source[key] = example[key]
        return codes, self._lm_audio_boundary_metadata(boundary_source)

    def collate_audio_tokens(self, examples: list[dict]) -> dict:
        if not self._train_language_model:
            raise ValueError("collate_audio_tokens is only used when --minimax_music_train_component=language_model.")
        self.load_text_tokenizer()
        tokenizer = self.tokenizers[0]
        max_frames = int(getattr(self.config, "minimax_music_lm_max_frames", 0) or 0)
        window_mode = self._lm_window_mode()

        input_id_rows = []
        code_rows = []
        prompt_lengths = []
        audio_lengths = []
        audio_window_start_frames = []
        audio_loss_start_frames = []
        audio_total_frames = []
        has_audio_end = []
        prompts = []
        for example in examples:
            caption = example.get("prompt") or example.get("tags")
            lyrics = example.get("lyrics")
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError("MiniMax Music 3 language model training requires 'prompt' (or 'tags') metadata.")
            if not isinstance(lyrics, str):
                raise ValueError(
                    "MiniMax Music 3 language model training requires 'lyrics' metadata (an empty string is "
                    "allowed for instrumental or regularisation tracks)."
                )
            codes, boundary_metadata = self._lm_load_audio_codes(example)
            source_start_frame, total_frames, source_crop_is_terminal = self._lm_boundary_frame_context(
                boundary_metadata,
                int(codes.shape[0]),
            )
            codes, start_frame, loss_start_frame, truncated = self._lm_slice_audio_codes(codes, max_frames, window_mode)
            absolute_start_frame = source_start_frame + start_frame
            sequence_start_frame = source_start_frame if window_mode == "continuation" else absolute_start_frame
            sequence_end_frame = sequence_start_frame + int(codes.shape[0])
            cached_source_excerpt = source_start_frame > 0 or not source_crop_is_terminal
            prefix_truncated_excerpt = window_mode == "prefix" and truncated
            random_excerpt = window_mode == "random" and (truncated or sequence_end_frame < total_frames)
            include_window_context = cached_source_excerpt or prefix_truncated_excerpt or random_excerpt
            prompt_lyrics = lyrics
            if include_window_context:
                prompt_lyrics = example.get("lyrics_window") if isinstance(example.get("lyrics_window"), str) else ""
            prompt_text = self._lm_prompt_text(
                caption,
                prompt_lyrics,
                window_start_frame=sequence_start_frame,
                window_frame_count=int(codes.shape[0]) if include_window_context else None,
                total_frame_count=total_frames,
                window_mode=window_mode,
            )
            input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].squeeze(0)
            if input_ids.shape[0] > _MAX_PROMPT_TOKENS:
                raise ValueError(
                    f"The assembled MiniMax Music 3 prompt has {input_ids.shape[0]} tokens; "
                    f"the maximum is {_MAX_PROMPT_TOKENS}."
                )
            input_id_rows.append(input_ids)
            code_rows.append(codes)
            prompt_lengths.append(int(input_ids.shape[0]))
            audio_lengths.append(int(codes.shape[0]))
            audio_window_start_frames.append(absolute_start_frame)
            audio_loss_start_frames.append(loss_start_frame)
            audio_total_frames.append(total_frames)
            has_audio_end.append((not truncated) and source_crop_is_terminal)
            prompts.append(caption)

        input_ids = pad_sequence(input_id_rows, batch_first=True, padding_value=0)
        audio_codes = pad_sequence(code_rows, batch_first=True, padding_value=0)
        return {
            "input_ids": input_ids,
            "audio_codes": audio_codes,
            "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
            "audio_lengths": torch.tensor(audio_lengths, dtype=torch.long),
            "audio_window_start_frames": torch.tensor(audio_window_start_frames, dtype=torch.long),
            "audio_loss_start_frames": torch.tensor(audio_loss_start_frames, dtype=torch.long),
            "audio_total_frames": torch.tensor(audio_total_frames, dtype=torch.long),
            "has_audio_end": torch.tensor(has_audio_end, dtype=torch.bool),
            "prompts": prompts,
        }

    def _lm_frame_embeds(self, codes: torch.Tensor, language_model=None) -> torch.Tensor:
        # codes: [frames, codebooks] raw per-book indices -> [frames, hidden] audio-frame input embeddings.
        if language_model is None:
            language_model = self.unwrap_model(self.model)
        embed_tokens = language_model.get_input_embeddings()
        depth = self.rvq_depth_decoder
        num_codebooks = int(depth.config.num_codebooks)
        audio_vocab = int(depth.config.audio_vocab_size)
        semantic = embed_tokens(codes[:, 0] + _AUDIO_CODE_OFFSET)
        offsets = (torch.arange(num_codebooks - 1, device=codes.device) * audio_vocab).unsqueeze(0)
        residual = depth.audio_embeddings(codes[:, 1:] + offsets).sum(dim=1)
        return (semantic + residual.to(semantic.dtype)) * num_codebooks**-0.5

    def _lm_xm_route_enabled(self) -> bool:
        xm_config = getattr(self, "xm_config", None)
        return bool(
            self._train_language_model
            and xm_config is not None
            and xm_config.enabled
            and xm_config.training_target == "route"
        )

    def _lm_hidden_size(self, language_model=None) -> int:
        if language_model is None:
            language_model = self.unwrap_model(self.model)
        config = getattr(language_model, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is not None:
            return int(hidden_size)
        return int(language_model.get_input_embeddings().embedding_dim)

    @staticmethod
    def _unwrap_lm_route_embedding(module) -> nn.Embedding:
        if isinstance(module, nn.Embedding):
            return module
        original = getattr(module, "original_module", None)
        if isinstance(original, nn.Embedding):
            return original
        modules_to_save = getattr(module, "modules_to_save", None)
        if isinstance(modules_to_save, nn.ModuleDict):
            for saved in modules_to_save.values():
                if isinstance(saved, nn.Embedding):
                    return saved
        raise TypeError(f"MiniMax Music 3 XM route module must wrap an nn.Embedding, got {type(module)}.")

    def _ensure_lm_xm_route_embeddings(self, language_model=None) -> nn.Module:
        if not self._lm_xm_route_enabled():
            raise ValueError("MiniMax Music 3 XM route embeddings were requested while XM route training is disabled.")
        if language_model is None:
            language_model = self.unwrap_model(self.model)
        if language_model is None:
            raise ValueError("MiniMax Music 3 XM route embeddings require a loaded language model.")

        candidate_count = int(self.xm_config.candidate_count)
        hidden_size = self._lm_hidden_size(language_model)
        existing = getattr(language_model, self.XM_ROUTE_EMBEDDING_MODULE_NAME, None)
        if existing is not None:
            unwrapped = self._unwrap_lm_route_embedding(existing)
            if unwrapped.num_embeddings != candidate_count or unwrapped.embedding_dim != hidden_size:
                raise ValueError(
                    f"{self.XM_ROUTE_EMBEDDING_MODULE_NAME} has shape "
                    f"({unwrapped.num_embeddings}, {unwrapped.embedding_dim}), expected "
                    f"({candidate_count}, {hidden_size})."
                )
            return existing

        route_embeddings = nn.Embedding(candidate_count, hidden_size)
        initializer_range = float(getattr(getattr(language_model, "config", None), "initializer_range", 0.02) or 0.02)
        nn.init.normal_(route_embeddings.weight, mean=0.0, std=initializer_range)
        route_embeddings.to(
            device=language_model.get_input_embeddings().weight.device,
            dtype=language_model.get_input_embeddings().weight.dtype,
        )
        setattr(language_model, self.XM_ROUTE_EMBEDDING_MODULE_NAME, route_embeddings)
        return route_embeddings

    def _lm_supervised_targets(self, prepared_batch: dict, *, seq_len: int, device: torch.device) -> torch.Tensor:
        audio_codes = prepared_batch["audio_codes"]
        prompt_lengths = prepared_batch["prompt_lengths"]
        audio_lengths = prepared_batch["audio_lengths"]
        audio_loss_start_frames = prepared_batch.get("audio_loss_start_frames")
        has_audio_end = prepared_batch["has_audio_end"]
        targets = torch.full((prompt_lengths.shape[0], seq_len), -100, dtype=torch.long, device=device)
        for index in range(prompt_lengths.shape[0]):
            prompt_len = int(prompt_lengths[index])
            audio_len = int(audio_lengths[index])
            loss_start_frame = int(audio_loss_start_frames[index]) if audio_loss_start_frames is not None else 0
            if audio_len == 0:
                raise ValueError("MiniMax Music 3 language model training received a sample with zero audio frames.")
            if not 0 <= loss_start_frame < audio_len:
                raise ValueError(
                    "MiniMax Music 3 LM continuation loss start must be within the audio sequence; "
                    f"received {loss_start_frame} for {audio_len} frames."
                )
            start = prompt_len - 1 + loss_start_frame
            end = prompt_len - 1 + audio_len
            if end > seq_len:
                raise ValueError(f"MiniMax Music 3 supervised audio span [{start}, {end}) exceeds logits length {seq_len}.")
            targets[index, start:end] = (
                audio_codes[index, loss_start_frame:audio_len, 0].to(device=device) + _AUDIO_CODE_OFFSET
            )
            if bool(has_audio_end[index]):
                if end >= seq_len:
                    raise ValueError(f"MiniMax Music 3 audio-end target position {end} exceeds logits length {seq_len}.")
                targets[index, end] = _AUDIO_END_TOKEN_ID
        return targets

    def _lm_supervised_position_mask(self, prepared_batch: dict, *, seq_len: int, device: torch.device) -> torch.Tensor:
        targets = self._lm_supervised_targets(prepared_batch, seq_len=seq_len, device=device)
        return targets.ne(-100)

    @staticmethod
    def _lm_pack_supervised_hidden_states(hidden_states: torch.Tensor, supervised_mask: torch.Tensor) -> torch.Tensor:
        counts = supervised_mask.sum(dim=1)
        min_count = int(counts.min().item()) if counts.numel() else 0
        if min_count < 2:
            raise ValueError("NextLat for MiniMax Music 3 language model training requires at least two supervised tokens.")
        packed = [hidden_states[index, supervised_mask[index]][:min_count] for index in range(hidden_states.shape[0])]
        return torch.stack(packed, dim=0)

    def _lm_capture_hidden_states(self, outputs, hidden_states_buffer, supervised_mask: torch.Tensor) -> None:
        if hidden_states_buffer is None:
            return
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("NextLat is enabled, but Qwen did not return hidden states for MiniMax Music 3 LM training.")
        capture_layers = getattr(hidden_states_buffer, "capture_layers", None)
        for block_idx, block_hidden in enumerate(hidden_states[1:]):
            if capture_layers is not None and block_idx not in capture_layers:
                continue
            hidden_states_buffer[f"layer_{block_idx}"] = self._lm_pack_supervised_hidden_states(
                block_hidden,
                supervised_mask,
            )

    def _lm_apply_xm_routes(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        supervised_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        route_embeddings = self._ensure_lm_xm_route_embeddings()
        candidate_count = int(self.xm_config.candidate_count)
        batch_size = inputs_embeds.shape[0]
        inputs_embeds = inputs_embeds.repeat((candidate_count, 1, 1))
        attention_mask = attention_mask.repeat((candidate_count, 1))
        supervised_mask = supervised_mask.repeat((candidate_count, 1))
        candidate_ids = torch.arange(candidate_count, device=inputs_embeds.device).repeat_interleave(batch_size)
        routes = route_embeddings(candidate_ids).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        inputs_embeds = inputs_embeds + supervised_mask.unsqueeze(-1).to(dtype=inputs_embeds.dtype) * routes[:, None, :]
        return inputs_embeds, attention_mask, supervised_mask

    def _lm_select_xm_hidden_states(self, hidden_states_buffer, winner_indices: torch.Tensor) -> None:
        if hidden_states_buffer is None:
            return
        candidate_count = int(self.xm_config.candidate_count)
        for key, value in list(hidden_states_buffer.items()):
            if torch.is_tensor(value) and value.shape[0] % candidate_count == 0:
                hidden_states_buffer[key] = select_winning_candidates(value, winner_indices, candidate_count)

    def _lm_predict(self, prepared_batch: dict, *, apply_xm_routes: bool = True) -> Dict[str, object]:
        language_model = self.model
        embed_tokens = self.unwrap_model(language_model).get_input_embeddings()
        input_ids = prepared_batch["input_ids"]
        if next(self.rvq_depth_decoder.parameters()).device != input_ids.device:
            self.rvq_depth_decoder.to(input_ids.device)
        audio_codes = prepared_batch["audio_codes"]
        prompt_lengths = prepared_batch["prompt_lengths"]
        audio_lengths = prepared_batch["audio_lengths"]
        sequences = []
        for index in range(input_ids.shape[0]):
            prompt_len = int(prompt_lengths[index])
            audio_len = int(audio_lengths[index])
            prompt_embeds = embed_tokens(input_ids[index, :prompt_len])
            frame_embeds = self._lm_frame_embeds(audio_codes[index, :audio_len])
            sequences.append(torch.cat((prompt_embeds, frame_embeds.to(prompt_embeds.dtype)), dim=0))
        lengths = torch.tensor([seq.shape[0] for seq in sequences], device=input_ids.device)
        inputs_embeds = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        attention_mask = torch.arange(inputs_embeds.shape[1], device=input_ids.device)[None, :] < lengths[:, None]
        hidden_states_buffer = self._new_hidden_state_buffer()
        supervised_mask = self._lm_supervised_position_mask(
            prepared_batch,
            seq_len=inputs_embeds.shape[1],
            device=input_ids.device,
        )
        if apply_xm_routes and self._lm_xm_route_enabled():
            inputs_embeds, attention_mask, supervised_mask = self._lm_apply_xm_routes(
                inputs_embeds,
                attention_mask,
                supervised_mask,
            )
        outputs = language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask.to(dtype=torch.long),
            output_hidden_states=hidden_states_buffer is not None,
        )
        self._lm_capture_hidden_states(outputs, hidden_states_buffer, supervised_mask)
        return {"logits": outputs.logits, "hidden_states_buffer": hidden_states_buffer}

    @staticmethod
    def _lm_xm_teacher_candidate_losses(
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        supervised_mask: torch.Tensor,
        *,
        candidate_count: int,
        block_size: int,
    ) -> torch.Tensor:
        batch_size, seq_len = teacher_logits.shape[:2]
        expected_shape = (candidate_count * batch_size, seq_len, teacher_logits.shape[-1])
        if tuple(logits.shape) != expected_shape:
            raise ValueError(f"XM student logits must have shape {expected_shape}, got {tuple(logits.shape)}.")
        if tuple(supervised_mask.shape) != (batch_size, seq_len):
            raise ValueError(
                f"XM supervised mask must have shape {(batch_size, seq_len)}, got {tuple(supervised_mask.shape)}."
            )

        candidates = logits.reshape(candidate_count, batch_size, seq_len, logits.shape[-1])
        token_losses = torch.empty(
            (candidate_count, batch_size, seq_len),
            device=logits.device,
            dtype=torch.float32,
        )
        for start in range(0, seq_len, 64):
            end = min(start + 64, seq_len)
            with torch.no_grad():
                teacher_piece = teacher_logits[:, start:end].float()
                top_probs, top_indices = teacher_piece.softmax(dim=-1).topk(64, dim=-1)
                top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
            student_piece = candidates[:, :, start:end].float()
            gather_indices = top_indices.unsqueeze(0).expand(candidate_count, -1, -1, -1)
            student_top = student_piece.gather(-1, gather_indices)
            log_normalizer = student_piece.logsumexp(dim=-1, keepdim=True)
            token_losses[:, :, start:end] = -(top_probs.unsqueeze(0) * (student_top - log_normalizer)).sum(dim=-1)

        valid = supervised_mask.unsqueeze(0).expand(candidate_count, -1, -1)
        if block_size <= 0:
            denominator = valid.sum(dim=-1).clamp_min(1)
            return (token_losses * valid).sum(dim=-1) / denominator

        pad = (-seq_len) % block_size
        if pad:
            token_losses = F.pad(token_losses, (0, pad))
            valid = F.pad(valid, (0, pad))
        block_losses = token_losses.reshape(candidate_count, batch_size, -1, block_size)
        block_valid = valid.reshape(candidate_count, batch_size, -1, block_size)
        block_denominator = block_valid.sum(dim=-1).clamp_min(1)
        reduced_blocks = (block_losses * block_valid).sum(dim=-1) / block_denominator
        has_block = block_valid.any(dim=-1)
        sample_denominator = has_block.sum(dim=-1).clamp_min(1)
        return (reduced_blocks * has_block).sum(dim=-1) / sample_denominator

    @contextmanager
    def _lm_adapters_disabled(self):
        language_model = self.unwrap_model(self.model)
        language_model.disable_adapters()
        try:
            yield
        finally:
            language_model.enable_adapters()

    def loss(self, prepared_batch: dict, model_output, apply_conditioning_mask: bool = True):
        if not self._train_language_model:
            return super().loss(prepared_batch, model_output, apply_conditioning_mask)
        del apply_conditioning_mask
        logits = model_output["logits"]
        base_targets = self._lm_supervised_targets(prepared_batch, seq_len=logits.shape[1], device=logits.device)
        xm_route_active = self._lm_xm_route_enabled()
        teacher_logits = None
        if (
            prepared_batch.get("is_regularisation_data")
            and str(getattr(self.config, "model_type", "lora")).lower() == "lora"
        ):
            # Prior preservation: on regularisation batches the target is the frozen base model's own
            # next-token distribution, so unrelated songs keep predicting as they would without the LoRA.
            with torch.no_grad(), self._lm_adapters_disabled():
                teacher_logits = self._lm_predict(prepared_batch, apply_xm_routes=False)["logits"]

        if xm_route_active:
            candidate_count = int(self.xm_config.candidate_count)
            if logits.shape[0] != base_targets.shape[0] * candidate_count:
                raise ValueError(
                    f"MiniMax Music 3 XM expected {base_targets.shape[0] * candidate_count} candidate logits rows, "
                    f"got {logits.shape[0]}."
                )
            block_size = self.xm_config.block_size if self.xm_config.selection_scope == "block" else 0
            if teacher_logits is None:
                targets = base_targets.repeat((candidate_count, 1))
                per_candidate = blockwise_cross_entropy(logits, targets, block_size=block_size)
                candidate_losses = per_candidate.reshape(candidate_count, base_targets.shape[0])
            else:
                candidate_losses = self._lm_xm_teacher_candidate_losses(
                    logits,
                    teacher_logits,
                    base_targets.ne(-100),
                    candidate_count=candidate_count,
                    block_size=block_size,
                )
            selected_loss, winner_indices = select_min_candidate_loss(candidate_losses)
            model_output["xm_winner_indices"] = winner_indices.detach()
            usage = route_usage_histogram(winner_indices.detach(), candidate_count)
            if usage is not None:
                model_output["xm_route_usage"] = usage.to(device=logits.device)
            self._lm_select_xm_hidden_states(model_output.get("hidden_states_buffer"), winner_indices)
            return selected_loss

        # Chunked losses: upcasting the full-vocab logits at once costs several GiB at long sequence lengths.
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = base_targets.reshape(-1)
        total = flat_logits.new_zeros((), dtype=torch.float32)
        count = 0
        chunk = 512 if teacher_logits is not None else 1024
        flat_teacher = teacher_logits.reshape(-1, teacher_logits.shape[-1]) if teacher_logits is not None else None
        for start in range(0, flat_logits.shape[0], chunk):
            piece_targets = flat_targets[start : start + chunk]
            mask = piece_targets != -100
            if not mask.any():
                continue
            piece = flat_logits[start : start + chunk][mask].float()
            if flat_teacher is not None:
                # Top-K soft targets: gathering the student's logits at the teacher's top tokens keeps the
                # per-position autograd footprint at O(K) instead of O(vocab).
                with torch.no_grad():
                    teacher_piece = flat_teacher[start : start + chunk][mask].float()
                    top_probs, top_indices = teacher_piece.softmax(dim=-1).topk(64, dim=-1)
                    top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
                student_top = piece.gather(1, top_indices)
                log_normalizer = piece.logsumexp(dim=-1, keepdim=True)
                total = total - (top_probs * (student_top - log_normalizer)).sum()
            else:
                total = total + F.cross_entropy(piece, piece_targets[mask], reduction="sum")
            count += int(mask.sum())
        if count == 0:
            raise ValueError("MiniMax Music 3 language model loss found no supervised positions.")
        return total / count

    def loss_with_logs(self, prepared_batch: dict, model_output, apply_conditioning_mask: bool = True):
        loss = self.loss(prepared_batch, model_output, apply_conditioning_mask=apply_conditioning_mask)
        if not self._lm_xm_route_enabled():
            return loss, None
        logs = {}
        usage = model_output.get("xm_route_usage")
        if torch.is_tensor(usage):
            for index, value in enumerate(usage.detach().float().cpu().tolist()):
                logs[f"xm_route_{index}_usage"] = value
        return loss, logs or None

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
        if getattr(self.config, "minimax_music_lm_adapter", None):
            self._apply_lm_precache_adapter(language_model)
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

    def move_text_encoders_for_vae_cache(self, target_device):
        components = (
            ("language_model", self.language_model),
            ("rvq_depth_decoder", self.rvq_depth_decoder),
            ("condition_encoder", self.condition_encoder),
        )
        target = torch.device(target_device)
        for component_name, component in components:
            if component is None:
                continue
            if component_name == "language_model" and self._ramtorch_text_encoders_requested():
                logger.debug("Skipping %s VAE-cache move because ramtorch_text_encoder is enabled.", component_name)
                continue
            if target.type == "cpu":
                component.to(target)
            else:
                component.to(target, dtype=self.config.weight_dtype)
        if self.language_model is not None:
            self.text_encoders = [self.language_model]
            self.text_encoder_1 = self.language_model

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

    def _apply_lm_precache_adapter(self, language_model) -> None:
        from peft import LoraConfig
        from safetensors.torch import load_file

        adapter_path = str(self.config.minimax_music_lm_adapter)
        strength = float(getattr(self.config, "minimax_music_lm_adapter_strength", 1.0) or 1.0)
        state = load_file(adapter_path)
        ranks = {value.shape[0] for key, value in state.items() if key.endswith(".lora_A.weight")}
        if len(ranks) != 1:
            raise ValueError(f"LM adapter {adapter_path} has mixed or missing LoRA ranks: {sorted(ranks)}")
        rank = ranks.pop()
        target_modules = sorted({key.split(".")[-3] for key in state if key.endswith(".lora_A.weight")})
        language_model.add_adapter(LoraConfig(r=rank, lora_alpha=rank, target_modules=target_modules))
        mapped = {}
        for key, value in state.items():
            new_key = (
                key.removeprefix("language_model.")
                .replace(".lora_A.weight", ".lora_A.default.weight")
                .replace(".lora_B.weight", ".lora_B.default.weight")
            )
            tensor = value.to(self.config.weight_dtype)
            if ".lora_B." in new_key:
                tensor = tensor * strength
            mapped[new_key] = tensor
        _missing, unexpected = language_model.load_state_dict(mapped, strict=False)
        if unexpected:
            raise ValueError(f"LM adapter {adapter_path} carries unknown keys: {sorted(unexpected)[:4]}")
        lora_names = [name for name, _ in language_model.named_parameters() if "lora_" in name]
        loaded = sum(1 for name in lora_names if name in mapped)
        if loaded != len(lora_names):
            raise ValueError(f"LM adapter {adapter_path} loaded {loaded}/{len(lora_names)} LoRA tensors.")
        logger.info("Applied LM precache adapter %s (%d tensors, strength %.2f).", adapter_path, loaded, strength)

    @torch.no_grad()
    def _teacher_forced_depth_hiddens(self, lm_hidden: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        # Teacher-forced prefix pass through the causal depth decoder reproduces the per-book hiddens the
        # rollout collects one step at a time. lm_hidden: [frames, dim], codes: [frames, codebooks] raw.
        depth = self.rvq_depth_decoder
        num_codebooks = int(depth.config.num_codebooks)
        audio_vocab = int(depth.config.audio_vocab_size)
        sequence = [depth.projection(lm_hidden).unsqueeze(1)]
        semantic_embed = self.language_model.model.embed_tokens(codes[:, 0] + _AUDIO_CODE_OFFSET)
        sequence.append(depth.projection(semantic_embed).unsqueeze(1))
        for index in range(1, num_codebooks - 1):
            embed = depth.audio_embeddings(codes[:, index] + (index - 1) * audio_vocab)
            sequence.append(depth.projection(embed).unsqueeze(1))
        hidden = depth(torch.cat(sequence, dim=1))
        return hidden[:, 1:num_codebooks, :].reshape(codes.shape[0], -1)

    @torch.no_grad()
    def _encode_teacher_forced_prompt(self, prompt: str, context: dict, mode: str) -> Dict[str, torch.Tensor]:
        tokenizer = self.tokenizers[0]
        language_model = self.language_model
        codes, _boundary_metadata = self._lm_load_audio_codes(context)
        codes = codes.to(self.accelerator.device)
        window_seconds = self._audio_duration_for_context(context)
        window_frames = max(1, int(round(window_seconds * self._frame_rate())))
        codes = codes[: min(window_frames, _MAX_AUDIO_FRAMES)]
        if mode == "audio+text":
            prompt_text = self._lm_prompt_text(prompt, self._lyrics_for_context(context, str(prompt)))
        else:
            prompt_text = "<|audio_start|>"
        input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to(self.accelerator.device)
        if input_ids.shape[1] > _MAX_PROMPT_TOKENS:
            raise ValueError(f"The assembled MiniMax Music 3 prompt has {input_ids.shape[1]} tokens.")
        prompt_embeds = language_model.model.embed_tokens(input_ids[0])
        frame_embeds = self._lm_frame_embeds(codes, language_model=language_model)
        sequence = torch.cat([prompt_embeds, frame_embeds.to(prompt_embeds.dtype)], dim=0).unsqueeze(0)
        hidden = language_model.model(inputs_embeds=sequence).last_hidden_state[0]
        prompt_len = input_ids.shape[1]
        predictor_hidden = hidden[prompt_len - 1 : prompt_len - 1 + codes.shape[0]]
        depth_hidden = self._teacher_forced_depth_hiddens(predictor_hidden, codes)
        frame_hiddens = torch.cat([predictor_hidden, depth_hidden.to(predictor_hidden.dtype)], dim=-1)
        return {"prompt_embeds": frame_hiddens.unsqueeze(0).to(dtype=self.config.weight_dtype)}

    @torch.no_grad()
    def _encode_single_prompt(self, prompt: str, context: dict) -> Dict[str, torch.Tensor]:
        self.load_text_encoder(move_to_device=True)
        tokenizer = self.tokenizers[0]
        language_model = self.language_model
        rvq_depth_decoder = self.rvq_depth_decoder
        if language_model is None or rvq_depth_decoder is None:
            raise ValueError("MiniMax Music 3 text components are not loaded.")
        precache_mode = str(getattr(self.config, "minimax_music_lm_precache_mode", "text-only") or "text-only")
        if precache_mode in ("audio-only", "audio+text"):
            return self._encode_teacher_forced_prompt(str(prompt), context, precache_mode)

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
        if self._mixflow_enabled():
            return super().sample_flow_sigmas(batch=batch, state=state)
        bsz = batch["latents"].shape[0]
        device = self.accelerator.device
        dtype = getattr(self.config, "weight_dtype", torch.float32)
        mean = float(getattr(self.config, "logit_mean", 0.0) or 0.0)
        std = float(getattr(self.config, "logit_std", 1.0) or 1.0)
        if self._uses_flow_cubic_schedule():
            u = self._sample_flow_cubic_values(bsz, device)
        else:
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
        if self._train_language_model:
            device = self.accelerator.device
            for key in ("input_ids", "audio_codes", "prompt_lengths", "audio_lengths", "has_audio_end"):
                value = batch.get(key)
                if value is None:
                    raise ValueError(f"MiniMax Music 3 language model batch is missing {key}.")
                batch[key] = value.to(device=device)
            loss_starts = batch.get("audio_loss_start_frames")
            if loss_starts is None:
                loss_starts = torch.zeros_like(batch["audio_lengths"])
            batch["audio_loss_start_frames"] = loss_starts.to(device=device)
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
            self._prepare_flow_noisy_latents(batch)
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
        if self._train_language_model:
            return self._lm_predict(prepared_batch)
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
        if self._train_language_model:
            raise NotImplementedError(
                "MiniMax Music 3 language model training does not build an inference pipeline yet; "
                "merge the LoRA into language_model/ and use the standard generation stack."
            )
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
