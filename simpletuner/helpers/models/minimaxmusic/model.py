# Copyright 2026 The MiniMax Team and The HuggingFace Team.
# Modifications for SimpleTuner are distributed under the AGPL-3.0-or-later.

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.guiders import ClassifierFreeGuidance
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
from simpletuner.helpers.models.common import AudioModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
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
from simpletuner.helpers.models.minimaxmusic.vocoder import MiniMaxMusic3Vocoder
from simpletuner.helpers.models.registry import ModelRegistry

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
    AUTOENCODER_CLASS = MiniMaxMusic3Vocoder
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

    @classmethod
    def register_config_requirements(cls):
        rules = [
            ConfigRule(
                field_name="dataset_type",
                rule_type=RuleType.CUSTOM,
                value=None,
                message="MiniMax Music 3 expects audio datasets with precomputed Flow-VAE latents.",
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
                    suggestion="Set dataset_type: audio and provide cached Flow-VAE latents.",
                )
            ]
        return []

    def supports_crepa_self_flow(self) -> bool:
        return True

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
        return pretrained_load_args

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

    def get_lora_target_layers(self):
        manual_targets = self._get_peft_lora_target_modules()
        if manual_targets:
            return manual_targets
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

    def load_vae(self, move_to_device: bool = True):
        if self.vae is None:
            self.vae = MiniMaxMusic3Vocoder.from_pretrained(
                self.config.pretrained_vae_model_name_or_path or self._checkpoint_path(),
                subfolder="vocoder",
                torch_dtype=self.config.weight_dtype,
            )
            self.vae.requires_grad_(False)
        if move_to_device and self.vae is not None:
            self.vae.to(self.accelerator.device, dtype=self.config.weight_dtype)
        return self.vae

    def encode_cache_batch(self, vae, samples, metadata_entries: Optional[list] = None):
        raise NotImplementedError(
            "MiniMax Music 3 support currently requires precomputed Flow-VAE latents; the public Diffusers PR "
            "includes the decoder/vocoder but not an audio encoder."
        )

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
        if self.language_model is not None and self.rvq_depth_decoder is not None and self.condition_encoder is not None:
            return
        self.load_text_tokenizer()
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
        condition_encoder = MiniMaxMusic3ConditionEncoder.from_pretrained(
            base_path,
            subfolder="condition_encoder",
            torch_dtype=self.config.weight_dtype,
        )
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

    def _audio_duration_for_context(self, context: dict) -> float:
        for key in ("audio_duration", "duration", "duration_seconds"):
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
            self.load_text_encoder(move_to_device=True)
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
        if batch.get("latent_batch") is None:
            raise ValueError("MiniMax Music 3 training requires cached Flow-VAE latents in latent_batch.")
        device = self.accelerator.device
        dtype = self.config.weight_dtype
        latents = batch["latent_batch"].to(device=device, dtype=dtype)
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
        r_timestep = r_timestep.to(device=timestep.device, dtype=timestep.dtype)
        if r_timestep.numel() > 0 and torch.max(r_timestep.detach().abs()) > 1.0:
            r_timestep = (r_timestep / 1000.0).clamp(0.0, 1.0)
        return r_timestep

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
        if self.language_model is None or self.rvq_depth_decoder is None or self.condition_encoder is None:
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
        if "lyrics" not in pipeline_kwargs:
            configured_lyrics = getattr(self.config, "validation_lyrics", None)
            if configured_lyrics:
                pipeline_kwargs["lyrics"] = configured_lyrics
        return pipeline_kwargs


ModelRegistry.register("minimaxmusic", MiniMaxMusic)
