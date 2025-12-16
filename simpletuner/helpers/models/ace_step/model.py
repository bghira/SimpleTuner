# NOTE: This file originates from the ACE-Step project (Apache-2.0).
#       Modifications for SimpleTuner are © 2024 SimpleTuner contributors
#       and distributed under the AGPL-3.0-or-later.

"""
ACEStep audio model integration for SimpleTuner.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import snapshot_download
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer, UMT5EncoderModel

from simpletuner.helpers.configuration.registry import (
    ConfigRegistry,
    ConfigRule,
    RuleType,
    ValidationResult,
    make_default_rule,
)
from simpletuner.helpers.models.ace_step.language_segmentation import LangSegment, language_filters
from simpletuner.helpers.models.ace_step.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer
from simpletuner.helpers.models.ace_step.music_dcae.music_dcae_pipeline import MusicDCAE
from simpletuner.helpers.models.ace_step.pipeline import SUPPORT_LANGUAGES, ACEStepPipeline, structure_pattern
from simpletuner.helpers.models.ace_step.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from simpletuner.helpers.models.ace_step.transformer import ACEStepTransformer2DModel
from simpletuner.helpers.models.common import (
    AudioModelFoundation,
    ModelTypes,
    PipelineTypes,
    PredictionTypes,
    get_model_config_path,
)
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)


class ACEStep(AudioModelFoundation):
    NAME = "ACE-Step"
    MODEL_DESCRIPTION = "Audio generation foundation model (ACE-Step v1 3.5B)"
    ENABLED_IN_WIZARD = True
    MODEL_LICENSE = "apache-2.0"
    MODEL_TYPE = ModelTypes.TRANSFORMER
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_CLASS = ACEStepTransformer2DModel
    MODEL_SUBFOLDER = "ace_step_transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2AUDIO: ACEStepPipeline,
        # Provide TEXT2IMG alias to satisfy callers expecting image pipelines in hooks.
        PipelineTypes.TEXT2IMG: ACEStepPipeline,
    }
    DEFAULT_PIPELINE_TYPE = PipelineTypes.TEXT2AUDIO
    AUTOENCODER_CLASS = MusicDCAE
    LATENT_CHANNEL_COUNT = 8
    DEFAULT_MODEL_FLAVOUR = "base"
    HUGGINGFACE_PATHS = {
        "base": "ACE-Step/ACE-Step-v1-3.5B",
    }
    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "UMT5 Encoder",
            "tokenizer": AutoTokenizer,
            "tokenizer_subfolder": "umt5-base",
            "model": UMT5EncoderModel,
            "subfolder": "umt5-base",
        }
    }
    SPEAKER_EMBED_DIM = 512
    SSL_ENCODER_NAMES = ["mert", "m-hubert"]
    SSL_LATENT_DIMS = [1024, 768]
    DEFAULT_LORA_TARGET = [
        "linear_q",
        "linear_k",
        "linear_v",
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
    ]
    SUPPORTS_LYRICS_EMBEDDER_TRAINING = True

    def __init__(self, config: dict, accelerator):
        super().__init__(config, accelerator)
        self.text_tokenizer_max_length = getattr(self.config, "tokenizer_max_length", 256)
        self.mert_model = None
        self.hubert_model = None
        self.resampler_mert = None
        self.resampler_mhubert = None
        self._ssl_models_ready = False
        self._ssl_device = accelerator.device if accelerator else torch.device("cpu")
        self.controlnet = None
        self.lyric_tokenizer = VoiceBpeTokenizer()
        self.lang_segment = LangSegment()
        self.lang_segment.setfilters(language_filters.default)
        self.tokenizers = []
        self.text_encoders = []
        self._checkpoint_base: Optional[str] = None

    def get_lora_target_layers(self):
        if getattr(self.config, "slider_lora_target", False) and self.config.lora_type.lower() == "standard":
            return getattr(self, "SLIDER_LORA_TARGET", None) or self.DEFAULT_SLIDER_LORA_TARGET
        if getattr(self.config, "controlnet", False):
            return self.DEFAULT_CONTROLNET_LORA_TARGET

        target_option = getattr(self.config, "acestep_lora_target", "attn_qkv+linear_qkv")

        if target_option == "attn_qkv":
            return ["to_q", "to_k", "to_v", "to_out.0"]
        elif target_option == "attn_qkv+linear_qkv":
            return [
                "linear_q",
                "linear_k",
                "linear_v",
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
            ]
        elif target_option == "attn_qkv+linear_qkv+speech_embedder":
            return [
                "speaker_embedder",
                "linear_q",
                "linear_k",
                "linear_v",
                "to_q",
                "to_k",
                "to_v",
                "to_out.0",
            ]

        return self.DEFAULT_LORA_TARGET

    def setup_training_noise_schedule(self):
        """
        ACE-Step ships its own flow-matching scheduler; avoid diffusers hub lookups.
        """
        shift = getattr(self.config, "flow_schedule_shift", None)
        if shift is None:
            shift = 3.0

        self.noise_schedule = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=shift,
        )
        return self.config, self.noise_schedule

    def _find_cached_snapshot(self, repo_id: str) -> Optional[str]:
        """
        Try to locate an existing Hugging Face snapshot on disk for the given repo.
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

    def _resolve_checkpoint_base(self) -> str:
        if self._checkpoint_base and os.path.exists(self._checkpoint_base):
            return self._checkpoint_base
        base = get_model_config_path(self.config.model_family, self.config.pretrained_model_name_or_path)
        cached = self._find_cached_snapshot(base)
        if cached:
            self._checkpoint_base = cached
            return self._checkpoint_base
        if os.path.exists(base):
            self._checkpoint_base = base
            return self._checkpoint_base
        logger.info("Downloading ACE-Step assets for %s", base)
        self._checkpoint_base = snapshot_download(
            repo_id=base,
            allow_patterns=[
                "music_dcae_f8c8/*",
                "music_vocoder/*",
                "ace_step_transformer/*",
                "umt5-base/*",
                "ace_step_transformer/*",
            ],
        )
        return self._checkpoint_base

    def load_text_tokenizer(self):
        base_path = self._resolve_checkpoint_base()
        logger.info("Loading ACE-Step tokenizer from %s (subfolder=umt5-base)", base_path)
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=base_path,
            subfolder="umt5-base",
            use_fast=True,
        )
        self.tokenizers = [tokenizer]
        self.tokenizer_1 = tokenizer

    def load_text_encoder(self, move_to_device: bool = True):
        if not self.tokenizers:
            self.load_text_tokenizer()
        base_path = self._resolve_checkpoint_base()
        logger.info("Loading ACE-Step text encoder from %s (subfolder=umt5-base)", base_path)
        text_encoder = UMT5EncoderModel.from_pretrained(
            pretrained_model_name_or_path=base_path,
            subfolder="umt5-base",
            torch_dtype=self.config.weight_dtype,
        )
        if move_to_device:
            text_encoder.to(self.accelerator.device, dtype=self.config.weight_dtype)

        self.text_encoders = [text_encoder]
        self.text_encoder_1 = text_encoder

    def load_vae(self, move_to_device: bool = True):
        """
        Load the ACE-Step DCAE/Vocoder bundle from the model checkpoint directory.
        The upstream weights live under `music_dcae_f8c8` and `music_vocoder` subfolders.
        """
        # Always resolve to the cached snapshot location; the repo root lacks a top-level config.json.
        base_path = self._resolve_checkpoint_base()
        self.config.vae_path = base_path
        self.config.pretrained_model_name_or_path = base_path
        logger.info("Loading ACE-Step MusicDCAE from %s (subfolders music_dcae_f8c8/music_vocoder)", base_path)
        self.vae = MusicDCAE(
            dcae_checkpoint_path=base_path,
            vocoder_checkpoint_path=base_path,
        )

    def scale_vae_latents_for_cache(self, latents, vae):
        # ACE-Step autoencoder outputs are already scaled; avoid double scaling.
        return latents

    def load_model(self, move_to_device: bool = True):
        """
        Override to ensure the transformer weights are loaded from the ACE-Step snapshot,
        not the repo root (which lacks a top-level config.json).
        """
        base_path = self._resolve_checkpoint_base()

        # If the user hasn't specified a specific transformer model path, or if they have specified the
        # upstream repo ID, we should default to the resolved snapshot path.
        current_transformer_path = getattr(self.config, "pretrained_transformer_model_name_or_path", None)
        if not current_transformer_path or not os.path.exists(current_transformer_path):
            self.config.pretrained_transformer_model_name_or_path = base_path

        # Force the common loader to use the resolved snapshot path.
        self.config.pretrained_model_name_or_path = base_path
        # Ensure we look inside the transformer subfolder
        self.config.pretrained_transformer_subfolder = self.MODEL_SUBFOLDER
        logger.info(
            "Loading ACE-Step transformer from %s (subfolder=%s)",
            self.config.pretrained_transformer_model_name_or_path,
            self.MODEL_SUBFOLDER,
        )
        return super().load_model(move_to_device=move_to_device)

    def encode_dropout_caption(self, positive_prompt_embeds: dict = None):
        """
        Encode a null caption for dropout using the ACE-Step text encoder/tokenizer.
        """
        encoded_text = self._encode_prompts([""], is_negative_prompt=False)
        return self._format_text_embedding(encoded_text)

    def get_pipeline(
        self,
        pipeline_type: str = PipelineTypes.TEXT2AUDIO,
        load_base_model: bool = True,
        cache_pipeline: bool = True,
    ):
        """
        Return the ACE-Step inference pipeline wired to the already-loaded components when available.
        """
        checkpoint_dir = getattr(self.config, "pretrained_model_name_or_path", None) or self._resolve_checkpoint_base()
        pipeline = ACEStepPipeline(checkpoint_dir=checkpoint_dir)
        # Wire in already loaded components to avoid reloading.
        pipeline.transformer = getattr(self, "model", None)
        pipeline.ace_step_transformer = getattr(self, "model", None)
        pipeline.music_dcae = getattr(self, "vae", None)
        pipeline.text_encoder_model = getattr(self, "text_encoder_1", None)
        pipeline.text_tokenizer = getattr(self, "tokenizer_1", None)
        pipeline.text_encoder = getattr(self, "text_encoder_1", None)
        pipeline.text_encoder_2 = getattr(self, "text_encoder_2", None)
        pipeline.text_encoder_3 = getattr(self, "text_encoder_3", None)
        pipeline.text_encoder_4 = getattr(self, "text_encoder_4", None)
        pipeline.lyric_tokenizer = getattr(self, "lyric_tokenizer", None)
        pipeline.lang_segment = getattr(self, "lang_segment", None)
        # Mark pipeline as loaded so it doesn't try to reload weights from disk
        pipeline.loaded = True
        return pipeline

    def get_lyrics_embedder_modules(self, unwrap: bool = True) -> list[tuple[str, torch.nn.Module]]:
        """
        Return the ACE-Step lyrics embedder components (embedding, encoder, projection).
        """
        component = self.model
        if component is None:
            return []
        if unwrap:
            component = self.unwrap_model(component)
        modules: list[tuple[str, torch.nn.Module]] = []
        for name in ("lyric_embs", "lyric_encoder", "lyric_proj"):
            module = getattr(component, name, None)
            if module is not None:
                modules.append((name, module))
        return modules

    @classmethod
    def caption_field_preferences(cls, dataset_type: Optional[str] = None) -> list[str]:
        if dataset_type and str(dataset_type).lower() == "audio":
            return ["prompt", "lyrics", "tags"]
        return []

    @classmethod
    def register_config_requirements(cls):
        rules = [
            make_default_rule(
                field_name="tokenizer_max_length",
                default_value=256,
                message="ACE-Step defaults to 256 token UMT5 context.",
            ),
            make_default_rule(
                field_name="ace_step_ssl_loss_weight",
                default_value=0.1,
                message="Projection-alignment losses default to weight 0.1 for ACE-Step (matches upstream ssl_coeff).",
            ),
            make_default_rule(
                field_name="max_grad_norm",
                default_value=1.0,
                message="ACE-Step defaults to grad-norm clipping at 1.0 to tame early training spikes.",
            ),
            ConfigRule(
                field_name="dataset_type",
                rule_type=RuleType.CUSTOM,
                value=None,
                message="ACE-Step expects audio datasets. Configure `dataset_type: audio` for training backends.",
                error_level="warning",
            ),
        ]
        ConfigRegistry.register_rules("ace_step", rules)
        ConfigRegistry.register_validator(
            "ace_step",
            cls._validate_audio_dataset_usage,
            "Validates ACE-Step specific dataset requirements.",
        )

    @staticmethod
    def _validate_audio_dataset_usage(config: dict) -> List[ValidationResult]:
        dataset_type = (config or {}).get("dataset_type")
        if dataset_type and str(dataset_type).lower() != "audio":
            return [
                ValidationResult(
                    passed=False,
                    field="dataset_type",
                    message="ACE-Step requires audio datasets. Update your data backend configuration.",
                    level="warning",
                    suggestion="Set dataset_type: audio for every training backend when using ACE-Step.",
                )
            ]
        return []

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False) -> Dict[str, torch.Tensor]:
        """
        Encode prompts with the UMT5 encoder used by ACE-Step.
        """
        if not isinstance(prompts, (list, tuple)):
            prompts = [prompts]
        tokenizer = self.tokenizers[0]
        text_encoder = self.text_encoders[0]
        tokenizer_kwargs = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt",
            "max_length": self.text_tokenizer_max_length,
        }
        text_inputs = tokenizer(prompts, **tokenizer_kwargs)
        text_inputs = {k: v.to(self.accelerator.device) for k, v in text_inputs.items()}
        if text_encoder.device != self.accelerator.device:
            text_encoder.to(self.accelerator.device, dtype=self.config.weight_dtype)
        text_encoder.eval()
        with torch.no_grad():
            encoder_outputs = text_encoder(**text_inputs)
        prompt_embeds = encoder_outputs.last_hidden_state.to(dtype=self.config.weight_dtype)
        attention_mask = text_inputs.get("attention_mask")
        return {
            "prompt_embeds": prompt_embeds,
            "attention_masks": attention_mask,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: Dict[str, torch.Tensor]) -> dict:
        return {
            "encoder_text_hidden_states": text_embedding["prompt_embeds"],
            "text_attention_mask": text_embedding.get("attention_masks"),
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: Dict[str, torch.Tensor]) -> dict:
        return {
            "negative_encoder_text_hidden_states": text_embedding["prompt_embeds"].unsqueeze(0),
            "negative_text_attention_mask": text_embedding.get("attention_masks"),
        }

    def collate_prompt_embeds(self, text_encoder_output: list[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate UMT5 embeddings so that:
        - prompt_embeds: (batch, seq, dim)
        - attention_masks: (batch, seq)
        """
        if not text_encoder_output:
            return {}

        def _norm_emb(t: torch.Tensor) -> torch.Tensor:
            # Expect [1, seq, dim] or [seq, dim]
            if t.dim() == 3 and t.shape[0] == 1:
                return t.squeeze(0)
            return t

        def _norm_mask(t: torch.Tensor) -> torch.Tensor:
            # Expect [1, seq] or [seq]
            if t.dim() == 2 and t.shape[0] == 1:
                return t.squeeze(0)
            return t

        embeds = [_norm_emb(e["prompt_embeds"]) for e in text_encoder_output]
        masks = [_norm_mask(e["attention_masks"]) for e in text_encoder_output if "attention_masks" in e]

        prompt_embeds = torch.stack(embeds, dim=0)
        attention_masks = torch.stack(masks, dim=0) if masks else None

        out: Dict[str, torch.Tensor] = {"prompt_embeds": prompt_embeds}
        if attention_masks is not None:
            out["attention_masks"] = attention_masks
        return out

    def _speaker_embedding_dim(self) -> int:
        component = getattr(self, "model", None)
        config = getattr(component, "config", None)
        if config is not None and getattr(config, "speaker_embedding_dim", None) is not None:
            return int(config.speaker_embedding_dim)
        return self.SPEAKER_EMBED_DIM

    def _build_audio_attention_mask(
        self,
        latents: torch.Tensor,
        latent_metadata: Optional[List[Dict]],
    ) -> torch.Tensor:
        """
        Create a per-token mask so the denoiser can ignore padded latent regions.
        """
        batch_size = latents.shape[0]
        seq_len = latents.shape[-1]
        mask = torch.ones(
            batch_size,
            seq_len,
            device=self.accelerator.device,
            dtype=self.config.weight_dtype,
        )
        if not latent_metadata:
            return mask

        def _infer_latent_length(meta: Dict, fallback: int) -> Optional[int]:
            # Prefer cached latent_lengths
            latent_lengths = meta.get("latent_lengths") or meta.get("latent_length")
            if latent_lengths is not None:
                return latent_lengths
            # Try deriving from raw sample lengths if present
            num_samples = meta.get("num_samples")
            if num_samples is None:
                return None
            try:
                sr = meta.get("sample_rate") or meta.get("sampling_rate") or meta.get("target_sampling_rate") or 48000
                stride = getattr(self.vae, "time_dimention_multiple", 8)
                return int(round((num_samples / sr) * 44100 / 512 / stride))
            except Exception:
                return None

        for idx, metadata in enumerate(latent_metadata):
            latent_lengths = None
            if isinstance(metadata, dict):
                latent_lengths = _infer_latent_length(metadata, seq_len)
            elif hasattr(metadata, "get"):
                try:
                    latent_lengths = _infer_latent_length(metadata, seq_len)
                except Exception:
                    latent_lengths = None
            if latent_lengths is None:
                continue
            if torch.is_tensor(latent_lengths):
                if latent_lengths.ndim == 0:
                    length_value = int(latent_lengths.item())
                elif latent_lengths.ndim == 1:
                    capped_idx = min(idx, latent_lengths.shape[0] - 1)
                    length_value = int(latent_lengths[capped_idx].item())
                else:
                    length_value = int(latent_lengths.flatten()[0].item())
            else:
                try:
                    length_value = int(latent_lengths)
                except (TypeError, ValueError):
                    continue
            length_value = max(0, min(seq_len, length_value))
            if length_value < seq_len:
                mask[idx, length_value:] = 0
        return mask

    def encode_cache_batch(self, vae, samples, metadata_entries: Optional[list] = None):
        """
        Use sample metadata (lyrics, lengths, etc.) while encoding audio with the VAE so
        cache entries can ship all ACE-Step specific conditioning data.
        """
        resolved_metadata = self._resolve_cache_metadata_entries(metadata_entries)
        result = super().encode_cache_batch(vae, samples, metadata_entries)
        payload = result
        ssl_payload = self._compute_ssl_embeddings_for_cache(samples, resolved_metadata)
        lyrics_payload = self._extract_lyrics_from_metadata(resolved_metadata)
        if ssl_payload is None and lyrics_payload is None:
            return result
        if not isinstance(payload, dict):
            payload = {"latents": payload}
        if ssl_payload is not None:
            payload["ssl_hidden_states"] = ssl_payload
        if lyrics_payload is not None:
            payload["lyrics"] = lyrics_payload
        return payload

    def _resolve_cache_metadata_entries(self, metadata_entries: Optional[list]) -> List[Dict]:
        if not metadata_entries:
            return []
        resolved = []
        for entry in metadata_entries:
            metadata = {}
            if isinstance(entry, dict):
                metadata = entry.get("metadata") or {}
                if not metadata:
                    filepath = entry.get("filepath")
                    backend_id = entry.get("data_backend_id")
                    if filepath is not None and backend_id is not None:
                        metadata = StateTracker.get_metadata_by_filepath(filepath, backend_id) or {}
            resolved.append(metadata or {})
        return resolved

    def _extract_lyrics_from_metadata(self, metadata_entries: List[Dict]):
        if not metadata_entries:
            return None
        lyrics = []
        has_lyrics = False
        for entry in metadata_entries:
            lyric_value = None
            if isinstance(entry, dict):
                lyric_value = entry.get("lyrics")
            lyrics.append(lyric_value)
            if lyric_value:
                has_lyrics = True
        if not has_lyrics:
            return None
        return lyrics

    def _ensure_ssl_models(self):
        if self._ssl_models_ready:
            return
        try:
            self.mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True).eval()
            self.mert_model.requires_grad_(False).to(self._ssl_device)
        except Exception as exc:
            logger.warning("Failed to load MERT SSL encoder: %s", exc)
            self.mert_model = None
        try:
            self.hubert_model = AutoModel.from_pretrained("utter-project/mHuBERT-147").eval()
            self.hubert_model.requires_grad_(False).to(self._ssl_device)
        except Exception as exc:
            logger.warning("Failed to load mHuBERT SSL encoder: %s", exc)
            self.hubert_model = None
        if self.mert_model is not None:
            self.resampler_mert = torchaudio.transforms.Resample(orig_freq=48000, new_freq=24000).to(self._ssl_device)
        if self.hubert_model is not None:
            self.resampler_mhubert = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000).to(self._ssl_device)
        self._ssl_models_ready = True

    def _infer_mert_ssl(self, target_wavs: torch.Tensor, wav_lengths: torch.Tensor):
        if self.mert_model is None or self.resampler_mert is None:
            return None
        target_wavs = target_wavs.to(device=self._ssl_device, dtype=torch.float32)
        wav_lengths = wav_lengths.to(device=self._ssl_device, dtype=torch.long)
        mert_input = self.resampler_mert(target_wavs.mean(dim=1))
        bsz = target_wavs.shape[0]
        actual_lengths_24k = wav_lengths // 2

        means = torch.stack([mert_input[i, : actual_lengths_24k[i]].mean() for i in range(bsz)])
        vars = torch.stack([mert_input[i, : actual_lengths_24k[i]].var() for i in range(bsz)])
        mert_input = (mert_input - means.view(-1, 1)) / torch.sqrt(vars.view(-1, 1) + 1e-7)

        chunk_size = 24000 * 5
        num_chunks_per_audio = (actual_lengths_24k + chunk_size - 1) // chunk_size

        all_chunks = []
        chunk_actual_lengths = []
        for i in range(bsz):
            audio = mert_input[i]
            actual_length = actual_lengths_24k[i]
            for start in range(0, actual_length, chunk_size):
                end = min(start + chunk_size, actual_length)
                chunk = audio[start:end]
                if len(chunk) < chunk_size:
                    chunk = F.pad(chunk, (0, chunk_size - len(chunk)))
                all_chunks.append(chunk)
                chunk_actual_lengths.append(end - start)

        if not all_chunks:
            return None

        all_chunks = torch.stack(all_chunks, dim=0).to(self._ssl_device)
        with torch.no_grad():
            hidden_states = self.mert_model(all_chunks).last_hidden_state

        chunk_features = [(length + 319) // 320 for length in chunk_actual_lengths]
        chunk_hidden_states = [hidden_states[i, : chunk_features[i], :] for i in range(len(all_chunks))]

        mert_ssl_hidden_states = []
        chunk_idx = 0
        for i in range(bsz):
            audio_chunks = chunk_hidden_states[chunk_idx : chunk_idx + num_chunks_per_audio[i]]
            if not audio_chunks:
                mert_ssl_hidden_states.append(torch.zeros(1, self.mert_model.config.hidden_size))
            else:
                mert_ssl_hidden_states.append(torch.cat(audio_chunks, dim=0))
            chunk_idx += num_chunks_per_audio[i]
        return mert_ssl_hidden_states

    def _infer_mhubert_ssl(self, target_wavs: torch.Tensor, wav_lengths: torch.Tensor):
        if self.hubert_model is None or self.resampler_mhubert is None:
            return None
        target_wavs = target_wavs.to(device=self._ssl_device, dtype=torch.float32)
        wav_lengths = wav_lengths.to(device=self._ssl_device, dtype=torch.long)
        mhubert_input = self.resampler_mhubert(target_wavs.mean(dim=1))
        bsz = target_wavs.shape[0]
        actual_lengths_16k = wav_lengths // 3

        means = torch.stack([mhubert_input[i, : actual_lengths_16k[i]].mean() for i in range(bsz)])
        vars = torch.stack([mhubert_input[i, : actual_lengths_16k[i]].var() for i in range(bsz)])
        mhubert_input = (mhubert_input - means.view(-1, 1)) / torch.sqrt(vars.view(-1, 1) + 1e-7)

        chunk_size = 16000 * 30
        num_chunks_per_audio = (actual_lengths_16k + chunk_size - 1) // chunk_size
        all_chunks = []
        chunk_actual_lengths = []
        for i in range(bsz):
            audio = mhubert_input[i]
            actual_length = actual_lengths_16k[i]
            for start in range(0, actual_length, chunk_size):
                end = min(start + chunk_size, actual_length)
                chunk = audio[start:end]
                if len(chunk) < chunk_size:
                    chunk = F.pad(chunk, (0, chunk_size - len(chunk)))
                all_chunks.append(chunk)
                chunk_actual_lengths.append(end - start)

        if not all_chunks:
            return None

        all_chunks = torch.stack(all_chunks, dim=0).to(self._ssl_device)
        with torch.no_grad():
            hidden_states = self.hubert_model(all_chunks).last_hidden_state

        chunk_features = [(length + 319) // 320 for length in chunk_actual_lengths]
        chunk_hidden_states = [hidden_states[i, : chunk_features[i], :] for i in range(len(all_chunks))]

        mhubert_ssl_hidden_states = []
        chunk_idx = 0
        for i in range(bsz):
            audio_chunks = chunk_hidden_states[chunk_idx : chunk_idx + num_chunks_per_audio[i]]
            if not audio_chunks:
                mhubert_ssl_hidden_states.append(torch.zeros(1, self.hubert_model.config.hidden_size))
            else:
                mhubert_ssl_hidden_states.append(torch.cat(audio_chunks, dim=0))
            chunk_idx += num_chunks_per_audio[i]
        return mhubert_ssl_hidden_states

    def _compute_ssl_embeddings_for_cache(self, waveforms: torch.Tensor, metadata_entries: List[Dict]):
        if waveforms is None or len(metadata_entries) == 0:
            return None
        if waveforms.ndim != 3:
            return None
        self._ensure_ssl_models()
        if self.mert_model is None and self.hubert_model is None:
            return None
        wav_tensor = waveforms.detach().to(device=self._ssl_device, dtype=torch.float32)
        default_length = wav_tensor.shape[-1]
        lengths = []
        for meta in metadata_entries:
            if isinstance(meta, dict):
                length = meta.get("num_samples")
            else:
                length = None
            lengths.append(int(length or default_length))
        wav_lengths = torch.tensor(lengths, dtype=torch.long)
        outputs = [[] for _ in range(wav_tensor.shape[0])]
        if self.mert_model is not None:
            mert_states = self._infer_mert_ssl(wav_tensor, wav_lengths)
            if mert_states:
                for idx, state in enumerate(mert_states):
                    outputs[idx].append(state.cpu())
        if self.hubert_model is not None:
            mhubert_states = self._infer_mhubert_ssl(wav_tensor, wav_lengths)
            if mhubert_states:
                for idx, state in enumerate(mhubert_states):
                    outputs[idx].append(state.cpu())
        if not any(entry for entry in outputs):
            return None
        return outputs

    def _gather_cached_ssl(self, latent_metadata: Optional[List[Dict]]):
        if not latent_metadata:
            return None
        samples = []
        for metadata in latent_metadata:
            if isinstance(metadata, dict):
                samples.append(metadata.get("ssl_hidden_states"))
            else:
                samples.append(None)
        if not any(samples):
            return None
        per_encoder = [[] for _ in range(len(self.SSL_ENCODER_NAMES))]
        for sample_ssl in samples:
            for idx in range(len(self.SSL_ENCODER_NAMES)):
                tensor = None
                if sample_ssl and idx < len(sample_ssl):
                    tensor = sample_ssl[idx]
                if tensor is None:
                    tensor = torch.zeros(1, self.SSL_LATENT_DIMS[idx], dtype=self.config.weight_dtype)
                else:
                    tensor = tensor.clone()
                per_encoder[idx].append(tensor)
        return per_encoder

    def _tokenize_lyrics_batch(self, lyrics_list: List[Optional[str]]) -> tuple[torch.Tensor, torch.Tensor]:
        token_tensors = []
        mask_tensors = []
        for lyrics in lyrics_list:
            token_ids = self._tokenize_single_lyrics(lyrics) if lyrics else []
            if not token_ids:
                token_ids = [0]
                mask_tensor = torch.zeros(1, dtype=torch.long)
            else:
                mask_tensor = torch.ones(len(token_ids), dtype=torch.long)
            token_tensors.append(torch.tensor(token_ids, dtype=torch.long))
            mask_tensors.append(mask_tensor)
        padded_ids = pad_sequence(token_tensors, batch_first=True, padding_value=0)
        padded_mask = pad_sequence(mask_tensors, batch_first=True, padding_value=0)
        return padded_ids, padded_mask

    def _tokenize_single_lyrics(self, lyrics: Optional[str]) -> List[int]:
        if not lyrics:
            return []
        lines = lyrics.split("\n")
        lyric_token_idx = [261]
        for line in lines:
            line = line.strip()
            if not line:
                lyric_token_idx += [2]
                continue
            lang = self._detect_lyrics_language(line)
            if lang not in SUPPORT_LANGUAGES:
                lang = "en"
            if "zh" in lang:
                lang = "zh"
            if "spa" in lang:
                lang = "es"
            try:
                if structure_pattern.match(line):
                    token_idx = self.lyric_tokenizer.encode(line, "en")
                else:
                    token_idx = self.lyric_tokenizer.encode(line, lang)
                lyric_token_idx = lyric_token_idx + token_idx + [2]
            except Exception as exc:
                logger.debug("Lyric tokenization failed for line '%s': %s", line, exc)
        return lyric_token_idx

    def _detect_lyrics_language(self, text: str) -> str:
        language = "en"
        try:
            _ = self.lang_segment.getTexts(text)
            lang_counts = self.lang_segment.getCounts()
            if lang_counts:
                language = lang_counts[0][0]
                if len(lang_counts) > 1 and language == "en":
                    language = lang_counts[1][0]
        except Exception:
            language = "en"
        if language not in SUPPORT_LANGUAGES:
            language = "en"
        if "zh" in language:
            language = "zh"
        if "spa" in language:
            language = "es"
        return language

    def _prepare_conditioning_features(self, batch: dict):
        """
        Ensure speaker/lyric condition tensors exist for every sample.
        """
        latent_batch = batch.get("latent_batch")
        if latent_batch is None:
            raise ValueError("ACEStep requires VAE latents to prepare conditioning features.")
        batch_size = latent_batch.shape[0]

        speaker_embeds = batch.get("speaker_embeds")
        if speaker_embeds is None:
            speaker_embeds = torch.zeros(
                batch_size,
                self._speaker_embedding_dim(),
                dtype=self.config.weight_dtype,
            )
        batch["speaker_embeds"] = speaker_embeds

        lyric_token_ids = batch.get("lyric_token_ids")
        if lyric_token_ids is None:
            lyric_token_ids = torch.zeros(batch_size, 1, dtype=torch.long)
        elif not torch.is_tensor(lyric_token_ids):
            lyric_token_ids = torch.tensor(lyric_token_ids, dtype=torch.long)
        batch["lyric_token_ids"] = lyric_token_ids

        lyric_mask = batch.get("lyric_mask")
        if lyric_mask is None:
            lyric_mask = torch.zeros_like(lyric_token_ids)
        elif not torch.is_tensor(lyric_mask):
            lyric_mask = torch.tensor(lyric_mask, dtype=torch.long)
        batch["lyric_mask"] = lyric_mask

    def prepare_batch(self, batch: dict, state: dict) -> dict:
        """
        Mirror the upstream ACE-Step training preprocess: build masks from cached latent lengths,
        sample logit-normal timesteps, derive sigmas from the FlowMatch scheduler, add noise,
        and apply simple CFG-style conditioning dropout.
        """
        if not batch:
            return batch

        latent_batch = batch.get("latent_batch")
        if latent_batch is None:
            raise ValueError("ACEStep training batch is missing VAE latents.")
        if batch.get("prompt_embeds") is None:
            raise ValueError(
                "ACE-Step requires cached UMT5 embeddings, but none were provided. Rebuild the text embed cache."
            )

        latent_metadata = batch.get("latent_metadata")
        batch["latent_attention_mask"] = self._build_audio_attention_mask(latent_batch, latent_metadata)
        lyrics = batch.get("lyrics")
        if not lyrics:
            lyrics = self._extract_lyrics_from_metadata(latent_metadata)
            if lyrics:
                batch["lyrics"] = lyrics
        if lyrics:
            lyric_token_ids, lyric_mask = self._tokenize_lyrics_batch(lyrics)
            batch["lyric_token_ids"] = lyric_token_ids
            batch["lyric_mask"] = lyric_mask
        # Ensure text attention mask is present and 2D
        if batch.get("encoder_attention_mask") is None and batch.get("prompt_embeds") is not None:
            pe = batch["prompt_embeds"]
            mask_shape = pe.shape[:2] if pe.dim() >= 2 else (pe.shape[0], 1)
            batch["encoder_attention_mask"] = torch.ones(mask_shape, dtype=torch.float32)
        self._prepare_conditioning_features(batch)

        # Move cached tensors to device
        device = self.accelerator.device
        dtype = getattr(self.config, "weight_dtype", torch.float32)
        latents = latent_batch.to(device=device, dtype=dtype)
        attention_mask = batch["latent_attention_mask"].to(device=device, dtype=dtype)
        encoder_hidden_states = batch["prompt_embeds"].to(device=device, dtype=dtype)
        text_attention_mask = batch.get("encoder_attention_mask")
        if text_attention_mask is not None:
            text_attention_mask = batch["encoder_attention_mask"].to(device=device, dtype=dtype)
        speaker_embeds = batch["speaker_embeds"].to(device=device, dtype=dtype)
        lyric_token_ids = batch["lyric_token_ids"].to(device=device, dtype=torch.long)
        lyric_mask = batch["lyric_mask"].to(device=device, dtype=torch.long)

        # Sample timesteps via logit-normal to index scheduler sigmas (upstream behavior).
        timesteps_tensor = self.noise_schedule.timesteps.to(device)
        sigmas_tensor = self.noise_schedule.sigmas.to(device)
        bsz = latents.shape[0]
        mean = getattr(self.config, "logit_mean", 0.0)
        std = getattr(self.config, "logit_std", 1.0)
        u = torch.normal(mean=mean, std=std, size=(bsz,), device=device)
        u = torch.sigmoid(u)
        indices = (u * (timesteps_tensor.shape[0] - 1)).long().clamp(0, timesteps_tensor.shape[0] - 1)
        timesteps = timesteps_tensor[indices]
        sigmas = sigmas_tensor[indices]
        # Expand sigmas to latent shape for mixing/noise and to feed the model
        view_shape = [bsz] + [1] * (latents.ndim - 1)
        sigmas_expanded = sigmas.view(*view_shape).to(dtype=dtype)

        noise = torch.randn_like(latents, device=device, dtype=dtype)
        noisy_latents = sigmas_expanded * noise + (1.0 - sigmas_expanded) * latents
        if not torch.isfinite(noisy_latents).all():
            raise ValueError(
                f"Non-finite noisy_latents detected (min={noisy_latents.min().item()}, max={noisy_latents.max().item()})"
            )

        # Apply classifier-free guidance style dropout during training.
        is_training = not state.get("is_validation", False) if isinstance(state, dict) else True
        if is_training:
            text_keep_mask = (torch.rand(size=(bsz,), device=device) >= 0.15).float().view(bsz, 1, 1)
            encoder_hidden_states = encoder_hidden_states * text_keep_mask

            speaker_keep_mask = (torch.rand(size=(bsz,), device=device) >= 0.50).float().view(bsz, 1)
            speaker_embeds = speaker_embeds * speaker_keep_mask

            lyric_keep_mask = (torch.rand(size=(bsz,), device=device) >= 0.15).float().view(bsz, 1)
            lyric_token_ids = (lyric_token_ids * lyric_keep_mask).to(dtype=torch.long)
            lyric_mask = (lyric_mask * lyric_keep_mask).to(dtype=torch.long)

        prepared = {
            "latents": latents,
            "noisy_latents": noisy_latents,
            "sigmas": sigmas_expanded,
            "timesteps": timesteps.to(device=device, dtype=dtype),
            "attention_mask": attention_mask,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": text_attention_mask,
            "speaker_embeds": speaker_embeds,
            "lyric_token_ids": lyric_token_ids,
            "lyric_mask": lyric_mask,
        }

        ssl_hidden_states = self._gather_cached_ssl(latent_metadata)
        if ssl_hidden_states is not None:
            prepared["ssl_hidden_states"] = [
                [tensor.to(device, dtype=dtype) for tensor in encoder_states] for encoder_states in ssl_hidden_states
            ]

        return prepared

    def sample_flow_sigmas(self, batch: dict, state: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ACE-Step mirrors the upstream trainer: sample timesteps via a logit-normal,
        then look up sigmas from FlowMatchEulerDiscreteScheduler.
        """
        bsz = batch["latents"].shape[0]
        timesteps_tensor = self.noise_schedule.timesteps.to(self.accelerator.device)
        sigmas_tensor = self.noise_schedule.sigmas.to(self.accelerator.device)

        mean = getattr(self.config, "logit_mean", 0.0)
        std = getattr(self.config, "logit_std", 1.0)
        u = torch.normal(mean=mean, std=std, size=(bsz,), device=self.accelerator.device)
        u = torch.sigmoid(u)
        indices = (u * (timesteps_tensor.shape[0] - 1)).long().clamp(0, timesteps_tensor.shape[0] - 1)
        timesteps = timesteps_tensor[indices]
        sigmas = sigmas_tensor[indices]
        # Flow sigmas are expected to be 1D; expand happens later via expand_sigmas.
        return sigmas, timesteps

    def model_predict(self, prepared_batch: dict) -> Dict[str, object]:
        transformer = self.get_trained_component()
        if transformer is None:
            raise ValueError("ACE-Step transformer has not been loaded before model_predict was invoked.")

        noise_latents = prepared_batch["noisy_latents"]
        attention_mask = prepared_batch["attention_mask"]
        text_hidden_states = prepared_batch["encoder_hidden_states"]
        text_attention_mask = prepared_batch.get("encoder_attention_mask")
        speaker_embeds = prepared_batch["speaker_embeds"]
        lyric_token_ids = prepared_batch["lyric_token_ids"]
        lyric_mask = prepared_batch["lyric_mask"]
        timesteps = prepared_batch["timesteps"]
        sigmas = prepared_batch["sigmas"]
        ssl_hidden_states = prepared_batch.get("ssl_hidden_states")

        output = transformer(
            hidden_states=noise_latents,
            attention_mask=attention_mask,
            encoder_text_hidden_states=text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embeds,
            lyric_token_idx=lyric_token_ids,
            lyric_mask=lyric_mask,
            timestep=timesteps,
            timestep_sign=(
                prepared_batch.get("twinflow_time_sign") if getattr(self.config, "twinflow_enabled", False) else None
            ),
            ssl_hidden_states=ssl_hidden_states,
            return_dict=True,
        )

        # Precondition velocity to data prediction: x0_hat = noisy - sigma * v_theta(noisy, sigma)
        data_pred = (-sigmas) * output.sample + noise_latents
        if not torch.isfinite(data_pred).all():
            raise ValueError(
                f"Non-finite model_prediction detected (min={data_pred.min().item()}, max={data_pred.max().item()})"
            )

        return {
            "model_prediction": data_pred,
            "proj_losses": output.proj_losses,
        }

    def loss(self, prepared_batch: dict, model_output, apply_conditioning_mask: bool = True):
        """
        Override base loss to mask out padding regions in audio latents.
        Upstream ACE-Step masks predictions/targets before computing loss to avoid
        learning from padding tokens, which causes gradient explosion.
        """
        import torch.nn.functional as F

        model_pred = model_output.get("model_prediction")
        if model_pred is None:
            model_pred = model_output.get("sample")
        target = prepared_batch.get("latents")
        attention_mask = prepared_batch.get("attention_mask")

        # If no attention mask, fall back to base implementation
        if attention_mask is None:
            return super().loss(prepared_batch, model_output, apply_conditioning_mask)

        # Expand mask to match model_pred shape: [N, C, W, T]
        # attention_mask is [N, T], we need [N, C, W, T]
        bsz = model_pred.shape[0]
        mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(-1, model_pred.shape[1], model_pred.shape[2], -1)

        # Mask predictions and targets to zero out padding regions
        selected_model_pred = (model_pred * mask).reshape(bsz, -1).contiguous()
        selected_target = (target * mask).reshape(bsz, -1).contiguous()

        # Compute MSE loss on masked regions
        loss = F.mse_loss(selected_model_pred.float(), selected_target.float(), reduction="none")
        loss = loss.mean(1)
        # Weight by the proportion of valid (non-padded) tokens
        loss = loss * mask.reshape(bsz, -1).mean(1)
        loss = loss.mean()

        return loss

    def auxiliary_loss(self, model_output, prepared_batch: dict, loss: torch.Tensor):
        proj_losses = model_output.get("proj_losses")
        if not proj_losses:
            return loss, None

        logs = {}
        collected = []
        for entry in proj_losses:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            name, value = entry
            if value is None:
                continue
            logs[f"ssl/{name}"] = value.detach().float().item()
            collected.append(value)

        if not collected:
            return loss, logs or None

        weight = float(getattr(self.config, "ace_step_ssl_loss_weight", 0.1) or 0.0)
        if weight == 0.0:
            return loss, logs

        stacked = torch.stack(collected).to(device=loss.device, dtype=loss.dtype)
        mean_proj = stacked.mean()
        logs["ssl/mean"] = mean_proj.detach().float().item()
        updated_loss = loss + mean_proj * weight
        return updated_loss, logs

    def custom_model_card_schedule_info(self) -> str:
        """
        Provide scheduler details for the model card, matching ACE-Step's flow-matching setup.
        """
        return (
            "\n"
            "    - Scheduler: FlowMatchEulerDiscreteScheduler (shift=3.0 by default)\n"
            "    - Timestep sampling: logit-normal (mean=0, std=1) -> sigmas from scheduler.sigmas\n"
            "    - Objective: flow-matching velocity preconditioned to data prediction (x0_hat = noisy - sigma * v)\n"
        )


ACEStep.register_config_requirements()
ModelRegistry.register("ace_step", ACEStep)
