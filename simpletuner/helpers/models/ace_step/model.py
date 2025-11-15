# NOTE: This file originates from the ACE-Step project (Apache-2.0).
#       Modifications for SimpleTuner are © 2024 SimpleTuner contributors
#       and distributed under the AGPL-3.0-or-later.

"""
ACEStep audio model integration for SimpleTuner.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import torchaudio
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
from simpletuner.helpers.models.ace_step.transformer import ACEStepTransformer2DModel
from simpletuner.helpers.models.common import AudioModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)


class ACEStep(AudioModelFoundation):
    NAME = "ACE-Step"
    MODEL_DESCRIPTION = "Audio generation foundation model (ACE-Step v1 3.5B)"
    ENABLED_IN_WIZARD = False
    MODEL_LICENSE = "apache-2.0"
    MODEL_TYPE = ModelTypes.TRANSFORMER
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_CLASS = ACEStepTransformer2DModel
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: ACEStepPipeline,
    }
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

    def __init__(self, config: dict, accelerator):
        super().__init__(config, accelerator)
        self.text_tokenizer_max_length = getattr(self.config, "tokenizer_max_length", 256)
        self.mert_model = None
        self.hubert_model = None
        self.resampler_mert = None
        self.resampler_mhubert = None
        self._ssl_models_ready = False
        self.lyric_tokenizer = VoiceBpeTokenizer()
        self.lang_segment = LangSegment(language_filters)

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
                default_value=1.0,
                message="Projection-alignment losses default to weight 1.0 for ACE-Step.",
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

    def convert_text_embed_for_pipeline(self, text_embedding: Dict[str, torch.Tensor], prompt: str) -> dict:
        return {
            "encoder_text_hidden_states": text_embedding["prompt_embeds"].unsqueeze(0),
            "text_attention_mask": text_embedding.get("attention_masks"),
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: Dict[str, torch.Tensor], prompt: str) -> dict:
        return {
            "negative_encoder_text_hidden_states": text_embedding["prompt_embeds"].unsqueeze(0),
            "negative_text_attention_mask": text_embedding.get("attention_masks"),
        }

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
        for idx, metadata in enumerate(latent_metadata):
            latent_lengths = None
            if isinstance(metadata, dict):
                latent_lengths = metadata.get("latent_lengths") or metadata.get("latent_length")
            elif hasattr(metadata, "get"):
                latent_lengths = metadata.get("latent_lengths")
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
            self.mert_model.requires_grad_(False)
        except Exception as exc:
            logger.warning("Failed to load MERT SSL encoder: %s", exc)
            self.mert_model = None
        try:
            self.hubert_model = AutoModel.from_pretrained("utter-project/mHuBERT-147").eval()
            self.hubert_model.requires_grad_(False)
        except Exception as exc:
            logger.warning("Failed to load mHuBERT SSL encoder: %s", exc)
            self.hubert_model = None
        if self.mert_model is not None:
            self.resampler_mert = torchaudio.transforms.Resample(orig_freq=48000, new_freq=24000)
        if self.hubert_model is not None:
            self.resampler_mhubert = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
        self._ssl_models_ready = True

    def _infer_mert_ssl(self, target_wavs: torch.Tensor, wav_lengths: torch.Tensor):
        if self.mert_model is None or self.resampler_mert is None:
            return None
        target_wavs = target_wavs.to(torch.float32)
        wav_lengths = wav_lengths.to(torch.long)
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

        all_chunks = torch.stack(all_chunks, dim=0)
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
        target_wavs = target_wavs.to(torch.float32)
        wav_lengths = wav_lengths.to(torch.long)
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

        all_chunks = torch.stack(all_chunks, dim=0)
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
        wav_tensor = waveforms.detach().to(torch.float32).cpu()
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
        self._prepare_conditioning_features(batch)

        prepared = super().prepare_batch(batch=batch, state=state)

        prepared["attention_mask"] = batch["latent_attention_mask"].to(
            device=self.accelerator.device,
            dtype=self.config.weight_dtype,
        )
        prepared["speaker_embeds"] = batch["speaker_embeds"].to(
            device=self.accelerator.device,
            dtype=self.config.weight_dtype,
        )
        prepared["lyric_token_ids"] = batch["lyric_token_ids"].to(
            device=self.accelerator.device,
            dtype=torch.long,
        )
        prepared["lyric_mask"] = batch["lyric_mask"].to(
            device=self.accelerator.device,
            dtype=torch.long,
        )
        ssl_hidden_states = self._gather_cached_ssl(latent_metadata)
        if ssl_hidden_states is not None:
            prepared["ssl_hidden_states"] = [
                [tensor.to(self.accelerator.device, dtype=self.config.weight_dtype) for tensor in encoder_states]
                for encoder_states in ssl_hidden_states
            ]

        return prepared

    def model_predict(self, prepared_batch: dict) -> Dict[str, object]:
        transformer = self.get_trained_component()
        if transformer is None:
            raise ValueError("ACE-Step transformer has not been loaded before model_predict was invoked.")

        noise_latents = prepared_batch["noisy_latents"].to(
            device=self.accelerator.device,
            dtype=self.config.weight_dtype,
        )
        attention_mask = prepared_batch["attention_mask"].to(
            device=self.accelerator.device,
            dtype=self.config.weight_dtype,
        )
        text_hidden_states = prepared_batch["encoder_hidden_states"].to(
            device=self.accelerator.device,
            dtype=self.config.weight_dtype,
        )
        text_attention_mask = prepared_batch.get("encoder_attention_mask")
        if text_attention_mask is not None:
            text_attention_mask = text_attention_mask.to(device=self.accelerator.device)

        speaker_embeds = prepared_batch["speaker_embeds"].to(
            device=self.accelerator.device,
            dtype=self.config.weight_dtype,
        )
        lyric_token_ids = prepared_batch["lyric_token_ids"].to(
            device=self.accelerator.device,
            dtype=torch.long,
        )
        lyric_mask = prepared_batch["lyric_mask"].to(
            device=self.accelerator.device,
            dtype=torch.long,
        )
        timesteps = prepared_batch["timesteps"].to(
            device=self.accelerator.device,
            dtype=self.config.weight_dtype,
        )

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
            ssl_hidden_states=ssl_hidden_states,
            return_dict=True,
        )

        return {
            "model_prediction": output.sample,
            "proj_losses": output.proj_losses,
        }

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

        weight = float(getattr(self.config, "ace_step_ssl_loss_weight", 1.0) or 0.0)
        if weight == 0.0:
            return loss, logs

        stacked = torch.stack(collected).to(device=loss.device, dtype=loss.dtype)
        mean_proj = stacked.mean()
        logs["ssl/mean"] = mean_proj.detach().float().item()
        updated_loss = loss + mean_proj * weight
        return updated_loss, logs


ACEStep.register_config_requirements()
ModelRegistry.register("ace_step", ACEStep)
