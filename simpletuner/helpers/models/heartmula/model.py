# Copyright 2025 SimpleTuner contributors
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from huggingface_hub import snapshot_download
from tokenizers import Tokenizer

from simpletuner.helpers.configuration.registry import ConfigRegistry, ConfigRule, RuleType, ValidationResult
from simpletuner.helpers.models.common import AudioModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.heartmula.gen_config import HeartMuLaGenConfig
from simpletuner.helpers.models.heartmula.modeling_heartmula import HeartMuLaModel
from simpletuner.helpers.models.heartmula.pipeline import HeartMuLaGenPipeline
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.training.adapter import load_lora_weights
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)


class HeartMuLa(AudioModelFoundation):
    NAME = "HeartMuLa"
    MODEL_DESCRIPTION = "Autoregressive audio foundation model (HeartMuLa oss 3B)"
    ENABLED_IN_WIZARD = True
    MODEL_TYPE = ModelTypes.TRANSFORMER
    PREDICTION_TYPE = PredictionTypes.AUTOREGRESSIVE_NEXT_TOKEN
    MODEL_CLASS = HeartMuLaModel
    MODEL_SUBFOLDER = "transformer"
    DEFAULT_MODEL_FLAVOUR = "3b"
    HUGGINGFACE_PATHS = {
        "3b": "HeartMuLa/HeartMuLa-oss-3B",
    }
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2AUDIO: HeartMuLaGenPipeline,
    }
    DEFAULT_PIPELINE_TYPE = PipelineTypes.TEXT2AUDIO
    TEXT_ENCODER_CONFIGURATION = {}
    AUTOENCODER_CLASS = None
    SUPPORTS_LORA = True
    DEFAULT_LORA_TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    VALIDATION_USES_NEGATIVE_PROMPT = False

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        self.model = None
        self.tokenizer: Optional[Tokenizer] = None
        self.gen_config: Optional[HeartMuLaGenConfig] = None
        self._gen_asset_path: Optional[str] = None

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        return 27

    @classmethod
    def caption_field_preferences(cls, dataset_type: Optional[str] = None) -> list[str]:
        if dataset_type == "audio":
            return ["tags", "lyrics"]
        return []

    @classmethod
    def register_config_requirements(cls):
        rules = [
            ConfigRule(
                field_name="dataset_type",
                rule_type=RuleType.CUSTOM,
                value=None,
                message="HeartMuLa expects audio datasets. Configure `dataset_type: audio` for training backends.",
                error_level="warning",
            ),
        ]
        ConfigRegistry.register_rules("heartmula", rules)
        ConfigRegistry.register_validator(
            "heartmula",
            cls._validate_audio_dataset_usage,
            "Validates HeartMuLa audio dataset requirements.",
        )

    @staticmethod
    def _validate_audio_dataset_usage(config: dict) -> list[ValidationResult]:
        dataset_type = (config or {}).get("dataset_type")
        if dataset_type and str(dataset_type).lower() != "audio":
            return [
                ValidationResult(
                    passed=False,
                    field="dataset_type",
                    message="HeartMuLa requires audio datasets. Update your data backend configuration.",
                    level="warning",
                    suggestion="Set dataset_type: audio for every training backend when using HeartMuLa.",
                )
            ]
        return []

    def uses_audio_latents(self) -> bool:
        return False

    def uses_audio_tokens(self) -> bool:
        return True

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        raise NotImplementedError("HeartMuLa does not use encode_prompts during training.")

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        return {}

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        return {}

    def validation_audio_sample_rate(self) -> Optional[int]:
        return 48000

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        if "prompt" in pipeline_kwargs and "tags" not in pipeline_kwargs:
            pipeline_kwargs["tags"] = pipeline_kwargs.pop("prompt")
        return pipeline_kwargs

    def _resolve_pretrained_path(self) -> str:
        model_path = self.config.pretrained_model_name_or_path
        if not model_path:
            raise ValueError("pretrained_model_name_or_path is required for HeartMuLa.")
        if os.path.exists(model_path):
            return model_path
        logger.info("Downloading HeartMuLa weights from %s", model_path)
        return snapshot_download(model_path)

    def _resolve_gen_assets_path(self, model_path: str) -> str:
        if self._gen_asset_path is not None:
            return self._gen_asset_path
        candidates = []
        if model_path:
            path = Path(model_path)
            candidates.extend([path, path.parent, path / "HeartMuLaGen", path.parent / "HeartMuLaGen"])
        for candidate in candidates:
            if candidate is None:
                continue
            tokenizer_path = candidate / "tokenizer.json"
            gen_path = candidate / "gen_config.json"
            if tokenizer_path.is_file() and gen_path.is_file():
                self._gen_asset_path = str(candidate)
                return self._gen_asset_path
        logger.info("Downloading HeartMuLaGen assets from HeartMuLa/HeartMuLaGen")
        self._gen_asset_path = snapshot_download("HeartMuLa/HeartMuLaGen")
        return self._gen_asset_path

    def _load_tokenizer(self):
        if self.tokenizer is not None and self.gen_config is not None:
            return
        model_path = self._resolve_pretrained_path()
        assets_path = self._resolve_gen_assets_path(model_path)
        tokenizer_path = os.path.join(assets_path, "tokenizer.json")
        gen_config_path = os.path.join(assets_path, "gen_config.json")
        if not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(f"HeartMuLa tokenizer.json not found at {tokenizer_path}.")
        if not os.path.isfile(gen_config_path):
            raise FileNotFoundError(f"HeartMuLa gen_config.json not found at {gen_config_path}.")
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.gen_config = HeartMuLaGenConfig.from_file(gen_config_path)

    def load_model(self, move_to_device: bool = True):
        model_path = self._resolve_pretrained_path()
        self.model = self.MODEL_CLASS.from_pretrained(
            model_path,
            torch_dtype=self.config.weight_dtype,
            revision=self.config.revision,
        )
        if move_to_device:
            self.model.to(self.accelerator.device)
        return self.model

    def add_lora_adapter(self):
        from peft import LoraConfig, get_peft_model

        if self.model is None:
            raise ValueError("HeartMuLa must load the base model before applying LoRA adapters.")

        target_modules = self.get_lora_target_layers()
        save_modules = self.get_lora_save_layers()
        addkeys, misskeys = [], []

        lora_config_cls = LoraConfig
        lora_config_kwargs = {}
        if getattr(self.config, "peft_lora_mode", None) is not None:
            if self.config.peft_lora_mode.lower() == "singlora":
                from peft_singlora import SingLoRAConfig, setup_singlora

                lora_config_cls = SingLoRAConfig
                lora_config_kwargs = {"ramp_up_steps": self.config.singlora_ramp_up_steps or 100}
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
            from simpletuner.helpers.utils import ramtorch as ramtorch_utils

            ramtorch_utils.register_lora_custom_module(self.lora_config)

        self.model = get_peft_model(self.model, self.lora_config)

        if getattr(self.config, "init_lora", None):
            use_dora = getattr(self.config, "use_dora", False) if isinstance(self.lora_config, LoraConfig) else False
            addkeys, misskeys = load_lora_weights(
                {self.MODEL_TYPE.value: self.model},
                self.config.init_lora,
                use_dora=use_dora,
            )

        return addkeys, misskeys

    def prepare_batch(self, batch: dict, state: dict) -> dict:
        if not batch:
            return batch
        tokens = batch.get("tokens")
        tokens_mask = batch.get("tokens_mask")
        audio_frame_mask = batch.get("audio_frame_mask")
        if tokens is None or tokens_mask is None or audio_frame_mask is None:
            raise ValueError("HeartMuLa batch is missing tokens, tokens_mask, or audio_frame_mask.")
        batch["tokens"] = tokens.to(device=self.accelerator.device, dtype=torch.long)
        batch["tokens_mask"] = tokens_mask.to(device=self.accelerator.device)
        batch["audio_frame_mask"] = audio_frame_mask.to(device=self.accelerator.device)
        return batch

    def model_predict(self, prepared_batch, custom_timesteps: list = None):
        del custom_timesteps
        tokens = prepared_batch["tokens"]
        tokens_mask = prepared_batch["tokens_mask"]
        attention_mask = tokens_mask.any(dim=-1).to(dtype=torch.long)
        return self.model(tokens=tokens, tokens_mask=tokens_mask, attention_mask=attention_mask)

    def loss(self, prepared_batch: dict, model_output, apply_conditioning_mask: bool = True):
        del apply_conditioning_mask
        tokens = prepared_batch["tokens"]
        audio_frame_mask = prepared_batch["audio_frame_mask"]
        if tokens.dim() != 3:
            raise ValueError(f"HeartMuLa tokens must have shape [batch, seq_len, num_codebooks+1], got {tokens.shape}.")
        if audio_frame_mask is None:
            raise ValueError("HeartMuLa loss requires audio_frame_mask.")

        target_audio = tokens[:, 1:, :-1]
        frame_mask = audio_frame_mask[:, 1:]
        if frame_mask.numel() == 0:
            raise ValueError("HeartMuLa batch does not contain audio frames.")

        codebook0_logits = model_output["codebook0_logits"]
        codebook_logits = model_output["codebook_logits"]
        vocab_size = codebook0_logits.shape[-1]

        loss0 = torch.nn.functional.cross_entropy(
            codebook0_logits.reshape(-1, vocab_size),
            target_audio[:, :, 0].reshape(-1),
            reduction="none",
        )
        loss0 = loss0.view(frame_mask.shape) * frame_mask.to(loss0.dtype)
        denom_frames = frame_mask.sum().clamp_min(1)
        loss0 = loss0.sum() / denom_frames

        rest_logits = codebook_logits.reshape(-1, vocab_size)
        rest_targets = target_audio[:, :, 1:].reshape(-1)
        loss_rest = torch.nn.functional.cross_entropy(rest_logits, rest_targets, reduction="none")
        loss_rest = loss_rest.view(*frame_mask.shape, -1) * frame_mask.unsqueeze(-1).to(loss_rest.dtype)
        denom_rest = denom_frames * (target_audio.shape[-1] - 1)
        loss_rest = loss_rest.sum() / denom_rest

        return loss0 + loss_rest

    def _normalize_tags(self, tags: str) -> str:
        tags = tags.strip().lower()
        if not tags.startswith("<tag>"):
            tags = f"<tag>{tags}"
        if not tags.endswith("</tag>"):
            tags = f"{tags}</tag>"
        return tags

    def _encode_text(self, text: str) -> list[int]:
        if self.tokenizer is None or self.gen_config is None:
            raise ValueError("HeartMuLa tokenizer was not loaded.")
        token_ids = self.tokenizer.encode(text).ids
        if not token_ids:
            return [self.gen_config.text_bos_id, self.gen_config.text_eos_id]
        if token_ids[0] != self.gen_config.text_bos_id:
            token_ids = [self.gen_config.text_bos_id] + token_ids
        if token_ids[-1] != self.gen_config.text_eos_id:
            token_ids = token_ids + [self.gen_config.text_eos_id]
        return token_ids

    def _resolve_tokens_path(self, token_path: str, example: dict) -> str:
        if os.path.isabs(token_path):
            return token_path
        backend_id = example.get("data_backend_id")
        backend_cfg = StateTracker.get_data_backend_config(backend_id) if backend_id else {}
        dataset_root = backend_cfg.get("instance_data_dir") if backend_cfg else None
        if dataset_root:
            return os.path.join(dataset_root, token_path)
        return token_path

    def _load_audio_tokens(self, example: dict) -> torch.Tensor:
        tokens = example.get("audio_tokens")
        if tokens is None:
            token_path = example.get("audio_tokens_path")
            if not token_path:
                raise ValueError("HeartMuLa requires audio_tokens or audio_tokens_path in metadata.")
            resolved = self._resolve_tokens_path(str(token_path), example)
            if not os.path.exists(resolved):
                raise FileNotFoundError(f"HeartMuLa audio token file not found: {resolved}")
            suffix = os.path.splitext(resolved)[1].lower()
            if suffix in {".npy", ".npz"}:
                payload = np.load(resolved, allow_pickle=False)
                if hasattr(payload, "files"):
                    if "tokens" in payload:
                        tokens = payload["tokens"]
                    elif payload.files:
                        tokens = payload[payload.files[0]]
                    else:
                        raise ValueError(f"Audio token archive {resolved} is empty.")
                else:
                    tokens = payload
            else:
                tokens = torch.load(resolved, map_location="cpu")
        if isinstance(tokens, torch.Tensor):
            tensor = tokens.detach().cpu()
        else:
            tensor = torch.as_tensor(tokens)
        if tensor.ndim != 2:
            raise ValueError(
                f"HeartMuLa audio tokens must have shape [codebooks, frames] or [frames, codebooks], got {tensor.shape}."
            )
        num_codebooks = self.model.config.audio_num_codebooks
        if tensor.shape[0] == num_codebooks and tensor.shape[1] != num_codebooks:
            tensor = tensor.transpose(0, 1)
        elif tensor.shape[0] == num_codebooks and tensor.shape[1] == num_codebooks:
            raise ValueError(
                f"Audio token matrix shape {tensor.shape} is ambiguous because both dimensions equal "
                f"num_codebooks ({num_codebooks}). Provide tokens with distinct frame and codebook dimensions."
            )
        elif tensor.shape[1] != num_codebooks:
            raise ValueError(f"Audio token matrix does not match expected codebooks ({num_codebooks}): got {tensor.shape}.")
        return tensor.to(dtype=torch.long)

    def collate_audio_tokens(self, examples: list[dict]) -> dict:
        self._load_tokenizer()
        if self.gen_config is None:
            raise ValueError("HeartMuLa gen_config was not loaded.")
        num_codebooks = self.model.config.audio_num_codebooks if self.model is not None else None
        if num_codebooks is None:
            raise ValueError("HeartMuLa model must be loaded before collation.")

        tokens_list = []
        tokens_mask_list = []
        audio_frame_masks = []
        prompt_lengths = []
        audio_lengths = []
        prompt_texts = []

        for example in examples:
            tags = example.get("tags") or example.get("prompt")
            lyrics = example.get("lyrics")
            if tags is None:
                raise ValueError("HeartMuLa requires 'tags' (or 'prompt') in metadata.")
            if lyrics is None:
                raise ValueError("HeartMuLa requires 'lyrics' in metadata.")
            if not isinstance(tags, str) or not isinstance(lyrics, str):
                raise ValueError("HeartMuLa tags and lyrics must be strings.")

            tags_text = self._normalize_tags(tags)
            lyrics_text = lyrics.strip().lower()
            tag_ids = self._encode_text(tags_text)
            lyric_ids = self._encode_text(lyrics_text)

            prompt_len = len(tag_ids) + 1 + len(lyric_ids)
            audio_tokens = self._load_audio_tokens(example)
            audio_len = audio_tokens.shape[0]
            seq_len = prompt_len + audio_len

            token_matrix = torch.full(
                (seq_len, num_codebooks + 1),
                fill_value=self.gen_config.empty_id,
                dtype=torch.long,
            )
            token_matrix[: len(tag_ids), -1] = torch.tensor(tag_ids, dtype=torch.long)
            token_matrix[len(tag_ids) + 1 : prompt_len, -1] = torch.tensor(lyric_ids, dtype=torch.long)
            token_matrix[prompt_len:, :num_codebooks] = audio_tokens

            token_mask = torch.zeros_like(token_matrix, dtype=torch.bool)
            token_mask[:prompt_len, -1] = True
            token_mask[prompt_len:, :num_codebooks] = True

            frame_mask = torch.zeros(seq_len, dtype=torch.bool)
            frame_mask[prompt_len:] = True

            tokens_list.append(token_matrix)
            tokens_mask_list.append(token_mask)
            audio_frame_masks.append(frame_mask)
            prompt_lengths.append(prompt_len)
            audio_lengths.append(audio_len)
            prompt_texts.append(tags_text)

        max_len = max(tokens.shape[0] for tokens in tokens_list)
        batch_size = len(tokens_list)
        batch_tokens = torch.full(
            (batch_size, max_len, num_codebooks + 1),
            fill_value=self.gen_config.empty_id,
            dtype=torch.long,
        )
        batch_masks = torch.zeros((batch_size, max_len, num_codebooks + 1), dtype=torch.bool)
        batch_frame_masks = torch.zeros((batch_size, max_len), dtype=torch.bool)
        for idx, (tokens, mask, frame_mask) in enumerate(zip(tokens_list, tokens_mask_list, audio_frame_masks)):
            length = tokens.shape[0]
            batch_tokens[idx, :length] = tokens
            batch_masks[idx, :length] = mask
            batch_frame_masks[idx, :length] = frame_mask

        return {
            "tokens": batch_tokens,
            "tokens_mask": batch_masks,
            "audio_frame_mask": batch_frame_masks,
            "prompt_lengths": prompt_lengths,
            "audio_lengths": audio_lengths,
            "prompts": prompt_texts,
        }


HeartMuLa.register_config_requirements()
ModelRegistry.register("heartmula", HeartMuLa)
