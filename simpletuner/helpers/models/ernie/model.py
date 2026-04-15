import json
import logging
import os
from typing import List, Optional

import torch
from diffusers import AutoencoderKLFlux2
from huggingface_hub import hf_hub_download
from transformers import AutoModel, Mistral3Config, PreTrainedTokenizerFast
from transformers.utils import ContextManagers

from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.ernie.pipeline import ErnieImagePipeline
from simpletuner.helpers.models.ernie.transformer import ErnieImageTransformer2DModel
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.training.deepspeed import deepspeed_zero_init_disabled_context_manager

logger = logging.getLogger(__name__)


class Ernie(ImageModelFoundation):
    SUPPORTS_MUON_CLIP = True
    NAME = "ERNIE-Image"
    MODEL_DESCRIPTION = "Baidu ERNIE single-stream diffusion transformer"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTO_LORA_FORMAT_DETECTION = True
    AUTOENCODER_CLASS = AutoencoderKLFlux2
    LATENT_CHANNEL_COUNT = 128
    VAE_SCALE_FACTOR = 16
    VALIDATION_USES_NEGATIVE_PROMPT = True
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    SLIDER_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]

    MODEL_CLASS = ErnieImageTransformer2DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: ErnieImagePipeline,
    }

    DEFAULT_MODEL_FLAVOUR = "base"
    HUGGINGFACE_PATHS = {
        "base": "baidu/ERNIE-Image",
        "turbo": "baidu/ERNIE-Image-Turbo",
    }
    MODEL_LICENSE = "apache-2.0"

    ASSISTANT_LORA_FLAVOURS = ["turbo"]
    ASSISTANT_LORA_PATH = ""
    ASSISTANT_LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"
    TEXT_EMBED_DIM = 3072

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "ERNIE text encoder",
            "tokenizer": PreTrainedTokenizerFast,
            "tokenizer_subfolder": "tokenizer",
            "model": AutoModel,
            "subfolder": "text_encoder",
        },
    }

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        return 23

    @staticmethod
    def _resolve_component_file_path(model_path: str, subfolder: str, filename: str, revision: Optional[str] = None) -> str:
        if os.path.isdir(model_path):
            candidate = os.path.join(model_path, subfolder, filename)
            if os.path.exists(candidate):
                return candidate
            direct_candidate = os.path.join(model_path, filename)
            if os.path.exists(direct_candidate):
                return direct_candidate
        return hf_hub_download(model_path, filename=f"{subfolder}/{filename}", revision=revision)

    @staticmethod
    def _patch_text_encoder_config_dict(config_dict: dict) -> dict:
        patched_config = dict(config_dict)
        text_config = dict(patched_config.get("text_config") or {})
        if text_config.get("model_type") == "ministral3":
            text_config["model_type"] = "ministral"
        if text_config.get("model_type") == "ministral" and text_config.get("sliding_window") is None:
            text_config["sliding_window"] = 4096
        patched_config["text_config"] = text_config
        return patched_config

    @staticmethod
    def _build_ernie_tokenizer(tokenizer_json_path: str, tokenizer_config: dict) -> PreTrainedTokenizerFast:
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_json_path,
            bos_token=tokenizer_config.get("bos_token"),
            eos_token=tokenizer_config.get("eos_token"),
            unk_token=tokenizer_config.get("unk_token"),
            pad_token=tokenizer_config.get("pad_token"),
            clean_up_tokenization_spaces=tokenizer_config.get("clean_up_tokenization_spaces", False),
        )
        model_max_length = tokenizer_config.get("model_max_length")
        if model_max_length is not None:
            tokenizer.model_max_length = model_max_length
        return tokenizer

    def _load_ernie_tokenizer(self):
        text_encoder_config = self.TEXT_ENCODER_CONFIGURATION["text_encoder"]
        tokenizer_path = self._resolve_text_encoder_path(text_encoder_config)
        tokenizer_subfolder = text_encoder_config.get("tokenizer_subfolder", "tokenizer")
        tokenizer_config_path = self._resolve_component_file_path(
            tokenizer_path, tokenizer_subfolder, "tokenizer_config.json", revision=self.config.revision
        )
        tokenizer_json_path = self._resolve_component_file_path(
            tokenizer_path, tokenizer_subfolder, "tokenizer.json", revision=self.config.revision
        )
        with open(tokenizer_config_path, "r", encoding="utf-8") as handle:
            tokenizer_config = json.load(handle)
        return self._build_ernie_tokenizer(tokenizer_json_path, tokenizer_config)

    def _load_ernie_text_encoder_config(self) -> Mistral3Config:
        text_encoder_config = self.TEXT_ENCODER_CONFIGURATION["text_encoder"]
        text_encoder_path = self._resolve_text_encoder_path(text_encoder_config)
        config_path = self._resolve_component_file_path(
            text_encoder_path,
            text_encoder_config.get("subfolder", "text_encoder"),
            "config.json",
            revision=self.config.revision,
        )
        with open(config_path, "r", encoding="utf-8") as handle:
            config_dict = json.load(handle)
        patched_config = self._patch_text_encoder_config_dict(config_dict)
        return Mistral3Config.from_dict(patched_config)

    def load_text_tokenizer(self):
        if getattr(self, "tokenizers", None):
            return

        tokenizer = self._load_ernie_tokenizer()
        self.tokenizers = [tokenizer]
        self.tokenizer_1 = tokenizer

    def load_text_encoder(self, move_to_device: bool = True):
        if getattr(self, "text_encoders", None):
            return

        self.load_text_tokenizer()
        text_encoder_config = self.TEXT_ENCODER_CONFIGURATION["text_encoder"]
        text_encoder_path = self._resolve_text_encoder_path(text_encoder_config)
        patched_config = self._load_ernie_text_encoder_config()

        load_kwargs = {
            "pretrained_model_name_or_path": text_encoder_path,
            "config": patched_config,
            "variant": self.config.variant,
            "revision": self.config.revision,
            "subfolder": text_encoder_config.get("subfolder", "text_encoder") or "",
            "torch_dtype": self.config.weight_dtype,
        }

        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            logger.info("Loading ERNIE text encoder with patched Mistral3 config")
            text_encoder = AutoModel.from_pretrained(**load_kwargs)

        if self._ramtorch_text_encoders_requested():
            self._apply_ramtorch_layers(
                text_encoder,
                "text_encoder_1",
                full_ramtorch=True,
                percent=self._ramtorch_text_encoder_percent(),
            )
        elif move_to_device:
            text_encoder.to(self.accelerator.device, dtype=self.config.weight_dtype)
        if hasattr(text_encoder, "eval"):
            text_encoder.eval()
        text_encoder.requires_grad_(False)

        self.text_encoders = [text_encoder]
        self.text_encoder = text_encoder
        self.text_encoder_1 = text_encoder

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = latents.shape
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError(f"Latent spatial dims must be even to patchify, got {(height, width)}")
        latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        return latents.reshape(batch_size, channels * 4, height // 2, width // 2)

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels_x4, height_x2, width_x2 = latents.shape
        channels = channels_x4 // 4
        latents = latents.view(batch_size, channels, 2, 2, height_x2, width_x2)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        return latents.reshape(batch_size, channels, height_x2 * 2, width_x2 * 2)

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae = self.get_vae()
        if vae is None:
            raise ValueError("Cannot normalize ERNIE latents without a loaded VAE.")
        bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        return (latents - bn_mean) / bn_std

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

        logger.info("TREAD training is enabled for ERNIE-Image")

    def post_model_load_setup(self):
        super().post_model_load_setup()
        self._maybe_load_assistant_lora()

    def _maybe_load_assistant_lora(self):
        if getattr(self.config, "disable_assistant_lora", False):
            return
        if not self.supports_assistant_lora(self.config):
            return
        if getattr(self.config, "model_type", "").lower() != "lora":
            return

        assistant_path = getattr(self.config, "assistant_lora_path", None) or self.ASSISTANT_LORA_PATH
        if not assistant_path:
            return

        from simpletuner.helpers.assistant_lora import load_assistant_adapter

        weight_name = getattr(self.config, "assistant_lora_weight_name", None) or self.ASSISTANT_LORA_WEIGHT_NAME
        self.assistant_lora_loaded = load_assistant_adapter(
            transformer=self.unwrap_model(model=self.model),
            pipeline_cls=ErnieImagePipeline,
            lora_path=assistant_path,
            adapter_name=self.assistant_adapter_name,
            low_cpu_mem_usage=getattr(self.config, "low_cpu_mem_usage", False),
            weight_name=weight_name,
        )

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        del is_negative_prompt
        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        text_encoder = self.text_encoders[0]
        tokenizer = self.tokenizers[0]
        tokenization_kwargs = {
            "add_special_tokens": True,
            "padding": False,
            "truncation": True,
            "return_tensors": "pt",
        }
        max_length = getattr(self.config, "tokenizer_max_length", None)
        if max_length is not None:
            tokenization_kwargs["max_length"] = max_length

        tokenized = tokenizer(prompts, **tokenization_kwargs)

        input_ids = tokenized.input_ids.to(self.accelerator.device)
        attention_mask = tokenized.attention_mask.to(self.accelerator.device).bool()
        if hasattr(text_encoder, "language_model") and hasattr(text_encoder, "get_input_embeddings"):
            input_embeds = text_encoder.get_input_embeddings()(input_ids)
            outputs = text_encoder.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            outputs = text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states:
            prompt_embeds = hidden_states[-2]
        else:
            prompt_embeds = outputs.last_hidden_state
        if prompt_embeds.shape[-1] != self.TEXT_EMBED_DIM:
            raise ValueError(
                f"ERNIE prompt embeddings must have hidden size {self.TEXT_EMBED_DIM}, got {prompt_embeds.shape[-1]}"
            )
        return {
            "prompt_embeds": prompt_embeds,
            "attention_mask": attention_mask,
        }

    def collate_prompt_embeds(self, text_encoder_output: list[dict]) -> dict:
        if not text_encoder_output:
            return {}

        prompt_embeds = []
        attention_masks = []
        max_seq_len = 0
        hidden_dim = None

        for item in text_encoder_output:
            embeds = item.get("prompt_embeds")
            mask = item.get("attention_mask")
            if embeds is None or mask is None:
                return {}

            if embeds.dim() == 2:
                embeds = embeds.unsqueeze(0)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)

            prompt_embeds.append(embeds)
            attention_masks.append(mask)
            max_seq_len = max(max_seq_len, embeds.shape[1])
            hidden_dim = embeds.shape[-1]

        padded_embeds = []
        padded_masks = []
        for embeds, mask in zip(prompt_embeds, attention_masks):
            if embeds.shape[1] < max_seq_len:
                pad_len = max_seq_len - embeds.shape[1]
                embed_pad = torch.zeros(
                    (embeds.shape[0], pad_len, hidden_dim),
                    dtype=embeds.dtype,
                    device=embeds.device,
                )
                mask_pad = torch.zeros((mask.shape[0], pad_len), dtype=mask.dtype, device=mask.device)
                embeds = torch.cat([embeds, embed_pad], dim=1)
                mask = torch.cat([mask, mask_pad], dim=1)

            padded_embeds.append(embeds)
            padded_masks.append(mask)

        return {
            "prompt_embeds": torch.cat(padded_embeds, dim=0),
            "attention_masks": torch.cat(padded_masks, dim=0),
        }

    def convert_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        attention_mask = text_embedding["attention_mask"]
        prompt_list: List[torch.Tensor] = []
        for embeds, mask in zip(prompt_embeds, attention_mask):
            prompt_list.append(embeds[mask.view(-1).bool()])
        return {"prompt_embeds": prompt_list}

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        attention_mask = text_embedding["attention_mask"]
        prompt_list: List[torch.Tensor] = []
        for embeds, mask in zip(prompt_embeds, attention_mask):
            prompt_list.append(embeds[mask.view(-1).bool()])
        return {"negative_prompt_embeds": prompt_list}

    def post_vae_encode_transform_sample(self, sample):
        if sample is None:
            return sample
        if hasattr(sample, "latent_dist"):
            sample = sample.latent_dist.mode()
        elif hasattr(sample, "sample"):
            sample = sample.sample
        if isinstance(sample, torch.Tensor) and sample.dim() == 4:
            if sample.shape[1] == 32:
                sample = self._patchify_latents(sample)
            sample = self._normalize_latents(sample)
        return sample

    def pre_latent_decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.dim() == 4 and latents.shape[1] == 128:
            return self._unpatchify_latents(latents)
        return latents

    def _pad_text_embeds(self, prompt_embeds: torch.Tensor, attention_mask: torch.Tensor):
        selected_embeds = []
        lengths = []
        for embeds, mask in zip(prompt_embeds, attention_mask):
            current = embeds[mask.view(-1).bool()].to(device=self.accelerator.device, dtype=self.config.weight_dtype)
            selected_embeds.append(current)
            lengths.append(current.shape[0])

        max_length = max(lengths, default=0)
        hidden_size = prompt_embeds.shape[-1]
        padded = torch.zeros(
            (len(selected_embeds), max_length, hidden_size),
            device=self.accelerator.device,
            dtype=self.config.weight_dtype,
        )
        for idx, current in enumerate(selected_embeds):
            padded[idx, : current.shape[0]] = current
        return padded, torch.tensor(lengths, device=self.accelerator.device, dtype=torch.long)

    def model_predict(self, prepared_batch, custom_timesteps: list = None):
        del custom_timesteps
        latents = prepared_batch["noisy_latents"]
        if latents.dim() != 4:
            raise ValueError(f"Expected 4D patchified latents for ERNIE-Image, got shape {latents.shape}")

        prompt_embeds = prepared_batch["encoder_hidden_states"]
        attention_mask = prepared_batch.get("encoder_attention_mask")
        if attention_mask is None:
            raise ValueError("encoder_attention_mask is required for ERNIE-Image training.")

        text_bth, text_lens = self._pad_text_embeds(prompt_embeds, attention_mask)

        raw_timesteps = prepared_batch["timesteps"]
        if not torch.is_tensor(raw_timesteps):
            raw_timesteps = torch.tensor(raw_timesteps, device=self.accelerator.device, dtype=torch.float32)
        else:
            raw_timesteps = raw_timesteps.to(device=self.accelerator.device, dtype=torch.float32)
        timesteps = raw_timesteps.expand(latents.shape[0])

        hidden_states_buffer = self._new_hidden_state_buffer()
        call_kwargs = {
            "hidden_states": latents.to(device=self.accelerator.device, dtype=self.config.weight_dtype),
            "timestep": timesteps,
            "text_bth": text_bth,
            "text_lens": text_lens,
            "return_dict": False,
        }
        force_keep_mask = prepared_batch.get("force_keep_mask")
        if force_keep_mask is not None:
            call_kwargs["force_keep_mask"] = force_keep_mask.to(device=self.accelerator.device, dtype=torch.bool)
        if hidden_states_buffer is not None:
            call_kwargs["hidden_states_buffer"] = hidden_states_buffer

        model_pred = self.model(**call_kwargs)[0].float()

        crepa_hidden = None
        crepa = getattr(self, "crepa_regularizer", None)
        if crepa and crepa.enabled and hidden_states_buffer is not None:
            crepa_hidden = hidden_states_buffer.get(f"layer_{crepa.block_index}")

        return {
            "model_prediction": model_pred,
            "crepa_hidden_states": crepa_hidden,
            "hidden_states_buffer": hidden_states_buffer,
        }

    def check_user_config(self):
        super().check_user_config()
        if (
            getattr(self.config, "model_type", "").lower() == "lora"
            and not getattr(self.config, "disable_assistant_lora", False)
            and self.supports_assistant_lora(self.config)
            and getattr(self.config, "assistant_lora_weight_name", None) in (None, "", "None")
        ):
            self.config.assistant_lora_weight_name = self.ASSISTANT_LORA_WEIGHT_NAME


ModelRegistry.register("ernie", Ernie)
