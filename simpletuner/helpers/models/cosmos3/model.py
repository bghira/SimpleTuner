import hashlib
import json
import logging
import os
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers import UniPCMultistepScheduler
from transformers import AutoTokenizer

from simpletuner.helpers.acceleration import (
    AccelerationBackend,
    AccelerationPreset,
    get_bitsandbytes_presets,
    get_deepspeed_presets,
    get_quanto_presets,
    get_sdnq_presets,
    get_torchao_presets,
)
from simpletuner.helpers.models.common import PipelineTypes, PredictionTypes, TextEmbedCacheKey, get_model_config_path
from simpletuner.helpers.models.cosmos3.audio_tokenizer import Cosmos3AVAEAudioTokenizer
from simpletuner.helpers.models.cosmos3.pipeline import (
    _SYSTEM_PROMPT_IMAGE,
    _SYSTEM_PROMPT_VIDEO,
    Cosmos3OmniPipeline,
    get_3d_mrope_ids_text_tokens,
)
from simpletuner.helpers.models.cosmos3.reasoner import (
    COSMOS3_GENERATOR_COMPONENTS,
    COSMOS3_REASONER_COMPONENTS,
    Cosmos3Reasoner,
)
from simpletuner.helpers.models.cosmos3.transformer import Cosmos3OmniTransformer, Cosmos3ReasonerMemoryState
from simpletuner.helpers.models.cosmos.model import Cosmos2Image
from simpletuner.helpers.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class Cosmos3Image(Cosmos2Image):
    NAME = "Cosmos3 (T2I)"
    MODEL_DESCRIPTION = "NVIDIA's Cosmos3 omni transformer model"
    ENABLED_IN_WIZARD = False
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    DEFAULT_MODEL_FLAVOUR = "nano"
    HUGGINGFACE_PATHS = {
        "edge": "nvidia/Cosmos3-Edge",
        "nano": "nvidia/Cosmos3-Nano",
        "super": "nvidia/Cosmos3-Super",
        "super-i2v": "nvidia/Cosmos3-Super-Image2Video",
        "super-t2i": "nvidia/Cosmos3-Super-Text2Image",
    }
    MODEL_CLASS = Cosmos3OmniTransformer
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: Cosmos3OmniPipeline,
    }
    MODEL_LICENSE = "openmdw-1.1"
    DEFAULT_LORA_TARGET = ["add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]
    DEFAULT_LYCORIS_TARGET = DEFAULT_LORA_TARGET
    DEFAULT_LORA_EXCLUDE_TARGETS = (
        r"(^|\.)(embed_tokens|lm_head|norm|to_q|to_k|to_v|to_out|norm_q|norm_k|mlp|"
        r"input_layernorm|post_attention_layernorm)$"
    )

    TEXT_ENCODER_CONFIGURATION = {}

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        return 35

    @classmethod
    def get_acceleration_presets(cls) -> list[AccelerationPreset]:
        base_memory_config = {
            "base_model_precision": "no_change",
            "gradient_checkpointing": True,
        }

        light_layers = ",".join(f"layers.{idx}.*" for idx in range(9))
        balanced_layers = ",".join(f"layers.{idx}.*" for idx in range(18))

        return [
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="light",
                name="RamTorch - Light",
                description="Streams 9 of 36 transformer layers (~25%).",
                tab="basic",
                tradeoff_vram="Reduces transformer VRAM by streaming early layers from CPU.",
                tradeoff_speed="Adds host-device transfer overhead.",
                tradeoff_notes="Requires enough system RAM for CPU-resident weights.",
                requires_min_system_ram_gb=64,
                config={**base_memory_config, "ramtorch": True, "ramtorch_target_modules": light_layers},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="balanced",
                name="RamTorch - Balanced",
                description="Streams 18 of 36 transformer layers (~50%).",
                tab="basic",
                tradeoff_vram="Reduces transformer VRAM by streaming half the layers from CPU.",
                tradeoff_speed="Adds substantial host-device transfer overhead.",
                tradeoff_notes="Requires enough system RAM for CPU-resident weights.",
                requires_min_system_ram_gb=64,
                config={**base_memory_config, "ramtorch": True, "ramtorch_target_modules": balanced_layers},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="aggressive",
                name="RamTorch - Aggressive",
                description="Streams all transformer layers.",
                tab="basic",
                tradeoff_vram="Maximizes transformer weight offload.",
                tradeoff_speed="Highest host-device transfer overhead.",
                tradeoff_notes="Requires enough system RAM for CPU-resident weights.",
                requires_min_system_ram_gb=64,
                config={**base_memory_config, "ramtorch": True, "ramtorch_target_modules": "layers.*"},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="light",
                name="Block Swap - Light",
                description="Swaps 9 of 36 transformer layers (~25%).",
                tab="basic",
                tradeoff_vram="Keeps most layers GPU-resident.",
                tradeoff_speed="Adds moderate block transfer overhead.",
                tradeoff_notes="Requires enough system RAM for CPU-resident blocks.",
                requires_min_system_ram_gb=64,
                config={**base_memory_config, "musubi_blocks_to_swap": 9},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="balanced",
                name="Block Swap - Balanced",
                description="Swaps 18 of 36 transformer layers (~50%).",
                tab="basic",
                tradeoff_vram="Keeps half of the layers GPU-resident.",
                tradeoff_speed="Adds significant block transfer overhead.",
                tradeoff_notes="Requires enough system RAM for CPU-resident blocks.",
                requires_min_system_ram_gb=64,
                config={**base_memory_config, "musubi_blocks_to_swap": 18},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="aggressive",
                name="Block Swap - Aggressive",
                description="Swaps 30 of 36 transformer layers (~83%).",
                tab="basic",
                tradeoff_vram="Keeps only a small layer tail GPU-resident.",
                tradeoff_speed="Highest block transfer overhead.",
                tradeoff_notes="Requires enough system RAM for CPU-resident blocks.",
                requires_min_system_ram_gb=64,
                config={**base_memory_config, "musubi_blocks_to_swap": 30},
            ),
            *get_deepspeed_presets(base_memory_config),
            *get_sdnq_presets(base_memory_config),
            *get_torchao_presets(base_memory_config),
            *get_quanto_presets(base_memory_config),
            *get_bitsandbytes_presets(base_memory_config),
        ]

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        self.text_tokenizer = None
        self.sound_tokenizer = None
        self.reasoner = None
        self._reasoner_component_cache_key = None
        self._training_pipeline_adapter = None

    def _encode_prompts(self, prompts: list | str, is_negative_prompt: bool = False):
        if isinstance(prompts, str):
            prompts = [prompts]
        contexts = getattr(self, "_current_prompt_contexts", None) or [{} for _ in prompts]
        if len(contexts) != len(prompts):
            raise ValueError(f"Cosmos3 received {len(prompts)} prompts but {len(contexts)} prompt contexts.")

        reasoner = self._load_reasoner()
        samples = []
        logger.info("Encoding Cosmos3 reasoner cache for %s prompt(s).", len(prompts))
        for prompt, context in zip(prompts, contexts):
            signature = self._cosmos3_text_cache_signature(prompt=prompt, metadata=context)
            required = ("cosmos3_num_frames", "cosmos3_height", "cosmos3_width", "cosmos3_fps")
            missing = [key for key in required if key not in context]
            if missing:
                raise ValueError(
                    "Cosmos3 reasoner cache requires geometry metadata for every prompt. "
                    f"Missing {missing}; prompt={prompt!r}."
                )
            input_ids = self._tokenize_reasoner_prompt(
                prompt=prompt,
                num_frames=int(context["cosmos3_num_frames"]),
                height=int(context["cosmos3_height"]),
                width=int(context["cosmos3_width"]),
                fps=float(context["cosmos3_fps"]),
            )
            text_segment = self._prepare_reasoner_text_segment(input_ids, self.accelerator.device)
            memory_state = reasoner(
                input_ids=text_segment["input_ids"],
                position_ids=text_segment["text_mrope_ids"],
            )
            samples.append(
                {
                    "cache_version": 1,
                    "cache_signature": signature,
                    "input_ids": text_segment["input_ids"].detach().cpu(),
                    "text_indexes": text_segment["text_indexes"].detach().cpu(),
                    "und_len": torch.tensor(text_segment["und_len"], dtype=torch.long),
                    "text_mrope_ids": text_segment["text_mrope_ids"].detach().cpu(),
                    "vision_start_temporal_offset": torch.tensor(
                        text_segment["vision_start_temporal_offset"], dtype=torch.float32
                    ),
                    "reasoner_memory_state": {"layer_kv": memory_state.layer_kv},
                }
            )
        logger.info("Finished Cosmos3 reasoner cache encoding for %s prompt(s).", len(samples))
        return {"cosmos3_reasoner_cache": samples[0] if len(samples) == 1 else samples}

    def uses_text_embeddings_cache(self) -> bool:
        return True

    @torch.no_grad()
    def encode_text_batch(
        self,
        text_batch: list,
        is_negative_prompt: bool = False,
        prompt_contexts: Optional[list[dict]] = None,
    ) -> dict:
        previous_context = getattr(self, "_current_prompt_contexts", None)
        self._current_prompt_contexts = prompt_contexts
        try:
            return self._encode_prompts(text_batch, is_negative_prompt)
        finally:
            self._current_prompt_contexts = previous_context

    def use_text_cache_dropout_sentinel(self) -> bool:
        return False

    def text_embed_cache_key(self) -> TextEmbedCacheKey:
        return TextEmbedCacheKey.DATASET_AND_FILENAME

    def text_embed_cache_metadata_for_sample(
        self,
        *,
        example: dict,
        latent: torch.Tensor,
        prompt: str,
        data_backend_id: str | None,
        dataset_relative_path: str | None,
    ) -> dict:
        fps = float(getattr(self.config, "framerate", 24.0) or 24.0)
        metadata = {
            "cosmos3_num_frames": int(latent.shape[-3]),
            "cosmos3_height": int(latent.shape[-2]) * 16,
            "cosmos3_width": int(latent.shape[-1]) * 16,
            "cosmos3_fps": fps,
            "cosmos3_reasoner_component": self._reasoner_component_id(),
        }
        metadata["cosmos3_cache_signature"] = self._cosmos3_text_cache_signature(prompt=prompt, metadata=metadata)
        return metadata

    def text_embed_cache_metadata_for_filepath(
        self,
        *,
        init_backend: dict,
        image_path: str,
        prompt: str,
        data_backend_id: str | None,
        dataset_relative_path: str | None,
    ) -> dict:
        metadata_backend = init_backend.get("metadata_backend")
        sample_metadata = {}
        if metadata_backend is not None:
            sample_metadata = metadata_backend.get_metadata_by_filepath(image_path) or {}
        target_size = sample_metadata.get("target_size") or sample_metadata.get("original_size")
        if not target_size or len(target_size) != 2:
            raise ValueError(f"Cosmos3 text cache precompute requires target_size metadata for {image_path}.")
        width, height = int(target_size[0]), int(target_size[1])
        num_frames = int(sample_metadata.get("bucket_frames") or sample_metadata.get("num_frames") or 1)
        metadata = {
            "cosmos3_num_frames": num_frames,
            "cosmos3_height": height,
            "cosmos3_width": width,
            "cosmos3_fps": float(sample_metadata.get("fps") or getattr(self.config, "framerate", 24.0) or 24.0),
            "cosmos3_reasoner_component": self._reasoner_component_id(),
        }
        metadata["cosmos3_cache_signature"] = self._cosmos3_text_cache_signature(prompt=prompt, metadata=metadata)
        return metadata

    def text_embed_cache_key_value(self, *, prompt: str, default_key: str, metadata: dict) -> str:
        signature = metadata.get("cosmos3_cache_signature") or self._cosmos3_text_cache_signature(
            prompt=prompt, metadata=metadata
        )
        return f"{default_key}:cosmos3:{signature}"

    def after_text_embed_cache_precompute(self):
        self._offload_reasoner_after_text_cache()

    def slice_text_embedding_for_cache(self, text_encoder_output: dict, batch_index: int, batch_size: int):
        cache = text_encoder_output.get("cosmos3_reasoner_cache")
        if isinstance(cache, list):
            if len(cache) != batch_size:
                raise ValueError(f"Cosmos3 reasoner cache batch has {len(cache)} entries, expected {batch_size}.")
            return {"cosmos3_reasoner_cache": cache[batch_index]}
        return None

    def collate_prompt_embeds(self, text_encoder_output: list[dict]) -> dict:
        if not text_encoder_output or "cosmos3_reasoner_cache" not in text_encoder_output[0]:
            return {}
        return {"cosmos3_reasoner_cache": [entry["cosmos3_reasoner_cache"] for entry in text_encoder_output]}

    def _is_i2v_flavour(self) -> bool:
        flavour = str(getattr(self.config, "model_flavour", None) or self.DEFAULT_MODEL_FLAVOUR).lower()
        return flavour == "super-i2v"

    def requires_conditioning_dataset(self) -> bool:
        return self._is_i2v_flavour() or super().requires_conditioning_dataset()

    def requires_conditioning_latents(self) -> bool:
        return self._is_i2v_flavour()

    def requires_conditioning_validation_inputs(self) -> bool:
        return self._is_i2v_flavour() or super().requires_conditioning_validation_inputs()

    def requires_validation_i2v_samples(self) -> bool:
        return self._is_i2v_flavour() or super().requires_validation_i2v_samples()

    @staticmethod
    def _cosmos3_text_cache_signature(prompt: str, metadata: dict) -> str:
        prompt_for_cache = prompt
        if all(
            metadata.get(key) is not None for key in ("cosmos3_num_frames", "cosmos3_height", "cosmos3_width", "cosmos3_fps")
        ):
            prompt_for_cache = Cosmos3OmniPipeline._build_generation_json_prompt(
                prompt,
                num_frames=int(metadata["cosmos3_num_frames"]),
                height=int(metadata["cosmos3_height"]),
                width=int(metadata["cosmos3_width"]),
                fps=float(metadata["cosmos3_fps"]),
            )
        payload = {
            "prompt": prompt_for_cache,
            "num_frames": metadata.get("cosmos3_num_frames"),
            "height": metadata.get("cosmos3_height"),
            "width": metadata.get("cosmos3_width"),
            "fps": metadata.get("cosmos3_fps"),
            "reasoner_component": metadata.get("cosmos3_reasoner_component"),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def convert_negative_text_embed_for_pipeline(self, text_embedding):
        return {}

    def convert_text_embed_for_pipeline(self, text_embedding):
        return {}

    def _reasoner_component_id(self) -> str:
        override = getattr(self.config, "cosmos3_reasoner_component", None)
        if override not in (None, "", "None", "auto"):
            return str(override)
        flavour = str(getattr(self.config, "model_flavour", None) or self.DEFAULT_MODEL_FLAVOUR).lower()
        if flavour not in COSMOS3_REASONER_COMPONENTS:
            raise ValueError(
                f"Cosmos3 reasoner component is set to auto, but model_flavour={flavour!r} is not supported. "
                f"Expected one of {sorted(COSMOS3_REASONER_COMPONENTS)}."
            )
        return COSMOS3_REASONER_COMPONENTS[flavour]

    def _generator_component_id(self) -> str:
        override = getattr(self.config, "cosmos3_generator_component", None)
        if override not in (None, "", "None", "auto"):
            return str(override)
        flavour = str(getattr(self.config, "model_flavour", None) or self.DEFAULT_MODEL_FLAVOUR).lower()
        if flavour not in COSMOS3_GENERATOR_COMPONENTS:
            raise ValueError(
                f"Cosmos3 generator component is set to auto, but model_flavour={flavour!r} is not supported. "
                f"Expected one of {sorted(COSMOS3_GENERATOR_COMPONENTS)}."
            )
        return COSMOS3_GENERATOR_COMPONENTS[flavour]

    def load_model(self, move_to_device: bool = True):
        custom_transformer_path = getattr(self.config, "pretrained_transformer_model_name_or_path", None)
        if custom_transformer_path in (None, "", "None"):
            self.config.pretrained_transformer_model_name_or_path = self._generator_component_id()
            self.config.pretrained_transformer_subfolder = None
        return super().load_model(move_to_device=move_to_device)

    def _load_reasoner(self):
        component_id = self._reasoner_component_id()
        device = torch.device(self.accelerator.device)
        cache_key = (
            component_id,
            str(device),
            self.config.weight_dtype,
            getattr(self.config, "revision", None),
        )
        if self.reasoner is not None and self._reasoner_component_cache_key == cache_key:
            if self.reasoner.device != device:
                logger.info("Moving Cosmos3 reasoner to %s for text cache encoding.", device)
                self.reasoner.to(device=device, dtype=self.config.weight_dtype)
            return self.reasoner
        logger.info("Loading Cosmos3 reasoner component %s on %s.", component_id, device)
        self.reasoner = Cosmos3Reasoner.from_pretrained(
            component_id,
            device=device,
            dtype=self.config.weight_dtype,
            revision=getattr(self.config, "revision", None),
        )
        self._reasoner_component_cache_key = cache_key
        return self.reasoner

    def _offload_reasoner_after_text_cache(self):
        if self.reasoner is None:
            return
        if self.reasoner.device.type == "cuda":
            logger.info("Moving Cosmos3 reasoner back to CPU after text cache encoding.")
            self.reasoner.to(device=torch.device("cpu"))
            torch.cuda.empty_cache()

    def _tokenize_reasoner_prompt(self, *, prompt: str, num_frames: int, height: int, width: int, fps: float) -> list[int]:
        self._load_text_tokenizer()
        is_image = num_frames == 1
        text = Cosmos3OmniPipeline._build_generation_json_prompt(
            prompt,
            num_frames=num_frames,
            fps=fps,
            height=height,
            width=width,
        )
        conversations = []
        if getattr(self.config, "default_use_system_prompt", True):
            conversations.append({"role": "system", "content": _SYSTEM_PROMPT_IMAGE if is_image else _SYSTEM_PROMPT_VIDEO})
        conversations.append({"role": "user", "content": text})
        encodings = self.text_tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=False,
            return_dict=True,
        )
        return list(encodings.input_ids) + [
            self.text_tokenizer.eos_token_id,
            self.text_tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        ]

    def _prepare_reasoner_text_segment(self, input_ids: list[int], device: torch.device | str) -> dict:
        transformer = self.unwrap_model(self.model) if self.model is not None else None
        config = transformer.config if transformer is not None else self._load_reasoner().config
        und_len = len(input_ids)
        text_mrope_ids, next_mrope_offset = get_3d_mrope_ids_text_tokens(
            num_tokens=und_len,
            temporal_offset=0,
            use_float_positions=getattr(config, "enable_fps_modulation", True),
        )
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "text_indexes": torch.arange(und_len, dtype=torch.long, device=device),
            "und_len": und_len,
            "text_mrope_ids": text_mrope_ids.to(device),
            "vision_start_temporal_offset": next_mrope_offset
            + getattr(config, "unified_3d_mrope_temporal_modality_margin", 15000),
        }

    def _load_preprocessor(self):
        self._load_text_tokenizer()

    def _load_text_tokenizer(self):
        if self.text_tokenizer is not None:
            return self.text_tokenizer

        self.text_tokenizer = AutoTokenizer.from_pretrained(
            get_model_config_path(self.config.model_family, self.config.pretrained_model_name_or_path),
            subfolder="text_tokenizer",
            revision=self.config.revision,
        )
        self.tokenizers = [self.text_tokenizer]
        return self.text_tokenizer

    def setup_training_noise_schedule(self):
        self.noise_schedule = UniPCMultistepScheduler.from_pretrained(
            get_model_config_path(self.config.model_family, self.config.pretrained_model_name_or_path),
            subfolder="scheduler",
        )
        return self.config, self.noise_schedule

    def supports_audio_inputs(self) -> bool:
        return True

    def uses_audio_latents(self) -> bool:
        return True

    def get_vae_for_dataset_type(self, dataset_type: str):
        if dataset_type == "audio":
            self._load_sound_tokenizer(move_to_device=True)
            return self.sound_tokenizer
        return self.get_vae()

    def _resolve_vae_dtype(self):
        vae_dtype = getattr(self.config, "vae_dtype", None)
        if vae_dtype == "bf16":
            return torch.bfloat16
        if vae_dtype == "fp16":
            return torch.float16
        if vae_dtype == "fp32":
            return torch.float32
        return self.config.weight_dtype

    def _load_sound_tokenizer(self, move_to_device: bool = True):
        if self.sound_tokenizer is not None:
            return self.sound_tokenizer

        self.sound_tokenizer = Cosmos3AVAEAudioTokenizer.from_pretrained(
            get_model_config_path(self.config.model_family, self.config.pretrained_model_name_or_path),
            subfolder="sound_tokenizer",
            revision=self.config.revision,
        )
        if move_to_device:
            self.sound_tokenizer.to(self.accelerator.device, dtype=self._resolve_vae_dtype())
        return self.sound_tokenizer

    def freeze_reasoning_layers(self, transformer=None) -> list[str]:
        transformer = transformer if transformer is not None else getattr(self, "model", None)
        if transformer is None:
            return []
        if getattr(self, "model", None) is None:
            transformer = transformer.module if hasattr(transformer, "module") else transformer
        else:
            transformer = self.unwrap_model(transformer)
        if transformer is None:
            return []
        frozen = []
        for module_name in ("embed_tokens", "norm"):
            module = getattr(transformer, module_name, None)
            if module is not None:
                module.requires_grad_(False)
                module.eval()
                frozen.append(module_name)
        for layer_idx, layer in enumerate(getattr(transformer, "layers", [])):
            for module_name in ("input_layernorm", "post_attention_layernorm", "mlp"):
                module = getattr(layer, module_name, None)
                if module is not None:
                    module.requires_grad_(False)
                    module.eval()
                    frozen.append(f"layers.{layer_idx}.{module_name}")
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue
            for module_name in ("to_q", "to_k", "to_v", "to_out", "norm_q", "norm_k"):
                module = getattr(attn, module_name, None)
                if module is not None:
                    module.requires_grad_(False)
                    module.eval()
                    frozen.append(f"layers.{layer_idx}.self_attn.{module_name}")
        if self.reasoner is not None:
            self.reasoner.requires_grad_(False)
            self.reasoner.eval()
            frozen.append("reasoner")
        return frozen

    def apply_model_specific_freeze(self):
        self.freeze_reasoning_layers()

    def freeze_components(self):
        super().freeze_components()
        self.apply_model_specific_freeze()

    def add_lora_adapter(self):
        result = super().add_lora_adapter()
        self.apply_model_specific_freeze()
        return result

    def pre_vae_encode_transform_sample(self, sample):
        if sample.ndim == 3:
            return sample
        return super().pre_vae_encode_transform_sample(sample)

    def encode_cache_batch(self, vae, samples, metadata_entries: Optional[list] = None):
        if isinstance(vae, Cosmos3AVAEAudioTokenizer):
            encoded = vae.encode(samples, return_dict=True, force_pad=True)
            return encoded.latent_dist.mode()
        return super().encode_cache_batch(vae, samples, metadata_entries=metadata_entries)

    def scale_vae_latents_for_cache(self, latents, vae):
        if isinstance(vae, Cosmos3AVAEAudioTokenizer):
            return latents
        mean = torch.tensor(vae.config.latents_mean, device=latents.device, dtype=latents.dtype)
        inv_std = 1.0 / torch.tensor(vae.config.latents_std, device=latents.device, dtype=latents.dtype)
        return (latents - mean.view(1, -1, 1, 1, 1)) * inv_std.view(1, -1, 1, 1, 1)

    def prepare_batch(self, batch: dict, state: dict) -> dict:
        if not batch:
            return batch

        target_kwargs = {
            "device": self.accelerator.device,
            "dtype": self.config.weight_dtype,
        }
        latents = batch["latent_batch"].to(**target_kwargs)
        batch["latents"] = latents

        noise = torch.randn_like(latents)
        batch["noise"] = noise
        batch["input_noise"] = noise

        bsz = latents.shape[0]
        u = torch.normal(mean=0.0, std=1.0, size=(bsz,), device=latents.device)
        t = torch.sigmoid(u)
        batch["sigmas"] = t
        batch["timesteps"] = t * 1000.0
        batch["noisy_latents"] = self._interpolate_flow_latents(latents, noise, t)

        if self._is_i2v_flavour():
            conditioning_latents = batch.get("conditioning_latents")
            if conditioning_latents is None:
                raise ValueError("Cosmos3 super-i2v training requires conditioning_latents in the batch.")
            if isinstance(conditioning_latents, list):
                selected_index = 0
                latent_types = batch.get("conditioning_latents_type")
                if (
                    batch.get("conditioning_type") == "reference_strict"
                    and isinstance(latent_types, list)
                    and "reference_strict" in latent_types
                ):
                    selected_index = latent_types.index("reference_strict")
                conditioning_latents = conditioning_latents[selected_index] if conditioning_latents else None
                if isinstance(latent_types, list) and selected_index < len(latent_types):
                    batch["conditioning_latents_type"] = latent_types[selected_index]
            if conditioning_latents is None:
                raise ValueError("Cosmos3 super-i2v training requires conditioning_latents in the batch.")
            conditioning_latents = conditioning_latents.to(**target_kwargs)
            if conditioning_latents.ndim == 4:
                conditioning_latents = conditioning_latents.unsqueeze(2)
            if conditioning_latents.ndim != 5:
                raise ValueError(
                    "Cosmos3 super-i2v conditioning_latents must be [B,C,H,W] or [B,C,F,H,W], "
                    f"got {tuple(conditioning_latents.shape)}."
                )
            if conditioning_latents.shape[0] != latents.shape[0] or conditioning_latents.shape[1] != latents.shape[1]:
                raise ValueError(
                    "Cosmos3 super-i2v conditioning_latents must match target batch and channel dimensions. "
                    f"Got {tuple(conditioning_latents.shape)} vs target {tuple(latents.shape)}."
                )
            if conditioning_latents.shape[-2:] != latents.shape[-2:]:
                raise ValueError(
                    "Cosmos3 super-i2v conditioning_latents must match target latent height and width. "
                    f"Got {tuple(conditioning_latents.shape[-2:])} vs target {tuple(latents.shape[-2:])}."
                )
            batch["conditioning_latents"] = conditioning_latents
            batch["noisy_latents"][:, :, 0] = conditioning_latents[:, :, 0]
            vision_loss_mask = torch.ones_like(latents[:, :1])
            vision_loss_mask[:, :, 0] = 0
            batch["vision_loss_mask"] = vision_loss_mask

        audio_latents = batch.get("audio_latent_batch")
        if torch.is_tensor(audio_latents):
            audio_latents = audio_latents.to(**target_kwargs)
            audio_noise = torch.randn_like(audio_latents)
            batch["audio_latents"] = audio_latents
            batch["audio_noise"] = audio_noise
            batch["audio_noisy_latents"] = self._interpolate_flow_latents(audio_latents, audio_noise, t)
            batch["audio_timesteps"] = batch["timesteps"]
            audio_latent_mask = batch.get("audio_latent_mask")
            if audio_latent_mask is not None and hasattr(audio_latent_mask, "to"):
                batch["audio_latent_mask"] = audio_latent_mask.to(device=latents.device, dtype=latents.dtype)

        batch = self.prepare_batch_conditions(batch=batch, state=state)
        return batch

    @staticmethod
    def _interpolate_flow_latents(latents: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        view_shape = (latents.shape[0],) + (1,) * (latents.ndim - 1)
        t_view = t.view(view_shape).to(device=latents.device, dtype=latents.dtype)
        return (1.0 - t_view) * noise + t_view * latents

    def _get_training_pipeline_adapter(self):
        self._load_text_tokenizer()
        if self._training_pipeline_adapter is None:
            self._training_pipeline_adapter = Cosmos3OmniPipeline(
                transformer=self.unwrap_model(self.model),
                text_tokenizer=self.text_tokenizer,
                vae=self.get_vae(),
                scheduler=self.noise_schedule,
                sound_tokenizer=self.sound_tokenizer,
            )
        else:
            self._training_pipeline_adapter.transformer = self.unwrap_model(self.model)
            self._training_pipeline_adapter.vae = self.get_vae()
            self._training_pipeline_adapter.sound_tokenizer = self.sound_tokenizer
        return self._training_pipeline_adapter

    def get_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True):
        self._load_text_tokenizer()
        if pipeline_type not in self.PIPELINE_CLASSES:
            raise NotImplementedError(f"Pipeline type {pipeline_type} not defined in {self.__class__.__name__}.")

        if pipeline_type in self.pipelines:
            pipeline = self.pipelines[pipeline_type]
            pipeline.transformer = self.unwrap_model(self.model)
            pipeline.vae = self.get_vae()
            pipeline.text_tokenizer = self.text_tokenizer
            pipeline.sound_tokenizer = self.sound_tokenizer
            return pipeline

        pipeline_class = self.PIPELINE_CLASSES[pipeline_type]
        scheduler = (
            self.noise_schedule.__class__.from_config(self.noise_schedule.config)
            if self.noise_schedule is not None
            else None
        )
        pipeline_kwargs = {
            "transformer": self.unwrap_model(self.model),
            "text_tokenizer": self.text_tokenizer,
            "vae": self.get_vae(),
            "scheduler": scheduler,
            "sound_tokenizer": self.sound_tokenizer,
        }
        if load_base_model:
            pipeline = pipeline_class.from_pretrained(
                self._model_config_path(),
                **pipeline_kwargs,
            )
        else:
            pipeline = pipeline_class(**pipeline_kwargs)
        self.pipelines[pipeline_type] = pipeline
        return pipeline

    def model_predict(self, prepared_batch):
        adapter = self._get_training_pipeline_adapter()
        noisy_latents = prepared_batch["noisy_latents"]
        timesteps = prepared_batch["timesteps"].to(device=noisy_latents.device, dtype=torch.float32)
        prompts = prepared_batch["prompts"]
        audio_noisy_latents = prepared_batch.get("audio_noisy_latents")
        audio_timesteps = prepared_batch.get("audio_timesteps", timesteps)
        fps = float(getattr(self.config, "framerate", 24.0) or 24.0)
        reasoner_cache_batch = (prepared_batch.get("text_encoder_output") or {}).get("cosmos3_reasoner_cache")
        condition_frame_indexes = [0] if self._is_i2v_flavour() else []

        vision_predictions = []
        sound_predictions = [] if torch.is_tensor(audio_noisy_latents) else None
        for sample_idx, (prompt, sample_latents) in enumerate(zip(prompts, noisy_latents)):
            reasoner_cache = reasoner_cache_batch[sample_idx] if isinstance(reasoner_cache_batch, list) else None
            model_out = self._predict_single_sample(
                adapter=adapter,
                prompt=prompt,
                vision_tokens=sample_latents.unsqueeze(0),
                vision_timestep=timesteps[sample_idx],
                fps=fps,
                sound_tokens=audio_noisy_latents[sample_idx] if torch.is_tensor(audio_noisy_latents) else None,
                sound_timestep=audio_timesteps[sample_idx] if torch.is_tensor(audio_timesteps) else timesteps[sample_idx],
                reasoner_cache=reasoner_cache,
                condition_frame_indexes=condition_frame_indexes,
            )
            vision_predictions.append(model_out[0][0])
            if sound_predictions is not None:
                if model_out[1] is None:
                    raise RuntimeError(
                        "Cosmos3 transformer returned no sound prediction for a batch containing sound tokens."
                    )
                sound_predictions.append(model_out[1][0])

        output = {"model_prediction": torch.cat(vision_predictions, dim=0)}
        if sound_predictions is not None:
            output["audio_model_prediction"] = torch.stack(sound_predictions, dim=0)
        return output

    def _predict_single_sample(
        self,
        *,
        adapter: Cosmos3OmniPipeline,
        prompt: str,
        vision_tokens: torch.Tensor,
        vision_timestep: torch.Tensor,
        fps: float,
        sound_tokens: torch.Tensor | None = None,
        sound_timestep: torch.Tensor | None = None,
        reasoner_cache: dict | None = None,
        condition_frame_indexes: list[int] | None = None,
    ):
        device = vision_tokens.device
        _, _, latent_t, latent_h, latent_w = vision_tokens.shape
        height = latent_h * adapter.vae_scale_factor_spatial
        width = latent_w * adapter.vae_scale_factor_spatial

        reasoner_memory_state = None
        if reasoner_cache is None:
            cond_input_ids, _ = adapter.tokenize_prompt(
                prompt=prompt,
                negative_prompt="",
                num_frames=latent_t,
                height=height,
                width=width,
                fps=fps,
            )
            text_segment = adapter._prepare_text_segment(cond_input_ids, device)
            reasoner_memory_state = self._load_reasoner()(
                input_ids=text_segment["input_ids"],
                position_ids=text_segment["text_mrope_ids"],
            )
        else:
            expected_signature = self._cosmos3_text_cache_signature(
                prompt=prompt,
                metadata={
                    "cosmos3_num_frames": int(latent_t),
                    "cosmos3_height": int(height),
                    "cosmos3_width": int(width),
                    "cosmos3_fps": float(fps),
                    "cosmos3_reasoner_component": self._reasoner_component_id(),
                },
            )
            if reasoner_cache.get("cache_signature") != expected_signature:
                raise ValueError("Cosmos3 reasoner cache signature does not match the current sample geometry.")
            text_segment = {
                "input_ids": reasoner_cache["input_ids"].to(device=device, dtype=torch.long),
                "text_indexes": reasoner_cache["text_indexes"].to(device=device, dtype=torch.long),
                "und_len": int(reasoner_cache["und_len"].item()),
                "text_mrope_ids": reasoner_cache["text_mrope_ids"].to(device=device),
                "vision_start_temporal_offset": float(reasoner_cache["vision_start_temporal_offset"].item()),
            }
            reasoner_memory_state = Cosmos3ReasonerMemoryState(layer_kv=reasoner_cache["reasoner_memory_state"]["layer_kv"])
        vision_segment = adapter._prepare_vision_segment(
            input_vision_tokens=vision_tokens,
            has_image_condition=bool(condition_frame_indexes),
            mrope_offset=text_segment["vision_start_temporal_offset"],
            vision_fps=fps,
            curr=text_segment["und_len"],
            device=device,
            condition_frame_indexes=condition_frame_indexes or [],
        )

        position_ids = [text_segment["text_mrope_ids"], vision_segment["vision_mrope_ids"]]
        sequence_length = text_segment["und_len"] + vision_segment["num_vision_tokens"]
        sound_kwargs = {}
        if sound_tokens is not None:
            sound_segment = adapter._prepare_sound_segment(
                input_sound_tokens=sound_tokens,
                mrope_offset=text_segment["vision_start_temporal_offset"],
                sound_fps=fps,
                curr=sequence_length,
                device=device,
            )
            position_ids.append(sound_segment["sound_mrope_ids"])
            sequence_length += sound_segment["sound_len"]
            if sound_timestep is None:
                sound_timestep = vision_timestep
            sound_kwargs = {
                "sound_tokens": [sound_tokens],
                "sound_token_shapes": sound_segment["sound_token_shapes"],
                "sound_sequence_indexes": sound_segment["sound_sequence_indexes"],
                "sound_mse_loss_indexes": sound_segment["sound_mse_loss_indexes"],
                "sound_timesteps": sound_timestep.expand(sound_segment["sound_len"]),
                "sound_noisy_frame_indexes": sound_segment["sound_noisy_frame_indexes"],
            }

        position_ids = torch.cat(position_ids, dim=1)
        return self.model(
            input_ids=text_segment["input_ids"],
            text_indexes=text_segment["text_indexes"],
            position_ids=position_ids,
            und_len=text_segment["und_len"],
            sequence_length=sequence_length,
            vision_tokens=[vision_tokens],
            vision_token_shapes=vision_segment["vision_token_shapes"],
            vision_sequence_indexes=vision_segment["vision_sequence_indexes"],
            vision_mse_loss_indexes=vision_segment["vision_mse_loss_indexes"],
            vision_timesteps=vision_timestep.expand(vision_segment["num_noisy_vision_tokens"]),
            vision_noisy_frame_indexes=vision_segment["vision_noisy_frame_indexes"],
            reasoner_memory_state=reasoner_memory_state,
            return_dict=False,
            **sound_kwargs,
        )

    def loss(self, prepared_batch, model_output, apply_conditioning_mask=True):
        video_target = prepared_batch["noise"] - prepared_batch["latents"]
        video_prediction = model_output["model_prediction"].float()
        video_target = video_target.float()
        vision_loss_mask = prepared_batch.get("vision_loss_mask")
        if vision_loss_mask is not None:
            vision_loss_mask = vision_loss_mask.to(device=video_prediction.device, dtype=video_prediction.dtype)
            video_loss = F.mse_loss(video_prediction, video_target, reduction="none")
            video_loss = (video_loss * vision_loss_mask).sum() / (
                vision_loss_mask.sum().clamp_min(1.0) * video_prediction.shape[1]
            )
        else:
            video_loss = F.mse_loss(
                video_prediction,
                video_target,
                reduction="mean",
            )

        if os.environ.get("SIMPLETUNER_COSMOS3_LOSS_DEBUG") or getattr(self.config, "cosmos3_loss_debug", False):
            count = getattr(self, "_cosmos3_loss_debug_count", 0)
            if count < 8:
                self._cosmos3_loss_debug_count = count + 1
                with torch.no_grad():
                    flipped_target = prepared_batch["latents"].float() - prepared_batch["noise"].float()
                    flipped_loss = F.mse_loss(video_prediction, flipped_target, reduction="mean")
                    zero_loss = F.mse_loss(torch.zeros_like(video_target), video_target, reduction="mean")
                    latents = prepared_batch["latents"].float()
                    noise = prepared_batch["noise"].float()
                    sigmas = prepared_batch.get("sigmas")
                    timesteps = prepared_batch.get("timesteps")
                    sigma_summary = "n/a"
                    if torch.is_tensor(sigmas):
                        sigmas_f = sigmas.float()
                        sigma_summary = (
                            f"{sigmas_f.mean().item():.4f}/" f"{sigmas_f.min().item():.4f}/" f"{sigmas_f.max().item():.4f}"
                        )
                    timestep_summary = "n/a"
                    if torch.is_tensor(timesteps):
                        timesteps_f = timesteps.float()
                        timestep_summary = (
                            f"{timesteps_f.mean().item():.2f}/"
                            f"{timesteps_f.min().item():.2f}/"
                            f"{timesteps_f.max().item():.2f}"
                        )
                    logger.warning(
                        "Cosmos3 loss debug #%s: loss=%.6f flipped_loss=%.6f zero_target_loss=%.6f "
                        "pred_mean=%.6f pred_std=%.6f target_mean=%.6f target_std=%.6f "
                        "latent_mean=%.6f latent_std=%.6f noise_mean=%.6f noise_std=%.6f "
                        "sigma_mean/min/max=%s timestep_mean/min/max=%s",
                        count + 1,
                        video_loss.detach().float().item(),
                        flipped_loss.detach().float().item(),
                        zero_loss.detach().float().item(),
                        video_prediction.mean().item(),
                        video_prediction.std().item(),
                        video_target.mean().item(),
                        video_target.std().item(),
                        latents.mean().item(),
                        latents.std().item(),
                        noise.mean().item(),
                        noise.std().item(),
                        sigma_summary,
                        timestep_summary,
                    )

        if "audio_model_prediction" not in model_output:
            return video_loss

        audio_target = prepared_batch["audio_noise"] - prepared_batch["audio_latents"]
        audio_loss = F.mse_loss(
            model_output["audio_model_prediction"].float(),
            audio_target.float(),
            reduction="none",
        )
        audio_mask = prepared_batch.get("audio_latent_mask")
        if audio_mask is not None:
            audio_loss = audio_loss * audio_mask.to(device=audio_loss.device, dtype=audio_loss.dtype).view(-1, 1, 1)
            denom = audio_mask.sum().clamp_min(1.0) * audio_loss[0].numel()
            audio_loss = audio_loss.sum() / denom
        else:
            audio_loss = audio_loss.mean()
        return video_loss + audio_loss

    def check_user_config(self):
        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )

        if self.config.aspect_bucket_alignment != 16:
            logger.warning(
                f"{self.NAME} requires an alignment value of 16px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 16

        if self.config.tokenizer_max_length is None:
            self.config.tokenizer_max_length = 4096
            logger.info(f"Setting tokenizer max length to {self.config.tokenizer_max_length}")
        if not getattr(self.config, "text_cache_disable", False) and not getattr(self.config, "text_cache_ondemand", False):
            logger.info("Cosmos3 reasoner K/V text cache will be pre-computed from bucket metadata.")

        self.config.pretrained_vae_model_name_or_path = None


ModelRegistry.register("cosmos3", Cosmos3Image)
