# HunyuanVideo 1.5 integration (AGPL-3.0-or-later), SimpleTuner © 2025

import logging
import os
from typing import Dict, List, Optional

import torch
from diffusers.guiders import ClassifierFreeGuidance
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from transformers import (
    ByT5Tokenizer,
    Qwen2_5_VLTextModel,
    Qwen2Tokenizer,
    SiglipImageProcessor,
    SiglipVisionModel,
    T5EncoderModel,
)

from simpletuner.helpers.acceleration import AccelerationBackend, AccelerationPreset
from simpletuner.helpers.data_backend.dataset_types import DatasetType
from simpletuner.helpers.models.common import ModelTypes, PipelineTypes, PredictionTypes, VideoModelFoundation
from simpletuner.helpers.models.hunyuanvideo.autoencoder import AutoencoderKLConv3D
from simpletuner.helpers.models.hunyuanvideo.commons import PIPELINE_CONFIGS, TRANSFORMER_VERSION_TO_SR_VERSION
from simpletuner.helpers.models.hunyuanvideo.pipeline import HunyuanVideo15Pipeline
from simpletuner.helpers.models.hunyuanvideo.pipeline_i2v import HunyuanVideo15ImageToVideoPipeline
from simpletuner.helpers.models.hunyuanvideo.transformer import HunyuanVideo15Transformer3DModel
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.musubi_block_swap import apply_musubi_pretrained_defaults
from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class HunyuanVideo(VideoModelFoundation):
    """
    Hunyuan Video 1.5 (8.3B) text-to-video and image-to-video transformer.
    """

    NAME = "HunyuanVideo"
    MODEL_DESCRIPTION = "Text-to-video / image-to-video flow-matching transformer (8.3B)"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKLConv3D
    LATENT_CHANNEL_COUNT = 32
    DEFAULT_NOISE_SCHEDULER = "flow_match_euler"
    MODEL_CLASS = HunyuanVideo15Transformer3DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: HunyuanVideo15Pipeline,
        PipelineTypes.IMG2VIDEO: HunyuanVideo15ImageToVideoPipeline,
    }
    DEFAULT_PIPELINE_TYPE = PipelineTypes.TEXT2IMG
    DEFAULT_MODEL_FLAVOUR = "t2v-480p"
    HUGGINGFACE_PATHS: Dict[str, str] = {
        "t2v-480p": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        "t2v-720p": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_t2v",
        "t2v-480p-distilled": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v_distilled",
        "i2v-480p": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v",
        "i2v-720p": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v",
        "i2v-480p-distilled": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v_distilled",
        "i2v-720p-distilled": "hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-720p_i2v_distilled",
    }
    MODEL_LICENSE = "agpl-3.0"

    # Component repositories - direct loading without subfolders
    VAE_REPO = "DiffusersVersionsOfModels/HunyuanVideo-1.5-vae"
    GLYPH_BYT5_REPO = "DiffusersVersionsOfModels/Glyph-ByT5"
    TEXT_ENCODER_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
    VISION_ENCODER_REPO = "google/siglip-so400m-patch14-384"

    # Transformer version mapping for pipeline configs
    TRANSFORMER_VERSIONS: Dict[str, str] = {
        "t2v-480p": "480p_t2v",
        "t2v-720p": "720p_t2v",
        "t2v-480p-distilled": "480p_t2v",
        "i2v-480p": "480p_i2v",
        "i2v-720p": "720p_i2v",
        "i2v-480p-distilled": "480p_i2v",
        "i2v-720p-distilled": "720p_i2v",
    }
    STRICT_I2V_FLAVOURS = ("i2v-480p", "i2v-720p", "i2v-480p-distilled", "i2v-720p-distilled")

    # Only required to satisfy encode_text_batch checks; loading is handled manually.
    TEXT_ENCODER_CONFIGURATION = {"text_encoder": {"name": "Hunyuan LLM"}}
    # Align LoRA target modules with the actual attention layers used by the transformer blocks.
    DEFAULT_LORA_TARGET = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
    ]

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        # HunyuanVideo has 54 transformer layers (double blocks)
        return 53

    @classmethod
    def get_acceleration_presets(cls) -> List[AccelerationPreset]:
        return [
            # Basic tab - RamTorch presets
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="basic",
                name="RamTorch - Basic",
                description="Streams half of transformer block weights from CPU RAM.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~30%",
                tradeoff_speed="Increases training time by ~20%",
                tradeoff_notes="Requires 64GB+ system RAM. CUDA/ROCm only.",
                requires_cuda=True,
                requires_min_system_ram_gb=64,
                config={
                    "ramtorch": True,
                    "ramtorch_target_modules": "transformer_blocks.0-26.*",
                },
            ),
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="aggressive",
                name="RamTorch - Aggressive",
                description="Streams all transformer block weights from CPU RAM.",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~65%",
                tradeoff_speed="Increases training time by ~55%",
                tradeoff_notes="Requires 64GB+ system RAM. CUDA/ROCm only.",
                requires_cuda=True,
                requires_min_system_ram_gb=64,
                config={
                    "ramtorch": True,
                    "ramtorch_target_modules": "*",
                },
            ),
            # Basic tab - Block swap presets
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="light",
                name="Block Swap - Light",
                description="Swaps 14 of 54 blocks (~25%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~20%",
                tradeoff_speed="Increases training time by ~15%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={"musubi_blocks_to_swap": 14},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="balanced",
                name="Block Swap - Balanced",
                description="Swaps 27 of 54 blocks (~50%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~45%",
                tradeoff_speed="Increases training time by ~30%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={"musubi_blocks_to_swap": 27},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="aggressive",
                name="Block Swap - Aggressive",
                description="Swaps 40 of 54 blocks (~75%).",
                tab="basic",
                tradeoff_vram="Reduces VRAM by ~65%",
                tradeoff_speed="Increases training time by ~50%",
                tradeoff_notes="Requires 64GB+ system RAM.",
                requires_min_system_ram_gb=64,
                config={"musubi_blocks_to_swap": 40},
            ),
            # Advanced tab - DeepSpeed presets
            AccelerationPreset(
                backend=AccelerationBackend.DEEPSPEED_ZERO_1,
                level="zero1",
                name="DeepSpeed ZeRO Stage 1",
                description="Shards optimizer states across GPUs.",
                tab="advanced",
                tradeoff_vram="Reduces optimizer memory by ~75% per GPU",
                tradeoff_speed="Minimal overhead",
                tradeoff_notes="Requires multi-GPU. Not compatible with FSDP.",
                requires_cuda=True,
                config={"deepspeed_config": "zero1"},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.DEEPSPEED_ZERO_2,
                level="zero2",
                name="DeepSpeed ZeRO Stage 2",
                description="Shards optimizer states and gradients across GPUs.",
                tab="advanced",
                tradeoff_vram="Reduces memory by ~85% per GPU",
                tradeoff_speed="Moderate overhead from gradient sync",
                tradeoff_notes="Requires multi-GPU. Not compatible with FSDP.",
                requires_cuda=True,
                config={"deepspeed_config": "zero2"},
            ),
        ]

    def __init__(self, config: dict, accelerator):
        super().__init__(config, accelerator)
        self._transformer_version = self._resolve_transformer_version()
        self._sr_version = TRANSFORMER_VERSION_TO_SR_VERSION.get(self._transformer_version)
        if getattr(self.config, "pretrained_transformer_subfolder", None) is None:
            # Default to the standard diffusers transformer folder layout.
            self.config.pretrained_transformer_subfolder = self.MODEL_SUBFOLDER
        if getattr(self.config, "flow_schedule_shift", None) is None:
            default_cfg = PIPELINE_CONFIGS.get(self._transformer_version, {})
            self.config.flow_schedule_shift = default_cfg.get("flow_shift", 7.0)
        if getattr(self.config, "validation_guidance", None) is None:
            default_cfg = PIPELINE_CONFIGS.get(self._transformer_version, {})
            self.config.validation_guidance = default_cfg.get("guidance_scale", 6.0)

    def _resolve_transformer_version(self) -> str:
        flavour = getattr(self.config, "model_flavour", self.DEFAULT_MODEL_FLAVOUR) or self.DEFAULT_MODEL_FLAVOUR
        if flavour not in self.TRANSFORMER_VERSIONS:
            raise ValueError(
                f"Unsupported HunyuanVideo flavour '{flavour}'. Expected one of {list(self.TRANSFORMER_VERSIONS)}"
            )
        return self.TRANSFORMER_VERSIONS[flavour]

    def _is_i2v_like_flavour(self) -> bool:
        flavour = getattr(self.config, "model_flavour", None) or self.DEFAULT_MODEL_FLAVOUR
        if not flavour:
            return False
        try:
            return str(flavour).strip().lower().startswith("i2v")
        except Exception:
            return False

    def requires_conditioning_dataset(self) -> bool:
        return self._is_i2v_like_flavour() or super().requires_conditioning_dataset()

    def requires_conditioning_latents(self) -> bool:
        return self._is_i2v_like_flavour()

    def setup_training_noise_schedule(self):
        self.noise_schedule = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=self.config.flow_schedule_shift,
        )
        return self.config, self.noise_schedule

    def load_text_encoder(self, move_to_device: bool = True):
        """
        Load the Qwen2.5 VL text encoder and ByT5 glyph encoder.
        """
        device = self.accelerator.device if move_to_device else torch.device("cpu")
        qwen_path = getattr(self.config, "hunyuan_text_encoder_path", None) or self.TEXT_ENCODER_REPO

        logger.info(f"Loading HunyuanVideo text encoder from {qwen_path}")
        tokenizer = Qwen2Tokenizer.from_pretrained(qwen_path)
        text_encoder = Qwen2_5_VLTextModel.from_pretrained(qwen_path, torch_dtype=torch.bfloat16)
        text_encoder.requires_grad_(False)
        if move_to_device:
            text_encoder = text_encoder.to(device)

        glyph_repo = getattr(self.config, "glyph_byt5_repo", self.GLYPH_BYT5_REPO)
        fallback_glyph_repo = getattr(self.config, "glyph_byt5_fallback_repo", "google/byt5-small")
        logger.info(f"Loading Glyph ByT5 encoder from {glyph_repo}")
        try:
            byt5_tokenizer = ByT5Tokenizer.from_pretrained(glyph_repo)
            byt5_model = T5EncoderModel.from_pretrained(glyph_repo, torch_dtype=torch.bfloat16)
        except OSError as e:
            logger.warning("Failed to load Glyph ByT5 from %s (%s). Falling back to %s.", glyph_repo, e, fallback_glyph_repo)
            byt5_tokenizer = ByT5Tokenizer.from_pretrained(fallback_glyph_repo)
            byt5_model = T5EncoderModel.from_pretrained(fallback_glyph_repo, torch_dtype=torch.bfloat16)
        try:
            ckpt_path = hf_hub_download(glyph_repo, filename="checkpoints/byt5_model.pt", repo_type="model")
            glyph_state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(glyph_state, dict) and "model" in glyph_state:
                glyph_state = glyph_state["model"]
            model_state = byt5_model.state_dict()
            mapped_state = {}
            for key, value in glyph_state.items():
                if key == "embed_tokens.weight":
                    continue  # Token embeddings would require a matching tokenizer; skip to avoid shape mismatches.
                new_key = key
                if key.startswith("block.") or key.startswith("final_layer_norm"):
                    new_key = f"encoder.{key}"
                if new_key in model_state and model_state[new_key].shape == value.shape:
                    mapped_state[new_key] = value
            load_result = byt5_model.load_state_dict(mapped_state, strict=False)
            missing, unexpected = load_result.missing_keys, load_result.unexpected_keys
            logger.info(
                "Loaded Glyph ByT5 finetuned weights from %s (applied=%d, missing=%d, unexpected=%d).",
                ckpt_path,
                len(mapped_state),
                len(missing),
                len(unexpected),
            )
        except Exception as glyph_load_error:
            logger.debug("No Glyph ByT5 finetuned weights applied (%s).", glyph_load_error)
        byt5_model.requires_grad_(False)
        if move_to_device:
            byt5_model = byt5_model.to(device)

        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.text_encoder_2 = byt5_model
        self.tokenizer_2 = byt5_tokenizer

        # Maintain attributes expected by the training stack.
        self.text_encoders = [text_encoder]
        self.text_encoder_1 = text_encoder
        self.tokenizers = [tokenizer]
        self.tokenizer_1 = tokenizer
        self._image_encoder = None
        self._image_processor = None

    def _load_image_encoder(self, move_to_device: bool = True):
        if not hasattr(self, "_image_encoder"):
            self._image_encoder = None
            self._image_processor = None
        if self._image_encoder is not None and self._image_processor is not None:
            return self._image_encoder, self._image_processor

        device = self.accelerator.device if move_to_device else torch.device("cpu")
        repo = getattr(self.config, "hunyuan_image_encoder_path", None) or self.VISION_ENCODER_REPO
        logger.info(f"Loading SigLIP vision encoder from {repo}")
        feature_extractor = SiglipImageProcessor.from_pretrained(repo)
        image_encoder = SiglipVisionModel.from_pretrained(repo, torch_dtype=torch.bfloat16)
        image_encoder.requires_grad_(False)
        if move_to_device:
            image_encoder = image_encoder.to(device)

        self._image_encoder = image_encoder
        self._image_processor = feature_extractor
        return image_encoder, feature_extractor

    def load_text_tokenizer(self):
        """
        Tokenizers are loaded alongside the text encoders.
        """
        return

    def load_vae(self, move_to_device: bool = True):
        """
        Load VAE from the dedicated HunyuanVideo VAE repo.
        """
        from transformers.utils import ContextManagers

        from simpletuner.helpers.models.common import deepspeed_zero_init_disabled_context_manager

        logger.info(f"Loading {self.AUTOENCODER_CLASS.__name__} from {self.VAE_REPO}")
        self.vae = None
        self.config.vae_kwargs = {
            "pretrained_model_name_or_path": self.VAE_REPO,
            "subfolder": None,  # Direct repo, no subfolder
            "revision": self.config.revision,
            "force_upcast": False,
            "variant": self.config.variant,
            "enable_temporal_roll": getattr(self.config, "vae_enable_temporal_roll", False),
        }
        if getattr(self.config, "vae_enable_patch_conv", False):
            logger.info("Enabling VAE patch-based convolution for HunyuanVideo VAE.")
            self.config.vae_kwargs["enable_patch_conv"] = True
        if getattr(self.config, "vae_enable_temporal_roll", False):
            logger.info("Enabling temporal rolling for HunyuanVideo VAE to reduce VRAM.")
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            self.vae = self.AUTOENCODER_CLASS.from_pretrained(**self.config.vae_kwargs)
        if self.vae is None:
            raise ValueError(f"Could not load VAE from {self.VAE_REPO}.")
        if self.config.vae_enable_tiling and hasattr(self.vae, "enable_tiling"):
            logger.info("Enabling VAE tiling.")
            self.vae.enable_tiling()
        if move_to_device and self.vae.device != self.accelerator.device:
            _vae_dtype = torch.bfloat16
            if hasattr(self.config, "vae_dtype") and self.config.vae_dtype == "fp32":
                _vae_dtype = torch.float32
            self.vae.to(self.accelerator.device, dtype=_vae_dtype)
        self.AUTOENCODER_SCALING_FACTOR = getattr(self.vae.config, "scaling_factor", 1.0)

    def get_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True):
        """
        Return the HunyuanVideo inference pipeline with pre-loaded components.

        Unlike the base class implementation which uses from_pretrained(),
        we instantiate the pipeline directly since our transformer repos
        don't contain model_index.json.
        """

        def _is_meta(module: Optional[torch.nn.Module]) -> bool:
            if module is None or not isinstance(module, torch.nn.Module):
                return False
            device = getattr(module, "device", None)
            if device == "meta" or getattr(device, "type", None) == "meta":
                return True
            return self._module_has_meta_tensors(module)

        def _prune_meta_components(pipeline_obj):
            """
            Drop any meta-resident modules from the pipeline so .to() won't try to move empty tensors.
            """
            try:
                component_names = []
                if hasattr(pipeline_obj, "components") and isinstance(pipeline_obj.components, dict):
                    component_names.extend(list(pipeline_obj.components.keys()))
                if hasattr(pipeline_obj, "_modules"):
                    component_names.extend(list(pipeline_obj._modules.keys()))
                seen = set()
                for name in component_names:
                    if name in seen:
                        continue
                    seen.add(name)
                    module = getattr(pipeline_obj, name, None)
                    if _is_meta(module):
                        setattr(pipeline_obj, name, None)
            except Exception:
                logger.debug("Failed to prune meta components from HunyuanVideo pipeline.", exc_info=True)

        active_pipelines = getattr(self, "pipelines", {})
        if pipeline_type in active_pipelines:
            pipeline = active_pipelines[pipeline_type]
            if load_base_model and self.model is not None:
                pipeline.transformer = self.unwrap_model(self.model)
            return pipeline

        # Ensure required components are resident before constructing the pipeline (validation may reload after meta-offload).
        if self.vae is None or _is_meta(self.vae):
            self.load_vae(move_to_device=True)

        device = self.accelerator.device
        flow_shift = getattr(self.config, "flow_schedule_shift", 7.0)
        guidance_scale = getattr(self.config, "validation_guidance", 6.0)

        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=flow_shift)
        guider = ClassifierFreeGuidance(guidance_scale=guidance_scale)

        transformer = self.unwrap_model(self.model) if load_base_model and self.model is not None else None
        vae = self.unwrap_model(self.vae) if self.vae is not None else None

        # Respect memory-offload: text encoders may be offloaded to meta/None when not training.
        txt_encoder = None if _is_meta(self.text_encoder) else self.text_encoder
        byt5_encoder = None if _is_meta(self.text_encoder_2) else self.text_encoder_2
        tokenizer = None if txt_encoder is None else self.tokenizer
        tokenizer_2 = None if byt5_encoder is None else self.tokenizer_2

        pipeline_kwargs = {
            "text_encoder": txt_encoder,
            "tokenizer": tokenizer,
            "transformer": transformer,
            "vae": vae,
            "scheduler": scheduler,
            "text_encoder_2": byt5_encoder,
            "tokenizer_2": tokenizer_2,
            "guider": guider,
        }

        if pipeline_type == PipelineTypes.IMG2VIDEO:
            image_encoder, feature_extractor = self._load_image_encoder(move_to_device=True)
            pipeline = HunyuanVideo15ImageToVideoPipeline(
                image_encoder=image_encoder,
                feature_extractor=feature_extractor,
                **pipeline_kwargs,
            )
        else:
            pipeline = HunyuanVideo15Pipeline(**pipeline_kwargs)

        _prune_meta_components(pipeline)
        pipeline.to(device)

        if not hasattr(self, "pipelines"):
            self.pipelines = {}
        self.pipelines[pipeline_type] = pipeline
        return pipeline

    def _prepare_cond_latents(self, cond_latents: Optional[torch.Tensor], latents: torch.Tensor, task_type: str):
        batch, _, frames, height, width = latents.shape
        dtype = latents.dtype
        device = latents.device

        if cond_latents is None:
            cond = torch.zeros(batch, self.LATENT_CHANNEL_COUNT, frames, height, width, device=device, dtype=dtype)
        else:
            if cond_latents.dim() == 4:
                cond_latents = cond_latents.unsqueeze(2)
            cond = cond_latents.to(device=device, dtype=dtype).repeat(1, 1, frames, 1, 1)
            if task_type == "i2v":
                cond[:, :, 1:, :, :] = 0.0

        mask = torch.zeros(batch, 1, frames, height, width, device=device, dtype=dtype)
        if task_type == "i2v":
            mask[:, :, 0, :, :] = 1.0
        return cond, mask

    def _format_text_embedding(self, text_embedding: dict):
        prompt_embeds, prompt_attention_mask, prompt_embeds_2, prompt_attention_mask_2 = text_embedding
        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
            "prompt_embeds_2": prompt_embeds_2,
            "prompt_attention_mask_2": prompt_attention_mask_2,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        return {
            "prompt_embeds": text_embedding.get("prompt_embeds"),
            "prompt_embeds_mask": text_embedding.get("prompt_attention_mask"),
            "prompt_embeds_2": text_embedding.get("prompt_embeds_2"),
            "prompt_embeds_mask_2": text_embedding.get("prompt_attention_mask_2"),
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        return {
            "negative_prompt_embeds": text_embedding.get("prompt_embeds"),
            "negative_prompt_embeds_mask": text_embedding.get("prompt_attention_mask"),
            "negative_prompt_embeds_2": text_embedding.get("prompt_embeds_2"),
            "negative_prompt_embeds_mask_2": text_embedding.get("prompt_attention_mask_2"),
        }

    def collate_prompt_embeds(self, text_encoder_output: dict) -> dict:
        """
        Stack prompt embeddings and masks for the current batch.
        """

        def _collate(field):
            tensors = [entry[field] for entry in text_encoder_output if field in entry]
            if not tensors:
                return None
            first = tensors[0]
            if first.dim() == 3:
                return torch.cat(tensors, dim=0)
            return torch.stack(tensors, dim=0)

        collated = {}
        if text_encoder_output and isinstance(text_encoder_output, list):
            prompt_embeds = _collate("prompt_embeds")
            if prompt_embeds is not None:
                collated["prompt_embeds"] = prompt_embeds
            prompt_masks = _collate("prompt_attention_mask")
            if prompt_masks is not None:
                collated["attention_masks"] = prompt_masks
            prompt_embeds_2 = _collate("prompt_embeds_2")
            if prompt_embeds_2 is not None:
                collated["prompt_embeds_2"] = prompt_embeds_2
            prompt_masks_2 = _collate("prompt_attention_mask_2")
            if prompt_masks_2 is not None:
                collated["attention_masks_2"] = prompt_masks_2
        return collated

    def prepare_batch_conditions(self, batch: dict, state: dict) -> dict:
        batch = super().prepare_batch_conditions(batch=batch, state=state)
        text_output = batch.get("text_encoder_output") or {}

        prompt_embeds = batch.get("prompt_embeds")
        if prompt_embeds is None:
            prompt_embeds = text_output.get("prompt_embeds")
        if prompt_embeds is not None:
            batch["encoder_hidden_states"] = prompt_embeds

        attention_masks = batch.get("attention_masks")
        if attention_masks is None:
            attention_masks = text_output.get("attention_masks")
        if attention_masks is not None:
            batch["encoder_attention_mask"] = attention_masks

        prompt_embeds_2 = batch.get("prompt_embeds_2")
        if prompt_embeds_2 is None:
            prompt_embeds_2 = text_output.get("prompt_embeds_2")
        if prompt_embeds_2 is not None:
            batch["encoder_hidden_states_2"] = prompt_embeds_2

        attention_masks_2 = batch.get("attention_masks_2")
        if attention_masks_2 is None:
            attention_masks_2 = text_output.get("attention_masks_2")
        if attention_masks_2 is not None:
            batch["encoder_attention_mask_2"] = attention_masks_2
        return batch

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        prompt_list = prompts if isinstance(prompts, list) else [prompts]
        prompt_embeds, prompt_attention_mask, prompt_embeds_2, prompt_attention_mask_2 = pipeline.encode_prompt(
            prompt=prompt_list,
            device=self.accelerator.device,
            batch_size=len(prompt_list),
            num_videos_per_prompt=1,
        )
        return prompt_embeds, prompt_attention_mask, prompt_embeds_2, prompt_attention_mask_2

    def model_predict(self, prepared_batch):
        latents = prepared_batch["noisy_latents"].to(self.config.weight_dtype)
        if latents.dim() != 5:
            raise ValueError(f"Expected 5D video latents, got shape {latents.shape}")

        hidden_states_buffer = self._new_hidden_state_buffer()
        encoder_hidden_states = prepared_batch["encoder_hidden_states"].to(self.config.weight_dtype)
        encoder_attention_mask = prepared_batch.get("encoder_attention_mask")
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(device=latents.device, dtype=torch.bool)

        encoder_hidden_states_2 = prepared_batch.get("encoder_hidden_states_2")
        if encoder_hidden_states_2 is not None:
            encoder_hidden_states_2 = encoder_hidden_states_2.to(self.config.weight_dtype)

        encoder_attention_mask_2 = prepared_batch.get("encoder_attention_mask_2")
        if encoder_attention_mask_2 is not None:
            encoder_attention_mask_2 = encoder_attention_mask_2.to(device=latents.device, dtype=torch.bool)

        timesteps = prepared_batch["timesteps"]
        wants_i2v_batch = bool(prepared_batch.get("is_i2v_data", False))
        is_i2v_model = self._is_i2v_like_flavour()

        if wants_i2v_batch and not is_i2v_model and should_log() and not getattr(self, "_warned_spurious_i2v_batch", False):
            logger.warning(
                "Received an i2v-labelled batch for a t2v flavour; ignoring the flag and continuing with t2v training."
            )
            self._warned_spurious_i2v_batch = True

        if is_i2v_model and prepared_batch.get("conditioning_latents") is None:
            raise ValueError("HunyuanVideo i2v training requires conditioning_latents in the batch.")

        task_type = "i2v" if is_i2v_model else "t2v"
        cond_latents, cond_mask = self._prepare_cond_latents(prepared_batch.get("conditioning_latents"), latents, task_type)
        latent_model_input = torch.cat([latents, cond_latents, cond_mask], dim=1)

        batch_size = latents.shape[0]
        base_model = self.unwrap_model(self.model)
        model_config = getattr(base_model, "config", None)

        def _cfg(attr: str, default):
            if model_config is None:
                return default
            if hasattr(model_config, attr):
                try:
                    return getattr(model_config, attr)
                except Exception:
                    pass
            if isinstance(model_config, dict):
                return model_config.get(attr, default)
            try:
                return model_config[attr]
            except Exception:
                return default

        vision_tokens = _cfg("vision_num_semantic_tokens", getattr(self.config, "vision_num_semantic_tokens", 729))
        vision_dim = _cfg("image_embed_dim", getattr(self.config, "vision_states_dim", 1152))
        text_embed_2_dim = _cfg("text_embed_2_dim", getattr(self.config, "text_embed_2_dim", 1472))

        image_embeds = prepared_batch.get("vision_states")
        if image_embeds is None:
            image_embeds = torch.zeros(
                batch_size,
                vision_tokens,
                vision_dim,
                device=latents.device,
                dtype=latents.dtype,
            )

        if encoder_hidden_states_2 is None:
            encoder_hidden_states_2 = torch.zeros(
                encoder_hidden_states.shape[0],
                1,
                text_embed_2_dim,
                device=latents.device,
                dtype=self.config.weight_dtype,
            )
            encoder_attention_mask_2 = torch.zeros(
                encoder_hidden_states_2.shape[0],
                encoder_hidden_states_2.shape[1],
                device=latents.device,
                dtype=torch.bool,
            )
        else:
            if encoder_attention_mask_2 is None:
                encoder_attention_mask_2 = torch.zeros(
                    encoder_hidden_states_2.shape[0],
                    encoder_hidden_states_2.shape[1],
                    device=latents.device,
                    dtype=torch.bool,
                )

        model_pred = self.model(
            hidden_states=latent_model_input,
            timestep=timesteps,
            timestep_sign=(
                prepared_batch.get("twinflow_time_sign") if getattr(self.config, "twinflow_enabled", False) else None
            ),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            encoder_hidden_states_2=encoder_hidden_states_2,
            encoder_attention_mask_2=encoder_attention_mask_2,
            image_embeds=image_embeds,
            return_dict=False,
            hidden_states_buffer=hidden_states_buffer,
        )[0]
        return {
            "model_prediction": model_pred,
            "hidden_states_buffer": hidden_states_buffer,
        }

    def save_lora_weights(self, *args, **kwargs):
        """
        Map transformer LoRA weights to Diffusers' expected naming before delegating to the pipeline mixin.
        """
        if args:
            save_directory, *remaining = args
        else:
            save_directory = kwargs.pop("save_directory", None)
            remaining = []
        if save_directory is None:
            raise ValueError("save_directory is required to save LoRA weights.")

        transformer_key = kwargs.pop("transformer_lora_layers", None)
        model_key = kwargs.pop(f"{self.MODEL_SUBFOLDER}_lora_layers", None)
        transformer_layers = transformer_key if transformer_key is not None else model_key
        transformer_adapter_metadata = kwargs.pop("transformer_lora_adapter_metadata", None)
        # Drop extra text encoder adapter metadata and unsupported adapter names.
        kwargs.pop("text_encoder_lora_adapter_metadata", None)
        kwargs.pop("text_encoder_2_lora_adapter_metadata", None)
        kwargs.pop("adapter_name", None)
        # Drop any secondary text encoder LoRA payloads (the mixin only supports one).
        kwargs.pop("text_encoder_2_lora_layers", None)

        if transformer_layers is not None:
            kwargs["unet_lora_layers"] = transformer_layers
        if transformer_adapter_metadata is not None:
            kwargs["unet_lora_adapter_metadata"] = transformer_adapter_metadata

        return self.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG].save_lora_weights(
            save_directory=save_directory,
            *remaining,
            **kwargs,
        )

    def pretrained_load_args(self, pretrained_load_args: dict) -> dict:
        args = super().pretrained_load_args(pretrained_load_args)
        return apply_musubi_pretrained_defaults(self.config, args)

    def tread_init(self):
        raise NotImplementedError("TREAD routing is not supported for the diffusers HunyuanVideo 1.5 transformer.")

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        """
        Populate pipeline kwargs for validation runs.
        """
        default_resolution = getattr(self.config, "validation_resolution", None)
        aspect_ratio = pipeline_kwargs.get("aspect_ratio")
        if aspect_ratio is None:
            if isinstance(default_resolution, str) and "x" in default_resolution:
                width, height = default_resolution.lower().split("x")
                aspect_ratio = f"{width}:{height}"
            else:
                aspect_ratio = "1280:720"
        pipeline_kwargs["aspect_ratio"] = aspect_ratio
        pipeline_kwargs.setdefault("video_length", getattr(self.config, "validation_num_video_frames", 25) or 25)
        pipeline_kwargs.setdefault("num_inference_steps", getattr(self.config, "validation_num_inference_steps", 30) or 30)
        pipeline_kwargs.setdefault("guidance_scale", getattr(self.config, "validation_guidance", 6.0))
        pipeline_kwargs.setdefault("flow_shift", getattr(self.config, "flow_schedule_shift", 7.0))
        pipeline_kwargs.setdefault("enable_sr", True)
        return pipeline_kwargs

    def get_transforms(self, dataset_type: str = "image"):
        if dataset_type == DatasetType.VIDEO.value or dataset_type == "video":
            return super().get_transforms(dataset_type="video")
        return super().get_transforms(dataset_type=dataset_type)


ModelRegistry.register("hunyuanvideo", HunyuanVideo)
