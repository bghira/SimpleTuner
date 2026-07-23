import logging
from typing import Optional

import torch
from einops import rearrange
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

from simpletuner.helpers.acceleration import (
    AccelerationBackend,
    AccelerationPreset,
    get_bitsandbytes_presets,
    get_deepspeed_presets,
    get_quanto_presets,
    get_sdnq_presets,
    get_torchao_presets,
)
from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.mageflow.autoencoder import MageVAE
from simpletuner.helpers.models.mageflow.pipeline import MageFlowPipeline
from simpletuner.helpers.models.mageflow.pipeline_edit import MageFlowEditPipeline
from simpletuner.helpers.models.mageflow.transformer import MageFlowTransformer2DModel
from simpletuner.helpers.models.mageflow.vendor.pipeline import _lens_to_cu
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.musubi_block_swap import apply_musubi_pretrained_defaults

logger = logging.getLogger(__name__)


class MageFlow(ImageModelFoundation):
    SUPPORTS_MUON_CLIP = True
    NAME = "Mage-Flow"
    MODEL_DESCRIPTION = "Microsoft Mage-Flow 4B rectified-flow image generation and editing"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    USES_DYNAMIC_SHIFT = False
    AUTOENCODER_CLASS = MageVAE
    AUTOENCODER_SCALING_FACTOR = 1.0
    LATENT_CHANNEL_COUNT = 128
    VALIDATION_PREVIEW_SPEC = None
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]
    SLIDER_LORA_TARGET = DEFAULT_LORA_TARGET

    MODEL_CLASS = MageFlowTransformer2DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: MageFlowPipeline,
    }

    DEFAULT_MODEL_FLAVOUR = "base"
    HUGGINGFACE_PATHS = {
        "base": "microsoft/Mage-Flow-Base",
        "default": "microsoft/Mage-Flow",
        "turbo": "microsoft/Mage-Flow-Turbo",
        "edit-base": "microsoft/Mage-Flow-Edit-Base",
        "edit": "microsoft/Mage-Flow-Edit",
        "edit-turbo": "microsoft/Mage-Flow-Edit-Turbo",
    }
    MODEL_LICENSE = "mit"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Qwen3-VL",
            "tokenizer": AutoTokenizer,
            "tokenizer_subfolder": "text_encoder",
            "model": Qwen3VLForConditionalGeneration,
            "subfolder": "text_encoder",
        },
    }
    PROCESSOR_CLASS = AutoProcessor
    PROCESSOR_SUBFOLDER = "text_encoder"
    VALIDATION_USES_NEGATIVE_PROMPT = True

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        return 11

    @classmethod
    def get_acceleration_presets(cls) -> list[AccelerationPreset]:
        base_config = {
            "base_model_precision": "no_change",
            "gradient_checkpointing": True,
        }
        return [
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="light",
                name="RamTorch - Light",
                description="Streams the first 4 Mage-Flow transformer blocks from CPU RAM.",
                tab="basic",
                tradeoff_vram="Reduces transformer VRAM use.",
                tradeoff_speed="Increases training time from CPU-GPU transfers.",
                tradeoff_notes="Requires high system RAM. CUDA/ROCm only.",
                requires_cuda=True,
                requires_min_system_ram_gb=64,
                config={
                    **base_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "transformer_blocks.0.*,transformer_blocks.1.*,transformer_blocks.2.*,transformer_blocks.3.*",
                },
            ),
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="balanced",
                name="RamTorch - Balanced",
                description="Streams the first 8 Mage-Flow transformer blocks from CPU RAM.",
                tab="basic",
                tradeoff_vram="Reduces transformer VRAM use more aggressively.",
                tradeoff_speed="Increases training time from CPU-GPU transfers.",
                tradeoff_notes="Requires high system RAM. CUDA/ROCm only.",
                requires_cuda=True,
                requires_min_system_ram_gb=96,
                config={
                    **base_config,
                    "ramtorch": True,
                    "ramtorch_target_modules": "transformer_blocks.0.*,transformer_blocks.1.*,transformer_blocks.2.*,transformer_blocks.3.*,transformer_blocks.4.*,transformer_blocks.5.*,transformer_blocks.6.*,transformer_blocks.7.*",
                },
            ),
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="aggressive",
                name="RamTorch - Aggressive",
                description="Streams all Mage-Flow transformer blocks from CPU RAM.",
                tab="basic",
                tradeoff_vram="Minimizes resident transformer block weights.",
                tradeoff_speed="Largest CPU-GPU transfer overhead.",
                tradeoff_notes="Requires high system RAM. CUDA/ROCm only.",
                requires_cuda=True,
                requires_min_system_ram_gb=128,
                config={**base_config, "ramtorch": True, "ramtorch_target_modules": "transformer_blocks.*"},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="light",
                name="Block Swap - Light",
                description="Swaps 3 of 12 Mage-Flow transformer blocks.",
                tab="basic",
                tradeoff_vram="Reduces transformer residency in VRAM.",
                tradeoff_speed="Adds CPU/GPU transfer overhead.",
                tradeoff_notes="Mutually exclusive with RamTorch and group offload.",
                requires_min_system_ram_gb=64,
                config={**base_config, "musubi_blocks_to_swap": 3},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="balanced",
                name="Block Swap - Balanced",
                description="Swaps 6 of 12 Mage-Flow transformer blocks.",
                tab="basic",
                tradeoff_vram="Reduces transformer residency in VRAM more aggressively.",
                tradeoff_speed="Adds more CPU/GPU transfer overhead.",
                tradeoff_notes="Mutually exclusive with RamTorch and group offload.",
                requires_min_system_ram_gb=96,
                config={**base_config, "musubi_blocks_to_swap": 6},
            ),
            AccelerationPreset(
                backend=AccelerationBackend.MUSUBI_BLOCK_SWAP,
                level="aggressive",
                name="Block Swap - Aggressive",
                description="Swaps 11 of 12 Mage-Flow transformer blocks.",
                tab="advanced",
                tradeoff_vram="Leaves only one transformer block resident.",
                tradeoff_speed="Largest CPU/GPU transfer overhead.",
                tradeoff_notes="Use only when VRAM is the limiting factor.",
                requires_min_system_ram_gb=128,
                config={**base_config, "musubi_blocks_to_swap": 11},
            ),
            *get_deepspeed_presets(base_config),
            *get_sdnq_presets(base_config),
            *get_torchao_presets(base_config),
            *get_quanto_presets(base_config),
            *get_bitsandbytes_presets(base_config),
        ]

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        self.vae_scale_factor = 16
        self.processor = None
        if self._is_edit_flavour():
            self.PIPELINE_CLASSES = {
                PipelineTypes.TEXT2IMG: MageFlowEditPipeline,
                PipelineTypes.IMG2IMG: MageFlowEditPipeline,
            }

    def _model_flavour(self) -> str:
        return str(getattr(self.config, "model_flavour", None) or self.DEFAULT_MODEL_FLAVOUR)

    def _is_edit_flavour(self) -> bool:
        return self._model_flavour() in {"edit-base", "edit", "edit-turbo"}

    @staticmethod
    def _normalise_attention_mechanism(attention_mechanism: str | None) -> str:
        return str(attention_mechanism or "").strip().lower().replace("_", "-")

    @staticmethod
    def _recommended_attention_mechanism() -> str:
        if not torch.cuda.is_available():
            return "flash-attn-varlen-hub"

        major, _ = torch.cuda.get_device_capability()
        if major >= 10:
            logger.warning(
                "Blackwell CUDA capability detected for Mage-Flow; selecting flash-attn-4-hub for packed varlen "
                "attention. This backend has not been validated by the SimpleTuner Mage-Flow presets yet."
            )
            return "flash-attn-4-hub"
        if major >= 9:
            return "flash-attn-3-varlen-hub"
        return "flash-attn-varlen-hub"

    def _normalise_mageflow_attention_config(self) -> None:
        requested = getattr(self.config, "attention_mechanism", None)
        requested_normalised = self._normalise_attention_mechanism(requested)
        recommended = self._recommended_attention_mechanism()
        supported = {
            "fa2",
            "flash-attn-2",
            "flash-attn-2-hub",
            "flash-attention-2",
            "flash-attention-2-hub",
            "flash-attn-varlen",
            "flash-attn-varlen-hub",
            "flash-attention-varlen",
            "flash-attention-varlen-hub",
            "flash-varlen-hub",
            "flash2",
            "flash2-hub",
            "fa3",
            "flash-attn-3",
            "flash-attn-3-hub",
            "flash-attn-3-varlen",
            "flash-attn-3-varlen-hub",
            "flash-attention-3",
            "flash-attention-3-hub",
            "flash-attention-3-varlen",
            "flash-attention-3-varlen-hub",
            "flash3",
            "flash3-hub",
            "flash3-varlen",
            "flash3-varlen-hub",
            "fa4",
            "flash-attn-4",
            "flash-attn-4-hub",
            "flash-attention-4",
            "flash-attention-4-hub",
            "flash4",
            "flash4-hub",
            "auto",
            "automatic",
        }
        supported_aliases = {
            "fa2": "flash-attn-varlen-hub",
            "flash-attn-2": "flash-attn-varlen-hub",
            "flash-attn-2-hub": "flash-attn-varlen-hub",
            "flash-attention-2": "flash-attn-varlen-hub",
            "flash-attention-2-hub": "flash-attn-varlen-hub",
            "flash-attn-varlen": "flash-attn-varlen-hub",
            "flash-attention-varlen": "flash-attn-varlen-hub",
            "flash-attention-varlen-hub": "flash-attn-varlen-hub",
            "flash-varlen-hub": "flash-attn-varlen-hub",
            "flash2": "flash-attn-varlen-hub",
            "flash2-hub": "flash-attn-varlen-hub",
            "fa3": "flash-attn-3-varlen-hub",
            "flash-attn-3": "flash-attn-3-varlen-hub",
            "flash-attn-3-hub": "flash-attn-3-varlen-hub",
            "flash-attention-3": "flash-attn-3-varlen-hub",
            "flash-attention-3-hub": "flash-attn-3-varlen-hub",
            "flash-attention-3-varlen": "flash-attn-3-varlen-hub",
            "flash-attention-3-varlen-hub": "flash-attn-3-varlen-hub",
            "flash3": "flash-attn-3-varlen-hub",
            "flash3-hub": "flash-attn-3-varlen-hub",
            "flash3-varlen": "flash-attn-3-varlen-hub",
            "flash3-varlen-hub": "flash-attn-3-varlen-hub",
            "fa4": "flash-attn-4-hub",
            "flash-attn-4": "flash-attn-4-hub",
            "flash-attention-4": "flash-attn-4-hub",
            "flash-attention-4-hub": "flash-attn-4-hub",
            "flash4": "flash-attn-4-hub",
            "flash4-hub": "flash-attn-4-hub",
            "auto": recommended,
            "automatic": recommended,
        }

        if requested_normalised in supported:
            resolved = supported_aliases.get(requested_normalised, requested_normalised or recommended)
            if resolved != requested:
                logger.info(
                    "Normalising Mage-Flow --attention_mechanism=%s to %s for packed varlen attention.",
                    requested,
                    resolved,
                )
                self.config.attention_mechanism = resolved
            return

        if requested_normalised:
            logger.warning(
                "Mage-Flow uses packed varlen attention and does not support --attention_mechanism=%s. "
                "Overriding with %s for this CUDA target.",
                requested,
                recommended,
            )
        else:
            logger.info("Mage-Flow attention mechanism unset; using %s for packed varlen attention.", recommended)
        self.config.attention_mechanism = recommended

    def check_user_config(self):
        if self.config.aspect_bucket_alignment != 16:
            logger.warning("Mage-Flow uses a 16px latent alignment. Overriding --aspect_bucket_alignment.")
            self.config.aspect_bucket_alignment = 16
        self._normalise_mageflow_attention_config()
        if int(getattr(self.config, "context_parallel_size", 1) or 1) > 1:
            raise ValueError("Mage-Flow does not support --context_parallel_size because it uses packed varlen attention.")
        self.config.tokenizer_max_length = int(getattr(self.config, "tokenizer_max_length", None) or 2048)
        if self._is_edit_flavour():
            self.PIPELINE_CLASSES[PipelineTypes.TEXT2IMG] = MageFlowEditPipeline
            self.PIPELINE_CLASSES[PipelineTypes.IMG2IMG] = MageFlowEditPipeline
        validation_steps = getattr(self.config, "validation_num_inference_steps", None)
        if self._model_flavour().endswith("turbo") and validation_steps is not None and validation_steps > 8:
            logger.warning("Mage-Flow Turbo checkpoints are intended for few-step validation; consider 4 steps.")

    def pretrained_load_args(self, pretrained_load_args: dict) -> dict:
        args = super().pretrained_load_args(pretrained_load_args)
        args["attn_type"] = getattr(self.config, "attention_mechanism", None) or "sdpa"
        return apply_musubi_pretrained_defaults(self.config, args)

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

    def _load_processor_for_pipeline(self):
        if self.processor is not None:
            return self.processor
        processor_path = getattr(self.config, "processor_pretrained_model_name_or_path", None) or self._model_config_path()
        processor_subfolder = getattr(self.config, "processor_subfolder", self.PROCESSOR_SUBFOLDER)
        self.processor = self.PROCESSOR_CLASS.from_pretrained(
            processor_path,
            subfolder=processor_subfolder,
            revision=getattr(self.config, "revision", None),
            local_files_only=getattr(self.config, "local_files_only", False),
        )
        return self.processor

    def requires_conditioning_dataset(self) -> bool:
        return False

    def supports_conditioning_dataset(self) -> bool:
        return self._is_edit_flavour()

    def requires_conditioning_latents(self) -> bool:
        return self._is_edit_flavour()

    def requires_conditioning_validation_inputs(self) -> bool:
        return False

    def requires_validation_edit_captions(self) -> bool:
        return False

    def requires_text_embed_image_context(self) -> bool:
        return False

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        if self._is_edit_flavour() and "conditioning_image" in pipeline_kwargs and "image" not in pipeline_kwargs:
            pipeline_kwargs["image"] = pipeline_kwargs.pop("conditioning_image")
        return pipeline_kwargs

    def _extract_prompt_image_from_context(self, context: dict):
        if not isinstance(context, dict):
            return None
        tensor = context.get("conditioning_pixel_values")
        if tensor is None:
            return None
        if torch.is_tensor(tensor):
            if tensor.dim() == 4 and tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
            return tensor
        return tensor

    def _prepare_prompt_images(self, prompt_contexts, batch_size: int):
        if not prompt_contexts or len(prompt_contexts) != batch_size:
            return None
        from PIL import Image
        from torchvision.transforms import functional as TF

        images = []
        for context in prompt_contexts:
            extracted = self._extract_prompt_image_from_context(context)
            if extracted is None:
                return None
            if isinstance(extracted, Image.Image):
                images.append(extracted.convert("RGB"))
            elif torch.is_tensor(extracted):
                tensor = extracted.detach().cpu()
                if tensor.min() < 0:
                    tensor = (tensor + 1.0) / 2.0
                images.append(TF.to_pil_image(tensor.clamp(0, 1)).convert("RGB"))
            else:
                return None
        return images

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        del is_negative_prompt
        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()
        self.load_text_tokenizer()
        text_encoder = self.text_encoders[0]
        if text_encoder.device != self.accelerator.device:
            text_encoder.to(self.accelerator.device, dtype=self.config.weight_dtype)

        if self._is_edit_flavour():
            pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
            prompt_contexts = getattr(self, "_current_prompt_contexts", None)
            images = self._prepare_prompt_images(prompt_contexts, len(prompts))
            return pipeline.encode_prompt(
                prompts,
                images=images,
                device=self.accelerator.device,
                max_sequence_length=int(self.config.tokenizer_max_length),
            )

        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        return pipeline.encode_prompt(
            prompts,
            device=self.accelerator.device,
            max_sequence_length=int(self.config.tokenizer_max_length),
        )

    def _format_text_embedding(self, text_embedding):
        prompt_embeds, attention_mask = text_embedding
        return {
            "prompt_embeds": prompt_embeds,
            "attention_masks": attention_mask,
        }

    def collate_prompt_embeds(self, text_encoder_output: list) -> dict:
        if not text_encoder_output:
            return {}
        embeds = [item["prompt_embeds"] for item in text_encoder_output]
        masks = [item["attention_masks"] for item in text_encoder_output]
        max_seq_len = max(embed.shape[-2] for embed in embeds)
        padded_embeds = []
        padded_masks = []
        for embed, mask in zip(embeds, masks, strict=True):
            if embed.dim() == 2:
                embed = embed.unsqueeze(0)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if embed.shape[1] < max_seq_len:
                pad_len = max_seq_len - embed.shape[1]
                embed = torch.cat([embed, embed.new_zeros(embed.shape[0], pad_len, embed.shape[2])], dim=1)
                mask = torch.cat([mask, mask.new_zeros(mask.shape[0], pad_len)], dim=1)
            padded_embeds.append(embed)
            padded_masks.append(mask)
        return {
            "prompt_embeds": torch.cat(padded_embeds, dim=0),
            "attention_masks": torch.cat(padded_masks, dim=0),
        }

    def convert_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        return {
            "prompt_embeds": text_embedding["prompt_embeds"],
            "prompt_embeds_mask": text_embedding["attention_masks"],
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        return {
            "negative_prompt_embeds": text_embedding["prompt_embeds"],
            "negative_prompt_embeds_mask": text_embedding["attention_masks"],
        }

    def _pack_text(self, prepared_batch: dict):
        prompt_embeds = prepared_batch["prompt_embeds"].to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        attention_mask = prepared_batch.get("encoder_attention_mask")
        if attention_mask is None:
            attention_mask = prepared_batch.get("attention_masks")
        if attention_mask is None:
            attention_mask = torch.ones(prompt_embeds.shape[:2], device=prompt_embeds.device, dtype=torch.bool)
        attention_mask = attention_mask.to(device=self.accelerator.device, dtype=torch.bool)
        chunks = []
        lengths = []
        for embeds, mask in zip(prompt_embeds, attention_mask, strict=True):
            selected = embeds[mask]
            chunks.append(selected)
            lengths.append(int(selected.shape[0]))
        return (
            torch.cat(chunks, dim=0).unsqueeze(0),
            _lens_to_cu(lengths, self.accelerator.device),
        )

    def _pack_latents_for_model(self, latents: torch.Tensor, conditioning_latents=None):
        batch_size, channels, height, width = latents.shape
        target_tokens = rearrange(latents, "b c h w -> b (h w) c")
        sample_lens = []
        target_lens = []
        shape_seq = []
        target_indices = []
        parts = []
        offset = 0
        for idx in range(batch_size):
            target = target_tokens[idx : idx + 1]
            parts.append(target)
            total_len = target.shape[1]
            target_lens.append(total_len)
            shape_seq.append((1, height, width))
            if conditioning_latents is not None:
                refs = conditioning_latents
                if torch.is_tensor(refs):
                    refs = [refs]
                for ref_group in refs:
                    ref = ref_group[idx : idx + 1] if ref_group.dim() == 4 else ref_group.unsqueeze(0)
                    ref_tokens = rearrange(ref.to(latents.device, latents.dtype), "b c h w -> b (h w) c")
                    parts.append(ref_tokens)
                    total_len += ref_tokens.shape[1]
                    shape_seq.append((1, ref.shape[-2], ref.shape[-1]))
            sample_lens.append(total_len)
            target_indices.append(torch.arange(offset, offset + target.shape[1], device=latents.device))
            offset += total_len
        return (
            torch.cat(parts, dim=1),
            _lens_to_cu(sample_lens, latents.device),
            [shape_seq],
            sample_lens,
            torch.cat(target_indices),
            target_lens,
            height,
            width,
        )

    def model_predict(self, prepared_batch, custom_timesteps: list = None):
        del custom_timesteps
        hidden_states_buffer = self._new_hidden_state_buffer()
        latents = prepared_batch["noisy_latents"].to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        conditioning_latents = prepared_batch.get("conditioning_latents") if self._is_edit_flavour() else None
        (
            img,
            img_cu,
            img_shapes,
            img_lens,
            target_indices,
            target_lens,
            latent_h,
            latent_w,
        ) = self._pack_latents_for_model(latents, conditioning_latents=conditioning_latents)
        txt, txt_cu = self._pack_text(prepared_batch)
        timestep = prepared_batch["timesteps"].to(device=self.accelerator.device, dtype=self.config.weight_dtype) / 1000.0
        if timestep.ndim == 0:
            timestep = timestep.expand(len(img_lens))

        model_pred = self.get_trained_component(base_model=True, unwrap_model=False)(
            img=img,
            txt=txt,
            timesteps=timestep,
            img_shapes=img_shapes,
            img_cu_seqlens=img_cu,
            txt_cu_seqlens=txt_cu,
            timestep_sign=(
                prepared_batch.get("twinflow_time_sign") if getattr(self.config, "twinflow_enabled", False) else None
            ),
            hidden_states_buffer=hidden_states_buffer,
        )
        model_pred = model_pred[:, target_indices]
        cursor = 0
        samples = []
        for target_len in target_lens:
            sample = model_pred[:, cursor : cursor + target_len]
            samples.append(rearrange(sample, "1 (h w) c -> c h w", h=latent_h, w=latent_w))
            cursor += target_len
        return {
            "model_prediction": torch.stack(samples, dim=0),
            "crepa_hidden_states": self._select_crepa_hidden_states(prepared_batch, hidden_states_buffer),
            "hidden_states_buffer": hidden_states_buffer,
        }


ModelRegistry.register("mageflow", MageFlow)
