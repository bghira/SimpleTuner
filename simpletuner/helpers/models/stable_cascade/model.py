import logging
import os
from typing import Dict, Optional

import torch
from transformers import CLIPTextModelWithProjection, PreTrainedTokenizerFast

from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.training.multi_process import should_log

from .autoencoder import TORCHVISION_IMPORT_ERROR, StableCascadeStageCAutoencoder
from .pipeline_combined import StableCascadeCombinedPipeline
from .pipeline_prior import StableCascadePriorPipeline
from .scheduler_ddpm_wuerstchen import ensure_wuerstchen_scheduler
from .unet import StableCascadeUNet

logger = logging.getLogger(__name__)
logger.setLevel(logging._nameToLevel.get(str(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO")).upper(), logging.INFO))


STAGE_C_FLAVOURS = frozenset({"stage-c", "stage-c-lite"})


class _LatentSampleWrapper:
    """
    Lightweight wrapper to mimic diffusers VAE outputs for caching.
    """

    def __init__(self, latents: torch.Tensor) -> None:
        self._latents = latents

    def sample(self) -> torch.Tensor:
        return self._latents


class StableCascadeStageC(ImageModelFoundation):
    NAME = "Stable Cascade (Stage C)"
    MODEL_DESCRIPTION = "Text-conditioned Stage C prior from Stable Cascade."
    ENABLED_IN_WIZARD = True

    PREDICTION_TYPE = PredictionTypes.EPSILON
    MODEL_TYPE = ModelTypes.UNET
    AUTOENCODER_CLASS = StableCascadeStageCAutoencoder
    AUTOENCODER_SCALING_FACTOR = 1.0
    LATENT_CHANNEL_COUNT = 16

    MODEL_CLASS = StableCascadeUNet
    MODEL_SUBFOLDER = "prior"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: StableCascadePriorPipeline,
    }

    DEFAULT_MODEL_FLAVOUR = "stage-c"
    HUGGINGFACE_PATHS: Dict[str, str] = {
        "stage-c": "stabilityai/stable-cascade-prior",
        "stage-c-lite": "stabilityai/stable-cascade-prior",
    }
    MODEL_LICENSE = "other"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "CLIP-ViT-bigG-14",
            "tokenizer": PreTrainedTokenizerFast,
            "tokenizer_subfolder": "tokenizer",
            "model": CLIPTextModelWithProjection,
            "subfolder": "text_encoder",
        },
    }

    STAGE_C_SUBFOLDER_OVERRIDES = {
        "stage-c-lite": "prior_lite",
    }

    DEFAULT_DECODER_REPO = "stabilityai/stable-cascade"
    DEFAULT_VALIDATION_PRIOR_GUIDANCE = 4.0
    DEFAULT_VALIDATION_PRIOR_STEPS = 20

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        self._combined_validation_pipeline = None

    def setup_model_flavour(self):
        super().setup_model_flavour()
        flavour = getattr(self.config, "model_flavour", self.DEFAULT_MODEL_FLAVOUR)
        if flavour not in STAGE_C_FLAVOURS:
            raise ValueError(f"Unsupported Stable Cascade flavour '{flavour}'.")
        subfolder = self.STAGE_C_SUBFOLDER_OVERRIDES.get(flavour, self.MODEL_SUBFOLDER)
        self.config.pretrained_unet_subfolder = subfolder
        if getattr(self.config, "pretrained_unet_model_name_or_path", None) is None:
            self.config.pretrained_unet_model_name_or_path = self.config.pretrained_model_name_or_path

    def _use_combined_pipeline(self) -> bool:
        return bool(getattr(self.config, "stable_cascade_use_decoder_for_validation", True))

    @staticmethod
    def _coerce_dtype(candidate, fallback=None):
        def _convert(value):
            if isinstance(value, torch.dtype):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                mapping = {
                    "fp32": torch.float32,
                    "float32": torch.float32,
                    "f32": torch.float32,
                    "fp16": torch.float16,
                    "float16": torch.float16,
                    "half": torch.float16,
                    "f16": torch.float16,
                    "bf16": torch.bfloat16,
                    "bfloat16": torch.bfloat16,
                }
                return mapping.get(normalized)
            return None

        resolved = _convert(candidate)
        if resolved is not None:
            return resolved
        resolved = _convert(fallback)
        if resolved is not None:
            return resolved
        return torch.float16

    def _decoder_repo_id(self) -> str:
        repo = getattr(self.config, "stable_cascade_decoder_model_name_or_path", None)
        if isinstance(repo, str) and repo:
            return repo
        return self.DEFAULT_DECODER_REPO

    def _decoder_variant(self) -> Optional[str]:
        variant = getattr(self.config, "stable_cascade_decoder_variant", None)
        if isinstance(variant, str) and variant.strip():
            return variant.strip()
        return None

    def _decoder_subfolder(self) -> Optional[str]:
        subfolder = getattr(self.config, "stable_cascade_decoder_subfolder", None)
        if isinstance(subfolder, str) and subfolder.strip():
            return subfolder.strip()
        return None

    def _decoder_pipeline_dtype(self) -> torch.dtype:
        return self._coerce_dtype(
            getattr(self.config, "stable_cascade_decoder_dtype", None),
            getattr(self.config, "weight_dtype", None),
        )

    def _apply_prior_overrides(self, pipeline: StableCascadeCombinedPipeline) -> None:
        prior_model = self.unwrap_model(model=self.model)
        if prior_model is None:
            raise RuntimeError("Stable Cascade prior weights must be loaded before building the combined pipeline.")
        pipeline.prior_prior = prior_model
        if hasattr(pipeline, "prior_pipe"):
            pipeline.prior_pipe.prior = prior_model

        if self.text_encoders:
            prior_text_encoder = self.text_encoders[0]
            pipeline.prior_text_encoder = prior_text_encoder
            if hasattr(pipeline, "prior_pipe"):
                pipeline.prior_pipe.text_encoder = prior_text_encoder

        if self.tokenizers:
            prior_tokenizer = self.tokenizers[0]
            pipeline.prior_tokenizer = prior_tokenizer
            if hasattr(pipeline, "prior_pipe"):
                pipeline.prior_pipe.tokenizer = prior_tokenizer

        pipeline.prior_scheduler = ensure_wuerstchen_scheduler(getattr(pipeline, "prior_scheduler", None))
        if hasattr(pipeline, "prior_pipe"):
            pipeline.prior_pipe.scheduler = pipeline.prior_scheduler

    def _maybe_override_decoder_components(self, pipeline: StableCascadeCombinedPipeline, dtype: torch.dtype) -> None:
        subfolder = self._decoder_subfolder()
        if not subfolder:
            return
        repo_id = self._decoder_repo_id()
        decoder = StableCascadeUNet.from_pretrained(
            repo_id,
            subfolder=subfolder,
            torch_dtype=dtype,
            use_safetensors=True,
        )
        pipeline.decoder = decoder
        if hasattr(pipeline, "decoder_pipe"):
            pipeline.decoder_pipe.decoder = decoder

    def _build_combined_validation_pipeline(self) -> StableCascadeCombinedPipeline:
        repo_id = self._decoder_repo_id()
        dtype = self._decoder_pipeline_dtype()
        load_kwargs = {"torch_dtype": dtype, "use_safetensors": True}
        variant = self._decoder_variant()
        if variant:
            load_kwargs["variant"] = variant
        try:
            pipeline = StableCascadeCombinedPipeline.from_pretrained(repo_id, **load_kwargs)
        except Exception as exc:  # pragma: no cover - initialization failure is surfaced to the user
            raise RuntimeError(f"Failed to load Stable Cascade decoder pipeline from '{repo_id}': {exc}") from exc

        self._apply_prior_overrides(pipeline)
        self._maybe_override_decoder_components(pipeline, dtype)
        return pipeline

    def load_vae(self, move_to_device: bool = True):
        vae_path = getattr(self.config, "vae_path", None)
        self.vae = self.AUTOENCODER_CLASS.from_pretrained(
            pretrained_model_name_or_path=vae_path,
            torch_dtype=torch.float32,
        )
        if move_to_device:
            self.vae.to(self.accelerator.device, dtype=torch.float32)
        self.vae.eval()
        self.AUTOENCODER_SCALING_FACTOR = getattr(self.vae.config, "scaling_factor", 1.0)

    def get_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True):
        if pipeline_type is PipelineTypes.TEXT2IMG and not load_base_model and self._use_combined_pipeline():
            if self._combined_validation_pipeline is None:
                self._combined_validation_pipeline = self._build_combined_validation_pipeline()
            return self._combined_validation_pipeline
        return super().get_pipeline(pipeline_type=pipeline_type, load_base_model=load_base_model)

    def unload_model(self):
        self._combined_validation_pipeline = None
        parent = super()
        if hasattr(parent, "unload_model"):
            parent.unload_model()

    def encode_with_vae(self, vae, samples):
        latents = vae(samples)
        return _LatentSampleWrapper(latents)

    def post_vae_encode_transform_sample(self, sample):
        if isinstance(sample, torch.Tensor):
            return sample
        if hasattr(sample, "sample"):
            return sample.sample()
        return sample

    def get_transforms(self, dataset_type: str = "image"):
        if dataset_type != "image":
            raise ValueError(f"{self.NAME} only supports image datasets.")
        if TORCHVISION_IMPORT_ERROR is not None:
            raise RuntimeError(
                "torchvision is required to compute Stable Cascade preprocessing transforms."
            ) from TORCHVISION_IMPORT_ERROR
        from torchvision import transforms as vision_transforms

        return vision_transforms.Compose([vision_transforms.ToTensor()])

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        tokenizer = self.tokenizers[0]
        device = self.accelerator.device

        text_inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)

        encoder: CLIPTextModelWithProjection = self.text_encoders[0]
        text_encoder_output = encoder(
            text_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        prompt_embeds = text_encoder_output.hidden_states[-1]
        pooled_embeds = text_encoder_output.text_embeds.unsqueeze(1)

        return {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_embeds,
            "attention_masks": attention_mask,
        }

    def _format_text_embedding(self, text_embedding):
        return text_embedding

    def convert_text_embed_for_pipeline(self, text_embedding: Dict[str, torch.Tensor], prompt: Optional[str] = None) -> dict:
        # Only unsqueeze if it's missing the batch dimension
        prompt_embeds = text_embedding["prompt_embeds"]
        pooled_prompt_embeds = text_embedding["pooled_prompt_embeds"]

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:  # Shape: [seq, dim]
            prompt_embeds = prompt_embeds.unsqueeze(0)  # Shape: [1, seq, dim]
        if pooled_prompt_embeds.dim() == 2:  # Shape: [batch, dim]
            # Already has batch dimension - keep as is
            pass
        elif pooled_prompt_embeds.dim() == 1:  # Shape: [dim]
            pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0)  # Shape: [1, dim]

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_embeds_pooled": pooled_prompt_embeds,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: Dict[str, torch.Tensor]) -> dict:
        # Only unsqueeze if it's missing the batch dimension
        prompt_embeds = text_embedding["prompt_embeds"]
        pooled_prompt_embeds = text_embedding["pooled_prompt_embeds"]

        # Add batch dimension if missing
        if prompt_embeds.dim() == 2:  # Shape: [seq, dim]
            prompt_embeds = prompt_embeds.unsqueeze(0)  # Shape: [1, seq, dim]
        if pooled_prompt_embeds.dim() == 2:  # Shape: [batch, dim]
            # Already has batch dimension - keep as is
            pass
        elif pooled_prompt_embeds.dim() == 1:  # Shape: [dim]
            pooled_prompt_embeds = pooled_prompt_embeds.unsqueeze(0)  # Shape: [1, dim]

        return {
            "negative_prompt_embeds": prompt_embeds,
            "negative_prompt_embeds_pooled": pooled_prompt_embeds,
        }

    def prepare_batch_conditions(self, batch: dict, state: dict) -> dict:
        batch = super().prepare_batch_conditions(batch=batch, state=state)
        batch_size = batch["latents"].shape[0]
        model_config = getattr(getattr(self, "model", None), "config", None)
        clip_channels = getattr(model_config, "clip_image_in_channels", 768)
        clip_img = batch.get("cascade_clip_image_embeds")
        device = self.accelerator.device
        dtype = self.config.weight_dtype

        if clip_img is None:
            batch["cascade_clip_image_embeds"] = torch.zeros(
                batch_size,
                1,
                clip_channels,
                device=device,
                dtype=dtype,
            )
        else:
            batch["cascade_clip_image_embeds"] = clip_img.to(device=device, dtype=dtype)

        return batch

    def _compute_timestep_ratio(self, timesteps: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, device=self.accelerator.device)
        timesteps = timesteps.to(self.accelerator.device).float()
        denom = max(self.noise_schedule.config.num_train_timesteps - 1, 1)
        return timesteps / denom

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        pipeline_kwargs = super().update_pipeline_call_kwargs(pipeline_kwargs)
        if not self._use_combined_pipeline():
            return pipeline_kwargs

        prior_steps = getattr(
            self.config,
            "stable_cascade_validation_prior_num_inference_steps",
            self.DEFAULT_VALIDATION_PRIOR_STEPS,
        )
        try:
            pipeline_kwargs.setdefault("prior_num_inference_steps", int(prior_steps))
        except (TypeError, ValueError) as exc:
            raise ValueError("`stable_cascade_validation_prior_num_inference_steps` must be an integer.") from exc

        prior_guidance = getattr(
            self.config,
            "stable_cascade_validation_prior_guidance_scale",
            getattr(self.config, "validation_guidance", self.DEFAULT_VALIDATION_PRIOR_GUIDANCE),
        )
        try:
            pipeline_kwargs.setdefault("prior_guidance_scale", float(prior_guidance))
        except (TypeError, ValueError) as exc:
            raise ValueError("`stable_cascade_validation_prior_guidance_scale` must be numeric.") from exc

        decoder_guidance = getattr(
            self.config,
            "stable_cascade_validation_decoder_guidance_scale",
            getattr(self.config, "validation_guidance", None),
        )
        if decoder_guidance is not None:
            try:
                pipeline_kwargs["decoder_guidance_scale"] = float(decoder_guidance)
            except (TypeError, ValueError) as exc:
                raise ValueError("`stable_cascade_validation_decoder_guidance_scale` must be numeric.") from exc

        return pipeline_kwargs

    def model_predict(self, prepared_batch):
        device = self.accelerator.device
        weight_dtype = self.config.base_weight_dtype

        noisy_latents = prepared_batch["noisy_latents"].to(device=device, dtype=weight_dtype)
        clip_text = prepared_batch["encoder_hidden_states"].to(device=device, dtype=weight_dtype)

        pooled = prepared_batch.get("added_cond_kwargs", {}).get("text_embeds")
        if pooled is None:
            raise ValueError("Stable Cascade Stage C requires pooled text embeddings.")
        pooled = pooled.to(device=device, dtype=weight_dtype)
        if pooled.ndim == 2:
            pooled = pooled.unsqueeze(1)

        clip_img = prepared_batch.get("cascade_clip_image_embeds")
        clip_img = clip_img.to(device=device, dtype=weight_dtype)

        timestep_ratio = self._compute_timestep_ratio(prepared_batch["timesteps"]).to(device=device, dtype=weight_dtype)

        model_output = self.model(
            sample=noisy_latents,
            timestep_ratio=timestep_ratio,
            clip_text=clip_text,
            clip_text_pooled=pooled,
            clip_img=clip_img,
            return_dict=False,
        )[0]

        return {"model_prediction": model_output}

    def check_user_config(self):
        mixed_precision = getattr(self.config, "mixed_precision", None)
        if mixed_precision not in (None, "no", "fp32"):
            if not getattr(self.config, "i_know_what_i_am_doing", False):
                raise ValueError("Stable Cascade Stage C requires --mixed_precision=no to run in full precision.")
            logger.warning(
                "Stable Cascade Stage C is running with mixed_precision=%s due to i_know_what_i_am_doing.",
                mixed_precision,
            )

        if getattr(self.config, "base_model_precision", "no_change") != "no_change":
            logger.warning("Stable Cascade Stage C ignores base_model_precision. Quantising the UNet is not recommended.")

        max_tokens = 77
        if getattr(self.config, "tokenizer_max_length", None) not in (None, max_tokens):
            if getattr(self.config, "i_know_what_i_am_doing", False):
                logger.warning(
                    "Tokenizer max length differs from the CLIP-G default of %s tokens; proceeding due to override.",
                    max_tokens,
                )
            else:
                logger.warning(
                    "Stable Cascade Stage C uses a fixed CLIP-G sequence length. Overriding tokenizer_max_length to %s.",
                    max_tokens,
                )
                self.config.tokenizer_max_length = max_tokens


ModelRegistry.register("stable_cascade", StableCascadeStageC)
