import logging
import os
from pathlib import Path
from typing import Optional

import torch
from diffusers import AutoencoderKLQwenImage
from huggingface_hub import hf_hub_download
from transformers import Qwen3Model

from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.registry import ModelRegistry

from .loading import (
    _QWEN_TOKENIZER_SOURCE,
    _T5_TOKENIZER_SOURCE,
    coerce_anima_scheduler,
    load_prompt_tokenizer,
    load_text_encoder_single_file,
    load_vae_single_file,
    resolve_text_encoder_dtype,
)
from .options import AnimaLoaderOptions
from .pipeline import AnimaPipeline
from .scheduler import AnimaFlowMatchEulerDiscreteScheduler
from .text_encoding import build_condition, prepare_condition_inputs
from .transformer import AnimaTransformerModel

logger = logging.getLogger(__name__)

DEFAULT_ANIMA_TEXT_ENCODER_FILENAME = "model.safetensors"
DEFAULT_ANIMA_VAE_FILENAME = "qwen_image_vae.safetensors"


def _resolve_weight_path(
    pretrained_model_name_or_path: str,
    *,
    filename: str,
    subfolder: Optional[str] = None,
    revision: Optional[str] = None,
) -> str:
    if pretrained_model_name_or_path is None:
        raise ValueError("pretrained_model_name_or_path is required")
    if os.path.isfile(pretrained_model_name_or_path):
        return pretrained_model_name_or_path
    if os.path.isdir(pretrained_model_name_or_path):
        base = os.path.join(pretrained_model_name_or_path, subfolder) if subfolder else pretrained_model_name_or_path
        return os.path.join(base, filename)
    relative = os.path.join(subfolder, filename) if subfolder else filename
    return hf_hub_download(pretrained_model_name_or_path, filename=relative, revision=revision)


class Anima(ImageModelFoundation):
    NAME = "Anima"
    MODEL_DESCRIPTION = "CircleStone Labs image model with vendored diffusers-anima components"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKLQwenImage
    LATENT_CHANNEL_COUNT = 16
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]

    MODEL_CLASS = AnimaTransformerModel
    MODEL_SUBFOLDER = "split_files/diffusion_models"
    PIPELINE_CLASSES = {PipelineTypes.TEXT2IMG: AnimaPipeline}

    DEFAULT_MODEL_FLAVOUR = "preview"
    HUGGINGFACE_PATHS = {
        "preview": "circlestone-labs/Anima",
    }
    MODEL_LICENSE = "other"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Qwen3 0.6B",
            "model": Qwen3Model,
            "subfolder": "split_files/text_encoders",
        }
    }

    def _loader_options(self) -> AnimaLoaderOptions:
        return AnimaLoaderOptions(
            local_files_only=bool(getattr(self.config, "local_files_only", False)),
            cache_dir=getattr(self.config, "cache_dir", None),
            force_download=bool(getattr(self.config, "force_download", False)),
            token=getattr(self.config, "token", None),
            revision=getattr(self.config, "revision", None),
            proxies=getattr(self.config, "proxies", None),
        )

    def _prompt_tokenizer_sources(self) -> tuple[str, str]:
        model_path = getattr(self.config, "pretrained_model_name_or_path", None)
        if isinstance(model_path, str):
            model_dir = Path(model_path)
            qwen_dir = model_dir / "prompt_tokenizer_qwen"
            t5_dir = model_dir / "prompt_tokenizer_t5"
            if qwen_dir.is_dir() and t5_dir.is_dir():
                return str(qwen_dir), str(t5_dir)
        return _QWEN_TOKENIZER_SOURCE, _T5_TOKENIZER_SOURCE

    def load_text_tokenizer(self):
        qwen_source, t5_source = self._prompt_tokenizer_sources()
        self.prompt_tokenizer = load_prompt_tokenizer(
            qwen_tokenizer_source=qwen_source,
            t5_tokenizer_source=t5_source,
            options=self._loader_options(),
        )
        self.tokenizers = [self.prompt_tokenizer.qwen_tokenizer]
        self.tokenizer = self.prompt_tokenizer.qwen_tokenizer
        self.tokenizer_1 = self.prompt_tokenizer.qwen_tokenizer
        self.tokenizer_t5 = self.prompt_tokenizer.t5_tokenizer

    def load_text_encoder(self, move_to_device: bool = True):
        if self.text_encoders is not None and len(self.text_encoders) > 0:
            return
        model_path = (
            getattr(self.config, "pretrained_text_encoder_model_name_or_path", None)
            or self.config.pretrained_model_name_or_path
        )
        revision = getattr(self.config, "text_encoder_revision", None) or getattr(self.config, "revision", None)
        weight_path = _resolve_weight_path(
            model_path,
            filename=DEFAULT_ANIMA_TEXT_ENCODER_FILENAME,
            subfolder="split_files/text_encoders",
            revision=revision,
        )
        dtype = resolve_text_encoder_dtype(
            model_dtype=self.config.weight_dtype,
            text_encoder_dtype="auto",
            execution_device=self.accelerator.device.type,
        )
        load_device = self.accelerator.device.type if move_to_device else "cpu"
        text_encoder = load_text_encoder_single_file(
            file_path=weight_path,
            device=load_device,
            dtype=dtype,
        )
        self.text_encoders = [text_encoder]
        self.text_encoder = text_encoder
        self.text_encoder_1 = text_encoder
        if not move_to_device:
            text_encoder.to("cpu")
        if getattr(self, "prompt_tokenizer", None) is None:
            self.load_text_tokenizer()

    def unload_text_encoder(self):
        super().unload_text_encoder()
        self.text_encoder = None
        self.prompt_tokenizer = None
        self.tokenizer_t5 = None

    def load_vae(self, move_to_device: bool = True):
        model_path = self.config.pretrained_model_name_or_path
        revision = getattr(self.config, "revision", None)
        weight_path = _resolve_weight_path(
            model_path,
            filename=DEFAULT_ANIMA_VAE_FILENAME,
            subfolder="split_files/vae",
            revision=revision,
        )
        load_device = self.accelerator.device.type if move_to_device else "cpu"
        self.vae = load_vae_single_file(
            weight_path,
            device=load_device,
            dtype=self.config.weight_dtype,
        )
        self.AUTOENCODER_SCALING_FACTOR = getattr(self.vae.config, "scaling_factor", 1.0)
        if not move_to_device:
            self.vae.to("cpu")

    def _load_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True):
        active_pipelines = getattr(self, "pipelines", {})
        if pipeline_type in active_pipelines:
            pipeline_instance = active_pipelines[pipeline_type]
            if self.model is not None:
                pipeline_instance.transformer = self.unwrap_model(self.model)
            return pipeline_instance

        if pipeline_type not in self.PIPELINE_CLASSES:
            raise NotImplementedError(f"Pipeline type {pipeline_type} not defined in {self.__class__.__name__}.")

        if load_base_model:
            if self.model is None:
                self.load_model(move_to_device=True)
            if self.vae is None:
                self.load_vae(move_to_device=True)
            if self.text_encoders is None:
                self.load_text_encoder(move_to_device=True)
            if getattr(self, "prompt_tokenizer", None) is None:
                self.load_text_tokenizer()

        transformer = self.unwrap_model(self.model)
        text_encoder = self.text_encoders[0] if self.text_encoders else None
        if transformer is None or text_encoder is None or self.vae is None:
            raise RuntimeError("Anima pipeline requires transformer, text_encoder, and vae to be loaded.")

        scheduler = getattr(self, "noise_schedule", None)
        if scheduler is None:
            scheduler = AnimaFlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
                use_dynamic_shifting=False,
            )
        else:
            scheduler = coerce_anima_scheduler(scheduler.__class__.from_config(scheduler.config))

        text_encoder_dtype = next(text_encoder.parameters()).dtype
        pipeline_instance = self.PIPELINE_CLASSES[pipeline_type](
            transformer=transformer,
            vae=self.get_vae(),
            scheduler=scheduler,
            text_encoder=text_encoder,
            prompt_tokenizer=self.prompt_tokenizer,
            execution_device=self.accelerator.device.type,
            model_dtype=self.config.weight_dtype,
            text_encoder_dtype=text_encoder_dtype,
            use_module_cpu_offload=False,
        )
        self.pipelines[pipeline_type] = pipeline_instance
        return pipeline_instance

    def pre_vae_encode_transform_sample(self, sample):
        if sample.ndim == 4:
            return sample.unsqueeze(2)
        return sample

    def pre_latent_decode(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.ndim == 4:
            return latents.unsqueeze(2)
        return latents

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        del is_negative_prompt
        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()
        if getattr(self, "prompt_tokenizer", None) is None:
            self.load_text_tokenizer()

        prompt_embeds, t5xxl_ids, t5xxl_weights = prepare_condition_inputs(
            self.prompt_tokenizer,
            self.text_encoders[0],
            prompts,
            execution_device=self.accelerator.device.type,
            model_dtype=self.config.weight_dtype,
        )

        return {
            "prompt_embeds": prompt_embeds,
            "t5xxl_ids": t5xxl_ids,
            "t5xxl_weights": t5xxl_weights,
        }

    def _format_text_embedding(self, text_embedding):
        return text_embedding

    def _build_pipeline_condition(self, text_embedding: dict) -> torch.Tensor:
        if self.model is None:
            self.load_model(move_to_device=True)
        condition = build_condition(
            self.unwrap_model(self.model),
            qwen_hidden=text_embedding["prompt_embeds"].to(
                device=self.accelerator.device,
                dtype=self.config.weight_dtype,
            ),
            t5_ids=text_embedding["t5xxl_ids"].to(device=self.accelerator.device, dtype=torch.int32),
            t5_weights=text_embedding["t5xxl_weights"].to(device=self.accelerator.device, dtype=torch.float32),
        )
        if condition.dim() == 2:
            condition = condition.unsqueeze(0)
        return condition

    def convert_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        return {
            "prompt_embeds": self._build_pipeline_condition(text_embedding),
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        return {
            "negative_prompt_embeds": self._build_pipeline_condition(text_embedding),
        }

    def prepare_batch(self, batch: dict, state: dict) -> dict:
        batch = super().prepare_batch(batch, state)
        if batch.get("t5xxl_ids") is not None:
            batch["t5xxl_ids"] = batch["t5xxl_ids"].to(device=self.accelerator.device, dtype=torch.int32)
        if batch.get("t5xxl_weights") is not None:
            batch["t5xxl_weights"] = batch["t5xxl_weights"].to(device=self.accelerator.device, dtype=torch.float32)
        return batch

    def model_predict(self, prepared_batch, custom_timesteps: list = None):
        del custom_timesteps
        latents = prepared_batch["noisy_latents"]
        if latents.dim() == 4:
            latents = latents.unsqueeze(2)
        timesteps = prepared_batch["timesteps"]
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, device=self.accelerator.device, dtype=torch.float32)
        else:
            timesteps = timesteps.to(device=self.accelerator.device, dtype=torch.float32)

        noise_pred = self.model(
            latents.to(device=self.accelerator.device, dtype=self.config.weight_dtype),
            timesteps,
            prepared_batch["encoder_hidden_states"].to(device=self.accelerator.device, dtype=self.config.weight_dtype),
            t5xxl_ids=prepared_batch.get("t5xxl_ids"),
            t5xxl_weights=prepared_batch.get("t5xxl_weights"),
        )
        if hasattr(noise_pred, "sample"):
            noise_pred = noise_pred.sample
        elif isinstance(noise_pred, tuple):
            noise_pred = noise_pred[0]
        if noise_pred.dim() == 5 and noise_pred.shape[2] == 1:
            noise_pred = noise_pred.squeeze(2)
        return {"model_prediction": noise_pred.float(), "hidden_states_buffer": None}

    def get_latent_shapes(self, resolution: tuple) -> tuple:
        height, width = resolution
        return (height // 8, width // 8)

    def check_user_config(self):
        super().check_user_config()
        if self.config.tokenizer_max_length is None:
            self.config.tokenizer_max_length = 512
        if getattr(self.config, "aspect_bucket_alignment", None) != 16:
            logger.warning("%s requires 16px aspect bucket alignment; overriding.", self.NAME)
            self.config.aspect_bucket_alignment = 16


ModelRegistry.register("anima", Anima)
