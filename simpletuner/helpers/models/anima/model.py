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
from simpletuner.helpers.training.crepa import CrepaMode

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
    DIFFUSERS_MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {PipelineTypes.TEXT2IMG: AnimaPipeline}

    DEFAULT_MODEL_FLAVOUR = "preview-3"
    HUGGINGFACE_PATHS = {
        "preview-3": "CalamitousFelicitousness/Anima-Preview-3-sdnext-diffusers",
        "preview-2": "CalamitousFelicitousness/Anima-Preview-2-sdnext-diffusers",
        "preview": "CalamitousFelicitousness/Anima-sdnext-diffusers",
    }
    DIFFUSERS_LAYOUT_PATHS = set(HUGGINGFACE_PATHS.values())
    MODEL_LICENSE = "other"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Qwen3 0.6B",
            "model": Qwen3Model,
            "subfolder": "split_files/text_encoders",
        }
    }

    @property
    def crepa_mode(self) -> CrepaMode:
        return CrepaMode.IMAGE

    def supports_crepa_self_flow(self) -> bool:
        return True

    def _select_crepa_hidden_states(self, prepared_batch: dict, hidden_states_buffer):
        if hidden_states_buffer is None:
            return None
        crepa = getattr(self, "crepa_regularizer", None)
        capture_layer = prepared_batch.get(
            "crepa_capture_block_index",
            getattr(crepa, "block_index", None),
        )
        if capture_layer is None:
            return None
        return hidden_states_buffer.get(f"layer_{int(capture_layer)}")

    def _prepare_crepa_self_flow_batch(self, batch: dict, state: dict) -> dict:
        latents = batch["latents"]
        input_noise = batch["input_noise"]
        base_sigmas = batch["sigmas"].to(device=latents.device, dtype=latents.dtype).view(-1)
        base_timesteps = batch["timesteps"].to(device=latents.device, dtype=latents.dtype).view(-1)
        alt_sigmas, alt_timesteps = self.sample_flow_sigmas(batch=batch, state=state)
        alt_sigmas = alt_sigmas.to(device=latents.device, dtype=latents.dtype)
        alt_timesteps = alt_timesteps.to(device=latents.device, dtype=latents.dtype)

        _, _, num_frames, height, width = latents.shape
        p_t, p_h, p_w = self._crepa_self_flow_patch_size()
        token_frames = max(num_frames // p_t, 1)
        token_height = max(height // p_h, 1)
        token_width = max(width // p_w, 1)

        mask_ratio = float(getattr(self.config, "crepa_self_flow_mask_ratio", 0.1) or 0.0)
        token_mask = (
            torch.rand(
                latents.shape[0],
                token_frames,
                token_height,
                token_width,
                device=latents.device,
                dtype=latents.dtype,
            )
            < mask_ratio
        )

        base_sigma_view = base_sigmas.view(-1, 1, 1, 1)
        alt_sigma_view = alt_sigmas.view(-1, 1, 1, 1)
        student_token_sigmas = torch.where(token_mask, alt_sigma_view, base_sigma_view)
        student_sigma_grid = self._expand_crepa_self_flow_patch_values(student_token_sigmas, (p_t, p_h, p_w), latents.shape)
        student_sigma_grid = student_sigma_grid.to(dtype=latents.dtype)

        base_timestep_view = base_timesteps.view(-1, 1, 1, 1)
        alt_timestep_view = alt_timesteps.view(-1, 1, 1, 1)
        student_token_timesteps = torch.where(token_mask, alt_timestep_view, base_timestep_view)

        teacher_sigmas = torch.minimum(base_sigmas, alt_sigmas).view(-1, 1, 1, 1, 1)
        teacher_timesteps = torch.minimum(base_timesteps, alt_timesteps)

        batch["sigmas"] = student_sigma_grid
        batch["timesteps"] = student_token_timesteps.flatten(1)
        batch["noisy_latents"] = (1 - student_sigma_grid) * latents + student_sigma_grid * input_noise
        batch["crepa_teacher_sigmas"] = teacher_sigmas.to(dtype=latents.dtype)
        batch["crepa_teacher_timesteps"] = teacher_timesteps.to(dtype=latents.dtype)
        batch["crepa_teacher_noisy_latents"] = (1 - teacher_sigmas) * latents + teacher_sigmas * input_noise
        batch["crepa_self_flow_mask"] = token_mask
        return batch

    def _crepa_self_flow_patch_size(self) -> tuple[int, int, int]:
        transformer = self.unwrap_model(self.model) if getattr(self, "model", None) is not None else None
        config = getattr(transformer, "config", None)
        patch_size = getattr(config, "patch_size", (1, 2, 2))
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        if not isinstance(patch_size, (tuple, list)) or len(patch_size) != 3:
            raise ValueError(f"Unexpected Anima patch size for Self-Flow: {patch_size!r}")
        return tuple(max(int(value), 1) for value in patch_size)

    def _expand_crepa_self_flow_patch_values(
        self,
        values: torch.Tensor,
        patch_size: tuple[int, int, int],
        target_shape: tuple[int, ...],
    ) -> torch.Tensor:
        p_t, p_h, p_w = patch_size
        expanded = values.unsqueeze(1)
        expanded = expanded.repeat_interleave(p_t, dim=2)
        expanded = expanded.repeat_interleave(p_h, dim=3)
        expanded = expanded.repeat_interleave(p_w, dim=4)
        _, _, target_t, target_h, target_w = target_shape
        return expanded[:, :, :target_t, :target_h, :target_w]

    def _latent_sequence_length(self, latent_tensor: torch.Tensor) -> int:
        p_t, p_h, p_w = self._crepa_self_flow_patch_size()
        return max(
            (latent_tensor.shape[2] // p_t) * (latent_tensor.shape[3] // p_h) * (latent_tensor.shape[4] // p_w),
            1,
        )

    def _prepare_model_predict_timesteps(
        self,
        raw_timesteps,
        *,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not torch.is_tensor(raw_timesteps):
            raw_timesteps = torch.tensor(raw_timesteps, device=device, dtype=dtype)
        else:
            raw_timesteps = raw_timesteps.to(device=device, dtype=dtype)

        if raw_timesteps.ndim == 0:
            return raw_timesteps.expand(batch_size)
        if raw_timesteps.ndim == 1:
            if raw_timesteps.shape[0] == 1:
                return raw_timesteps.expand(batch_size)
            if raw_timesteps.shape[0] == batch_size:
                return raw_timesteps
            raise ValueError(f"Anima expected 1 timestep or {batch_size} per-batch timesteps, got {raw_timesteps.shape[0]}.")
        if raw_timesteps.ndim == 2:
            if raw_timesteps.shape[1] != sequence_length:
                raise ValueError(
                    f"Anima expected tokenwise timesteps with sequence length {sequence_length}, got {raw_timesteps.shape[1]}."
                )
            if raw_timesteps.shape[0] == 1:
                return raw_timesteps.expand(batch_size, -1)
            if raw_timesteps.shape[0] == batch_size:
                return raw_timesteps
            raise ValueError(
                f"Anima expected tokenwise timesteps for batch size {batch_size}, got {raw_timesteps.shape[0]}."
            )
        raise ValueError(
            f"Anima expected a scalar, 1D batch tensor, or 2D tokenwise tensor, got shape {tuple(raw_timesteps.shape)}."
        )

    def _loader_options(self) -> AnimaLoaderOptions:
        return AnimaLoaderOptions(
            local_files_only=bool(getattr(self.config, "local_files_only", False)),
            cache_dir=getattr(self.config, "cache_dir", None),
            force_download=bool(getattr(self.config, "force_download", False)),
            token=getattr(self.config, "token", None),
            revision=getattr(self.config, "revision", None),
            proxies=getattr(self.config, "proxies", None),
        )

    def _uses_diffusers_repo_layout(
        self,
        model_path: Optional[str] = None,
        *,
        component_subfolder: str = "transformer",
    ) -> bool:
        path_was_provided = model_path is not None
        if model_path is None:
            model_path = getattr(self.config, "pretrained_model_name_or_path", None)
        if isinstance(model_path, str):
            normalized_path = model_path.rstrip("/")
            if normalized_path in self.DIFFUSERS_LAYOUT_PATHS:
                return True
            model_dir = Path(model_path)
            if (model_dir / component_subfolder / "config.json").is_file():
                return True

        if path_was_provided or model_path is not None:
            return False

        flavour = getattr(self.config, "model_flavour", None)
        return flavour in self.HUGGINGFACE_PATHS and self.HUGGINGFACE_PATHS[flavour] in self.DIFFUSERS_LAYOUT_PATHS

    def _add_hf_token_kwarg(self, load_kwargs: dict) -> None:
        token = getattr(self.config, "token", None)
        if token not in (None, False):
            load_kwargs["token"] = token

    def load_model(self, move_to_device: bool = True):
        self.MODEL_SUBFOLDER = (
            self.DIFFUSERS_MODEL_SUBFOLDER if self._uses_diffusers_repo_layout() else "split_files/diffusion_models"
        )
        return super().load_model(move_to_device=move_to_device)

    def _prompt_tokenizer_sources(self) -> tuple[str, str]:
        model_path = getattr(self.config, "pretrained_model_name_or_path", None)
        if isinstance(model_path, str):
            model_dir = Path(model_path)
            qwen_dir = model_dir / "tokenizer"
            t5_dir = model_dir / "t5_tokenizer"
            if qwen_dir.is_dir() and t5_dir.is_dir():
                return str(qwen_dir), str(t5_dir)
            qwen_dir = model_dir / "prompt_tokenizer_qwen"
            t5_dir = model_dir / "prompt_tokenizer_t5"
            if qwen_dir.is_dir() and t5_dir.is_dir():
                return str(qwen_dir), str(t5_dir)
            if self._uses_diffusers_repo_layout():
                return f"{model_path}::tokenizer", f"{model_path}::t5_tokenizer"
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
        dtype = resolve_text_encoder_dtype(
            model_dtype=self.config.weight_dtype,
            text_encoder_dtype="auto",
            execution_device=self.accelerator.device.type,
        )
        load_device = self.accelerator.device.type if move_to_device else "cpu"
        if self._uses_diffusers_repo_layout(model_path, component_subfolder="text_encoder"):
            load_kwargs = {
                "pretrained_model_name_or_path": model_path,
                "subfolder": "text_encoder",
                "revision": revision,
                "torch_dtype": dtype,
                "local_files_only": bool(getattr(self.config, "local_files_only", False)),
                "cache_dir": getattr(self.config, "cache_dir", None),
                "force_download": bool(getattr(self.config, "force_download", False)),
            }
            self._add_hf_token_kwarg(load_kwargs)
            text_encoder = Qwen3Model.from_pretrained(**load_kwargs)
            text_encoder.eval().requires_grad_(False)
            text_encoder.to(device=load_device, dtype=dtype)
            self.text_encoders = [text_encoder]
            self.text_encoder = text_encoder
            self.text_encoder_1 = text_encoder
            if not move_to_device:
                text_encoder.to("cpu")
            if getattr(self, "prompt_tokenizer", None) is None:
                self.load_text_tokenizer()
            return

        weight_path = _resolve_weight_path(
            model_path,
            filename=DEFAULT_ANIMA_TEXT_ENCODER_FILENAME,
            subfolder="split_files/text_encoders",
            revision=revision,
        )
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
        load_device = self.accelerator.device.type if move_to_device else "cpu"
        if self._uses_diffusers_repo_layout(model_path, component_subfolder="vae"):
            load_kwargs = {
                "pretrained_model_name_or_path": model_path,
                "subfolder": "vae",
                "revision": revision,
                "torch_dtype": self.config.weight_dtype,
                "local_files_only": bool(getattr(self.config, "local_files_only", False)),
                "cache_dir": getattr(self.config, "cache_dir", None),
                "force_download": bool(getattr(self.config, "force_download", False)),
            }
            self._add_hf_token_kwarg(load_kwargs)
            self.vae = self.AUTOENCODER_CLASS.from_pretrained(**load_kwargs)
            self.vae.eval().requires_grad_(False)
            self.vae.to(device=load_device, dtype=self.config.weight_dtype)
            self.AUTOENCODER_SCALING_FACTOR = getattr(self.vae.config, "scaling_factor", 1.0)
            if not move_to_device:
                self.vae.to("cpu")
            return

        weight_path = _resolve_weight_path(
            model_path,
            filename=DEFAULT_ANIMA_VAE_FILENAME,
            subfolder="split_files/vae",
            revision=revision,
        )
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
        batch_size = latents.shape[0]
        timesteps = self._prepare_model_predict_timesteps(
            prepared_batch["timesteps"],
            batch_size=batch_size,
            sequence_length=self._latent_sequence_length(latents),
            device=self.accelerator.device,
            dtype=torch.float32,
        )
        hidden_states_buffer = self._new_hidden_state_buffer()

        noise_pred = self.model(
            latents.to(device=self.accelerator.device, dtype=self.config.weight_dtype),
            timesteps,
            prepared_batch["encoder_hidden_states"].to(device=self.accelerator.device, dtype=self.config.weight_dtype),
            t5xxl_ids=prepared_batch.get("t5xxl_ids"),
            t5xxl_weights=prepared_batch.get("t5xxl_weights"),
            hidden_states_buffer=hidden_states_buffer,
        )
        if hasattr(noise_pred, "sample"):
            noise_pred = noise_pred.sample
        elif isinstance(noise_pred, tuple):
            noise_pred = noise_pred[0]
        if noise_pred.dim() == 5 and noise_pred.shape[2] == 1:
            noise_pred = noise_pred.squeeze(2)
        return {
            "model_prediction": noise_pred.float(),
            "crepa_hidden_states": self._select_crepa_hidden_states(prepared_batch, hidden_states_buffer),
            "hidden_states_buffer": hidden_states_buffer,
        }

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
