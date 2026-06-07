from __future__ import annotations

import logging
import types
from typing import Optional

import torch
from huggingface_hub import hf_hub_download

from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.ideogram.constants import (
    IMAGE_POSITION_OFFSET,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
)
from simpletuner.helpers.models.ideogram.autoencoder import AutoEncoder
from simpletuner.helpers.models.ideogram.pipeline import (
    Ideogram4Config,
    Ideogram4Pipeline,
    Ideogram4PipelineConfig,
    _build_transformer,
    _load_autoencoder,
    _load_indexed_or_single_state_dict,
    _load_qwen3_vl,
)
from simpletuner.helpers.models.ideogram.prompt_enhancer import Ideogram4PromptEnhancerHead
from simpletuner.helpers.models.ideogram.prompting import maybe_convert_prompt_to_ideogram_json
from simpletuner.helpers.models.ideogram.scheduler import get_schedule_for_resolution
from simpletuner.helpers.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


class Ideogram4(ImageModelFoundation):
    NAME = "Ideogram 4"
    MODEL_DESCRIPTION = "Ideogram 4 conditional transformer"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    MODEL_CLASS = types.SimpleNamespace
    AUTOENCODER_CLASS = AutoEncoder
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {PipelineTypes.TEXT2IMG: Ideogram4Pipeline}
    TEXT_ENCODER_CONFIGURATION = {"text_encoder": {"name": "Qwen3-VL-8B-Instruct"}}
    LATENT_CHANNEL_COUNT = 32
    DEFAULT_MODEL_FLAVOUR = "fp8"
    HUGGINGFACE_PATHS = {
        "fp8": "ideogram-ai/ideogram-4-fp8",
        "nf4": "ideogram-ai/ideogram-4-nf4",
    }
    MODEL_LICENSE = "ideogram-4-non-commercial"
    VALIDATION_USES_NEGATIVE_PROMPT = True
    VALIDATION_USE_AUTOCAST = False
    SUPPORTS_LORA = True
    DEFAULT_LORA_TARGET = ["qkv", "o", "w1", "w2", "w3"]
    DEFAULT_PROMPT_ENHANCER_HEAD = "diffusers/qwen3-vl-8b-instruct-lm-head"
    PATCH_SIZE = 2
    AE_SCALE_FACTOR = 8

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        return 33

    def setup_model_flavour(self):
        super().setup_model_flavour()
        flavour = getattr(self.config, "model_flavour", None) or self.DEFAULT_MODEL_FLAVOUR
        if getattr(self.config, "pretrained_model_name_or_path", None) in (None, "", "None"):
            self.config.pretrained_model_name_or_path = self.HUGGINGFACE_PATHS.get(flavour, self.HUGGINGFACE_PATHS["fp8"])

    def _repo_id(self) -> str:
        return (
            getattr(self.config, "pretrained_transformer_model_name_or_path", None)
            or getattr(self.config, "pretrained_model_name_or_path", None)
            or self.HUGGINGFACE_PATHS["fp8"]
        )

    def load_model(self, move_to_device: bool = True):
        repo_id = self._repo_id()
        pipe_config = Ideogram4PipelineConfig(weights_repo=repo_id)
        state_dict = _load_indexed_or_single_state_dict(repo_id, pipe_config.conditional_index_filename)
        self.model = _build_transformer(
            Ideogram4Config(),
            state_dict,
            self.accelerator.device,
            self.config.weight_dtype,
        )
        if move_to_device:
            self.model.to(self.accelerator.device)
        return self.model

    def load_vae(self, move_to_device: bool = True):
        repo_id = getattr(self.config, "pretrained_model_name_or_path", None) or self.HUGGINGFACE_PATHS["fp8"]
        pipe_config = Ideogram4PipelineConfig(weights_repo=repo_id)
        autoencoder_weights = hf_hub_download(repo_id=repo_id, filename=pipe_config.autoencoder_filename)
        device = self.accelerator.device if move_to_device else torch.device("cpu")
        self.vae = _load_autoencoder(autoencoder_weights, device, self.config.weight_dtype)
        return self.vae

    def load_text_encoder(self, move_to_device: bool = True):
        repo_id = getattr(self.config, "pretrained_model_name_or_path", None) or self.HUGGINGFACE_PATHS["fp8"]
        pipe_config = Ideogram4PipelineConfig(weights_repo=repo_id)
        tokenizer, text_encoder = _load_qwen3_vl(
            repo_id,
            self.accelerator.device,
            self.config.weight_dtype,
            tokenizer_subfolder=pipe_config.tokenizer_subfolder,
            text_encoder_subfolder=pipe_config.text_encoder_subfolder,
        )
        self.tokenizers = [tokenizer]
        self.text_encoders = [text_encoder]
        if getattr(self.config, "ideogram_prompt_upsample", False):
            self.load_prompt_enhancer_head(move_to_device=move_to_device)
        return text_encoder

    def load_prompt_enhancer_head(self, move_to_device: bool = True):
        repo_id = (
            getattr(self.config, "ideogram_prompt_enhancer_head_id", None)
            or self.DEFAULT_PROMPT_ENHANCER_HEAD
        )
        self.prompt_enhancer_head = Ideogram4PromptEnhancerHead.from_pretrained(repo_id)
        self.prompt_enhancer_head.to(dtype=self.config.weight_dtype)
        if move_to_device:
            self.prompt_enhancer_head.to(self.accelerator.device)
        self.prompt_enhancer_head.eval()
        return self.prompt_enhancer_head

    def get_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True):
        if pipeline_type != PipelineTypes.TEXT2IMG:
            raise ValueError(f"{self.NAME} only supports text-to-image validation.")
        if not getattr(self.config, "ideogram_validation", False):
            raise ValueError("Ideogram validation is disabled unless --ideogram_validation is supplied.")
        if self.pipelines.get(pipeline_type) is not None:
            return self.pipelines[pipeline_type]

        if getattr(self, "model", None) is None:
            self.load_model(move_to_device=True)
        if getattr(self, "vae", None) is None:
            self.load_vae()
        if getattr(self, "text_encoders", None) is None:
            self.load_text_encoder(move_to_device=True)

        repo_id = getattr(self.config, "pretrained_model_name_or_path", None) or self.HUGGINGFACE_PATHS["fp8"]
        transformer = self.unwrap_model(self.model, keep_fp32_wrapper=False)
        pipeline = Ideogram4Pipeline(
            conditional_transformer=transformer,
            unconditional_transformer=None,
            text_encoder=self.text_encoders[0],
            text_tokenizer=self.tokenizers[0],
            autoencoder=self.vae,
            config=Ideogram4PipelineConfig(weights_repo=repo_id),
            device=self.accelerator.device,
            dtype=self.config.weight_dtype,
            prompt_enhancer_head=getattr(self, "prompt_enhancer_head", None),
        )
        self.pipelines[pipeline_type] = pipeline
        self.pipeline = pipeline
        return pipeline

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        if "prompt" in pipeline_kwargs and pipeline_kwargs["prompt"] is not None:
            pipeline_kwargs["prompts"] = maybe_convert_prompt_to_ideogram_json(
                pipeline_kwargs.pop("prompt"),
                enabled=getattr(self.config, "ideogram_auto_json", True),
            )
        elif pipeline_kwargs.get("prompt") is None:
            pipeline_kwargs.pop("prompt", None)
        pipeline_kwargs.pop("negative_prompt", None)
        if "num_inference_steps" in pipeline_kwargs:
            pipeline_kwargs["num_steps"] = pipeline_kwargs.pop("num_inference_steps")
        if "generator" in pipeline_kwargs:
            generator = pipeline_kwargs.pop("generator")
            seed = getattr(generator, "initial_seed", lambda: None)()
            if seed is not None:
                pipeline_kwargs["seed"] = int(seed)
        pipeline_kwargs.pop("num_images_per_prompt", None)
        pipeline_kwargs.setdefault("raise_on_caption_issues", False)
        return pipeline_kwargs

    def _prompt_upsample_resolution(self) -> tuple[int, int]:
        resolution = (
            getattr(self.config, "resolution", None)
            or getattr(self.config, "validation_resolution", None)
            or getattr(self.config, "maximum_image_size", None)
            or "1024x1024"
        )
        if isinstance(resolution, (tuple, list)) and len(resolution) >= 2:
            return int(resolution[0]), int(resolution[1])
        if isinstance(resolution, int):
            return resolution, resolution
        if isinstance(resolution, str):
            parts = resolution.lower().replace(",", "x").split("x")
            if len(parts) >= 2:
                try:
                    return int(parts[0].strip()), int(parts[1].strip())
                except ValueError:
                    pass
            try:
                parsed = int(resolution)
                return parsed, parsed
            except ValueError:
                pass
        return 1024, 1024

    def _maybe_upsample_prompt(self, prompt: str, encoder_shell: Ideogram4Pipeline) -> str:
        if not getattr(self.config, "ideogram_prompt_upsample", False):
            return prompt
        if getattr(self, "prompt_enhancer_head", None) is not None and hasattr(self.prompt_enhancer_head, "to"):
            self.prompt_enhancer_head.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        upsample_prompt = getattr(encoder_shell, "upsample_prompt", None)
        if upsample_prompt is None:
            raise ValueError("--ideogram_prompt_upsample requires an Ideogram pipeline with upsample_prompt().")
        height, width = self._prompt_upsample_resolution()
        upsampled = upsample_prompt(prompt, height=height, width=width, device=self.accelerator.device)
        if isinstance(upsampled, list):
            return upsampled[0]
        return upsampled

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        if getattr(self, "text_encoders", None) is None:
            self.load_text_encoder(move_to_device=True)
        if getattr(self.config, "ideogram_prompt_upsample", False) and getattr(self, "prompt_enhancer_head", None) is None:
            self.load_prompt_enhancer_head(move_to_device=True)

        tokenizer = self.tokenizers[0]
        if not getattr(self, "_logged_text_encoder_devices", False):
            text_encoder = self.text_encoders[0]
            text_encoder_device = getattr(text_encoder, "device", None)
            if text_encoder_device is None and hasattr(text_encoder, "parameters"):
                try:
                    text_encoder_device = next(text_encoder.parameters()).device
                except StopIteration:
                    text_encoder_device = "unknown"
            if text_encoder_device is None:
                text_encoder_device = "unknown"
            logger.info(
                "Ideogram text encoder device: %s; prompt upsample: %s",
                text_encoder_device,
                bool(getattr(self.config, "ideogram_prompt_upsample", False)),
            )
            head = getattr(self, "prompt_enhancer_head", None)
            if head is not None:
                head_device = getattr(head, "device", None)
                if head_device is None and hasattr(head, "parameters"):
                    try:
                        head_device = next(head.parameters()).device
                    except StopIteration:
                        head_device = "unknown"
                if head_device is None:
                    head_device = "unknown"
                logger.info("Ideogram prompt enhancer head device: %s", head_device)
            self._logged_text_encoder_devices = True
        encoder_shell = Ideogram4Pipeline(
            conditional_transformer=None,
            unconditional_transformer=None,
            text_encoder=self.text_encoders[0],
            text_tokenizer=tokenizer,
            autoencoder=None,
            config=Ideogram4PipelineConfig(weights_repo=self._repo_id()),
            device=self.accelerator.device,
            dtype=self.config.weight_dtype,
            prompt_enhancer_head=getattr(self, "prompt_enhancer_head", None),
        )
        tokenized = []
        for prompt in prompts:
            prompt = self._maybe_upsample_prompt(str(prompt), encoder_shell)
            prompt = maybe_convert_prompt_to_ideogram_json(
                prompt,
                enabled=getattr(self.config, "ideogram_auto_json", True),
            )
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            tokenized.append(encoded["input_ids"][0])

        max_text_tokens = max(int(tokens.shape[0]) for tokens in tokenized)
        if max_text_tokens > 2048:
            raise ValueError(f"prompt has {max_text_tokens} tokens, exceeds max_text_tokens=2048")

        batch_size = len(tokenized)
        token_ids = torch.zeros(batch_size, max_text_tokens, dtype=torch.long, device=self.accelerator.device)
        text_position_ids = torch.zeros(batch_size, max_text_tokens, 3, dtype=torch.long, device=self.accelerator.device)
        attention_mask = torch.zeros(batch_size, max_text_tokens, dtype=torch.bool, device=self.accelerator.device)
        for batch_idx, tokens in enumerate(tokenized):
            tokens = tokens.to(self.accelerator.device)
            num_text = int(tokens.shape[0])
            token_ids[batch_idx, :num_text] = tokens
            text_pos = torch.arange(num_text, device=self.accelerator.device)
            text_position_ids[batch_idx, :num_text] = torch.stack([text_pos, text_pos, text_pos], dim=1)
            attention_mask[batch_idx, :num_text] = True

        indicator = torch.full_like(token_ids, LLM_TOKEN_INDICATOR)
        prompt_embeds = encoder_shell._encode_text(token_ids, text_position_ids, indicator)
        prompt_embeds = prompt_embeds * attention_mask.to(prompt_embeds.dtype).unsqueeze(-1)
        return {"prompt_embeds": prompt_embeds, "attention_mask": attention_mask}

    def _format_text_embedding(self, text_embedding: dict):
        return text_embedding

    def convert_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        attention_mask = text_embedding.get("attention_mask")
        if attention_mask is None:
            attention_mask = text_embedding.get("attention_masks")
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": attention_mask,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        attention_mask = text_embedding.get("attention_mask")
        if attention_mask is None:
            attention_mask = text_embedding.get("attention_masks")
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        return {
            "negative_prompt_embeds": prompt_embeds,
            "negative_prompt_attention_mask": attention_mask,
        }

    def collate_prompt_embeds(self, text_encoder_output: list[dict]) -> dict:
        if not text_encoder_output:
            return {}

        embeds = []
        masks = []
        max_len = 0
        for item in text_encoder_output:
            prompt_embeds = item["prompt_embeds"]
            attention_mask = item.get("attention_mask")
            if attention_mask is None:
                attention_mask = item.get("attention_masks")
            if prompt_embeds.dim() == 3:
                prompt_embeds = prompt_embeds.squeeze(0)
            if attention_mask is None:
                attention_mask = torch.ones(prompt_embeds.shape[0], dtype=torch.bool, device=prompt_embeds.device)
            elif attention_mask.dim() == 2:
                attention_mask = attention_mask.squeeze(0)
            attention_mask = attention_mask.to(dtype=torch.bool)
            length = int(attention_mask.sum().item())
            prompt_embeds = prompt_embeds[:length]
            attention_mask = attention_mask[:length]
            embeds.append(prompt_embeds)
            masks.append(attention_mask)
            max_len = max(max_len, length)

        padded_embeds = []
        padded_masks = []
        for prompt_embeds, attention_mask in zip(embeds, masks):
            pad_len = max_len - prompt_embeds.shape[0]
            if pad_len > 0:
                prompt_embeds = torch.nn.functional.pad(prompt_embeds, (0, 0, 0, pad_len))
                attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=False)
            padded_embeds.append(prompt_embeds.unsqueeze(0))
            padded_masks.append(attention_mask.unsqueeze(0))

        return {
            "prompt_embeds": torch.cat(padded_embeds, dim=0),
            "attention_masks": torch.cat(padded_masks, dim=0),
        }

    @torch.no_grad()
    def encode_with_vae(self, vae, samples):
        encoded = vae.encoder(samples.to(device=self.accelerator.device, dtype=self.config.weight_dtype))
        mean, _logvar = encoded.chunk(2, dim=1)
        return mean

    def _patchify_vae_latents(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = latents.shape
        patch = self.PATCH_SIZE
        if height % patch != 0 or width % patch != 0:
            raise ValueError(f"Ideogram latent height/width must be divisible by {patch}, got {height}x{width}.")
        latents = latents.view(batch_size, channels, height // patch, patch, width // patch, patch)
        latents = latents.permute(0, 2, 4, 3, 5, 1).contiguous()
        latents = latents.view(batch_size, height // patch, width // patch, patch * patch * channels)
        return latents.permute(0, 3, 1, 2).contiguous()

    def _unpatchify_vae_latents(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = latents.shape
        patch = self.PATCH_SIZE
        ae_channels = channels // (patch * patch)
        latents = latents.permute(0, 2, 3, 1).contiguous()
        latents = latents.view(batch_size, height, width, patch, patch, ae_channels)
        latents = latents.permute(0, 5, 1, 3, 2, 4).contiguous()
        return latents.view(batch_size, ae_channels, height * patch, width * patch)

    def _normalize_packed_vae_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae = self.get_vae()
        if vae is None:
            raise ValueError("Cannot normalize Ideogram latents without a loaded VAE.")
        mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device=latents.device, dtype=latents.dtype)
        std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.bn.eps).to(device=latents.device, dtype=latents.dtype)
        return (latents - mean) / std

    def post_vae_encode_transform_sample(self, sample):
        if hasattr(sample, "latent_dist"):
            sample = sample.latent_dist.mode()
        elif hasattr(sample, "sample") and not torch.is_tensor(sample):
            sample = sample.sample
        packed = self._patchify_vae_latents(sample)
        return self._normalize_packed_vae_latents(packed)

    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = latents.shape
        expected_channels = self.LATENT_CHANNEL_COUNT * self.PATCH_SIZE * self.PATCH_SIZE
        if channels != expected_channels:
            raise ValueError(f"Ideogram expects packed 128-channel latents, got {channels} channels.")
        return latents.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)

    def _unpack_latents(self, packed: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, _tokens, channels = packed.shape
        return packed.view(batch_size, height, width, channels).permute(0, 3, 1, 2).contiguous()

    def _image_position_ids(self, batch_size: int, grid_h: int, grid_w: int) -> torch.Tensor:
        h_idx = torch.arange(grid_h, device=self.accelerator.device).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
        w_idx = torch.arange(grid_w, device=self.accelerator.device).view(1, -1).expand(grid_h, grid_w).reshape(-1)
        t_idx = torch.zeros_like(h_idx)
        image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET
        return image_pos.unsqueeze(0).expand(batch_size, -1, -1)

    def model_predict(self, prepared_batch):
        noisy_latents = prepared_batch["noisy_latents"].to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        batch_size, _channels, latent_height, latent_width = noisy_latents.shape
        packed_latents = self._pack_latents(noisy_latents)
        image_tokens = packed_latents.shape[1]

        prompt_embeds = prepared_batch.get("prompt_embeds")
        if prompt_embeds is None:
            prompt_embeds = prepared_batch["encoder_hidden_states"]
        prompt_embeds = prompt_embeds.to(device=self.accelerator.device, dtype=torch.float32)
        attention_mask = prepared_batch.get("encoder_attention_mask")
        if attention_mask is None:
            attention_mask = prepared_batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = prepared_batch.get("attention_masks")
        if attention_mask is None:
            attention_mask = torch.ones(prompt_embeds.shape[:2], dtype=torch.bool, device=self.accelerator.device)
        else:
            attention_mask = attention_mask.to(device=self.accelerator.device, dtype=torch.bool)
        text_tokens = prompt_embeds.shape[1]

        text_position = torch.arange(text_tokens, device=self.accelerator.device)
        text_position_ids = torch.stack([text_position, text_position, text_position], dim=1).unsqueeze(0).expand(
            batch_size, -1, -1
        )
        image_position_ids = self._image_position_ids(batch_size, latent_height, latent_width)
        position_ids = torch.cat([text_position_ids, image_position_ids], dim=1)
        segment_ids = torch.cat(
            [
                attention_mask.to(torch.long),
                torch.ones(batch_size, image_tokens, dtype=torch.long, device=self.accelerator.device),
            ],
            dim=1,
        )
        indicator = torch.cat(
            [
                torch.full((batch_size, text_tokens), LLM_TOKEN_INDICATOR, dtype=torch.long, device=self.accelerator.device),
                torch.full((batch_size, image_tokens), OUTPUT_IMAGE_INDICATOR, dtype=torch.long, device=self.accelerator.device),
            ],
            dim=1,
        )
        llm_features = torch.cat(
            [
                prompt_embeds,
                torch.zeros(
                    batch_size,
                    image_tokens,
                    prompt_embeds.shape[-1],
                    dtype=prompt_embeds.dtype,
                    device=self.accelerator.device,
                ),
            ],
            dim=1,
        )
        text_z_padding = torch.zeros(
            batch_size,
            text_tokens,
            packed_latents.shape[-1],
            dtype=packed_latents.dtype,
            device=self.accelerator.device,
        )
        model_input = torch.cat([text_z_padding, packed_latents], dim=1)
        timesteps = prepared_batch["timesteps"].to(device=self.accelerator.device, dtype=torch.float32)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(batch_size)
        if timesteps.max() > 1:
            timesteps = timesteps / 1000.0
        timesteps = 1.0 - timesteps

        model_output = self.model(
            llm_features=llm_features,
            x=model_input,
            t=timesteps,
            position_ids=position_ids,
            segment_ids=segment_ids,
            indicator=indicator,
        )
        packed_prediction = model_output[:, text_tokens:]
        model_prediction = self._unpack_latents(packed_prediction, latent_height, latent_width)
        return {"model_prediction": model_prediction * -1}

    def sample_flow_sigmas(self, batch: dict, state: dict) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = batch["latents"].shape[0]
        device = self.accelerator.device
        latents = batch["latents"]
        image_height = int(latents.shape[-2] * self.AE_SCALE_FACTOR * self.PATCH_SIZE)
        image_width = int(latents.shape[-1] * self.AE_SCALE_FACTOR * self.PATCH_SIZE)
        mu = float(getattr(self.config, "ideogram_schedule_mu", 0.0) or 0.0)
        std = float(getattr(self.config, "ideogram_schedule_std", 1.5) or 1.5)
        schedule = get_schedule_for_resolution((image_height, image_width), known_mean=mu, std=std)
        schedule_u = torch.rand((bsz,), device=device, dtype=torch.float32)
        model_t = schedule(schedule_u).to(device=device, dtype=torch.float32)
        sigmas = (1.0 - model_t).clamp(0.0, 1.0)
        timesteps = sigmas * 1000.0
        return sigmas, timesteps

    def requires_special_scheduler_setup(self) -> bool:
        return True

    def setup_training_noise_schedule(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        self.noise_schedule = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=getattr(self.config, "flow_schedule_shift", 1.0) or 1.0,
        )
        self.config.prediction_type = "flow_matching"
        return self.config, self.noise_schedule

    def check_user_config(self):
        super().check_user_config()
        self.config.prediction_type = "flow_matching"

    def log_model_devices(self):
        super().log_model_devices()
        head = getattr(self, "prompt_enhancer_head", None)
        if head is not None:
            try:
                device = next(head.parameters()).device
            except StopIteration:
                device = "unknown"
            logger.debug(f"Prompt enhancer head device: {device}")


ModelRegistry.register("ideogram", Ideogram4)
