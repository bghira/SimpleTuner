import logging
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from simpletuner.helpers.acceleration import (
    AccelerationBackend,
    AccelerationPreset,
    get_bitsandbytes_presets,
    get_deepspeed_presets,
    get_quanto_presets,
    get_sdnq_presets,
    get_torchao_presets,
)
from simpletuner.helpers.models.boogu_image.pipeline import BooguImagePipeline
from simpletuner.helpers.models.boogu_image.pipeline_edit import BooguImageEditPipeline
from simpletuner.helpers.models.boogu_image.pipeline_img2img import BooguImageImg2ImgPipeline
from simpletuner.helpers.models.boogu_image.pipeline_turbo import BooguImageTurboPipeline
from simpletuner.helpers.models.boogu_image.schedulers.scheduling_flow_match_euler_discrete_time_shifting import (
    FlowMatchEulerDiscreteScheduler,
)
from simpletuner.helpers.models.boogu_image.transformer import BooguImageTransformer2DModel
from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.flux.model import Flux
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.training.deepspeed import deepspeed_zero_init_disabled_context_manager

logger = logging.getLogger(__name__)


class BooguImage(ImageModelFoundation):
    SUPPORTS_MUON_CLIP = True
    NAME = "Boogu-Image"
    MODEL_DESCRIPTION = "Boogu-Image 0.1 multimodal image generation and editing model"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    USES_DYNAMIC_SHIFT = True
    AUTO_LORA_FORMAT_DETECTION = True
    AUTOENCODER_CLASS = AutoencoderKL
    AUTOENCODER_SCALING_FACTOR = 0.3611
    LATENT_CHANNEL_COUNT = 16
    VALIDATION_PREVIEW_SPEC = Flux.VALIDATION_PREVIEW_SPEC
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    SLIDER_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]

    MODEL_CLASS = BooguImageTransformer2DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: BooguImagePipeline,
        PipelineTypes.IMG2IMG: BooguImageImg2ImgPipeline,
    }

    DEFAULT_MODEL_FLAVOUR = "v0.1-turbo"
    HUGGINGFACE_PATHS = {
        "v0.1-base": "SimpleTuner/Boogu-Image-0.1-Base",
        "v0.1-base-fp8": "SimpleTuner/Boogu-Image-0.1-Base",
        "v0.1-turbo": "SimpleTuner/Boogu-Image-0.1-Turbo",
        "v0.1-turbo-fp8": "SimpleTuner/Boogu-Image-0.1-Turbo",
        "v0.1-edit": "SimpleTuner/Boogu-Image-0.1-Edit",
        "v0.1-edit-fp8": "SimpleTuner/Boogu-Image-0.1-Edit",
    }
    MODEL_LICENSE = "apache-2.0"

    ASSISTANT_LORA_FLAVOURS = ["v0.1-turbo", "v0.1-turbo-fp8"]
    ASSISTANT_LORA_PATH = None
    ASSISTANT_LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"

    TEXT_ENCODER_CONFIGURATION = {
        "mllm": {
            "name": "Qwen3-VL",
            "model": Qwen3VLForConditionalGeneration,
            "subfolder": "mllm",
        },
    }
    PROCESSOR_CLASS = Qwen3VLProcessor
    PROCESSOR_SUBFOLDER = "processor"

    SYSTEM_PROMPT_4_T2I = "You are a helpful assistant that describes images for text-to-image generation."
    SYSTEM_PROMPT_4_TI2I = "You are a helpful assistant that describes image editing instructions."
    SYSTEM_PROMPT_4_I2I = "You are a helpful assistant that describes image transformations."
    SYSTEM_PROMPT_DROP = "You are a helpful assistant."

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        # 40 single-stream layers plus 8 double-stream layers and 2 refiners.
        return 49

    @classmethod
    def get_acceleration_presets(cls) -> list[AccelerationPreset]:
        base_config = {
            "base_model_precision": "no_change",
            "gradient_checkpointing": True,
        }
        return [
            AccelerationPreset(
                backend=AccelerationBackend.RAMTORCH,
                level="balanced",
                name="RamTorch - Balanced",
                description="Streams the Boogu transformer blocks from CPU RAM.",
                tab="basic",
                tradeoff_vram="Substantial VRAM savings for the 10B transformer.",
                tradeoff_speed="Increases training time from CPU-GPU transfers.",
                tradeoff_notes="Requires high system RAM.",
                requires_min_system_ram_gb=128,
                config={**base_config, "ramtorch": True, "ramtorch_target_modules": "*layers.*"},
            ),
            *get_deepspeed_presets(base_config),
            *get_sdnq_presets(base_config),
            *get_torchao_presets(base_config),
            *get_quanto_presets(base_config),
            *get_bitsandbytes_presets(base_config),
        ]

    def __init__(self, config: dict, accelerator):
        super().__init__(config, accelerator)
        self.vae_scale_factor = 8
        self.processor = None
        self._validate_xm_support()
        if self._is_turbo_flavour():
            self.PIPELINE_CLASSES = {
                PipelineTypes.TEXT2IMG: BooguImageTurboPipeline,
                PipelineTypes.IMG2IMG: BooguImageImg2ImgPipeline,
            }
        elif self._is_edit_flavour():
            self.PIPELINE_CLASSES = {
                PipelineTypes.TEXT2IMG: BooguImageEditPipeline,
                PipelineTypes.IMG2IMG: BooguImageEditPipeline,
            }

    def _model_flavour(self) -> Optional[str]:
        return getattr(self.config, "model_flavour", None)

    def _is_edit_flavour(self) -> bool:
        flavour = self._model_flavour()
        return flavour in {"v0.1-edit", "v0.1-edit-fp8"}

    def _is_turbo_flavour(self) -> bool:
        flavour = self._model_flavour()
        return flavour in {"v0.1-turbo", "v0.1-turbo-fp8"}

    def requires_conditioning_dataset(self) -> bool:
        return self._is_edit_flavour()

    def requires_conditioning_latents(self) -> bool:
        return self._is_edit_flavour()

    def requires_conditioning_validation_inputs(self) -> bool:
        return self._is_edit_flavour()

    def requires_validation_edit_captions(self) -> bool:
        return self._is_edit_flavour()

    def requires_text_embed_image_context(self) -> bool:
        return self._is_edit_flavour()

    def requires_special_scheduler_setup(self) -> bool:
        return True

    def _load_scheduler_for_pipeline(self, pipeline_type: str):
        return FlowMatchEulerDiscreteScheduler.from_pretrained(
            self._model_config_path(),
            subfolder="scheduler",
            revision=getattr(self.config, "revision", None),
            local_files_only=getattr(self.config, "local_files_only", False),
        )

    def sample_flow_sigmas(self, batch: dict, state: dict) -> tuple[torch.Tensor, torch.Tensor]:
        noise_sigmas, _ = super().sample_flow_sigmas(batch=batch, state=state)
        boogu_timesteps = 1.0 - noise_sigmas
        return noise_sigmas, boogu_timesteps

    def flow_matching_target_direction(self) -> float:
        return -1.0

    def get_prediction_target(self, prepared_batch: dict):
        if prepared_batch.get("target") is not None:
            return prepared_batch["target"]
        return self.get_flow_matching_target(prepared_batch, prefer_explicit_target=False)

    def text_embed_cache_key(self):
        if self._is_edit_flavour():
            from simpletuner.helpers.models.common import TextEmbedCacheKey

            return TextEmbedCacheKey.DATASET_AND_FILENAME
        return super().text_embed_cache_key()

    def _load_processor_for_pipeline(self):
        if self.processor is not None:
            return self.processor
        processor_path = self._resolve_qwen_processor_path(self._model_config_path())
        processor_subfolder = self._resolve_qwen_processor_subfolder(self.PROCESSOR_SUBFOLDER)
        processor_kwargs = {
            "pretrained_model_name_or_path": processor_path,
            "subfolder": processor_subfolder,
            "revision": getattr(self.config, "revision", None),
        }
        self.processor = self.PROCESSOR_CLASS.from_pretrained(**processor_kwargs)
        return self.processor

    def load_text_tokenizer(self):
        if getattr(self, "tokenizers", None):
            return
        processor = self._load_processor_for_pipeline()
        tokenizer = getattr(processor, "tokenizer", processor)
        self.tokenizers = [tokenizer]
        self.tokenizer_1 = tokenizer

    def load_text_encoder(self, move_to_device: bool = True):
        if getattr(self, "text_encoders", None):
            return

        from transformers.utils import ContextManagers

        self.load_text_tokenizer()
        text_encoder_config = self.TEXT_ENCODER_CONFIGURATION["mllm"]
        text_encoder_path = self._resolve_text_encoder_path(text_encoder_config)
        load_kwargs = {
            "pretrained_model_name_or_path": text_encoder_path,
            "subfolder": text_encoder_config.get("subfolder", "mllm"),
            "revision": self.config.revision,
            "torch_dtype": self.config.weight_dtype,
        }
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            text_encoder = text_encoder_config["model"].from_pretrained(**load_kwargs)

        if hasattr(text_encoder, "model") and hasattr(text_encoder, "lm_head"):
            text_encoder = text_encoder.model
        if move_to_device and not self._ramtorch_text_encoders_requested():
            text_encoder.to(self.accelerator.device, dtype=self.config.weight_dtype)
        if hasattr(text_encoder, "eval"):
            text_encoder.eval()
        text_encoder.requires_grad_(False)

        self.text_encoders = [text_encoder]
        self.mllm = text_encoder
        self.text_encoder = text_encoder
        self.text_encoder_1 = text_encoder

    def get_pipeline(self, pipeline_type: str = PipelineTypes.TEXT2IMG, load_base_model: bool = True):
        pipeline = super().get_pipeline(pipeline_type=pipeline_type, load_base_model=load_base_model)
        if pipeline is not None and hasattr(pipeline, "processor") and self.processor is None:
            self.processor = pipeline.processor
        return pipeline

    def _system_prompt(self, has_images: bool, instruction: str) -> str:
        if not has_images:
            return self.SYSTEM_PROMPT_4_T2I if instruction else self.SYSTEM_PROMPT_DROP
        return self.SYSTEM_PROMPT_4_TI2I if instruction else self.SYSTEM_PROMPT_4_I2I

    def _build_prompt_messages(self, instruction: str, input_images: Optional[List[Image.Image]] = None):
        input_images = input_images or []
        system_prompt = self._system_prompt(bool(input_images), instruction)
        content = [{"type": "image", "image": image} for image in input_images]
        content.append({"type": "text", "text": instruction})
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": content},
        ]

    def _prompt_context_images(self, prompts: list) -> Optional[List[List[Image.Image]]]:
        if not self.requires_text_embed_image_context():
            return None
        contexts = getattr(self, "_current_prompt_contexts", None)
        if not contexts or len(contexts) != len(prompts):
            raise ValueError("Boogu edit text encoding requires prompt image context for each caption.")
        images = []
        for context in contexts:
            extracted = self._extract_prompt_image_from_context(context)
            images.append(extracted if isinstance(extracted, list) else [extracted])
        return images

    def _extract_prompt_image_from_context(self, context: dict):
        tensor = context.get("conditioning_pixel_values") if isinstance(context, dict) else None
        if torch.is_tensor(tensor):
            if tensor.dim() == 4:
                return [self._tensor_to_pil(tensor[idx]) for idx in range(tensor.shape[0])]
            return self._tensor_to_pil(tensor)
        image_path = context.get("image_path") if isinstance(context, dict) else None
        data_backend_id = context.get("data_backend_id") if isinstance(context, dict) else None
        if not image_path or not data_backend_id:
            raise ValueError("Boogu edit prompt image context must include conditioning pixels or image backend metadata.")
        from simpletuner.helpers.training.state_tracker import StateTracker

        backend_entry = StateTracker.get_data_backend(data_backend_id)
        data_backend = backend_entry.get("data_backend") if backend_entry else None
        if data_backend is None:
            raise ValueError(f"Unable to resolve data backend '{data_backend_id}' for Boogu edit prompt context.")
        image = data_backend.read_image(image_path)
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        if torch.is_tensor(image):
            return self._tensor_to_pil(image)
        raise ValueError(f"Unsupported Boogu edit prompt image type: {type(image)}.")

    def _tensor_to_pil(self, tensor: torch.Tensor):
        tensor = tensor.detach().float().cpu()
        if tensor.dim() == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.dim() != 3:
            raise ValueError(f"Expected image tensor with shape (C, H, W), got {tuple(tensor.shape)}.")
        if tensor.max().item() > 1.0 or tensor.min().item() < 0.0:
            tensor = (tensor + 1.0) / 2.0
        array = (tensor.clamp(0.0, 1.0).permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
        return Image.fromarray(array)

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        text_encoder = self.text_encoders[0]
        processor = self._load_processor_for_pipeline()
        prompt_images = self._prompt_context_images(prompts)
        messages = [
            self._build_prompt_messages(prompt, None if prompt_images is None else prompt_images[idx])
            for idx, prompt in enumerate(prompts)
        ]
        max_length = getattr(self.config, "tokenizer_max_length", None) or 1024
        inputs = processor.apply_chat_template(
            messages,
            padding="longest",
            max_length=max_length,
            truncation=True,
            padding_side="right",
            return_tensors="pt",
            tokenize=True,
            return_dict=True,
        )
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(self.accelerator.device)

        outputs = text_encoder(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        return {
            "prompt_embeds": hidden_states.to(dtype=self.config.weight_dtype),
            "attention_masks": inputs["attention_mask"].to(dtype=torch.bool),
        }

    def convert_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        return {
            "instruction_embeds": text_embedding["prompt_embeds"],
            "instruction_attention_mask": text_embedding["attention_masks"],
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        return {
            "negative_instruction_embeds": text_embedding["prompt_embeds"],
            "negative_instruction_attention_mask": text_embedding["attention_masks"],
        }

    def update_pipeline_call_kwargs(self, pipeline_kwargs: dict) -> dict:
        if "prompt" in pipeline_kwargs and "instruction" not in pipeline_kwargs:
            instruction = pipeline_kwargs.pop("prompt")
            if instruction is not None:
                pipeline_kwargs["instruction"] = instruction
        if "negative_prompt" in pipeline_kwargs and "negative_instruction" not in pipeline_kwargs:
            negative_instruction = pipeline_kwargs.pop("negative_prompt")
            if negative_instruction is not None:
                pipeline_kwargs["negative_instruction"] = negative_instruction
        if "num_images_per_prompt" in pipeline_kwargs and "num_images_per_instruction" not in pipeline_kwargs:
            pipeline_kwargs["num_images_per_instruction"] = pipeline_kwargs.pop("num_images_per_prompt")
        if "guidance_scale" in pipeline_kwargs and "text_guidance_scale" not in pipeline_kwargs:
            pipeline_kwargs["text_guidance_scale"] = pipeline_kwargs.pop("guidance_scale")
        pipeline_kwargs.pop("guidance_scale_real", None)
        return pipeline_kwargs

    def collate_prompt_embeds(self, text_encoder_output: list) -> dict:
        if not text_encoder_output:
            return {}
        embeds = [item["prompt_embeds"] for item in text_encoder_output]
        masks = [item["attention_masks"] for item in text_encoder_output]
        max_seq_len = max(embed.shape[-2] for embed in embeds)
        padded_embeds = []
        padded_masks = []
        for embed, mask in zip(embeds, masks):
            if embed.dim() == 2:
                embed = embed.unsqueeze(0)
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            if embed.shape[1] < max_seq_len:
                pad_len = max_seq_len - embed.shape[1]
                embed = torch.cat(
                    [embed, embed.new_zeros(embed.shape[0], pad_len, embed.shape[2])],
                    dim=1,
                )
                mask = torch.cat([mask, mask.new_zeros(mask.shape[0], pad_len)], dim=1)
            padded_embeds.append(embed)
            padded_masks.append(mask)
        return {
            "prompt_embeds": torch.cat(padded_embeds, dim=0),
            "attention_masks": torch.cat(padded_masks, dim=0),
        }

    def prepare_batch(self, batch: dict, state: dict) -> dict:
        batch = super().prepare_batch(batch, state)
        if self._is_edit_flavour():
            conditioning_latents = batch.get("conditioning_latents")
            if conditioning_latents is None:
                raise ValueError("Boogu edit training requires cached conditioning_latents.")
            batch["ref_image_hidden_states"] = self._build_ref_latent_list(conditioning_latents, batch["latents"].shape[0])
        return batch

    def _build_ref_latent_list(self, conditioning_latents, batch_size: int):
        ref_latents: List[Optional[List[torch.Tensor]]] = [[] for _ in range(batch_size)]
        if not isinstance(conditioning_latents, list):
            conditioning_latents = [conditioning_latents]
        for latent_group in conditioning_latents:
            if torch.is_tensor(latent_group):
                if latent_group.dim() == 3:
                    ref_latents[0].append(latent_group.to(self.accelerator.device, self.config.weight_dtype))
                elif latent_group.dim() == 4:
                    for idx in range(min(batch_size, latent_group.shape[0])):
                        ref_latents[idx].append(latent_group[idx].to(self.accelerator.device, self.config.weight_dtype))
            elif isinstance(latent_group, list):
                for idx, latent in enumerate(latent_group[:batch_size]):
                    if torch.is_tensor(latent):
                        ref_latents[idx].append(latent.to(self.accelerator.device, self.config.weight_dtype))
        return [items or None for items in ref_latents]

    def _freqs_cis(self):
        from simpletuner.helpers.models.boogu_image.rope import BooguImageRotaryPosEmbed

        config = self.unwrap_model(self.get_trained_component(base_model=True)).config
        return BooguImageRotaryPosEmbed.get_freqs_cis(config.axes_dim_rope, config.axes_lens, theta=10000)

    def model_predict(self, prepared_batch, custom_timesteps: list = None):
        if self._xm_noise_candidates_enabled():
            self._prepare_xm_noise_candidates(prepared_batch)
            model_output = self._model_predict_single(prepared_batch, custom_timesteps=custom_timesteps)
            model_output["xm_candidate_count"] = self.xm_config.candidate_count
            return model_output
        return self._model_predict_single(prepared_batch, custom_timesteps=custom_timesteps)

    def _repeat_xm_candidate_value(self, value, candidate_count: int):
        if torch.is_tensor(value):
            if value.ndim == 0:
                return value
            return self._repeat_xm_candidate_tensor(value, candidate_count)
        if isinstance(value, list):
            return [self._repeat_xm_candidate_value(item, candidate_count) for item in value]
        if isinstance(value, tuple):
            return tuple(self._repeat_xm_candidate_value(item, candidate_count) for item in value)
        return value

    @staticmethod
    def _repeat_xm_candidate_sequence(value, candidate_count: int):
        if isinstance(value, tuple):
            return tuple(item for _ in range(candidate_count) for item in value)
        return [item for _ in range(candidate_count) for item in value]

    def _prepare_xm_noise_candidates(self, prepared_batch: dict) -> dict:
        xm_config = self.xm_config
        candidate_count = xm_config.candidate_count
        latents = prepared_batch.get("latents")
        timesteps = prepared_batch.get("timesteps")
        sigmas = prepared_batch.get("sigmas")
        if not torch.is_tensor(latents) or not torch.is_tensor(timesteps) or not torch.is_tensor(sigmas):
            raise ValueError("Boogu-Image XM noise-candidate training requires latents, timesteps, and sigmas tensors.")
        if "noisy_latents" not in prepared_batch:
            raise ValueError("Boogu-Image XM noise-candidate training requires prepared noisy_latents.")
        if prepared_batch.get("target") is not None:
            raise ValueError("Boogu-Image XM noise-candidate training cannot be used with an explicit prepared target.")

        expanded_batch = dict(prepared_batch)
        batch_size = latents.shape[0]
        for key, value in list(expanded_batch.items()):
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == batch_size:
                expanded_batch[key] = self._repeat_xm_candidate_tensor(value, candidate_count)
            elif key == "conditioning_latents" and isinstance(value, (list, tuple)):
                expanded_batch[key] = self._repeat_xm_candidate_value(value, candidate_count)
            elif key == "ref_image_hidden_states" and isinstance(value, (list, tuple)) and len(value) == batch_size:
                expanded_batch[key] = self._repeat_xm_candidate_sequence(value, candidate_count)
            elif key == "metadata" and isinstance(value, (list, tuple)) and len(value) == batch_size:
                expanded_batch[key] = self._repeat_xm_candidate_sequence(value, candidate_count)

        expanded_latents = expanded_batch["latents"]
        candidate_noise = torch.randn_like(expanded_latents)
        sigma_source = expanded_batch.get("mixflow_interpolation_sigmas")
        if sigma_source is None:
            sigma_source = expanded_batch["sigmas"]
        if not torch.is_tensor(sigma_source):
            raise ValueError("Boogu-Image XM noise-candidate training requires tensor sigmas for interpolation.")
        interpolation_grid = self._expand_sigma_values(sigma_source, expanded_latents)
        expanded_batch["noise"] = candidate_noise
        expanded_batch["input_noise"] = candidate_noise
        expanded_batch["noisy_latents"] = (
            1.0 - interpolation_grid
        ) * expanded_latents + interpolation_grid * candidate_noise
        expanded_batch["flow_target"] = self.get_flow_matching_target(
            expanded_batch,
            latents=expanded_latents,
            noise=candidate_noise,
            prefer_explicit_target=False,
        ).to(device=expanded_latents.device, dtype=expanded_latents.dtype)
        expanded_batch["xm_candidate_count"] = candidate_count
        expanded_batch["xm_original_batch_size"] = batch_size

        prepared_batch.clear()
        prepared_batch.update(expanded_batch)
        return prepared_batch

    def _model_predict_single(self, prepared_batch, custom_timesteps: list = None):
        hidden_states_buffer = self._new_hidden_state_buffer()
        latents = prepared_batch["noisy_latents"].to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        timestep = prepared_batch["timesteps"].to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        if timestep.ndim == 0:
            timestep = timestep.expand(latents.shape[0])
        instruction_embeds = prepared_batch.get("encoder_hidden_states")
        if instruction_embeds is None:
            instruction_embeds = prepared_batch.get("instruction_embeds")
        instruction_mask = prepared_batch.get("encoder_attention_mask")
        if instruction_mask is None:
            instruction_mask = prepared_batch.get("instruction_attention_mask")
        if instruction_embeds is None or instruction_mask is None:
            raise ValueError("Boogu training requires cached instruction embeddings and attention masks.")

        capture_kwargs = {"hidden_states_buffer": hidden_states_buffer} if hidden_states_buffer is not None else {}
        model_pred = self.get_trained_component(base_model=True)(
            latents,
            timestep,
            instruction_embeds.to(device=self.accelerator.device, dtype=self.config.weight_dtype),
            self._freqs_cis(),
            instruction_mask.to(device=self.accelerator.device, dtype=torch.bool),
            ref_image_hidden_states=prepared_batch.get("ref_image_hidden_states"),
            **capture_kwargs,
        )
        if hasattr(model_pred, "sample"):
            model_pred = model_pred.sample
        output = {"model_prediction": model_pred}
        if hidden_states_buffer is not None:
            output["hidden_states_buffer"] = hidden_states_buffer
        return output

    def loss_with_logs(self, prepared_batch: dict, model_output, apply_conditioning_mask: bool = True):
        candidate_count = model_output.get("xm_candidate_count") if isinstance(model_output, dict) else None
        if not candidate_count:
            return super().loss_with_logs(
                prepared_batch,
                model_output,
                apply_conditioning_mask=apply_conditioning_mask,
            )
        return self._xm_noise_loss_with_logs(
            prepared_batch,
            model_output,
            candidate_count=int(candidate_count),
            apply_conditioning_mask=apply_conditioning_mask,
        )

    def _xm_diffusion_loss_tensor(
        self, prepared_batch: dict, model_output: dict, apply_conditioning_mask: bool
    ) -> torch.Tensor:
        target = self.get_prediction_target(prepared_batch)
        if target is None:
            raise ValueError("Target is None. Cannot compute Boogu-Image XM loss.")
        model_pred = model_output["model_prediction"]
        loss_type = getattr(self.config, "loss_type", "l2")
        if loss_type in ["huber", "smooth_l1"]:
            timesteps = prepared_batch["timesteps"]
            if getattr(self.config, "huber_schedule", "constant") != "constant":
                losses = []
                for idx in range(model_pred.shape[0]):
                    huber_c = self.compute_scheduled_huber_c(timesteps[idx : idx + 1]).item()
                    losses.append(
                        self.conditional_loss(
                            model_pred[idx : idx + 1].float(),
                            target[idx : idx + 1].float(),
                            reduction="none",
                            loss_type=loss_type,
                            huber_c=huber_c,
                        )
                    )
                loss = torch.cat(losses, dim=0)
            else:
                loss = self.conditional_loss(
                    model_pred.float(),
                    target.float(),
                    reduction="none",
                    loss_type=loss_type,
                    huber_c=getattr(self.config, "huber_c", 0.1),
                )
        elif loss_type == "l2":
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        else:
            raise ValueError(f"Boogu-Image XM noise-candidate training does not support loss_type={loss_type!r}.")

        loss_mask_type = prepared_batch.get("loss_mask_type")
        if not loss_mask_type:
            legacy_type = prepared_batch.get("conditioning_type")
            if legacy_type in ("mask", "segmentation"):
                loss_mask_type = legacy_type
        if loss_mask_type == "mask" and apply_conditioning_mask:
            mask_image = (
                prepared_batch["conditioning_pixel_values"].to(dtype=loss.dtype, device=loss.device)[:, 0].unsqueeze(1)
            )
            mask_image = torch.nn.functional.interpolate(mask_image, size=loss.shape[2:], mode="area")
            mask_image = mask_image / 2 + 0.5
            loss = loss * mask_image
        elif loss_mask_type == "segmentation" and apply_conditioning_mask:
            raise ValueError("Boogu-Image XM noise-candidate training does not support stochastic segmentation masked loss.")
        return loss

    def post_model_load_setup(self):
        super().post_model_load_setup()
        self._validate_xm_support()
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

        loaded = load_assistant_adapter(
            transformer=self.unwrap_model(model=self.model),
            pipeline_cls=BooguImageTurboPipeline,
            lora_path=assistant_path,
            adapter_name=self.assistant_adapter_name,
            low_cpu_mem_usage=getattr(self.config, "low_cpu_mem_usage", False),
            weight_name=getattr(self.config, "assistant_lora_weight_name", None) or self.ASSISTANT_LORA_WEIGHT_NAME,
        )
        self.assistant_lora_loaded = loaded


ModelRegistry.register("boogu_image", BooguImage)
