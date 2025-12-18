import logging
import os
from typing import Dict, List, Tuple

import torch
from diffusers import AutoencoderKL
from torch.nn import functional as F
from transformers import AutoTokenizer, T5EncoderModel

from simpletuner.helpers.configuration.registry import (
    ConfigRegistry,
    ConfigRule,
    RuleType,
    ValidationResult,
    make_override_rule,
)
from simpletuner.helpers.models.chroma import pack_latents, prepare_latent_image_ids, unpack_latents
from simpletuner.helpers.models.chroma.controlnet import ChromaControlNetModel
from simpletuner.helpers.models.chroma.pipeline import ChromaPipeline
from simpletuner.helpers.models.chroma.pipeline_controlnet import ChromaControlNetPipeline
from simpletuner.helpers.models.chroma.transformer import ChromaTransformer2DModel
from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.models.tae.types import ImageTAESpec
from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger(__name__)

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class Chroma(ImageModelFoundation):
    SUPPORTS_MUON_CLIP = True
    NAME = "Chroma 1"
    MODEL_DESCRIPTION = "Flow-matching image transformer from Lodestone Labs"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 16
    VALIDATION_PREVIEW_SPEC = ImageTAESpec(repo_id="madebyollin/taef1")
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0", "to_qkv"]
    SLIDER_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0", "to_qkv"]
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = ChromaTransformer2DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: ChromaPipeline,
        PipelineTypes.CONTROLNET: ChromaControlNetPipeline,
    }

    DEFAULT_MODEL_FLAVOUR = "base"
    HUGGINGFACE_PATHS = {
        "base": "lodestones/Chroma1-Base",
        "hd": "lodestones/Chroma1-HD",
        "flash": "lodestones/Chroma1-Flash",
    }
    MODEL_LICENSE = "apache-2.0"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "T5 XXL v1.1",
            "tokenizer": AutoTokenizer,
            "tokenizer_subfolder": "tokenizer",
            "model": T5EncoderModel,
        },
    }

    def requires_conditioning_latents(self) -> bool:
        if self.config.controlnet or self.config.control:
            return True
        return False

    def tread_init(self):
        """
        Initialize the TREAD router for Chroma training.
        """
        from simpletuner.helpers.training.tread import TREADRouter

        if (
            getattr(self.config, "tread_config", None) is None
            or getattr(self.config, "tread_config", None) is {}
            or getattr(self.config, "tread_config", {}).get("routes", None) is None
        ):
            logger.error("TREAD training requires you to configure the routes in the TREAD config")
            import sys

            sys.exit(1)

        self.unwrap_model(model=self.model).set_router(
            TREADRouter(
                seed=getattr(self.config, "seed", None) or 42,
                device=self.accelerator.device,
            ),
            self.config.tread_config["routes"],
        )

        logger.info("TREAD training is enabled")

    def controlnet_init(self):
        """
        Initialize the Chroma ControlNet module.
        """
        logger.info("Creating the Chroma ControlNet...")

        if getattr(self.config, "controlnet_model_name_or_path", None):
            logger.info("Loading ControlNet weights from %s", self.config.controlnet_model_name_or_path)
            self.controlnet = ChromaControlNetModel.from_pretrained(
                self.config.controlnet_model_name_or_path,
                torch_dtype=self.config.weight_dtype,
            )
        else:
            logger.info("Initializing ControlNet weights from base transformer.")
            base_transformer = self.unwrap_model(self.model)
            if base_transformer is None:
                raise ValueError("Base transformer must be initialized before creating the ControlNet.")
            self.controlnet = ChromaControlNetModel.from_transformer(base_transformer)

        self.controlnet.to(self.accelerator.device, self.config.weight_dtype)

    def requires_conditioning_validation_inputs(self) -> bool:
        if self.config.controlnet or self.config.control:
            return True
        return False

    def _encode_prompts(self, prompts: List[str], is_negative_prompt: bool = False):
        pipeline = self.pipelines.get(PipelineTypes.TEXT2IMG)
        if pipeline is None:
            pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        prompt_embeds, _, prompt_attention_mask, *_ = pipeline.encode_prompt(
            prompt=prompts,
            negative_prompt=None,
            device=self.accelerator.device,
            num_images_per_prompt=1,
            max_sequence_length=int(self.config.tokenizer_max_length or 512),
            do_classifier_free_guidance=False,
        )
        if getattr(self.config, "t5_padding", "unmodified") == "zero":
            prompt_embeds = prompt_embeds * prompt_attention_mask.to(device=prompt_embeds.device).unsqueeze(-1)
        return prompt_embeds, prompt_attention_mask

    def _format_text_embedding(self, text_embedding: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prompt_embeds, attention_mask = text_embedding
        return {
            "prompt_embeds": prompt_embeds,
            "attention_masks": attention_mask,
        }

    def collate_prompt_embeds(self, text_encoder_output: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        if not text_encoder_output:
            return {}

        prompt_embeds_list = [t["prompt_embeds"] for t in text_encoder_output]
        attention_masks_list = [t["attention_masks"] for t in text_encoder_output]

        # Handle both 2D (seq, dim) and 3D (1, seq, dim) cached embeddings
        if prompt_embeds_list[0].dim() == 3:
            prompt_embeds = torch.cat(prompt_embeds_list, dim=0)
        else:
            prompt_embeds = torch.stack(prompt_embeds_list)

        if attention_masks_list[0].dim() == 2:
            attention_masks = torch.cat(attention_masks_list, dim=0)
        else:
            attention_masks = torch.stack(attention_masks_list)

        return {
            "prompt_embeds": prompt_embeds,
            "attention_masks": attention_masks,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prompt_embeds = text_embedding["prompt_embeds"]
        attention_mask = text_embedding["attention_masks"]

        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if attention_mask.dim() == 3 and attention_mask.size(1) == 1:
            attention_mask = attention_mask.squeeze(1)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": attention_mask,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: Dict[str, torch.Tensor]) -> dict:
        neg_embeds = text_embedding["prompt_embeds"]
        neg_mask = text_embedding["attention_masks"]

        if neg_embeds.dim() == 2:
            neg_embeds = neg_embeds.unsqueeze(0)
        if neg_mask.dim() == 3 and neg_mask.size(1) == 1:
            neg_mask = neg_mask.squeeze(1)
        if neg_mask.dim() == 1:
            neg_mask = neg_mask.unsqueeze(0)

        result = {
            "negative_prompt_embeds": neg_embeds,
            "negative_prompt_attention_mask": neg_mask,
        }

        if self.config.validation_guidance_real is not None and self.config.validation_guidance_real > 1.0:
            result["guidance_scale_real"] = float(self.config.validation_guidance_real)

        return result

    def get_lora_target_layers(self):
        if getattr(self.config, "slider_lora_target", False) and self.config.lora_type.lower() == "standard":
            return getattr(self, "SLIDER_LORA_TARGET", None) or self.DEFAULT_SLIDER_LORA_TARGET
        if self.config.lora_type.lower() == "standard":
            if getattr(self.config, "flux_lora_target", None) is None:
                return self.DEFAULT_LORA_TARGET
            target = self.config.flux_lora_target
            if target == "all":
                return [
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_qkv",
                    "add_qkv_proj",
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "to_out.0",
                    "to_add_out",
                ]
            if target == "context":
                return [
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "add_qkv_proj",
                    "to_add_out",
                ]
            if target == "context+ffs":
                return [
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "add_qkv_proj",
                    "to_add_out",
                    "ff_context.net.0.proj",
                    "ff_context.net.2",
                ]
            if target == "all+ffs":
                return [
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_qkv",
                    "add_qkv_proj",
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "to_out.0",
                    "to_add_out",
                    "ff.net.0.proj",
                    "ff.net.2",
                    "ff_context.net.0.proj",
                    "ff_context.net.2",
                    "proj_mlp",
                    "proj_out",
                ]
            if target == "all+ffs+embedder":
                return [
                    "x_embedder",
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_qkv",
                    "add_qkv_proj",
                    "to_out.0",
                    "add_k_proj",
                    "add_q_proj",
                    "add_v_proj",
                    "to_add_out",
                    "ff.net.0.proj",
                    "ff.net.2",
                    "ff_context.net.0.proj",
                    "ff_context.net.2",
                    "proj_mlp",
                    "proj_out",
                ]
            if target == "ai-toolkit":
                return [
                    "to_q",
                    "to_k",
                    "to_qkv",
                    "add_qkv_proj",
                    "to_v",
                    "add_q_proj",
                    "add_k_proj",
                    "add_v_proj",
                    "to_out.0",
                    "to_add_out",
                    "ff.net.0.proj",
                    "ff.net.2",
                    "ff_context.net.0.proj",
                    "ff_context.net.2",
                    "norm.linear",
                    "norm1.linear",
                    "norm1_context.linear",
                    "proj_mlp",
                    "proj_out",
                ]
            if target == "tiny":
                return [
                    "to_q",
                    "to_k",
                    "to_v",
                    "to_out.0",
                    "ff.net.0.proj",
                ]
            if target == "tiny+embedder":
                return [
                    "x_embedder",
                    "to_q",
                    "to_k",
                    "to_v",
                    "to_out.0",
                    "ff.net.0.proj",
                ]
            logger.warning("Unknown flux_lora_target '%s', falling back to default layers.", target)
            return self.DEFAULT_LORA_TARGET
        if self.config.lora_type.lower() == "lycoris":
            return self.DEFAULT_LYCORIS_TARGET
        raise NotImplementedError(f"Unknown LoRA target type {self.config.lora_type}.")

    def model_predict(self, prepared_batch):
        batch_size, _, height, width = prepared_batch["latents"].shape
        hidden_states_buffer = self._new_hidden_state_buffer()
        packed_noisy_latents = pack_latents(
            prepared_batch["noisy_latents"],
            batch_size=batch_size,
            num_channels_latents=prepared_batch["noisy_latents"].shape[1],
            height=height,
            width=width,
        ).to(dtype=self.config.base_weight_dtype, device=self.accelerator.device)

        img_ids = prepare_latent_image_ids(
            batch_size,
            height,
            width,
            self.accelerator.device,
            self.config.weight_dtype,
        )
        if img_ids.dim() == 3:
            img_ids = img_ids[0]

        timesteps = (
            torch.tensor(prepared_batch["timesteps"], device=self.accelerator.device).expand(batch_size)
            / self.noise_schedule.config.num_train_timesteps
        )

        text_ids = torch.zeros(
            prepared_batch["prompt_embeds"].shape[1],
            3,
            device=self.accelerator.device,
            dtype=self.config.base_weight_dtype,
        )

        attention_mask = prepared_batch.get("encoder_attention_mask")
        if attention_mask is None:
            raise ValueError(
                "No attention mask was found for Chroma training. Regenerate your text embedding cache to include masks."
            )
        if attention_mask.dim() == 3 and attention_mask.size(1) == 1:
            attention_mask = attention_mask.squeeze(1)

        # Match pipeline behaviour by extending the attention mask to cover image tokens
        seq_length = packed_noisy_latents.shape[1]
        attention_mask = attention_mask.to(device=self.accelerator.device)
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask > 0
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    attention_mask.shape[0],
                    seq_length,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )

        transformer_kwargs = {
            "hidden_states": packed_noisy_latents,
            "timestep": timesteps,
            "encoder_hidden_states": prepared_batch["prompt_embeds"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            "txt_ids": text_ids,
            "img_ids": img_ids,
            "attention_mask": attention_mask,
            "joint_attention_kwargs": None,
            "timestep_sign": prepared_batch.get("twinflow_time_sign"),
            "return_dict": False,
        }
        if hidden_states_buffer is not None:
            transformer_kwargs["hidden_states_buffer"] = hidden_states_buffer

        if (
            getattr(self.config, "tread_config", None) is not None
            and self.config.tread_config is not None
            and "conditioning_pixel_values" in prepared_batch
            and prepared_batch["conditioning_pixel_values"] is not None
            and prepared_batch.get("conditioning_type") in ("mask", "segmentation")
        ):
            with torch.no_grad():
                h_tokens = prepared_batch["latents"].shape[2] // 2
                w_tokens = prepared_batch["latents"].shape[3] // 2
                mask_img = prepared_batch["conditioning_pixel_values"]
                mask_img = (mask_img.sum(1, keepdim=True) / 3 + 1) / 2
                mask_lat = F.interpolate(mask_img, size=(h_tokens, w_tokens), mode="area")
                force_keep = mask_lat.flatten(2).squeeze(1) > 0.5
                transformer_kwargs["force_keep_mask"] = force_keep

        model_pred = self.model(**transformer_kwargs)[0]

        return {
            "model_prediction": unpack_latents(
                model_pred,
                height=prepared_batch["latents"].shape[2] * 8,
                width=prepared_batch["latents"].shape[3] * 8,
                vae_scale_factor=16,
            ),
            "hidden_states_buffer": hidden_states_buffer,
        }

    def controlnet_predict(self, prepared_batch: dict) -> dict:
        conditioning_latents = prepared_batch.get("conditioning_latents")
        if conditioning_latents is None:
            raise ValueError("conditioning_latents must be provided for ControlNet training")

        batch_size, _, height, width = conditioning_latents.shape

        packed_controlnet_cond = pack_latents(
            conditioning_latents,
            batch_size=batch_size,
            num_channels_latents=conditioning_latents.shape[1],
            height=height,
            width=width,
        ).to(device=self.accelerator.device, dtype=self.config.base_weight_dtype)

        packed_noisy_latents = pack_latents(
            prepared_batch["noisy_latents"],
            batch_size=batch_size,
            num_channels_latents=prepared_batch["noisy_latents"].shape[1],
            height=height,
            width=width,
        ).to(device=self.accelerator.device, dtype=self.config.base_weight_dtype)

        img_ids = prepare_latent_image_ids(
            batch_size,
            height,
            width,
            self.accelerator.device,
            self.config.weight_dtype,
        )
        if img_ids.dim() == 3:
            img_ids = img_ids[0]

        timesteps = (
            torch.tensor(prepared_batch["timesteps"], device=self.accelerator.device).expand(batch_size)
            / self.noise_schedule.config.num_train_timesteps
        )

        text_ids = torch.zeros(
            prepared_batch["prompt_embeds"].shape[1],
            3,
            device=self.accelerator.device,
            dtype=self.config.base_weight_dtype,
        )

        attention_mask = prepared_batch.get("encoder_attention_mask")
        if attention_mask is None:
            raise ValueError(
                "No attention mask was found for Chroma ControlNet training. Regenerate your text embedding cache to include masks."
            )
        if attention_mask.dim() == 3 and attention_mask.size(1) == 1:
            attention_mask = attention_mask.squeeze(1)

        seq_length = packed_noisy_latents.shape[1]
        attention_mask = attention_mask.to(device=self.accelerator.device)
        if attention_mask.dtype != torch.bool:
            attention_mask = attention_mask > 0
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    attention_mask.shape[0],
                    seq_length,
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )

        conditioning_scale = getattr(self.config, "controlnet_conditioning_scale", 1.0)

        controlnet_block_samples, controlnet_single_block_samples = self.controlnet(
            hidden_states=packed_noisy_latents,
            controlnet_cond=packed_controlnet_cond,
            conditioning_scale=conditioning_scale,
            encoder_hidden_states=prepared_batch["prompt_embeds"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            timestep=timesteps,
            txt_ids=text_ids,
            img_ids=img_ids,
            attention_mask=attention_mask,
            joint_attention_kwargs=None,
            return_dict=False,
        )

        transformer_kwargs = {
            "hidden_states": packed_noisy_latents,
            "timestep": timesteps,
            "encoder_hidden_states": prepared_batch["prompt_embeds"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            "txt_ids": text_ids,
            "img_ids": img_ids,
            "attention_mask": attention_mask,
            "joint_attention_kwargs": None,
            "return_dict": False,
        }

        if controlnet_block_samples is not None:
            transformer_kwargs["controlnet_block_samples"] = [
                sample.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
                for sample in controlnet_block_samples
            ]

        if controlnet_single_block_samples is not None:
            transformer_kwargs["controlnet_single_block_samples"] = [
                sample.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
                for sample in controlnet_single_block_samples
            ]

        model_pred = self.get_trained_component(base_model=True)(**transformer_kwargs)[0]

        return {
            "model_prediction": unpack_latents(
                model_pred,
                height=prepared_batch["latents"].shape[2] * 8,
                width=prepared_batch["latents"].shape[3] * 8,
                vae_scale_factor=16,
            )
        }

    def prepare_batch(self, batch: dict, state: dict) -> dict:
        batch = super().prepare_batch(batch, state)
        if (
            self.PREDICTION_TYPE is PredictionTypes.FLOW_MATCHING
            and "sigmas" in batch
            and batch["sigmas"] is not None
            and not getattr(self.config, "flux_fast_schedule", False)
            and not getattr(self.config, "flow_use_beta_schedule", False)
            and not getattr(self.config, "flow_use_uniform_schedule", False)
        ):
            sigmas = batch["sigmas"]
            if sigmas.dim() > 1:
                sigmas = sigmas.view(sigmas.shape[0])
            adjusted = torch.where(
                sigmas < 0.5,
                sigmas.square(),
                1 - (1 - sigmas).square(),
            )
            batch["sigmas"] = adjusted
            batch["timesteps"] = adjusted * 1000.0
            batch["noisy_latents"] = (1 - adjusted).view(-1, 1, 1, 1) * batch["latents"] + adjusted.view(
                -1, 1, 1, 1
            ) * batch["input_noise"]
            self.expand_sigmas(batch)
        return batch

    def custom_model_card_schedule_info(self):
        output_args = []
        if getattr(self.config, "flux_fast_schedule", False):
            output_args.append("flux_fast_schedule")
        if getattr(self.config, "flow_schedule_auto_shift", False):
            output_args.append("flow_schedule_auto_shift")
        if getattr(self.config, "flow_schedule_shift", None) is not None:
            output_args.append(f"shift={self.config.flow_schedule_shift}")
        guidance_mode = getattr(self.config, "flux_guidance_mode", None)
        if guidance_mode is not None:
            output_args.append(f"guidance_mode={guidance_mode}")
        if getattr(self.config, "flux_guidance_value", None):
            output_args.append(f"guidance_value={self.config.flux_guidance_value}")
        if guidance_mode == "random-range":
            if getattr(self.config, "flux_guidance_min", None) is not None:
                output_args.append(f"guidance_min={self.config.flux_guidance_min}")
            if getattr(self.config, "flux_guidance_max", None) is not None:
                output_args.append(f"guidance_max={self.config.flux_guidance_max}")
        if getattr(self.config, "flow_use_beta_schedule", False):
            output_args.append(f"flow_beta_schedule_alpha={self.config.flow_beta_schedule_alpha}")
            output_args.append(f"flow_beta_schedule_beta={self.config.flow_beta_schedule_beta}")
        if getattr(self.config, "flux_attention_masked_training", False):
            output_args.append("attention_masked_training")
        if (
            self.config.model_type == "lora"
            and self.config.lora_type == "standard"
            and getattr(self.config, "flux_lora_target", None) is not None
        ):
            output_args.append(f"flux_lora_target={self.config.flux_lora_target}")
        return f" (extra parameters={output_args})" if output_args else " (no special parameters set)"

    @classmethod
    def register_config_requirements(cls):
        rules = [
            make_override_rule(
                field_name="aspect_bucket_alignment",
                value=64,
                message="Chroma requires aspect bucket alignment of 64px",
                example="aspect_bucket_alignment: 64",
            ),
            ConfigRule(
                field_name="tokenizer_max_length",
                rule_type=RuleType.MAX,
                value=512,
                message="Chroma supports a maximum of 512 tokens",
                example="tokenizer_max_length: 512  # Maximum supported",
                error_level="warning",
            ),
            ConfigRule(
                field_name="base_model_precision",
                rule_type=RuleType.CHOICES,
                value=["int8-quanto", "fp8-torchao", "no_change", "int4-quanto", "nf4-torchao", "fp8-torchao-compile"],
                message="Chroma supports limited precision options",
                example="base_model_precision: fp8-torchao",
                error_level="warning",
            ),
            ConfigRule(
                field_name="prediction_type",
                rule_type=RuleType.CUSTOM,
                value=None,
                message="Chroma uses flow matching and does not support custom prediction types",
                error_level="warning",
            ),
        ]

        ConfigRegistry.register_rules("chroma", rules)
        ConfigRegistry.register_validator(
            "chroma",
            cls._validate_chroma_specific,
            """Validates Chroma-specific requirements:
- Warns about unsupported prediction_type overrides
- Ensures aspect bucket alignment is correct
- Emits warnings if Control/ControlNet options are enabled""",
        )

    @staticmethod
    def _validate_chroma_specific(config: dict) -> List[ValidationResult]:
        results = []
        if config.get("prediction_type") is not None:
            results.append(
                ValidationResult(
                    passed=False,
                    field="prediction_type",
                    message="Chroma does not support custom prediction types - it uses flow matching",
                    level="warning",
                    suggestion="Remove prediction_type from your configuration",
                )
            )
        if config.get("controlnet") or config.get("control"):
            results.append(
                ValidationResult(
                    passed=False,
                    field="controlnet",
                    message="Chroma currently does not support Control or ControlNet training",
                    level="warning",
                    suggestion="Disable control/controlnet when training Chroma models",
                )
            )
        return results


Chroma.register_config_requirements()
ModelRegistry.register("chroma", Chroma)
