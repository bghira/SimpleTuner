import logging
import os
from typing import List, Optional

import numpy as np
import torch
from diffusers import AutoencoderKL
from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from simpletuner.helpers.models.common import (
    ImageModelFoundation,
    ModelTypes,
    PipelineTypes,
    PredictionTypes,
    TextEmbedCacheKey,
)
from simpletuner.helpers.models.longcat_image import pack_latents, prepare_pos_ids, unpack_latents
from simpletuner.helpers.models.longcat_image.pipeline import LongCatImagePipeline
from simpletuner.helpers.models.longcat_image.pipeline_edit import LongCatImageEditPipeline
from simpletuner.helpers.models.longcat_image.transformer import LongCatImageTransformer2DModel
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.models.tae.types import ImageTAESpec
from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class LongCatImage(ImageModelFoundation):
    SUPPORTS_MUON_CLIP = True
    NAME = "LongCat-Image"
    MODEL_DESCRIPTION = "Bilingual 6B image generation and editing"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 16
    VALIDATION_PREVIEW_SPEC = ImageTAESpec(repo_id="madebyollin/taef1")

    MODEL_CLASS = LongCatImageTransformer2DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: LongCatImagePipeline,
        PipelineTypes.IMG2IMG: LongCatImageEditPipeline,
    }

    DEFAULT_MODEL_FLAVOUR = "final"
    HUGGINGFACE_PATHS = {
        "final": "meituan-longcat/LongCat-Image",
        "dev": "meituan-longcat/LongCat-Image-Dev",
        "edit": "meituan-longcat/LongCat-Image-Edit",
    }
    MODEL_LICENSE = "apache-2.0"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Qwen-2.5 VL",
            "tokenizer": Qwen2Tokenizer,
            "tokenizer_subfolder": "tokenizer",
            "model": Qwen2_5_VLForConditionalGeneration,
            "subfolder": "text_encoder",
        },
    }

    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0", "to_qkv"]
    EDIT_FLAVOURS = frozenset({"edit"})

    def __init__(self, config: dict, accelerator):
        super().__init__(config, accelerator)
        self.vae_scale_factor = 8
        pipeline_classes = dict(self.PIPELINE_CLASSES)
        if self._is_edit_flavour():
            pipeline_classes[PipelineTypes.TEXT2IMG] = LongCatImageEditPipeline
        self.PIPELINE_CLASSES = pipeline_classes

    def _get_model_flavour(self) -> Optional[str]:
        return getattr(self.config, "model_flavour", None)

    def _is_edit_flavour(self) -> bool:
        flavour = self._get_model_flavour()
        return flavour in self.EDIT_FLAVOURS if flavour is not None else False

    def requires_conditioning_latents(self) -> bool:
        return self._is_edit_flavour()

    def requires_conditioning_validation_inputs(self) -> bool:
        return self._is_edit_flavour()

    def requires_validation_edit_captions(self) -> bool:
        return self._is_edit_flavour()

    def requires_text_embed_image_context(self) -> bool:
        return self._is_edit_flavour()

    def conditioning_validation_dataset_type(self) -> bool:
        if self._is_edit_flavour():
            return "image"
        return super().conditioning_validation_dataset_type()

    def text_embed_cache_key(self) -> TextEmbedCacheKey:
        if self._is_edit_flavour():
            return TextEmbedCacheKey.DATASET_AND_FILENAME
        return super().text_embed_cache_key()

    def _create_dummy_image(self):
        from PIL import Image

        return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

    def encode_dropout_caption(self, positive_prompt_embeds: dict = None):
        if not self._is_edit_flavour():
            return super().encode_dropout_caption(positive_prompt_embeds)

        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        dummy_image = self._create_dummy_image()
        prompt_embeds, text_ids = pipeline.encode_prompt(
            dummy_image,
            [""],
            device=self.accelerator.device,
            dtype=torch.float64,
        )

        return {
            "prompt_embeds": prompt_embeds[0],
            "text_ids": text_ids,
        }

    def encode_validation_negative_prompt(self, negative_prompt: str, positive_prompt_embeds: dict = None):
        if not self._is_edit_flavour():
            return super().encode_validation_negative_prompt(negative_prompt, positive_prompt_embeds)

        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        dummy_image = self._create_dummy_image()
        prompt_embeds, text_ids = pipeline.encode_prompt(
            dummy_image,
            [negative_prompt],
            device=self.accelerator.device,
            dtype=torch.float64,
        )

        return {
            "prompt_embeds": prompt_embeds[0],
            "text_ids": text_ids,
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        prompt_image_tensor = None
        if self._is_edit_flavour():
            prompt_contexts = getattr(self, "_current_prompt_contexts", None)
            if not prompt_contexts or len(prompt_contexts) != len(prompts):
                raise ValueError(
                    "LongCat edit prompt encoding requires image context for each caption, but none was provided."
                )
            prompt_image_tensor = self._prepare_prompt_image_batch(prompt_contexts, len(prompts), pipeline)
            if prompt_image_tensor is None:
                raise ValueError("Failed to resolve prompt image tensors for LongCat edit text encoding.")
            prompt_embeds, text_ids = pipeline.encode_prompt(
                prompt_image_tensor,
                prompts,
                device=self.accelerator.device,
                dtype=torch.float64,
            )
        else:
            prompt_embeds, text_ids = pipeline.encode_prompt(prompts, device=self.accelerator.device, dtype=torch.float64)

        return prompt_embeds, text_ids

    def _prepare_prompt_image_batch(self, prompt_contexts, batch_size: int, pipeline):
        if not prompt_contexts or len(prompt_contexts) != batch_size:
            return None
        image_tensors = []
        for idx, context in enumerate(prompt_contexts):
            tensor = self._extract_prompt_image_from_context(context)
            if tensor is None:
                logger.warning(f"Failed to extract image tensor from context {idx}: {context}")
                return None
            if tensor.dim() == 4 and tensor.size(0) == 1:
                tensor = tensor.squeeze(0)
            if tensor.dim() != 3:
                raise ValueError(f"Expected conditioning tensor with shape (C, H, W); received {tensor.shape}.")
            image_tensors.append(tensor)
        pil_images = [self._tensor_to_pil(tensor) for tensor in image_tensors]
        return pil_images

    def _extract_prompt_image_from_context(self, context: dict):
        if not isinstance(context, dict):
            return None
        embed = context.get("conditioning_image_embeds")
        tensor = None
        if isinstance(embed, dict):
            tensor = self._coerce_prompt_tensor(embed.get("pixel_values"))
        elif torch.is_tensor(embed):
            tensor = self._coerce_prompt_tensor(embed)
        if tensor is not None:
            return tensor

        tensor = context.get("conditioning_pixel_values")
        tensor = self._coerce_prompt_tensor(tensor)
        if tensor is not None:
            return tensor
        return self._load_prompt_image_from_backend(context)

    def _tensor_to_pil(self, tensor: torch.Tensor | np.ndarray):
        from PIL import Image

        if isinstance(tensor, np.ndarray):
            array = tensor
            if array.ndim == 3 and array.shape[2] in (1, 3):
                if array.dtype != np.uint8:
                    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
                if array.shape[2] == 1:
                    array = np.repeat(array, 3, axis=2)
                return Image.fromarray(array)
            raise ValueError(f"Unsupported numpy image shape: {array.shape}")
        if not torch.is_tensor(tensor):
            raise ValueError(f"Unsupported prompt image type: {type(tensor)}")

        converted = tensor.detach().float().cpu()
        if converted.dim() == 4 and converted.size(0) == 1:
            converted = converted.squeeze(0)
        if converted.dim() != 3:
            raise ValueError(f"Expected conditioning tensor with shape (C, H, W); received {tuple(converted.shape)}.")
        if converted.max().item() > 1.0 or converted.min().item() < 0.0:
            converted = (converted + 1.0) / 2.0
        converted = converted.clamp_(0.0, 1.0)
        array = (converted.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        return Image.fromarray(array)

    def _load_prompt_image_from_backend(self, context: dict):
        if not self._is_edit_flavour():
            return None
        image_path = context.get("image_path")
        data_backend_id = context.get("data_backend_id")
        if not image_path or not data_backend_id:
            return None
        from simpletuner.helpers.training.state_tracker import StateTracker

        backend_entry = StateTracker.get_data_backend(data_backend_id)
        if backend_entry is None:
            return None
        data_backend = backend_entry.get("data_backend")
        if data_backend is None:
            return None
        image = data_backend.read_image(image_path)
        tensor = self._convert_image_to_tensor(image)
        if tensor is None:
            return None
        return tensor.to(device=self.accelerator.device, dtype=self.config.weight_dtype)

    def _coerce_prompt_tensor(self, tensor):
        if tensor is None:
            return None
        if isinstance(tensor, torch.Tensor):
            coerced = tensor
        elif isinstance(tensor, np.ndarray):
            coerced = torch.from_numpy(tensor)
        else:
            return None
        if coerced.dim() == 4 and coerced.size(0) == 1:
            coerced = coerced.squeeze(0)
        if coerced.dim() != 3:
            return None
        return coerced.to(device=self.accelerator.device, dtype=self.config.weight_dtype)

    def _convert_image_to_tensor(self, image):
        from PIL import Image

        if isinstance(image, torch.Tensor):
            tensor = image.clone().detach()
        elif isinstance(image, Image.Image):
            array = np.array(image.convert("RGB"), copy=True)
            tensor = torch.from_numpy(array)
        elif isinstance(image, np.ndarray):
            array = image
            if array.ndim == 4:
                array = array[0]
            if array.ndim == 3 and array.shape[2] == 4:
                array = array[:, :, :3]
            tensor = torch.from_numpy(array)
        else:
            return None
        if tensor.dim() == 3 and tensor.shape[0] not in (1, 3):
            tensor = tensor.permute(2, 0, 1)
        elif tensor.dim() == 4 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        tensor = tensor.to(dtype=torch.float32)
        if tensor.max() > 1.0 or tensor.min() < 0.0:
            tensor = tensor / 255.0
        return tensor.clamp_(0.0, 1.0)

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        prompt_embeds, text_ids = text_embedding
        return {
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        text_ids = text_embedding.get("text_ids")

        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if text_ids is not None and text_ids.dim() == 3:
            text_ids = text_ids[0]

        return {
            "prompt_embeds": prompt_embeds,
            "text_ids": text_ids,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        if self.config.validation_guidance is None or self.config.validation_guidance <= 1.0:
            return {}
        prompt_embeds = text_embedding["prompt_embeds"]
        text_ids = text_embedding.get("text_ids")
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if text_ids is not None and text_ids.dim() == 3:
            text_ids = text_ids[0]
        return {
            "negative_prompt_embeds": prompt_embeds,
            "negative_text_ids": text_ids,
        }

    def model_predict(self, prepared_batch):
        prompt_embeds = prepared_batch["prompt_embeds"].to(
            device=self.accelerator.device,
            dtype=self.config.base_weight_dtype if hasattr(self.config, "base_weight_dtype") else self.config.weight_dtype,
        )
        prompt_length = prompt_embeds.shape[1]
        timesteps = (
            torch.tensor(prepared_batch["timesteps"])
            .expand(prepared_batch["noisy_latents"].shape[0])
            .to(device=self.accelerator.device, dtype=torch.float32)
            / self.noise_schedule.config.num_train_timesteps
        )
        dtype_pos = torch.float64 if self.accelerator.device.type != "mps" else torch.float32
        text_ids = prepare_pos_ids(
            modality_id=0,
            type="text",
            start=(0, 0),
            num_token=prompt_length,
        ).to(self.accelerator.device, dtype=dtype_pos)
        img_ids = prepare_pos_ids(
            modality_id=1,
            type="image",
            start=(prompt_length, prompt_length),
            height=prepared_batch["latents"].shape[2] // 2,
            width=prepared_batch["latents"].shape[3] // 2,
        ).to(self.accelerator.device, dtype=dtype_pos)

        packed_noisy_latents = pack_latents(
            prepared_batch["noisy_latents"],
            batch_size=prepared_batch["latents"].shape[0],
            num_channels_latents=prepared_batch["latents"].shape[1],
            height=prepared_batch["latents"].shape[2],
            width=prepared_batch["latents"].shape[3],
        ).to(dtype=self.config.base_weight_dtype if hasattr(self.config, "base_weight_dtype") else self.config.weight_dtype)

        hidden_states = packed_noisy_latents
        img_ids_input = img_ids
        if self._is_edit_flavour():
            conditioning = prepared_batch.get("conditioning_latents")
            if conditioning is None:
                raise ValueError("Edit flavour requires conditioning_latents in the prepared batch.")
            packed_ref_latents = pack_latents(
                conditioning,
                batch_size=conditioning.shape[0],
                num_channels_latents=conditioning.shape[1],
                height=conditioning.shape[2],
                width=conditioning.shape[3],
            ).to(
                dtype=(
                    self.config.base_weight_dtype if hasattr(self.config, "base_weight_dtype") else self.config.weight_dtype
                )
            )
            hidden_states = torch.cat([packed_noisy_latents, packed_ref_latents], dim=1)
            ref_ids = prepare_pos_ids(
                modality_id=2,
                type="image",
                start=(prompt_length, prompt_length),
                height=conditioning.shape[2] // 2,
                width=conditioning.shape[3] // 2,
            ).to(self.accelerator.device, dtype=dtype_pos)
            img_ids_input = torch.cat([img_ids, ref_ids], dim=0)

        model_pred = self.model(
            hidden_states=hidden_states,
            timestep=timesteps,
            timestep_sign=(
                prepared_batch.get("twinflow_time_sign") if getattr(self.config, "twinflow_enabled", False) else None
            ),
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids_input,
            return_dict=False,
        )[0]

        if self._is_edit_flavour():
            scene_seq_len = packed_noisy_latents.shape[1]
            model_pred = model_pred[:, :scene_seq_len, :]

        return {
            "model_prediction": unpack_latents(
                model_pred,
                height=prepared_batch["latents"].shape[2] * 8,
                width=prepared_batch["latents"].shape[3] * 8,
                vae_scale_factor=16,
            )
        }


ModelRegistry.register("longcat_image", LongCatImage)
