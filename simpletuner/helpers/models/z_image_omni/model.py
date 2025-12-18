import logging
from typing import List, Optional

import torch
from diffusers import AutoencoderKL
from transformers import AutoModelForCausalLM, AutoTokenizer, Siglip2ImageProcessorFast, Siglip2VisionModel

from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.models.tae.types import ImageTAESpec
from simpletuner.helpers.models.z_image_omni.pipeline import ZImageOmniPipeline
from simpletuner.helpers.models.z_image_omni.transformer import ZImageOmniTransformer2DModel

logger = logging.getLogger(__name__)


class ZImageOmni(ImageModelFoundation):
    SUPPORTS_MUON_CLIP = True
    NAME = "Z-Image Omni"
    MODEL_DESCRIPTION = "Z-Image Omni flow-matching transformer"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 16
    VALIDATION_PREVIEW_SPEC = ImageTAESpec(repo_id="madebyollin/taef1")

    MODEL_CLASS = ZImageOmniTransformer2DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: ZImageOmniPipeline,
    }

    HUGGINGFACE_PATHS: dict = {
        "base": "TONGYI-Lab/Z-Image-Base",
        "edit": "TONGYI-Lab/Z-Image-Edit",
    }
    DEFAULT_MODEL_FLAVOUR = "base"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Qwen3 4B",
            "tokenizer": AutoTokenizer,
            "tokenizer_subfolder": "tokenizer",
            "model": AutoModelForCausalLM,
            "subfolder": "text_encoder",
        },
    }

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        self._conditioning_image_embedder = None

    def pretrained_load_args(self, pretrained_load_args: dict) -> dict:
        args = super().pretrained_load_args(pretrained_load_args)
        if "low_cpu_mem_usage" not in args:
            args["low_cpu_mem_usage"] = bool(getattr(self.config, "low_cpu_mem_usage", False))
        return args

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        text_encoder = self.text_encoders[0]
        tokenizer = self.tokenizers[0]

        processed: List[str] = []
        for prompt in prompts:
            processed.append(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")

        max_length = getattr(self.config, "tokenizer_max_length", None) or 512
        tokenized = tokenizer(
            processed,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokenized.input_ids.to(self.accelerator.device)
        attention_mask = tokenized.attention_mask.to(self.accelerator.device).bool()
        outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-2]
        return {
            "prompt_embeds": hidden_states,
            "attention_mask": attention_mask,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        attention_mask = text_embedding["attention_mask"]

        prompt_list: List[List[torch.Tensor]] = []
        for embeds, mask in zip(prompt_embeds, attention_mask):
            flat_mask = mask.view(-1).bool()
            prompt_list.append([embeds[flat_mask]])

        return {
            "prompt_embeds": prompt_list,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        attention_mask = text_embedding["attention_mask"]

        prompt_list: List[List[torch.Tensor]] = []
        for embeds, mask in zip(prompt_embeds, attention_mask):
            flat_mask = mask.view(-1).bool()
            prompt_list.append([embeds[flat_mask]])

        return {
            "negative_prompt_embeds": prompt_list,
        }

    def requires_conditioning_image_embeds(self) -> bool:
        """
        SigLIP conditioning is optional; enable when a conditioning dataset is configured.
        The conditioning dataset images (not the primary/edited images) are encoded via the provider below.
        """
        try:
            from simpletuner.helpers.training.state_tracker import StateTracker

            if StateTracker.get_conditioning_mappings():
                return True
        except Exception:
            # Fallback to config-based detection when StateTracker is not yet initialised.
            pass

        cond_data = getattr(self.config, "conditioning_data", None)
        if isinstance(cond_data, (list, tuple, set)):
            return len(cond_data) > 0
        return cond_data is not None

    def conditioning_image_embeds_use_reference_dataset(self) -> bool:
        # Condition on the reference/conditioning dataset, not the primary image.
        return True

    class _SiglipConditioningImageEmbedder:
        def __init__(self, siglip: Siglip2VisionModel, processor: Siglip2ImageProcessorFast, device, dtype):
            self.siglip = siglip
            self.processor = processor
            self.device = device
            self.dtype = dtype
            self.siglip.eval()
            if self.dtype is not None:
                self.siglip.to(device=self.device, dtype=self.dtype)
            else:
                self.siglip.to(device=self.device)
            for param in self.siglip.parameters():
                param.requires_grad_(False)

        @torch.no_grad()
        def encode(self, images, **_: dict):
            inputs = self.processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            outputs = self.siglip(**inputs)
            hidden = outputs.last_hidden_state
            spatial_shapes = inputs.get("spatial_shapes", None)
            if spatial_shapes is None:
                raise ValueError("SigLIP processor did not return spatial_shapes needed to reshape embeddings.")
            h, w = int(spatial_shapes[0][0]), int(spatial_shapes[0][1])
            hidden = hidden[:, : h * w]
            hidden = hidden.view(hidden.shape[0], h, w, hidden.shape[-1])
            if self.dtype is not None:
                hidden = hidden.to(self.dtype)
            return [hidden[i] for i in range(hidden.shape[0])]

    def _load_siglip_components(self):
        pipeline = self.pipelines.get(PipelineTypes.TEXT2IMG)
        siglip = getattr(pipeline, "siglip", None) if pipeline is not None else None
        siglip_processor = getattr(pipeline, "siglip_processor", None) if pipeline is not None else None

        if siglip is not None and siglip_processor is not None:
            return siglip, siglip_processor

        repo_id = getattr(self.config, "siglip_pretrained_model_name_or_path", None) or getattr(
            self.config, "pretrained_model_name_or_path", None
        )
        if repo_id is None:
            raise ValueError(
                "SigLIP conditioning embeds are required, but no siglip_pretrained_model_name_or_path was provided."
            )

        siglip_subfolder = getattr(self.config, "siglip_subfolder", None) or "siglip"
        siglip_processor_subfolder = getattr(self.config, "siglip_processor_subfolder", None) or "siglip_processor"
        siglip_revision = getattr(self.config, "siglip_revision", getattr(self.config, "revision", None))
        siglip_processor_revision = getattr(self.config, "siglip_processor_revision", getattr(self.config, "revision", None))

        siglip = Siglip2VisionModel.from_pretrained(
            repo_id,
            subfolder=siglip_subfolder,
            revision=siglip_revision,
        )
        siglip_processor = Siglip2ImageProcessorFast.from_pretrained(
            repo_id,
            subfolder=siglip_processor_subfolder,
            revision=siglip_processor_revision,
        )
        return siglip, siglip_processor

    def get_conditioning_image_embedder(self):
        if self._conditioning_image_embedder is not None:
            return self._conditioning_image_embedder

        siglip, siglip_processor = self._load_siglip_components()

        device = getattr(self.config, "conditioning_image_embed_device", self.accelerator.device)
        if isinstance(device, str):
            device = torch.device(device)
        dtype = getattr(self.config, "weight_dtype", None)
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype, None)

        self._conditioning_image_embedder = self._SiglipConditioningImageEmbedder(
            siglip=siglip,
            processor=siglip_processor,
            device=device,
            dtype=dtype,
        )
        return self._conditioning_image_embedder

    def model_predict(self, prepared_batch, custom_timesteps: Optional[list] = None):
        latents = prepared_batch["noisy_latents"]
        if latents.dim() == 4:
            latents = latents.unsqueeze(2)
        elif latents.dim() != 5:
            raise ValueError(f"Unexpected latent rank {latents.dim()} for Z-Image Omni.")

        batch_size = latents.shape[0]
        prompt_embeds = prepared_batch["encoder_hidden_states"]
        attention_mask = prepared_batch.get("encoder_attention_mask")
        if attention_mask is None:
            raise ValueError("encoder_attention_mask is required for Z-Image Omni training.")

        prompt_list: List[List[torch.Tensor]] = []
        for idx in range(batch_size):
            mask = attention_mask[idx].view(-1).bool()
            prompt_list.append([prompt_embeds[idx][mask].to(device=self.accelerator.device, dtype=self.config.weight_dtype)])

        latents = latents.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
        latent_list = [sample for sample in latents]

        cond_latents_input = prepared_batch.get("conditioning_latents")
        cond_latent_list: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]
        if cond_latents_input is not None:
            if isinstance(cond_latents_input, torch.Tensor):
                cond_latents_tensor = cond_latents_input
                if cond_latents_tensor.dim() == 4:
                    cond_latents_tensor = cond_latents_tensor.unsqueeze(2)
                cond_latents_tensor = cond_latents_tensor.to(device=self.accelerator.device, dtype=self.config.weight_dtype)
                cond_latent_list = [[sample] for sample in cond_latents_tensor]
            elif isinstance(cond_latents_input, list):
                cond_latent_list = []
                for sample in cond_latents_input:
                    if sample is None:
                        cond_latent_list.append([])
                        continue
                    if not torch.is_tensor(sample):
                        raise ValueError("conditioning_latents items must be tensors or None.")
                    if sample.dim() == 4:
                        sample = sample.unsqueeze(2)
                    cond_latent_list.append([sample.to(device=self.accelerator.device, dtype=self.config.weight_dtype)])
            if len(cond_latent_list) != batch_size:
                raise ValueError("conditioning_latents length must match batch size.")

        siglip_feats: List[Optional[List[torch.Tensor]]] = [None for _ in range(batch_size)]
        siglip_input = prepared_batch.get("siglip_embeds") or prepared_batch.get("conditioning_image_embeds")
        if siglip_input is not None:
            siglip_feats = []
            if isinstance(siglip_input, torch.Tensor):
                for sample in siglip_input:
                    siglip_feats.append([sample.to(device=self.accelerator.device, dtype=self.config.weight_dtype)])
            elif isinstance(siglip_input, list):
                for sample in siglip_input:
                    if sample is None:
                        siglip_feats.append(None)
                    elif torch.is_tensor(sample):
                        siglip_feats.append([sample.to(device=self.accelerator.device, dtype=self.config.weight_dtype)])
                    elif isinstance(sample, list):
                        siglip_feats.append(
                            [s.to(device=self.accelerator.device, dtype=self.config.weight_dtype) for s in sample]
                        )
                    else:
                        raise ValueError("siglip_embeds items must be tensors, lists of tensors or None.")
            else:
                raise ValueError("siglip_embeds must be a tensor or list when provided.")
            if len(siglip_feats) != batch_size:
                raise ValueError("siglip_embeds length must match batch size.")
            cond_counts = [len(c) for c in cond_latent_list]
            for idx, sels in enumerate(siglip_feats):
                if cond_counts[idx] == 0:
                    siglip_feats[idx] = None
                    continue
                if sels is None:
                    raise ValueError(f"Missing SigLIP embeds for batch item {idx} with conditioning latents present.")
                if len(sels) < cond_counts[idx]:
                    raise ValueError(
                        f"SigLIP embeds count ({len(sels)}) is less than conditioning latents ({cond_counts[idx]}) "
                        f"for batch item {idx}."
                    )
                if len(sels) == cond_counts[idx]:
                    siglip_feats[idx] = sels + [None]
                else:
                    siglip_feats[idx] = sels

        timesteps = prepared_batch["timesteps"]
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, device=self.accelerator.device, dtype=torch.float32)
        else:
            timesteps = timesteps.to(device=self.accelerator.device, dtype=torch.float32)
        normalized_t = (1000.0 - timesteps) / 1000.0

        model_out_list = self.model(
            latent_list,
            normalized_t,
            prompt_list,
            cond_latent_list,
            siglip_feats,
            return_dict=False,
        )[0]

        noise_pred = torch.stack([out.float() for out in model_out_list], dim=0)
        if noise_pred.dim() == 5 and noise_pred.shape[2] == 1:
            noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred

        return {"model_prediction": noise_pred}


ModelRegistry.register("z_image_omni", ZImageOmni)
ModelRegistry.register("z-image-omni", ZImageOmni)
