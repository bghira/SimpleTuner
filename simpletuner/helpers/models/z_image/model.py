import logging
from typing import List

import torch
from diffusers import AutoencoderKL
from transformers import AutoModelForCausalLM, AutoTokenizer

from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.models.tae.types import ImageTAESpec
from simpletuner.helpers.models.z_image.pipeline import ZImagePipeline
from simpletuner.helpers.models.z_image.transformer import ZImageTransformer2DModel

logger = logging.getLogger(__name__)


class ZImage(ImageModelFoundation):
    NAME = "Z-Image"
    MODEL_DESCRIPTION = "Z-Image flow-matching transformer"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 16
    VALIDATION_PREVIEW_SPEC = ImageTAESpec(repo_id="madebyollin/taef1")

    MODEL_CLASS = ZImageTransformer2DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: ZImagePipeline,
    }
    ASSISTANT_LORA_FLAVOURS = ["turbo"]
    ASSISTANT_LORA_PATH = "ostris/zimage_turbo_training_adapter"
    ASSISTANT_LORA_WEIGHT_NAME = "zimage_turbo_training_adapter_v1.safetensors"

    # We do not bundle a default HF path; users must point at a released checkpoint.
    HUGGINGFACE_PATHS: dict = {"base": "TONGYI-MAI/Z-Image-Base", "turbo": "TONGYI-MAI/Z-Image-Turbo"}
    DEFAULT_MODEL_FLAVOUR = "base"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Z-Image text encoder",
            "tokenizer": AutoTokenizer,
            "tokenizer_subfolder": "tokenizer",
            "model": AutoModelForCausalLM,
            "subfolder": "text_encoder",
        },
    }

    def pretrained_load_args(self, pretrained_load_args: dict) -> dict:
        args = super().pretrained_load_args(pretrained_load_args)
        # Default to full-weight loading to avoid meta tensors from diffusers low_cpu_mem_usage defaults.
        if "low_cpu_mem_usage" not in args:
            args["low_cpu_mem_usage"] = bool(getattr(self.config, "low_cpu_mem_usage", False))
        return args

    def tread_init(self):
        """Initialize the TREAD router when training with token routing enabled."""
        from simpletuner.helpers.training.tread import TREADRouter

        tread_cfg = getattr(self.config, "tread_config", None)
        if not isinstance(tread_cfg, dict) or tread_cfg == {} or tread_cfg.get("routes") is None:
            logger.error("TREAD training requires you to configure the routes in the TREAD config")
            import sys

            sys.exit(1)

        self.unwrap_model(model=self.model).set_router(
            TREADRouter(
                seed=getattr(self.config, "seed", None) or 42,
                device=self.accelerator.device,
            ),
            tread_cfg["routes"],
        )

        logger.info("TREAD training is enabled for Z-Image")

    def post_model_load_setup(self):
        super().post_model_load_setup()
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
            pipeline_cls=ZImagePipeline,
            lora_path=assistant_path,
            adapter_name=self.assistant_adapter_name,
            low_cpu_mem_usage=getattr(self.config, "low_cpu_mem_usage", False),
            weight_name=getattr(self.config, "assistant_lora_weight_name", None) or self.ASSISTANT_LORA_WEIGHT_NAME,
        )
        self.assistant_lora_loaded = loaded

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        text_encoder = self.text_encoders[0]
        tokenizer = self.tokenizers[0]
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Z-Image tokenizer must implement apply_chat_template.")

        processed = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            processed_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            processed.append(processed_prompt)

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

        prompt_list: List[torch.Tensor] = []
        for embeds, mask in zip(prompt_embeds, attention_mask):
            flat_mask = mask.view(-1).bool()
            prompt_list.append(embeds[flat_mask])

        return {
            "prompt_embeds": prompt_list,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: dict) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        attention_mask = text_embedding["attention_mask"]

        prompt_list: List[torch.Tensor] = []
        for embeds, mask in zip(prompt_embeds, attention_mask):
            flat_mask = mask.view(-1).bool()
            prompt_list.append(embeds[flat_mask])

        return {
            "negative_prompt_embeds": prompt_list,
        }

    def check_user_config(self):
        super().check_user_config()
        if (
            self.config.model_flavour == "turbo"
            and getattr(self.config, "model_type", "").lower() == "lora"
            and not getattr(self.config, "disable_assistant_lora", False)
        ):
            if getattr(self.config, "assistant_lora_path", None) in (None, "", "None"):
                if self.ASSISTANT_LORA_PATH:
                    self.config.assistant_lora_path = self.ASSISTANT_LORA_PATH
                    if getattr(self.config, "assistant_lora_weight_name", None) in (None, "", "None"):
                        self.config.assistant_lora_weight_name = self.ASSISTANT_LORA_WEIGHT_NAME
                else:
                    raise ValueError(
                        "Z-Image turbo training expects an assistant LoRA. Provide --assistant_lora_path pointing to the turbo assistant adapter."
                    )

    def model_predict(self, prepared_batch, custom_timesteps: list = None):
        latents = prepared_batch["noisy_latents"]
        if latents.dim() == 4:
            latents = latents.unsqueeze(2)
        elif latents.dim() != 5:
            raise ValueError(f"Unexpected latent rank {latents.dim()} for Z-Image.")

        batch_size = latents.shape[0]
        prompt_embeds = prepared_batch["encoder_hidden_states"]
        attention_mask = prepared_batch.get("encoder_attention_mask")
        if attention_mask is None:
            raise ValueError("encoder_attention_mask is required for Z-Image training.")

        prompt_list: List[torch.Tensor] = []
        for idx in range(batch_size):
            mask = attention_mask[idx].view(-1).bool()
            prompt_list.append(prompt_embeds[idx][mask].to(device=self.accelerator.device, dtype=self.config.weight_dtype))

        latent_list = [sample.to(device=self.accelerator.device, dtype=self.config.weight_dtype) for sample in latents]

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
        )[0]

        noise_pred = torch.stack([out.float() for out in model_out_list], dim=0)
        if noise_pred.dim() == 5 and noise_pred.shape[2] == 1:
            noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred

        return {"model_prediction": noise_pred}


ModelRegistry.register("z-image", ZImage)
