import torch, os, logging
from helpers.models.common import (
    ImageModelFoundation,
    PredictionTypes,
    PipelineTypes,
    ModelTypes,
)
from transformers import (
    PreTrainedTokenizerFast,
    Gemma2Model,
)
from diffusers import AutoencoderKL
from diffusers.models.attention_processor import Attention
from diffusers import Lumina2Transformer2DModel, Lumina2Pipeline

from helpers.training.multi_process import _get_rank
from helpers.training.multi_process import _get_rank

logger = logging.getLogger(__name__)
from helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class Lumina2(ImageModelFoundation):
    NAME = "Lumina-T2I"
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 16
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0", "to_qkv"]
    # Only training the Attention blocks by default.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = Lumina2Transformer2DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: Lumina2Pipeline,
        # PipelineTypes.IMG2IMG: None,  # Not implemented yet
    }

    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "2.0"
    HUGGINGFACE_PATHS = {
        "2.0": "Alpha-VLLM/Lumina-Image-2.0",
        "neta-lumina": "terminusresearch/neta-lumina-v1",
    }
    MODEL_LICENSE = "apache-2.0"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Gemma2",
            "tokenizer": PreTrainedTokenizerFast,
            "tokenizer_subfolder": "tokenizer",
            "model": Gemma2Model,
            "subfolder": "text_encoder",
        },
    }

    # Lumina2 system prompt for image generation
    SYSTEM_PROMPT = "You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts."

    def fuse_qkv_projections(self):
        """Lumina2 may support QKV fusion similar to Flux"""
        if not self.config.fuse_qkv_projections or self._qkv_projections_fused:
            return

        try:
            # Try to use fused attention if available
            from diffusers.models.attention_processor import FusedAttnProcessor2_0

            attn_processor = FusedAttnProcessor2_0()
        except:
            logger.debug("Fused attention not available for Lumina2, using default")
            return

        if self.model is not None:
            logger.debug("Fusing QKV projections in the model..")
            for module in self.model.modules():
                if isinstance(module, Attention):
                    module.fuse_projections(fuse=True)
        else:
            logger.warning("Model does not support QKV projection fusing. Skipping.")

        self.unwrap_model(model=self.model).set_attn_processor(attn_processor)
        self._qkv_projections_fused = True

    def unfuse_qkv_projections(self):
        """Unfuse QKV projections if they were fused."""
        if not self.config.fuse_qkv_projections or not self._qkv_projections_fused:
            return
        self._qkv_projections_fused = False

        if self.model is not None:
            logger.debug("Temporarily unfusing QKV projections in the model..")
            for module in self.model.modules():
                if isinstance(module, Attention):
                    module.fuse_projections(fuse=False)

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        """
        Format the stored text embedding for Lumina2.

        Args:
            text_embedding: Tuple of (prompt_embeds, prompt_attention_mask)

        Returns:
            dict: Formatted embeddings
        """
        prompt_embeds, prompt_attention_mask = text_embedding

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        """Convert text embeddings for pipeline usage"""
        # Only unsqueeze if it's missing the batch dimension
        prompt_embeds = text_embedding["prompt_embeds"]
        if prompt_embeds.dim() == 2:  # Shape: [seq_len, hidden_dim]
            prompt_embeds = prompt_embeds.unsqueeze(
                0
            )  # Shape: [1, seq_len, hidden_dim]

        attention_mask = text_embedding.get("prompt_attention_mask", None)
        if attention_mask is not None and attention_mask.dim() == 1:  # Shape: [seq_len]
            attention_mask = attention_mask.unsqueeze(0)  # Shape: [1, seq_len]

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": attention_mask.to(torch.int32),
        }

    def convert_negative_text_embed_for_pipeline(
        self, text_embedding: torch.Tensor, prompt: str
    ) -> dict:
        """Convert negative text embeddings for pipeline usage"""
        if (
            self.config.validation_guidance is None
            or self.config.validation_guidance <= 1.0
        ):
            # CFG is disabled, no negative prompts.
            return {}

        # Only unsqueeze if it's missing the batch dimension
        prompt_embeds = text_embedding["prompt_embeds"]
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)

        attention_mask = text_embedding.get("prompt_attention_mask", None)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        return {
            "negative_prompt_embeds": prompt_embeds,
            "negative_prompt_attention_mask": attention_mask.to(torch.int32),
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode prompts using Gemma2 text encoder.

        Args:
            prompts: List of prompts to encode
            is_negative_prompt: Whether these are negative prompts

        Returns:
            Tuple of (prompt_embeds, prompt_attention_mask)
        """
        # Add system prompt to prompts
        if not is_negative_prompt and self.SYSTEM_PROMPT:
            prompts = [self.SYSTEM_PROMPT + " <Prompt Start> " + p for p in prompts]

        # Use the pipeline's encode method
        prompt_embeds, prompt_attention_mask = self.pipelines[
            PipelineTypes.TEXT2IMG
        ]._get_gemma_prompt_embeds(
            prompt=prompts,
            device=self.accelerator.device,
            max_sequence_length=int(self.config.tokenizer_max_length),
        )

        return prompt_embeds, prompt_attention_mask

    def model_predict(self, prepared_batch):
        """
        Perform model prediction for Lumina2.

        Key differences from Flux:
        - Uses reverse flow (multiply by -1)
        - Different timestep normalization (1 - t/num_train_timesteps)
        - No guidance embedding in transformer
        - Lumina2 expects unpacked latents (4D tensor)
        """
        # Lumina2 expects unpacked latents, not packed ones
        noisy_latents = prepared_batch["noisy_latents"].to(
            dtype=self.config.base_weight_dtype,
            device=self.accelerator.device,
        )

        # Lumina2 uses reverse timestep normalization (1 - t/T)
        timesteps = (
            1.0
            - torch.tensor(prepared_batch["timesteps"])
            .expand(prepared_batch["noisy_latents"].shape[0])
            .to(device=self.accelerator.device)
            / self.noise_schedule.config.num_train_timesteps
        )

        # Get attention mask
        encoder_attention_mask = prepared_batch.get("encoder_attention_mask")
        if encoder_attention_mask is not None and encoder_attention_mask.dim() == 3:
            # Squeeze out extra dimension if present [B, 1, S] -> [B, S]
            encoder_attention_mask = encoder_attention_mask.squeeze(1)

        logger.debug(
            "DTypes:"
            f"\n-> Timesteps shape: {timesteps.shape}, dtype: {timesteps.dtype}"
            f"\n-> Noisy Latents shape: {noisy_latents.shape}, dtype: {noisy_latents.dtype}"
            f"\n-> Encoder hidden states shape: {prepared_batch['prompt_embeds'].shape}"
            f"\n-> Attention mask shape: {encoder_attention_mask.shape if encoder_attention_mask is not None else None}"
        )

        # Lumina2 transformer kwargs
        lumina_transformer_kwargs = {
            "hidden_states": noisy_latents,
            "timestep": timesteps,
            "encoder_hidden_states": prepared_batch["prompt_embeds"].to(
                device=self.accelerator.device,
                dtype=self.config.base_weight_dtype,
            ),
            "encoder_attention_mask": encoder_attention_mask.to(torch.int32),
            "return_dict": False,
        }

        # Get model prediction
        model_pred = self.model(**lumina_transformer_kwargs)[0]

        # IMPORTANT: Lumina2 uses reverse flow, so we multiply by -1
        # This is the key difference mentioned in the prompt
        model_pred = -model_pred

        return {"model_prediction": model_pred}

    def check_user_config(self):
        """Check and validate user configuration for Lumina2"""
        if self.config.aspect_bucket_alignment != 64:
            logger.warning(
                f"{self.NAME} requires an alignment value of 64px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 64

        if self.config.prediction_type is not None:
            logger.warning(
                f"{self.NAME} uses flow matching and does not support prediction type {self.config.prediction_type}."
            )

        # Lumina2 supports up to 256 tokens by default
        if self.config.tokenizer_max_length is None:
            self.config.tokenizer_max_length = 256
        elif self.config.tokenizer_max_length > 512:
            logger.warning(
                f"{self.NAME} has a maximum token length of 512. Setting to 512."
            )
            self.config.tokenizer_max_length = 512

        # Lumina2 default inference steps
        if self.config.validation_num_inference_steps > 30:
            logger.warning(
                f"{self.NAME} typically uses around 30 inference steps. Consider reducing --validation_num_inference_steps."
            )

    def get_lora_target_layers(self):
        """Get LoRA target layers for Lumina2"""
        if self.config.lora_type.lower() == "standard":
            # Lumina2 uses standard attention layer targeting
            return self.DEFAULT_LORA_TARGET
        elif self.config.lora_type.lower() == "lycoris":
            return self.DEFAULT_LYCORIS_TARGET
        else:
            raise NotImplementedError(
                f"Unknown LoRA target type {self.config.lora_type}."
            )

    def custom_model_card_schedule_info(self):
        """Custom model card info for Lumina2"""
        output_args = []

        # Add training configuration info
        if hasattr(self.config, "flow_matching_loss"):
            output_args.append(f"flow_matching_loss={self.config.flow_matching_loss}")

        if self.config.model_type == "lora":
            output_args.append(f"lora_rank={self.config.lora_rank}")

        output_str = f" (parameters={output_args})" if output_args else ""

        return output_str
