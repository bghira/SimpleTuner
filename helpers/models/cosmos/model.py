import torch, os, logging
from helpers.models.cosmos.pipeline import Cosmos2TextToImagePipeline
from helpers.models.common import (
    VideoModelFoundation,
    PredictionTypes,
    PipelineTypes,
    ModelTypes,
)
from transformers import (
    T5TokenizerFast,
    T5EncoderModel,
)
from diffusers import AutoencoderKLWan
from diffusers import CosmosTransformer3DModel
from helpers.training.multi_process import _get_rank

logger = logging.getLogger(__name__)
logger.setLevel(
    os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO") if _get_rank() == 0 else "ERROR"
)


class Cosmos2Image(VideoModelFoundation):
    NAME = "Cosmos (T2I)"
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKLWan
    LATENT_CHANNEL_COUNT = 16
    DEFAULT_NOISE_SCHEDULER = "flow_matching"
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    # Only training the Attention blocks by default.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = CosmosTransformer3DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: Cosmos2TextToImagePipeline,
        # PipelineTypes.IMG2IMG: None,
        # PipelineTypes.CONTROLNET: None,
    }

    # The default model flavor to use when none is specified.
    DEFAULT_MODEL_FLAVOUR = "2b"
    HUGGINGFACE_PATHS = {
        "2b": "nvidia/Cosmos-Predict2-2B-Text2Image",
        "14b": "nvidia/Cosmos-Predict2-14B-Text2Image",
    }
    MODEL_LICENSE = "other"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "T5 11B",
            "tokenizer": T5TokenizerFast,
            "subfolder": "text_encoder",
            "tokenizer_subfolder": "tokenizer",
            "model": T5EncoderModel,
        },
    }
    sigma_max = 80.0
    sigma_min = 0.002
    sigma_data = 1.0
    final_sigmas_type = "sigma_min"

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        """
        Format the T5 text embedding for storage.
        
        Args:
            text_embedding (torch.Tensor): The embed to adjust.
        
        Returns:
            dict: Formatted embedding data.
        """
        return {
            "prompt_embeds": text_embedding,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        """Convert stored embeddings for pipeline use."""
        return {
            "prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
        }

    def convert_negative_text_embed_for_pipeline(
        self, text_embedding: torch.Tensor, prompt: str
    ) -> dict:
        """Convert negative embeddings for pipeline use."""
        return {
            "negative_prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0),
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode a prompt using T5 encoder.
        
        Args:
            prompts: The list of prompts to encode.
            is_negative_prompt: Whether encoding negative prompts.
        
        Returns:
            Text encoder output (raw)
        """
        max_sequence_length = self.config.tokenizer_max_length or 512
        device = self.accelerator.device
        
        # Ensure prompts is a list
        prompts = [prompts] if isinstance(prompts, str) else prompts
        
        # Tokenize
        text_inputs = self.tokenizers[0](
            prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        text_input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.bool().to(device)
        
        # Encode
        with torch.no_grad():
            prompt_embeds = self.text_encoders[0](
                text_input_ids,
                attention_mask=attention_mask
            ).last_hidden_state
        
        # Apply attention mask to zero out padding tokens
        lengths = attention_mask.sum(dim=1).cpu()
        for i, length in enumerate(lengths):
            prompt_embeds[i, length:] = 0
        
        return prompt_embeds

    def pre_vae_encode_transform_sample(self, sample):
        """
        We have to boost the thing from image to video w/ single frame.
        """
        if sample.ndim == 4:
            # Single frame, add a dummy dimension for num_frames
            sample = sample.unsqueeze(2)
        elif sample.ndim != 5:
            raise ValueError(
                f"Cosmos T2I expects input with 4 or 5 dimensions, got {sample.ndim}."
            )
        
        return sample

    def model_predict(self, prepared_batch):
        """
        Perform model prediction for training.
        
        Args:
            prepared_batch: Dictionary containing batch data
            
        Returns:
            Dictionary containing model prediction
        """
        if prepared_batch["noisy_latents"].shape[1] != 32:
            raise ValueError(
                f"Cosmos T2I requires a latent size of 32 channels. "
                f"Batch received: {prepared_batch['noisy_latents'].shape}"
            )
        # we have to split the mu and logvar channels on the noisy latents
        if prepared_batch["noisy_latents"].shape[1] != 16:
            # just slice the first 16 channels and discard the rest
            prepared_batch["noisy_latents"] = prepared_batch["noisy_latents"].narrow(
                1, 0, 16
            )
            # slice also the target latents
            prepared_batch["latents"] = prepared_batch["latents"].narrow(
                1, 0, 16
            )
            # and the noise
            prepared_batch["noise"] = prepared_batch["noise"].narrow(
                1, 0, 16
            )
        
        # For T2I, we use single frame (num_frames=1)
        batch_size, channels, num_frames, height, width = prepared_batch["noisy_latents"].shape
        
        # Create padding mask
        padding_mask = torch.zeros(
            1, 1, height, width,
            device=prepared_batch["noisy_latents"].device,
            dtype=prepared_batch["noisy_latents"].dtype
        )
        
        # Prepare timesteps - Cosmos uses a different timestep format
        timesteps = prepared_batch["timesteps"]
        current_sigma = timesteps  # Assuming timesteps are sigmas
        current_t = current_sigma / (current_sigma + 1)
        timestep = current_t.to(dtype=prepared_batch["noisy_latents"].dtype)
        
        # Model forward pass
        model_pred = self.model(
            hidden_states=prepared_batch["noisy_latents"].to(
                device=self.accelerator.device,
                dtype=self.config.weight_dtype,
            ),
            timestep=timestep,
            encoder_hidden_states=prepared_batch["encoder_hidden_states"].to(
                device=self.accelerator.device,
                dtype=self.config.weight_dtype,
            ),
            padding_mask=padding_mask,
            return_dict=False,
        )[0]
        # return the split mu and logvar channels
        model_pred = model_pred[:, :16, :, :, :]  # Keep only the first
        return {
            "model_prediction": model_pred,
        }

    def prepare_flow_matching_params(self, batch_size: int, device: torch.device):
        """
        Prepare flow matching specific parameters.
        
        Args:
            batch_size: Current batch size
            device: Device to create tensors on
            
        Returns:
            Dictionary of flow matching parameters
        """
        # Sample sigmas according to Cosmos schedule
        sigmas = torch.rand(batch_size, device=device)
        sigmas = self.sigma_min + (self.sigma_max - self.sigma_min) * sigmas
        
        return {
            "sigmas": sigmas,
            "sigma_data": self.sigma_data,
        }

    def check_user_config(self):
        """
        Checks self.config values against important issues.
        """
        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(
                f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision level instead."
            )
        
        if self.config.aspect_bucket_alignment != 16:
            logger.warning(
                f"{self.NAME} requires an alignment value of 16px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 16
        
        # T5 tokenizer settings
        if self.config.tokenizer_max_length is None:
            self.config.tokenizer_max_length = 512
            logger.info(f"Setting tokenizer max length to {self.config.tokenizer_max_length}")
        
        # Validation settings
        if self.config.validation_num_inference_steps < 30:
            logger.warning(
                f"{self.NAME} expects around 35 or more inference steps. "
                f"Consider increasing --validation_num_inference_steps to 35."
            )
        
        # Disable custom VAEs
        self.config.pretrained_vae_model_name_or_path = None
        
        # Ensure proper scheduler settings
        if hasattr(self.config, "flow_schedule_shift") and self.config.flow_schedule_shift is None:
            self.config.flow_schedule_shift = 1.0  # Cosmos default

    def custom_model_card_schedule_info(self):
        """
        Provide custom schedule information for model card.
        """
        output_args = []
        
        output_args.append(f"sigma_max={self.sigma_max}")
        output_args.append(f"sigma_min={self.sigma_min}")
        output_args.append(f"sigma_data={self.sigma_data}")
        
        if self.config.flow_schedule_auto_shift:
            output_args.append("flow_schedule_auto_shift")
        if self.config.flow_schedule_shift is not None:
            output_args.append(f"shift={self.config.flow_schedule_shift}")
        if self.config.flow_use_beta_schedule:
            output_args.append(
                f"flow_beta_schedule_alpha={self.config.flow_beta_schedule_alpha}"
            )
            output_args.append(
                f"flow_beta_schedule_beta={self.config.flow_beta_schedule_beta}"
            )
        
        output_str = (
            f" (parameters={output_args})"
            if output_args
            else " (default parameters)"
        )
        
        return output_str

    def get_latent_shapes(self, resolution: tuple) -> tuple:
        """
        Calculate latent shapes for given resolution.
        
        Args:
            resolution: (height, width) tuple
            
        Returns:
            (latent_height, latent_width) tuple
        """
        height, width = resolution
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        
        return (latent_height, latent_width)