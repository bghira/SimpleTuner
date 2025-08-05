import torch
import torch.nn.functional as F
import random
import logging
import os
from helpers.training.multi_process import _get_rank
from helpers.models.common import (
    ImageModelFoundation,
    PredictionTypes,
    PipelineTypes,
    ModelTypes,
)
from transformers import Qwen2Tokenizer, Qwen2_5_VLForConditionalGeneration
from diffusers import (
    AutoencoderKLQwenImage,
    QwenImageTransformer2DModel,
    QwenImagePipeline,
)
from diffusers.models.attention_processor import Attention
from helpers.training.custom_schedule import (
    apply_flow_schedule_shift,
)

logger = logging.getLogger(__name__)
from helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class QwenImage(ImageModelFoundation):
    NAME = "Qwen-Image"
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKLQwenImage
    LATENT_CHANNEL_COUNT = 16
    AUTOENCODER_SCALING_FACTOR = 1.0  # Qwen Image doesn't use additional scaling

    MODEL_CLASS = QwenImageTransformer2DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: QwenImagePipeline,
    }

    # Default model flavor
    DEFAULT_MODEL_FLAVOUR = "v1.0"
    HUGGINGFACE_PATHS = {
        "v1.0": "Qwen/Qwen-Image",
    }
    MODEL_LICENSE = "other"

    # Qwen Image uses a different text encoder configuration
    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Qwen2.5-VL",
            "tokenizer": Qwen2Tokenizer,
            "tokenizer_subfolder": "tokenizer",
            "model": Qwen2_5_VLForConditionalGeneration,
            "subfolder": "text_encoder",
        },
    }

    # LoRA configuration
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]

    def __init__(self, config: dict, accelerator):
        super().__init__(config, accelerator)
        # Qwen Image specific initialization
        self.is_flow_matching = True
        self.is_transformer = True

    def setup_training_noise_schedule(self):
        """
        Loads the noise schedule for Qwen Image (flow matching).
        """
        from diffusers import FlowMatchEulerDiscreteScheduler

        scheduler_config = {
            "base_image_seq_len": 256,
            "base_shift": 0.5,
            "invert_sigmas": False,
            "max_image_seq_len": 8192,
            "max_shift": 0.9,
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": 0.02,
            "stochastic_sampling": False,
            "time_shift_type": "exponential",
            "use_beta_sigmas": False,
            "use_dynamic_shifting": True,
            "use_exponential_sigmas": False,
            "use_karras_sigmas": False
        }

        self.noise_schedule = FlowMatchEulerDiscreteScheduler(**scheduler_config)
        self.config.prediction_type = "flow_matching"

        return self.config, self.noise_schedule

    def load_model(self, move_to_device: bool = True):
        """
        Load the Qwen Image model components.
        """
        from transformers.utils import ContextManagers
        from helpers.training.deepspeed import deepspeed_zero_init_disabled_context_manager

        dtype = self.config.weight_dtype
        logger.info("Loading Qwen Image model")

        # Determine model paths
        model_path = self.config.pretrained_model_name_or_path
        transformer_subfolder = 'transformer'

        # Check if it's a full checkpoint
        if os.path.exists(model_path):
            transformer_path = os.path.join(model_path, 'transformer')
            if os.path.exists(transformer_path):
                transformer_subfolder = None
                te_folder_path = os.path.join(model_path, 'text_encoder')
                if os.path.exists(te_folder_path):
                    # Full checkpoint, use as base
                    pass

        # Load transformer
        logger.info("Loading transformer")
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            self.model = QwenImageTransformer2DModel.from_pretrained(
                transformer_path if transformer_subfolder is None else model_path,
                subfolder=transformer_subfolder,
                torch_dtype=dtype
            )

        if move_to_device:
            self.model.to(self.accelerator.device, dtype=dtype)

        self.post_model_load_setup()

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """
        Encode prompts using Qwen's text encoder.
        
        Args:
            prompts: List of text prompts to encode
            is_negative_prompt: Whether these are negative prompts
            
        Returns:
            Tuple of (prompt_embeds, prompt_embeds_mask)
        """
        if self.text_encoders is None or len(self.text_encoders) == 0:
            self.load_text_encoder()

        text_encoder = self.text_encoders[0]
        tokenizer = self.tokenizers[0]

        # Move to device if needed
        if text_encoder.device != self.accelerator.device:
            text_encoder.to(self.accelerator.device)

        # Get the pipeline for encoding
        pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        
        # Use pipeline's encode_prompt method
        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
            prompts,
            device=self.accelerator.device,
            num_images_per_prompt=1,
        )

        return prompt_embeds, prompt_embeds_mask, None, prompt_embeds_mask

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        """
        Format the text embeddings for Qwen Image.
        
        Args:
            text_embedding: The embedding tuple from _encode_prompts
            
        Returns:
            Dictionary with formatted embeddings
        """
        prompt_embeds, prompt_embeds_mask, _, masks = text_embedding

        return {
            "prompt_embeds": prompt_embeds,
            "attention_masks": prompt_embeds_mask,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        """
        Convert text embeddings for pipeline use.
        """
        attention_mask = text_embedding.get("attention_masks", None)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        return {
            "prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0)
            if text_embedding["prompt_embeds"].dim() == 2
            else text_embedding["prompt_embeds"],
            "prompt_embeds_mask": attention_mask.to(dtype=torch.int64)
            if attention_mask is not None
            else None,
        }

    def convert_negative_text_embed_for_pipeline(
        self, text_embedding: torch.Tensor, prompt: str
    ) -> dict:
        """
        Convert negative text embeddings for pipeline use.
        """
        attention_mask = text_embedding.get("attention_masks", None)
        if attention_mask is not None and attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        return {
            "negative_prompt_embeds": text_embedding["prompt_embeds"].unsqueeze(0)
            if text_embedding["prompt_embeds"].dim() == 2
            else text_embedding["prompt_embeds"],
            "negative_prompt_embeds_mask": attention_mask.to(dtype=torch.int64)
            if attention_mask is not None
            else None,
        }

    def model_predict(self, prepared_batch):
        """
        Perform a forward pass with the Qwen Image model.
        
        Args:
            prepared_batch: Dictionary containing prepared batch data
            
        Returns:
            Dictionary with model prediction
        """
        latent_model_input = prepared_batch["noisy_latents"]
        timesteps = prepared_batch["timesteps"]
        
        # Prepare the latents for Qwen Image (patchify)
        batch_size, num_channels, height, width = latent_model_input.shape
        
        # Reshape to patches: (B, C, H, W) -> (B, H//2 * W//2, C*4)
        latent_model_input = latent_model_input.view(
            batch_size, num_channels, height // 2, 2, width // 2, 2
        )
        latent_model_input = latent_model_input.permute(0, 2, 4, 1, 3, 5)
        latent_model_input = latent_model_input.reshape(
            batch_size, (height // 2) * (width // 2), num_channels * 4
        )
        
        # Prepare text embeddings
        prompt_embeds = prepared_batch["prompt_embeds"].to(
            device=self.accelerator.device,
            dtype=self.config.weight_dtype,
        )
        
        # Get attention mask
        prompt_embeds_mask = prepared_batch.get("encoder_attention_mask")
        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask.to(
                self.accelerator.device, dtype=torch.int64
            )
            if prompt_embeds_mask.dim() == 3 and prompt_embeds_mask.size(1) == 1:
                prompt_embeds_mask = prompt_embeds_mask.squeeze(1)
        
        # Prepare image shapes
        img_shapes = [(1, height // 2, width // 2)] * batch_size
        
        # Prepare timesteps (normalize to 0-1 range)
        timesteps = (
            torch.tensor(prepared_batch["timesteps"])
            .expand(batch_size)
            .to(device=self.accelerator.device)
            / self.noise_schedule.config.num_train_timesteps
        )
        
        # Get text sequence lengths
        txt_seq_lens = (
            prompt_embeds_mask.sum(dim=1).tolist()
            if prompt_embeds_mask is not None
            else [prompt_embeds.shape[1]] * batch_size
        )
        
        # Forward pass through transformer
        noise_pred = self.model(
            hidden_states=latent_model_input.to(
                self.accelerator.device, self.config.weight_dtype
            ),
            timestep=timesteps,
            guidance=None,  # Qwen Image doesn't use guidance during training
            encoder_hidden_states=prompt_embeds,
            encoder_hidden_states_mask=prompt_embeds_mask,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]
        
        # Unpack the noise prediction back to original shape
        noise_pred = noise_pred.view(
            batch_size, height // 2, width // 2, num_channels, 2, 2
        )
        noise_pred = noise_pred.permute(0, 3, 1, 4, 2, 5)
        noise_pred = noise_pred.reshape(batch_size, num_channels, height, width)
        
        return {"model_prediction": noise_pred}

    def check_user_config(self):
        """
        Check and validate user configuration for Qwen Image.
        """
        super().check_user_config()
        
        # Qwen Image specific checks
        if self.config.aspect_bucket_alignment != 32:
            logger.warning(
                f"{self.NAME} requires an alignment value of 32px. "
                "Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 32
        
        # Ensure we're using flow matching
        if self.config.prediction_type != "flow_matching":
            logger.warning(
                f"{self.NAME} uses flow matching. "
                "Overriding prediction_type to 'flow_matching'."
            )
            self.config.prediction_type = "flow_matching"

    def pre_vae_encode_transform_sample(self, sample):
        """
        Pre-encode transform for the sample before passing it to the VAE.
        For QwenImage, we need to add a frame dimension since it uses Wan VAE.
        """
        # Add frame dimension: (B, C, H, W) -> (B, C, 1, H, W)
        if sample.dim() == 4:
            sample = sample.unsqueeze(2)
        return sample

    def post_vae_encode_transform_sample(self, sample):
        """
        Post-encode transform for Qwen Image VAE output.
        """
        # Remove frame dimension if present: (B, C, 1, H, W) -> (B, C, H, W)
        if sample.dim() == 5 and sample.shape[2] == 1:
            sample = sample.squeeze(2)
        
        # Qwen Image VAE normalization
        if hasattr(self.vae.config, "latents_mean") and hasattr(
            self.vae.config, "latents_std"
        ):
            # Adjust dimensions based on whether frame dim is present
            if sample.dim() == 5:
                latents_mean = torch.tensor(self.vae.config.latents_mean).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                ).to(sample.device, sample.dtype)
                
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1, 1
                ).to(sample.device, sample.dtype)
            else:
                latents_mean = torch.tensor(self.vae.config.latents_mean).view(
                    1, self.vae.config.z_dim, 1, 1
                ).to(sample.device, sample.dtype)
                
                latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                    1, self.vae.config.z_dim, 1, 1
                ).to(sample.device, sample.dtype)
            
            sample = (sample - latents_mean) * latents_std
        
        return sample