import torch, os, logging
import torch.nn.functional as F
from helpers.models.cosmos.pipeline import Cosmos2TextToImagePipeline
from helpers.models.cosmos.transformer import CosmosTransformer3DModel
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

logger = logging.getLogger(__name__)
from helpers.training.multi_process import should_log

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class Cosmos2Image(VideoModelFoundation):
    NAME = "Cosmos (T2I)"
    PREDICTION_TYPE = PredictionTypes.SAMPLE
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
                text_input_ids, attention_mask=attention_mask
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

    def prepare_batch(self, batch: dict, state: dict) -> dict:
        """
        1. Move tensors to the accelerator device / dtype.
        2. Draw σ from the log-uniform EDM schedule.
        3. Add additive Gaussian noise  xₜ = x₀ + σ ε.
        4. Store `sigmas` (broadcast shape B×1×1×1×1) and `noisy_latents`.
        Everything else (prompt embeds, masks, etc.) follows the base
        implementation.
        """
        if not batch:
            return batch

        # ---------- move prompt embeds & latents to device --------------
        target_kwargs = {
            "device": self.accelerator.device,
            "dtype": self.config.weight_dtype,
        }

        if batch.get("prompt_embeds") is not None:
            batch["encoder_hidden_states"] = batch["prompt_embeds"].to(**target_kwargs)

        latents = batch["latent_batch"].to(**target_kwargs)  # clean x0
        batch["latents"] = latents

        # ---------- plain Gaussian noise ε ------------------------------
        noise = torch.randn_like(latents)
        batch["noise"] = noise
        batch["input_noise"] = noise  # no extra perturbation

        # ---------- draw σ and form x_t ---------------------------------
        bsz = latents.size(0)
        sigmas = self.prepare_edm_sigmas(bsz, self.accelerator.device)["sigmas"]  # (B,)
        sigmas_exp = sigmas.view(-1, 1, 1, 1, 1)  # B×1×1×1×1

        batch["sigmas"] = sigmas_exp
        batch["timesteps"] = sigmas  # unused but kept for API
        batch["noisy_latents"] = latents + sigmas_exp * noise  # x_t

        # ---------- any ControlNet / mask specific tweaks ---------------
        batch = self.prepare_batch_conditions(batch=batch, state=state)
        return batch

    def model_predict(self, prepared_batch):
        xt = prepared_batch["noisy_latents"]
        sigmas = prepared_batch["sigmas"].view(-1, 1, 1, 1, 1)  # B×1×1×1×1
        B, _, _, H, W = xt.shape
        device = self.accelerator.device
        dtype = self.config.weight_dtype

        inv = 1.0 / (sigmas + 1.0)  # == c_in == c_skip
        cout = -sigmas * inv

        latent_in = xt * inv
        timestep = (sigmas / (sigmas + 1)).view(B).to(dtype=dtype)  # == current_t

        pad_mask = torch.zeros(B, 1, H, W, device=device, dtype=latent_in.dtype)
        r_pred = self.model(
            hidden_states=latent_in.to(dtype),
            timestep=timestep,
            encoder_hidden_states=prepared_batch["encoder_hidden_states"].to(dtype),
            padding_mask=pad_mask,
            return_dict=False,
        )[
            0
        ]  # transformer output

        x0_pred = inv * xt + cout * r_pred.float()  # behaviour identical to NVIDIA loop
        return {"model_prediction": x0_pred}

    def loss(self, prepared_batch, model_output, apply_conditioning_mask=True):
        x0 = prepared_batch["latents"].float()
        x0_pred = model_output["model_prediction"].float()
        sigmas = prepared_batch["sigmas"]

        w = (sigmas**2 + self.sigma_data**2) / (sigmas * self.sigma_data) ** 2
        while w.ndim < x0.ndim:
            w = w.unsqueeze(-1)

        loss = F.mse_loss(x0_pred, x0, reduction="none") * w

        if apply_conditioning_mask:
            ctype = prepared_batch.get("conditioning_type")
            if ctype == "mask":
                m = prepared_batch["conditioning_pixel_values"][:, :1]
                m = torch.nn.functional.interpolate(m, size=loss.shape[2:], mode="area")
                loss *= m / 2 + 0.5
            elif ctype == "segmentation":
                m = prepared_batch["conditioning_pixel_values"]
                m = torch.sum(m, dim=1, keepdim=True) / 3
                m = torch.nn.functional.interpolate(m, size=loss.shape[2:], mode="area")
                loss *= ((m / 2 + 0.5) > 0).to(loss.dtype)

        return loss.mean()

    def prepare_edm_sigmas(self, bsz: int, device: torch.device) -> torch.Tensor:
        log_min, log_max = map(
            torch.log10, (torch.tensor(self.sigma_min), torch.tensor(self.sigma_max))
        )
        u = torch.rand(bsz, device=device)
        return (10.0 ** (log_min + (log_max - log_min) * u)).to(device)

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
            logger.info(
                f"Setting tokenizer max length to {self.config.tokenizer_max_length}"
            )

        # Validation settings
        if self.config.validation_num_inference_steps < 30:
            logger.warning(
                f"{self.NAME} expects around 35 or more inference steps. "
                f"Consider increasing --validation_num_inference_steps to 35."
            )

        # Disable custom VAEs
        self.config.pretrained_vae_model_name_or_path = None

        # Ensure proper scheduler settings
        if (
            hasattr(self.config, "flow_schedule_shift")
            and self.config.flow_schedule_shift is None
        ):
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
            f" (parameters={output_args})" if output_args else " (default parameters)"
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
