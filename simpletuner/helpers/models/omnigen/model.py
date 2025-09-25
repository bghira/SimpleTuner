import logging

import torch
from diffusers import AutoencoderKL, OmniGenPipeline
from diffusers.models.transformers import OmniGenTransformer2DModel
from diffusers.pipelines.omnigen.processor_omnigen import OmniGenMultiModalProcessor
from transformers import AutoTokenizer

from simpletuner.helpers.models.common import ImageModelFoundation, ModelTypes, PipelineTypes, PredictionTypes
from simpletuner.helpers.models.omnigen.collator import OmniGenTrainingCollator

logger = logging.getLogger(__name__)


class OmniGen(ImageModelFoundation):
    NAME = "OmniGen"
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKL
    LATENT_CHANNEL_COUNT = 4  # Default, user should override if different
    MODEL_CLASS = OmniGenTransformer2DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: OmniGenPipeline,
    }
    DEFAULT_MODEL_FLAVOUR = "v1"
    HUGGINGFACE_PATHS = {
        "v1": "Shitao/OmniGen-v1-diffusers",
    }

    TEXT_ENCODER_CONFIGURATION = {}

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        self.processor = None

    def get_transforms(self, dataset_type: str = "image"):
        from torchvision import transforms

        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def _encode_prompts(self, prompts, is_negative_prompt=False):
        # OmniGen uses token IDs directly; the text encoder caching is skipped during factory init.
        raise NotImplementedError("OmniGen does not use encode_prompts during training.")

    def convert_negative_text_embed_for_pipeline(self, negative_text_embed):
        # OmniGen does not use negative text embeddings
        return {}

    def convert_text_embed_for_pipeline(self, text_embed):
        # OmniGen does not use text embeddings
        return {}

    def _load_preprocessor(self):
        if self.processor is not None:
            return

        # Load the model and tokenizer
        self.tokenizers = [
            AutoTokenizer.from_pretrained(
                self.config.pretrained_model_name_or_path,
                subfolder="tokenizer",
                revision=self.config.revision,
            )
        ]
        self.processor = OmniGenMultiModalProcessor(self.tokenizers[0], max_image_size=1024)
        self.processor.collator = OmniGenTrainingCollator(
            # keep_raw_resolution=self.config.keep_raw_resolution,
        )

    def prepare_batch_conditions(self, batch: dict, state: dict) -> dict:
        """OmniGen-specific flow matching preparation"""
        if self.PREDICTION_TYPE is PredictionTypes.FLOW_MATCHING:
            # Get batch size and device
            bsz = batch["latents"].shape[0]
            device = batch["latents"].device

            # Sample t using OmniGen's approach (normal -> sigmoid)
            u = torch.normal(mean=0.0, std=1.0, size=(bsz,), device=device)
            t = 1.0 / (1.0 + torch.exp(-u))  # sigmoid transformation

            # OmniGen uses timesteps from 0 to 1, NOT scaled by 1000
            batch["timesteps"] = t  # Keep as 0-1 range

            # OmniGen doesn't use sigmas in the traditional sense
            # They directly use t for interpolation
            batch["sigmas"] = t  # For compatibility with base class

            # Expand t to match latent dimensions
            t_view = t.view(-1, 1, 1, 1)

            # Create noisy samples using OmniGen's formulation
            # xt = t * x1 + (1-t) * x0
            # where x1 = clean latents, x0 = noise
            batch["noisy_latents"] = t_view * batch["latents"].float() + (1.0 - t_view) * batch["noise"].float()

            # Store t for potential debugging
            batch["t"] = t

        return batch

    def loss(self, prepared_batch, model_output, apply_conditioning_mask=True):
        """OmniGen-specific loss calculation"""
        # Get the model prediction
        model_pred = model_output["model_prediction"]

        # Calculate target as in OmniGen
        target = prepared_batch["latents"] - prepared_batch["noise"]
        # print(f"Model pred: min={model_pred.min().item():.4f}, max={model_pred.max().item():.4f}, mean={model_pred.mean().item():.4f}, std={model_pred.std().item():.4f}")
        # print(f"Target: min={target.min().item():.4f}, max={target.max().item():.4f}, mean={target.mean().item():.4f}, std={target.std().item():.4f}")
        # print(f"Loss contribution: {((model_pred - target)**2).mean().item():.4f}")
        # Calculate MSE loss
        loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="none")

        # Average over all dimensions
        loss = loss.mean()

        return loss

    def model_predict(self, prepared_batch):
        self._load_preprocessor()
        batch_latents = prepared_batch["noisy_latents"]  # shape [B, 4, H, W]

        # Build list of input text and image features
        all_features = []
        for prompt, latents in zip(prepared_batch["prompts"], batch_latents):
            prompt_dict = self.processor.process_multi_modal_prompt(prompt, input_images=None)
            features = (prompt_dict, latents)  # Pass latents directly
            all_features.append(features)

        # Run the custom collator to collect latent features into a batch with padding and masks
        processed_data = self.processor.collator(all_features)

        # Ensure timesteps are in 0-1 range for OmniGen
        timesteps = prepared_batch["timesteps"]
        if timesteps.max() > 1.0:
            # If timesteps were scaled to 0-1000, scale back to 0-1
            timesteps = timesteps / 1000.0
        # Then call the model

        model_out = self.model(
            hidden_states=processed_data["output_latents"].to(self.accelerator.device),
            timestep=prepared_batch["timesteps"].to(self.accelerator.device),
            input_ids=processed_data["input_ids"].to(self.accelerator.device),
            attention_mask=processed_data["attention_mask"].to(self.accelerator.device),
            position_ids=processed_data["position_ids"].to(self.accelerator.device),
            input_img_latents=processed_data["input_img_latents"] or [],
            input_image_sizes=processed_data["input_image_sizes"] or {},
            return_dict=False,
        )[0]

        return {"model_prediction": model_out}


from simpletuner.helpers.models.registry import ModelRegistry
ModelRegistry.register("omnigen", OmniGen)
