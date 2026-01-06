import logging
import os
import threading
from typing import Dict, Optional

import torch
import torchaudio
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from transformers import T5TokenizerFast, UMT5EncoderModel, Wav2Vec2Model, Wav2Vec2Processor

from simpletuner.helpers.models.common import ModelTypes, PipelineTypes, PredictionTypes, VideoModelFoundation
from simpletuner.helpers.models.tae.types import VideoTAESpec
from simpletuner.helpers.models.wan_s2v import WAV2VEC2_DIM, WAV2VEC2_NUM_LAYERS
from simpletuner.helpers.models.wan_s2v.pipeline import WanS2VPipeline
from simpletuner.helpers.models.wan_s2v.transformer import WanS2VTransformer3DModel
from simpletuner.helpers.training.multi_process import should_log

logger = logging.getLogger(__name__)

if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class WanS2V(VideoModelFoundation):
    """
    Training model for Wan2.2-S2V (Speech-to-Video).

    This model generates video from audio, text, and reference images using
    Wav2Vec2 for audio conditioning.
    """

    NAME = "WanS2V"
    MODEL_DESCRIPTION = "Speech-to-Video generation model"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKLWan
    LATENT_CHANNEL_COUNT = 16
    _TAE_SPEC = VideoTAESpec(filename="taew2_1.pth", description="Wan 2.1 / 2.2 14B VAE")
    DEFAULT_NOISE_SCHEDULER = "euler"

    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    SLIDER_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = WanS2VTransformer3DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: WanS2VPipeline,
    }

    DEFAULT_MODEL_FLAVOUR = "s2v-14b-2.2"
    HUGGINGFACE_PATHS = {
        "s2v-14b-2.2": "tolgacangoz/Wan2.2-S2V-14B-Diffusers",
    }
    MODEL_LICENSE = "apache-2.0"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "UMT5",
            "tokenizer": T5TokenizerFast,
            "subfolder": "text_encoder",
            "tokenizer_subfolder": "tokenizer",
            "model": UMT5EncoderModel,
        },
    }

    # Audio encoder configuration
    AUDIO_ENCODER_MODEL = "facebook/wav2vec2-large-xlsr-53"
    AUDIO_SAMPLE_RATE = 16000

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        # WanS2V has 40 transformer blocks
        return 39

    def get_validation_preview_spec(self):
        return self._TAE_SPEC

    @classmethod
    def supports_chunked_feed_forward(cls) -> bool:
        return True

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        self._audio_encoder = None
        self._audio_processor = None
        self._audio_encoder_lock = threading.Lock()
        self._s2v_warned_missing_audio = False

    def _load_audio_encoder(self):
        """Lazy-load the Wav2Vec2 audio encoder."""
        if self._audio_encoder is not None:
            return

        with self._audio_encoder_lock:
            if self._audio_encoder is not None:
                return

            logger.info(f"Loading Wav2Vec2 audio encoder from {self.AUDIO_ENCODER_MODEL}")
            self._audio_processor = Wav2Vec2Processor.from_pretrained(self.AUDIO_ENCODER_MODEL)
            self._audio_encoder = Wav2Vec2Model.from_pretrained(
                self.AUDIO_ENCODER_MODEL,
                torch_dtype=torch.float32,  # Wav2Vec2 needs fp32
            )
            self._audio_encoder.eval()
            self._audio_encoder.requires_grad_(False)

            # Move to appropriate device
            device = self.accelerator.device
            self._audio_encoder = self._audio_encoder.to(device)
            logger.info(f"Wav2Vec2 audio encoder loaded to {device}")

    @property
    def audio_encoder(self):
        self._load_audio_encoder()
        return self._audio_encoder

    @property
    def audio_processor(self):
        self._load_audio_encoder()
        return self._audio_processor

    def requires_s2v_datasets(self) -> bool:
        """S2V always requires audio datasets."""
        return True

    def supports_audio_inputs(self) -> bool:
        """Wan S2V accepts audio conditioning alongside text/video inputs."""
        return True

    def requires_s2v_validation_inputs(self) -> bool:
        """S2V validation requires audio inputs."""
        return True

    def requires_conditioning_validation_inputs(self) -> bool:
        """S2V requires conditioning (reference image + audio) for validation."""
        return True

    def conditioning_validation_dataset_type(self) -> str:
        """S2V uses video datasets for conditioning validation samples."""
        return "video"

    def encode_audio(self, audio_path: str) -> torch.Tensor:
        """
        Encode audio file to Wav2Vec2 embeddings.

        Args:
            audio_path: Path to audio file

        Returns:
            Audio embeddings [1, num_layers, audio_dim, T]
        """
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample if needed
        if sample_rate != self.AUDIO_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sample_rate, self.AUDIO_SAMPLE_RATE)
            waveform = resampler(waveform)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        waveform = waveform.squeeze(0)  # Remove channel dim

        # Process with Wav2Vec2
        device = self.audio_encoder.device
        inputs = self.audio_processor(
            waveform.numpy(),
            sampling_rate=self.AUDIO_SAMPLE_RATE,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            outputs = self.audio_encoder(
                input_values,
                output_hidden_states=True,
                return_dict=True,
            )

        # Stack all hidden states [1, num_layers, T, audio_dim]
        hidden_states = torch.stack(outputs.hidden_states, dim=1)
        # Permute to [1, num_layers, audio_dim, T]
        hidden_states = hidden_states.permute(0, 1, 3, 2)

        return hidden_states

    def interpolate_audio_to_frames(self, audio_embeds: torch.Tensor, num_latent_frames: int) -> torch.Tensor:
        """Interpolate audio embeddings to match video latent frame count."""
        import torch.nn.functional as F

        # audio_embeds: [B, num_layers, audio_dim, T_audio]
        B, L, D, T = audio_embeds.shape
        audio_flat = audio_embeds.reshape(B * L, D, T)
        audio_interp = F.interpolate(
            audio_flat,
            size=num_latent_frames,
            mode="linear",
            align_corners=False,
        )
        return audio_interp.reshape(B, L, D, num_latent_frames)

    def convert_text_embed_for_training(self, text_embedding):
        """Convert text embedding to training format."""
        embeds = text_embedding["prompt_embeds"]
        masks = text_embedding.get("attention_masks")
        return {
            "prompt_embeds": embeds,
            "attention_masks": masks,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        return {"prompt_embeds": prompt_embeds}

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        return {"negative_prompt_embeds": prompt_embeds}

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        """Encode text prompts using T5."""
        pipeline = self.pipelines.get(PipelineTypes.TEXT2IMG)
        if pipeline is None:
            pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)

        prompt_embeds, masks = pipeline.encode_prompt(
            prompt=prompts,
            device=self.accelerator.device,
        )

        if self.config.t5_padding == "zero":
            prompt_embeds = prompt_embeds * masks.to(device=prompt_embeds.device).unsqueeze(-1).expand(prompt_embeds.shape)

        return prompt_embeds, masks

    def prepare_batch_conditions(self, batch: dict) -> dict:
        """
        Prepare batch with audio conditioning.

        This method handles fetching and processing audio from s2v_datasets.
        """
        # Get audio paths from batch (linked via s2v_datasets)
        audio_embeds_list = batch.get("s2v_audio_embeds")

        if audio_embeds_list is not None:
            # Audio was pre-processed and included in batch
            batch["audio_embeds"] = audio_embeds_list
        elif batch.get("s2v_audio_paths") is not None:
            # Audio paths provided, encode on-the-fly
            audio_paths = batch["s2v_audio_paths"]
            audio_embeds = []
            B = batch["latents"].shape[0]
            T = batch["latents"].shape[2]  # Temporal dimension

            for path in audio_paths:
                if path is not None:
                    embed = self.encode_audio(path)
                    audio_embeds.append(embed)
                else:
                    # No audio for this sample, use zeros
                    zero_embed = torch.zeros(
                        1,
                        WAV2VEC2_NUM_LAYERS,
                        WAV2VEC2_DIM,
                        T,
                        device=batch["latents"].device,
                        dtype=batch["latents"].dtype,
                    )
                    audio_embeds.append(zero_embed)

            batch["audio_embeds"] = torch.cat(audio_embeds, dim=0)
        else:
            if not self._s2v_warned_missing_audio:
                logger.warning(
                    "S2V model requires audio conditioning but none was provided in batch. "
                    "Ensure s2v_datasets is configured properly."
                )
                self._s2v_warned_missing_audio = True
            # Create zero audio embeds as fallback
            B = batch["latents"].shape[0]
            T = batch["latents"].shape[2]  # Temporal dimension
            batch["audio_embeds"] = torch.zeros(
                B,
                WAV2VEC2_NUM_LAYERS,
                WAV2VEC2_DIM,
                T,
                device=batch["latents"].device,
                dtype=batch["latents"].dtype,
            )

        return batch

    def model_predict(self, prepared_batch):
        """Forward pass through S2V transformer with TREAD and CREPA support."""
        B, C, T, H, W = prepared_batch["noisy_latents"].shape
        device = prepared_batch["noisy_latents"].device
        dtype = self.config.weight_dtype

        # Get audio embeddings
        audio_embeds = prepared_batch.get("audio_embeds")
        if audio_embeds is None:
            # Fallback to zeros if no audio
            num_latent_frames = T
            audio_embeds = torch.zeros(
                B,
                WAV2VEC2_NUM_LAYERS,
                WAV2VEC2_DIM,
                num_latent_frames,
                device=device,
                dtype=dtype,
            )
        else:
            # Interpolate audio to match latent frames
            audio_embeds = self.interpolate_audio_to_frames(audio_embeds, T)
            audio_embeds = audio_embeds.to(device=device, dtype=dtype)

        # Get or create reference image latents
        image_latents = prepared_batch.get("conditioning_latents")
        if image_latents is None:
            # Use first frame of noisy latents as reference (during training this would be clean)
            image_latents = prepared_batch["latents"][:, :, :1, :, :]

        # Create empty pose latents
        pose_latents = torch.zeros(
            B,
            16,
            T,
            H,
            W,
            device=device,
            dtype=dtype,
        )

        # Create empty motion latents (for first clip training)
        motion_latents = torch.zeros(
            B,
            C,
            0,
            H,
            W,
            device=device,
            dtype=dtype,
        )

        hidden_states_buffer = self._new_hidden_state_buffer()

        # Build transformer kwargs
        wan_s2v_kwargs = {
            "hidden_states": prepared_batch["noisy_latents"].to(dtype),
            "timestep": prepared_batch["timesteps"],
            "encoder_hidden_states": prepared_batch["encoder_hidden_states"].to(dtype),
            "motion_latents": motion_latents,
            "audio_embeds": audio_embeds,
            "image_latents": image_latents.to(dtype),
            "pose_latents": pose_latents,
            "motion_frames": [17, 5],
            "drop_motion_frames": True,
            "add_last_motion": 0,
            "return_dict": False,
        }

        # CREPA support
        capture_hidden = bool(getattr(self, "crepa_regularizer", None) and self.crepa_regularizer.wants_hidden_states())
        if capture_hidden:
            wan_s2v_kwargs["output_hidden_states"] = True
            wan_s2v_kwargs["hidden_state_layer"] = self.crepa_regularizer.block_index
        if hidden_states_buffer is not None:
            wan_s2v_kwargs["hidden_states_buffer"] = hidden_states_buffer

        # TREAD force_keep_mask support
        force_keep_mask = prepared_batch.get("force_keep_mask")
        if force_keep_mask is not None:
            wan_s2v_kwargs["force_keep_mask"] = force_keep_mask

        model_output = self.model(**wan_s2v_kwargs)

        if capture_hidden:
            if isinstance(model_output, tuple) and len(model_output) >= 2:
                model_pred = model_output[0]
                crepa_hidden = model_output[1]
            else:
                model_pred = model_output[0] if isinstance(model_output, tuple) else model_output
                crepa_hidden = None
                if capture_hidden:
                    logger.warning(
                        f"CREPA requested hidden states from layer {self.crepa_regularizer.block_index} "
                        "but none were returned. Check that crepa_block_index is within the model's block count."
                    )
        else:
            model_pred = model_output[0] if isinstance(model_output, tuple) else model_output
            crepa_hidden = None

        return {
            "model_prediction": model_pred,
            "crepa_hidden_states": crepa_hidden,
            "hidden_states_buffer": hidden_states_buffer,
        }

    def check_user_config(self):
        """Validate configuration for S2V training."""
        if self.config.base_model_precision == "fp8-quanto":
            raise ValueError(f"{self.NAME} does not support fp8-quanto. Please use fp8-torchao or int8 precision instead.")

        if self.config.aspect_bucket_alignment != 32:
            logger.warning(f"{self.NAME} requires an alignment value of 32px. Overriding --aspect_bucket_alignment.")
            self.config.aspect_bucket_alignment = 32

        if self.config.tokenizer_max_length is not None:
            logger.warning(f"{self.NAME} supports max 512 tokens, --tokenizer_max_length is ignored.")
        self.config.tokenizer_max_length = 512

        if self.config.validation_num_inference_steps > 50:
            logger.warning(f"{self.NAME} may waste compute with >50 steps. Consider reducing.")
        if self.config.validation_num_inference_steps < 40:
            logger.warning(f"{self.NAME} expects ~40 inference steps. Consider increasing --validation_num_inference_steps.")

        if not self.config.validation_disable_unconditional:
            logger.info("Disabling unconditional validation for S2V.")
            self.config.validation_disable_unconditional = True

        if self.config.framerate is None:
            self.config.framerate = 15

        self.config.vae_enable_tiling = True
        self.config.vae_enable_slicing = True

    def pretrained_load_args(self, pretrained_load_args: dict) -> dict:
        """Arguments for loading pretrained model."""
        load_args = super().pretrained_load_args(pretrained_load_args)
        return load_args

    def get_pipeline(self, pipeline_type: PipelineTypes, load_base_model: bool = True):
        """Get inference pipeline for validation."""
        if pipeline_type not in self.PIPELINE_CLASSES:
            raise ValueError(f"{self.NAME} does not support pipeline type {pipeline_type}")

        pipeline_class = self.PIPELINE_CLASSES[pipeline_type]

        # Get components
        if load_base_model:
            transformer = self.model
        else:
            transformer = None

        tokenizer = self.get_tokenizer()
        text_encoder = self.get_text_encoder()
        vae = self.get_vae()
        scheduler = FlowMatchEulerDiscreteScheduler()

        # Load audio components
        self._load_audio_encoder()

        pipeline = pipeline_class(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            audio_encoder=self._audio_encoder,
            audio_processor=self._audio_processor,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
        )

        self.pipelines[pipeline_type] = pipeline
        return pipeline

    def update_pipeline_call_kwargs(self, pipeline_kwargs: dict) -> dict:
        """
        Update pipeline kwargs for S2V validation.

        Handles loading audio from the validation conditioning if provided.
        """
        # Check if audio path is in conditioning
        conditioning = pipeline_kwargs.get("_s2v_conditioning")
        if conditioning is not None:
            audio_path = conditioning.get("audio_path")
            if audio_path is not None:
                # Load audio waveform for pipeline
                waveform, sample_rate = torchaudio.load(audio_path)
                if sample_rate != self.AUDIO_SAMPLE_RATE:
                    resampler = torchaudio.transforms.Resample(sample_rate, self.AUDIO_SAMPLE_RATE)
                    waveform = resampler(waveform)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                # Pipeline expects audio as 1D tensor
                pipeline_kwargs["audio"] = waveform.squeeze(0)
                logger.debug(f"Loaded audio for S2V validation from: {audio_path}")

            # Handle reference image from conditioning
            ref_image = conditioning.get("image")
            if ref_image is not None:
                pipeline_kwargs["image"] = ref_image

            # Remove internal conditioning key
            del pipeline_kwargs["_s2v_conditioning"]

        return pipeline_kwargs

    def setup_model_flavour(self):
        """Configure model based on flavour."""
        flavour = getattr(self.config, "model_flavour", self.DEFAULT_MODEL_FLAVOUR)
        if flavour not in self.HUGGINGFACE_PATHS:
            logger.warning(f"Unknown flavour {flavour}, using default {self.DEFAULT_MODEL_FLAVOUR}")
            flavour = self.DEFAULT_MODEL_FLAVOUR
            self.config.model_flavour = flavour

        # Set HuggingFace path if not specified
        if not self.config.pretrained_model_name_or_path:
            self.config.pretrained_model_name_or_path = self.HUGGINGFACE_PATHS[flavour]

        logger.info(f"Configured {self.NAME} with flavour: {flavour}")
