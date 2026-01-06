import logging
import os
import threading
from typing import Optional, Sequence

import torch
from transformers import Gemma3ForConditionalGeneration, GemmaTokenizerFast

from simpletuner.helpers.models.common import ModelTypes, PipelineTypes, PredictionTypes, VideoModelFoundation
from simpletuner.helpers.models.ltxvideo2 import (
    pack_ltx2_audio_latents,
    pack_ltx2_latents,
    unpack_ltx2_audio_latents,
    unpack_ltx2_latents,
)
from simpletuner.helpers.models.ltxvideo2.audio_autoencoder import AutoencoderKLLTX2Audio
from simpletuner.helpers.models.ltxvideo2.autoencoder import AutoencoderKLLTX2Video
from simpletuner.helpers.models.ltxvideo2.connectors import LTX2TextConnectors
from simpletuner.helpers.models.ltxvideo2.pipeline_ltx2 import LTX2Pipeline
from simpletuner.helpers.models.ltxvideo2.pipeline_ltx2_image2video import LTX2ImageToVideoPipeline
from simpletuner.helpers.models.ltxvideo2.transformer import LTX2VideoTransformer3DModel
from simpletuner.helpers.musubi_block_swap import apply_musubi_pretrained_defaults
from simpletuner.helpers.training.multi_process import _get_rank, should_log
from simpletuner.helpers.training.state_tracker import StateTracker

logger = logging.getLogger(__name__)
if should_log():
    logger.setLevel(os.environ.get("SIMPLETUNER_LOG_LEVEL", "INFO"))
else:
    logger.setLevel("ERROR")


class LTXVideo2(VideoModelFoundation):
    NAME = "LTXVideo2"
    MODEL_DESCRIPTION = "Audio-video generation model with flow matching"
    ENABLED_IN_WIZARD = True
    PREDICTION_TYPE = PredictionTypes.FLOW_MATCHING
    MODEL_TYPE = ModelTypes.TRANSFORMER
    AUTOENCODER_CLASS = AutoencoderKLLTX2Video
    LATENT_CHANNEL_COUNT = 128
    DEFAULT_NOISE_SCHEDULER = "flow_matching"
    # The safe diffusers default value for LoRA training targets.
    DEFAULT_LORA_TARGET = ["to_k", "to_q", "to_v", "to_out.0"]
    # Only training the Attention blocks by default.
    DEFAULT_LYCORIS_TARGET = ["Attention"]

    MODEL_CLASS = LTX2VideoTransformer3DModel
    MODEL_SUBFOLDER = "transformer"
    PIPELINE_CLASSES = {
        PipelineTypes.TEXT2IMG: LTX2Pipeline,
        PipelineTypes.IMG2VIDEO: LTX2ImageToVideoPipeline,
    }

    DEFAULT_MODEL_FLAVOUR = "2.0"
    HUGGINGFACE_PATHS = {
        "2.0": "Lightricks/LTX-2",
    }
    MODEL_LICENSE = "apache-2.0"

    TEXT_ENCODER_CONFIGURATION = {
        "text_encoder": {
            "name": "Gemma3",
            "tokenizer": GemmaTokenizerFast,
            "subfolder": "text_encoder",
            "tokenizer_subfolder": "tokenizer",
            "use_fast": True,
            "model": Gemma3ForConditionalGeneration,
        },
    }

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        self.audio_vae = None
        self.connectors = None
        self._audio_vae_lock = threading.Lock()
        self._connector_lock = threading.Lock()
        self._warned_missing_audio = False

    @classmethod
    def max_swappable_blocks(cls, config=None) -> Optional[int]:
        # LTX-2 transformer blocks default to 48; leave at least 1 on GPU.
        return 47

    def supports_audio_inputs(self) -> bool:
        return True

    def uses_audio_latents(self) -> bool:
        return True

    def get_vae_for_dataset_type(self, dataset_type: str):
        if dataset_type == "audio":
            self._load_audio_vae(move_to_device=True)
            return self.audio_vae
        return self.get_vae()

    def _resolve_audio_vae_dtype(self):
        if hasattr(self.config, "vae_dtype"):
            if self.config.vae_dtype == "bf16":
                return torch.bfloat16
            if self.config.vae_dtype == "fp16":
                return torch.float16
            if self.config.vae_dtype == "fp32":
                return torch.float32
        return self.config.weight_dtype

    def _load_audio_vae(self, move_to_device: bool = True):
        if self.audio_vae is not None:
            return
        with self._audio_vae_lock:
            if self.audio_vae is not None:
                return
            audio_vae_path = getattr(self.config, "pretrained_audio_vae_model_name_or_path", None)
            if audio_vae_path is None:
                audio_vae_path = self.config.pretrained_model_name_or_path
            if audio_vae_path is None:
                raise ValueError("Unable to resolve audio VAE path for LTX-2.")
            logger.info(f"Loading LTX-2 audio VAE from {audio_vae_path}")
            audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
                audio_vae_path,
                subfolder="audio_vae",
                torch_dtype=self._resolve_audio_vae_dtype(),
                revision=self.config.revision,
                variant=self.config.variant,
                use_safetensors=True,
            )
            audio_vae.requires_grad_(False)
            if move_to_device:
                audio_vae.to(self.accelerator.device, dtype=self._resolve_audio_vae_dtype())
            self.audio_vae = audio_vae

    def _load_connectors(self, move_to_device: bool = True):
        if self.connectors is not None:
            return
        with self._connector_lock:
            if self.connectors is not None:
                return
            model_path = self._model_config_path()
            logger.info(f"Loading LTX-2 text connectors from {model_path}")
            connectors = LTX2TextConnectors.from_pretrained(
                model_path,
                subfolder="connectors",
                torch_dtype=self.config.weight_dtype,
                revision=self.config.revision,
                variant=self.config.variant,
                use_safetensors=True,
            )
            if move_to_device:
                connectors.to(self.accelerator.device, dtype=self.config.weight_dtype)
            self.connectors = connectors
            if self.model is not None and getattr(self.model, "connectors", None) is None:
                self.model.connectors = connectors

    def post_model_load_setup(self):
        super().post_model_load_setup()
        self._load_connectors(move_to_device=True)

    def load_vae(self, move_to_device: bool = True):
        super().load_vae(move_to_device=move_to_device)
        enable_patch_conv = getattr(self.config, "vae_enable_patch_conv", False)
        enable_temporal_roll = getattr(self.config, "vae_enable_temporal_roll", False)
        if enable_patch_conv or enable_temporal_roll:
            if hasattr(self.vae, "enable_temporal_chunking"):
                logger.info(
                    "Enabling LTX-2 VAE temporal chunking%s.",
                    " (temporal roll)" if enable_temporal_roll else "",
                )
                self.vae.enable_temporal_chunking()
            else:
                logger.warning("VAE temporal chunking is enabled, but not yet supported by LTX-2.")
        else:
            if hasattr(self.vae, "disable_temporal_chunking"):
                self.vae.disable_temporal_chunking()
        self._load_audio_vae(move_to_device=move_to_device)

    def encode_cache_batch(self, vae, samples, metadata_entries: Optional[list] = None):
        if isinstance(vae, AutoencoderKLLTX2Audio):
            sample_rates = self._resolve_audio_sample_rates(metadata_entries, samples.shape[0])
            return vae.encode_waveform(samples, sample_rates=sample_rates, return_dict=True)
        return super().encode_cache_batch(vae, samples, metadata_entries=metadata_entries)

    def _resolve_audio_sample_rates(self, metadata_entries: Optional[list], batch_size: int) -> Sequence[int]:
        target_rate = getattr(self.audio_vae, "sample_rate", None)
        if target_rate is None:
            target_rate = getattr(getattr(self.audio_vae, "config", None), "sample_rate", 16000)
        resolved: list[int] = []
        for idx in range(batch_size):
            rate = None
            if metadata_entries and idx < len(metadata_entries):
                entry = metadata_entries[idx]
                metadata = entry.get("metadata") if isinstance(entry, dict) else None
                if isinstance(metadata, dict):
                    rate = metadata.get("sample_rate") or metadata.get("sampling_rate")
            if rate is None:
                rate = target_rate
            resolved.append(int(rate))
        return resolved

    def check_user_config(self):
        if self.config.aspect_bucket_alignment != 64:
            logger.warning(
                f"{self.NAME} requires an alignment value of 64px. Overriding the value of --aspect_bucket_alignment."
            )
            self.config.aspect_bucket_alignment = 64

        if self.config.prediction_type is not None:
            logger.warning(f"{self.NAME} does not support prediction type {self.config.prediction_type}.")

        if self.config.tokenizer_max_length is not None and self.config.tokenizer_max_length != 1024:
            logger.warning(f"-!- {self.NAME} supports a max length of 1024 tokens, --tokenizer_max_length is ignored -!-")
        self.config.tokenizer_max_length = 1024

        if self.config.framerate is None:
            self.config.framerate = 25

    def update_pipeline_call_kwargs(self, pipeline_kwargs):
        pipeline_kwargs["num_frames"] = min(125, self.config.validation_num_video_frames or 125)
        pipeline_kwargs["frame_rate"] = self.config.framerate or 25
        conditioning = pipeline_kwargs.pop("_s2v_conditioning", None)
        if conditioning is not None:
            audio_path = conditioning.get("audio_path")
            if audio_path is not None:
                audio_latents = self._load_audio_latents_for_validation(audio_path)
                if audio_latents is not None:
                    pipeline_kwargs["audio_latents"] = audio_latents
        return pipeline_kwargs

    def requires_s2v_validation_inputs(self) -> bool:
        args = StateTracker.get_args()
        if not getattr(args, "validation_using_datasets", False):
            return False
        eval_ids = getattr(args, "eval_dataset_id", None)
        eval_list: list[str] = []
        if isinstance(eval_ids, (list, tuple)):
            eval_list = [str(item) for item in eval_ids]
        elif eval_ids:
            eval_list = [str(eval_ids)]
        if eval_list:
            return any(bool(StateTracker.get_s2v_datasets(dataset_id)) for dataset_id in eval_list)
        return bool(StateTracker.get_s2v_mappings())

    def _load_audio_latents_for_validation(self, audio_path: str) -> Optional[torch.Tensor]:
        backend_id = self._resolve_audio_backend_id(audio_path)
        if backend_id is None:
            raise ValueError(
                f"Unable to resolve audio backend for validation audio '{audio_path}'. "
                "Ensure the audio dataset is linked via s2v_datasets."
            )
        cache = StateTracker.get_vaecache(id=backend_id)
        latents = cache.retrieve_from_cache(audio_path)
        if isinstance(latents, dict):
            latents = latents.get("latents")
        if not torch.is_tensor(latents):
            raise ValueError(f"Expected audio latents to be a Tensor, got {type(latents)}.")
        return latents.to(device=self.accelerator.device, dtype=self.config.weight_dtype)

    def _resolve_audio_backend_id(self, audio_path: str) -> Optional[str]:
        audio_backends = StateTracker.get_data_backends(_type="audio")
        if not audio_backends:
            return None
        for backend_id, backend in audio_backends.items():
            config = backend.get("config", {})
            root = config.get("instance_data_dir")
            if not root:
                continue
            try:
                root_abs = os.path.abspath(root)
                path_abs = os.path.abspath(audio_path)
                if os.path.commonpath([root_abs, path_abs]) == root_abs:
                    return backend_id
            except ValueError:
                continue
        return None

    def _format_text_embedding(self, text_embedding: torch.Tensor):
        prompt_embeds, prompt_attention_mask, _, _ = text_embedding
        return {
            "prompt_embeds": prompt_embeds,
            "attention_masks": prompt_attention_mask,
        }

    def convert_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        attention_masks = text_embedding["attention_masks"]

        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if attention_masks.dim() == 1:
            attention_masks = attention_masks.unsqueeze(0)

        return {
            "prompt_embeds": prompt_embeds,
            "prompt_attention_mask": attention_masks,
        }

    def convert_negative_text_embed_for_pipeline(self, text_embedding: torch.Tensor) -> dict:
        prompt_embeds = text_embedding["prompt_embeds"]
        attention_masks = text_embedding["attention_masks"]

        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        if attention_masks.dim() == 1:
            attention_masks = attention_masks.unsqueeze(0)

        return {
            "negative_prompt_embeds": prompt_embeds,
            "negative_prompt_attention_mask": attention_masks,
        }

    def _encode_prompts(self, prompts: list, is_negative_prompt: bool = False):
        pipeline = self.pipelines.get(PipelineTypes.TEXT2IMG)
        if pipeline is None:
            pipeline = self.get_pipeline(PipelineTypes.TEXT2IMG, load_base_model=False)
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        ) = pipeline.encode_prompt(
            prompt=prompts,
            device=self.accelerator.device,
            max_sequence_length=self.config.tokenizer_max_length or 1024,
        )
        return (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
        )

    def _extract_sigmas_1d(self, sigmas: torch.Tensor) -> torch.Tensor:
        if sigmas.dim() == 1:
            return sigmas
        return sigmas.view(sigmas.shape[0], -1)[:, 0]

    def _build_empty_audio_latents(self, batch: dict, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        self._load_audio_vae(move_to_device=True)
        if self.audio_vae is None:
            raise ValueError("Audio VAE is required to construct placeholder audio latents.")
        latent_channels = getattr(self.audio_vae.config, "latent_channels", None) or 8
        mel_bins = getattr(self.audio_vae.config, "mel_bins", None) or 64
        mel_compression = getattr(self.audio_vae, "mel_compression_ratio", 4)
        temporal_compression = getattr(self.audio_vae, "temporal_compression_ratio", 4)
        latent_mel_bins = mel_bins // mel_compression

        video_latents = batch.get("latents")
        if video_latents is None:
            raise ValueError("Cannot infer audio latent length without video latents.")
        latent_frames = int(video_latents.shape[2])
        temporal_ratio = getattr(self.get_vae(), "temporal_compression_ratio", 8)
        video_frames = int((latent_frames - 1) * temporal_ratio + 1)
        frame_rate = self.config.framerate or 25
        duration_s = video_frames / frame_rate
        sampling_rate = getattr(self.audio_vae.config, "sample_rate", 16000)
        hop_length = getattr(self.audio_vae.config, "mel_hop_length", 160)
        latents_per_second = float(sampling_rate) / float(hop_length) / float(temporal_compression)
        latent_length = max(1, int(duration_s * latents_per_second))
        shape = (video_latents.shape[0], latent_channels, latent_length, latent_mel_bins)
        return torch.zeros(shape, device=device, dtype=dtype)

    def prepare_batch_conditions(self, batch: dict, state: dict) -> dict:
        batch = super().prepare_batch_conditions(batch=batch, state=state)

        audio_latents = batch.get("audio_latent_batch")
        audio_mask = batch.get("audio_latent_mask")
        target_device = self.accelerator.device
        target_dtype = self.config.weight_dtype

        if audio_latents is None:
            audio_latents = self._build_empty_audio_latents(batch, target_device, target_dtype)
            if audio_mask is None:
                audio_mask = torch.zeros(audio_latents.shape[0], device=target_device, dtype=torch.float32)
            if not self._warned_missing_audio and _get_rank() == 0:
                logger.warning(
                    "LTX-2 received no audio latents for this batch; using zeros and masking audio loss. "
                    "Provide s2v_datasets with cached audio latents to train audio generation."
                )
                self._warned_missing_audio = True
        else:
            if isinstance(audio_latents, dict):
                audio_latents = audio_latents.get("latents")
            if not torch.is_tensor(audio_latents):
                raise ValueError(f"Expected audio latents to be a Tensor, got {type(audio_latents)}.")
            audio_latents = audio_latents.to(device=target_device, dtype=target_dtype)

        if audio_mask is None:
            audio_mask = torch.ones(audio_latents.shape[0], device=target_device, dtype=torch.float32)
        else:
            audio_mask = audio_mask.to(device=target_device, dtype=torch.float32)

        if audio_latents.ndim != 4:
            raise ValueError(f"Expected audio latents to have shape [B, C, L, M], got {audio_latents.shape}.")

        audio_noise = torch.randn_like(audio_latents)
        audio_input_noise = audio_noise
        if self.config.input_perturbation != 0 and (
            not getattr(self.config, "input_perturbation_steps", None)
            or state.get("global_step", 0) < self.config.input_perturbation_steps
        ):
            input_perturbation = self.config.input_perturbation
            if getattr(self.config, "input_perturbation_steps", None):
                input_perturbation *= 1.0 - (state.get("global_step", 0) / self.config.input_perturbation_steps)
            audio_input_noise = audio_noise + input_perturbation * torch.randn_like(audio_latents)

        if self.PREDICTION_TYPE is PredictionTypes.FLOW_MATCHING:
            sigmas_1d = self._extract_sigmas_1d(batch["sigmas"])
            audio_sigmas = sigmas_1d.view(sigmas_1d.shape[0], *([1] * (audio_latents.ndim - 1)))
            audio_noisy = (1 - audio_sigmas) * audio_latents + audio_sigmas * audio_input_noise
            batch["audio_sigmas"] = audio_sigmas
        else:
            audio_noisy = self.noise_schedule.add_noise(
                audio_latents.float(),
                audio_input_noise.float(),
                batch["timesteps"],
            ).to(device=target_device, dtype=target_dtype)

        batch["audio_latents"] = audio_latents
        batch["audio_latent_mask"] = audio_mask
        batch["audio_noise"] = audio_noise
        batch["audio_noisy_latents"] = audio_noisy

        return batch

    def _prepare_force_keep_mask(self, latents: torch.Tensor, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """
        Validate and reshape the optional force_keep_mask to match token count for TREAD routing.
        """
        if mask is None:
            return None

        patch_size = getattr(self.unwrap_model(self.model).config, "patch_size", 1)
        patch_size_t = getattr(self.unwrap_model(self.model).config, "patch_size_t", 1)
        if not isinstance(patch_size, int) or not isinstance(patch_size_t, int):
            raise ValueError(f"Unexpected patch_size values: {patch_size}, {patch_size_t}")

        tokens_expected = (
            (latents.shape[2] // patch_size_t) * (latents.shape[3] // patch_size) * (latents.shape[4] // patch_size)
        )

        if mask.dim() > 2:
            mask = mask.view(mask.shape[0], -1)

        if mask.numel() == mask.shape[0] * tokens_expected and mask.shape[1] != tokens_expected:
            mask = mask.view(mask.shape[0], tokens_expected)

        if mask.shape[1] != tokens_expected:
            raise ValueError(
                f"force_keep_mask length {mask.shape[1]} does not match expected token count {tokens_expected} "
                f"for patch_size {patch_size}/{patch_size_t}."
            )

        return mask.to(device=latents.device, dtype=torch.bool)

    def model_predict(self, prepared_batch):
        noisy_latents = prepared_batch["noisy_latents"]
        if noisy_latents.shape[1] != self.LATENT_CHANNEL_COUNT:
            raise ValueError(
                "LTX-2 requires 128-channel video latents. Ensure the VAE cache matches the LTX-2 video autoencoder."
            )
        audio_noisy = prepared_batch.get("audio_noisy_latents")
        if audio_noisy is None:
            raise ValueError("LTX-2 requires audio latents for training.")

        self._load_connectors(move_to_device=True)
        if self.connectors is None:
            raise ValueError("LTX-2 text connectors failed to load.")

        num_frames = noisy_latents.shape[2]
        height = noisy_latents.shape[3]
        width = noisy_latents.shape[4]
        patch_size = getattr(self.model.config, "patch_size", 1)
        patch_size_t = getattr(self.model.config, "patch_size_t", 1)

        packed_noisy = pack_ltx2_latents(noisy_latents, patch_size, patch_size_t).to(self.config.weight_dtype)

        audio_latents = prepared_batch["audio_latents"]
        audio_num_frames = audio_latents.shape[2]
        audio_mel_bins = audio_latents.shape[3]
        packed_audio_noisy = pack_ltx2_audio_latents(audio_noisy).to(self.config.weight_dtype)

        encoder_hidden_states = prepared_batch["encoder_hidden_states"]
        encoder_attention_mask = prepared_batch.get("encoder_attention_mask")
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(
                encoder_hidden_states.shape[:2],
                device=encoder_hidden_states.device,
                dtype=torch.float32,
            )
        additive_attention_mask = (1 - encoder_attention_mask.to(encoder_hidden_states.dtype)) * -1000000.0
        connector_video_embeds, connector_audio_embeds, connector_attention_mask = self.connectors(
            encoder_hidden_states,
            additive_attention_mask,
            additive_mask=True,
        )

        force_keep_mask = None
        raw_force_keep = prepared_batch.get("force_keep_mask")
        if raw_force_keep is not None and getattr(self.config, "tread_config", None):
            force_keep_mask = self._prepare_force_keep_mask(noisy_latents, raw_force_keep)

        hidden_states_buffer = self._new_hidden_state_buffer()
        capture_hidden = bool(getattr(self, "crepa_regularizer", None) and self.crepa_regularizer.wants_hidden_states())
        transformer_kwargs = {
            "hidden_states": packed_noisy,
            "audio_hidden_states": packed_audio_noisy,
            "encoder_hidden_states": connector_video_embeds,
            "audio_encoder_hidden_states": connector_audio_embeds,
            "timestep": prepared_batch["timesteps"],
            "timestep_sign": (
                prepared_batch.get("twinflow_time_sign") if getattr(self.config, "twinflow_enabled", False) else None
            ),
            "encoder_attention_mask": connector_attention_mask,
            "audio_encoder_attention_mask": connector_attention_mask,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "fps": self.config.framerate or 25,
            "audio_num_frames": audio_num_frames,
            "return_dict": False,
        }
        if force_keep_mask is not None:
            transformer_kwargs["force_keep_mask"] = force_keep_mask
        if capture_hidden:
            transformer_kwargs["output_hidden_states"] = True
            transformer_kwargs["hidden_state_layer"] = self.crepa_regularizer.block_index
        if hidden_states_buffer is not None:
            transformer_kwargs["hidden_states_buffer"] = hidden_states_buffer

        model_output = self.model(**transformer_kwargs)
        if capture_hidden:
            if isinstance(model_output, tuple):
                video_pred = model_output[0]
                audio_pred = model_output[1] if len(model_output) > 1 else None
                crepa_hidden = model_output[2] if len(model_output) > 2 else None
            else:
                video_pred, audio_pred = model_output.sample, model_output.audio_sample
                crepa_hidden = getattr(model_output, "crepa_hidden_states", None)
            if crepa_hidden is None and not getattr(self.crepa_regularizer, "use_backbone_features", False):
                raise ValueError(
                    f"CREPA requested hidden states from layer {self.crepa_regularizer.block_index} "
                    "but none were returned. Check that crepa_block_index is within the model's block count."
                )
        else:
            crepa_hidden = None
            if isinstance(model_output, tuple):
                video_pred, audio_pred = model_output
            else:
                video_pred, audio_pred = model_output.sample, model_output.audio_sample

        video_pred = unpack_ltx2_latents(
            video_pred,
            num_frames=num_frames,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            height=height,
            width=width,
        )

        audio_pred = unpack_ltx2_audio_latents(
            audio_pred,
            latent_length=audio_num_frames,
            num_mel_bins=audio_mel_bins,
            patch_size=None,
            patch_size_t=None,
        )

        return {
            "model_prediction": video_pred,
            "audio_prediction": audio_pred,
            "crepa_hidden_states": crepa_hidden,
            "hidden_states_buffer": hidden_states_buffer,
        }

    def tread_init(self):
        """
        Initialize the TREAD model training method for LTX-2 checkpoints.
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

        logger.info("TREAD training is enabled for LTX-2")

    def pretrained_load_args(self, pretrained_load_args: dict) -> dict:
        args = super().pretrained_load_args(pretrained_load_args)
        return apply_musubi_pretrained_defaults(self.config, args)

    def loss(self, prepared_batch: dict, model_output, apply_conditioning_mask: bool = True):
        loss = super().loss(prepared_batch, model_output, apply_conditioning_mask=apply_conditioning_mask)

        audio_pred = model_output.get("audio_prediction")
        if audio_pred is None:
            return loss
        audio_target = prepared_batch.get("audio_noise") - prepared_batch.get("audio_latents")
        if audio_target is None:
            return loss

        audio_loss = (audio_pred.float() - audio_target.float()) ** 2
        audio_mask = prepared_batch.get("audio_latent_mask")
        if audio_mask is not None:
            mask = audio_mask.view(audio_mask.shape[0], *([1] * (audio_loss.ndim - 1)))
            audio_loss = audio_loss * mask
        audio_loss = audio_loss.mean()
        weight = float(getattr(self.config, "audio_loss_weight", 1.0) or 1.0)

        return loss + audio_loss * weight


from simpletuner.helpers.models.registry import ModelRegistry

ModelRegistry.register("ltxvideo2", LTXVideo2)
