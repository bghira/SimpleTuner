import threading
from typing import Optional

import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from simpletuner.helpers.models.common import PipelineTypes
from simpletuner.helpers.models.infinitetalk import INFINITETALK_AUDIO_FPS
from simpletuner.helpers.models.infinitetalk.audio import (
    align_waveform_to_video_frames,
    encode_wav2vec_hidden_states,
    window_audio_embeddings,
)
from simpletuner.helpers.models.infinitetalk.pipeline import InfiniteTalkPipeline
from simpletuner.helpers.models.infinitetalk.transformer import InfiniteTalkTransformer3DModel
from simpletuner.helpers.models.registry import ModelRegistry
from simpletuner.helpers.models.wan.model import Wan


class InfiniteTalk(Wan):
    NAME = "InfiniteTalk"
    MODEL_DESCRIPTION = "Audio-driven Wan 2.1 image-to-video model"
    MODEL_CLASS = InfiniteTalkTransformer3DModel
    PIPELINE_CLASSES = {
        **Wan.PIPELINE_CLASSES,
        PipelineTypes.IMG2VIDEO: InfiniteTalkPipeline,
    }
    DEFAULT_MODEL_FLAVOUR = "single-14b-480p"
    HUGGINGFACE_PATHS = {
        "single-14b-480p": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
    }
    AUDIO_ENCODER_MODEL = "TencentGameMate/chinese-wav2vec2-base"
    AUDIO_SAMPLE_RATE = 16000

    I2V_FLAVOURS = frozenset({"single-14b-480p"})
    I2V_CLIP_CONDITIONED_FLAVOURS = I2V_FLAVOURS
    FLF2V_FLAVOURS = frozenset()
    TI2V_FLAVOURS = frozenset()
    EXPAND_TIMESTEP_FLAVOURS = frozenset()
    STRICT_I2V_FLAVOURS = tuple(I2V_FLAVOURS)

    DEFAULT_LORA_TARGET = [
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "audio_cross_attn.q_linear",
        "audio_cross_attn.kv_linear",
        "audio_cross_attn.proj",
    ]
    DEFAULT_LYCORIS_TARGET = ["Attention", "InfiniteTalkAudioAttention", "InfiniteTalkAudioProjector"]

    def __init__(self, config, accelerator):
        super().__init__(config, accelerator)
        self._audio_encoder = None
        self._audio_processor = None
        self._audio_encoder_lock = threading.Lock()

    def requires_s2v_datasets(self) -> bool:
        return True

    def supports_audio_inputs(self) -> bool:
        return True

    def requires_s2v_validation_inputs(self) -> bool:
        return True

    def conditioning_validation_dataset_type(self) -> str:
        return "video"

    def _load_audio_encoder(self) -> None:
        if self._audio_encoder is not None:
            return
        with self._audio_encoder_lock:
            if self._audio_encoder is not None:
                return
            self._audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(self.AUDIO_ENCODER_MODEL)
            self._audio_encoder = Wav2Vec2Model.from_pretrained(self.AUDIO_ENCODER_MODEL, torch_dtype=torch.float32)
            self._audio_encoder.eval().requires_grad_(False)
            self._audio_encoder.to(self.accelerator.device)

    @torch.no_grad()
    def encode_audio(self, audio_path: str, num_frames: int) -> torch.Tensor:
        self._load_audio_encoder()
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.AUDIO_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.AUDIO_SAMPLE_RATE)
        waveform = align_waveform_to_video_frames(waveform, self.AUDIO_SAMPLE_RATE, num_frames)
        processed = self._audio_processor(
            waveform.squeeze(0).cpu().numpy(),
            sampling_rate=self.AUDIO_SAMPLE_RATE,
            return_tensors="pt",
        )
        input_values = processed.input_values.to(self.accelerator.device)
        embeddings = encode_wav2vec_hidden_states(self._audio_encoder, input_values, num_frames)
        return window_audio_embeddings(embeddings)

    def prepare_batch_conditions(self, batch: dict, state: Optional[dict] = None) -> dict:
        batch = super().prepare_batch_conditions(batch, state)
        audio_paths = batch.get("s2v_audio_paths")
        if not audio_paths or any(path is None for path in audio_paths):
            raise ValueError(
                "InfiniteTalk requires one aligned audio input for every video. Configure s2v_datasets or audio.auto_split."
            )
        latent_frames = batch["latents"].shape[2]
        video_frames = (latent_frames - 1) * 4 + 1
        batch["infinitetalk_audio_hidden_states"] = torch.cat(
            [self.encode_audio(path, video_frames) for path in audio_paths],
            dim=0,
        )
        return batch

    def update_model_predict_kwargs(self, prepared_batch: dict, transformer_kwargs: dict) -> dict:
        transformer_kwargs["audio_hidden_states"] = prepared_batch["infinitetalk_audio_hidden_states"].to(
            device=prepared_batch["noisy_latents"].device,
            dtype=self.config.weight_dtype,
        )
        return transformer_kwargs

    def get_pipeline(self, pipeline_type: str = PipelineTypes.IMG2VIDEO, load_base_model: bool = True):
        pipeline = super().get_pipeline(pipeline_type, load_base_model)
        if pipeline_type == PipelineTypes.IMG2VIDEO:
            self._load_audio_encoder()
            pipeline.set_audio_encoder(self._audio_encoder, self._audio_processor)
        return pipeline

    def update_pipeline_call_kwargs(self, pipeline_kwargs: dict) -> dict:
        pipeline_kwargs = super().update_pipeline_call_kwargs(pipeline_kwargs)
        pipeline_kwargs.pop("_validation_prompt_text", None)
        pipeline_kwargs.pop("_validation_negative_prompt_text", None)
        if "num_images_per_prompt" in pipeline_kwargs:
            pipeline_kwargs["num_videos_per_prompt"] = pipeline_kwargs.pop("num_images_per_prompt")
        conditioning = pipeline_kwargs.pop("_s2v_conditioning", None)
        if conditioning is not None:
            pipeline_kwargs["audio"] = conditioning.get("audio_path")
            pipeline_kwargs["image"] = conditioning.get("image")
        return pipeline_kwargs

    def check_user_config(self):
        super().check_user_config()
        if getattr(self.config, "tread_config", None):
            raise ValueError("InfiniteTalk does not support TREAD because its audio attention is frame-local.")
        if getattr(self.config, "context_parallel_size", 1) not in (None, 1):
            raise ValueError("InfiniteTalk does not yet support context parallelism.")
        if self.config.framerate is not None:
            try:
                configured_framerate = float(self.config.framerate)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"InfiniteTalk audio alignment requires --framerate={INFINITETALK_AUDIO_FPS}.") from exc
            if configured_framerate != INFINITETALK_AUDIO_FPS:
                raise ValueError(f"InfiniteTalk audio alignment requires --framerate={INFINITETALK_AUDIO_FPS}.")
        self.config.framerate = INFINITETALK_AUDIO_FPS


ModelRegistry.register("infinitetalk", InfiniteTalk)
