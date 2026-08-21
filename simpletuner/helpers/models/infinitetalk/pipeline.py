from pathlib import Path
from typing import Any, Optional

import torch
import torchaudio
from diffusers import WanImageToVideoPipeline

from simpletuner.helpers.models.infinitetalk.audio import (
    align_waveform_to_video_frames,
    encode_wav2vec_hidden_states,
    window_audio_embeddings,
)


class InfiniteTalkPipeline(WanImageToVideoPipeline):
    def set_audio_encoder(self, audio_encoder, audio_processor) -> None:
        self.audio_encoder = audio_encoder
        self.audio_processor = audio_processor

    @torch.no_grad()
    def encode_audio(
        self,
        audio: str | Path | torch.Tensor,
        num_frames: int,
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        if not hasattr(self, "audio_encoder") or not hasattr(self, "audio_processor"):
            raise RuntimeError("InfiniteTalk validation requires a configured Wav2Vec2 audio encoder.")
        if isinstance(audio, (str, Path)):
            waveform, source_rate = torchaudio.load(str(audio))
        elif torch.is_tensor(audio):
            waveform, source_rate = audio, sample_rate
        else:
            raise TypeError(f"audio must be a path or tensor, got {type(audio).__name__}.")

        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if source_rate != sample_rate:
            waveform = torchaudio.functional.resample(waveform, source_rate, sample_rate)
        waveform = align_waveform_to_video_frames(waveform, sample_rate, num_frames)

        processed = self.audio_processor(
            waveform.squeeze(0).cpu().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        device = next(self.audio_encoder.parameters()).device
        input_values = processed.input_values.to(device)
        embeddings = encode_wav2vec_hidden_states(self.audio_encoder, input_values, num_frames)
        return window_audio_embeddings(embeddings).to(device=self._execution_device, dtype=self.transformer.dtype)

    def __call__(
        self,
        *args,
        audio: Optional[str | Path | torch.Tensor] = None,
        audio_sample_rate: int = 16000,
        num_frames: int = 81,
        attention_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        if audio is None:
            raise ValueError("InfiniteTalk validation requires an audio input.")
        audio_hidden_states = self.encode_audio(audio, num_frames, audio_sample_rate)
        attention_kwargs = dict(attention_kwargs or {})
        attention_kwargs["_infinitetalk_audio_hidden_states"] = audio_hidden_states
        return super().__call__(
            *args,
            num_frames=num_frames,
            attention_kwargs=attention_kwargs,
            **kwargs,
        )
