import torch
import torch.nn.functional as F

from simpletuner.helpers.models.infinitetalk import INFINITETALK_AUDIO_FPS


def align_waveform_to_video_frames(
    waveform: torch.Tensor,
    sample_rate: int,
    num_frames: int,
    fps: int = INFINITETALK_AUDIO_FPS,
) -> torch.Tensor:
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform with shape [channels, samples], got {tuple(waveform.shape)}.")
    if sample_rate < 1 or num_frames < 1 or fps < 1:
        raise ValueError(f"sample_rate, num_frames, and fps must be positive; got {sample_rate}, {num_frames}, and {fps}.")

    target_samples = int(round(num_frames / fps * sample_rate))
    if waveform.shape[-1] >= target_samples:
        return waveform[..., :target_samples]
    return F.pad(waveform, (0, target_samples - waveform.shape[-1]))


def interpolate_wav2vec_features(features: torch.Tensor, num_frames: int) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError(f"Expected Wav2Vec features with shape [batch, frames, channels], got {tuple(features.shape)}.")
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}.")
    return F.interpolate(features.transpose(1, 2), size=num_frames, mode="linear", align_corners=True).transpose(1, 2)


def encode_wav2vec_hidden_states(
    audio_encoder,
    input_values: torch.Tensor,
    num_frames: int,
) -> torch.Tensor:
    extract_features = audio_encoder.feature_extractor(input_values).transpose(1, 2)
    extract_features = interpolate_wav2vec_features(extract_features, num_frames)
    hidden_states, _ = audio_encoder.feature_projection(extract_features)
    encoder_outputs = audio_encoder.encoder(
        hidden_states,
        output_hidden_states=True,
        return_dict=True,
    )
    layer_states = encoder_outputs.hidden_states
    if layer_states is None or len(layer_states) < 2:
        raise ValueError("The InfiniteTalk audio encoder did not return per-layer hidden states.")
    return torch.stack(layer_states[1:], dim=2)


def window_audio_embeddings(embeddings: torch.Tensor, window_size: int = 5) -> torch.Tensor:
    if embeddings.ndim != 4:
        raise ValueError(
            f"Expected audio embeddings with shape [batch, frames, layers, channels], got {tuple(embeddings.shape)}."
        )
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(f"window_size must be a positive odd integer, got {window_size}.")

    radius = window_size // 2
    frame_count = embeddings.shape[1]
    frame_ids = torch.arange(frame_count, device=embeddings.device).unsqueeze(1)
    offsets = torch.arange(-radius, radius + 1, device=embeddings.device).unsqueeze(0)
    indices = (frame_ids + offsets).clamp_(0, frame_count - 1)
    return embeddings[:, indices]
