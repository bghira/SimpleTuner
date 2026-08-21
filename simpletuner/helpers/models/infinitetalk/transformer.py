from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.normalization import FP32LayerNorm
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from simpletuner.helpers.models.infinitetalk import (
    INFINITETALK_AUDIO_DIM,
    INFINITETALK_AUDIO_LAYERS,
    INFINITETALK_AUDIO_WINDOW,
    INFINITETALK_VAE_TEMPORAL_SCALE,
)
from simpletuner.helpers.models.wan.transformer import WanTransformer3DModel, WanTransformerBlock
from simpletuner.helpers.training.offloaded_gradient_checkpointer import activation_offload_context


class InfiniteTalkAudioAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int = INFINITETALK_AUDIO_DIM,
        num_heads: int = 40,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads}).")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_linear = nn.Linear(dim, dim, bias=True)
        self.kv_linear = nn.Linear(encoder_hidden_states_dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.eps = eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        batch_size, sequence_length, dim = hidden_states.shape
        if num_frames < 1 or sequence_length % num_frames != 0:
            raise ValueError(
                f"Video token count {sequence_length} must be divisible by num_frames={num_frames} for audio attention."
            )
        if encoder_hidden_states.shape[0] != batch_size * num_frames:
            raise ValueError(
                "InfiniteTalk audio frame count does not match the video: "
                f"expected {batch_size * num_frames}, got {encoder_hidden_states.shape[0]}."
            )

        tokens_per_frame = sequence_length // num_frames
        frame_hidden_states = hidden_states.reshape(batch_size * num_frames, tokens_per_frame, dim)
        query = self.q_linear(frame_hidden_states).unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        key, value = self.kv_linear(encoder_hidden_states).chunk(2, dim=-1)
        key = key.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        value = value.unflatten(-1, (self.num_heads, self.head_dim)).transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        attended = attended.transpose(1, 2).flatten(2)
        return self.proj(attended).reshape(batch_size, sequence_length, dim)


class InfiniteTalkAudioProjector(nn.Module):
    def __init__(
        self,
        audio_window: int = INFINITETALK_AUDIO_WINDOW,
        vae_scale: int = INFINITETALK_VAE_TEMPORAL_SCALE,
        audio_layers: int = INFINITETALK_AUDIO_LAYERS,
        audio_dim: int = INFINITETALK_AUDIO_DIM,
        intermediate_dim: int = 512,
        output_dim: int = INFINITETALK_AUDIO_DIM,
        context_tokens: int = 32,
    ) -> None:
        super().__init__()
        self.audio_window = audio_window
        self.vae_scale = vae_scale
        self.audio_layers = audio_layers
        self.audio_dim = audio_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim
        self.proj1 = nn.Linear(audio_window * audio_layers * audio_dim, intermediate_dim)
        self.proj1_vf = nn.Linear((audio_window + vae_scale - 1) * audio_layers * audio_dim, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, audio_hidden_states: torch.Tensor) -> torch.Tensor:
        if audio_hidden_states.ndim != 5:
            raise ValueError(
                "Expected windowed audio with shape [batch, video_frames, window, layers, channels], "
                f"got {tuple(audio_hidden_states.shape)}."
            )
        batch_size, video_frames, window, layers, channels = audio_hidden_states.shape
        if (window, layers, channels) != (self.audio_window, self.audio_layers, self.audio_dim):
            raise ValueError(
                "InfiniteTalk audio geometry mismatch: expected "
                f"({self.audio_window}, {self.audio_layers}, {self.audio_dim}), got {(window, layers, channels)}."
            )
        if (video_frames - 1) % self.vae_scale != 0:
            raise ValueError(
                f"InfiniteTalk requires video_frames = 4k + 1; received {video_frames} with vae_scale={self.vae_scale}."
            )

        first_frame = audio_hidden_states[:, :1].flatten(2)
        projected = [F.relu(self.proj1(first_frame))]

        latent_tail_frames = (video_frames - 1) // self.vae_scale
        if latent_tail_frames:
            tail = audio_hidden_states[:, 1:].reshape(
                batch_size,
                latent_tail_frames,
                self.vae_scale,
                self.audio_window,
                self.audio_layers,
                self.audio_dim,
            )
            middle = self.audio_window // 2
            first = tail[:, :, :1, : middle + 1].flatten(2, 3)
            center = tail[:, :, 1:-1, middle : middle + 1].flatten(2, 3)
            last = tail[:, :, -1:, middle:].flatten(2, 3)
            tail_context = torch.cat([first, center, last], dim=2).flatten(2)
            projected.append(F.relu(self.proj1_vf(tail_context)))

        context = torch.cat(projected, dim=1)
        context = F.relu(self.proj2(context))
        context = self.proj3(context).reshape(batch_size, -1, self.context_tokens, self.output_dim)
        context = self.norm(context.float()).to(dtype=audio_hidden_states.dtype)
        return context.flatten(0, 1)


class InfiniteTalkTransformerBlock(WanTransformerBlock):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        added_kv_proj_dim: Optional[int] = None,
        audio_output_dim: int = INFINITETALK_AUDIO_DIM,
    ) -> None:
        super().__init__(dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim)
        self.audio_cross_attn = InfiniteTalkAudioAttention(dim, audio_output_dim, num_heads, eps)
        self.norm_x = FP32LayerNorm(dim, eps, elementwise_affine=True)

    def _ensure_module_dtype(self, device: torch.device, dtype: torch.dtype) -> None:
        super()._ensure_module_dtype(device, dtype)
        self.audio_cross_attn.to(device=device, dtype=dtype)
        self.norm_x.to(device=device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        audio_hidden_states: Optional[torch.Tensor] = None,
        num_frames: Optional[int] = None,
        checkpoint_ffn: bool = False,
        checkpoint_fn=None,
        offload_attention: bool = False,
    ) -> torch.Tensor:
        if audio_hidden_states is None or num_frames is None:
            raise ValueError("InfiniteTalk transformer blocks require audio_hidden_states and num_frames.")
        self._ensure_module_dtype(hidden_states.device, hidden_states.dtype)

        temb = temb.to(device=self.scale_shift_table.device, dtype=self.scale_shift_table.dtype, non_blocking=True)
        if temb.ndim == 3:
            modulation = self.scale_shift_table + temb
            chunk_dim = 1
        elif temb.ndim == 4:
            modulation = self.scale_shift_table.unsqueeze(1) + temb
            chunk_dim = 2
        else:
            raise ValueError(f"InfiniteTalkTransformerBlock expected temb with 3 or 4 dims, got {tuple(temb.shape)}.")
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = modulation.chunk(6, dim=chunk_dim)
        if chunk_dim == 2:
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [
                value.squeeze(2) for value in (shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa)
            ]

        norm_hidden_states = self.norm1(hidden_states) * (1 + scale_msa) + shift_msa
        with activation_offload_context(offload_attention, label=f"{self.__class__.__qualname__}:attention"):
            hidden_states = hidden_states + self.attn1(hidden_states=norm_hidden_states, rotary_emb=rotary_emb) * gate_msa
            hidden_states = hidden_states + self.attn2(
                hidden_states=self.norm2(hidden_states),
                encoder_hidden_states=encoder_hidden_states,
            )
            hidden_states = hidden_states + self.audio_cross_attn(
                self.norm_x(hidden_states),
                audio_hidden_states,
                num_frames,
            )

        norm_hidden_states = self.norm3(hidden_states) * (1 + c_scale_msa) + c_shift_msa
        if checkpoint_ffn:
            if checkpoint_fn is None:
                raise ValueError("checkpoint_fn is required when checkpoint_ffn=True")
            ff_output = checkpoint_fn(self._run_feed_forward, norm_hidden_states, use_reentrant=False)
        else:
            ff_output = self._run_feed_forward(norm_hidden_states)
        return hidden_states + ff_output * c_gate_msa


class InfiniteTalkTransformer3DModel(WanTransformer3DModel):
    INFINITETALK_REPO = "MeiGen-AI/InfiniteTalk"
    INFINITETALK_WEIGHT = "single/infinitetalk.safetensors"
    _no_split_modules = ["InfiniteTalkTransformerBlock"]
    _repeated_blocks = ["InfiniteTalkTransformerBlock"]
    _keep_in_fp32_modules = [*WanTransformer3DModel._keep_in_fp32_modules, "norm_x", "audio_proj.norm"]

    def __init__(
        self,
        patch_size: Tuple[int] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 36,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: Optional[str] = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: Optional[int] = 1280,
        added_kv_proj_dim: Optional[int] = None,
        rope_max_seq_len: int = 1024,
        feed_forward_chunk_size: Optional[int] = None,
        feed_forward_chunk_dim: int = 0,
        musubi_blocks_to_swap: int = 0,
        musubi_block_swap_device: str = "cpu",
        enable_time_sign_embed: bool = False,
        gate_value: Optional[float] = None,
        deltatime_type: Optional[str] = None,
        audio_window: int = INFINITETALK_AUDIO_WINDOW,
        audio_layers: int = INFINITETALK_AUDIO_LAYERS,
        audio_dim: int = INFINITETALK_AUDIO_DIM,
        audio_intermediate_dim: int = 512,
        audio_output_dim: int = INFINITETALK_AUDIO_DIM,
        audio_context_tokens: int = 32,
        vae_temporal_scale: int = INFINITETALK_VAE_TEMPORAL_SCALE,
    ) -> None:
        super().__init__(
            patch_size=patch_size,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            text_dim=text_dim,
            freq_dim=freq_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            cross_attn_norm=cross_attn_norm,
            qk_norm=qk_norm,
            eps=eps,
            image_dim=image_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            rope_max_seq_len=rope_max_seq_len,
            feed_forward_chunk_size=feed_forward_chunk_size,
            feed_forward_chunk_dim=feed_forward_chunk_dim,
            musubi_blocks_to_swap=musubi_blocks_to_swap,
            musubi_block_swap_device=musubi_block_swap_device,
            enable_time_sign_embed=enable_time_sign_embed,
            gate_value=gate_value,
            deltatime_type=deltatime_type,
        )
        self.register_to_config(
            audio_window=audio_window,
            audio_layers=audio_layers,
            audio_dim=audio_dim,
            audio_intermediate_dim=audio_intermediate_dim,
            audio_output_dim=audio_output_dim,
            audio_context_tokens=audio_context_tokens,
            vae_temporal_scale=vae_temporal_scale,
        )
        inner_dim = num_attention_heads * attention_head_dim
        self.blocks = nn.ModuleList(
            [
                InfiniteTalkTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                    added_kv_proj_dim,
                    audio_output_dim,
                )
                for _ in range(num_layers)
            ]
        )
        self.audio_proj = InfiniteTalkAudioProjector(
            audio_window=audio_window,
            vae_scale=vae_temporal_scale,
            audio_layers=audio_layers,
            audio_dim=audio_dim,
            intermediate_dim=audio_intermediate_dim,
            output_dim=audio_output_dim,
            context_tokens=audio_context_tokens,
        )
        if feed_forward_chunk_size is not None:
            self.set_chunk_feed_forward(feed_forward_chunk_size, feed_forward_chunk_dim)

    def _expected_delta_keys(self) -> set[str]:
        return {
            name
            for name in self.state_dict()
            if name.startswith("audio_proj.") or ".audio_cross_attn." in name or ".norm_x." in name
        }

    @staticmethod
    def _set_parameter(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
        target = module
        parts = name.split(".")
        for part in parts[:-1]:
            target = getattr(target, part)
        current = getattr(target, parts[-1])
        requires_grad = current.requires_grad if isinstance(current, nn.Parameter) else True
        setattr(target, parts[-1], nn.Parameter(tensor, requires_grad=requires_grad))

    def load_audio_conditioning_weights(self, delta_path: str) -> None:
        expected = self._expected_delta_keys()
        with safe_open(delta_path, framework="pt", device="cpu") as checkpoint:
            available = set(checkpoint.keys())
            missing = expected - available
            if missing:
                preview = ", ".join(sorted(missing)[:5])
                raise ValueError(f"InfiniteTalk checkpoint is missing required tensors: {preview}")
            state = self.state_dict()
            for name in sorted(expected):
                tensor = checkpoint.get_tensor(name)
                target = state[name]
                if tuple(tensor.shape) != tuple(target.shape):
                    raise ValueError(
                        f"InfiniteTalk tensor {name} has shape {tuple(tensor.shape)}; expected {tuple(target.shape)}."
                    )
                self._set_parameter(self, name, tensor.to(dtype=target.dtype))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        delta_path = kwargs.pop("infinitetalk_delta_path", None)
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        if delta_path is None:
            delta_path = hf_hub_download(cls.INFINITETALK_REPO, cls.INFINITETALK_WEIGHT)
        model.load_audio_conditioning_weights(delta_path)
        return model

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        audio_hidden_states: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        if audio_hidden_states is None and attention_kwargs is not None:
            audio_hidden_states = attention_kwargs.get("_infinitetalk_audio_hidden_states")
        if audio_hidden_states is None:
            raise ValueError("InfiniteTalk requires audio_hidden_states.")
        audio_context = self.audio_proj(audio_hidden_states.to(device=hidden_states.device, dtype=hidden_states.dtype))
        if attention_kwargs is not None and "_infinitetalk_audio_hidden_states" in attention_kwargs:
            attention_kwargs = dict(attention_kwargs)
            attention_kwargs.pop("_infinitetalk_audio_hidden_states")
        return super().forward(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            audio_hidden_states=audio_context,
            attention_kwargs=attention_kwargs,
            **kwargs,
        )
