# SPDX-License-Identifier: Apache-2.0
# Adapted from OpenMOSS-Team/MOSS-Image-Base, revision 1ceede80badddd6a7812069619edaa5633587192.
import logging

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import apply_lora_scale
from torch import nn

from simpletuner.helpers.models.flowmap import (
    blend_flowmap_embeddings,
    clone_flowmap_embedder,
    prepare_flowmap_delta_timestep,
    register_flowmap_config,
    register_flowmap_gate_buffer,
    set_flowmap_gate,
    validate_flowmap_deltatime_type,
)
from simpletuner.helpers.musubi_block_swap import MusubiBlockSwapManager
from simpletuner.helpers.training.checkpointing import checkpoint
from simpletuner.helpers.training.gradient_checkpointing_interval import checkpoint_sequential_state, should_checkpoint_block
from simpletuner.helpers.training.offloaded_gradient_checkpointer import activation_offload_context

logger = logging.getLogger(__name__)


class _RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        return y.to(x.dtype) * self.weight


def _situ(hidden, gate):
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(hidden / 25.0))


def _validate_skip_layers(layers, depth, name="skip_layers"):
    layers = () if layers is None else tuple(layers)
    if any(type(index) is not int or not 0 <= index < depth for index in layers):
        raise ValueError(f"{name} must contain integer block indices in [0, {depth}).")
    return frozenset(layers)


def _rotate(x, rope):
    sin, cos = rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1).unsqueeze(-2)
    cos = torch.cat((cos, cos), -1).unsqueeze(-2)
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat((-b, a), -1) * sin


class MOSSImageAttentionProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn, q, k, v, rope):
        shape = (*q.shape[:2], attn.heads, q.shape[-1] // attn.heads)
        q = _rotate(q.reshape(shape), rope)
        k = _rotate(k.reshape(shape), rope)
        v = v.reshape(shape)
        with activation_offload_context(attn.offload_attention, label=f"{attn.__class__.__qualname__}:attention"):
            return dispatch_attention_fn(q.to(v.dtype), k.to(v.dtype), v, backend=self._attention_backend).flatten(2)


class MOSSImageTextBlock(nn.Module, AttentionModuleMixin):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.offload_attention = False
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.ffn_gate, self.ffn_up = nn.Linear(dim, ffn_dim), nn.Linear(dim, ffn_dim)
        self.ffn_down = nn.Linear(ffn_dim, dim)
        self.processor = MOSSImageAttentionProcessor()

    def _feed_forward(self, y):
        return self.ffn_down(_situ(self.ffn_up(y), self.ffn_gate(y)))

    def forward(self, x, rope, checkpoint_ffn=False, checkpoint_fn=checkpoint):
        q, k, v = self.qkv(self.norm1(x)).chunk(3, -1)
        q = self.q_norm(q.unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(k.unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj(self.processor(self, q, k, v, rope))
        y = self.norm2(x)
        ffn = checkpoint_fn(self._feed_forward, y, use_reentrant=False) if checkpoint_ffn else self._feed_forward(y)
        return x + ffn


class _TextProjection(nn.Module):
    def __init__(self, text_dim, dim, ffn_dim, heads):
        super().__init__()
        self.proj = nn.Linear(text_dim, dim)
        self.blocks = nn.ModuleList([MOSSImageTextBlock(dim, ffn_dim, heads) for _ in range(2)])

    def forward(self, x, rope, run_block):
        x = self.proj(x.to(self.proj.weight.dtype))
        for index, block in enumerate(self.blocks):
            x = run_block(block, index, x, rope)
        return x


class MOSSImageTransformerBlock(nn.Module, AttentionModuleMixin):
    def __init__(self, dim, ffn_dim, heads):
        super().__init__()
        self.heads = heads
        self.offload_attention = False
        self.norm1, self.norm2 = _RMSNorm(dim), _RMSNorm(dim)
        self.q_norm, self.k_norm = _RMSNorm(dim // heads), _RMSNorm(dim // heads)
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)
        self.processor = MOSSImageAttentionProcessor()

    def _feed_forward(self, y):
        hidden, gate = self.proj["ffn_up_gate"](y.to(self.proj["ffn_up_gate"].weight.dtype)).chunk(2, -1)
        return self.proj["ffn_down"](_situ(hidden, gate))

    def forward(self, x, rope, modulation, checkpoint_ffn=False, checkpoint_fn=checkpoint):
        if modulation.ndim == 3:
            modulation = modulation.unsqueeze(2)
        shift_a, scale_a, gate_a, shift_f, scale_f, gate_f = modulation.unbind(1)
        y = self.norm1(x) * (1 + scale_a) + shift_a
        dtype = self.proj["q"].weight.dtype
        y = y.to(dtype)
        q = self.q_norm(self.proj["q"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        k = self.k_norm(self.proj["k"](y).unflatten(-1, (self.heads, -1))).flatten(2)
        x = x + self.proj["o"](self.processor(self, q, k, self.proj["v"](y), rope)) * gate_a
        y = self.norm2(x) * (1 + scale_f) + shift_f
        ffn = checkpoint_fn(self._feed_forward, y, use_reentrant=False) if checkpoint_ffn else self._feed_forward(y)
        return x + ffn * gate_f


class _RoPE(nn.Module):
    def __init__(self, head_dim, base):
        super().__init__()
        self.num_bands = head_dim // 8
        self.base = base
        self.register_buffer("bands", self._default_bands(torch.device("cpu")))

    def _default_bands(self, device):
        # Match training's CPU fp32 construction, including under meta init.
        exponent = torch.arange(self.num_bands, device="cpu", dtype=torch.float32) / self.num_bands
        return (1 / (self.base**exponent)).to(device)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # .float() cannot recover frequencies already rounded by .to(bf16).
        self.bands = self._default_bands(self.bands.device)
        return self

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        # Frequencies are derived constants; retain the key for checkpoint compatibility.
        self.bands = self._default_bands(self.bands.device)

    def forward(self, height, width, text_length, device):
        # Diffusers/Accelerate can install tensors directly, bypassing load hooks.
        if self.bands.dtype != torch.float32:
            self.bands = self._default_bands(device)
        # Coordinates: (r, f, h, w); sequence order is image followed by text.
        h, w = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing="ij")
        image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_length), h, w), -1).flatten(0, 1).float()
        image[:, 2] -= (height - 1) / 2
        image[:, 3] -= (width - 1) / 2
        text = torch.zeros(text_length, 4, device=device, dtype=torch.float32)
        text[:, 1] = torch.arange(text_length, device=device)
        phase = torch.cat((image, text), 0).unsqueeze(-1) * self.bands
        return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0)


class MOSSImageTransformer(ModelMixin, ConfigMixin, PeftAdapterMixin, AttentionMixin):
    """MOSSImage DiT with training-compatible state-dict names and SDPA attention.

    Each sample is evaluated at its true text length, including text refinement;
    padded tokens cannot influence either text or joint attention.
    """

    _supports_gradient_checkpointing = True
    _supports_ffn_gradient_checkpointing = True
    _supports_attention_activation_offload = True
    _no_split_modules = ["MOSSImageTransformerBlock", "MOSSImageTextBlock"]
    _repeated_blocks = ["MOSSImageTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        dim=3072,
        ffn_dim=8192,
        num_heads=24,
        num_layers=42,
        text_dim=4096,
        freq_dim=256,
        latent_channels=128,
        rope_base=10000.0,
        enable_time_sign_embed=False,
        gate_value=None,
        deltatime_type=None,
        musubi_blocks_to_swap=0,
        musubi_block_swap_device="cpu",
    ):
        super().__init__()
        if dim % num_heads or (dim // num_heads) % 8 or freq_dim % 2:
            raise ValueError("Require dim divisible by heads, head_dim divisible by 8, and even freq_dim.")
        self.video_embedder = nn.Linear(latent_channels, dim)
        self.text_embedder = _TextProjection(text_dim, dim, ffn_dim, num_heads)
        self.rope = _RoPE(dim // num_heads, rope_base)
        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.shared_block_modulation = nn.Linear(dim, 6 * dim)
        self.layers = nn.ModuleList([MOSSImageTransformerBlock(dim, ffn_dim, num_heads) for _ in range(num_layers)])
        self.final_norm_video = _RMSNorm(dim)
        self.final_linear_video = nn.Linear(dim, latent_channels, bias=False)
        self.video_head_modulation = nn.Linear(dim, 2 * dim)
        self.time_sign_embed = nn.Embedding(2, dim) if enable_time_sign_embed else None
        if self.time_sign_embed is not None:
            nn.init.zeros_(self.time_sign_embed.weight)
        self.delta_timestep_embedder = None
        self.flowmap_deltatime_type = None
        register_flowmap_gate_buffer(self)
        if deltatime_type is not None:
            self.enable_flowmap_time_conditioning(0.25 if gate_value is None else gate_value, deltatime_type)
        self.gradient_checkpointing = False
        self.gradient_checkpointing_interval = 1
        self.gradient_checkpointing_segment_stride = None
        self.gradient_checkpointing_backend = "torch"
        self._checkpoint_fn = checkpoint
        self._musubi_block_swap = MusubiBlockSwapManager.build(
            depth=num_layers, blocks_to_swap=musubi_blocks_to_swap, swap_device=musubi_block_swap_device, logger=logger
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        output_loading_info = kwargs.pop("output_loading_info", False)
        model, info = super().from_pretrained(pretrained_model_name_or_path, output_loading_info=True, **kwargs)
        if model.time_sign_embed is not None and "time_sign_embed.weight" in info["missing_keys"]:
            model.time_sign_embed = nn.Embedding(2, model.config.dim, device=model.device, dtype=model.dtype)
            nn.init.zeros_(model.time_sign_embed.weight)
        return (model, info) if output_loading_info else model

    def enable_flowmap_time_conditioning(self, gate_value=0.25, deltatime_type="r"):
        self.flowmap_deltatime_type = validate_flowmap_deltatime_type(deltatime_type, model_name="MOSS-Image")
        if self.delta_timestep_embedder is None:
            self.delta_timestep_embedder = clone_flowmap_embedder(self.time_embedding)
        set_flowmap_gate(self, gate_value)
        register_flowmap_config(self, gate_value, deltatime_type)

    def set_gradient_checkpointing_backend(self, backend):
        if backend not in ("torch", "unsloth", "torch-ffn", "unsloth-ffn"):
            raise ValueError("Unknown MOSS-Image gradient checkpointing backend.")
        self.gradient_checkpointing_backend = backend
        if backend.startswith("unsloth"):
            from simpletuner.helpers.training.offloaded_gradient_checkpointer import offloaded_checkpoint

            self._checkpoint_fn = offloaded_checkpoint
        else:
            self._checkpoint_fn = checkpoint

    def set_gradient_checkpointing_interval(self, interval):
        self.gradient_checkpointing_interval = interval

    def set_gradient_checkpointing_offload_attention(self, enabled):
        for block in (*self.text_embedder.blocks, *self.layers):
            block.offload_attention = bool(enabled)

    def set_gradient_checkpointing_segment_stride(self, stride):
        self.gradient_checkpointing_segment_stride = stride

    def _checkpoint_block(self, index):
        return torch.is_grad_enabled() and should_checkpoint_block(
            index,
            self.gradient_checkpointing,
            self.gradient_checkpointing_interval,
            self.gradient_checkpointing_segment_stride,
        )

    def _run_block(self, block, index, *args):
        checkpointed = self._checkpoint_block(index)
        if self.gradient_checkpointing_backend.endswith("-ffn"):
            return block(*args, checkpoint_ffn=checkpointed, checkpoint_fn=self._checkpoint_fn)
        if checkpointed:
            return self._checkpoint_fn(block, *args, use_reentrant=False)
        return block(*args)

    def _time_embedding(self, times, embedder):
        half = self.config.freq_dim // 2
        dtype = torch.float32 if times.device.type == "mps" else torch.float64
        frequencies = 10000 ** (-torch.arange(half, device=times.device, dtype=dtype) / half)
        phase = times.to(dtype)[..., None] * frequencies
        sinusoid = torch.cat((phase.cos(), phase.sin()), -1).float()
        return embedder(sinusoid.to(next(embedder.parameters()).dtype)).float()

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        return_dict=True,
        skip_layers=None,
        hidden_states_buffer=None,
        timestep_sign=None,
        r_timestep=None,
        attention_kwargs=None,
    ):
        skip_layers = _validate_skip_layers(skip_layers, len(self.layers))
        if hidden_states.ndim != 4 or hidden_states.shape[1] != self.config.latent_channels:
            raise ValueError("hidden_states must be [B, latent_channels, H, W].")
        batch, _, height, width = hidden_states.shape
        if encoder_hidden_states.ndim != 3 or encoder_hidden_states.shape[0] != batch:
            raise ValueError("encoder_hidden_states must be [B, S, text_dim].")
        mask = encoder_attention_mask
        if mask is None:
            mask = torch.ones(encoder_hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)
        if mask.shape != encoder_hidden_states.shape[:2] or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError("encoder_attention_mask must be a binary [B, S] mask.")
        times = torch.as_tensor(timestep, device=hidden_states.device, dtype=torch.float32)
        tokenwise = times.ndim == 2
        if tokenwise:
            if times.shape != (batch, height * width):
                raise ValueError("Tokenwise timesteps must have shape [B, H * W].")
        else:
            times = times.reshape(-1).expand(batch)
        time = self._time_embedding(times, self.time_embedding)
        if r_timestep is not None:
            if self.delta_timestep_embedder is None:
                raise ValueError("Enable FlowMap time conditioning before providing r_timestep.")
            delta_time = prepare_flowmap_delta_timestep(
                times, r_timestep, self.flowmap_deltatime_type, model_name="MOSS-Image"
            )
            time = blend_flowmap_embeddings(
                time, self._time_embedding(delta_time, self.delta_timestep_embedder), self.flowmap_delta_emb_gate
            )
        if timestep_sign is not None:
            if self.time_sign_embed is None:
                raise ValueError("TwinFlow signed time requires enable_time_sign_embed=True.")
            sign = torch.as_tensor(timestep_sign, device=times.device)
            if tokenwise and sign.ndim == 1:
                sign = sign[:, None]
            sign = torch.broadcast_to(sign, times.shape)
            time = time + self.time_sign_embed((sign < 0).long()).to(time)
        mod = self.shared_block_modulation(time.to(self.shared_block_modulation.weight.dtype))
        head = self.video_head_modulation(time.to(self.video_head_modulation.weight.dtype))
        if tokenwise:
            mod = mod.reshape(batch, height * width, 6, -1).transpose(1, 2)
            head = head.reshape(batch, height * width, 2, -1).transpose(1, 2)
        else:
            mod, head = mod.reshape(batch, 6, -1), head.reshape(batch, 2, -1)
        manager = self._musubi_block_swap
        swapping = manager is not None and manager.activate(self.layers, hidden_states.device, torch.is_grad_enabled())
        captured = {}
        capture_layers = getattr(hidden_states_buffer, "capture_layers", None)
        outputs = []
        for index in range(batch):
            context = encoder_hidden_states[index : index + 1, mask[index].bool()]
            if context.shape[1] == 0:
                raise ValueError("Each sample needs at least one valid text token.")
            image = hidden_states[index : index + 1].flatten(2).transpose(1, 2)
            image = self.video_embedder(image.to(self.video_embedder.weight.dtype).contiguous())
            rope = self.rope(height, width, context.shape[1], hidden_states.device).to(image.dtype)
            text = self.text_embedder(context, rope[:, height * width :], self._run_block)
            x = torch.cat((image, text), 1)
            modulation = mod[index : index + 1]
            if tokenwise:
                text_modulation = modulation.mean(dim=2, keepdim=True).expand(-1, -1, context.shape[1], -1)
                modulation = torch.cat((modulation, text_modulation), dim=2)
            active = [(i, block) for i, block in enumerate(self.layers) if i not in skip_layers]
            segmented = (
                torch.is_grad_enabled()
                and self.gradient_checkpointing
                and (self.gradient_checkpointing_interval or 1) > 1
                and not self.gradient_checkpointing_backend.endswith("-ffn")
                and not swapping
                and hidden_states_buffer is None
            )
            if segmented:

                def run_segment_block(_index, entry, value, rotation, conditioning):
                    return entry[1](value, rotation, conditioning), rotation, conditioning

                x, _, _ = checkpoint_sequential_state(
                    active,
                    self.gradient_checkpointing_interval,
                    (x, rope, modulation),
                    run_segment_block,
                    self._checkpoint_fn,
                    {"use_reentrant": False},
                    self.gradient_checkpointing_segment_stride,
                )
            else:
                for layer_index, layer in active:
                    managed = swapping and manager.is_managed_block(layer_index)
                    if managed:
                        manager.stream_in(
                            layer,
                            hidden_states.device,
                            checkpointed=(
                                self._checkpoint_block(layer_index)
                                and not self.gradient_checkpointing_backend.endswith("-ffn")
                            ),
                        )
                    x = self._run_block(layer, layer_index, x, rope, modulation)
                    if hidden_states_buffer is not None and (capture_layers is None or layer_index in capture_layers):
                        captured.setdefault(layer_index, []).append(x[:, : height * width])
                    if managed:
                        manager.stream_out(layer)
            x = self.final_norm_video(x[:, : height * width].to(self.final_norm_video.weight.dtype))
            offset, scale = head[index : index + 1].unbind(1)
            if not tokenwise:
                offset, scale = offset[:, None], scale[:, None]
            x = x * (1 + scale) + offset
            x = self.final_linear_video(x.to(self.final_linear_video.weight.dtype))
            outputs.append(x.transpose(1, 2).reshape(1, -1, height, width))
        if hidden_states_buffer is not None:
            for layer_index, rows in captured.items():
                hidden_states_buffer[f"layer_{layer_index}"] = torch.cat(rows)
        sample = torch.cat(outputs)
        return Transformer2DModelOutput(sample=sample) if return_dict else (sample,)
