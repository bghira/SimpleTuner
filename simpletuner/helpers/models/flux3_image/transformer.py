"""Image transformer with independent image, reference, and text preparation streams."""

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import apply_lora_scale
from torch import nn

from simpletuner.helpers.models.flux3_image import text_ids
from simpletuner.helpers.training.checkpointing import checkpoint
from simpletuner.helpers.training.gradient_checkpointing_interval import should_checkpoint_block


class Flux3ImageRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value):
        value = value.float()
        return value * (value.square().mean(-1, keepdim=True) + self.eps).rsqrt() * self.weight.float()


class Flux3ImageRotaryEmbedding(nn.Module):
    def __init__(self, axes_dims: tuple[int, ...], theta: float):
        super().__init__()
        self.axes_dims = axes_dims
        self.theta = theta

    def forward(self, ids):
        phases = []
        for axis, width in enumerate(self.axes_dims):
            dtype = torch.float32 if ids.device.type == "mps" else torch.float64
            frequencies = self.theta ** (-torch.arange(0, width, 2, device=ids.device, dtype=dtype) / width)
            phases.append(ids[..., axis, None].to(dtype) * frequencies)
        phase = torch.cat(phases, dim=-1)
        return phase.cos().float().unsqueeze(-2), phase.sin().float().unsqueeze(-2)


def apply_rotary(value, rotary):
    cosine, sine = rotary
    paired = value.float().unflatten(-1, (-1, 2))
    even, odd = paired.unbind(-1)
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


class Flux3ImageAttentionProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn, hidden_states, rotary, attention_mask=None):
        query = attn.to_q(hidden_states).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(-1, (attn.heads, -1))
        query = apply_rotary(attn.norm_q(query), rotary).to(value.dtype)
        key = apply_rotary(attn.norm_k(key), rotary).to(value.dtype)
        attended = dispatch_attention_fn(query, key, value, attn_mask=attention_mask, backend=self._attention_backend)
        return attn.to_out[0](attended.flatten(-2))


class Flux3ImageAttention(nn.Module, AttentionModuleMixin):
    def __init__(self, width: int, heads: int, eps: float):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(width, width, bias=False)
        self.to_v = nn.Linear(width, width, bias=False)
        self.norm_q = Flux3ImageRMSNorm(width // heads, eps)
        self.norm_k = Flux3ImageRMSNorm(width // heads, eps)
        self.to_out = nn.ModuleList([nn.Linear(width, width, bias=False)])
        self.processor = Flux3ImageAttentionProcessor()

    def forward(self, hidden_states, rotary, attention_mask=None):
        return self.processor(self, hidden_states, rotary, attention_mask)


class Flux3ImageFeedForward(nn.Module):
    def __init__(self, width: int, mlp_ratio: float):
        super().__init__()
        intermediate = int(width * mlp_ratio)
        self.linear_in = nn.Linear(width, intermediate * 2, bias=False)
        self.linear_out = nn.Linear(intermediate, width, bias=False)

    def forward(self, hidden_states):
        gate, value = self.linear_in(hidden_states).chunk(2, dim=-1)
        return self.linear_out(F.silu(gate) * value)


class Flux3ImageModulation(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear = nn.Linear(width, width * 3, bias=False)

    def forward(self, embedding):
        return self.linear(F.silu(embedding))


class Flux3ImageTransformerBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_ratio: float, eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(width, eps=eps, elementwise_affine=False)
        self.attn = Flux3ImageAttention(width, heads, eps)
        self.ff = Flux3ImageFeedForward(width, mlp_ratio)

    def forward(self, hidden_states, modulation, rotary, attention_mask=None):
        shift, scale, gate = modulation.chunk(3, dim=-1)
        normalized = self.norm(hidden_states) * (1 + scale) + shift
        return hidden_states + gate * (self.attn(normalized, rotary, attention_mask) + self.ff(normalized))


class Flux3ImageOutputNorm(nn.Module):
    def __init__(self, width: int, eps: float):
        super().__init__()
        self.linear = nn.Linear(width, width * 2, bias=False)
        self.norm = nn.LayerNorm(width, eps=eps, elementwise_affine=False)

    def forward(self, hidden_states, embedding):
        scale, shift = self.linear(F.silu(embedding)).chunk(2, dim=-1)
        return self.norm(hidden_states) * (1 + scale) + shift


class Flux3ImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, AttentionMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["Flux3ImageTransformerBlock"]
    _repeated_blocks = ["Flux3ImageTransformerBlock"]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        output_loading_info = kwargs.pop("output_loading_info", False)
        model, info = super().from_pretrained(pretrained_model_name_or_path, output_loading_info=True, **kwargs)
        if model.time_sign_embed is not None and "time_sign_embed.weight" in info["missing_keys"]:
            model.time_sign_embed = nn.Embedding(2, model.inner_dim, device=model.device, dtype=model.dtype)
            nn.init.zeros_(model.time_sign_embed.weight)
        return (model, info) if output_loading_info else model

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        cond_in_channels: int = 128,
        joint_attention_dim: int = 20480,
        num_attention_heads: int = 24,
        attention_head_dim: int = 128,
        num_layers: int = 5,
        num_single_layers: int = 28,
        mlp_ratio: float = 3.0,
        axes_dims_rope: tuple[int, ...] = (32, 32, 32, 32),
        rope_theta: float = 10000,
        timestep_channels: int = 256,
        eps: float = 1e-6,
        flow_map: bool = False,
        enable_time_sign_embed: bool = False,
    ):
        super().__init__()
        if sum(axes_dims_rope) != attention_head_dim or any(width % 2 for width in axes_dims_rope):
            raise ValueError("FLUX.3 image RoPE axes must be even and sum to the attention head width.")
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pos_embed = Flux3ImageRotaryEmbedding(tuple(axes_dims_rope), rope_theta)
        self.time_proj = Timesteps(timestep_channels, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(timestep_channels, self.inner_dim, sample_proj_bias=False)
        self.target_timestep_embedder = (
            TimestepEmbedding(timestep_channels, self.inner_dim, sample_proj_bias=False) if flow_map else None
        )
        self.time_sign_embed = nn.Embedding(2, self.inner_dim) if enable_time_sign_embed else None
        if self.time_sign_embed is not None:
            nn.init.zeros_(self.time_sign_embed.weight)
        self.x_embedders = nn.ModuleDict(
            {
                "image": nn.Linear(in_channels, self.inner_dim, bias=False),
                "image_cond": nn.Linear(cond_in_channels, self.inner_dim, bias=False),
            }
        )
        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim, bias=False)
        self.early_stream_modulations = nn.ModuleDict(
            {mode: Flux3ImageModulation(self.inner_dim) for mode in ("image", "image_cond", "txt")}
        )
        self.single_stream_modulations = nn.ModuleDict(
            {mode: Flux3ImageModulation(self.inner_dim) for mode in ("image", "image_cond", "txt")}
        )

        def blocks(count):
            return nn.ModuleList(
                [Flux3ImageTransformerBlock(self.inner_dim, num_attention_heads, mlp_ratio, eps) for _ in range(count)]
            )

        self.mode_blocks = nn.ModuleDict({"image": blocks(num_layers), "image_cond": blocks(num_layers)})
        self.txt_mode_blocks = blocks(num_layers)
        self.single_transformer_blocks = blocks(num_single_layers)
        self.norm_out = Flux3ImageOutputNorm(self.inner_dim, eps)
        self.proj_out = nn.Linear(self.inner_dim, in_channels, bias=False)
        self.gradient_checkpointing = False
        self.gradient_checkpointing_interval = 1
        self.gradient_checkpointing_segment_stride = None
        self._checkpoint_fn = checkpoint

    def set_gradient_checkpointing_backend(self, backend):
        if backend == "torch":
            self._checkpoint_fn = checkpoint
        elif backend == "unsloth":
            from simpletuner.helpers.training.offloaded_gradient_checkpointer import offloaded_checkpoint

            self._checkpoint_fn = offloaded_checkpoint
        else:
            raise ValueError("FLUX.3 supports torch and unsloth whole-block gradient checkpointing.")

    def set_gradient_checkpointing_interval(self, interval):
        self.gradient_checkpointing_interval = interval

    def set_gradient_checkpointing_segment_stride(self, stride):
        self.gradient_checkpointing_segment_stride = stride

    def _run_block(self, block, index, hidden_states, modulation, rotary, mask):
        if torch.is_grad_enabled() and should_checkpoint_block(
            index,
            self.gradient_checkpointing,
            self.gradient_checkpointing_interval,
            self.gradient_checkpointing_segment_stride,
        ):
            return self._checkpoint_fn(block, hidden_states, modulation, rotary, mask, use_reentrant=False)
        return block(hidden_states, modulation, rotary, mask)

    def _time_embedding(self, timestep, dtype):
        shape = timestep.shape
        projected = self.time_proj(timestep.flatten().to(dtype) * 1000).to(dtype)
        return self.timestep_embedder(projected).reshape(*shape, self.inner_dim)

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        img_ids=None,
        txt_ids=None,
        conditioning_hidden_states=None,
        conditioning_ids=None,
        conditioning_attention_mask=None,
        encoder_attention_mask=None,
        timestep_sign=None,
        target_timestep=None,
        hidden_states_buffer=None,
        attention_kwargs=None,
        return_dict=True,
    ):
        batch, image_length, _ = hidden_states.shape
        if timestep.ndim == 0:
            timestep = timestep.expand(batch)
        if timestep.ndim == 1:
            timestep = timestep[:, None]
        image_embedding = self._time_embedding(timestep, hidden_states.dtype)
        clean_embedding = self._time_embedding(
            torch.zeros(batch, 1, device=timestep.device, dtype=timestep.dtype), hidden_states.dtype
        )
        if self.target_timestep_embedder is not None:
            if target_timestep is None:
                raise ValueError("FLUX.3 distilled flow-map prediction requires a destination target_timestep.")
            target = torch.as_tensor(target_timestep, device=timestep.device).to(image_embedding.dtype).flatten()
            if target.numel() == 1:
                target = target.expand(batch)
            if target.numel() != batch:
                raise ValueError("FLUX.3 distilled flow-map destination times must be scalar or one per image.")
            projected = self.time_proj(target.flatten() * 1000).to(image_embedding.dtype)
            destination = self.target_timestep_embedder(projected).reshape(batch, -1, self.inner_dim)
            image_embedding = image_embedding + destination
            clean_embedding = clean_embedding + destination
        elif target_timestep is not None:
            raise ValueError("Destination timesteps require a FLUX.3 flow-map checkpoint.")
        if timestep_sign is not None:
            if self.time_sign_embed is None:
                raise ValueError("TwinFlow signed time requires enable_time_sign_embed=True.")
            sign = torch.as_tensor(timestep_sign, device=timestep.device)
            if sign.numel() == 1:
                sign = sign.reshape(1).expand(batch)
            sign = sign.reshape(batch, -1)
            image_embedding = image_embedding + self.time_sign_embed((sign < 0).long()).to(image_embedding.dtype)
        if img_ids is None:
            raise ValueError("FLUX.3 image prediction requires spatial position IDs.")
        if txt_ids is None:
            txt_ids = text_ids(encoder_hidden_states.shape[1], hidden_states.device)
        text_length = encoder_hidden_states.shape[1]
        streams = {"txt": self.context_embedder(encoder_hidden_states), "image": self.x_embedders.image(hidden_states)}
        positions = {"txt": txt_ids, "image": img_ids}
        embeddings = {"txt": clean_embedding, "image": image_embedding}
        stages = {"txt": self.txt_mode_blocks, "image": self.mode_blocks.image}
        masks = {}
        if encoder_attention_mask is not None:
            masks["txt"] = encoder_attention_mask[:, None, None, :].bool()
        if conditioning_hidden_states is not None:
            if conditioning_ids is None:
                raise ValueError("Reference image tokens require reference position IDs.")
            streams["image_cond"] = self.x_embedders.image_cond(conditioning_hidden_states)
            positions["image_cond"] = conditioning_ids
            embeddings["image_cond"] = clean_embedding
            stages["image_cond"] = self.mode_blocks.image_cond
            if conditioning_attention_mask is not None:
                masks["image_cond"] = conditioning_attention_mask[:, None, None, :].bool()
        for mode in streams:
            rotary = self.pos_embed(positions[mode])
            modulation = getattr(self.early_stream_modulations, mode)(embeddings[mode])
            for index, block in enumerate(stages[mode]):
                streams[mode] = self._run_block(block, index, streams[mode], modulation, rotary, masks.get(mode))
                if mode == "image" and hidden_states_buffer is not None:
                    hidden_states_buffer[f"layer_{index}"] = streams[mode]
        lengths = [value.shape[1] for value in streams.values()]
        joint = torch.cat(list(streams.values()), dim=1)
        if any(value.ndim == 3 for value in positions.values()):
            positions = {
                mode: value.expand(batch, -1, -1) if value.ndim == 2 else value for mode, value in positions.items()
            }
        joint_ids = torch.cat(list(positions.values()), dim=-2)
        rotary = self.pos_embed(joint_ids)
        modulation = torch.cat(
            [
                getattr(self.single_stream_modulations, mode)(embeddings[mode]).expand(-1, length, -1)
                for mode, length in zip(streams, lengths)
            ],
            dim=1,
        )
        mask = None
        if masks:
            valid = torch.cat(
                [
                    (
                        masks[mode][:, 0, 0]
                        if mode in masks
                        else torch.ones(batch, length, device=joint.device, dtype=torch.bool)
                    )
                    for mode, length in zip(streams, lengths)
                ],
                dim=1,
            )
            mask = valid[:, None, None, :]
        for index, block in enumerate(self.single_transformer_blocks):
            joint = self._run_block(block, index, joint, modulation, rotary, mask)
            if hidden_states_buffer is not None:
                hidden_states_buffer[f"layer_{self.config.num_layers + index}"] = joint[
                    :, text_length : text_length + image_length
                ]
        generated = joint[:, text_length : text_length + image_length]
        prediction = self.proj_out(self.norm_out(generated, image_embedding))
        if not return_dict:
            return (prediction,)
        return Transformer2DModelOutput(sample=prediction)
