from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.transformers.transformer_flux import (
    AdaLayerNormContinuous,
    FluxPosEmbed,
    FluxSingleTransformerBlock,
    FluxTransformerBlock,
)

from simpletuner.helpers.musubi_block_swap import MusubiBlockSwapManager

logger = get_logger(__name__, log_level="INFO")


def _store_hidden_state(buffer, key: str, hidden_states: torch.Tensor, image_tokens_start: int | None = None):
    if buffer is None:
        return
    if image_tokens_start is not None and hidden_states.dim() >= 3:
        buffer[key] = hidden_states[:, image_tokens_start:, ...]
    else:
        buffer[key] = hidden_states


class TimestepEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, enable_time_sign_embed: bool = False):
        super().__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        # Signed-time embedding for TwinFlow-style negative time handling.
        self.time_sign_embed: Optional[nn.Embedding] = None
        if enable_time_sign_embed:
            self.time_sign_embed = nn.Embedding(2, embedding_dim)
            nn.init.zeros_(self.time_sign_embed.weight)

    def forward(self, timestep, hidden_dtype, timestep_sign: Optional[torch.Tensor] = None):
        if timestep.ndim == 2:
            batch_size, sequence_length = timestep.shape
            flat_timestep = timestep.reshape(-1)
            timesteps_proj = self.time_proj(flat_timestep)
            temb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype)).view(batch_size, sequence_length, -1)
        else:
            timesteps_proj = self.time_proj(timestep)
            temb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))
        if timestep_sign is not None:
            if self.time_sign_embed is None:
                raise ValueError(
                    "timestep_sign was provided but the model was loaded without `enable_time_sign_embed=True`. "
                    "Enable TwinFlow (or load a TwinFlow-compatible checkpoint) to use signed-timestep conditioning."
                )
            sign_tensor = timestep_sign.to(device=temb.device)
            if timestep.ndim == 2:
                batch_size, sequence_length = timestep.shape
                if sign_tensor.ndim == 0:
                    sign_tensor = sign_tensor.expand(batch_size, sequence_length)
                elif sign_tensor.ndim == 1:
                    if sign_tensor.shape[0] == 1:
                        sign_tensor = sign_tensor.expand(batch_size)
                    elif sign_tensor.shape[0] != batch_size:
                        raise ValueError(
                            f"LongCat-Image tokenwise timestep_sign expected 1 or {batch_size} batch values, got {sign_tensor.shape[0]}."
                        )
                    sign_tensor = sign_tensor[:, None].expand(-1, sequence_length)
                elif sign_tensor.ndim == 2:
                    if sign_tensor.shape[1] != sequence_length:
                        raise ValueError(
                            f"LongCat-Image tokenwise timestep_sign expected sequence length {sequence_length}, got {sign_tensor.shape[1]}."
                        )
                    if sign_tensor.shape[0] == 1:
                        sign_tensor = sign_tensor.expand(batch_size, -1)
                    elif sign_tensor.shape[0] != batch_size:
                        raise ValueError(
                            f"LongCat-Image tokenwise timestep_sign expected batch size {batch_size}, got {sign_tensor.shape[0]}."
                        )
                else:
                    raise ValueError(
                        "LongCat-Image timestep_sign expected scalar, 1D batch tensor, or 2D tokenwise tensor, "
                        f"got shape {tuple(sign_tensor.shape)}."
                    )
                sign_idx = (sign_tensor.reshape(-1) < 0).long().to(device=temb.device)
                sign_emb = self.time_sign_embed(sign_idx).to(dtype=temb.dtype, device=temb.device)
                temb = temb + sign_emb.view(batch_size, sequence_length, -1)
            else:
                if sign_tensor.ndim == 0:
                    sign_tensor = sign_tensor.expand(temb.shape[0])
                elif sign_tensor.ndim == 1:
                    if sign_tensor.shape[0] == 1:
                        sign_tensor = sign_tensor.expand(temb.shape[0])
                    elif sign_tensor.shape[0] != temb.shape[0]:
                        raise ValueError(
                            f"LongCat-Image timestep_sign expected 1 or {temb.shape[0]} batch values, got {sign_tensor.shape[0]}."
                        )
                else:
                    raise ValueError(
                        "LongCat-Image timestep_sign expected scalar or 1D batch tensor for batchwise timesteps, "
                        f"got shape {tuple(sign_tensor.shape)}."
                    )
                sign_idx = (sign_tensor.view(-1) < 0).long().to(device=temb.device)
                temb = temb + self.time_sign_embed(sign_idx).to(dtype=temb.dtype, device=temb.device)
        return temb


def _longcat_apply_ada_layer_norm_zero(norm, hidden_states: torch.Tensor, emb: torch.Tensor):
    if emb.ndim == 2:
        return norm(hidden_states, emb=emb)

    modulation = norm.linear(norm.silu(emb))
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
    hidden_states = norm.norm(hidden_states) * (1 + scale_msa) + shift_msa
    return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp


def _longcat_apply_ada_layer_norm_zero_single(norm, hidden_states: torch.Tensor, emb: torch.Tensor):
    if emb.ndim == 2:
        return norm(hidden_states, emb=emb)

    modulation = norm.linear(norm.silu(emb))
    shift_msa, scale_msa, gate_msa = modulation.chunk(3, dim=-1)
    hidden_states = norm.norm(hidden_states) * (1 + scale_msa) + shift_msa
    return hidden_states, gate_msa


def _longcat_apply_ada_layer_norm_continuous(norm, hidden_states: torch.Tensor, conditioning_embedding: torch.Tensor):
    if conditioning_embedding.ndim == 2:
        return norm(hidden_states, conditioning_embedding)

    emb = norm.linear(norm.silu(conditioning_embedding).to(hidden_states.dtype))
    scale, shift = torch.chunk(emb, 2, dim=-1)
    return norm.norm(hidden_states) * (1 + scale) + shift


def _run_longcat_transformer_block(
    block: FluxTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    context_temb: torch.Tensor,
    image_rotary_emb,
):
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = _longcat_apply_ada_layer_norm_zero(
        block.norm1, hidden_states, temb
    )
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = _longcat_apply_ada_layer_norm_zero(
        block.norm1_context, encoder_hidden_states, context_temb
    )

    attention_outputs = block.attn(
        hidden_states=norm_hidden_states,
        encoder_hidden_states=norm_encoder_hidden_states,
        image_rotary_emb=image_rotary_emb,
    )
    if len(attention_outputs) == 2:
        attn_output, context_attn_output = attention_outputs
    elif len(attention_outputs) == 3:
        attn_output, context_attn_output, ip_attn_output = attention_outputs

    if gate_msa.ndim == 2:
        gate_msa = gate_msa.unsqueeze(1)
    hidden_states = hidden_states + gate_msa * attn_output

    norm_hidden_states = block.norm2(hidden_states)
    if scale_mlp.ndim == 2:
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    else:
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp
    ff_output = block.ff(norm_hidden_states)
    if gate_mlp.ndim == 2:
        gate_mlp = gate_mlp.unsqueeze(1)
    hidden_states = hidden_states + gate_mlp * ff_output
    if len(attention_outputs) == 3:
        hidden_states = hidden_states + ip_attn_output

    if c_gate_msa.ndim == 2:
        c_gate_msa = c_gate_msa.unsqueeze(1)
    encoder_hidden_states = encoder_hidden_states + c_gate_msa * context_attn_output

    norm_encoder_hidden_states = block.norm2_context(encoder_hidden_states)
    if c_scale_mlp.ndim == 2:
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
    else:
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp) + c_shift_mlp
    context_ff_output = block.ff_context(norm_encoder_hidden_states)
    if c_gate_mlp.ndim == 2:
        c_gate_mlp = c_gate_mlp.unsqueeze(1)
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp * context_ff_output

    if encoder_hidden_states.dtype == torch.float16:
        encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)
    return encoder_hidden_states, hidden_states


def _run_longcat_single_transformer_block(
    block: FluxSingleTransformerBlock,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    image_rotary_emb,
):
    text_seq_len = encoder_hidden_states.shape[1]
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    residual = hidden_states
    norm_hidden_states, gate = _longcat_apply_ada_layer_norm_zero_single(block.norm, hidden_states, temb)
    mlp_hidden_states = block.act_mlp(block.proj_mlp(norm_hidden_states))
    attn_output = block.attn(hidden_states=norm_hidden_states, image_rotary_emb=image_rotary_emb)

    hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
    if gate.ndim == 2:
        gate = gate.unsqueeze(1)
    hidden_states = gate * block.proj_out(hidden_states)
    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16:
        hidden_states = hidden_states.clip(-65504, 65504)

    encoder_hidden_states, hidden_states = hidden_states[:, :text_seq_len], hidden_states[:, text_seq_len:]
    return encoder_hidden_states, hidden_states


class LongCatImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    The Transformer model introduced in Flux, adapted for LongCat-Image.
    """

    _supports_gradient_checkpointing = True
    _cp_plan = {
        "": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "encoder_hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
            "img_ids": ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
            "txt_ids": ContextParallelInput(split_dim=0, expected_dims=2, split_output=False),
        },
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
    }

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 3584,
        pooled_projection_dim: int = 3584,
        axes_dims_rope: List[int] = [16, 56, 56],
        enable_time_sign_embed: bool = False,
        musubi_blocks_to_swap: int = 0,
        musubi_block_swap_device: str = "cpu",
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.pooled_projection_dim = pooled_projection_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)
        self.time_embed = TimestepEmbeddings(embedding_dim=self.inner_dim, enable_time_sign_embed=enable_time_sign_embed)

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = torch.nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                FluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                FluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False
        self.initialize_weights()

        self.use_checkpoint = [True] * num_layers
        self.use_single_checkpoint = [True] * num_single_layers

        total_layers = num_layers + num_single_layers
        self._musubi_block_swap = MusubiBlockSwapManager.build(
            depth=total_layers,
            blocks_to_swap=musubi_blocks_to_swap,
            swap_device=musubi_block_swap_device,
            logger=logger,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        timestep_sign: Optional[torch.Tensor] = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        return_dict: bool = True,
        hidden_states_buffer: Optional[dict] = None,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, sequence, channel)`):
                Input hidden states.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings to use.
            timestep (`torch.LongTensor`):
                Timestep for denoising.
            return_dict (`bool`):
                Whether to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a tuple.
        """
        hidden_states = self.x_embedder(hidden_states)
        batch_size, num_img_tokens, _ = hidden_states.shape

        timestep = timestep.to(device=hidden_states.device, dtype=hidden_states.dtype)
        if timestep.ndim == 0:
            timestep = timestep.expand(batch_size)
        elif timestep.ndim == 1:
            if timestep.shape[0] == 1:
                timestep = timestep.expand(batch_size)
            elif timestep.shape[0] != batch_size:
                raise ValueError(
                    f"LongCat-Image expected 1 timestep or {batch_size} per-batch timesteps, got {timestep.shape[0]}."
                )
        elif timestep.ndim == 2:
            if timestep.shape[1] != num_img_tokens:
                raise ValueError(
                    f"LongCat-Image expected tokenwise timesteps with sequence length {num_img_tokens}, got {timestep.shape[1]}."
                )
            if timestep.shape[0] == 1:
                timestep = timestep.expand(batch_size, -1)
            elif timestep.shape[0] != batch_size:
                raise ValueError(
                    f"LongCat-Image expected tokenwise timesteps for batch size {batch_size}, got {timestep.shape[0]}."
                )
        else:
            raise ValueError(
                "LongCat-Image expected a scalar, 1D batch tensor, or 2D tokenwise tensor, "
                f"got shape {tuple(timestep.shape)}."
            )

        timestep = timestep * 1000
        guidance = guidance.to(hidden_states.dtype) * 1000 if guidance is not None else None

        temb = self.time_embed(timestep, hidden_states.dtype, timestep_sign=timestep_sign)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        txt_len = encoder_hidden_states.shape[1]

        if timestep.ndim == 2:
            temb_img = temb
            temb_txt = temb.mean(dim=1)
            temb_single = torch.cat([temb_txt.unsqueeze(1).expand(-1, txt_len, -1), temb_img], dim=1)
        else:
            temb_img = temb
            temb_txt = temb
            temb_single = temb

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` 3d torch.Tensor is deprecated. Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` 3d torch.Tensor is deprecated. Please remove the batch dimension and pass it as a 2d torch Tensor"
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        # Musubi block swap activation
        combined_blocks = list(self.transformer_blocks) + list(self.single_transformer_blocks)
        musubi_manager = self._musubi_block_swap
        musubi_offload_active = False
        grad_enabled = torch.is_grad_enabled()
        if musubi_manager is not None:
            musubi_offload_active = musubi_manager.activate(combined_blocks, hidden_states.device, grad_enabled)

        capture_idx = 0
        for index_block, block in enumerate(self.transformer_blocks):
            if musubi_offload_active and musubi_manager.is_managed_block(capture_idx):
                musubi_manager.stream_in(block, hidden_states.device)
            if torch.is_grad_enabled() and self.gradient_checkpointing and self.use_checkpoint[index_block]:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    _run_longcat_transformer_block,
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb_img,
                    temb_txt,
                    image_rotary_emb,
                )
            else:
                encoder_hidden_states, hidden_states = _run_longcat_transformer_block(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb_img,
                    temb_txt,
                    image_rotary_emb,
                )
            if musubi_offload_active and musubi_manager.is_managed_block(capture_idx):
                musubi_manager.stream_out(block)
            _store_hidden_state(hidden_states_buffer, f"layer_{capture_idx}", hidden_states)
            capture_idx += 1

        for index_block, block in enumerate(self.single_transformer_blocks):
            if musubi_offload_active and musubi_manager.is_managed_block(capture_idx):
                musubi_manager.stream_in(block, hidden_states.device)
            if torch.is_grad_enabled() and self.gradient_checkpointing and self.use_single_checkpoint[index_block]:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    _run_longcat_single_transformer_block,
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb_single,
                    image_rotary_emb,
                )
            else:
                encoder_hidden_states, hidden_states = _run_longcat_single_transformer_block(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb_single,
                    image_rotary_emb,
                )
            if musubi_offload_active and musubi_manager.is_managed_block(capture_idx):
                musubi_manager.stream_out(block)
            _store_hidden_state(hidden_states_buffer, f"layer_{capture_idx}", hidden_states)
            capture_idx += 1

        hidden_states = _longcat_apply_ada_layer_norm_continuous(self.norm_out, hidden_states, temb_img)
        output = self.proj_out(hidden_states)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        nn.init.normal_(self.context_embedder.weight, std=0.02)

        for block in self.transformer_blocks:
            nn.init.constant_(block.norm1.linear.weight, 0)
            nn.init.constant_(block.norm1.linear.bias, 0)
            nn.init.constant_(block.norm1_context.linear.weight, 0)
            nn.init.constant_(block.norm1_context.linear.bias, 0)

        for block in self.single_transformer_blocks:
            nn.init.constant_(block.norm.linear.weight, 0)
            nn.init.constant_(block.norm.linear.bias, 0)

        nn.init.constant_(self.norm_out.linear.weight, 0)
        nn.init.constant_(self.norm_out.linear.bias, 0)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)
