# Copyright 2024 The HuggingFace Team and 2025 bghira. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Tuple, Union

import torch
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention_processor import Attention, AttentionProcessor, AttnProcessor2_0, SanaLinearAttnProcessor2_0
from diffusers.models.embeddings import PatchEmbed, PixArtAlphaTextProjection
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from torch import nn

from simpletuner.helpers.training.tread import TREADRouter
from simpletuner.helpers.utils.patching import CallableDict, MutableModuleList, PatchableModule

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _store_hidden_state(buffer, key: str, hidden_states: torch.Tensor, image_tokens_start: int | None = None):
    if buffer is None:
        return
    if image_tokens_start is not None and hidden_states.dim() >= 3:
        buffer[key] = hidden_states[:, image_tokens_start:, ...]
    else:
        buffer[key] = hidden_states


class GLUMBConv(PatchableModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 4,
        norm_type: Optional[str] = None,
        residual_connection: bool = True,
    ) -> None:
        super().__init__()

        self.expand_ratio = float(expand_ratio)
        hidden_channels = int(self.expand_ratio * in_channels)
        self.norm_type = norm_type
        self.residual_connection = residual_connection

        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(in_channels, hidden_channels * 2, 1, 1, 0)
        self.conv_depth = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            3,
            1,
            1,
            groups=hidden_channels * 2,
        )
        self.conv_point = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False)

        self.norm = None
        if norm_type == "rms_norm":
            self.norm = RMSNorm(out_channels, eps=1e-5, elementwise_affine=True, bias=True)

        self._parameter_dtype = self.conv_inverted.weight.dtype

    def _ensure_parameter_dtype(self, dtype: torch.dtype) -> None:
        if dtype == self._parameter_dtype:
            return

        for module in (self.conv_inverted, self.conv_depth, self.conv_point):
            module.to(dtype=dtype)

        if self.norm is not None:
            self.norm.to(dtype=dtype)

        self._parameter_dtype = dtype

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.residual_connection:
            residual = hidden_states

        self._ensure_parameter_dtype(hidden_states.dtype)

        hidden_states = self.conv_inverted(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)

        hidden_states = self.conv_point(hidden_states)

        if self.norm_type == "rms_norm":
            # channel last for rmsnorm
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.residual_connection and residual.shape == hidden_states.shape:
            hidden_states = hidden_states + residual

        return hidden_states


class SanaTransformerBlock(PatchableModule):
    r"""
    Transformer block introduced in [Sana](https://huggingface.co/papers/2410.10629).
    """

    def __init__(
        self,
        dim: int = 2240,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        dropout: float = 0.0,
        num_cross_attention_heads: Optional[int] = 20,
        cross_attention_head_dim: Optional[int] = 112,
        cross_attention_dim: Optional[int] = 2240,
        attention_bias: bool = True,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        attention_out_bias: bool = True,
        mlp_ratio: float = 2.5,
    ) -> None:
        super().__init__()

        # self attention
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=norm_eps)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=None,
            processor=SanaLinearAttnProcessor2_0(),
        )

        # cross attention (optional)
        self.cross_norm: Optional[nn.LayerNorm] = None
        self.attn2: Optional[Attention] = None
        self.cross_attention_projection: Optional[nn.Linear] = None

        if cross_attention_dim is not None:
            if num_cross_attention_heads is None:
                num_cross_attention_heads = num_attention_heads
            if cross_attention_head_dim is None:
                cross_attention_head_dim = attention_head_dim

            self.cross_norm = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_cross_attention_heads,
                dim_head=cross_attention_head_dim,
                dropout=dropout,
                bias=True,
                out_bias=attention_out_bias,
                processor=AttnProcessor2_0(),
            )

        # feedforward
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
        self.ff = GLUMBConv(dim, dim, mlp_ratio, norm_type=None, residual_connection=False)

        # expand ratio is surfaced via GLUMBConv

        self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        height: int = None,
        width: int = None,
    ) -> torch.Tensor:
        batch_size = hidden_states.shape[0]

        # modulation
        modulation = self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1).to(self.scale_shift_table.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [
            tensor.to(hidden_states.dtype) for tensor in modulation.chunk(6, dim=1)
        ]

        # self attention
        norm_hidden_states = self.norm1(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        norm_hidden_states = norm_hidden_states.to(hidden_states.dtype)

        attn_output = self.attn1(norm_hidden_states, attention_mask=attention_mask)
        hidden_states = hidden_states + gate_msa * attn_output

        # cross attention
        if self.attn2 is not None and encoder_hidden_states is not None:
            cross_hidden_states = hidden_states
            if self.cross_norm is not None:
                cross_hidden_states = self.cross_norm(cross_hidden_states)
            cross_hidden_states = cross_hidden_states.to(hidden_states.dtype)

            encoder_states = encoder_hidden_states
            expected_dim = self.attn2.to_k.in_features
            current_dim = encoder_states.shape[-1]
            if current_dim != expected_dim:
                if (
                    self.cross_attention_projection is None
                    or self.cross_attention_projection.in_features != current_dim
                    or self.cross_attention_projection.out_features != expected_dim
                ):
                    projection = nn.Linear(current_dim, expected_dim, bias=False)
                    self.cross_attention_projection = projection
                self.cross_attention_projection = self.cross_attention_projection.to(
                    encoder_states.device, dtype=encoder_states.dtype
                )
                encoder_states = self.cross_attention_projection(encoder_states)

            attn_output = self.attn2(
                cross_hidden_states,
                encoder_hidden_states=encoder_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = hidden_states + attn_output

        # feedforward
        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        height, width = self._resolve_spatial_dimensions(hidden_states, height, width)

        norm_hidden_states = norm_hidden_states.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
        ff_output = self.ff(norm_hidden_states)
        ff_output = ff_output.permute(0, 2, 3, 1).reshape(batch_size, -1, hidden_states.shape[-1])
        hidden_states = hidden_states + gate_mlp * ff_output

        return hidden_states

    @staticmethod
    def _resolve_spatial_dimensions(
        hidden_states: torch.Tensor, height: Optional[int], width: Optional[int]
    ) -> Tuple[int, int]:
        _, seq_len, _ = hidden_states.shape

        if height is None or width is None or height * width != seq_len:
            if height is None and width is None:
                root = int(round(seq_len**0.5))
                if root * root == seq_len:
                    height = width = root
                else:
                    height = seq_len
                    width = 1
            elif height is None and width is not None:
                height = max(1, seq_len // width)
            elif width is None and height is not None:
                width = max(1, seq_len // height)

            if height * width != seq_len:
                height = seq_len
                width = 1

        return int(height), int(width)


class SanaTransformer2DModel(PatchableModule, ModelMixin, ConfigMixin, PeftAdapterMixin):
    r"""
    A 2D Transformer model introduced in [Sana](https://huggingface.co/papers/2410.10629) family of models.

    Args:
        in_channels (`int`, defaults to `32`):
            The number of channels in the input.
        out_channels (`int`, *optional*, defaults to `32`):
            The number of channels in the output.
        num_attention_heads (`int`, defaults to `70`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, defaults to `32`):
            The number of channels in each head.
        num_layers (`int`, defaults to `20`):
            The number of layers of Transformer blocks to use.
        num_cross_attention_heads (`int`, *optional*, defaults to `20`):
            The number of heads to use for cross-attention.
        cross_attention_head_dim (`int`, *optional*, defaults to `112`):
            The number of channels in each head for cross-attention.
        cross_attention_dim (`int`, *optional*, defaults to `2240`):
            The number of channels in the cross-attention output.
        caption_channels (`int`, defaults to `2304`):
            The number of channels in the caption embeddings.
        mlp_ratio (`float`, defaults to `2.5`):
            The expansion ratio to use in the GLUMBConv layer.
        dropout (`float`, defaults to `0.0`):
            The dropout probability.
        attention_bias (`bool`, defaults to `False`):
            Whether to use bias in the attention layer.
        sample_size (`int`, defaults to `32`):
            The base size of the input latent.
        patch_size (`int`, defaults to `1`):
            The size of the patches to use in the patch embedding layer.
        norm_elementwise_affine (`bool`, defaults to `False`):
            Whether to use elementwise affinity in the normalization layer.
        norm_eps (`float`, defaults to `1e-6`):
            The epsilon value for the normalization layer.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["SanaTransformerBlock", "PatchEmbed"]
    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: Optional[int] = 32,
        num_attention_heads: int = 70,
        attention_head_dim: int = 32,
        num_layers: int = 20,
        num_cross_attention_heads: Optional[int] = 20,
        cross_attention_head_dim: Optional[int] = 112,
        cross_attention_dim: Optional[int] = 2240,
        caption_channels: int = 2304,
        mlp_ratio: float = 2.5,
        dropout: float = 0.0,
        attention_bias: bool = False,
        sample_size: int = 32,
        patch_size: Union[int, Tuple[int, int]] = 1,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Normalise patch size to a 2D tuple for consistency with diffusers' PatchEmbed
        if isinstance(patch_size, int):
            patch_size_int = patch_size
            patch_size_tuple: Tuple[int, int] = (patch_size, patch_size)
        elif isinstance(patch_size, (tuple, list)) and len(patch_size) == 2:
            if patch_size[0] != patch_size[1]:
                raise ValueError("SanaTransformer2DModel expects square patches.")
            patch_size_int = int(patch_size[0])
            patch_size_tuple = (patch_size_int, patch_size_int)
        else:
            raise ValueError("`patch_size` must be an int or a tuple/list of length 2.")

        patch_area = patch_size_tuple[0] * patch_size_tuple[1]

        effective_out_channels = out_channels or in_channels
        self.register_to_config(
            in_channels=in_channels,
            out_channels=effective_out_channels,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            num_layers=num_layers,
            num_cross_attention_heads=num_cross_attention_heads,
            cross_attention_head_dim=cross_attention_head_dim,
            cross_attention_dim=cross_attention_dim,
            caption_channels=caption_channels,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_bias=attention_bias,
            sample_size=sample_size,
            patch_size=patch_size_int,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            interpolation_scale=interpolation_scale,
        )

        out_channels = effective_out_channels
        inner_dim = num_attention_heads * attention_head_dim

        # Ensure the normalised patch size is reflected in the stored config
        self.config.patch_size = patch_size_int

        # patch embedding
        self.patch_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size_int,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=interpolation_scale,
            pos_embed_type="sincos" if interpolation_scale is not None else None,
        )

        # condition embeddings
        self.time_embed = AdaLayerNormSingle(inner_dim)
        # Signed-time embedding for TwinFlow-style negative time handling.
        self.time_sign_embed = nn.Embedding(2, inner_dim)
        nn.init.zeros_(self.time_sign_embed.weight)

        self.caption_projection = PixArtAlphaTextProjection(in_features=caption_channels, hidden_size=inner_dim)
        self.caption_norm = RMSNorm(inner_dim, eps=1e-5, elementwise_affine=True)

        # transformer blocks
        self.transformer_blocks = MutableModuleList(
            [
                SanaTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    num_cross_attention_heads=num_cross_attention_heads,
                    cross_attention_head_dim=cross_attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_layers)
            ]
        )

        # output blocks
        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim**0.5)

        self.norm_out = nn.LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(inner_dim, patch_area * out_channels)

        self.gradient_checkpointing = False
        self.gradient_checkpointing_interval = None

        # tread support
        self._tread_router = None
        self._tread_routes = None

    def set_router(self, router: TREADRouter, routes: Optional[List[Dict]] = None):
        """Set TREAD router and routes for token reduction during training."""
        self._tread_router = router
        self._tread_routes = routes

    def set_gradient_checkpointing_interval(self, interval: int):
        r"""
        Sets the gradient checkpointing interval for the model.

        Parameters:
            interval (`int`):
                The interval at which to checkpoint the gradients.
        """
        self.gradient_checkpointing_interval = interval

    @staticmethod
    def _coerce_module_output(output: Union[torch.Tensor, Transformer2DModelOutput, Tuple, List]):
        if isinstance(output, Transformer2DModelOutput):
            return output.sample
        if isinstance(output, dict):
            return output.get("sample", next(iter(output.values())))
        if isinstance(output, (list, tuple)):
            return SanaTransformer2DModel._coerce_module_output(output[0])
        return output

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return CallableDict(processors)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        timestep_sign: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[dict] = None,
        force_keep_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        hidden_states_buffer: Optional[dict] = None,
    ) -> Union[Tuple[torch.Tensor, ...], Transformer2DModelOutput]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # scale lora layers
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # input
        batch_size, num_channels, height, width = hidden_states.shape
        p = self.config.patch_size
        post_patch_height, post_patch_width = height // p, width // p

        hidden_states = self.patch_embed(hidden_states)

        timestep, embedded_timestep = self.time_embed(timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype)
        if timestep_sign is not None:
            sign_idx = (timestep_sign.view(-1) < 0).long().to(device=hidden_states.device)
            sign_emb = self.time_sign_embed(sign_idx).to(dtype=embedded_timestep.dtype, device=hidden_states.device)
            embedded_timestep = embedded_timestep + sign_emb.view(batch_size, -1)

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        encoder_hidden_states = self.caption_norm(encoder_hidden_states)

        # transformer blocks
        use_reentrant = is_torch_version("<=", "1.11.0")
        ckpt_kwargs = {"use_reentrant": use_reentrant} if is_torch_version(">=", "1.11.0") else {}

        # tread init
        routes = self._tread_routes or []
        router = self._tread_router
        use_routing = self.training and len(routes) > 0 and torch.is_grad_enabled()

        capture_idx = 0
        for i, block in enumerate(self.transformer_blocks):
            mask_info = None
            original_hidden_states = None

            if use_routing:
                for route in routes:
                    start_idx = route["start_layer_idx"]
                    end_idx = route["end_layer_idx"]
                    if start_idx < 0:
                        start_idx = len(self.transformer_blocks) + start_idx
                    if end_idx < 0:
                        end_idx = len(self.transformer_blocks) + end_idx

                    if start_idx <= i <= end_idx:
                        mask_info = router.get_mask(
                            hidden_states,
                            route["selection_ratio"],
                            force_keep=force_keep_mask,
                        )
                        original_hidden_states = hidden_states
                        hidden_states = router.start_route(hidden_states, mask_info)
                        break
            if (
                self.training
                and self.gradient_checkpointing
                and (self.gradient_checkpointing_interval is None or i % self.gradient_checkpointing_interval == 0)
            ):

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )

            hidden_states = self._coerce_module_output(hidden_states)

            if mask_info is not None and router is not None:
                hidden_states = router.end_route(hidden_states, mask_info, original_x=original_hidden_states)

            _store_hidden_state(hidden_states_buffer, f"layer_{capture_idx}", hidden_states)
            capture_idx += 1
        tokens = hidden_states.shape[1]
        expected_tokens = post_patch_height * post_patch_width
        if tokens != expected_tokens:
            post_patch_height, post_patch_width = self._infer_token_grid(tokens, post_patch_height, post_patch_width)

        # normalization
        modulation_out = self.scale_shift_table[None].to(hidden_states.dtype) + embedded_timestep[:, None].to(
            hidden_states.dtype
        )
        shift, scale = modulation_out.chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states)

        # modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_height,
            post_patch_width,
            self.config.patch_size,
            self.config.patch_size,
            -1,
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(batch_size, -1, post_patch_height * p, post_patch_width * p)

        if USE_PEFT_BACKEND:
            # unscale lora layers
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)

    @staticmethod
    def _infer_token_grid(num_tokens: int, target_height: int, target_width: int) -> Tuple[int, int]:
        if target_height > 0 and num_tokens % target_height == 0:
            return int(target_height), int(num_tokens // target_height)
        if target_width > 0 and num_tokens % target_width == 0:
            return int(num_tokens // target_width), int(target_width)

        root = int(round(num_tokens**0.5))
        if root > 0 and root * root == num_tokens:
            return root, root

        return num_tokens, 1
