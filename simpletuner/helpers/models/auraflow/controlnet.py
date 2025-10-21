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

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention_processor import (
    Attention,
    AttentionProcessor,
    AuraFlowAttnProcessor2_0,
    FusedAuraFlowAttnProcessor2_0,
)
from diffusers.models.controlnet import zero_module
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormZero, FP32LayerNorm
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph

from simpletuner.helpers.utils.patching import CallableDict

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


# Import or redefine the necessary AuraFlow components
from simpletuner.helpers.models.auraflow.transformer import (
    AuraFlowFeedForward,
    AuraFlowJointTransformerBlock,
    AuraFlowPatchEmbed,
    AuraFlowPreFinalBlock,
    AuraFlowSingleTransformerBlock,
)


@dataclass
class AuraFlowControlNetOutput(BaseOutput):

    controlnet_block_samples: Tuple[torch.Tensor]


class AuraFlowControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):

    _supports_gradient_checkpointing = True
    _no_split_modules = [
        "AuraFlowJointTransformerBlock",
        "AuraFlowSingleTransformerBlock",
        "AuraFlowPatchEmbed",
    ]

    def __init__(
        self,
        sample_size: int = 64,
        patch_size: int = 2,
        in_channels: int = 4,
        num_mmdit_layers: int = 4,
        num_single_dit_layers: int = 32,
        attention_head_dim: int = 256,
        num_attention_heads: int = 12,
        joint_attention_dim: int = 2048,
        caption_projection_dim: int = 3072,
        out_channels: int = 4,
        pos_embed_max_size: int = 1024,
        num_layers: Optional[int] = None,
        extra_conditioning_channels: int = 0,
    ):
        super().__init__()

        default_out_channels = in_channels
        effective_out_channels = out_channels if out_channels is not None else default_out_channels
        self.register_to_config(
            sample_size=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_mmdit_layers=num_mmdit_layers,
            num_single_dit_layers=num_single_dit_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            caption_projection_dim=caption_projection_dim,
            out_channels=effective_out_channels,
            pos_embed_max_size=pos_embed_max_size,
            num_layers=num_layers,
            extra_conditioning_channels=extra_conditioning_channels,
        )
        self.out_channels = effective_out_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        # limit blocks if num_layers specified
        if num_layers is not None:
            # distribute layers proportionally
            total_layers = num_mmdit_layers + num_single_dit_layers
            mmdit_ratio = num_mmdit_layers / total_layers

            actual_num_mmdit_layers = int(num_layers * mmdit_ratio)
            actual_num_single_dit_layers = num_layers - actual_num_mmdit_layers
        else:
            actual_num_mmdit_layers = num_mmdit_layers
            actual_num_single_dit_layers = num_single_dit_layers

        self.num_layers = actual_num_mmdit_layers + actual_num_single_dit_layers

        self.pos_embed = AuraFlowPatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,
        )

        self.context_embedder = nn.Linear(
            joint_attention_dim,
            caption_projection_dim,
            bias=False,
        )

        self.time_step_embed = Timesteps(num_channels=256, downscale_freq_shift=0, scale=1000, flip_sin_to_cos=True)
        self.time_step_proj = TimestepEmbedding(in_channels=256, time_embed_dim=self.inner_dim)

        self.joint_transformer_blocks = nn.ModuleList(
            [
                AuraFlowJointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for i in range(actual_num_mmdit_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                AuraFlowSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(actual_num_single_dit_layers)
            ]
        )

        # zero-initialized linear layers for each transformer block
        self.controlnet_blocks = nn.ModuleList([])
        total_blocks = len(self.joint_transformer_blocks) + len(self.single_transformer_blocks)

        for _ in range(total_blocks):
            controlnet_block = nn.Linear(self.inner_dim, self.inner_dim)
            controlnet_block = zero_module(controlnet_block)
            self.controlnet_blocks.append(controlnet_block)

        if extra_conditioning_channels > 0:
            self.pos_embed_input = AuraFlowPatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels + extra_conditioning_channels,
                embed_dim=self.inner_dim,
                pos_embed_max_size=pos_embed_max_size,
            )
            self.pos_embed_input = zero_module(self.pos_embed_input)
        else:
            self.pos_embed_input = None

        self.register_tokens = nn.Parameter(torch.randn(1, 8, self.inner_dim) * 0.02)

        self.gradient_checkpointing = False

    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    def disable_forward_chunking(self):

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)

    @property
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
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

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
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

    # def _set_gradient_checkpointing(self, module, enable=False, **kwargs):
    #     if hasattr(module, "gradient_checkpointing"):
    #         module.gradient_checkpointing = enable

    @classmethod
    def from_transformer(
        cls,
        transformer,
        num_layers: Optional[int] = None,
        num_extra_conditioning_channels: int = 0,
        load_weights_from_transformer: bool = True,
    ):
        config = transformer.config

        controlnet_config = {
            "sample_size": config.sample_size,
            "patch_size": config.patch_size,
            "in_channels": config.in_channels,
            "num_mmdit_layers": config.num_mmdit_layers,
            "num_single_dit_layers": config.num_single_dit_layers,
            "attention_head_dim": config.attention_head_dim,
            "num_attention_heads": config.num_attention_heads,
            "joint_attention_dim": config.joint_attention_dim,
            "caption_projection_dim": config.caption_projection_dim,
            "out_channels": config.out_channels,
            "pos_embed_max_size": config.pos_embed_max_size,
        }

        if num_layers is not None:
            controlnet_config["num_layers"] = num_layers
        controlnet_config["extra_conditioning_channels"] = num_extra_conditioning_channels

        controlnet = cls(**controlnet_config)

        if load_weights_from_transformer:
            # load weights from transformer, handle layer differences
            if hasattr(transformer, "pos_embed"):
                controlnet.pos_embed.load_state_dict(transformer.pos_embed.state_dict())
            if hasattr(transformer, "time_step_embed"):
                controlnet.time_step_embed.load_state_dict(transformer.time_step_embed.state_dict())
            if hasattr(transformer, "time_step_proj"):
                controlnet.time_step_proj.load_state_dict(transformer.time_step_proj.state_dict())
            if hasattr(transformer, "context_embedder"):
                controlnet.context_embedder.load_state_dict(transformer.context_embedder.state_dict())
            if hasattr(transformer, "register_tokens"):
                controlnet.register_tokens.data = transformer.register_tokens.data.clone()

            for i in range(len(controlnet.joint_transformer_blocks)):
                if i < len(transformer.joint_transformer_blocks):
                    controlnet.joint_transformer_blocks[i].load_state_dict(
                        transformer.joint_transformer_blocks[i].state_dict()
                    )

            for i in range(len(controlnet.single_transformer_blocks)):
                if i < len(transformer.single_transformer_blocks):
                    controlnet.single_transformer_blocks[i].load_state_dict(
                        transformer.single_transformer_blocks[i].state_dict()
                    )

            if controlnet.pos_embed_input is not None:
                controlnet.pos_embed_input = zero_module(controlnet.pos_embed_input)

        return controlnet

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[AuraFlowControlNetOutput, Tuple]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0
            attention_kwargs = {}

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        hidden_states = self.pos_embed(hidden_states)

        if self.pos_embed_input is not None:
            hidden_states = hidden_states + self.pos_embed_input(controlnet_cond)

        temb = self.time_step_embed(timestep).to(dtype=next(self.parameters()).dtype)
        temb = self.time_step_proj(temb)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)
        encoder_hidden_states = torch.cat(
            [
                self.register_tokens.repeat(encoder_hidden_states.size(0), 1, 1),
                encoder_hidden_states,
            ],
            dim=1,
        )

        block_res_samples = []

        for block in self.joint_transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    **ckpt_kwargs,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    attention_kwargs=attention_kwargs,
                )

            block_res_samples.append(hidden_states)

        if len(self.single_transformer_blocks) > 0:
            encoder_seq_len = encoder_hidden_states.size(1)
            combined_hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

            for block in self.single_transformer_blocks:
                if self.training and self.gradient_checkpointing:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    combined_hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        combined_hidden_states,
                        temb,
                        **ckpt_kwargs,
                    )
                else:
                    combined_hidden_states = block(
                        hidden_states=combined_hidden_states,
                        temb=temb,
                        attention_kwargs=attention_kwargs,
                    )

                hidden_states = combined_hidden_states[:, encoder_seq_len:]
                block_res_samples.append(hidden_states)

        controlnet_block_res_samples = []
        for block_res_sample, controlnet_block in zip(block_res_samples, self.controlnet_blocks):
            block_res_sample = controlnet_block(block_res_sample)
            block_res_sample = block_res_sample * conditioning_scale
            controlnet_block_res_samples.append(block_res_sample)

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (controlnet_block_res_samples,)

        return AuraFlowControlNetOutput(controlnet_block_samples=controlnet_block_res_samples)
