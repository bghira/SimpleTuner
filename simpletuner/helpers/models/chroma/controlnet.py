from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.controlnets.controlnet import ControlNetConditioningEmbedding, zero_module
from diffusers.models.embeddings import FluxPosEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, logging, scale_lora_layers, unscale_lora_layers

from .transformer import (
    ChromaApproximator,
    ChromaCombinedTimestepTextProjEmbeddings,
    ChromaSingleTransformerBlock,
    ChromaTransformerBlock,
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@dataclass
class ChromaControlNetOutput(BaseOutput):
    controlnet_block_samples: Optional[Tuple[torch.Tensor, ...]]
    controlnet_single_block_samples: Optional[Tuple[torch.Tensor, ...]]


class ChromaControlNetModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        axes_dims_rope: Tuple[int, ...] = (16, 56, 56),
        approximator_num_channels: int = 64,
        approximator_hidden_dim: int = 5120,
        approximator_layers: int = 5,
        conditioning_embedding_channels: Optional[int] = None,
    ):
        super().__init__()
        self.out_channels = in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)
        self.time_text_embed = ChromaCombinedTimestepTextProjEmbeddings(
            num_channels=approximator_num_channels // 4,
            out_dim=3 * num_single_layers + 2 * 6 * num_layers + 2,
        )
        self.distilled_guidance_layer = ChromaApproximator(
            in_dim=approximator_num_channels,
            out_dim=self.inner_dim,
            hidden_dim=approximator_hidden_dim,
            n_layers=approximator_layers,
        )

        self.context_embedder = nn.Linear(joint_attention_dim, self.inner_dim)
        self.x_embedder = nn.Linear(in_channels, self.inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                ChromaTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                ChromaSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                )
                for _ in range(num_single_layers)
            ]
        )

        self.controlnet_blocks = nn.ModuleList(
            [zero_module(nn.Linear(self.inner_dim, self.inner_dim)) for _ in range(len(self.transformer_blocks))]
        )
        self.controlnet_single_blocks = nn.ModuleList(
            [zero_module(nn.Linear(self.inner_dim, self.inner_dim)) for _ in range(len(self.single_transformer_blocks))]
        )

        if conditioning_embedding_channels is not None:
            self.input_hint_block = ControlNetConditioningEmbedding(
                conditioning_embedding_channels=conditioning_embedding_channels,
                block_out_channels=(16, 16, 16, 16),
            )
            self.controlnet_x_embedder = nn.Linear(in_channels, self.inner_dim)
        else:
            self.input_hint_block = None
            self.controlnet_x_embedder = zero_module(nn.Linear(in_channels, self.inner_dim))

        self.gradient_checkpointing = False

    @classmethod
    def from_transformer(
        cls,
        transformer,
        load_weights_from_transformer: bool = True,
    ):
        config = dict(transformer.config)
        controlnet = cls.from_config(config)

        if load_weights_from_transformer:
            controlnet.pos_embed.load_state_dict(transformer.pos_embed.state_dict())
            controlnet.time_text_embed.load_state_dict(transformer.time_text_embed.state_dict())
            controlnet.distilled_guidance_layer.load_state_dict(transformer.distilled_guidance_layer.state_dict())
            controlnet.context_embedder.load_state_dict(transformer.context_embedder.state_dict())
            controlnet.x_embedder.load_state_dict(transformer.x_embedder.state_dict())
            controlnet.transformer_blocks.load_state_dict(transformer.transformer_blocks.state_dict(), strict=False)
            controlnet.single_transformer_blocks.load_state_dict(
                transformer.single_transformer_blocks.state_dict(), strict=False
            )

            controlnet.controlnet_x_embedder = zero_module(controlnet.controlnet_x_embedder)

        return controlnet

    @property
    def attn_processors(self):
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """

        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

        """
        if isinstance(processor, dict) and len(processor) != len(self.attn_processors):
            raise ValueError(
                f"You have passed {len(processor.keys())} attention processors, but the model has {len(self.attn_processors)} attention processors."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor: AttentionProcessor, index: int = 0):
            if hasattr(module, "set_processor"):
                if isinstance(processor, dict):
                    module.set_processor(processor.pop(f"{name}.processor"))
                else:
                    module.set_processor(processor)

            for sub_name, child in module.named_children():
                if processor is not None:
                    fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor, index)
                else:
                    fn_recursive_attn_processor(f"{name}.{sub_name}", child, None, index)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def enable_xformers_memory_efficient_attention(self, attention_op: Optional[str] = None):
        r"""
        Enable memory efficient attention from xFormers.

        See https://github.com/facebookresearch/xformers for more details.
        """
        for module in self.modules():
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(True, attention_op=attention_op)

    def disable_xformers_memory_efficient_attention(self):
        r"""
        Disable memory efficient attention from xFormers.
        """
        for module in self.modules():
            if hasattr(module, "set_use_memory_efficient_attention_xformers"):
                module.set_use_memory_efficient_attention_xformers(False)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        encoder_hidden_states: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]], ChromaControlNetOutput]:
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            joint_attention_kwargs = {}
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            scale_lora_layers(self, lora_scale)
        elif "scale" in joint_attention_kwargs:
            logger.warning("Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective.")

        hidden_states = self.x_embedder(hidden_states)

        if self.input_hint_block is not None:
            controlnet_cond = self.input_hint_block(controlnet_cond)
            batch_size, channels, height_pw, width_pw = controlnet_cond.shape
            height = height_pw // self.config.patch_size
            width = width_pw // self.config.patch_size
            controlnet_cond = controlnet_cond.reshape(
                batch_size, channels, height, self.config.patch_size, width, self.config.patch_size
            )
            controlnet_cond = controlnet_cond.permute(0, 2, 4, 1, 3, 5)
            controlnet_cond = controlnet_cond.reshape(batch_size, height * width, -1)

        hidden_states = hidden_states + self.controlnet_x_embedder(controlnet_cond)

        timestep = timestep.to(hidden_states.dtype) * 1000
        temb_inputs = self.time_text_embed(timestep)
        pooled_temb = self.distilled_guidance_layer(temb_inputs)

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            logger.warning(
                "Passing `txt_ids` with a batch dimension is deprecated. Please remove the batch dimension and pass it as a 2d torch Tensor."
            )
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            logger.warning(
                "Passing `img_ids` with a batch dimension is deprecated. Please remove the batch dimension and pass it as a 2d torch Tensor."
            )
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.pos_embed(ids)

        txt_len = encoder_hidden_states.shape[1]

        block_samples: Tuple[torch.Tensor, ...] = ()
        current_hidden_states = hidden_states
        current_encoder_states = encoder_hidden_states

        for index_block, block in enumerate(self.transformer_blocks):
            img_offset = 3 * len(self.single_transformer_blocks)
            txt_offset = img_offset + 6 * len(self.transformer_blocks)
            img_modulation = img_offset + 6 * index_block
            text_modulation = txt_offset + 6 * index_block
            temb = torch.cat(
                (
                    pooled_temb[:, img_modulation : img_modulation + 6],
                    pooled_temb[:, text_modulation : text_modulation + 6],
                ),
                dim=1,
            )

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                current_encoder_states, current_hidden_states = self._gradient_checkpointing_func(
                    block,
                    current_hidden_states,
                    current_encoder_states,
                    temb,
                    image_rotary_emb,
                    attention_mask,
                )
            else:
                current_encoder_states, current_hidden_states = block(
                    hidden_states=current_hidden_states,
                    encoder_hidden_states=current_encoder_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    attention_mask=attention_mask,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            block_samples = block_samples + (current_hidden_states,)

        single_block_samples: Tuple[torch.Tensor, ...] = ()
        for index_block, block in enumerate(self.single_transformer_blocks):
            temb = pooled_temb[:, 3 * index_block : 3 * index_block + 3]

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                current_hidden_states = self._gradient_checkpointing_func(
                    block,
                    current_hidden_states,
                    temb,
                    image_rotary_emb,
                    attention_mask,
                )
            else:
                current_hidden_states = block(
                    hidden_states=current_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    attention_mask=attention_mask,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            single_block_samples = single_block_samples + (current_hidden_states,)

        controlnet_block_samples: Tuple[torch.Tensor, ...] = tuple(
            control_block(sample) * conditioning_scale
            for sample, control_block in zip(block_samples, self.controlnet_blocks)
        )
        controlnet_single_block_samples: Tuple[torch.Tensor, ...] = tuple(
            control_block(sample) * conditioning_scale
            for sample, control_block in zip(single_block_samples, self.controlnet_single_blocks)
        )

        if USE_PEFT_BACKEND:
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return controlnet_block_samples, controlnet_single_block_samples

        return ChromaControlNetOutput(
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
        )
