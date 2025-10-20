import argparse
import gc
import operator
import os
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.models.attention import FeedForward, _chunked_feed_forward
from diffusers.models.attention_processor import Attention, AttentionProcessor, JointAttnProcessor2_0
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.torch_utils import maybe_allow_in_graph

from simpletuner.helpers.utils.patching import CallableDict

ORIG_DEPTH = 24
FINAL_DEPTH = 36
M_VALUE = 6

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class JointTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        context_pre_only=False,
        qk_norm="layer_norm",
    ):
        super().__init__()

        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"

        self.norm1 = AdaLayerNormZero(dim)

        if context_norm_type == "ada_norm_continous":
            self.norm1_context = AdaLayerNormContinuous(
                dim,
                dim,
                elementwise_affine=False,
                eps=1e-6,
                bias=True,
                norm_type="layer_norm",
            )
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = AdaLayerNormZero(dim)
        else:
            raise ValueError(
                f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`"
            )
        if hasattr(F, "scaled_dot_product_attention"):
            processor = JointAttnProcessor2_0()
        else:
            raise ValueError("The current PyTorch version does not support the `scaled_dot_product_attention` function.")
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            qk_norm=qk_norm,
            dim_head=attention_head_dim // num_attention_heads,
            heads=num_attention_heads,
            out_dim=attention_head_dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=processor,
        )

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            self.norm2_context = None
            self.ff_context = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            (
                norm_encoder_hidden_states,
                c_gate_msa,
                c_shift_mlp,
                c_scale_mlp,
                c_gate_mlp,
            ) = self.norm1_context(encoder_hidden_states, emb=temb)

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context,
                    norm_encoder_hidden_states,
                    self._chunk_dim,
                    self._chunk_size,
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states


class SD3TransformerQKNorm2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    """
    The Transformer model introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        sample_size (`int`): The width of the latent images. This is fixed during training since
            it is used to learn a number of position embeddings.
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of Transformer blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        caption_projection_dim (`int`): Number of dimensions to use when projecting the `encoder_hidden_states`.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        out_channels (`int`, defaults to 16): Number of output channels.
        qk_norm (`str`, defaults to "layer_norm"): The type of qk_norm to use.

        TODO The SD3 paper uses RMSNorm instead of LayerNorm but it is unlikely
             that there is much difference betweens RMSNorm being faster.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        qk_norm: str | None = "layer_norm",
    ):
        super().__init__()
        default_out_channels = in_channels
        effective_out_channels = out_channels if out_channels is not None else default_out_channels
        self.register_to_config(
            sample_size=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            caption_projection_dim=caption_projection_dim,
            pooled_projection_dim=pooled_projection_dim,
            out_channels=effective_out_channels,
            pos_embed_max_size=pos_embed_max_size,
            qk_norm=qk_norm,
        )
        self.out_channels = effective_out_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=self.config.pooled_projection_dim,
        )
        self.context_embedder = nn.Linear(self.config.joint_attention_dim, self.config.caption_projection_dim)

        # `attention_head_dim` is doubled to account for the mixing.
        # It needs to crafted when we get the actual checkpoints.
        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.inner_dim,
                    context_pre_only=i == num_layers - 1,
                    qk_norm=qk_norm,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: torch.nn.Module, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

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
                processors[f"{name}.processor"] = module.get_processor(return_deprecated_lora=True)

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

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`SD3Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()
            lora_scale = joint_attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
                )

        height, width = hidden_states.shape[-2:]

        hidden_states = self.pos_embed(hidden_states)  # takes care of adding positional embeddings too.
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
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
                )

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        # unpatchify
        patch_size = self.config.patch_size
        height = height // patch_size
        width = width // patch_size

        hidden_states = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                height,
                width,
                patch_size,
                patch_size,
                self.out_channels,
            )
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(
                hidden_states.shape[0],
                self.out_channels,
                height * patch_size,
                width * patch_size,
            )
        )

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


def verify_all_parameters_offset_copy(
    model_old,
    model_new,
    layer_name_prefix,
    source_start_idx,
    dest_start_idx,
    num_layers_to_check,
):
    """
    Verifies that all parameters from a specified range in the old model are correctly copied to a new range in the scaled model.

    Parameters:
    - model_old: The original PyTorch model.
    - model_new: The depth-scaled PyTorch model.
    - layer_name_prefix: The prefix of the layer names to check, e.g., 'transformer_blocks'.
    - source_start_idx: The starting index of the layers in the old model from which parameters are copied.
    - dest_start_idx: The starting index of the layers in the new model where parameters are copied into.
    - num_layers_to_check: The number of layers to check from the source_start_idx.
    """
    for offset in range(num_layers_to_check):
        source_idx = source_start_idx + offset
        dest_idx = dest_start_idx + offset
        source_layer = getattr(model_old, layer_name_prefix)[source_idx]
        dest_layer = getattr(model_new, layer_name_prefix)[dest_idx]

        for param_name, source_param in source_layer.named_parameters():
            # Retrieve the corresponding parameter from the destination layer
            if isinstance(operator.attrgetter(param_name)(dest_layer), torch.Tensor):
                dest_param = operator.attrgetter(param_name)(dest_layer)

                # Check if the parameters are close enough (considering floating-point arithmetic)
                if not torch.allclose(source_param, dest_param, atol=1e-6):
                    raise AssertionError(
                        f"Parameter mismatch for {layer_name_prefix}.{source_idx}.{param_name} (original) -> {layer_name_prefix}.{dest_idx}.{param_name} (new)."
                    )
            else:
                raise AssertionError(f"Missing parameter {layer_name_prefix}.{dest_idx}.{param_name} in the new model.")

    print(
        f"All parameters from {source_start_idx} to {source_start_idx + num_layers_to_check - 1} ({num_layers_to_check} layers) in {layer_name_prefix} have been verified to be correctly copied to {dest_start_idx} to {dest_start_idx + num_layers_to_check - 1}."
    )


def expand_existing_sd3_model(model_old):
    # This model is 36 layers deep, versus 24 layers deep from the original model.
    # We will prune 12 layers off from the end and the start of the merged weights.
    model_new = SD3TransformerQKNorm2DModel.from_config(
        {
            "_class_name": "SD3Transformer2DModel",
            "_diffusers_version": "0.30.0.dev0",
            "_name_or_path": "stabilityai/stable-diffusion-3-medium-diffusers",
            "attention_head_dim": 64,
            "caption_projection_dim": 1536,
            "in_channels": 16,
            "joint_attention_dim": 4096,
            "num_attention_heads": 24,
            "num_layers": FINAL_DEPTH,
            "out_channels": 16,
            "patch_size": 2,
            "pooled_projection_dim": 2048,
            "pos_embed_max_size": 192,
            "qk_norm": "layer_norm",
            "sample_size": 128,
        }
    )

    # Copy in layers 0...23 and all other layers.
    with torch.no_grad():
        new_model_param_names = set(name for name, _ in model_new.named_parameters())

        # Iterate through parameters of the old model
        for name, param in model_old.named_parameters():
            if name in new_model_param_names:
                # Get the corresponding parameter from the new model and copy the old param in
                try:
                    model_new.state_dict()[name].copy_(param)
                except RuntimeError as e:
                    if (
                        "The size of tensor a (9216) must match the size of tensor b (3072) at non-singleton dimension 0"
                        in str(e)
                    ):
                        pass
                    else:
                        print(f"Got {str(e)} on layer {name}")
                        raise

    # We now need to deal with [18:] for both transformer_blocks.
    # We do this by copying in [6:] into [18:] for these blocks.
    with torch.no_grad():
        for layer_idx, injection_idx in zip(
            range(M_VALUE, FINAL_DEPTH),
            range(ORIG_DEPTH - M_VALUE, FINAL_DEPTH),
        ):
            for name, param in model_old.named_parameters():
                if "transformer_blocks" in name:
                    if f"transformer_blocks.{layer_idx}." in name:
                        name_to_inject_into = name.replace(
                            f"transformer_blocks.{layer_idx}.",
                            f"transformer_blocks.{injection_idx}.",
                        )
                        model_new.state_dict()[name_to_inject_into].copy_(param)

    # Finally, transform all the newly added qk norm layers in passthroughs.
    # Setting the weights to 1 and the bias to zero means that initially they
    # should do nothing to the model.
    with torch.no_grad():
        for name, param in model_new.named_parameters():
            if "transformer_blocks" in name and ("norm_q" in name or "norm_k" in name):
                if "norm_q.weight" in name:
                    param.fill_(1)
                elif "norm_q.bias" in name:
                    param.fill_(0)

    verify_all_parameters_offset_copy(
        model_old, model_new, "transformer_blocks", 0, 0, ORIG_DEPTH - M_VALUE
    )  # Adjust the index as needed
    verify_all_parameters_offset_copy(
        model_old, model_new, "transformer_blocks", 6, 18, ORIG_DEPTH - M_VALUE
    )  # Adjust the last parameter as needed based on the number of layers you're checking

    orig_params = sum(p.numel() for p in model_old.parameters())
    expanded_params = sum(p.numel() for p in model_new.parameters())
    print(f"Model has been successfully expanded from {orig_params / 1e6:.2f}M to {expanded_params / 1e6:.2f}M.")

    model_new.save_pretrained((os.path.join(args.output_model, "transformer")))
    return model_new


if __name__ == "__main__":
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

    parser = argparse.ArgumentParser(
        description="Make a 24 block deep SD3 2B into a 36 block deep version",
    )
    parser.add_argument(
        "input_model",
        action="store",
        type=str,
        help="The input pretrained model",
    )
    parser.add_argument(
        "output_model",
        action="store",
        type=str,
        help="The output pretrained model location",
    )

    args = parser.parse_args()

    model_old = SD3Transformer2DModel.from_pretrained(
        args.input_model,
        subfolder="transformer",
    )
    model_new = expand_existing_sd3_model(model_old)
    del model_old
    gc.collect()
    model_new = model_new.to("cuda", dtype=torch.bfloat16)
    with torch.no_grad(), torch.inference_mode():
        model_new(
            hidden_states=torch.rand((1, 16, 64, 64)).to("cuda", dtype=torch.bfloat16),
            encoder_hidden_states=torch.rand((1, 144, 4096)).to("cuda", dtype=torch.bfloat16),
            pooled_projections=torch.rand((1, 2048)).to("cuda", dtype=torch.bfloat16),
            timestep=torch.tensor([500]).to("cuda", dtype=torch.bfloat16),
        )
    print("Successfully expanded and tested model.")
