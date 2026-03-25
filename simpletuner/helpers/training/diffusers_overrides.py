from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import fsdp_utils as accelerate_fsdp_utils
from diffusers.models.attention import Attention
from diffusers.models.attention_dispatch import dispatch_attention_fn

logger = logging.getLogger(__name__)

# SimpleTuner always fuses.
PERMANENT_FUSION = True


@torch.no_grad()
def fuse_projections_smart(self, fuse=True, permanent=None):
    """
    Fuse QKV projections with option for permanent (delete originals) or reversible fusion.

    Args:
        fuse: Whether to fuse (always True for compatibility)
        permanent: Override for PERMANENT_FUSION setting. If None, uses global setting.
    """
    if self.fused_projections:
        return  # Already fused

    # Determine if this should be permanent
    is_permanent = PERMANENT_FUSION if permanent is None else permanent

    device = self.to_q.weight.data.device
    dtype = self.to_q.weight.data.dtype

    if not self.is_cross_attention:
        # Fuse Q, K, V for self-attention
        concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
        in_features = concatenated_weights.shape[1]
        out_features = concatenated_weights.shape[0]

        # Create fused layer
        self.to_qkv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
        self.to_qkv.weight.copy_(concatenated_weights)

        if self.use_bias:
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            self.to_qkv.bias.copy_(concatenated_bias)

        if is_permanent:
            # DELETE the original layers
            del self.to_q
            del self.to_k
            del self.to_v

            # Remove from _modules to ensure they're not accessible
            if "to_q" in self._modules:
                del self._modules["to_q"]
            if "to_k" in self._modules:
                del self._modules["to_k"]
            if "to_v" in self._modules:
                del self._modules["to_v"]

    else:
        # For cross-attention, keep to_q separate, only fuse k,v
        concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
        in_features = concatenated_weights.shape[1]
        out_features = concatenated_weights.shape[0]

        self.to_kv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
        self.to_kv.weight.copy_(concatenated_weights)

        if self.use_bias:
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            self.to_kv.bias.copy_(concatenated_bias)

        if is_permanent:
            # DELETE the original k,v layers
            del self.to_k
            del self.to_v

            if "to_k" in self._modules:
                del self._modules["to_k"]
            if "to_v" in self._modules:
                del self._modules["to_v"]

    # Handle added projections for SD3 and others
    if (
        getattr(self, "add_q_proj", None) is not None
        and getattr(self, "add_k_proj", None) is not None
        and getattr(self, "add_v_proj", None) is not None
    ):
        concatenated_weights = torch.cat(
            [
                self.add_q_proj.weight.data,
                self.add_k_proj.weight.data,
                self.add_v_proj.weight.data,
            ]
        )
        in_features = concatenated_weights.shape[1]
        out_features = concatenated_weights.shape[0]

        self.to_added_qkv = nn.Linear(
            in_features,
            out_features,
            bias=self.added_proj_bias,
            device=device,
            dtype=dtype,
        )
        self.to_added_qkv.weight.copy_(concatenated_weights)

        if self.added_proj_bias:
            concatenated_bias = torch.cat(
                [
                    self.add_q_proj.bias.data,
                    self.add_k_proj.bias.data,
                    self.add_v_proj.bias.data,
                ]
            )
            self.to_added_qkv.bias.copy_(concatenated_bias)

        if is_permanent:
            # DELETE the original added projection layers
            del self.add_q_proj
            del self.add_k_proj
            del self.add_v_proj

            if "add_q_proj" in self._modules:
                del self._modules["add_q_proj"]
            if "add_k_proj" in self._modules:
                del self._modules["add_k_proj"]
            if "add_v_proj" in self._modules:
                del self._modules["add_v_proj"]

    self.fused_projections = True
    fusion_type = "permanent" if is_permanent else "reversible"
    logger.debug(f"Fused projections for {self.__class__.__name__} ({fusion_type})")


@torch.no_grad()
def unfuse_projections_smart(self):
    """
    Unfuse the QKV projections back to their individual components.
    Will warn and return if fusion was permanent.
    """
    if not self.fused_projections:
        logger.debug("Projections are not fused, nothing to unfuse")
        return

    # Check if layers were deleted (permanent fusion)
    if not hasattr(self, "to_q") and hasattr(self, "to_qkv"):
        logger.warning(
            "Cannot unfuse projections - original layers were deleted during permanent fusion! "
            "Set PERMANENT_FUSION=False or use fuse_projections(permanent=False) for reversible fusion."
        )
        return

    logger.debug(f"Unfusing projections for {self.__class__.__name__}")

    # Handle self-attention unfusing
    if hasattr(self, "to_qkv"):
        # Get device and dtype from fused layer
        device = self.to_qkv.weight.device
        dtype = self.to_qkv.weight.dtype

        # Get the concatenated weights and bias
        concatenated_weights = self.to_qkv.weight.data

        # Calculate dimensions
        total_dim = concatenated_weights.shape[0]
        q_dim = self.inner_dim
        k_dim = self.inner_kv_dim
        v_dim = self.inner_kv_dim

        # Verify dimensions
        assert total_dim == q_dim + k_dim + v_dim, f"Dimension mismatch: {total_dim} != {q_dim} + {k_dim} + {v_dim}"

        # Split the weights
        q_weight = concatenated_weights[:q_dim]
        k_weight = concatenated_weights[q_dim : q_dim + k_dim]
        v_weight = concatenated_weights[q_dim + k_dim :]

        # Create individual linear layers
        self.to_q = nn.Linear(self.query_dim, q_dim, bias=self.use_bias, device=device, dtype=dtype)
        self.to_k = nn.Linear(
            self.cross_attention_dim,
            k_dim,
            bias=self.use_bias,
            device=device,
            dtype=dtype,
        )
        self.to_v = nn.Linear(
            self.cross_attention_dim,
            v_dim,
            bias=self.use_bias,
            device=device,
            dtype=dtype,
        )

        # Copy weights
        self.to_q.weight.data.copy_(q_weight)
        self.to_k.weight.data.copy_(k_weight)
        self.to_v.weight.data.copy_(v_weight)

        # Handle biases if they exist
        if self.use_bias and hasattr(self.to_qkv, "bias") and self.to_qkv.bias is not None:
            concatenated_bias = self.to_qkv.bias.data
            q_bias = concatenated_bias[:q_dim]
            k_bias = concatenated_bias[q_dim : q_dim + k_dim]
            v_bias = concatenated_bias[q_dim + k_dim :]

            self.to_q.bias.data.copy_(q_bias)
            self.to_k.bias.data.copy_(k_bias)
            self.to_v.bias.data.copy_(v_bias)

        # Remove the fused layer
        del self.to_qkv
        if "to_qkv" in self._modules:
            del self._modules["to_qkv"]

        logger.debug("Unfused to_qkv -> to_q, to_k, to_v")

    # Handle cross-attention unfusing (fused K,V only)
    elif hasattr(self, "to_kv"):
        # Get device and dtype
        device = self.to_kv.weight.device
        dtype = self.to_kv.weight.dtype

        # Get concatenated weights
        concatenated_weights = self.to_kv.weight.data

        # Calculate dimensions
        total_dim = concatenated_weights.shape[0]
        k_dim = self.inner_kv_dim
        v_dim = self.inner_kv_dim

        assert total_dim == k_dim + v_dim, f"Dimension mismatch for KV: {total_dim} != {k_dim} + {v_dim}"

        # Split weights
        k_weight = concatenated_weights[:k_dim]
        v_weight = concatenated_weights[k_dim:]

        # Create individual layers
        self.to_k = nn.Linear(
            self.cross_attention_dim,
            k_dim,
            bias=self.use_bias,
            device=device,
            dtype=dtype,
        )
        self.to_v = nn.Linear(
            self.cross_attention_dim,
            v_dim,
            bias=self.use_bias,
            device=device,
            dtype=dtype,
        )

        # Copy weights
        self.to_k.weight.data.copy_(k_weight)
        self.to_v.weight.data.copy_(v_weight)

        # Handle biases
        if self.use_bias and hasattr(self.to_kv, "bias") and self.to_kv.bias is not None:
            concatenated_bias = self.to_kv.bias.data
            k_bias = concatenated_bias[:k_dim]
            v_bias = concatenated_bias[k_dim:]

            self.to_k.bias.data.copy_(k_bias)
            self.to_v.bias.data.copy_(v_bias)

        # Remove fused layer
        del self.to_kv
        if "to_kv" in self._modules:
            del self._modules["to_kv"]

        logger.debug("Unfused to_kv -> to_k, to_v")

    # Handle added projections (SD3/Flux style)
    if hasattr(self, "to_added_qkv"):
        # Get device and dtype
        device = self.to_added_qkv.weight.device
        dtype = self.to_added_qkv.weight.dtype

        # Get concatenated weights
        concatenated_weights = self.to_added_qkv.weight.data

        # Calculate dimensions
        total_dim = concatenated_weights.shape[0]
        q_dim = self.inner_dim
        k_dim = self.inner_kv_dim
        v_dim = self.inner_kv_dim

        assert (
            total_dim == q_dim + k_dim + v_dim
        ), f"Dimension mismatch for added QKV: {total_dim} != {q_dim} + {k_dim} + {v_dim}"

        # Split weights
        add_q_weight = concatenated_weights[:q_dim]
        add_k_weight = concatenated_weights[q_dim : q_dim + k_dim]
        add_v_weight = concatenated_weights[q_dim + k_dim :]

        # Create individual layers
        self.add_q_proj = nn.Linear(
            self.added_kv_proj_dim,
            q_dim,
            bias=self.added_proj_bias,
            device=device,
            dtype=dtype,
        )
        self.add_k_proj = nn.Linear(
            self.added_kv_proj_dim,
            k_dim,
            bias=self.added_proj_bias,
            device=device,
            dtype=dtype,
        )
        self.add_v_proj = nn.Linear(
            self.added_kv_proj_dim,
            v_dim,
            bias=self.added_proj_bias,
            device=device,
            dtype=dtype,
        )

        # Copy weights
        self.add_q_proj.weight.data.copy_(add_q_weight)
        self.add_k_proj.weight.data.copy_(add_k_weight)
        self.add_v_proj.weight.data.copy_(add_v_weight)

        # Handle biases
        if self.added_proj_bias and hasattr(self.to_added_qkv, "bias") and self.to_added_qkv.bias is not None:
            concatenated_bias = self.to_added_qkv.bias.data
            add_q_bias = concatenated_bias[:q_dim]
            add_k_bias = concatenated_bias[q_dim : q_dim + k_dim]
            add_v_bias = concatenated_bias[q_dim + k_dim :]

            self.add_q_proj.bias.data.copy_(add_q_bias)
            self.add_k_proj.bias.data.copy_(add_k_bias)
            self.add_v_proj.bias.data.copy_(add_v_bias)

        # Remove fused layer
        del self.to_added_qkv
        if "to_added_qkv" in self._modules:
            del self._modules["to_added_qkv"]

        logger.debug("Unfused to_added_qkv -> add_q_proj, add_k_proj, add_v_proj")

    # Mark as unfused
    self.fused_projections = False
    logger.debug("Unfusing complete")


def patch_attention_flexible():
    """Apply flexible fusion/unfusion patches to Attention class"""
    # Store originals
    Attention._original_fuse_projections = Attention.fuse_projections
    Attention._original_unfuse_projections = getattr(Attention, "unfuse_projections", None)

    # Apply our versions
    Attention.fuse_projections = fuse_projections_smart
    Attention.unfuse_projections = unfuse_projections_smart

    logger.info(f"Patched Attention with flexible fusion (permanent={PERMANENT_FUSION})")


# Convenience functions for different use cases
def enable_permanent_fusion():
    """Enable permanent fusion mode globally"""
    global PERMANENT_FUSION
    PERMANENT_FUSION = True
    logger.info("Enabled permanent QKV fusion mode")


def enable_reversible_fusion():
    """Enable reversible fusion mode globally"""
    global PERMANENT_FUSION
    PERMANENT_FUSION = False
    logger.info("Enabled reversible QKV fusion mode")


patch_attention_flexible()


def _pad_qwen_hidden_states_to_fixed_length(
    hidden_states: Iterable[torch.Tensor],
    *,
    target_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    padded_hidden_states = []
    padded_attention_masks = []

    for sample_hidden_states in hidden_states:
        sample_seq_len = min(sample_hidden_states.size(0), target_length)
        sample_hidden_states = sample_hidden_states[:sample_seq_len]

        pad_tokens = target_length - sample_seq_len
        if pad_tokens > 0:
            sample_hidden_states = torch.cat(
                [
                    sample_hidden_states,
                    sample_hidden_states.new_zeros((pad_tokens, sample_hidden_states.size(1))),
                ],
                dim=0,
            )

        sample_attention_mask = torch.cat(
            [
                torch.ones(sample_seq_len, dtype=torch.long, device=sample_hidden_states.device),
                torch.zeros(pad_tokens, dtype=torch.long, device=sample_hidden_states.device),
            ]
        )

        padded_hidden_states.append(sample_hidden_states)
        padded_attention_masks.append(sample_attention_mask)

    return torch.stack(padded_hidden_states), torch.stack(padded_attention_masks)


def _patched_qwen_prompt_embeds_from_tokenizer(
    self,
    prompt=None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    device = device or self._execution_device
    dtype = dtype or self.text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt

    template = self.prompt_template_encode
    drop_idx = self.prompt_template_encode_start_idx
    txt = [template.format(entry) for entry in prompt]
    txt_tokens = self.tokenizer(
        txt,
        max_length=self.tokenizer_max_length + drop_idx,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    encoder_hidden_states = self.text_encoder(
        input_ids=txt_tokens.input_ids,
        attention_mask=txt_tokens.attention_mask,
        output_hidden_states=True,
    )
    hidden_states = encoder_hidden_states.hidden_states[-1]
    split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
    split_hidden_states = [entry[drop_idx:] for entry in split_hidden_states]
    prompt_embeds, encoder_attention_mask = _pad_qwen_hidden_states_to_fixed_length(
        split_hidden_states,
        target_length=self.tokenizer_max_length,
    )

    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    return prompt_embeds, encoder_attention_mask


def _patched_qwen_prompt_embeds_from_processor(
    self,
    prompt=None,
    image=None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    device = device or self._execution_device
    dtype = dtype or self.text_encoder.dtype

    prompt = [prompt] if isinstance(prompt, str) else prompt
    drop_idx = self.prompt_template_encode_start_idx

    if hasattr(self, "processor") and self.processor is not None:
        processor_kwargs = {
            "padding": "max_length",
            "max_length": self.tokenizer_max_length + drop_idx,
            "truncation": True,
            "return_tensors": "pt",
        }

        if image is None and hasattr(self, "prompt_template_encode"):
            text_inputs = [self.prompt_template_encode.format(entry) for entry in prompt]
        else:
            img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
            if isinstance(image, list):
                base_img_prompt = "".join(img_prompt_template.format(i + 1) for i, _ in enumerate(image))
            elif image is not None:
                base_img_prompt = img_prompt_template.format(1)
            else:
                base_img_prompt = ""
            text_inputs = [self.prompt_template_encode.format(base_img_prompt + entry) for entry in prompt]

        model_inputs = self.processor(text=text_inputs, images=image, **processor_kwargs).to(device)
        outputs = self.text_encoder(
            input_ids=model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            pixel_values=model_inputs.pixel_values,
            image_grid_thw=model_inputs.image_grid_thw,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
        split_hidden_states = [entry[drop_idx:] for entry in split_hidden_states]
        prompt_embeds, encoder_attention_mask = _pad_qwen_hidden_states_to_fixed_length(
            split_hidden_states,
            target_length=self.tokenizer_max_length,
        )
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        return prompt_embeds, encoder_attention_mask

    raise AttributeError("Qwen processor-based prompt patch requires a pipeline with `processor` and `text_encoder`.")


def _extract_qwen_text_mask(
    attention_mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    seq_txt: int,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None

    raw_mask = attention_mask
    if raw_mask.dim() == 4:
        raw_mask = raw_mask[:, 0, 0, :]
    elif raw_mask.dim() == 3:
        raw_mask = raw_mask[:, 0, :]

    if raw_mask.dim() != 2 or raw_mask.shape[0] != batch_size or raw_mask.shape[1] < seq_txt:
        return None

    raw_mask = raw_mask[:, :seq_txt]
    if raw_mask.dtype != torch.bool:
        raw_mask = raw_mask.to(torch.bool)

    return raw_mask


def _qwen_double_stream_split_attention(
    self,
    attn: Attention,
    hidden_states: torch.FloatTensor,
    encoder_hidden_states: torch.FloatTensor,
    raw_text_mask: torch.Tensor,
    image_rotary_emb: Optional[torch.Tensor],
):
    seq_txt = encoder_hidden_states.shape[1]

    img_query = attn.to_q(hidden_states)
    img_key = attn.to_k(hidden_states)
    img_value = attn.to_v(hidden_states)

    txt_query = attn.add_q_proj(encoder_hidden_states)
    txt_key = attn.add_k_proj(encoder_hidden_states)
    txt_value = attn.add_v_proj(encoder_hidden_states)

    img_query = img_query.unflatten(-1, (attn.heads, -1))
    img_key = img_key.unflatten(-1, (attn.heads, -1))
    img_value = img_value.unflatten(-1, (attn.heads, -1))

    txt_query = txt_query.unflatten(-1, (attn.heads, -1))
    txt_key = txt_key.unflatten(-1, (attn.heads, -1))
    txt_value = txt_value.unflatten(-1, (attn.heads, -1))

    if attn.norm_q is not None:
        img_query = attn.norm_q(img_query)
    if attn.norm_k is not None:
        img_key = attn.norm_k(img_key)
    if attn.norm_added_q is not None:
        txt_query = attn.norm_added_q(txt_query)
    if attn.norm_added_k is not None:
        txt_key = attn.norm_added_k(txt_key)

    if image_rotary_emb is not None:
        from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

        img_freqs, txt_freqs = image_rotary_emb
        img_query = apply_rotary_emb_qwen(img_query, img_freqs, use_real=False)
        img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)
        txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
        txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

    img_attn_outputs = []
    txt_attn_outputs = []

    for sample_idx in range(hidden_states.shape[0]):
        true_txt_len = int(raw_text_mask[sample_idx].sum().item())
        true_txt_len = max(true_txt_len, 1)

        sample_txt_query = txt_query[sample_idx : sample_idx + 1, :true_txt_len]
        sample_txt_key = txt_key[sample_idx : sample_idx + 1, :true_txt_len]
        sample_txt_value = txt_value[sample_idx : sample_idx + 1, :true_txt_len]

        sample_joint_query = torch.cat([sample_txt_query, img_query[sample_idx : sample_idx + 1]], dim=1)
        sample_joint_key = torch.cat([sample_txt_key, img_key[sample_idx : sample_idx + 1]], dim=1)
        sample_joint_value = torch.cat([sample_txt_value, img_value[sample_idx : sample_idx + 1]], dim=1)

        sample_joint_hidden_states = dispatch_attention_fn(
            sample_joint_query,
            sample_joint_key,
            sample_joint_value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=getattr(self, "_attention_backend", None),
            parallel_config=getattr(self, "_parallel_config", None),
        )
        sample_joint_hidden_states = sample_joint_hidden_states.flatten(2, 3).to(sample_joint_query.dtype)

        sample_txt_attn_output = sample_joint_hidden_states[:, :true_txt_len]
        sample_img_attn_output = sample_joint_hidden_states[:, true_txt_len:]

        if true_txt_len < seq_txt:
            sample_txt_attn_output = torch.cat(
                [
                    sample_txt_attn_output,
                    sample_txt_attn_output.new_zeros((1, seq_txt - true_txt_len, sample_txt_attn_output.shape[-1])),
                ],
                dim=1,
            )

        img_attn_outputs.append(sample_img_attn_output)
        txt_attn_outputs.append(sample_txt_attn_output)

    img_attn_output = torch.cat(img_attn_outputs, dim=0)
    txt_attn_output = torch.cat(txt_attn_outputs, dim=0)

    img_attn_output = attn.to_out[0](img_attn_output.contiguous())
    if len(attn.to_out) > 1:
        img_attn_output = attn.to_out[1](img_attn_output)

    txt_attn_output = attn.to_add_out(txt_attn_output.contiguous())

    return img_attn_output, txt_attn_output


def patch_qwen_image_batch_size_fixes() -> None:
    from diffusers import QwenImagePipeline
    from diffusers.models.transformers.transformer_qwenimage import QwenDoubleStreamAttnProcessor2_0

    if not hasattr(QwenImagePipeline, "_simpletuner_original_get_qwen_prompt_embeds"):
        QwenImagePipeline._simpletuner_original_get_qwen_prompt_embeds = QwenImagePipeline._get_qwen_prompt_embeds
        QwenImagePipeline._get_qwen_prompt_embeds = _patched_qwen_prompt_embeds_from_tokenizer

    if not hasattr(QwenDoubleStreamAttnProcessor2_0, "_simpletuner_original_call"):
        QwenDoubleStreamAttnProcessor2_0._simpletuner_original_call = QwenDoubleStreamAttnProcessor2_0.__call__

        def patched_double_stream_call(
            self,
            attn: Attention,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor = None,
            encoder_hidden_states_mask: torch.FloatTensor = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            image_rotary_emb: Optional[torch.Tensor] = None,
        ):
            if encoder_hidden_states is None:
                return QwenDoubleStreamAttnProcessor2_0._simpletuner_original_call(
                    self,
                    attn,
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )

            raw_text_mask = encoder_hidden_states_mask
            if raw_text_mask is None:
                raw_text_mask = _extract_qwen_text_mask(
                    attention_mask,
                    batch_size=hidden_states.shape[0],
                    seq_txt=encoder_hidden_states.shape[1],
                )

            if raw_text_mask is None or hidden_states.shape[0] <= 1:
                return QwenDoubleStreamAttnProcessor2_0._simpletuner_original_call(
                    self,
                    attn,
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    attention_mask=attention_mask,
                    image_rotary_emb=image_rotary_emb,
                )

            return _qwen_double_stream_split_attention(
                self,
                attn,
                hidden_states,
                encoder_hidden_states,
                raw_text_mask,
                image_rotary_emb,
            )

        QwenDoubleStreamAttnProcessor2_0.__call__ = patched_double_stream_call

    try:
        from simpletuner.helpers.models.qwen_image.pipeline import QwenImageEditPipeline
        from simpletuner.helpers.models.qwen_image.pipeline_edit_plus import QwenImageEditPlusPipeline
    except ImportError as exc:
        logger.debug(f"Skipping local Qwen pipeline prompt padding patches: {exc}")
    else:
        local_pipeline_classes = [QwenImageEditPipeline, QwenImageEditPlusPipeline]
        for pipeline_cls in local_pipeline_classes:
            if not hasattr(pipeline_cls, "_simpletuner_original_get_qwen_prompt_embeds"):
                pipeline_cls._simpletuner_original_get_qwen_prompt_embeds = pipeline_cls._get_qwen_prompt_embeds
                pipeline_cls._get_qwen_prompt_embeds = _patched_qwen_prompt_embeds_from_processor

    logger.info("Patched Qwen Image prompt padding and attention mask handling for batch size > 1.")


patch_qwen_image_batch_size_fixes()


def patch_fsdp2_state_dict_loader() -> None:
    original = getattr(accelerate_fsdp_utils, "fsdp2_load_full_state_dict", None)
    if original is None:
        return

    def replacement(accelerator: Accelerator, model: torch.nn.Module, full_sd: dict):
        import torch.distributed as dist
        from torch.distributed.tensor import distribute_tensor

        meta_state = model.state_dict()
        shards: Dict[str, torch.Tensor] = {}

        def to_contiguous_and_cast(tensor: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
            out = tensor
            if ref.dtype.is_floating_point and out.dtype != ref.dtype:
                out = out.to(dtype=ref.dtype)
            if ref.is_contiguous() and not out.is_contiguous():
                out = out.contiguous()
            return out

        if accelerator.is_main_process:
            for (name, full_param), (meta_name, sharded_param) in zip(full_sd.items(), meta_state.items()):
                if name != meta_name:
                    raise RuntimeError(f"State-dict key mismatch: {name} vs {meta_name}")

                if hasattr(sharded_param, "device_mesh"):
                    mesh = sharded_param.device_mesh
                    tensor = full_param.detach().to(mesh.device_type)
                    dist.broadcast(tensor, src=0, group=dist.group.WORLD)
                    shard = distribute_tensor(tensor, mesh, sharded_param.placements)
                    shard = to_contiguous_and_cast(shard, sharded_param)
                else:
                    device = accelerator.device if accelerator.device.type != "meta" else torch.device("cpu")
                    shard = full_param.detach().to(device)
                    dist.broadcast(shard, src=0, group=dist.group.WORLD)
                    shard = to_contiguous_and_cast(shard, full_param)

                shards[name] = shard
        else:
            for name, sharded_param in meta_state.items():
                if hasattr(sharded_param, "device_mesh"):
                    mesh = sharded_param.device_mesh
                    tensor = torch.empty(sharded_param.size(), device=mesh.device_type, dtype=sharded_param.dtype)
                    dist.broadcast(tensor, src=0, group=dist.group.WORLD)
                    shard = distribute_tensor(tensor, mesh, sharded_param.placements)
                    shard = to_contiguous_and_cast(shard, sharded_param)
                else:
                    device = accelerator.device if accelerator.device.type != "meta" else torch.device("cpu")
                    shard = torch.empty(sharded_param.size(), device=device, dtype=sharded_param.dtype)
                    dist.broadcast(shard, src=0, group=dist.group.WORLD)
                shards[name] = shard

        model.load_state_dict(shards, assign=True)
        return model

    accelerate_fsdp_utils.fsdp2_load_full_state_dict = replacement


patch_fsdp2_state_dict_loader()
