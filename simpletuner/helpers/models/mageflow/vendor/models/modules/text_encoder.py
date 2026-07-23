"""Qwen3-VL text encoder: custom HF model + packing-aware forward patches + TextEncoder wrapper."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack

import torch
from loguru import logger
from torch import nn
from transformers import AutoProcessor, AutoTokenizer, Cache, Qwen3VLForConditionalGeneration
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from transformers.utils import ModelOutput

from ._attn_backend import flash_attn_varlen_func

# ===========================================================================
# Custom Qwen3-VL model (customizable forward output)
# ===========================================================================


@dataclass
class Qwen3VLModelOutput(ModelOutput):
    """Flexible output class for custom Qwen3-VL model."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    last_hidden_state: torch.FloatTensor | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    rope_deltas: torch.LongTensor | None = None


class CustomQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """
    Custom Qwen3-VL model that allows customizing the forward output.

    This class inherits from Qwen3VLForConditionalGeneration and provides
    hooks to customize what is returned from the forward pass.

    Example usage:
        ```python
        model = CustomQwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            attn_implementation="flash_attention_2"  # Use flash attention for faster inference
        )

        # Option 1: Use built-in output modes
        model.set_output_mode("embedding")  # Only return last hidden state (default)
        model.set_output_mode("full")       # Return everything
        model.set_output_mode("logits")     # Only return logits

        # Option 2: Set a custom output processor
        def my_custom_output(hidden_states, logits, outputs, **kwargs):
            return {"embeddings": hidden_states, "pooled": hidden_states.mean(dim=1)}
        model.set_output_processor(my_custom_output)
        ```
    """

    # Output mode constants
    OUTPUT_MODE_FULL = "full"
    OUTPUT_MODE_EMBEDDING = "embedding"
    OUTPUT_MODE_LOGITS = "logits"
    OUTPUT_MODE_HIDDEN = "hidden"

    def __init__(self, config):
        super().__init__(config)
        self._output_mode = self.OUTPUT_MODE_EMBEDDING
        self._skip_lm_head = True

    def set_output_mode(self, mode: str):
        """
        Set the output mode for the forward pass.

        Args:
            mode: One of:
                - "full": Return full Qwen3VLCausalLMOutputWithPast
                - "embedding": Only return last hidden state (skip lm_head) (default)
                - "logits": Only return logits
                - "hidden": Return all hidden states
        """
        valid_modes = [
            self.OUTPUT_MODE_FULL,
            self.OUTPUT_MODE_EMBEDDING,
            self.OUTPUT_MODE_LOGITS,
            self.OUTPUT_MODE_HIDDEN,
        ]
        if mode not in valid_modes:
            raise ValueError(f"Invalid output mode: {mode}. Must be one of {valid_modes}")
        self._output_mode = mode
        self._skip_lm_head = mode == self.OUTPUT_MODE_EMBEDDING

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> Qwen3VLCausalLMOutputWithPast | Qwen3VLModelOutput | dict | torch.Tensor:
        """
        Forward pass with customizable output.

        Returns different outputs based on the configured output mode or custom processor.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        # Get outputs from the base model (Qwen3VLModel)
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        # Get the last hidden state
        hidden_states = outputs[0]  # This is the last hidden state

        # Compute logits if not skipping lm_head
        logits = None
        if not self._skip_lm_head:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        # Compute loss if labels are provided
        loss = None
        if labels is not None and logits is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs)

        # Return based on output mode
        if self._output_mode == self.OUTPUT_MODE_EMBEDDING:
            return Qwen3VLModelOutput(
                last_hidden_state=hidden_states,
                past_key_values=outputs.past_key_values,
                attentions=outputs.attentions,
                rope_deltas=outputs.rope_deltas,
            )
        elif self._output_mode == self.OUTPUT_MODE_LOGITS:
            return logits
        elif self._output_mode == self.OUTPUT_MODE_HIDDEN:
            return Qwen3VLModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=outputs.hidden_states,
                past_key_values=outputs.past_key_values,
                attentions=outputs.attentions,
                rope_deltas=outputs.rope_deltas,
            )
        else:  # OUTPUT_MODE_FULL
            return Qwen3VLCausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                rope_deltas=outputs.rope_deltas,
            )


# ===========================================================================
# Packing-aware forward patches (cu_seqlens) for the Qwen3-VL text encoder
# ===========================================================================


def model_forward(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    # args for deepstack
    visual_pos_masks: torch.Tensor | None = None,
    deepstack_visual_embeds: list[torch.Tensor] | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple | BaseModelOutputWithPast:
    r"""
    visual_pos_masks (`torch.Tensor` of shape `(batch_size, seqlen)`, *optional*):
        The mask of the visual positions.
    deepstack_visual_embeds (`list[torch.Tensor]`, *optional*):
        The deepstack visual embeddings. The shape is (num_layers, visual_seqlen, embed_dim).
        The feature is extracted from the different visual encoder layers, and fed to the decoder
        hidden states. It's from the paper DeepStack(https://arxiv.org/abs/2406.04334).
    """
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    # torch.jit.trace() doesn't support cache objects in the output
    if use_cache and past_key_values is None and not torch.jit.is_tracing():
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    # the hard coded `3` is for temporal, height and width.
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        position_ids = position_ids[1:]
    else:
        text_position_ids = position_ids[0]

    if kwargs.get("cu_seqlens") is None:
        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    for layer_idx, decoder_layer in enumerate(self.layers):
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=text_position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = layer_outputs

        # add visual features to the hidden states of first several layers
        if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
            hidden_states = self._deepstack_process(
                hidden_states,
                visual_pos_masks,
                deepstack_visual_embeds[layer_idx],
            )

    hidden_states = self.norm(hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


def forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    cu_seqlens = kwargs.get("cu_seqlens", None)

    if cu_seqlens is None:
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )
    else:
        max_seqlen = torch.diff(cu_seqlens).max().item() if cu_seqlens is not None else None
        query_states = query_states.transpose(1, 2).squeeze(0)
        key_states = key_states.transpose(1, 2).squeeze(0)
        value_states = value_states.transpose(1, 2).squeeze(0)
        attn_output = flash_attn_varlen_func(
            q=query_states,
            k=key_states,
            v=value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
            window_size=(-1, -1),
            softmax_scale=self.head_dim**-0.5,
            dropout_p=0.0,
        )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def qwen3_patch_forward():
    """Patch the Qwen3-VL text model + attention forwards to support packed
    varlen (cu_seqlens) inputs used by ``TextEncoder.forward``."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextAttention, Qwen3VLTextModel

    Qwen3VLTextModel.forward = model_forward
    Qwen3VLTextAttention.forward = forward


# ===========================================================================
# TextEncoder wrapper (packed text -> DiT conditioning embeddings)
# ===========================================================================
_FA2_ALIASES = {"fa2", "flash-attn-2", "flash2", "flash-attention-2"}
_FA4_ALIASES = {"fa4", "flash-attn-4", "flash4", "flash-attention-4"}
_FLASH_HUB_ALIASES = {
    "flash-attn-2-hub",
    "flash-attn-3-hub",
    "flash-attn-3-varlen-hub",
    "flash-attn-4-hub",
    "flash-attn-varlen-hub",
    "flash-varlen-hub",
    "flash2-hub",
    "flash3-hub",
    "flash3-varlen-hub",
    "flash4-hub",
}
_SDPA_ALIASES = {
    "diffusers",
    "native-math",
    "sdpa",
    "scaled-dot-product-attention",
    "torch-sdpa",
}


def _resolve_hf_attn_impl(attn_type: str) -> str:
    """Map a project-level attn_type to a HuggingFace ``attn_implementation`` string.

    ``VF_HF_ATTN_IMPL`` env var, if set, takes precedence (useful for forcing
    sdpa on machines without flash-attn). For FA4 we additionally probe that
    the CUTE-DSL kernel is importable and (when available) ask the HF helper
    to confirm; if not, fall back to sdpa rather than crashing at load time.
    """
    override = os.environ.get("VF_HF_ATTN_IMPL")
    if override:
        return override

    name = attn_type.lower().strip().replace("_", "-")
    if name in _SDPA_ALIASES or name in _FLASH_HUB_ALIASES:
        return "sdpa"
    if name in _FA2_ALIASES:
        return "flash_attention_2"
    if name in _FA4_ALIASES:
        try:
            import flash_attn.cute  # noqa: F401

            fa4_importable = True
        except Exception:
            fa4_importable = False
        if fa4_importable:
            try:
                from transformers.utils.import_utils import is_flash_attn_4_available

                if is_flash_attn_4_available():
                    return "flash_attention_4"
            except ImportError:
                return "flash_attention_4"
        logger.warning(
            "attn_type=flash4 requested but flash_attn.cute is unavailable; " "falling back to sdpa for HF text encoder."
        )
        return "sdpa"
    if name in _SDPA_ALIASES:
        return "sdpa"
    raise ValueError(
        f"Unknown attn_type {attn_type!r}; expected one of " f"{sorted(_FA2_ALIASES | _FA4_ALIASES | _SDPA_ALIASES)}"
    )


SEQ_MULTI_OF = 32


class TextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        version: str,
        tokenizer_max_length: int,
        prompt_template: dict | None,
        dit_structure: dict,
        use_packed_text_infer: bool = False,
        attn_type: str = "flash2",
        **hf_kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.tokenizer_max_length = tokenizer_max_length
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(version)
        self.tokenizer.padding_side = "right"

        hf_attn_impl = _resolve_hf_attn_impl(attn_type)
        logger.info(f"TextEncoder attn_type={attn_type} -> attn_implementation={hf_attn_impl}")

        logger.info("init vl model: qwen3")
        self.hf_module: CustomQwen3VLForConditionalGeneration = CustomQwen3VLForConditionalGeneration.from_pretrained(
            version, attn_implementation=hf_attn_impl, **hf_kwargs
        )

        # Use local_files_only if version is a local path (absolute path or contains path separators)
        is_local = version.startswith("/") or os.sep in version
        self.processor = AutoProcessor.from_pretrained(version, local_files_only=is_local)

        self.hf_module = self.hf_module.eval().requires_grad_(False)

        prompt_template = prompt_template or {}
        self.prompt_template_encode = prompt_template.get("template", "")
        self.prompt_template_encode_start_idx = prompt_template.get("start_idx", 0)
        self.dit_structure = dit_structure
        self.use_packed_text_infer = use_packed_text_infer

    def forward(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        inputs: dict | None = None,
        drop_idx_override: int | None = None,
    ):
        """Encode packed text (varlen ``cu_seqlens``) into DiT conditioning embeddings.

        This is the sole text-embedding path — both t2i and edit call it. Uses
        Flash-Attention-2's varlen capability (``cu_seqlens``) via the patched
        Qwen3-VL forward to process several concatenated sequences in a single
        launch, with no padding. Verified numerically identical to a padded-batch
        forward with per-sample cu_seqlens isolation (zero cross-contamination).

        Args:
            input_ids: Packed token ids ``[Total_L]``.
            cu_seqlens: Cumulative sequence lengths ``[B+1]``.
            inputs: Optional dict with additional model inputs (e.g. ``pixel_values``,
                ``image_grid_thw`` for the multimodal edit path). Passed through to
                the text encoder.
            drop_idx_override: If set, override the number of leading (system-prompt)
                tokens to drop per sequence. Use 0 for multi-turn where the system
                prompt is embedded in the conversation and should not be stripped.

        Returns:
            dict with keys:
                - ``txt``: text embeddings ``[Total_L - B*drop_idx, D]`` (system prompt dropped)
                - ``vec``: pooled text embeddings ``[B, D]``
                - ``txt_seq_lens``: per-sequence lengths ``[B]`` (after dropping system prompt)
        """
        # Compute seqlens from cu_seqlens
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        seqlens_list = seqlens.cpu().tolist()

        # Build position_ids for packing: each sequence starts from 0
        position_ids_list = []
        for length in seqlens_list:
            position_ids_list.append(torch.arange(length, device=input_ids.device))
        position_ids = torch.cat(position_ids_list)  # [Total_L]

        # Reshape for model input: [1, Total_L]
        input_ids_packed = input_ids.unsqueeze(0)  # [1, Total_L]
        position_ids_packed = position_ids.unsqueeze(0)  # [1, Total_L]

        # Move to text encoder device
        device = self.hf_module.device
        input_ids_packed = input_ids_packed.to(device)
        position_ids_packed = position_ids_packed.to(device)

        # Get text embeddings (the text encoder is always frozen)
        with torch.no_grad():
            forward_kwargs = {
                "input_ids": input_ids_packed,
                "cu_seqlens": cu_seqlens,
                "position_ids": position_ids_packed,
                "output_hidden_states": False,
                "max_seqlen": None,
            }
            # Pass multimodal inputs for edit mode (reference images)
            if inputs is not None:
                for key in ("pixel_values", "image_grid_thw"):
                    if key in inputs and inputs[key] is not None:
                        val = inputs[key]
                        if hasattr(val, "to"):
                            val = val.to(device)
                        forward_kwargs[key] = val
            outputs = self.hf_module(**forward_kwargs)

        # Extract hidden state
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            hidden = outputs.last_hidden_state  # [1, Total_L, D]
        elif hasattr(outputs, "hidden_states"):
            hidden = outputs.hidden_states[-1]

        # Remove batch dimension: [Total_L, D]
        hidden = hidden.squeeze(0)

        # Get drop_idx (system prompt length to skip).
        # For multi-turn, drop_idx_override=0 is passed since system prompt is in the messages.
        if drop_idx_override is not None:
            drop_idx = drop_idx_override
        else:
            drop_idx = self.prompt_template_encode_start_idx

        # Split hidden states by sequence
        hidden_split = torch.split(hidden, seqlens_list, dim=0)

        # Extract valid embeddings (drop system prompt) and compute vec
        txt_list = []
        vec_list = []
        valid_lengths = []

        for h in hidden_split:
            # Drop system prompt tokens
            h_valid = h[drop_idx:]  # [seq_len - drop_idx, D]
            txt_list.append(h_valid)
            valid_lengths.append(h_valid.shape[0])

            # Compute pooled embedding (mean of valid tokens only, after dropping system prompt)
            vec_list.append(h_valid.mean(dim=0))  # [D]

        txt = torch.cat(txt_list, dim=0)  # [Total_valid, D]
        vec = torch.stack(vec_list, dim=0)  # [B, D]
        txt_seq_lens = torch.tensor(valid_lengths, device=input_ids.device)

        result = {
            "txt": txt,
            "vec": vec,
            "txt_seq_lens": txt_seq_lens,
        }

        return result

    # ------------------------------------------------------------------
    # Mandatory content-policy screening (same Qwen3-VL weights)
    # ------------------------------------------------------------------
    # The policy classifier lives HERE, on the text encoder, so it runs on the
    # exact weights that produce the diffusion conditioning and is not a
    # separable, toggleable pre-pass in the pipeline. The classifier needs
    # autoregressive ``.generate()`` (JSON verdict) whereas conditioning is a
    # single embedding forward — they cannot be one GPU forward without a
    # trained classification head, so "fused" here means: same module, same
    # weights, always run, FAIL-CLOSED (any error blocks).

    def screen_text(self, prompt: str, max_new_tokens: int = 160):
        """Classify a text-to-image ``prompt`` against the content policy.

        Returns a ``FilterVerdict``. FAIL-CLOSED: any error (generation, parse)
        returns ``violates=True`` so a broken classifier cannot be used as a
        bypass. An empty prompt is not a violation.
        """
        from .mage_text import CONTENT_FILTER_SYSTEM, FilterVerdict, _extract_json_object, _full_output_mode

        if not prompt or not prompt.strip():
            return FilterVerdict(False, [], "empty prompt", "")
        try:
            tokenizer = self.tokenizer
            hf = self.hf_module
            device = next(hf.parameters()).device

            messages = [
                {"role": "system", "content": CONTENT_FILTER_SYSTEM},
                {"role": "user", "content": f"Prompt to classify:\n{prompt}"},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)

            eos_id = tokenizer.eos_token_id
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

            with _full_output_mode(hf), torch.no_grad():
                out = hf.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id, eos_token_id=eos_id
                )
            gen = tokenizer.decode(out[0, inputs.input_ids.shape[1] :], skip_special_tokens=True).strip()

            parsed = _extract_json_object(gen)
            violates = bool(parsed.get("violates", False))
            cats = [c for c in (parsed.get("categories", []) or []) if isinstance(c, str)]
            reason = str(parsed.get("reason", "")).strip()
            return FilterVerdict(violates, cats, reason, gen)
        except Exception as exc:  # noqa: BLE001
            # FAIL-CLOSED: block on any screening error.
            return FilterVerdict(True, ["policy"], f"filter error (blocked): {type(exc).__name__}: {exc}", "")

    def screen_edit(self, prompt: str, ref_images, max_new_tokens: int = 192):
        """Classify an image-EDIT request (source image(s) + instruction).

        Considers BOTH the source image(s) and the instruction via multimodal
        Qwen3-VL. Falls back to :meth:`screen_text` when no image is given.
        FAIL-CLOSED: any error returns ``violates=True``.
        """
        from PIL import Image

        from .mage_text import CONTENT_FILTER_EDIT_SYSTEM, FilterVerdict, _extract_json_object, _full_output_mode

        pils = [ref_images] if isinstance(ref_images, Image.Image) else list(ref_images)
        pils = [p.convert("RGB") for p in pils if p is not None]
        if not pils:
            return self.screen_text(prompt, max_new_tokens=max_new_tokens)

        instruction = (prompt or "").strip() or "(no textual instruction)"
        try:
            processor = self.processor
            tokenizer = self.tokenizer
            hf = self.hf_module
            device = next(hf.parameters()).device

            user_content = [{"type": "image"} for _ in pils]
            user_content.append(
                {
                    "type": "text",
                    "text": (
                        f"There {'is' if len(pils) == 1 else 'are'} {len(pils)} source "
                        f"image(s) above. Edit instruction: {instruction}\n"
                        "Classify this edit request."
                    ),
                }
            )
            messages = [
                {"role": "system", "content": CONTENT_FILTER_EDIT_SYSTEM},
                {"role": "user", "content": user_content},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=pils, padding=True, return_tensors="pt").to(device)

            eos_id = tokenizer.eos_token_id
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

            # Keep only the kwargs Qwen3-VL .generate() consumes.
            gen_inputs = {
                k: inputs[k]
                for k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")
                if k in inputs and inputs[k] is not None
            }
            input_len = gen_inputs["input_ids"].shape[1]

            with _full_output_mode(hf), torch.no_grad():
                out = hf.generate(
                    **gen_inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=pad_id, eos_token_id=eos_id
                )
            gen = tokenizer.decode(out[0, input_len:], skip_special_tokens=True).strip()

            parsed = _extract_json_object(gen)
            violates = bool(parsed.get("violates", False))
            cats = [c for c in (parsed.get("categories", []) or []) if isinstance(c, str)]
            reason = str(parsed.get("reason", "")).strip()
            return FilterVerdict(violates, cats, reason, gen)
        except Exception as exc:  # noqa: BLE001
            # FAIL-CLOSED: block on any screening error.
            return FilterVerdict(True, ["policy"], f"edit filter error (blocked): {type(exc).__name__}: {exc}", "")
