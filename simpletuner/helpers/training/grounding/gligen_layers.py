"""Inject GLIGEN grounding layers (position_net + fuser) into a pretrained model."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Ordered list of attribute names to probe when looking for the attention
# sub-module inside a transformer block.
_ATTN_ATTR_CANDIDATES = ("attn1", "attn", "self_attention")


def _extract_block_dims(
    block: nn.Module,
    attn_attr: str | None = None,
) -> tuple[int, int] | None:
    """Return ``(query_dim, n_heads)`` from a block's attention sub-module.

    Tries ``attn_attr`` first (if given), then falls back to the candidates in
    ``_ATTN_ATTR_CANDIDATES``.  Returns ``None`` when no usable attention module
    is found.
    """
    candidates = (attn_attr,) if attn_attr else _ATTN_ATTR_CANDIDATES

    for attr in candidates:
        attn = getattr(block, attr, None)
        if attn is None:
            continue

        # Standard diffusers Attention: has ``to_q`` and ``heads``
        to_q = getattr(attn, "to_q", None)
        if to_q is not None and hasattr(attn, "heads"):
            return to_q.in_features, attn.heads

        # Kandinsky5Attention (and similar): use ``out_layer`` + ``num_heads``
        out_layer = getattr(attn, "out_layer", None)
        num_heads = getattr(attn, "num_heads", None)
        if out_layer is not None and num_heads is not None:
            return out_layer.in_features, num_heads

    return None


def inject_gligen_layers(
    model: nn.Module,
    positive_len: int,
    cross_attention_dim: int,
    feature_type: str = "text-image",
    block_types: tuple[type, ...] | None = None,
    attn_attr: str | None = None,
) -> nn.Module:
    """Add ``GatedSelfAttentionDense`` fuser layers and a ``position_net`` to a model.

    Works with both UNet (``BasicTransformerBlock``) and any transformer whose
    blocks expose an attention sub-module with ``(to_q, heads)`` or
    ``(out_layer, num_heads)``.

    Args:
        model: A ``UNet2DConditionModel``, transformer, or similar ``nn.Module``.
        positive_len: Dimensionality fed into position_net for both text and
            image branches.
        cross_attention_dim: The model's cross-attention hidden size (also
            ``out_dim`` of position_net and ``context_dim`` of the fuser).
        feature_type: ``"text-image"`` for dual branch, ``"text-only"`` for
            text-only grounding.
        block_types: Explicit tuple of block classes to inject fusers into.
            When ``None`` (default), auto-detects all blocks that have a
            recognisable attention sub-module.
        attn_attr: Force a specific attribute name for the attention sub-module
            (e.g. ``"self_attention"``).  When ``None``, tries ``attn1``,
            ``attn``, ``self_attention`` in order.
    Returns:
        The modified model (same object, mutated in-place).
    """
    from diffusers.models.attention import GatedSelfAttentionDense
    from diffusers.models.embeddings import GLIGENTextBoundingboxProjection

    # 1. Add position_net
    model.position_net = GLIGENTextBoundingboxProjection(
        positive_len=positive_len,
        out_dim=cross_attention_dim,
        feature_type=feature_type,
    )
    logger.info(
        f"Injected position_net (positive_len={positive_len}, "
        f"out_dim={cross_attention_dim}, feature_type={feature_type})"
    )

    # 2. Collect qualifying blocks first (avoid mutating during iteration)
    targets: list[tuple[nn.Module, int, int]] = []  # (module, query_dim, n_heads)
    seen_ids: set[int] = set()
    for module in model.modules():
        mid = id(module)
        if mid in seen_ids:
            continue
        seen_ids.add(mid)

        if hasattr(module, "fuser"):
            continue
        if block_types is not None and not isinstance(module, block_types):
            continue

        dims = _extract_block_dims(module, attn_attr)
        if dims is None:
            continue

        # Skip raw attention modules — we only want their parent blocks.
        # A block should contain children beyond just projection layers.
        if hasattr(module, "to_q") or hasattr(module, "to_k"):
            continue

        targets.append((module, dims[0], dims[1]))

    # 3. Inject fusers into collected targets
    for module, query_dim, n_heads in targets:
        d_head = query_dim // n_heads
        module.fuser = GatedSelfAttentionDense(
            query_dim=query_dim,
            context_dim=cross_attention_dim,
            n_heads=n_heads,
            d_head=d_head,
        )

    logger.info(f"Injected {len(targets)} GatedSelfAttentionDense fuser layers")
    return model


def apply_grounding_fuser(
    fuser: nn.Module,
    hidden_states: torch.Tensor,
    objs: torch.Tensor,
    txt_len: int | None = None,
    tokens_per_frame: int | None = None,
    num_frames: int = 1,
) -> torch.Tensor:
    """Apply a ``GatedSelfAttentionDense`` fuser to (possibly mixed) hidden states.

    Handles three layouts transparently:

    * **Image-only** (``txt_len is None``, ``num_frames == 1``):
      ``hidden_states`` is ``(B, S_image, D)``.
    * **Text + image** (``txt_len`` set):
      ``hidden_states`` is ``(B, txt_len + S_image, D)``; only the image
      portion is fused, then recombined with text.
    * **Video** (``num_frames > 1``):
      Image tokens are reshaped from ``(B, T*S_frame, D)`` to
      ``(B*T, S_frame, D)``, fused per-frame, then unfolded back.

    Args:
        fuser: A ``GatedSelfAttentionDense`` instance.
        hidden_states: ``(B, S, D)`` hidden states tensor.
        objs: Grounding objects from ``position_net``.  Shape is either
            ``(B, N_objs, D_ctx)`` for images or ``(B*T, N_objs, D_ctx)``
            for video (pre-flattened by ``_build_grounding_position_net_kwargs``).
        txt_len: If set, the first ``txt_len`` tokens in the sequence are
            text tokens that should not be fused.
        tokens_per_frame: Spatial tokens per frame.  Required when
            ``num_frames > 1``.
        num_frames: Number of temporal frames (1 for images).
    Returns:
        Updated ``hidden_states`` with the same shape as the input.
    """
    # 1. Split text / image if needed
    text_part = None
    if txt_len is not None:
        text_part = hidden_states[:, :txt_len]
        image_part = hidden_states[:, txt_len:]
    else:
        image_part = hidden_states

    # 2. Video: fold (B, T*S_frame, D) -> (B*T, S_frame, D)
    B = image_part.shape[0]
    did_fold = False
    if num_frames > 1:
        assert tokens_per_frame is not None, "tokens_per_frame is required when num_frames > 1"
        T = num_frames
        S_frame = tokens_per_frame
        assert image_part.shape[1] == T * S_frame, (
            f"Expected {T * S_frame} image tokens (T={T} * S_frame={S_frame}), " f"got {image_part.shape[1]}"
        )
        D = image_part.shape[2]
        image_part = image_part.reshape(B * T, S_frame, D)
        did_fold = True

    # 3. Apply fuser
    image_part = fuser(image_part, objs)

    # 4. Video: unfold back to (B, T*S_frame, D)
    if did_fold:
        image_part = image_part.reshape(B, T * S_frame, D)

    # 5. Recombine text + image
    if text_part is not None:
        return torch.cat([text_part, image_part], dim=1)
    return image_part


def get_gligen_trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    """Collect parameters from position_net, fuser, and projection layers."""
    params = []
    for name, p in model.named_parameters():
        if "fuser" in name or "position_net" in name or "grounding_projection" in name:
            p.requires_grad = True
            params.append(p)
        else:
            p.requires_grad = False
    return params
