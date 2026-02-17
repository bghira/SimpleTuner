"""Inject GLIGEN grounding layers (position_net + fuser) into a pretrained UNet."""

from __future__ import annotations

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


def inject_gligen_layers(
    unet: nn.Module,
    positive_len: int,
    cross_attention_dim: int,
    feature_type: str = "text-image",
) -> nn.Module:
    """Add ``GatedSelfAttentionDense`` fuser layers and a ``position_net`` to an existing UNet.

    The fuser ``alpha_attn`` / ``alpha_dense`` gates are initialized to 0
    by the diffusers implementation, so the model starts identical to vanilla
    SD and gradually learns to incorporate grounding signals.

    Args:
        unet: A ``UNet2DConditionModel`` (or compatible) instance.
        positive_len: Dimensionality fed into position_net for both text and
            image branches.  Typically ``cross_attention_dim``.
        cross_attention_dim: The UNet's cross-attention hidden size.
        feature_type: ``"text-image"`` for dual branch, ``"text-only"`` for
            text-only grounding.
    Returns:
        The modified UNet (same object, mutated in-place).
    """
    from diffusers.models.attention import BasicTransformerBlock, GatedSelfAttentionDense
    from diffusers.models.embeddings import GLIGENTextBoundingboxProjection

    # 1. Add position_net
    unet.position_net = GLIGENTextBoundingboxProjection(
        positive_len=positive_len,
        out_dim=cross_attention_dim,
        feature_type=feature_type,
    )
    logger.info(
        f"Injected position_net (positive_len={positive_len}, "
        f"out_dim={cross_attention_dim}, feature_type={feature_type})"
    )

    # 2. Add fuser to every BasicTransformerBlock that doesn't already have one
    fuser_count = 0
    for module in unet.modules():
        if isinstance(module, BasicTransformerBlock) and not hasattr(module, "fuser"):
            query_dim = module.attn1.to_q.in_features
            n_heads = module.attn1.heads
            d_head = query_dim // n_heads
            module.fuser = GatedSelfAttentionDense(
                query_dim=query_dim,
                context_dim=cross_attention_dim,
                n_heads=n_heads,
                d_head=d_head,
            )
            fuser_count += 1

    logger.info(f"Injected {fuser_count} GatedSelfAttentionDense fuser layers")
    return unet


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
