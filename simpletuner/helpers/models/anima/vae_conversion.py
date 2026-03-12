# Vendored from diffusers-anima: /src/diffusers-anima/src/diffusers_anima/pipelines/anima/vae_conversion.py
# Adapted for SimpleTuner local imports.

"""Anima VAE checkpoint key conversion utilities."""

from __future__ import annotations

import torch

_ANIMA_VAE_RESIDUAL_KEY_MAP = {
    "residual.0.gamma": "norm1.gamma",
    "residual.2.weight": "conv1.weight",
    "residual.2.bias": "conv1.bias",
    "residual.3.gamma": "norm2.gamma",
    "residual.6.weight": "conv2.weight",
    "residual.6.bias": "conv2.bias",
}

_ANIMA_VAE_DECODER_UP_RESNET_MAP = {
    0: "decoder.up_blocks.0.resnets.0",
    1: "decoder.up_blocks.0.resnets.1",
    2: "decoder.up_blocks.0.resnets.2",
    4: "decoder.up_blocks.1.resnets.0",
    5: "decoder.up_blocks.1.resnets.1",
    6: "decoder.up_blocks.1.resnets.2",
    8: "decoder.up_blocks.2.resnets.0",
    9: "decoder.up_blocks.2.resnets.1",
    10: "decoder.up_blocks.2.resnets.2",
    12: "decoder.up_blocks.3.resnets.0",
    13: "decoder.up_blocks.3.resnets.1",
    14: "decoder.up_blocks.3.resnets.2",
}

_ANIMA_VAE_DECODER_UP_UPSAMPLER_MAP = {
    3: "decoder.up_blocks.0.upsamplers.0",
    7: "decoder.up_blocks.1.upsamplers.0",
    11: "decoder.up_blocks.2.upsamplers.0",
}


def _map_residual_tail(tail: str) -> str | None:
    mapped = _ANIMA_VAE_RESIDUAL_KEY_MAP.get(tail)
    if mapped is not None:
        return mapped
    if tail.startswith("shortcut."):
        return "conv_shortcut." + tail.split(".", maxsplit=1)[1]
    if tail.startswith(("resample.", "time_conv.")):
        return tail
    return None


def _convert_anima_vae_head_key(key: str, *, prefix: str) -> str | None:
    if key == f"{prefix}.head.0.gamma":
        return f"{prefix}.norm_out.gamma"
    head_prefix = f"{prefix}.head.2."
    if key.startswith(head_prefix):
        return f"{prefix}.conv_out.{key.split('.', maxsplit=3)[3]}"
    return None


def _convert_anima_vae_mid_resnet_key(key: str, *, prefix: str, source_index: int, target_index: int) -> str | None:
    source_prefix = f"{prefix}.middle.{source_index}."
    if not key.startswith(source_prefix):
        return None
    mapped = _ANIMA_VAE_RESIDUAL_KEY_MAP.get(key.removeprefix(source_prefix))
    if mapped is None:
        return None
    return f"{prefix}.mid_block.resnets.{target_index}.{mapped}"


def _convert_anima_vae_mid_attention_key(key: str, *, prefix: str) -> str | None:
    middle_prefix = f"{prefix}.middle.1."
    if not key.startswith(middle_prefix):
        return None
    tail = key.removeprefix(middle_prefix)
    if tail == "norm.gamma":
        return f"{prefix}.mid_block.attentions.0.norm.gamma"
    if tail.startswith(("to_qkv.", "proj.")):
        return f"{prefix}.mid_block.attentions.0.{tail}"
    return None


def _convert_anima_vae_encoder_key(key: str) -> str | None:
    if key.startswith("encoder.conv1."):
        return "encoder.conv_in." + key.split(".", maxsplit=2)[2]

    mapped_head = _convert_anima_vae_head_key(key, prefix="encoder")
    if mapped_head is not None:
        return mapped_head

    for source_index, target_index in ((0, 0), (2, 1)):
        mapped_resnet = _convert_anima_vae_mid_resnet_key(
            key,
            prefix="encoder",
            source_index=source_index,
            target_index=target_index,
        )
        if mapped_resnet is not None:
            return mapped_resnet

    mapped_attention = _convert_anima_vae_mid_attention_key(key, prefix="encoder")
    if mapped_attention is not None:
        return mapped_attention

    if key.startswith("encoder.downsamples."):
        rest = key.removeprefix("encoder.downsamples.")
        idx, tail = rest.split(".", maxsplit=1)
        mapped = _map_residual_tail(tail)
        if mapped is not None:
            return f"encoder.down_blocks.{idx}.{mapped}"

    return None


def _convert_anima_vae_decoder_upsample_key(key: str) -> str | None:
    if not key.startswith("decoder.upsamples."):
        return None

    rest = key.removeprefix("decoder.upsamples.")
    idx_text, tail = rest.split(".", maxsplit=1)
    idx = int(idx_text)
    mapped = _map_residual_tail(tail)
    if mapped is not None and idx in _ANIMA_VAE_DECODER_UP_RESNET_MAP:
        return f"{_ANIMA_VAE_DECODER_UP_RESNET_MAP[idx]}.{mapped}"
    if idx in _ANIMA_VAE_DECODER_UP_UPSAMPLER_MAP and tail.startswith(("resample.", "time_conv.")):
        return f"{_ANIMA_VAE_DECODER_UP_UPSAMPLER_MAP[idx]}.{tail}"
    return None


def _convert_anima_vae_decoder_key(key: str) -> str | None:
    if key.startswith("decoder.conv1."):
        return "decoder.conv_in." + key.split(".", maxsplit=2)[2]

    mapped_head = _convert_anima_vae_head_key(key, prefix="decoder")
    if mapped_head is not None:
        return mapped_head

    for source_index, target_index in ((0, 0), (2, 1)):
        mapped_resnet = _convert_anima_vae_mid_resnet_key(
            key,
            prefix="decoder",
            source_index=source_index,
            target_index=target_index,
        )
        if mapped_resnet is not None:
            return mapped_resnet

    mapped_attention = _convert_anima_vae_mid_attention_key(key, prefix="decoder")
    if mapped_attention is not None:
        return mapped_attention

    return _convert_anima_vae_decoder_upsample_key(key)


def _convert_anima_vae_key(key: str) -> str:
    if key.startswith("conv1."):
        return "quant_conv." + key.split(".", maxsplit=1)[1]
    if key.startswith("conv2."):
        return "post_quant_conv." + key.split(".", maxsplit=1)[1]

    mapped_encoder = _convert_anima_vae_encoder_key(key)
    if mapped_encoder is not None:
        return mapped_encoder

    mapped_decoder = _convert_anima_vae_decoder_key(key)
    if mapped_decoder is not None:
        return mapped_decoder

    raise KeyError(key)


def convert_anima_vae_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert an Anima-format VAE state dict to Diffusers AutoencoderKLQwenImage layout."""
    if "encoder.conv_in.weight" in state_dict and "quant_conv.weight" in state_dict:
        return dict(state_dict)

    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        mapped_key = _convert_anima_vae_key(key)
        if mapped_key in converted:
            raise RuntimeError(f"Duplicate converted VAE key: {mapped_key}")
        converted[mapped_key] = value
    return converted
