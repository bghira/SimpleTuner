from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

import torch
from accelerate import init_empty_weights
from safetensors import safe_open

from simpletuner.helpers.models.ltxvideo2.audio_autoencoder import AutoencoderKLLTX2Audio
from simpletuner.helpers.models.ltxvideo2.autoencoder import AutoencoderKLLTX2Video
from simpletuner.helpers.models.ltxvideo2.connectors import LTX2TextConnectors
from simpletuner.helpers.models.ltxvideo2.transformer import LTX2VideoTransformer3DModel
from simpletuner.helpers.models.ltxvideo2.vocoder import LTX2Vocoder

LTX_2_0_TRANSFORMER_KEYS_RENAME_DICT = {
    "patchify_proj": "proj_in",
    "audio_patchify_proj": "audio_proj_in",
    "av_ca_video_scale_shift_adaln_single": "av_cross_attn_video_scale_shift",
    "av_ca_a2v_gate_adaln_single": "av_cross_attn_video_a2v_gate",
    "av_ca_audio_scale_shift_adaln_single": "av_cross_attn_audio_scale_shift",
    "av_ca_v2a_gate_adaln_single": "av_cross_attn_audio_v2a_gate",
    "scale_shift_table_a2v_ca_video": "video_a2v_cross_attn_scale_shift_table",
    "scale_shift_table_a2v_ca_audio": "audio_a2v_cross_attn_scale_shift_table",
    "q_norm": "norm_q",
    "k_norm": "norm_k",
}

LTX_2_0_VIDEO_VAE_RENAME_DICT = {
    "down_blocks.0": "down_blocks.0",
    "down_blocks.1": "down_blocks.0.downsamplers.0",
    "down_blocks.2": "down_blocks.1",
    "down_blocks.3": "down_blocks.1.downsamplers.0",
    "down_blocks.4": "down_blocks.2",
    "down_blocks.5": "down_blocks.2.downsamplers.0",
    "down_blocks.6": "down_blocks.3",
    "down_blocks.7": "down_blocks.3.downsamplers.0",
    "down_blocks.8": "mid_block",
    "up_blocks.0": "mid_block",
    "up_blocks.1": "up_blocks.0.upsamplers.0",
    "up_blocks.2": "up_blocks.0",
    "up_blocks.3": "up_blocks.1.upsamplers.0",
    "up_blocks.4": "up_blocks.1",
    "up_blocks.5": "up_blocks.2.upsamplers.0",
    "up_blocks.6": "up_blocks.2",
    "res_blocks": "resnets",
    "per_channel_statistics.mean-of-means": "latents_mean",
    "per_channel_statistics.std-of-means": "latents_std",
}

LTX_2_0_AUDIO_VAE_RENAME_DICT: Dict[str, str] = {}

LTX_2_0_VOCODER_RENAME_DICT = {
    "ups": "upsamplers",
    "resblocks": "resnets",
    "conv_pre": "conv_in",
    "conv_post": "conv_out",
}

LTX_2_0_VAE_SPECIAL_KEYS_REMAP = {
    "per_channel_statistics.channel": None,
    "per_channel_statistics.mean-of-stds": None,
}
LTX_2_0_AUDIO_VAE_SPECIAL_KEYS_REMAP: Dict[str, Any] = {}
LTX_2_0_VOCODER_SPECIAL_KEYS_REMAP = {}

LTX_2_0_CONNECTORS_KEYS_RENAME_DICT = {
    "connectors.": "",
    "video_embeddings_connector": "video_connector",
    "audio_embeddings_connector": "audio_connector",
    "transformer_1d_blocks": "transformer_blocks",
    "text_embedding_projection.aggregate_embed": "text_proj_in",
    "q_norm": "norm_q",
    "k_norm": "norm_k",
}


def _normalize_prefix(prefix: str) -> str:
    return prefix if prefix.endswith(".") else f"{prefix}."


def update_state_dict_inplace(state_dict: Dict[str, Any], old_key: str, new_key: str) -> None:
    if old_key == new_key:
        state_dict[new_key] = state_dict.pop(old_key)
        return
    state_dict[new_key] = state_dict.pop(old_key)


def _remove_key_inplace(key: str, state_dict: Dict[str, Any]) -> None:
    state_dict.pop(key)


def _convert_ltx2_transformer_adaln_single(key: str, state_dict: Dict[str, Any]) -> None:
    if ".weight" not in key and ".bias" not in key:
        return
    if key.startswith("adaln_single."):
        new_key = key.replace("adaln_single.", "time_embed.")
        update_state_dict_inplace(state_dict, key, new_key)
    if key.startswith("audio_adaln_single."):
        new_key = key.replace("audio_adaln_single.", "audio_time_embed.")
        update_state_dict_inplace(state_dict, key, new_key)


def _convert_ltx2_audio_vae_per_channel_statistics(key: str, state_dict: Dict[str, Any]) -> None:
    if key.startswith("per_channel_statistics"):
        new_key = ".".join(["decoder", key])
        update_state_dict_inplace(state_dict, key, new_key)


def _split_transformer_and_connector_state_dict(
    state_dict: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    connector_prefixes = (
        "video_embeddings_connector",
        "audio_embeddings_connector",
        "transformer_1d_blocks",
        "text_embedding_projection.aggregate_embed",
        "connectors.",
        "video_connector",
        "audio_connector",
        "text_proj_in",
    )
    transformer_state_dict, connector_state_dict = {}, {}
    for key, value in state_dict.items():
        if key.startswith(connector_prefixes):
            connector_state_dict[key] = value
        else:
            transformer_state_dict[key] = value
    return transformer_state_dict, connector_state_dict


def get_model_state_dict_from_combined_ckpt(combined_ckpt: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    prefix = _normalize_prefix(prefix)
    model_state_dict: Dict[str, Any] = {}
    for param_name, param in combined_ckpt.items():
        if param_name.startswith(prefix):
            model_state_dict[param_name.replace(prefix, "")] = param
    if prefix == "model.diffusion_model.":
        connector_key = "text_embedding_projection.aggregate_embed.weight"
        if connector_key in combined_ckpt and connector_key not in model_state_dict:
            model_state_dict[connector_key] = combined_ckpt[connector_key]
    return model_state_dict


def load_ltx2_state_dict_from_checkpoint(checkpoint_path: str, prefix: str) -> Dict[str, Any]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"LTX-2 checkpoint not found at {checkpoint_path}")

    prefix = _normalize_prefix(prefix)
    _, ext = os.path.splitext(checkpoint_path)
    if ext in [".safetensors", ".sft"]:
        state_dict: Dict[str, Any] = {}
        with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            for key in keys:
                if key.startswith(prefix):
                    state_dict[key.replace(prefix, "")] = handle.get_tensor(key)
            if prefix == "model.diffusion_model.":
                connector_key = "text_embedding_projection.aggregate_embed.weight"
                if connector_key in keys and connector_key not in state_dict:
                    state_dict[connector_key] = handle.get_tensor(connector_key)
        return state_dict

    combined_ckpt = torch.load(checkpoint_path, map_location="cpu")
    return get_model_state_dict_from_combined_ckpt(combined_ckpt, prefix)


def load_ltx2_metadata_config(checkpoint_path: str) -> Dict[str, Any]:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"LTX-2 checkpoint not found at {checkpoint_path}")
    _, ext = os.path.splitext(checkpoint_path)
    if ext not in [".safetensors", ".sft"]:
        return {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    raw_config = metadata.get("config")
    if not raw_config:
        return {}
    try:
        return json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise ValueError("Unable to parse LTX-2 metadata config JSON.") from exc


def _extract_audio_vae_config_from_metadata(metadata_config: Dict[str, Any]) -> Dict[str, Any] | None:
    if not metadata_config:
        return None
    audio_cfg = metadata_config.get("audio_vae")
    if not isinstance(audio_cfg, dict):
        return None
    model_cfg = audio_cfg.get("model", {}).get("params", {})
    ddconfig = model_cfg.get("ddconfig", {})
    preprocessing = audio_cfg.get("preprocessing", {})
    stft_cfg = preprocessing.get("stft", {})
    mel_cfg = preprocessing.get("mel", {})
    audio_cfg_meta = preprocessing.get("audio", {})

    if not ddconfig:
        return None

    attn_resolutions = ddconfig.get("attn_resolutions", [])
    if attn_resolutions is None:
        attn_resolutions = None
    else:
        attn_resolutions = tuple(attn_resolutions)

    mel_bins = ddconfig.get("mel_bins", None) or mel_cfg.get("n_mel_channels", None)
    sample_rate = model_cfg.get("sampling_rate", None) or audio_cfg_meta.get("sampling_rate", None)
    mel_hop_length = stft_cfg.get("hop_length", None)
    n_fft = stft_cfg.get("filter_length", None)
    is_causal = stft_cfg.get("causal", None)

    return {
        "base_channels": ddconfig.get("ch", 128),
        "output_channels": ddconfig.get("out_ch", 2),
        "ch_mult": tuple(ddconfig.get("ch_mult", (1, 2, 4))),
        "num_res_blocks": ddconfig.get("num_res_blocks", 2),
        "attn_resolutions": attn_resolutions,
        "in_channels": ddconfig.get("in_channels", 2),
        "resolution": ddconfig.get("resolution", 256),
        "latent_channels": ddconfig.get("z_channels", 8),
        "double_z": ddconfig.get("double_z", True),
        "norm_type": ddconfig.get("norm_type", "pixel"),
        "causality_axis": ddconfig.get("causality_axis", "height"),
        "dropout": ddconfig.get("dropout", 0.0),
        "mid_block_add_attention": ddconfig.get("mid_block_add_attention", False),
        "sample_rate": sample_rate or 16000,
        "mel_hop_length": mel_hop_length or 160,
        "n_fft": n_fft or 1024,
        "is_causal": True if is_causal is None else bool(is_causal),
        "mel_bins": mel_bins or 64,
    }


def _apply_remap_rules(state_dict: Dict[str, Any], rename_dict: Dict[str, str], special_keys_remap: Dict[str, Any]) -> None:
    for key in list(state_dict.keys()):
        new_key = key
        for replace_key, rename_key in rename_dict.items():
            new_key = new_key.replace(replace_key, rename_key)
        update_state_dict_inplace(state_dict, key, new_key)

    for key in list(state_dict.keys()):
        for special_key, handler in special_keys_remap.items():
            if special_key not in key:
                continue
            if handler is None:
                _remove_key_inplace(key, state_dict)
            elif handler == "audio_vae_stats":
                _convert_ltx2_audio_vae_per_channel_statistics(key, state_dict)
            else:
                handler(key, state_dict)


def _get_ltx2_transformer_config(version: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if version == "test":
        diffusers_config = {
            "in_channels": 4,
            "out_channels": 4,
            "patch_size": 1,
            "patch_size_t": 1,
            "num_attention_heads": 2,
            "attention_head_dim": 8,
            "cross_attention_dim": 16,
            "vae_scale_factors": (8, 32, 32),
            "pos_embed_max_pos": 20,
            "base_height": 2048,
            "base_width": 2048,
            "audio_in_channels": 4,
            "audio_out_channels": 4,
            "audio_patch_size": 1,
            "audio_patch_size_t": 1,
            "audio_num_attention_heads": 2,
            "audio_attention_head_dim": 4,
            "audio_cross_attention_dim": 8,
            "audio_scale_factor": 4,
            "audio_pos_embed_max_pos": 20,
            "audio_sampling_rate": 16000,
            "audio_hop_length": 160,
            "num_layers": 2,
            "activation_fn": "gelu-approximate",
            "qk_norm": "rms_norm_across_heads",
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
            "caption_channels": 16,
            "attention_bias": True,
            "attention_out_bias": True,
            "rope_theta": 10000.0,
            "rope_double_precision": False,
            "causal_offset": 1,
            "timestep_scale_multiplier": 1000,
            "cross_attn_timestep_scale_multiplier": 1,
        }
    elif version == "2.0":
        diffusers_config = {
            "in_channels": 128,
            "out_channels": 128,
            "patch_size": 1,
            "patch_size_t": 1,
            "num_attention_heads": 32,
            "attention_head_dim": 128,
            "cross_attention_dim": 4096,
            "vae_scale_factors": (8, 32, 32),
            "pos_embed_max_pos": 20,
            "base_height": 2048,
            "base_width": 2048,
            "audio_in_channels": 128,
            "audio_out_channels": 128,
            "audio_patch_size": 1,
            "audio_patch_size_t": 1,
            "audio_num_attention_heads": 32,
            "audio_attention_head_dim": 64,
            "audio_cross_attention_dim": 2048,
            "audio_scale_factor": 4,
            "audio_pos_embed_max_pos": 20,
            "audio_sampling_rate": 16000,
            "audio_hop_length": 160,
            "num_layers": 48,
            "activation_fn": "gelu-approximate",
            "qk_norm": "rms_norm_across_heads",
            "norm_elementwise_affine": False,
            "norm_eps": 1e-6,
            "caption_channels": 3840,
            "attention_bias": True,
            "attention_out_bias": True,
            "rope_theta": 10000.0,
            "rope_double_precision": True,
            "causal_offset": 1,
            "timestep_scale_multiplier": 1000,
            "cross_attn_timestep_scale_multiplier": 1000,
            "rope_type": "split",
        }
    else:
        raise ValueError(f"Unsupported LTX-2 transformer version: {version}")

    if overrides:
        diffusers_config = {**diffusers_config, **overrides}
    return diffusers_config


def _get_ltx2_connectors_config(version: str) -> Dict[str, Any]:
    if version == "test":
        return {
            "caption_channels": 16,
            "text_proj_in_factor": 3,
            "video_connector_num_attention_heads": 4,
            "video_connector_attention_head_dim": 8,
            "video_connector_num_layers": 1,
            "video_connector_num_learnable_registers": None,
            "audio_connector_num_attention_heads": 4,
            "audio_connector_attention_head_dim": 8,
            "audio_connector_num_layers": 1,
            "audio_connector_num_learnable_registers": None,
            "connector_rope_base_seq_len": 32,
            "rope_theta": 10000.0,
            "rope_double_precision": False,
            "causal_temporal_positioning": False,
        }
    if version == "2.0":
        return {
            "caption_channels": 3840,
            "text_proj_in_factor": 49,
            "video_connector_num_attention_heads": 30,
            "video_connector_attention_head_dim": 128,
            "video_connector_num_layers": 2,
            "video_connector_num_learnable_registers": 128,
            "audio_connector_num_attention_heads": 30,
            "audio_connector_attention_head_dim": 128,
            "audio_connector_num_layers": 2,
            "audio_connector_num_learnable_registers": 128,
            "connector_rope_base_seq_len": 4096,
            "rope_theta": 10000.0,
            "rope_double_precision": True,
            "causal_temporal_positioning": False,
            "rope_type": "split",
        }
    raise ValueError(f"Unsupported LTX-2 connectors version: {version}")


def _get_ltx2_video_vae_config(version: str) -> Dict[str, Any]:
    if version in {"test", "2.0"}:
        return {
            "in_channels": 3,
            "out_channels": 3,
            "latent_channels": 128,
            "block_out_channels": (256, 512, 1024, 2048),
            "down_block_types": (
                "LTX2VideoDownBlock3D",
                "LTX2VideoDownBlock3D",
                "LTX2VideoDownBlock3D",
                "LTX2VideoDownBlock3D",
            ),
            "decoder_block_out_channels": (256, 512, 1024),
            "layers_per_block": (4, 6, 6, 2, 2),
            "decoder_layers_per_block": (5, 5, 5, 5),
            "spatio_temporal_scaling": (True, True, True, True),
            "decoder_spatio_temporal_scaling": (True, True, True),
            "decoder_inject_noise": (False, False, False, False),
            "downsample_type": ("spatial", "temporal", "spatiotemporal", "spatiotemporal"),
            "upsample_residual": (True, True, True),
            "upsample_factor": (2, 2, 2),
            "timestep_conditioning": False,
            "patch_size": 4,
            "patch_size_t": 1,
            "resnet_norm_eps": 1e-6,
            "encoder_causal": True,
            "decoder_causal": False,
            "encoder_spatial_padding_mode": "zeros",
            "decoder_spatial_padding_mode": "reflect",
            "spatial_compression_ratio": 32,
            "temporal_compression_ratio": 8,
        }
    raise ValueError(f"Unsupported LTX-2 video VAE version: {version}")


def _get_ltx2_audio_vae_config(version: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if overrides:
        return overrides
    if version == "2.0":
        return {
            "base_channels": 128,
            "output_channels": 2,
            "ch_mult": (1, 2, 4),
            "num_res_blocks": 2,
            "attn_resolutions": None,
            "in_channels": 2,
            "resolution": 256,
            "latent_channels": 8,
            "norm_type": "pixel",
            "causality_axis": "height",
            "dropout": 0.0,
            "mid_block_add_attention": False,
            "sample_rate": 16000,
            "mel_hop_length": 160,
            "is_causal": True,
            "mel_bins": 64,
        }
    raise ValueError(f"Unsupported LTX-2 audio VAE version: {version}")


def _get_ltx2_vocoder_config(version: str) -> Dict[str, Any]:
    if version == "2.0":
        return {
            "in_channels": 128,
            "hidden_channels": 1024,
            "out_channels": 2,
            "upsample_kernel_sizes": [16, 15, 8, 4, 4],
            "upsample_factors": [6, 5, 2, 2, 2],
            "resnet_kernel_sizes": [3, 7, 11],
            "resnet_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "leaky_relu_negative_slope": 0.1,
            "output_sampling_rate": 24000,
        }
    raise ValueError(f"Unsupported LTX-2 vocoder version: {version}")


def convert_ltx2_transformer(
    original_state_dict: Dict[str, Any],
    version: str,
    config_overrides: Dict[str, Any] | None = None,
) -> LTX2VideoTransformer3DModel:
    diffusers_config = _get_ltx2_transformer_config(version, config_overrides)
    transformer_state_dict, _ = _split_transformer_and_connector_state_dict(original_state_dict)
    special_keys_remap = {
        "video_embeddings_connector": _remove_key_inplace,
        "audio_embeddings_connector": _remove_key_inplace,
        "adaln_single": _convert_ltx2_transformer_adaln_single,
    }

    with init_empty_weights():
        transformer = LTX2VideoTransformer3DModel.from_config(diffusers_config)

    _apply_remap_rules(transformer_state_dict, LTX_2_0_TRANSFORMER_KEYS_RENAME_DICT, special_keys_remap)
    transformer.load_state_dict(transformer_state_dict, strict=True, assign=True)
    return transformer


def convert_ltx2_connectors(original_state_dict: Dict[str, Any], version: str) -> LTX2TextConnectors:
    diffusers_config = _get_ltx2_connectors_config(version)
    _, connector_state_dict = _split_transformer_and_connector_state_dict(original_state_dict)
    if not connector_state_dict:
        raise ValueError("No connector weights found in the provided state dict.")

    with init_empty_weights():
        connectors = LTX2TextConnectors.from_config(diffusers_config)

    _apply_remap_rules(connector_state_dict, LTX_2_0_CONNECTORS_KEYS_RENAME_DICT, {})
    connectors.load_state_dict(connector_state_dict, strict=True, assign=True)
    return connectors


def convert_ltx2_video_vae(original_state_dict: Dict[str, Any], version: str) -> AutoencoderKLLTX2Video:
    diffusers_config = _get_ltx2_video_vae_config(version)
    with init_empty_weights():
        vae = AutoencoderKLLTX2Video.from_config(diffusers_config)
    _apply_remap_rules(original_state_dict, LTX_2_0_VIDEO_VAE_RENAME_DICT, LTX_2_0_VAE_SPECIAL_KEYS_REMAP)
    vae.load_state_dict(original_state_dict, strict=True, assign=True)
    return vae


def convert_ltx2_audio_vae(
    original_state_dict: Dict[str, Any],
    version: str,
    metadata_config: Dict[str, Any] | None = None,
) -> AutoencoderKLLTX2Audio:
    config_overrides = _extract_audio_vae_config_from_metadata(metadata_config or {})
    diffusers_config = _get_ltx2_audio_vae_config(version, config_overrides)
    with init_empty_weights():
        vae = AutoencoderKLLTX2Audio.from_config(diffusers_config)
    _apply_remap_rules(original_state_dict, LTX_2_0_AUDIO_VAE_RENAME_DICT, LTX_2_0_AUDIO_VAE_SPECIAL_KEYS_REMAP)
    vae.load_state_dict(original_state_dict, strict=True, assign=True)
    return vae


def convert_ltx2_vocoder(original_state_dict: Dict[str, Any], version: str) -> LTX2Vocoder:
    diffusers_config = _get_ltx2_vocoder_config(version)
    with init_empty_weights():
        vocoder = LTX2Vocoder.from_config(diffusers_config)
    _apply_remap_rules(original_state_dict, LTX_2_0_VOCODER_RENAME_DICT, LTX_2_0_VOCODER_SPECIAL_KEYS_REMAP)
    vocoder.load_state_dict(original_state_dict, strict=True, assign=True)
    return vocoder
