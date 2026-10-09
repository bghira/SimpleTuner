"""Export fused full-matrix Krea2 LoKR projections as separate ComfyUI projections."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def _remove_unused_qkv_projections(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converted = dict(weights)
    prefixes = {key.rsplit(".", 1)[0] for key in weights if key.startswith("lycoris_") and "_attn_to_qkv." in key}
    for prefix in prefixes:
        for projection in "qkv":
            target = prefix.removesuffix("qkv") + projection
            parts = {key.removeprefix(target + "."): value for key, value in weights.items() if key.startswith(target + ".")}
            if not parts:
                continue
            if not {"lokr_w1", "lokr_w2"} <= parts.keys() or parts.keys() - {"lokr_w1", "lokr_w2", "alpha", "dora_scale"}:
                raise ValueError(f"{target} must be a full-matrix LoKR projection.")
            w1, w2 = parts["lokr_w1"], parts["lokr_w2"]
            if w1.ndim != 2 or w2.ndim != 2:
                raise ValueError(f"{target} must contain linear LoKR factors.")
            if not (
                torch.isfinite(w1).all()
                and torch.isfinite(w2).all()
                and (torch.count_nonzero(w1) == 0 or torch.count_nonzero(w2) == 0)
            ):
                raise ValueError(f"{target} already contains nonzero separate projection weights.")
            # Fused attention never calls these zero-delta modules, including their DoRA scales.
            for suffix in parts:
                del converted[f"{target}.{suffix}"]
    return converted


def split_fused_lokr(
    weights: dict[str, torch.Tensor], config: dict, base_weights: dict[str, torch.Tensor] | None = None
) -> dict[str, torch.Tensor]:
    weights = _remove_unused_qkv_projections(weights)
    converted = dict(weights)
    prefixes = {key.rsplit(".", 1)[0] for key in weights if key.startswith("lycoris_") and "_attn_to_qkv." in key}
    if not prefixes:
        raise ValueError("No fused Krea2 LyCORIS attention projections were found.")
    for key in weights:
        if not key.endswith(".dora_scale"):
            continue
        prefix = key.removesuffix(".dora_scale")
        if base_weights is None:
            raise ValueError("DoRA export requires --base-model with the original Krea2 raw.safetensors checkpoint.")
        parts = {key.removeprefix(prefix + "."): value for key, value in weights.items() if key.startswith(prefix + ".")}
        if not {"lokr_w1", "lokr_w2"} <= parts.keys() or parts.keys() - {"lokr_w1", "lokr_w2", "alpha", "dora_scale"}:
            raise ValueError(f"{prefix} must be a full-matrix LoKR projection.")
        base = base_weights[prefix].float()
        dora = parts["dora_scale"]
        if base.ndim != 2 or dora.shape != (base.shape[0], 1):
            raise ValueError(f"{prefix} requires a linear base weight and output-normalized DoRA scales.")
        delta = torch.kron(parts["lokr_w1"].float().contiguous(), parts["lokr_w2"].float().contiguous())
        if delta.shape != base.shape:
            raise ValueError(f"{prefix} factors do not match the base model.")
        merged = (base + delta).to(dora.dtype)
        merged = merged * (dora / (merged.norm(dim=1, keepdim=True) + torch.finfo(dora.dtype).eps))
        # ComfyUI output DoRA uses the base norm; export the native merged difference instead.
        converted[f"{prefix}.lokr_w1"] = torch.ones((1, 1), dtype=torch.float32)
        converted[f"{prefix}.lokr_w2"] = merged.float() - base
        del converted[key]
        converted.pop(f"{prefix}.alpha", None)
    for prefix in sorted(prefixes):
        parts = {key.removeprefix(prefix + "."): value for key, value in converted.items() if key.startswith(prefix + ".")}
        if not {"lokr_w1", "lokr_w2"} <= parts.keys() or parts.keys() - {"lokr_w1", "lokr_w2", "alpha", "dora_scale"}:
            raise ValueError(f"{prefix} must be a full-matrix LoKR projection.")
        w1, w2 = parts["lokr_w1"], parts["lokr_w2"]
        if w1.ndim != 2 or w2.ndim != 2:
            raise ValueError(f"{prefix} must contain linear LoKR factors.")
        if prefix.startswith(("lycoris_text_fusion_layerwise_blocks_", "lycoris_text_fusion_refiner_blocks_")):
            head_dim = config["text_hidden_dim"] // config["text_num_attention_heads"]
            query_size = config["text_hidden_dim"]
            kv_size = head_dim * config["text_num_key_value_heads"]
        elif prefix.startswith("lycoris_transformer_blocks_"):
            query_size = config["attention_head_dim"] * config["num_attention_heads"]
            kv_size = config["attention_head_dim"] * config["num_key_value_heads"]
        else:
            raise ValueError(f"Unknown Krea2 fused attention projection: {prefix}")
        sizes = (query_size, kv_size, kv_size)
        if w1.shape[0] * w2.shape[0] != sum(sizes) or w1.shape[1] * w2.shape[1] != query_size:
            raise ValueError(f"{prefix} factors do not match the transformer config.")
        # A contiguous Q/K/V slice can cross Kronecker factor boundaries; materialize it exactly.
        delta = torch.kron(w1.float().contiguous(), w2.float().contiguous())
        offset = 0
        for projection, size in zip(("q", "k", "v"), sizes):
            target = prefix.removesuffix("qkv") + projection
            converted[f"{target}.lokr_w1"] = torch.ones((1, 1), dtype=torch.float32)
            converted[f"{target}.lokr_w2"] = delta[offset : offset + size].clone()
            offset += size
        for suffix in parts:
            del converted[f"{prefix}.{suffix}"]
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Native full-matrix LyCORIS safetensors checkpoint")
    parser.add_argument("--output", required=True, help="Separate ComfyUI export; use a different filename")
    parser.add_argument("--transformer-config", required=True, help="Base Krea2 transformer's config.json")
    parser.add_argument("--base-model", help="Original Krea2 raw.safetensors checkpoint; required for DoRA")
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        parser.error("The output must differ from the native checkpoint.")
    if args.base_model and Path(args.base_model).resolve() == Path(args.output).resolve():
        parser.error("The output must differ from the base model.")
    with open(args.transformer_config, encoding="utf-8") as handle:
        config = json.load(handle)
    with safe_open(args.input, framework="pt", device="cpu") as checkpoint:
        weights = {key: checkpoint.get_tensor(key) for key in checkpoint.keys()}
        metadata = checkpoint.metadata()
    weights = _remove_unused_qkv_projections(weights)
    base_weights = None
    if any(key.endswith(".dora_scale") for key in weights):
        if not args.base_model:
            parser.error("DoRA export requires --base-model with the original Krea2 raw.safetensors checkpoint.")
        from simpletuner.helpers.models.krea2.quantized_loading import _map_comfy_key_to_diffusers

        base_weights = {}
        with safe_open(args.base_model, framework="pt", device="cpu") as checkpoint:
            aliases = {
                "lycoris_"
                + _map_comfy_key_to_diffusers("model.diffusion_model." + key).removesuffix(".weight").replace(".", "_"): key
                for key in checkpoint.keys()
            }
            for key in weights:
                if key.endswith(".dora_scale"):
                    prefix = key.removesuffix(".dora_scale")
                    if prefix.endswith("_attn_to_qkv"):
                        base_weights[prefix] = torch.cat(
                            [
                                checkpoint.get_tensor(aliases[prefix.removesuffix("qkv") + projection]).float()
                                for projection in "qkv"
                            ]
                        )
                    else:
                        base_weights[prefix] = checkpoint.get_tensor(aliases[prefix])
    save_file(split_fused_lokr(weights, config, base_weights), args.output, metadata=metadata)


if __name__ == "__main__":
    main()
