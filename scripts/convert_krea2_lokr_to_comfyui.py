"""Export fused full-matrix Krea2 LoKR projections as separate ComfyUI projections."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def split_fused_lokr(weights: dict[str, torch.Tensor], config: dict) -> dict[str, torch.Tensor]:
    converted = dict(weights)
    prefixes = {key.rsplit(".", 1)[0] for key in weights if key.startswith("lycoris_") and "_attn_to_qkv." in key}
    if not prefixes:
        raise ValueError("No fused Krea2 LyCORIS attention projections were found.")
    for prefix in sorted(prefixes):
        parts = {key.removeprefix(prefix + "."): value for key, value in weights.items() if key.startswith(prefix + ".")}
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
        dora = parts.get("dora_scale")
        if dora is not None and dora.shape != (sum(sizes), 1):
            raise ValueError(f"{prefix} requires output-normalized DoRA scales.")
        # A contiguous Q/K/V slice can cross Kronecker factor boundaries; materialize it exactly.
        delta = torch.kron(w1.float().contiguous(), w2.float().contiguous())
        offset = 0
        for projection, size in zip(("q", "k", "v"), sizes):
            target = prefix.removesuffix("qkv") + projection
            for suffix in ("lokr_w1", "lokr_w2", "alpha", "dora_scale"):
                if f"{target}.{suffix}" in converted:
                    raise ValueError(f"{target} already contains separate projection weights.")
            converted[f"{target}.lokr_w1"] = torch.ones((1, 1), dtype=torch.float32)
            converted[f"{target}.lokr_w2"] = delta[offset : offset + size].clone()
            if dora is not None:
                converted[f"{target}.dora_scale"] = dora[offset : offset + size].clone()
            offset += size
        for suffix in parts:
            del converted[f"{prefix}.{suffix}"]
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Native full-matrix LyCORIS safetensors checkpoint")
    parser.add_argument("--output", required=True, help="Separate ComfyUI export; use a different filename")
    parser.add_argument("--transformer-config", required=True, help="Base Krea2 transformer's config.json")
    args = parser.parse_args()
    if Path(args.input).resolve() == Path(args.output).resolve():
        parser.error("The output must differ from the native checkpoint.")
    with open(args.transformer_config, encoding="utf-8") as handle:
        config = json.load(handle)
    with safe_open(args.input, framework="pt", device="cpu") as checkpoint:
        weights = {key: checkpoint.get_tensor(key) for key in checkpoint.keys()}
        metadata = checkpoint.metadata()
    save_file(split_fused_lokr(weights, config), args.output, metadata=metadata)


if __name__ == "__main__":
    main()
