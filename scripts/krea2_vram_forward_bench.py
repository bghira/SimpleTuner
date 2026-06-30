#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import time
from types import SimpleNamespace

import torch

from simpletuner.helpers.models.krea2.transformer import Krea2Transformer2DModel
from simpletuner.helpers.training.attention_backend import AttentionBackendController, AttentionPhase
from simpletuner.helpers.training.quantisation import _torchao_model


def _pack_latents(latents: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, int, int]:
    batch_size, channels, height, width = latents.shape
    packed = latents.view(batch_size, channels, height // patch_size, patch_size, width // patch_size, patch_size)
    packed = packed.permute(0, 2, 4, 1, 3, 5)
    packed = packed.reshape(batch_size, (height // patch_size) * (width // patch_size), channels * patch_size * patch_size)
    return packed, height // patch_size, width // patch_size


def _position_ids(text_seq_len: int, grid_height: int, grid_width: int, device: torch.device) -> torch.Tensor:
    text_ids = torch.zeros(text_seq_len, 3, device=device)
    image_ids = torch.zeros(grid_height, grid_width, 3, device=device)
    image_ids[..., 1] = torch.arange(grid_height, device=device)[:, None]
    image_ids[..., 2] = torch.arange(grid_width, device=device)[None, :]
    return torch.cat([text_ids, image_ids.reshape(grid_height * grid_width, 3)], dim=0)


def _gb(value: int) -> float:
    return value / (1024**3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="krea/Krea-2-Raw")
    parser.add_argument("--subfolder", default="transformer")
    parser.add_argument("--precision", choices=("bf16", "int8-torchao"), required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--text-seq-len", type=int, default=512)
    parser.add_argument("--attention-mechanism", default="diffusers")
    parser.add_argument("--fuse-qkv", action="store_true")
    parser.add_argument("--train-step", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    AttentionBackendController.apply(SimpleNamespace(attention_mechanism=args.attention_mechanism), AttentionPhase.TRAIN)

    result = {
        "precision": args.precision,
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "text_seq_len": args.text_seq_len,
        "attention_mechanism": args.attention_mechanism,
        "fuse_qkv": args.fuse_qkv,
    }

    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        started = time.perf_counter()
        model = Krea2Transformer2DModel.from_pretrained(
            args.model,
            subfolder=args.subfolder,
            torch_dtype=torch.bfloat16,
        )
        model.eval()
        if args.fuse_qkv:
            model.fuse_qkv_projections(preferred_backend=args.attention_mechanism)
        if args.precision == "int8-torchao":
            _torchao_model(model, model_precision="int8-torchao", base_model_precision="int8-torchao")
        optimizer = None
        if args.train_step:
            from peft import LoraConfig, get_peft_model

            target_modules = ["to_qkv", "to_out.0"] if args.fuse_qkv else ["to_q", "to_k", "to_v", "to_out.0"]
            model = get_peft_model(
                model,
                LoraConfig(
                    r=args.lora_rank,
                    lora_alpha=args.lora_rank,
                    init_lora_weights=True,
                    target_modules=target_modules,
                ),
            )
            model.train()
        model.to(device)
        if args.train_step:
            trainable_params = [param for param in model.parameters() if param.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
        torch.cuda.synchronize(device)
        result["load_seconds"] = time.perf_counter() - started
        result["post_load_allocated_gb"] = _gb(torch.cuda.memory_allocated(device))
        result["post_load_reserved_gb"] = _gb(torch.cuda.memory_reserved(device))

        patch_size = int(max(getattr(model.config, "patch_size", 2), 1))
        latent_channels = int(getattr(model.config, "in_channels", 64)) // (patch_size * patch_size)
        latent_size = args.resolution // 8
        latents = torch.randn(
            args.batch_size,
            latent_channels,
            latent_size,
            latent_size,
            device=device,
            dtype=torch.bfloat16,
        )
        hidden_states, grid_height, grid_width = _pack_latents(latents, patch_size)
        prompt_embeds = torch.randn(
            args.batch_size,
            args.text_seq_len,
            int(getattr(model.config, "num_text_layers", 12)),
            int(getattr(model.config, "text_hidden_dim", 2560)),
            device=device,
            dtype=torch.bfloat16,
        )
        prompt_mask = torch.ones(args.batch_size, args.text_seq_len, device=device, dtype=torch.int64)
        timesteps = torch.full((args.batch_size,), 0.5, device=device, dtype=torch.float32)
        pos_ids = _position_ids(args.text_seq_len, grid_height, grid_width, device)

        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        forward_started = time.perf_counter()
        with torch.set_grad_enabled(args.train_step):
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                position_ids=pos_ids,
                encoder_attention_mask=prompt_mask,
                return_dict=False,
            )[0]
            if args.train_step:
                loss = output.float().square().mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - forward_started
        if args.train_step:
            result["step_seconds"] = elapsed
        else:
            result["forward_seconds"] = elapsed
        result["output_shape"] = list(output.shape)
        peak_prefix = "peak_step" if args.train_step else "peak_forward"
        post_prefix = "post_step" if args.train_step else "post_forward"
        result[f"{peak_prefix}_allocated_gb"] = _gb(torch.cuda.max_memory_allocated(device))
        result[f"{peak_prefix}_reserved_gb"] = _gb(torch.cuda.max_memory_reserved(device))
        result[f"{post_prefix}_allocated_gb"] = _gb(torch.cuda.memory_allocated(device))
        result[f"{post_prefix}_reserved_gb"] = _gb(torch.cuda.memory_reserved(device))
        if args.train_step:
            result["trainable_params"] = sum(param.numel() for param in model.parameters() if param.requires_grad)
        result["status"] = "ok"
    except torch.cuda.OutOfMemoryError as exc:
        result["status"] = "oom"
        result["error"] = str(exc).splitlines()[0]
        result["allocated_gb"] = _gb(torch.cuda.memory_allocated(device))
        result["reserved_gb"] = _gb(torch.cuda.memory_reserved(device))
        torch.cuda.empty_cache()
    finally:
        AttentionBackendController.restore_default()
        gc.collect()
        torch.cuda.empty_cache()

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
