"""Verify real H3 Image weights, per-head gradients and ComfyUI adapter conversion.

Run from the repository root with a configured accelerator:
    .venv/bin/python scripts/verify_minimax_h3_image.py --resolutions 2048 4096
"""

import argparse
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model_state_dict
from safetensors.torch import save_file

from simpletuner.helpers.models.minimaxh3.image import MINIMAX_H3_IMAGE_REPO, MINIMAX_H3_IMAGE_SIGMAS
from simpletuner.helpers.models.minimaxh3.image_blocks import build_image_layout
from simpletuner.helpers.models.minimaxh3.modular_pipeline import (
    _convert_minimax_h3_comfy_lora_to_diffusers,
    _convert_minimax_h3_diffusers_lora_to_comfyui,
)
from simpletuner.helpers.models.minimaxh3.transformer import MiniMaxH3Transformer3DModel


def verify_comfyui_adapter(adapter):
    comfy = _convert_minimax_h3_diffusers_lora_to_comfyui(adapter)
    restored, alphas = _convert_minimax_h3_comfy_lora_to_diffusers(comfy, target_prefix="transformer")
    for key, down in adapter.items():
        if not key.endswith(".lora_A.weight"):
            continue
        module = key.removesuffix(".lora_A.weight")
        up = adapter[f"{module}.lora_B.weight"]
        prefix = f"transformer.{module}"
        restored_down = restored[f"{prefix}.lora.down.weight"]
        restored_up = restored[f"{prefix}.lora.up.weight"]
        scale = alphas[f"{prefix}.alpha"] / restored_down.shape[0]
        probe = torch.linspace(-1, 1, down.shape[1], device=down.device).expand(2, -1)
        expected = probe @ down.float().T @ up.float().T
        actual = probe @ restored_down.float().T @ restored_up.float().T * scale
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    return comfy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MINIMAX_H3_IMAGE_REPO)
    parser.add_argument("--revision")
    parser.add_argument("--resolutions", nargs="+", type=int, default=[2048, 4096])
    parser.add_argument("--output", type=Path, default=Path("output/h3-image-verification"))
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    parser.add_argument("--single-file", action="store_true")
    parser.add_argument("--activation-offload", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    if args.single_file:
        model = MiniMaxH3Transformer3DModel.from_single_file(args.model, torch_dtype=torch.bfloat16, revision=args.revision)
    else:
        model = MiniMaxH3Transformer3DModel.from_pretrained(
            args.model,
            subfolder="transformer_diffusers",
            revision=args.revision,
            torch_dtype=torch.bfloat16,
            audio_enabled=False,
        )
    model.requires_grad_(False)
    model.add_adapter(LoraConfig(r=32, lora_alpha=32, target_modules=["to_q", "to_k", "to_v", "to_out.0"]))
    model.enable_gradient_checkpointing()
    if args.activation_offload:
        model.set_gradient_checkpointing_backend("unsloth")
    model.to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    records = []
    for resolution in args.resolutions:
        if resolution % 32 or resolution < 32:
            raise ValueError("Resolutions must be positive multiples of 32.")
        latent_edge = resolution // 16
        layout = build_image_layout(torch.ones(32, dtype=torch.long), [], latent_edge, latent_edge, model.config.patch_size)
        inputs = {
            "hidden_states": torch.randn(1, (resolution // 32) ** 2, 96, device=device),
            "audio_hidden_states": torch.empty(1, 0, 32, device=device),
            "encoder_hidden_states": torch.randn(1, 32, model.config.text_dim, device=device, dtype=torch.bfloat16),
            "timestep": torch.tensor([1 - MINIMAX_H3_IMAGE_SIGMAS[0]], device=device),
            "timestep_indices": torch.zeros(layout.sequence_length, device=device, dtype=torch.long),
            "token_tags": layout.token_tags.to(device),
            "position_ids": layout.position_ids.to(device),
            "video_indices": layout.video_indices.to(device),
            "audio_indices": layout.audio_indices.to(device),
            "text_indices": layout.text_indices.to(device),
            "pdd_step_index": 0,
        }
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.monotonic()
        prediction = model(**inputs).sample
        loss = prediction.float().square().mean()
        loss.backward()
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError("H3 Image adapter gradients are missing or non-finite.")
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(gradient_norm) or gradient_norm <= 0:
            raise RuntimeError("H3 Image adapter gradient norm must be finite and nonzero.")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        record = {
            "resolution": resolution,
            "loss": loss.item(),
            "gradient_norm": gradient_norm.item(),
            "seconds": time.monotonic() - start,
            "peak_vram_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        del prediction, loss, inputs
    adapter = get_peft_model_state_dict(model)
    comfy = verify_comfyui_adapter(adapter)
    save_file(
        {key: value.detach().cpu().contiguous() for key, value in comfy.items()},
        str(args.output / "adapter-comfyui.safetensors"),
    )
    (args.output / "results.json").write_text(
        json.dumps({"checks": records, "comfyui_tensors": len(comfy)}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
