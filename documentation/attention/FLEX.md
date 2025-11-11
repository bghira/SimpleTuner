# FlexAttention Guide

**FlexAttention requires CUDA devices.**

FlexAttention is PyTorch’s block-level attention kernel that landed in PyTorch 2.5.0. It rewrites the SDPA computation as a programmable loop so you can express masking strategies without writing CUDA. Diffusers exposes it through the new `attention_backend` dispatcher, and SimpleTuner wires that dispatcher to `--attention_mechanism=flex`.

> ⚠️ FlexAttention is still labelled “prototype” upstream. Expect to recompile when you change drivers, CUDA versions, or PyTorch builds.

## Prerequisites

1. **Ampere+ GPU** – NVIDIA SM80 (A100), Ada (4090/L40S), or Hopper (H100/H200) are supported. Older cards fail the capability check during kernel registration.
2. **Compiler toolchain** – the kernels compile at runtime with `nvcc`. Install `cuda-nvcc` that matches the wheel (CUDA 12.8 for current releases) and make sure `nvcc` appears in `$PATH`.

## Building the kernels

The first import of `torch.nn.attention.flex_attention` builds the CUDA extension into PyTorch’s lazy cache. You can do this ahead of time to surface build errors early:

```bash
python - <<'PY'
import torch
from torch.nn.attention import flex_attention

assert torch.__version__ >= "2.5.0", torch.__version__
flex_attention.build_flex_attention_kernels()  # no-op when already compiled
print("FlexAttention kernels installed at", flex_attention.kernel_root)
PY
```

- If you see `AttributeError: flex_attention has no attribute build_flex_attention_kernels`, upgrade PyTorch – the helper shipped in 2.5.0+.
- Cache lives under `~/.cache/torch/kernels`. Remove it if you upgrade CUDA and need to force a rebuild.

## Enabling FlexAttention in SimpleTuner

Once the kernels exist, select the backend via `config.json`:

```json
{
  "attention_mechanism": "flex"
}
```

What to expect:

- Only dispatch-enabled transformer blocks (Flux, Wan 2.2, LTXVideo, QwenImage, etc.) route through Diffusers’ `attention_backend`. Classic SD/SDXL UNets still call PyTorch SDPA directly, so FlexAttention has no effect there.
- FlexAttention currently supports BF16/FP16 tensors. If you run FP32 or FP8 weights you’ll hit `ValueError: Query, key, and value must be either bfloat16 or float16`.
- The backend honours `is_causal=False` only. Supplying a mask converts it into the block mask the kernel expects, but arbitrary ragged masks are not yet supported (mirrors upstream behaviour).

## Troubleshooting checklist

| Symptom | Fix |
| --- | --- |
| `RuntimeError: Flex Attention backend 'flex' is not usable because of missing package` | PyTorch build is < 2.5 or does not include CUDA. Install a newer CUDA wheel. |
| `Could not compile flex_attention kernels` | Ensure `nvcc` matches the CUDA version your torch wheel expects (12.1+). Set `export CUDA_HOME=/usr/local/cuda-12.4` if the installer can’t locate headers. |
| `ValueError: Query, key, and value must be on a CUDA device` | FlexAttention is CUDA only. Remove the backend setting on Apple/ROCm runs. |
| Training never switches to the backend | Make sure you are using a model family that already uses Diffusers’ `dispatch_attention_fn` (Flux/Wan/LTXVideo). Standard SD UNets will continue to use PyTorch SDPA no matter what backend you select. |

Refer to the upstream documentation for deeper internals and API flags: [PyTorch FlexAttention docs](https://pytorch.org/docs/stable/nn.attention.html#flexattention).
