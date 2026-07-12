# Metal Flash Attention

`metal-flash-attention` routes eligible Apple Silicon MPS SDPA calls through the Universal Metal Flash Attention (UMFA) PyTorch FFI extension. It is experimental and currently intended for FLUX-style FP32 training paths where PyTorch SDPA either uses more memory or hits MPSGraph limits at long sequence lengths.

## Requirements

- Apple Silicon with MPS available.
- Xcode command line tools with the Metal toolchain.
- SimpleTuner installed with the Apple dependency set. The Apple extra requires PyTorch `>=2.13.0`.
- A UMFA build that exposes `metal_flash_attention_autograd`. Older forward-only builds are rejected by SimpleTuner.

SimpleTuner only dispatches UMFA for MPS FP32 4D attention calls with at least four heads and sequence length at least 64. FP16/BF16, masked, causal, grouped-query, tiny, 2D, and non-MPS calls fall back to PyTorch SDPA.

## Build And Install UMFA

Use the same Python environment that runs SimpleTuner:

```bash
export ST_ROOT=/path/to/SimpleTuner
export UMFA_ROOT=/path/to/universal-metal-flash-attention
export PYTHON="$ST_ROOT/.venv/bin/python"
```

Build the Swift package and install the PyTorch FFI package:

```bash
cd "$UMFA_ROOT"
git submodule update --init --recursive
swift build -c release

cd "$UMFA_ROOT/examples/pytorch-custom-op-ffi"
"$PYTHON" -m pip install --upgrade pip setuptools wheel pybind11 numpy
"$PYTHON" -m pip install --force-reinstall --no-deps --no-build-isolation --no-cache-dir .
```

Verify the extension exports the autograd binding:

```bash
"$PYTHON" - <<'PY'
import metal_sdpa_extension

print("metal_flash_attention_autograd" in dir(metal_sdpa_extension))
print([name for name in dir(metal_sdpa_extension) if "attention" in name])
PY
```

Verify SimpleTuner accepts the installed extension:

```bash
cd "$ST_ROOT"
"$PYTHON" - <<'PY'
from simpletuner.helpers.training.attention_backend import (
    get_metal_flash_attention_unavailable_reason,
    is_metal_flash_attention_available,
)

print("available", is_metal_flash_attention_available())
print("reason", get_metal_flash_attention_unavailable_reason())
PY
```

Expected result:

```text
available True
reason None
```

If the output says the UMFA output is detached, rebuild UMFA from a version that implements `metal_flash_attention_autograd`.

## Enable In SimpleTuner

Set the attention mechanism:

```json
{
  "attention_mechanism": "metal-flash-attention",
  "mixed_precision": "no",
  "base_model_default_dtype": "fp32"
}
```

`mixed_precision=no` and FP32 model defaults are important for the current integration. SimpleTuner falls back rather than sending BF16/FP16 attention to UMFA.

Quantized aliases are also available:

- `metal-flash-attention-int8`
- `metal-flash-attention-int4`

These call UMFA's `metal_quantized_flash_attention_autograd` with blockwise quantization (`quant_mode=2`). SimpleTuner runs an additional startup check that verifies attached autograd outputs and finite multi-head gradients before enabling either alias.

## FLUX Sequence Lengths

For the tested FLUX.1 square-image LoRA path, the attention shape is `B,H,S,D = 1,24,S,128`. Sequence length scales with image area:

| Resolution | Expected `S` | Attention Shape |
| --- | ---: | --- |
| 512 | 1536 | `(1, 24, 1536, 128)` |
| 1024 | 6144 | `(1, 24, 6144, 128)` |
| 2048 | 24576 | `(1, 24, 24576, 128)` |

So yes, `S=24576` is the target synthetic attention shape for a 2048px square FLUX.1 sample in this configuration.

## Synthetic Memory Probe

The following numbers are from isolated FP32 forward+backward attention probes on MPS with `B=1,H=24,D=128`. They measure `torch.mps.driver_allocated_memory()` peaks and do not include model weights, optimizer state, VAE cache generation, data loading, or checkpointing.

| Sequence Length | PyTorch SDPA Peak Driver Memory | UMFA Peak Driver Memory | Result |
| ---: | ---: | ---: | --- |
| 1536 | 1.010 GiB | 1.009 GiB | both pass |
| 6144 | 11.983 GiB | 1.009 GiB | both pass |
| 8192 | 20.516 GiB | 1.009 GiB | both pass |
| 10240 | fails | 1.009 GiB | SDPA hits an MPSGraph tensor-dimension limit |
| 24576 | not tested after SDPA failure | 3.008 GiB | UMFA passes |
| 65536 | not tested after SDPA failure | 6.040 GiB | UMFA passes |

The SDPA failure at `S=10240` was:

```text
RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX
```

Direct parity at the 1024px attention shape (`S=6144`) matched PyTorch SDPA:

```text
forward max_abs=6.11e-07
loss_sdpa=0.0004459388
loss_metal=0.0004459389
q/k/v gradient mean_abs <= 4.1e-16
```

## One-Step FLUX Probe

The real trainer probe used a one-step FP32 FLUX.1 LoRA config with `train_batch_size=1`, no quantization, no validation, and the small Domokun dataset. These numbers are for smoke testing memory and shape behavior, not model quality.

| Run | Result | Step Time | Peak Process-Tree RSS | Notes |
| --- | --- | ---: | ---: | --- |
| 1024px PyTorch SDPA | pass | 40.66s | 46.991 GiB | baseline one-step run |
| 1024px UMFA | pass | 47.64s | 49.758 GiB | autograd path active; direct `S=6144` parity passes |
| 2048px UMFA, no VAE tiling | fail before train step | n/a | n/a | failed while creating 2048px VAE latents |
| 2048px PyTorch SDPA, no VAE tiling | fail before train step | n/a | n/a | same VAE cache failure before attention |
| 2048px UMFA, VAE tiling enabled | pass | 512.66s | 46.747 GiB | 27 VAE latents cached; `Metal SDPA backend initialized successfully`; `step_loss=0.256` |
| 2048px PyTorch SDPA, VAE tiling enabled | crash at train step | n/a | 46.292 GiB | process exited with `rc=-11` after entering the first training step; no Python traceback |

The first 2048px runs did not enable `vae_enable_tiling` and did not reach transformer attention. They failed during VAE latent cache generation:

```text
MPS backend out of memory (MPS allocated: 20.48 GiB, other allocations: 146.27 GiB, max allowed: 163.20 GiB).
```

With `vae_enable_tiling=true`, the 2048px VAE cache completed and the UMFA run completed the training step. The matching PyTorch SDPA run reused the hot tiled VAE cache, entered the first training step, and crashed without a Python traceback. This means UMFA reduces attention memory pressure, but it does not solve every high-resolution FLUX bottleneck; VAE cache generation must still be configured separately.

## Troubleshooting

- `metal_flash_attention_autograd` is missing: rebuild UMFA from a version with autograd support and reinstall the FFI package.
- `available False`: read `get_metal_flash_attention_unavailable_reason()`; SimpleTuner reports the exact failed import, availability, forward parity, or autograd parity check.
- Training silently falls back: verify tensors are MPS FP32 4D BHSD with at least four heads and sequence length at least 64, and that `dropout_p=0`, `is_causal=False`, no mask, and no GQA are used.
- 2048px FLUX fails before attention: this is likely VAE cache memory pressure, not UMFA attention memory. Enable `vae_enable_tiling=true` or generate/reuse latents with a lower-memory cache workflow before treating attention as the blocker.
