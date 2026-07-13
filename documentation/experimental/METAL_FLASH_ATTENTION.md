# Metal Flash Attention

`metal-flash-attention` routes eligible Apple Silicon MPS SDPA calls through the Universal Metal Flash Attention (UMFA) PyTorch FFI extension. It is experimental and currently intended for FLUX-style FP32/FP16/BF16 paths where PyTorch SDPA either uses more memory or hits MPSGraph limits at long sequence lengths.

## Requirements

- Apple Silicon with MPS available.
- Xcode command line tools with the Metal toolchain.
- SimpleTuner installed with the Apple dependency set. The Apple extra requires PyTorch `>=2.13.0`.
- A UMFA build that exposes `metal_flash_attention_autograd`, registers the PyTorch `MPS` dispatch key, and exposes `clear_quantization_mode`. The quantized aliases also require `metal_quantized_flash_attention_autograd`, `set_quantization_mode`, `QUANT_INT8`, `QUANT_INT4`, and `QUANT_BLOCK_WISE`.

SimpleTuner routes attention through PyTorch's MPS SDPA dispatcher, which current UMFA builds register. Eligible calls are MPS FP32/FP16/BF16 4D attention with any head count (single-head works) and any sequence length, including transposed FLUX-style layouts, bool/additive masks up to 4D, and causal calls. Eligible calls encode directly into PyTorch's MPS command stream — no per-call synchronization, no FP32 promotion for FP16/BF16 inputs. Causal training is eligible too — causal backward passes exact gradient parity. Calls with dropout or `enable_gqa` fall back to PyTorch SDPA. Older UMFA builds that registered `PrivateUse1` instead of `MPS` will silently use native PyTorch SDPA.

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
  "attention_mechanism": "metal-flash-attention"
}
```

FP32, FP16, and BF16 attention all run natively: FP16/BF16 inputs use low-precision kernels (BF16 keeps FP32 softmax accumulation) and the output is produced in the input dtype, so `mixed_precision: bf16` works without forcing FP32 anywhere. SimpleTuner still falls back for dropout and `enable_gqa`.

Quantized aliases are also available:

- `metal-flash-attention-int8`
- `metal-flash-attention-int4`

These set UMFA's global quantization mode with blockwise quantization (`quant_mode=2`) and use the quantized autograd entry point for direct-dispatched calls:

- `metal-flash-attention-int8`: `set_quantization_mode(ext.QUANT_INT8, ext.QUANT_BLOCK_WISE)`
- `metal-flash-attention-int4`: `set_quantization_mode(ext.QUANT_INT4, ext.QUANT_BLOCK_WISE)`

SimpleTuner clears that mode when switching back to FP32 UMFA or another attention backend. It also runs an additional startup check that verifies attached autograd outputs, finite multi-head gradients, dispatcher-level masked SDPA, and no PyTorch fallback before enabling either alias.

Both the regular and quantized dispatchers support bool masks (`True` means attend), additive float masks, batched masks such as `[B, H, S_q, S_kv]`, and broadcast masks such as `[B, 1, 1, S_kv]`. All-true bool masks are detected and skipped as a fast path. Masked calls stay on the in-stream path; mask expansion is encoded on the same command buffer as the attention kernel.

To verify that the MPS dispatcher is taking the expected path during a run, inspect UMFA's dispatch counters:

```python
import metal_sdpa_extension as ext

print(ext.get_dispatch_stats())
```

For unquantized inference/validation, `fp32_instream` should increase while `pytorch_fallback` stays at `0` (attention with any input dtype counts here — the name refers to the dispatch path, not the compute dtype). For quantized Z-Image training, `quantized_autograd` should increase instead. If the `encoder_attention_mask` is all true, `mask_all_true_skipped` should increase too. Calls routed through the fused RoPE entry point count under `rope_instream`.

## Fused RoPE + SDPA

The extension exposes `metal_sdpa_extension.rope_scaled_dot_product_attention(query, key, value, rope_cos, rope_sin, attn_mask=None, is_causal=False, scale=None)`, which applies interleaved-pair rotary embeddings to Q/K on the GPU immediately before attention — one command-buffer submit, no eager rotation passes, no FP32 materializations. This covers the RoPE convention shared by FLUX.1, FLUX.2, Krea2, and Z-Image (Z-Image's complex-multiply formulation is the same rotation); the models differ only in table format, which the caller adapts.

- Tensors are BHSD; strided views (e.g. `transpose(1, 2)` of a BSHD projection, or fused-QKV `unbind` views) are consumed without copies.
- `rope_cos`/`rope_sin` are pair-duplicated tables (`cos[2i] == cos[2i+1]`), shape `[S, D]`, `[1, S, D]`, or per-sample `[B, S, D]`; any float dtype is normalized to FP32 internally.
- Training flows through the fused path: a custom autograd applies the inverse rotation (the same pairwise rotation with sin negated — RoPE is orthonormal) to dQ/dK in backward, so gradients are returned with respect to pre-RoPE Q/K. Verified exact against a differentiable reference in FP32, including per-sample batched tables. Causal is supported with gradients as well; masked and GQA calls that require gradients still use eager rotation.
- GQA inputs (fewer K/V heads) are rotated at the K/V head count and expanded afterwards.

At the Z-Image DiT shape `(1, 30, 4128, 128)` in BF16, the fused path measured 4.4 ms/layer faster than eager rotation + SDPA in a 12-layer chain benchmark. Model integration in SimpleTuner is pending; until then the entry point is available for direct use.

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
- Training silently falls back: verify tensors are MPS FP32/FP16/BF16 4D BHSD (any head count and sequence length; single-head is supported) and that `dropout_p=0` and `enable_gqa` is not set. Bool/additive masks up to 4D are eligible. Causal calls are eligible with and without gradients. Verify the UMFA build registers the PyTorch `MPS` dispatch key and exposes `get_dispatch_stats()`; builds registered only on `PrivateUse1` will be bypassed by `torch.device("mps")` tensors.
- 2048px FLUX fails before attention: this is likely VAE cache memory pressure, not UMFA attention memory. Enable `vae_enable_tiling=true` or generate/reuse latents with a lower-memory cache workflow before treating attention as the blocker.
