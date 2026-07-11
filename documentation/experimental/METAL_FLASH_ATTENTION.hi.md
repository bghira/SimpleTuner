# Metal Flash Attention

`metal-flash-attention` Apple Silicon पर eligible MPS SDPA calls को Universal Metal Flash Attention (UMFA) PyTorch FFI extension से चलाता है। यह experimental है और अभी FLUX-style FP32 training paths के लिए है, जहां PyTorch SDPA ज्यादा memory लेता है या long sequence lengths पर MPSGraph limits तक पहुंचता है।

## Requirements

- Apple Silicon with MPS available.
- Xcode command line tools with Metal toolchain.
- SimpleTuner Apple dependency set के साथ installed हो। Apple extra PyTorch `>=2.13.0` require करता है।
- UMFA build में `metal_flash_attention_autograd` expose होना चाहिए। पुराने forward-only builds SimpleTuner reject करता है।

SimpleTuner केवल MPS FP32 4D attention calls को UMFA पर dispatch करता है जब कम से कम चार heads हों और sequence length 64 या अधिक हो। FP16/BF16, masked, causal, grouped-query, tiny, 2D, और non-MPS calls PyTorch SDPA पर fallback होते हैं।

## UMFA Build Aur Install

SimpleTuner वाला Python environment use करें:

```bash
export ST_ROOT=/path/to/SimpleTuner
export UMFA_ROOT=/path/to/universal-metal-flash-attention
export PYTHON="$ST_ROOT/.venv/bin/python"
```

Swift package build करें और PyTorch FFI package install करें:

```bash
cd "$UMFA_ROOT"
git submodule update --init --recursive
swift build -c release

cd "$UMFA_ROOT/examples/pytorch-custom-op-ffi"
"$PYTHON" -m pip install --upgrade pip setuptools wheel pybind11 numpy
"$PYTHON" -m pip install --force-reinstall --no-deps --no-build-isolation --no-cache-dir .
```

Autograd binding verify करें:

```bash
"$PYTHON" - <<'PY'
import metal_sdpa_extension

print("metal_flash_attention_autograd" in dir(metal_sdpa_extension))
print([name for name in dir(metal_sdpa_extension) if "attention" in name])
PY
```

SimpleTuner availability verify करें:

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

Expected:

```text
available True
reason None
```

अगर output detached UMFA output बताता है, तो `metal_flash_attention_autograd` support वाली UMFA version rebuild करें।

## SimpleTuner Me Enable Karna

Config में attention mechanism set करें:

```json
{
  "attention_mechanism": "metal-flash-attention",
  "mixed_precision": "no",
  "base_model_default_dtype": "fp32"
}
```

Current integration के लिए `mixed_precision=no` और FP32 defaults important हैं। SimpleTuner BF16/FP16 attention को UMFA पर भेजने के बजाय fallback करता है।

## FLUX Sequence Lengths

Tested square-image FLUX.1 LoRA path में attention shape `B,H,S,D = 1,24,S,128` है। Sequence length image area के साथ scale करता है:

| Resolution | Expected `S` | Attention Shape |
| --- | ---: | --- |
| 512 | 1536 | `(1, 24, 1536, 128)` |
| 1024 | 6144 | `(1, 24, 6144, 128)` |
| 2048 | 24576 | `(1, 24, 24576, 128)` |

इसलिए इस config में 2048px square FLUX.1 sample के लिए synthetic target shape `S=24576` है।

## Synthetic Memory Probe

ये numbers isolated FP32 forward+backward attention probes से हैं, MPS पर `B=1,H=24,D=128` के साथ। ये `torch.mps.driver_allocated_memory()` peak मापते हैं और model weights, optimizer, VAE cache, data loading, checkpointing include नहीं करते।

| Sequence Length | PyTorch SDPA Peak | UMFA Peak | Result |
| ---: | ---: | ---: | --- |
| 1536 | 1.010 GiB | 1.009 GiB | दोनों pass |
| 6144 | 11.983 GiB | 1.009 GiB | दोनों pass |
| 8192 | 20.516 GiB | 1.009 GiB | दोनों pass |
| 10240 | fail | 1.009 GiB | SDPA MPSGraph limit hit करता है |
| 24576 | SDPA fail के बाद नहीं चलाया | 3.008 GiB | UMFA pass |
| 65536 | SDPA fail के बाद नहीं चलाया | 6.040 GiB | UMFA pass |

SDPA failure:

```text
RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX
```

1024px shape (`S=6144`) पर direct parity:

```text
forward max_abs=6.11e-07
loss_sdpa=0.0004459388
loss_metal=0.0004459389
q/k/v gradient mean_abs <= 4.1e-16
```

## One-Step FLUX Probe

Real trainer probe ने one-step FP32 FLUX.1 LoRA config use किया: `train_batch_size=1`, no quantization, no validation, small Domokun dataset। ये numbers smoke testing के लिए हैं, quality comparison के लिए नहीं।

| Run | Result | Step Time | Peak Process-Tree RSS | Notes |
| --- | --- | ---: | ---: | --- |
| 1024px PyTorch SDPA | pass | 40.66s | 46.991 GiB | baseline |
| 1024px UMFA | pass | 47.64s | 49.758 GiB | autograd active; direct `S=6144` parity pass |
| 2048px UMFA, no VAE tiling | train step से पहले fail | n/a | n/a | 2048px VAE latents बनाते समय fail |
| 2048px PyTorch SDPA, no VAE tiling | train step से पहले fail | n/a | n/a | attention से पहले वही VAE cache failure |
| 2048px UMFA, VAE tiling enabled | pass | 512.66s | 46.747 GiB | 27 VAE latents cached; `Metal SDPA backend initialized successfully`; `step_loss=0.256` |
| 2048px PyTorch SDPA, VAE tiling enabled | train step पर crash | n/a | 46.292 GiB | first training step में enter करने के बाद `rc=-11`; Python traceback नहीं |

पहले 2048px runs में `vae_enable_tiling` enabled नहीं था, इसलिए वे transformer attention तक नहीं पहुंचे:

```text
MPS backend out of memory (MPS allocated: 20.48 GiB, other allocations: 146.27 GiB, max allowed: 163.20 GiB).
```

`vae_enable_tiling=true` के साथ 2048px VAE cache complete हुआ और UMFA run ने training step complete किया। matching PyTorch SDPA run ने hot tiled VAE cache reuse किया, first training step में enter किया, और बिना Python traceback crash हुआ। UMFA attention memory pressure कम करता है, लेकिन high-resolution FLUX के हर bottleneck को solve नहीं करता; VAE cache generation को अलग से configure करना होगा।

## Troubleshooting

- `metal_flash_attention_autograd` missing: autograd support वाली UMFA rebuild करें और FFI package reinstall करें।
- `available False`: `get_metal_flash_attention_unavailable_reason()` पढ़ें।
- Unexpected fallback: MPS FP32 4D BHSD tensors, heads >= 4, sequence length >= 64, `dropout_p=0`, `is_causal=False`, no mask, no GQA verify करें।
- 2048px attention से पहले fail: यह likely VAE cache memory pressure है, UMFA attention memory नहीं। `vae_enable_tiling=true` enable करें या lower-memory cache workflow से latents generate/reuse करें।
