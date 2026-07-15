# Metal Flash Attention

`metal-flash-attention` Apple Silicon पर eligible MPS SDPA calls को Universal Metal Flash Attention (UMFA) PyTorch FFI extension से चलाता है। यह experimental है और अभी FLUX-style FP32/FP16/BF16 paths के लिए है, जहां PyTorch SDPA ज्यादा memory लेता है या long sequence lengths पर MPSGraph limits तक पहुंचता है।

## Requirements

- Apple Silicon with MPS available.
- Xcode command line tools with Metal toolchain.
- SimpleTuner Apple dependency set के साथ installed हो। Apple extra PyTorch `>=2.13.0` require करता है।
- UMFA build में `metal_flash_attention_autograd` expose होना चाहिए, PyTorch `MPS` dispatch key register होनी चाहिए, और `clear_quantization_mode` expose होना चाहिए। Quantized aliases के लिए `metal_quantized_flash_attention_autograd`, `set_quantization_mode`, `QUANT_INT8`, `QUANT_INT4`, और `QUANT_BLOCK_WISE` भी चाहिए।

SimpleTuner attention को PyTorch के MPS SDPA dispatcher के जरिए route करता है, जिसे मौजूदा UMFA builds register करते हैं। योग्य calls हैं: MPS FP32/FP16/BF16 4D attention, किसी भी head count के साथ (single-head काम करता है) और किसी भी sequence length पर — इसमें transposed FLUX-style layouts, 4D तक के bool/additive masks, और causal calls शामिल हैं। योग्य calls सीधे PyTorch के MPS command stream में encode होती हैं — per-call synchronization नहीं, और FP16/BF16 inputs के लिए FP32 promotion नहीं। causal training भी योग्य है — causal backward exact gradient parity पास करता है। dropout या `enable_gqa` वाली calls PyTorch SDPA पर fallback करती हैं। `MPS` की जगह `PrivateUse1` पर register हुए पुराने UMFA builds चुपचाप native PyTorch SDPA उपयोग करते हैं।

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
  "attention_mechanism": "metal-flash-attention"
}
```

FP32, FP16 और BF16 attention सभी natively चलते हैं: FP16/BF16 inputs low-precision kernels उपयोग करते हैं (BF16 में softmax accumulation FP32 में रहता है) और output input dtype में ही बनता है, इसलिए `mixed_precision: bf16` कहीं भी FP32 को force किए बिना काम करता है। dropout और `enable_gqa` पर SimpleTuner अभी भी fallback करता है।

Quantized aliases भी उपलब्ध हैं:

- `metal-flash-attention-int8`
- `metal-flash-attention-int4`

ये blockwise quantization (`quant_mode=2`) के साथ UMFA का global quantization mode set करते हैं और direct-dispatched calls के लिए quantized autograd entry point use करते हैं:

- `metal-flash-attention-int8`: `set_quantization_mode(ext.QUANT_INT8, ext.QUANT_BLOCK_WISE)`
- `metal-flash-attention-int4`: `set_quantization_mode(ext.QUANT_INT4, ext.QUANT_BLOCK_WISE)`

FP32 UMFA या किसी और attention backend पर लौटते समय SimpleTuner इस mode को clear करता है। SimpleTuner किसी भी alias को enable करने से पहले autograd से जुड़े outputs, finite multi-head gradients, dispatcher-level masked SDPA, और no PyTorch fallback verify करने वाला अतिरिक्त startup check भी चलाता है।

regular और quantized दोनों dispatchers bool masks (`True` का मतलब attend), additive float masks, `[B, H, S_q, S_kv]` जैसे batched masks, और `[B, 1, 1, S_kv]` जैसे broadcast masks सपोर्ट करते हैं। all-true bool masks detect होकर fast path के रूप में skip हो जाते हैं। masked calls in-stream path पर ही रहती हैं; mask expansion attention kernel के साथ उसी command buffer पर encode होता है।

Run के दौरान MPS dispatcher expected path ले रहा है या नहीं, यह verify करने के लिए UMFA dispatch counters देखें:

```python
import metal_sdpa_extension as ext

print(ext.get_dispatch_stats())
```

बिना quantization वाली inference/validation में `fp32_instream` बढ़ना चाहिए और `pytorch_fallback` `0` पर रहना चाहिए (किसी भी input dtype की attention यहीं count होती है — नाम dispatch path का है, compute dtype का नहीं)। quantized Z-Image training में इसकी जगह `quantized_autograd` बढ़ता है। अगर `encoder_attention_mask` all-true है तो `mask_all_true_skipped` भी बढ़ेगा। fused RoPE entry point से गई no-grad calls `rope_instream` में count होती हैं; gradient calls `rope_autograd` में count होती हैं।

## Fused RoPE + SDPA

extension `metal_sdpa_extension.rope_scaled_dot_product_attention(query, key, value, rope_cos, rope_sin, attn_mask=None, is_causal=False, scale=None)` उपलब्ध कराता है, जो attention से ठीक पहले GPU पर Q/K पर interleaved-pair rotary embeddings लागू करता है। Eligible no-grad calls in-stream attention path पर रहती हैं, बिना eager rotation passes या FP32 tensor materializations के। यह FLUX.1, FLUX.2, Krea2 और Z-Image की साझा RoPE convention को cover करता है (Z-Image का complex-multiply रूप वही rotation है); मॉडल केवल table format में अलग हैं, जिसे caller adapt करता है।

- tensors BHSD होते हैं; strided views (जैसे BSHD projection का `transpose(1, 2)`, या fused-QKV के `unbind` views) बिना copy के उपयोग होते हैं।
- `rope_cos`/`rope_sin` pair-duplicated tables हैं (`cos[2i] == cos[2i+1]`), आकार `[S, D]`, `[1, S, D]` या per-sample `[B, S, D]`; कोई भी float dtype internally FP32 में normalize होता है।
- training fused path से ही चलती है: एक custom autograd backward में dQ/dK पर inverse rotation लागू करता है (वही pairwise rotation, sin negate करके — RoPE orthonormal है), इसलिए gradients pre-RoPE Q/K के सापेक्ष लौटते हैं। FP32 में differentiable reference के विरुद्ध exact सत्यापित, per-sample batched tables समेत। causal gradients के साथ भी समर्थित है; gradients चाहने वाली masked या GQA calls अभी भी eager rotation उपयोग करती हैं।
- GQA inputs (कम K/V heads) K/V head count पर rotate होकर बाद में expand होते हैं।

Z-Image के DiT shape `(1, 30, 4128, 128)` पर BF16 में, 12-layer chain benchmark में fused path eager rotation + SDPA से 4.4 ms/layer तेज़ मापा गया। SimpleTuner में model integration अभी बाकी है; तब तक entry point सीधे उपयोग के लिए उपलब्ध है।

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
- Training चुपचाप fallback करती है: सुनिश्चित करें कि tensors MPS FP32/FP16/BF16 4D BHSD हैं (कोई भी head count और sequence length; single-head समर्थित है), `dropout_p=0` है और `enable_gqa` सेट नहीं है। 4D तक के bool/additive masks योग्य हैं। causal calls gradients के साथ और बिना दोनों योग्य हैं। सुनिश्चित करें कि UMFA build PyTorch `MPS` dispatch key register करता है और `get_dispatch_stats()` expose करता है; केवल `PrivateUse1` पर register हुए builds `torch.device("mps")` tensors द्वारा bypass हो जाते हैं।
- 2048px attention से पहले fail: यह likely VAE cache memory pressure है, UMFA attention memory नहीं। `vae_enable_tiling=true` enable करें या lower-memory cache workflow से latents generate/reuse करें।
