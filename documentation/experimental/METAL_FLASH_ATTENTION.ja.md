# Metal Flash Attention

`metal-flash-attention` は、Apple Silicon の対象 MPS SDPA 呼び出しを Universal Metal Flash Attention (UMFA) PyTorch FFI extension に送ります。これは experimental で、現時点では PyTorch SDPA が多くの memory を使う、または long sequence length で MPSGraph limit に当たる FLUX-style FP32 training path 向けです。

## Requirements

- MPS が使える Apple Silicon。
- Metal toolchain を含む Xcode command line tools。
- Apple dependency set で SimpleTuner をインストールしていること。Apple extra は PyTorch `>=2.13.0` を要求します。
- `metal_flash_attention_autograd` を公開する UMFA build。古い forward-only build は SimpleTuner が拒否します。

SimpleTuner は、MPS FP32 4D attention call のうち、heads が 4 以上で sequence length が 64 以上のものだけを UMFA に dispatch します。FP16/BF16、mask 付き、causal、grouped-query、tiny、2D、non-MPS call は PyTorch SDPA に fallback します。

## UMFA Build And Install

SimpleTuner と同じ Python environment を使います:

```bash
export ST_ROOT=/path/to/SimpleTuner
export UMFA_ROOT=/path/to/universal-metal-flash-attention
export PYTHON="$ST_ROOT/.venv/bin/python"
```

Swift package を build し、PyTorch FFI package を install します:

```bash
cd "$UMFA_ROOT"
git submodule update --init --recursive
swift build -c release

cd "$UMFA_ROOT/examples/pytorch-custom-op-ffi"
"$PYTHON" -m pip install --upgrade pip setuptools wheel pybind11 numpy
"$PYTHON" -m pip install --force-reinstall --no-deps --no-build-isolation --no-cache-dir .
```

autograd binding を確認します:

```bash
"$PYTHON" - <<'PY'
import metal_sdpa_extension

print("metal_flash_attention_autograd" in dir(metal_sdpa_extension))
print([name for name in dir(metal_sdpa_extension) if "attention" in name])
PY
```

SimpleTuner 側の availability を確認します:

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

UMFA output が detached と表示される場合は、`metal_flash_attention_autograd` を実装した UMFA version から rebuild してください。

## SimpleTuner De Enable

attention mechanism を設定します:

```json
{
  "attention_mechanism": "metal-flash-attention",
  "mixed_precision": "no",
  "base_model_default_dtype": "fp32"
}
```

現在の integration では `mixed_precision=no` と FP32 default が重要です。SimpleTuner は BF16/FP16 attention を UMFA に送らず fallback します。

## FLUX Sequence Lengths

テストした square-image FLUX.1 LoRA path では attention shape は `B,H,S,D = 1,24,S,128` です。Sequence length は image area に比例します:

| Resolution | Expected `S` | Attention Shape |
| --- | ---: | --- |
| 512 | 1536 | `(1, 24, 1536, 128)` |
| 1024 | 6144 | `(1, 24, 6144, 128)` |
| 2048 | 24576 | `(1, 24, 24576, 128)` |

したがって、この config の 2048px square FLUX.1 sample の synthetic attention target は `S=24576` です。

## Synthetic Memory Probe

以下は MPS 上で `B=1,H=24,D=128` の isolated FP32 forward+backward attention probe を実行した結果です。`torch.mps.driver_allocated_memory()` peak を測っており、model weights、optimizer state、VAE cache generation、data loading、checkpointing は含みません。

| Sequence Length | PyTorch SDPA Peak | UMFA Peak | Result |
| ---: | ---: | ---: | --- |
| 1536 | 1.010 GiB | 1.009 GiB | both pass |
| 6144 | 11.983 GiB | 1.009 GiB | both pass |
| 8192 | 20.516 GiB | 1.009 GiB | both pass |
| 10240 | fail | 1.009 GiB | SDPA hits MPSGraph limit |
| 24576 | not tested after SDPA failure | 3.008 GiB | UMFA passes |
| 65536 | not tested after SDPA failure | 6.040 GiB | UMFA passes |

SDPA failure:

```text
RuntimeError: MPSGraph does not support tensor dims larger than INT_MAX
```

1024px shape (`S=6144`) の direct parity:

```text
forward max_abs=6.11e-07
loss_sdpa=0.0004459388
loss_metal=0.0004459389
q/k/v gradient mean_abs <= 4.1e-16
```

## One-Step FLUX Probe

実際の trainer probe は、one-step FP32 FLUX.1 LoRA config、`train_batch_size=1`、no quantization、no validation、小さな Domokun dataset で実行しました。これは memory と shape の smoke test であり、quality comparison ではありません。

| Run | Result | Step Time | Peak Process-Tree RSS | Notes |
| --- | --- | ---: | ---: | --- |
| 1024px PyTorch SDPA | pass | 40.66s | 46.991 GiB | baseline |
| 1024px UMFA | pass | 47.64s | 49.758 GiB | autograd active; direct `S=6144` parity passes |
| 2048px UMFA, VAE tiling なし | fail before train step | n/a | n/a | 2048px VAE latents 作成中に fail |
| 2048px PyTorch SDPA, VAE tiling なし | fail before train step | n/a | n/a | attention 前に同じ VAE cache failure |
| 2048px UMFA, VAE tiling enabled | pass | 512.66s | 46.747 GiB | 27 VAE latents cached; `Metal SDPA backend initialized successfully`; `step_loss=0.256` |
| 2048px PyTorch SDPA, VAE tiling enabled | crash at train step | n/a | 46.292 GiB | first training step に入ったあと `rc=-11` で終了。Python traceback なし |

最初の 2048px run は `vae_enable_tiling` を有効にしておらず、transformer attention まで到達していません:

```text
MPS backend out of memory (MPS allocated: 20.48 GiB, other allocations: 146.27 GiB, max allowed: 163.20 GiB).
```

`vae_enable_tiling=true` では 2048px VAE cache が完了し、UMFA run は training step を完了しました。同じ tiled VAE cache を使った PyTorch SDPA run は first training step に入ったあと、Python traceback なしで crash しました。UMFA は attention memory pressure を下げますが、high-resolution FLUX の全 bottleneck を解決するわけではありません。VAE cache generation は別途設定が必要です。

## Troubleshooting

- `metal_flash_attention_autograd` がない: autograd support 付き UMFA を rebuild して FFI package を reinstall します。
- `available False`: `get_metal_flash_attention_unavailable_reason()` を確認します。
- 予期しない fallback: MPS FP32 4D BHSD tensors、heads >= 4、sequence length >= 64、`dropout_p=0`、`is_causal=False`、mask なし、GQA なしを確認します。
- 2048px が attention 前に fail する: UMFA attention memory ではなく VAE cache memory pressure の可能性が高いです。`vae_enable_tiling=true` を有効にするか、より低メモリの cache workflow で latents を生成または再利用してください。
