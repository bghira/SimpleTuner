# Metal Flash Attention

`metal-flash-attention` 会把 Apple Silicon 上符合条件的 MPS SDPA 调用路由到 Universal Metal Flash Attention (UMFA) 的 PyTorch FFI 扩展。它仍是 experimental，当前主要面向 FLUX-style FP32/FP16/BF16 paths，用于 PyTorch SDPA 占用更多内存或在长 sequence length 上触发 MPSGraph 限制的场景。

## Requirements

- 支持 MPS 的 Apple Silicon。
- 带 Metal toolchain 的 Xcode command line tools。
- 使用 Apple dependency set 安装的 SimpleTuner。Apple extra 要求 PyTorch `>=2.13.0`。
- UMFA build 必须暴露 `metal_flash_attention_autograd`，注册 PyTorch `MPS` dispatch key，并暴露 `clear_quantization_mode`。量化 aliases 还需要 `metal_quantized_flash_attention_autograd`、`set_quantization_mode`、`QUANT_INT8`、`QUANT_INT4` 和 `QUANT_BLOCK_WISE`。

SimpleTuner 通过 PyTorch 的 MPS SDPA dispatcher 路由注意力，当前的 UMFA build 会注册该 dispatcher。符合条件的调用为 MPS FP32/FP16/BF16 4D 注意力，支持任意 head 数量（single-head 可用）和任意 sequence length，包括 transposed FLUX-style layouts、最多 4D 的 bool/additive masks，以及 causal 调用。符合条件的调用会直接编码进 PyTorch 的 MPS command stream —— 没有每次调用的同步，FP16/BF16 输入也不会被提升为 FP32。causal 训练同样符合条件 —— causal backward 通过了精确的梯度一致性验证。带 dropout 或 `enable_gqa` 的调用会 fallback 到 PyTorch SDPA。注册在 `PrivateUse1` 而非 `MPS` 的旧 UMFA build 会被静默绕过，使用原生 PyTorch SDPA。

## Build And Install UMFA

使用运行 SimpleTuner 的同一个 Python environment:

```bash
export ST_ROOT=/path/to/SimpleTuner
export UMFA_ROOT=/path/to/universal-metal-flash-attention
export PYTHON="$ST_ROOT/.venv/bin/python"
```

构建 Swift package 并安装 PyTorch FFI package:

```bash
cd "$UMFA_ROOT"
git submodule update --init --recursive
swift build -c release

cd "$UMFA_ROOT/examples/pytorch-custom-op-ffi"
"$PYTHON" -m pip install --upgrade pip setuptools wheel pybind11 numpy
"$PYTHON" -m pip install --force-reinstall --no-deps --no-build-isolation --no-cache-dir .
```

确认 extension 暴露 autograd binding:

```bash
"$PYTHON" - <<'PY'
import metal_sdpa_extension

print("metal_flash_attention_autograd" in dir(metal_sdpa_extension))
print([name for name in dir(metal_sdpa_extension) if "attention" in name])
PY
```

确认 SimpleTuner 接受该 extension:

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

如果输出提示 UMFA output is detached，请从实现 `metal_flash_attention_autograd` 的 UMFA 版本重新构建。

## Enable In SimpleTuner

设置 attention mechanism:

```json
{
  "attention_mechanism": "metal-flash-attention"
}
```

FP32、FP16、BF16 注意力均为原生执行：FP16/BF16 输入使用低精度 kernel（BF16 的 softmax 仍以 FP32 累加），输出以输入 dtype 产出，因此 `mixed_precision: bf16` 可直接使用，任何环节都不会强制 FP32。带 dropout 或 `enable_gqa` 的调用仍会 fallback。

也可以使用量化 aliases:

- `metal-flash-attention-int8`
- `metal-flash-attention-int4`

这些 aliases 会以 blockwise quantization（`quant_mode=2`）设置 UMFA 的全局量化模式，并在直接分发的 calls 中使用量化 autograd 入口:

- `metal-flash-attention-int8`: `set_quantization_mode(ext.QUANT_INT8, ext.QUANT_BLOCK_WISE)`
- `metal-flash-attention-int4`: `set_quantization_mode(ext.QUANT_INT4, ext.QUANT_BLOCK_WISE)`

切回 FP32 UMFA 或其他 attention backend 时，SimpleTuner 会清除该模式。SimpleTuner 还会在启用任一 alias 前执行额外的启动检查，确认输出连接到 autograd、multi-head gradients 有限、dispatcher-level masked SDPA 可用且没有 PyTorch fallback。

常规与量化 dispatcher 都支持 bool masks（`True` 表示参与注意力）、additive float masks、批量 masks（如 `[B, H, S_q, S_kv]`）以及广播 masks（如 `[B, 1, 1, S_kv]`）。all-true bool masks 会被检测并作为 fast path 跳过。带 mask 的调用仍走 in-stream 路径；mask 展开与注意力 kernel 编码在同一个 command buffer 上。

要确认 MPS dispatcher 在实际运行中走了预期路径，可以查看 UMFA dispatch counters:

```python
import metal_sdpa_extension as ext

print(ext.get_dispatch_stats())
```

对于未量化的推理/验证，`fp32_instream` 应增加且 `pytorch_fallback` 保持为 `0`（任何输入 dtype 的注意力都计入此项 —— 名称指的是 dispatch 路径而非计算 dtype）。量化的 Z-Image 训练则应看到 `quantized_autograd` 增加。若 `encoder_attention_mask` 为 all-true，`mask_all_true_skipped` 也会增加。经由融合 RoPE 入口的 no-grad 调用计入 `rope_instream`；带梯度调用计入 `rope_autograd`。

## 融合 RoPE + SDPA

扩展提供 `metal_sdpa_extension.rope_scaled_dot_product_attention(query, key, value, rope_cos, rope_sin, attn_mask=None, is_causal=False, scale=None)`，在注意力之前直接在 GPU 上对 Q/K 应用 interleaved-pair rotary embeddings。符合条件的 no-grad 调用会留在 in-stream attention 路径中，没有 eager 旋转 pass，也没有 FP32 tensor 物化。它覆盖 FLUX.1、FLUX.2、Krea2 和 Z-Image 共享的 RoPE 约定（Z-Image 的 complex-multiply 写法是同一种旋转）；各模型仅在表格式上不同，由调用方适配。

- 张量为 BHSD；strided views（例如 BSHD 投影的 `transpose(1, 2)`，或 fused-QKV 的 `unbind` views）无需拷贝即可使用。
- `rope_cos`/`rope_sin` 为 pair-duplicated 表（`cos[2i] == cos[2i+1]`），形状为 `[S, D]`、`[1, S, D]` 或按样本的 `[B, S, D]`；任意 float dtype 会在内部归一化为 FP32。
- 训练走融合路径：自定义 autograd 在 backward 中对 dQ/dK 应用逆旋转（同一 pairwise 旋转、sin 取负 —— RoPE 是正交的），因此返回的是相对 pre-RoPE Q/K 的梯度。已用可微分参考在 FP32 下验证为精确一致，含按样本的 batched 表。causal 在带梯度时同样支持；需要梯度的带 mask 或 GQA 调用仍使用 eager 旋转。
- GQA 输入（K/V head 较少）按 K/V head 数量旋转后再扩展。

在 Z-Image DiT 形状 `(1, 30, 4128, 128)`、BF16 下，融合路径在 12 层链式 benchmark 中比 eager 旋转 + SDPA 每层快 4.4 ms。SimpleTuner 的模型集成尚未完成；在此之前该入口可直接使用。

## FLUX Sequence Lengths

在测试的 square-image FLUX.1 LoRA path 中，attention shape 是 `B,H,S,D = 1,24,S,128`。Sequence length 会随 image area 缩放:

| Resolution | Expected `S` | Attention Shape |
| --- | ---: | --- |
| 512 | 1536 | `(1, 24, 1536, 128)` |
| 1024 | 6144 | `(1, 24, 6144, 128)` |
| 2048 | 24576 | `(1, 24, 24576, 128)` |

因此，在此配置中，2048px square FLUX.1 sample 的 synthetic attention target shape 是 `S=24576`。

## Synthetic Memory Probe

以下数字来自 MPS 上 `B=1,H=24,D=128` 的 isolated FP32 forward+backward attention probes。它们测量 `torch.mps.driver_allocated_memory()` peak，不包含 model weights、optimizer state、VAE cache generation、data loading 或 checkpointing。

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

1024px shape (`S=6144`) 的 direct parity:

```text
forward max_abs=6.11e-07
loss_sdpa=0.0004459388
loss_metal=0.0004459389
q/k/v gradient mean_abs <= 4.1e-16
```

## One-Step FLUX Probe

真实 trainer probe 使用 one-step FP32 FLUX.1 LoRA config，`train_batch_size=1`，无 quantization，无 validation，数据集为小型 Domokun。以下数字用于 memory 和 shape smoke test，不用于质量比较。

| Run | Result | Step Time | Peak Process-Tree RSS | Notes |
| --- | --- | ---: | ---: | --- |
| 1024px PyTorch SDPA | pass | 40.66s | 46.991 GiB | baseline |
| 1024px UMFA | pass | 47.64s | 49.758 GiB | autograd active; direct `S=6144` parity passes |
| 2048px UMFA, no VAE tiling | fail before train step | n/a | n/a | failed while creating 2048px VAE latents |
| 2048px PyTorch SDPA, no VAE tiling | fail before train step | n/a | n/a | same VAE cache failure before attention |
| 2048px UMFA, VAE tiling enabled | pass | 512.66s | 46.747 GiB | 27 VAE latents cached; `Metal SDPA backend initialized successfully`; `step_loss=0.256` |
| 2048px PyTorch SDPA, VAE tiling enabled | crash at train step | n/a | 46.292 GiB | entered first training step, then exited with `rc=-11`; no Python traceback |

最初的 2048px runs 没有启用 `vae_enable_tiling`，因此没有到达 transformer attention:

```text
MPS backend out of memory (MPS allocated: 20.48 GiB, other allocations: 146.27 GiB, max allowed: 163.20 GiB).
```

启用 `vae_enable_tiling=true` 后，2048px VAE cache 可以完成，UMFA run 也完成了 training step。匹配的 PyTorch SDPA run 复用了已热的 tiled VAE cache，进入 first training step 后无 Python traceback 崩溃。UMFA 会降低 attention memory pressure，但不会解决 high-resolution FLUX 的所有瓶颈；VAE cache generation 仍需单独配置。

## Troubleshooting

- 缺少 `metal_flash_attention_autograd`: 重新构建带 autograd support 的 UMFA，并重新安装 FFI package。
- `available False`: 查看 `get_metal_flash_attention_unavailable_reason()`。
- 训练静默 fallback：确认张量为 MPS FP32/FP16/BF16 4D BHSD（任意 head 数量与 sequence length；支持 single-head），`dropout_p=0` 且未设置 `enable_gqa`。最多 4D 的 bool/additive masks 均符合条件。causal 调用在有无梯度时都符合条件。确认 UMFA build 注册了 PyTorch `MPS` dispatch key 并暴露 `get_dispatch_stats()`；仅注册在 `PrivateUse1` 的 build 会被 `torch.device("mps")` tensors 绕过。
- 2048px 在 attention 前失败: 这通常是 VAE cache memory pressure，而不是 UMFA attention memory。启用 `vae_enable_tiling=true`，或使用更低内存的 cache workflow 生成/复用 latents。
