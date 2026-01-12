# FlexAttention 指南

**FlexAttention 需要 CUDA 设备。**

FlexAttention 是 PyTorch 2.5.0 中引入的块级注意力内核。它将 SDPA 计算改写为可编程循环，使你无需编写 CUDA 就能表达掩码策略。Diffusers 通过新的 `attention_backend` 分发器暴露该能力，SimpleTuner 则将其连接到 `--attention_mechanism=flex`。

> ⚠️ FlexAttention 在上游仍标记为 “prototype”。当你更换驱动、CUDA 版本或 PyTorch 构建时，可能需要重新编译。

## 前置条件

1. **Ampere+ GPU** – 支持 NVIDIA SM80（A100）、Ada（4090/L40S）或 Hopper（H100/H200）。旧卡会在内核注册时失败。
2. **编译工具链** – 内核运行时用 `nvcc` 编译。请安装与 wheel 匹配的 `cuda-nvcc`（当前发行版为 CUDA 12.8），并确保 `$PATH` 中能找到 `nvcc`。

## 构建内核

首次导入 `torch.nn.attention.flex_attention` 时会将 CUDA 扩展编译到 PyTorch 的惰性缓存中。你也可以提前执行以尽早暴露编译错误：

```bash
python - <<'PY'
import torch
from torch.nn.attention import flex_attention

assert torch.__version__ >= "2.5.0", torch.__version__
flex_attention.build_flex_attention_kernels()  # no-op when already compiled
print("FlexAttention kernels installed at", flex_attention.kernel_root)
PY
```

- 若出现 `AttributeError: flex_attention has no attribute build_flex_attention_kernels`，请升级 PyTorch（2.5.0+ 才有该辅助函数）。
- 缓存在 `~/.cache/torch/kernels`。若升级 CUDA 需要强制重建，请删除该目录。

## 在 SimpleTuner 中启用 FlexAttention

内核存在后，通过 `config.json` 选择后端：

```json
{
  "attention_mechanism": "flex"
}
```

你可以预期：

- 只有启用 Diffusers `attention_backend` 的 Transformer 块（Flux、Wan 2.2、LTXVideo、QwenImage 等）才会走该路径。经典 SD/SDXL UNet 仍会直接调用 PyTorch SDPA，因此不会生效。
- FlexAttention 目前只支持 BF16/FP16 张量。若使用 FP32 或 FP8 权重会触发 `ValueError: Query, key, and value must be either bfloat16 or float16`。
- 该后端仅支持 `is_causal=False`。传入 mask 会转换为内核所需的块掩码，但任意 ragged mask 仍不支持（与上游一致）。

## 排障清单

| 症状 | 解决办法 |
| --- | --- |
| `RuntimeError: Flex Attention backend 'flex' is not usable because of missing package` | PyTorch 版本 < 2.5 或不含 CUDA。安装新的 CUDA wheel。 |
| `Could not compile flex_attention kernels` | 确保 `nvcc` 与 torch wheel 期望的 CUDA 版本一致（12.1+）。若安装器找不到头文件，请设置 `export CUDA_HOME=/usr/local/cuda-12.4`。 |
| `ValueError: Query, key, and value must be on a CUDA device` | FlexAttention 仅支持 CUDA。Apple/ROCm 运行时请移除该后端设置。 |
| 训练始终未切换到该后端 | 请确认模型家族已使用 Diffusers 的 `dispatch_attention_fn`（Flux/Wan/LTXVideo）。标准 SD UNet 无论选择何种后端仍使用 PyTorch SDPA。 |

更深入的内部机制与 API 标志请参考上游文档：[PyTorch FlexAttention docs](https://pytorch.org/docs/stable/nn.attention.html#flexattention)。
