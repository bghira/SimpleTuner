# Unsloth 风格 Checkpointing

短版：训练差一点放不下时再用；模型支持时先试 FFN-only。

`unsloth` backend 会把保存的 activation tensor 卸载到 CPU。`torch` backend 则丢掉这些 activation，在 backward 时重算。Unsloth 可以帮你挤出最后几个 GiB，用来提高 batch、分辨率或帧数。它不是免费加速。如果 `torch` 已经能跑，通常继续用 `torch`。

## 控制项

```json
{
  "gradient_checkpointing": true,
  "gradient_checkpointing_backend": "torch-ffn"
}
```

`gradient_checkpointing_backend` 有四个常用值：

| Value | Scope | Path | 适合场景 |
| --- | --- | --- | --- |
| `torch` | whole block | recompute | CPU offload 之前，需要最大内置显存节省。 |
| `torch-ffn` | feed-forward | recompute | Flash Attention 已处理 attention memory 后，想要便宜的节省。 |
| `unsloth` | whole block | CPU offload | torch layer checkpointing 仍然放不下。 |
| `unsloth-ffn` | feed-forward | CPU offload | torch FFN-only 差一点放下，CPU offload 买最后一点空间。 |

支持的模型族还可以减少 checkpoint 的 block 数量：

```json
{
  "gradient_checkpointing": true,
  "gradient_checkpointing_backend": "torch-ffn",
  "gradient_checkpointing_interval": 2
}
```

`gradient_checkpointing_interval: 2` 会在受支持的 whole-block 路径上 checkpoint 连续的两个 block chunk。值越大，重算越少，VRAM 里保留的 activation 越多。

在这些分段路径上，`gradient_checkpointing_segment_stride` 也可以和 `unsloth` 一起使用。把它当作 fit lever，而不是 speed lever：跳过的 blocks 仍留在 GPU，checkpointed blocks 仍会把保存 tensor CPU offload。Torch-only 概览和模型 benchmark 见 [Segmented Checkpointing](SEGMENTED_CHECKPOINTING.md)。

`gradient_checkpointing_offload_attention` 独立于 backend。在支持 attention/FFN 分离的 blocks 上，它会 offload attention 侧保存的 activations。它可以单独运行；当模型支持所选 backend 时，也可与 `torch`、`torch-ffn`、`unsloth` 或 `unsloth-ffn` 组合。

`gradient_checkpointing_offload_pin_memory_max_buckets` 控制 offloaded saved tensors 的 pinned CPU pooling。默认是 `12` 个不同 tensor buckets；设为 `0` 时只使用普通 CPU memory。

`torch-ffn` 和 `unsloth-ffn` 目前支持 Chroma、Flux、Krea 2、LTXVideo2、MageFlow、Wan 和 Z-Image。其他模型族会明确报错，直到它们的 block 暴露同样安全的边界。

## 它交换了什么

- `torch`：丢弃中间 activations，在 backward 重算。
- `unsloth`：把一部分 tensor 存到 CPU，backward 时再拷回 GPU。
- `*-ffn`：在有清晰 FFN 边界的模型上，只 checkpoint feed-forward 部分。
- Flash Attention 已经避免了巨大的 attention matrix。所谓“免费 checkpointing”主要是 attention，不是整个 transformer block。
- 当 activation 很大、峰值不是参数或 optimizer 主导时，CPU offload 最有用。

它需要 CUDA 和足够的 CPU RAM。PCIe 带宽很重要。如果 CPU-GPU 拷贝不能被其他计算隐藏，step 会变慢。

## 我们的 Sweep

合成 transformer block，bf16，flash SDPA，冻结 base weights，batch 1。这不是模型保证，只是展示 tradeoff。

### Packed Image Latents

2x2 packing 下，`64x64`、`128x128`、`256x256` 会变成 `1024`、`4096`、`16384` 个 transformer tokens。

| GPU | Tokens | No checkpoint | Torch FFN | Unsloth FFN | Torch layer | Unsloth layer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 80GB | 1024 | 0.0166s / 4.43 GiB | 0.0191s / 4.08 GiB | 0.0233s / 4.00 GiB | 0.0231s / 3.64 GiB | 0.0265s / 3.56 GiB |
| H100 80GB | 4096 | 0.0948s / 7.43 GiB | 0.1029s / 6.02 GiB | 0.1157s / 5.67 GiB | 0.1233s / 4.26 GiB | 0.1358s / 3.93 GiB |
| H100 80GB | 16384 | 0.8781s / 19.39 GiB | 0.9117s / 13.77 GiB | 0.9632s / 12.36 GiB | 1.1157s / 6.72 GiB | 1.1662s / 5.41 GiB |
| L40S | 1024 | 0.0500s / 4.39 GiB | 0.0575s / 4.04 GiB | 0.0627s / 3.95 GiB | 0.0666s / 3.60 GiB | 0.0725s / 3.51 GiB |
| L40S | 4096 | 0.2461s / 7.38 GiB | 0.2729s / 5.97 GiB | 0.2933s / 5.62 GiB | 0.3169s / 4.21 GiB | 0.3369s / 3.88 GiB |
| L40S | 16384 | 1.8153s / 19.35 GiB | 1.9639s / 13.72 GiB | 2.0250s / 12.31 GiB | 2.3360s / 6.67 GiB | 2.4218s / 5.36 GiB |

`1024` tokens 时，额外 offload 基本没意义，除非你已经顶到显存墙。`16384` tokens 时，`torch-ffn` 是便宜的一步，whole-layer checkpointing 才是大幅降低显存的手段。`unsloth` 比 torch layer checkpointing 再省大约 `1.3 GiB`。

### 更大的 Transformer

冻结 `32` 层，宽度 `4096`，`3072` tokens：

| GPU | No checkpoint | Torch FFN | Unsloth FFN | Torch layer | Unsloth layer |
| --- | ---: | ---: | ---: | ---: | ---: |
| H100 80GB | 0.1943s / 14.56 GiB | 0.2138s / 11.65 GiB | 0.2317s / 10.92 GiB | 0.2527s / 8.01 GiB | 0.2722s / 7.30 GiB |
| L40S | 0.5045s / 14.51 GiB | 0.5640s / 11.60 GiB | 0.5932s / 10.88 GiB | 0.6491s / 7.96 GiB | 0.6864s / 7.26 GiB |

全量可训练权重会改变结果：gradients 和 optimizer state 主导峰值时，`unsloth` 在我们的合成 run 里没有比 `torch` 多省显存。PEFT 更接近冻结权重的情况。

## 实用规则

1. 如果不开 checkpointing 就能跑，关掉它。
2. 如果放不下，先试 `gradient_checkpointing_backend: torch-ffn`。
3. 如果还是太紧，再试 `torch`。
4. 如果 torch layer checkpointing 仍然放不下，再试 `unsloth-ffn`，然后试 `unsloth`。
5. 如果模型支持 `gradient_checkpointing_interval`，等 run 已经能放下后再用 `2` 或更高来找回速度。

它的价值在于让你跑上想要的 batch、分辨率、帧数或 rank。小 token 数、或者峰值主要来自可训练权重、gradients、optimizer、VAE cache、validation 时，它价值不大。

## 备注

- 启用 FSDP activation checkpointing 时，SimpleTuner 会关闭 model-level gradient checkpointing，避免两个系统冲突。
- `torch-ffn` 和 `unsloth-ffn` 需要模型支持。SimpleTuner 会直接报错，而不是静默运行另一个 scope。
- `gradient_checkpointing_interval: 1` 等同于普通 every-block checkpointing。
- 有些模型族不支持 interval checkpointing。SimpleTuner 会警告并忽略该 interval。
- 我们的 sweep 里，`torch.compile` 没有拯救 offload 路径。
