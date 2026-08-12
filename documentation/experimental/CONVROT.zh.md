# ConvRot / Hadamard SDNQ

SimpleTuner 通过 SDNQ 的 Hadamard 路径提供 ConvRot 风格的旋转量化。它适合大型 PEFT 训练：冻结的基础模型使用 int8 计算，LoRA 或 LyCORIS 适配器继续以 bf16 等混合精度 dtype 训练。

SimpleTuner 不会把任意 ConvRot sidecar buffer 当作独立功能读取。常见路径是加载原始模型权重，然后让 SimpleTuner 在模型加载后用 SDNQ 量化训练组件。支持单文件量化 transformer 权重的模型 loader 也可以加载兼容的 INT8 ConvRot transformer safetensors，并通过 SDNQ Hadamard 执行。

## 快速配置

```json
{
  "base_model_precision": "int8-sdnq",
  "gradient_checkpointing": true,
  "sdnq_use_hadamard": true,
  "sdnq_hadamard_group_size": 256,
  "sdnq_group_size": -1,
  "sdnq_use_quantized_matmul": true,
  "sdnq_compile_mode": "compile"
}
```

对于大型模型，除非模型指南另有说明，建议保留 `quantize_via` 为 `cpu`。CPU 量化可以降低初始化阶段的显存峰值。

## 选项说明

- `base_model_precision: int8-sdnq` 为训练的基础组件选择 SDNQ int8 后加载量化。
- `sdnq_use_hadamard: true` 启用 Hadamard 旋转路径。
- `sdnq_hadamard_group_size: 256` 设置 SDNQ 使用的旋转块大小。ConvRot 使用 `256`；更小的块会选择类似 QuaRot 的路径。
- `sdnq_group_size: -1` 使用静态的按行权重 scale，避免主要面向全量微调的动态分组路径在训练中重新量化权重。
- `sdnq_use_quantized_matmul: true` 保持 SDNQ int8 matmul 路径启用。
- `sdnq_compile_mode: compile` 在 SDNQ 支持的位置编译量化 helper 和 kernel。
- `gradient_checkpointing: true` 让 SDNQ 在 PEFT 训练中使用开销更低的训练路径。SimpleTuner 会把它作为 `use_grad_ckpt=True` 传给 SDNQ；启用梯度 checkpointing 时，如果把该 SDNQ 标志设为 false，只会额外保存会被 checkpointing 立即丢弃的量化 backward 输入。

## PEFT 行为

基础 transformer 由 SDNQ 量化。适配器权重仍然可训练，并使用普通混合精度 dtype，通常是 bf16。

部分模型会在训练前加载固定辅助适配器。例如 Z-Image Turbo 有 assistant LoRA。SimpleTuner 会把该 assistant 适配器延后到 SDNQ 量化之后加载，这样 SDNQ 看到的是原始 transformer module，而不是 PEFT wrapper 的代理权重。

## 要求与限制

- SimpleTuner 会为支持的安装目标安装并配置 SDNQ 训练依赖。
- 该预设面向大型模型的 LoRA 和 LyCORIS 训练。SDNQ Hadamard 的全量微调需要单独验证。
- 初始 step 可能较慢，因为 SDNQ 和 Torch 会在初始化和早期训练时编译 kernel。
- 验证和推理使用量化后的基础模型加当前适配器，和训练一致。
- ConvRot 可以降低量化损伤，但不保证 INT8 在每个模型上都能匹配 BF16 或 FP8。开始长训练前，请同时验证 loss curve 和生成样本。
- 使用 SDNQ ConvRot 做 standalone inference 不属于本训练指南范围。若要直接使用 SDNQ inference API，请参考 [SDNQ 上游文档](https://github.com/Disty0/sdnq)，因为该 API 比 SimpleTuner 训练配置变化更频繁。

## 实测结果

这些是按模型测得的 SimpleTuner 真实 trainer 数据，不是只测 GEMM 的合成结果。`Loop s/step` 是 wrapper 记录的训练循环 wall time 每步。`Mean step` 排除了前五个 warmup step。

| 模型 | GPU | 步数 | 权重路径 | Loop s/step | Mean step | p50 | p95 | 峰值已分配 VRAM |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| Z-Image Turbo LoRA | H100 80GB | 1000 | SDNQ Hadamard post-load quantization | 1.107 | 1.087 | 1.071 | 1.109 | 9.70 GiB |
| Z-Image Turbo LoRA | L40S | 1000 | SDNQ Hadamard post-load quantization | 1.026 | 1.018 | 1.002 | 1.040 | 9.66 GiB |
| Z-Image Turbo LoRA | L40S | 1000 | baseline SDNQ Hadamard 路径 | 1.131 | 1.072 | 1.055 | 1.102 | 9.66 GiB |
| Krea 2 Raw LoRA | H100 80GB | 100 | `lilcheaty/Krea2-INT8-ConvRot` transformer 权重，diffusers attention | 0.787 | 0.399 | 0.397 | 0.411 | 32.15 GiB |
| Krea 2 Raw LoRA | L40S | 100 | `lilcheaty/Krea2-INT8-ConvRot` transformer 权重，cuDNN attention | 0.945 | 0.794 | 0.793 | 0.799 | 31.89 GiB |
| Mage-Flow LoRA，square crop | H100 80GB | 100 | SDNQ INT8 vanilla post-load quantization | 1.113 | 0.277 | 0.276 | 0.286 | 20.12 GiB |
| Mage-Flow LoRA，square crop | H100 80GB | 100 | SDNQ ConvRot 256 post-load quantization | 0.436 | 0.299 | 0.297 | 0.308 | 20.15 GiB |

在 warm cache 的 L40S Z-Image 对比中，当前路径按 train-loop wall time 比 baseline SDNQ Hadamard 路径快 10.3%，按实测 train-step 平均快 5.2%。Krea 2 各行验证了 Hugging Face INT8 ConvRot transformer 权重路径在真实 100 步训练运行中的可用性。Mage-Flow 各行说明为什么必须按模型验证：square crop 去掉了大部分 shape compile churn，ConvRot 相比 vanilla INT8 降低了总 train-loop 时间，但 warm 后的实测单步略慢于 vanilla INT8。

## 示例模型

SimpleTuner 为 Z-Image Turbo、Krea 2、FLUX.2、Cosmos 3 和 LTXVideo 2.3 提供 SDNQ Hadamard 示例。这些示例使用 `sdnq_group_size: -1`，因为它比动态分组训练默认值更符合 PEFT 工作负载。
