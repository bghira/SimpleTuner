# ConvRot / Hadamard SDNQ

SimpleTuner 通过 SDNQ 的 Hadamard 路径提供 ConvRot 风格的旋转量化。它适合大型 PEFT 训练：冻结的基础模型使用 int8 计算，LoRA 或 LyCORIS 适配器继续以 bf16 等混合精度 dtype 训练。

这不会直接读取外部 ConvRot checkpoint 中的自定义 buffer。请加载原始模型权重，然后让 SimpleTuner 在模型加载后用 SDNQ 量化训练组件。

## 快速配置

```json
{
  "base_model_precision": "int8-sdnq",
  "gradient_checkpointing": true,
  "sdnq_use_hadamard": true,
  "sdnq_hadamard_group_size": 128,
  "sdnq_group_size": -1,
  "sdnq_use_quantized_matmul": true,
  "sdnq_compile_mode": "compile"
}
```

对于大型模型，除非模型指南另有说明，建议保留 `quantize_via` 为 `cpu`。CPU 量化可以降低初始化阶段的显存峰值。

## 选项说明

- `base_model_precision: int8-sdnq` 为训练的基础组件选择 SDNQ int8 后加载量化。
- `sdnq_use_hadamard: true` 启用 Hadamard 旋转路径。
- `sdnq_hadamard_group_size: 128` 设置 SDNQ 使用的旋转块大小。
- `sdnq_group_size: -1` 使用静态的按行权重 scale，避免主要面向全量微调的动态分组路径在训练中重新量化权重。
- `sdnq_use_quantized_matmul: true` 保持 SDNQ int8 matmul 路径启用。
- `sdnq_compile_mode: compile` 在 SDNQ 支持的位置编译量化 helper 和 kernel。
- `gradient_checkpointing: true` 让 SDNQ 在 PEFT 训练中使用开销更低的训练路径。SimpleTuner 会把它作为 `use_grad_ckpt=True` 传给 SDNQ；启用梯度 checkpointing 时，如果把该 SDNQ 标志设为 false，只会额外保存会被 checkpointing 立即丢弃的量化 backward 输入。

## PEFT 行为

基础 transformer 由 SDNQ 量化。适配器权重仍然可训练，并使用普通混合精度 dtype，通常是 bf16。

部分模型会在训练前加载固定辅助适配器。例如 Z-Image Turbo 有 assistant LoRA。SimpleTuner 会把该 assistant 适配器延后到 SDNQ 量化之后加载，这样 SDNQ 看到的是原始 transformer module，而不是 PEFT wrapper 的代理权重。

## 要求与限制

- 使用带 Hadamard 支持的 SDNQ 构建。H100 验证使用的是上游 SDNQ `0.2.3`；PyPI `0.2.2` 不包含相同的 bf16 Hadamard 修复。
- 该预设面向大型模型的 LoRA 和 LyCORIS 训练。SDNQ Hadamard 的全量微调需要单独验证。
- 初始 step 可能较慢，因为 SDNQ 和 Torch 会在初始化和早期训练时编译 kernel。
- 验证和推理使用量化后的基础模型加当前适配器，和训练一致。

## 示例模型

SimpleTuner 为 Z-Image Turbo、Krea 2、FLUX.2、Cosmos 3 和 LTXVideo 2.3 提供 SDNQ Hadamard 示例。这些示例使用 `sdnq_group_size: -1`，因为它比动态分组训练默认值更符合 PEFT 工作负载。
