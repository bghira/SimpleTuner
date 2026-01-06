# TREAD 训练文档

> ⚠️ **实验性功能**：SimpleTuner 的 TREAD 支持是新实现的。虽然已经可用，但最佳配置仍在探索中，部分行为可能在未来版本中变化。

## 概述

TREAD（Token Routing for Efficient Architecture-agnostic Diffusion Training）是一种训练加速方法，通过在 Transformer 层中智能路由 token 来提升扩散模型训练速度。在某些层仅处理最重要的 token，可显著降低计算开销，同时保持模型质量。

基于 [Krause 等人（2025）](https://arxiv.org/abs/2501.04765) 的研究，TREAD 通过以下方式实现加速：
- 在每个 Transformer 层动态选择需要处理的 token
- 通过 skip 连接维持所有 token 的梯度流
- 基于重要度做路由决策

速度提升与 `selection_ratio` 成正比——越接近 1.0，丢弃的 token 越多，训练越快。

## TREAD 的工作方式

### 核心概念

训练时，TREAD：
1. **路由 token** - 在指定 Transformer 层中，根据重要度选择一部分 token 进行处理
2. **处理子集** - 仅选中的 token 经过高开销的 attention 与 MLP 操作
3. **还原完整序列** - 处理后恢复完整 token 序列，所有 token 都能获得梯度

### Token 选择

token 根据 L1 范数（重要度分数）选择，并可加入随机化以增加探索：
- 重要度更高的 token 更可能被保留
- 重要度选择与随机选择混合，避免过拟合某些模式
- 强制保留 mask 可确保特定 token（如遮罩区域）不会被丢弃

## 配置

### 基础设置

要在 SimpleTuner 中启用 TREAD 训练，请在配置中添加：

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 2,
        "end_layer_idx": 5
      }
    ]
  }
}
```

### 路由配置

每条 route 定义 token 路由生效的窗口：
- `selection_ratio`：丢弃 token 的比例（0.5 = 保留 50% 的 token）
- `start_layer_idx`：开始路由的层（0 起始）
- `end_layer_idx`：路由生效的最后一层

支持负索引：`-1` 表示最后一层。

### 高级示例

使用不同 selection ratio 的多个路由窗口：

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.3,
        "start_layer_idx": 1,
        "end_layer_idx": 3
      },
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 4,
        "end_layer_idx": 8
      },
      {
        "selection_ratio": 0.7,
        "start_layer_idx": -4,
        "end_layer_idx": -1
      }
    ]
  }
}
```

## 兼容性

### 支持的模型
- **FLUX Dev/Kontext、Wan、AuraFlow、PixArt、SD3** - 目前仅支持这些模型家族
- 未来计划支持其他扩散 Transformer

### 适配良好的功能
- **遮罩损失训练** - 与 mask/segmentation 条件结合时，TREAD 会自动保留遮罩区域
- **多 GPU 训练** - 与分布式训练兼容
- **量化训练** - 可与 int8/int4/NF4 量化一起使用

### 限制
- 仅在训练阶段生效（推理时无效）
- 需要梯度计算（eval 模式不可用）
- 目前实现仅针对 FLUX 和 Wan，暂不适用于 Lumina2 等其他架构

## 性能考量

### 速度收益
- 训练加速与 `selection_ratio` 成正比（越接近 1.0，丢弃 token 越多、训练越快）
- 由于 attention 的 O(n²) 复杂度，**长视频输入与高分辨率场景的收益最大**
- 通常可获得 20–40% 的加速，但结果因配置而异
- 使用遮罩损失训练时，遮罩 token 不能丢弃，速度提升会下降

### 质量权衡
- **丢弃更多 token 会导致 LoRA/LoKr 训练起始损失更高**
- 损失往往会较快纠正，除非使用过高的选择率，图像会很快恢复正常
  - 这可能是网络在适应中间层 token 减少
- 保守比率（0.1–0.25）通常能保持质量
- 激进比率（>0.35）会明显影响收敛

### LoRA 特定注意事项
- 性能可能依赖数据，最佳路由配置仍需探索
- 与全量微调相比，LoRA/LoKr 的初始损失峰值更明显

### 推荐设置

速度/质量平衡：
```json
{
  "routes": [
    {"selection_ratio": 0.5, "start_layer_idx": 2, "end_layer_idx": -2}
  ]
}
```

最大速度（预计明显损失峰值）：
```json
{
  "routes": [
    {"selection_ratio": 0.7, "start_layer_idx": 1, "end_layer_idx": -1}
  ]
}
```

高分辨率训练（1024px+）：
```json
{
  "routes": [
    {"selection_ratio": 0.6, "start_layer_idx": 2, "end_layer_idx": -3}
  ]
}
```

## 技术细节

### 路由器实现

TREAD 路由器（`TREADRouter` 类）负责：
- 通过 L1 范数计算 token 重要度
- 生成用于高效路由的排列
- 保留梯度的 token 复原

### 与 Attention 的集成

TREAD 会调整旋转位置嵌入（RoPE）以匹配路由后的序列：
- 文本 token 保持原位置
- 图像 token 使用打乱/切片后的位置信息
- 确保路由过程中的位置一致性
- **注记**：FLUX 的 RoPE 路由实现可能不完全正确，但在实践中似乎可用

### 遮罩损失兼容性

使用遮罩损失训练时：
- 遮罩内 token 会自动强制保留
- 防止重要训练信号被丢弃
- 通过 `conditioning_type` 为 ["mask", "segmentation"] 激活
- **注记**：由于需要保留 token，速度提升会降低

## 已知问题与限制

### 实现状态
- **实验性功能** - 新实现，可能存在未知问题
- **RoPE 处理** - token 路由的 RoPE 实现可能不完全准确
- **测试有限** - 最优路由配置尚未充分探索

### 训练行为
- **初始损失峰值** - 在 LoRA/LoKr 训练中使用 TREAD 时，初始损失偏高但会很快纠正
- **LoRA 性能** - 部分配置可能导致 LoRA 训练略有变慢
- **配置敏感** - 性能高度依赖路由配置选择

### 已知 Bug（已修复）
- 遮罩损失训练在早期版本中损坏，但已通过正确的模型风味检查（`kontext` guard）修复

## 排障

### 常见问题

**“TREAD training requires you to configure the routes”**
- 确保 `tread_config` 包含 `routes` 数组
- 每条 route 需要 `selection_ratio`、`start_layer_idx`、`end_layer_idx`

**训练比预期慢**
- 检查路由是否覆盖有意义的层范围
- 考虑更激进的 selection ratio
- 检查梯度检查点是否冲突
- LoRA 训练可能会变慢，尝试不同路由配置

**LoRA/LoKr 初始损失过高**
- 这是预期行为，网络需要适应 token 数减少
- 通常几百步内损失会改善
- 若损失无改善，降低 `selection_ratio`（保留更多 token）

**质量下降**
- 降低 selection ratio（保留更多 token）
- 避免在早期层（0–2）或末层进行路由
- 确保有足够的训练数据以匹配更高的效率

## 实用示例

### 高分辨率训练（1024px+）
高分辨率场景的最大收益配置：
```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.6, "start_layer_idx": 2, "end_layer_idx": -3}
    ]
  }
}
```

### LoRA 微调
保守配置以降低初始损失峰值：
```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.4, "start_layer_idx": 3, "end_layer_idx": -4}
    ]
  }
}
```

### 遮罩损失训练
使用遮罩训练时，遮罩区域 token 会被保留：
```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.7, "start_layer_idx": 2, "end_layer_idx": -2}
    ]
  }
}
```
注记：由于强制保留 token，实际加速会低于 0.7 所暗示的水平。

## 后续工作

由于 SimpleTuner 中的 TREAD 支持是新实现，仍有多个方向可改进：

- **配置优化** - 需要更多测试以找出不同场景的最优路由配置
- **LoRA 性能** - 研究部分 LoRA 配置变慢的原因
- **RoPE 实现** - 改进旋转位置嵌入的正确性
- **扩展模型支持** - 为 Flux 之外的扩散 Transformer 架构实现支持
- **自动化配置** - 基于模型与数据集特性自动生成最优路由配置的工具

欢迎社区贡献与测试结果，以帮助完善 TREAD 支持。

## 参考

- [TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training](https://arxiv.org/abs/2501.04765)
- [SimpleTuner Flux Documentation](quickstart/FLUX.md#tread-training)

## 引用

```bibtex
@misc{krause2025treadtokenroutingefficient,
      title={TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training},
      author={Felix Krause and Timy Phan and Vincent Tao Hu and Björn Ommer},
      year={2025},
      eprint={2501.04765},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.04765},
}
```
