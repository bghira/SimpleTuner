# REPA 与 U-REPA（图像正则化）

表示对齐（REPA）是一种正则化技术，它将扩散模型的隐藏状态与冻结的视觉编码器特征（通常是 DINOv2）对齐。通过利用预训练的视觉表示，可以提高生成质量和训练效率。

SimpleTuner 支持两种变体：

- **REPA** 用于基于 DiT 的图像模型（Flux、SD3、Chroma、Sana、PixArt 等）- PR #2562
- **U-REPA** 用于基于 UNet 的图像模型（SDXL、SD1.5、Kolors）- PR #2563

> **寻找视频模型？** 请参阅 [VIDEO_CREPA.zh.md](VIDEO_CREPA.zh.md) 了解带时间对齐的视频模型 CREPA 支持。

## 何时使用

### REPA（DiT 模型）
- 您正在训练基于 DiT 的图像模型，希望更快收敛
- 您注意到质量问题或希望获得更强的语义基础
- 支持的模型系列：`flux`、`flux2`、`sd3`、`chroma`、`sana`、`pixart`、`hidream`、`auraflow`、`lumina2` 等

### U-REPA（UNet 模型）
- 您正在训练基于 UNet 的图像模型（SDXL、SD1.5、Kolors）
- 您希望利用针对 UNet 架构优化的表示对齐
- U-REPA 使用**中间块**对齐（而非早期层），并添加**流形损失**以获得更好的相对相似性结构

## 快速设置（WebUI）

### DiT 模型（REPA）

1. 打开**训练 -> 损失函数**。
2. 启用 **CREPA**（相同选项为图像模型启用 REPA）。
3. 将 **CREPA Block Index** 设置为早期编码器层：
   - Flux / Flux2：`8`
   - SD3：`8`
   - Chroma：`8`
   - Sana / PixArt：`10`
4. 将**权重**设置为 `0.5` 作为起点。
5. 保持视觉编码器默认值（`dinov2_vitg14`，分辨率 `518`）。

### UNet 模型（U-REPA）

1. 打开**训练 -> 损失函数**。
2. 启用 **U-REPA**。
3. 将 **U-REPA Weight** 设置为 `0.5`（论文默认值）。
4. 将 **U-REPA Manifold Weight** 设置为 `3.0`（论文默认值）。
5. 保持视觉编码器默认值。

## 快速设置（配置 JSON / CLI）

### DiT 模型（REPA）

```json
{
  "crepa_enabled": true,
  "crepa_block_index": 8,
  "crepa_lambda": 0.5,
  "crepa_encoder": "dinov2_vitg14",
  "crepa_encoder_image_size": 518
}
```

### UNet 模型（U-REPA）

```json
{
  "urepa_enabled": true,
  "urepa_lambda": 0.5,
  "urepa_manifold_weight": 3.0,
  "urepa_model": "dinov2_vitg14",
  "urepa_encoder_image_size": 518
}
```

## 关键区别：REPA vs U-REPA

| 方面 | REPA（DiT） | U-REPA（UNet） |
|------|-----------|---------------|
| 架构 | Transformer 块 | 带中间块的 UNet |
| 对齐点 | 早期 transformer 层 | 中间块（瓶颈） |
| 隐藏状态形状 | `(B, S, D)` 序列 | `(B, C, H, W)` 卷积 |
| 损失组件 | 余弦对齐 | 余弦 + 流形损失 |
| 默认权重 | 0.5 | 0.5 |
| 配置前缀 | `crepa_*` | `urepa_*` |

## U-REPA 细节

U-REPA 通过两个关键创新为 UNet 架构适配 REPA：

### 中间块对齐
与使用早期 transformer 层的基于 DiT 的 REPA 不同，U-REPA 从 UNet 的**中间块**（瓶颈）提取特征。这是 UNet 压缩最多语义信息的位置。

- **SDXL/Kolors**：对于 1024x1024 图像，中间块输出 `(B, 1280, 16, 16)`
- **SD1.5**：对于 512x512 图像，中间块输出 `(B, 1280, 8, 8)`

### 流形损失
除了余弦对齐外，U-REPA 还添加了**流形损失**来对齐相对相似性结构：

```
L_manifold = ||sim(y[i],y[j]) - sim(h[i],h[j])||^2_F
```

这确保如果两个编码器补丁相似，相应的投影补丁也应该相似。`urepa_manifold_weight` 参数（默认 3.0）控制直接对齐和流形对齐之间的平衡。

## 调优参数

### REPA（DiT 模型）
- `crepa_lambda`：对齐损失权重（默认 0.5）
- `crepa_block_index`：要提取的 transformer 块（从 0 开始索引）
- `crepa_spatial_align`：插值 token 以匹配（默认 true）
- `crepa_encoder`：视觉编码器模型（默认 `dinov2_vitg14`）
- `crepa_encoder_image_size`：输入分辨率（默认 518）

### U-REPA（UNet 模型）
- `urepa_lambda`：对齐损失权重（默认 0.5）
- `urepa_manifold_weight`：流形损失权重（默认 3.0）
- `urepa_model`：视觉编码器模型（默认 `dinov2_vitg14`）
- `urepa_encoder_image_size`：输入分辨率（默认 518）
- `urepa_use_tae`：使用 Tiny AutoEncoder 加速解码

## 系数调度

REPA 和 U-REPA 都支持调度以在训练过程中衰减正则化：

```json
{
  "crepa_scheduler": "cosine",
  "crepa_warmup_steps": 100,
  "crepa_decay_steps": 5000,
  "crepa_lambda_end": 0.0
}
```

对于 U-REPA，使用 `urepa_` 前缀：

```json
{
  "urepa_scheduler": "cosine",
  "urepa_warmup_steps": 100,
  "urepa_cutoff_step": 5000
}
```

<details>
<summary>工作原理（实践者）</summary>

### REPA（DiT）
- 从选定的 transformer 块捕获隐藏状态
- 通过 LayerNorm + Linear 投影到编码器维度
- 计算与冻结 DINOv2 特征的余弦相似度
- 如果数量不同，插值空间 token 以匹配

### U-REPA（UNet）
- 在 UNet mid_block 上注册前向钩子
- 捕获卷积特征 `(B, C, H, W)`
- 重塑为序列 `(B, H*W, C)` 并投影到编码器维度
- 计算余弦对齐和流形损失
- 流形损失对齐成对相似性结构

</details>

<details>
<summary>技术细节（SimpleTuner 内部）</summary>

### REPA
- 实现：`simpletuner/helpers/training/crepa.py`（`CrepaRegularizer` 类）
- 模式检测：图像模型使用 `CrepaMode.IMAGE`，通过 `crepa_mode` 属性自动设置
- 隐藏状态存储在模型输出的 `crepa_hidden_states` 键中

### U-REPA
- 实现：`simpletuner/helpers/training/crepa.py`（`UrepaRegularizer` 类）
- 中间块捕获：`simpletuner/helpers/utils/hidden_state_buffer.py`（`UNetMidBlockCapture`）
- 隐藏大小从 `block_out_channels[-1]` 推断（SDXL/SD1.5/Kolors 为 1280）
- 仅对 `MODEL_TYPE == ModelTypes.UNET` 启用
- 隐藏状态存储在模型输出的 `urepa_hidden_states` 键中

</details>

## 常见问题

- **模型类型错误**：REPA（`crepa_*`）用于 DiT 模型；U-REPA（`urepa_*`）用于 UNet 模型。使用错误的类型将不会产生效果。
- **块索引过高**（REPA）：如果出现"hidden states not returned"错误，请降低索引。
- **显存峰值**：尝试更小的编码器（`dinov2_vits14` + 图像大小 `224`）或启用 `use_tae` 进行解码。
- **流形权重过高**（U-REPA）：如果训练变得不稳定，将 `urepa_manifold_weight` 从 3.0 降低到 1.0。

## 参考文献

- [REPA 论文](https://arxiv.org/abs/2402.17750) - 生成的表示对齐
- [U-REPA 论文](https://arxiv.org/abs/2410.xxxxx) - UNet 架构的通用 REPA（NeurIPS 2025）
- [DINOv2](https://github.com/facebookresearch/dinov2) - 自监督视觉编码器
