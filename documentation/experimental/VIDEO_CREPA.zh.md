# CREPA（视频正则化）

Cross-frame Representation Alignment（CREPA）是视频模型的轻量正则项。它将每帧隐藏状态对齐到冻结视觉编码器在当前帧**及其邻帧**上的特征，从而在不改变主损失的情况下提升时间一致性。

## 适用场景

- 训练包含复杂运动、场景变化或遮挡的视频。
- 微调视频 DiT（LoRA 或全量）时出现帧间闪烁/身份漂移。
- 支持的模型家族：`kandinsky5_video`、`ltxvideo`、`sanavideo`、`wan`（其他家族未暴露 CREPA 钩子）。
- 有额外 VRAM（根据设置约 1–2GB）用于 DINO 编码器与 VAE，并需在训练时保持内存驻留以将 latent 解码为像素。

## 快速设置（WebUI）

1. 打开 **Training → Loss functions**。
2. 启用 **CREPA**。
3. 将 **CREPA Block Index** 设为编码侧层。建议起点：
   - Kandinsky5 Video: `8`
   - LTXVideo / Wan: `8`
   - SanaVideo: `10`
4. **Weight** 先保持 `0.5`。
5. **Adjacent Distance** 设为 `1`，**Temporal Decay** 设为 `1.0`，以接近原始 CREPA 论文设置。
6. 视觉编码器使用默认值（`dinov2_vitg14`，分辨率 `518`）。仅在需要更小编码器时才调整（如 `dinov2_vits14` + 图像尺寸 `224` 以省显存）。
7. 正常训练。CREPA 会加入辅助损失并记录 `crepa_loss` / `crepa_similarity`。

## 快速设置（config JSON / CLI）

在 `config.json` 或 CLI 参数中添加：

```json
{
  "crepa_enabled": true,
  "crepa_block_index": 8,
  "crepa_lambda": 0.5,
  "crepa_adjacent_distance": 1,
  "crepa_adjacent_tau": 1.0,
  "crepa_encoder": "dinov2_vitg14",
  "crepa_encoder_image_size": 518
}
```

## 调参要点

- `crepa_spatial_align`：保留 patch 级结构（默认）。显存紧张时可设为 `false` 以进行池化。
- `crepa_normalize_by_frames`：保持不同剪辑长度下损失尺度稳定（默认）。若希望更长视频贡献更大，可关闭。
- `crepa_drop_vae_encoder`：如果仅**解码** latent 可释放内存（若需要编码像素则不安全）。
- `crepa_adjacent_distance=0`：类似每帧 REPA*（无邻帧帮助）；可结合 `crepa_adjacent_tau` 做距离衰减。
- `crepa_cumulative_neighbors=true`（仅配置项）：使用 `1..d` 的所有偏移，而非只用最近邻。
- `crepa_use_backbone_features=true`：跳过外部编码器，改为对齐更深的 Transformer 块；通过 `crepa_teacher_block_index` 指定教师。
- 编码器大小：显存紧张可用 `dinov2_vits14` + `224`；追求质量建议 `dinov2_vitg14` + `518`。

## 系数调度

CREPA 支持在训练过程中对系数（`crepa_lambda`）进行调度，包括预热、衰减以及基于相似度阈值的自动截止。这对于 text2video 训练尤其有用，因为如果 CREPA 应用过强或过久，可能会导致水平/垂直条纹或画面发灰。

### 基本调度

```json
{
  "crepa_enabled": true,
  "crepa_lambda": 0.5,
  "crepa_scheduler": "cosine",
  "crepa_warmup_steps": 100,
  "crepa_decay_steps": 5000,
  "crepa_lambda_end": 0.0
}
```

此配置：
1. 在前 100 步将 CREPA 权重从 0 预热到 0.5
2. 使用余弦调度在 5000 步内从 0.5 衰减到 0.0
3. 第 5100 步后，CREPA 实际上已被禁用

### 调度器类型

- `constant`：无衰减，权重保持在 `crepa_lambda`（默认）
- `linear`：从 `crepa_lambda` 到 `crepa_lambda_end` 的线性插值
- `cosine`：平滑余弦退火（大多数情况推荐）
- `polynomial`：多项式衰减，可通过 `crepa_power` 配置幂次

### 基于步数的截止

若需在特定步数后硬截止：

```json
{
  "crepa_cutoff_step": 3000
}
```

第 3000 步后 CREPA 将完全禁用。

### 基于相似度的截止

这是最灵活的方式——当相似度指标趋于平稳（表明模型已学会足够的时间对齐）时，CREPA 自动禁用：

```json
{
  "crepa_similarity_threshold": 0.9,
  "crepa_similarity_ema_decay": 0.99,
  "crepa_threshold_mode": "permanent"
}
```

- `crepa_similarity_threshold`：当相似度的指数移动平均达到此值时，CREPA 截止
- `crepa_similarity_ema_decay`：平滑系数（0.99 ≈ 100 步窗口）
- `crepa_threshold_mode`：`permanent`（保持关闭）或 `recoverable`（相似度下降时可重新启用）

### 推荐配置

**image2video (i2v)**：
```json
{
  "crepa_scheduler": "constant",
  "crepa_lambda": 0.5
}
```
标准 CREPA 对 i2v 效果良好，因为参考帧可锚定一致性。

**text2video (t2v)**：
```json
{
  "crepa_scheduler": "cosine",
  "crepa_lambda": 0.5,
  "crepa_warmup_steps": 100,
  "crepa_decay_steps": 0,
  "crepa_lambda_end": 0.1,
  "crepa_similarity_threshold": 0.85,
  "crepa_threshold_mode": "permanent"
}
```
在训练过程中衰减 CREPA，并在相似度饱和时截止以防止伪影。

**纯色背景 (t2v)**：
```json
{
  "crepa_cutoff_step": 2000
}
```
早期截止可防止均匀背景上的条纹伪影。

<details>
<summary>工作原理（实践视角）</summary>

- 捕获指定 DiT 块的隐藏状态，经 LayerNorm+Linear 头投影后与冻结视觉特征对齐。
- 默认使用 DINOv2 编码像素帧；主干模式复用更深的 Transformer 块。
- 以指数衰减（`crepa_adjacent_tau`）对齐邻帧；累计模式可将 `d` 内的所有偏移求和。
- 通过空间/时间对齐重新采样 token，使 DiT patch 与编码器 patch 对齐后再计算余弦相似度；损失在 patch 与帧上取平均。

</details>

<details>
<summary>技术细节（SimpleTuner 内部）</summary>

- 实现：`simpletuner/helpers/training/crepa.py`；由 `ModelFoundation._init_crepa_regularizer` 注册并挂到可训练模型（投影头挂在模型上以进入优化器）。
- 隐藏状态捕获：当 `crepa_enabled` 为 true，视频 Transformer 会保存 `crepa_hidden_states`（必要时保存 `crepa_frame_features`）；主干模式也可从共享缓冲中取 `layer_{idx}`。
- 损失路径：除非启用 `crepa_use_backbone_features`，否则用 VAE 将 latent 解码为像素；对投影隐藏状态与编码器特征做归一化，计算距离加权余弦相似度，记录 `crepa_loss` / `crepa_similarity` 并叠加缩放损失。
- 交互：在 LayerSync 之前执行，以便共享缓冲；结束后清理缓冲。需要有效 block index 且能从 Transformer 配置推断 hidden size。

</details>

## 常见问题

- 在不支持的家族上启用 CREPA 会导致缺失隐藏状态；仅限 `kandinsky5_video`、`ltxvideo`、`sanavideo`、`wan`。
- **Block index 太高** → “hidden states not returned”。降低索引；Transformer 块为 0-based。
- **显存峰值** → 尝试 `crepa_spatial_align=false`、更小编码器（`dinov2_vits14` + `224`）或更低 block index。
- **主干模式报错** → 同时设置 `crepa_block_index`（学生）与 `crepa_teacher_block_index`（教师）为存在的层。
- **内存不足** → 若 RamTorch 无效，可能只能使用更大的 GPU；若 H200 或 B200 也不够，请提交 issue。
