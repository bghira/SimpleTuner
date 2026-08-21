# iREPA

iREPA 通过保留对齐路径中的空间结构来改进表示对齐。它用空间卷积替代逐 token 线性投影，并在每张图像的 patch 维度上对教师特征逐通道进行 z-score 归一化。

SimpleTuner 会根据骨干类型使用现有对齐引擎：Transformer 图像模型使用 REPA/CREPA；Transformer 视频模型逐帧应用 iREPA，并保留 CREPA 的时间邻帧损失；UNet 图像模型使用 U-REPA 的 mid-block 与 manifold loss。矩形 token 网格从干净 latent 的形状推导，不要求方形 bucket。

```json
{
  "irepa_enabled": true,
  "irepa_spatial_norm_alpha": 0.6,
  "irepa_projector_kernel_size": 3,
  "crepa_enabled": true,
  "crepa_block_index": 8,
  "crepa_lambda": 1.0
}
```

Transformer 需同时启用 iREPA 和 `crepa_enabled`，UNet 需同时启用 iREPA 和 `urepa_enabled`。对应的 `crepa_*` 或 `urepa_*` 设置控制教师、权重、捕获层和调度。`0.6` 对应 latent-diffusion 参考配置；`3` 是论文中的卷积核大小。

iREPA 需要带空间 patch token 的隐藏状态和用于恢复网格的干净 latent。视频卷积不会跨帧混合；时间对齐仍由 CREPA 参数控制。

请使用完整模型训练或标准 PEFT LoRA。LyCORIS 无法保存辅助投影器，因此不受支持。

参考：[What Matters for Representation Alignment: Global Information or Spatial Structure?](https://arxiv.org/abs/2512.10794)
