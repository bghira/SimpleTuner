# Self-Flow（内部对齐）

Self-Flow 是一种 CREPA 模式，用同一模型更干净的 EMA 教师视图替代外部视觉编码器。它较为贴近 Black Forest Labs 论文中的做法：学生分支使用混合的 token 级噪声调度，EMA 教师看到更干净的视图，同时保持正常生成 loss 并对齐内部 hidden states。

与附近的 SimpleTuner 方法相比：

| 方法 | 教师来源 | 噪声非对称性 | 额外教师模型 | 核心思路 |
| --- | --- | --- | --- | --- |
| REPA / VIDEO_CREPA | 冻结的外部编码器，通常是 DINOv2 | 否 | 是 | 将模型 hidden states 对齐到外部语义特征 |
| LayerSync | 同一次 forward pass 中更深的层 | 否 | 否 | 将较早层对齐到更强的后续层 |
| TwinFlow | EMA 教师与递归轨迹目标 | 没有 token 级 cleaner / noisier 切分 | 无外部模型 | 少步数轨迹匹配，可选负时间符号语义 |
| Self-Flow | 同一模型在更干净视图上的 EMA 教师 | 是 | 无外部模型 | 通过 dual-timestep scheduling 学习更强的内部表示 |

> **如果你想要外部编码器对齐**：REPA / U-REPA 请看 [IMAGE_REPA.zh.md](IMAGE_REPA.zh.md)，视频时间对齐 CREPA 请看 [VIDEO_CREPA.zh.md](VIDEO_CREPA.zh.md)。

## 何时使用

- 你想用 BFL 风格的自监督正则，而不是外部编码器。
- 你正在训练已经在 SimpleTuner 中实现 Self-Flow hook 的 transformer 家族。
- 你希望同一个正则同时帮助普通生成、编辑和多模态训练。
- 你已经启用 EMA，或可以启用。Self-Flow 必须使用 EMA 教师。

当前支持的家族包括：

- 图像 / 编辑：`flux`, `flux2`, `sd3`, `pixart`, `sana`, `qwen_image`, `chroma`, `hidream`, `auraflow`, `lumina2`, `z_image`, `z_image_omni`, `kandinsky5_image`, `longcat_image`, `omnigen`, `ace_step`
- 视频 / 多模态：`wan`, `wan_s2v`, `ltxvideo`, `ltxvideo2`, `sanavideo`, `kandinsky5_video`, `hunyuanvideo`, `longcat_video`, `cosmos`, `anima`

## 快速设置（WebUI）

1. 打开 **Training → Loss functions**。
2. 启用 **CREPA**。
3. 将 **CREPA Feature Source** 设为 `self_flow`。
4. 将 **CREPA Block Index** 设为较早的学生块。24 层 DiT 可从 `8` 开始，更深的栈可从 `10` 开始。
5. 将 **CREPA Teacher Block Index** 设为更深的教师块。`16` 或 `20` 是不错的起点。
6. **Weight** 从 `0.5` 开始。
7. **Self-Flow Mask Ratio** 建议：
   - 图像：`0.25`
   - 视频：`0.10`
   - 音频较重的模型如 `ace_step`：`0.50`
8. 确保启用 **EMA**。
9. 不要与 TwinFlow 同时使用。

## 快速设置（config JSON / CLI）

```json
{
  "use_ema": true,
  "crepa_enabled": true,
  "crepa_feature_source": "self_flow",
  "crepa_block_index": 8,
  "crepa_teacher_block_index": 16,
  "crepa_lambda": 0.5,
  "crepa_self_flow_mask_ratio": 0.25
}
```

旧别名 `crepa_self_flow=true` 仍然可用，但新配置更推荐 `crepa_feature_source=self_flow`。

## 关键调参项

- `crepa_block_index`：学生块
- `crepa_teacher_block_index`：EMA 教师块，必填
- `crepa_lambda`：对齐强度，建议从 `0.5` 开始
- `crepa_self_flow_mask_ratio`：使用替代 timestep 的 token 比例，范围必须在 `[0.0, 0.5]`
- `crepa_scheduler`, `crepa_warmup_steps`, `crepa_decay_steps`, `crepa_lambda_end`, `crepa_cutoff_step`：与 CREPA 相同的系数调度控制
- `crepa_use_backbone_features`：这是另一种模式，不要和 Self-Flow 混用

## 采样 / 验证

Self-Flow 改变的是训练，而不是基础推理算法。

- 训练时学生分支使用混合 token 噪声，教师分支使用更干净的 EMA 视图。
- 验证 loss 仍按请求的均匀 timestep 调度来评估。
- 正常采样方式不变。推理时不会使用 dual-timestep masking。

<details>
<summary>工作方式（实用角度）</summary>

- 采样两个 timestep，并用随机 mask 将它们分配给不同 token。
- 构建一个混合污染的学生视图，以及一个更干净 timestep 的教师视图。
- 学生正常前向，EMA 教师在 `no_grad` 下前向。
- 使用余弦相似度将较早的学生层对齐到更深的教师层，同时继续训练正常的生成 loss。

</details>

<details>
<summary>技术说明（SimpleTuner 内部）</summary>

- 模式选择位于 `simpletuner/helpers/training/crepa.py` 中的 `CrepaFeatureSource.SELF_FLOW`
- 共享 batch builder 位于 `_prepare_image_crepa_self_flow_batch` 和 `_prepare_video_crepa_self_flow_batch`
- EMA 教师前向由 `auxiliary_loss` 通过 `_run_crepa_teacher_forward` 触发
- 当请求 `custom_timesteps` 时，验证会重建均匀评估 batch，避免训练用的混合 Self-Flow batch 污染 eval loss

</details>

## 常见问题

- **未启用 EMA**：Self-Flow 需要 `use_ema=true`
- **未设置教师块**：请设置 `crepa_teacher_block_index`
- **启用了 TwinFlow**：两者不兼容
- **模型家族不支持**：只有实现了 `supports_crepa_self_flow()` 的家族才能使用
- **mask ratio 太高**：请保持在 `0.5` 或以下
- **以为需要特殊 sampler**：推理仍然使用普通采样

## 参考

- [Self-Supervised Flow Matching for Scalable Multi-Modal Synthesis](https://bfl.ai/research/self-flow)
