# Slider LoRA 目标

本指南将介绍在 SimpleTuner 中训练滑块式适配器。我们使用 Z-Image Turbo，因为它训练速度快，采用 Apache 2.0 许可，并且以其体积提供很好的效果——即使是蒸馏权重也是如此。

完整兼容矩阵（LoRA、LyCORIS、全秩）请查看 [documentation/QUICKSTART.md](QUICKSTART.md) 中的 Sliders 列；本指南适用于所有架构。

Slider 目标适用于标准 LoRA、LyCORIS（包含 `full`）以及 ControlNet。开关在 CLI 和 WebUI 中都可用；SimpleTuner 已内置，无需额外安装。

## 步骤 1 — 先完成基础设置

- **CLI**: 按 `documentation/quickstart/ZIMAGE.md` 完成环境、安装、硬件注意事项以及 starter `config.json`。
- **WebUI**: 使用 `documentation/webui/TUTORIAL.md` 运行训练向导，正常选择 Z-Image Turbo。

上述指南可以一直跟到需要配置数据集的节点，因为滑块只改变适配器的放置位置与数据采样方式。

## 步骤 2 — 启用 slider 目标

- CLI: 添加 `"slider_lora_target": true`（或传入 `--slider_lora_target true`）。
- WebUI: Model → LoRA Config → Advanced → 勾选 “Use slider LoRA targets”。

LyCORIS 请保持 `lora_type: "lycoris"`，并在 `lycoris_config.json` 中使用下方详情部分的预设。

## 步骤 3 — 构建滑块友好的数据集

概念滑块会从“相反”对比数据集中学习。创建小规模的 before/after 对（4–6 对即可起步，有更多更好）：

- **Positive 桶**： “更多概念”（例如眼睛更明亮、笑容更强、沙子更多）。设置 `"slider_strength": 0.5`（任意正值）。
- **Negative 桶**： “更少概念”（例如眼睛更暗、表情中性）。设置 `"slider_strength": -0.5`（任意负值）。
- **Neutral 桶（可选）**：普通样本。省略 `slider_strength` 或设为 `0`。

正负目录不需要保持文件名匹配，只需确保每个桶的样本数量相同即可。

## 步骤 4 — 让 dataloader 指向你的桶

- 使用 Z-Image quickstart 中相同的 dataloader JSON 模式。
- 在每个 backend 条目中添加 `slider_strength`。SimpleTuner 会：
  - 以 **positive → negative → neutral** 的顺序轮转 batch，使两种方向保持新鲜。
  - 仍然遵守每个 backend 的概率，因此权重调节仍然生效。

无需额外标志——只要 `slider_strength` 字段即可。

## 步骤 5 — 训练

使用常规命令（`simpletuner train ...`）或从 WebUI 启动。只要开关启用，slider 目标会自动生效。

## 步骤 6 — 验证（可选滑块微调）

提示词库可以为每条提示词设置适配器强度，用于 A/B 检查：

```json
{
  "plain": "regular prompt",
  "slider_plus": { "prompt": "same prompt", "adapter_strength": 1.2 },
  "slider_minus": { "prompt": "same prompt", "adapter_strength": 0.5 }
}
```

如果省略，验证会使用你的全局强度。

---

## 参考与细节

<details>
<summary>为什么这些目标？（技术）</summary>

SimpleTuner 将滑块 LoRA 路由到自注意力、conv/proj 与 time-embedding 层，以模拟 Concept Sliders 的“只动视觉、不动文本”规则。ControlNet 训练同样会遵循 slider 目标。Assistant 适配器保持冻结。
</details>

<details>
<summary>默认 slider 目标列表（按架构）</summary>

- 通用（SD1.x、SDXL、SD3、Lumina2、Wan、HiDream、LTXVideo、Qwen-Image、Cosmos、Stable Cascade 等）:

  ```json
  [
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn1.to_qkv", "to_qkv",
    "proj_in", "proj_out",
    "conv_in", "conv_out",
    "time_embedding.linear_1", "time_embedding.linear_2"
  ]
  ```

- Flux / Flux2 / Chroma / AuraFlow（仅视觉流）:

  ```json
  ["to_q", "to_k", "to_v", "to_out.0", "to_qkv"]
  ```

  Flux2 变体还包含 `attn.to_q`、`attn.to_k`、`attn.to_v`、`attn.to_out.0`、`attn.to_qkv_mlp_proj`。

- Kandinsky 5（图像/视频）:

  ```json
  ["attn1.to_query", "attn1.to_key", "attn1.to_value", "conv_in", "conv_out", "time_embedding.linear_1", "time_embedding.linear_2"]
  ```

</details>

<details>
<summary>LyCORIS 预设（LoKr 示例）</summary>

多数模型：

```json
{
  "algo": "lokr",
  "multiplier": 1.0,
  "linear_dim": 4,
  "linear_alpha": 1,
  "apply_preset": {
    "target_module": [
      "attn1.to_q",
      "attn1.to_k",
      "attn1.to_v",
      "attn1.to_out.0",
      "conv_in",
      "conv_out",
      "time_embedding.linear_1",
      "time_embedding.linear_2"
    ]
  }
}
```

Flux/Chroma/AuraFlow：将目标替换为 `["attn.to_q","attn.to_k","attn.to_v","attn.to_out.0","attn.to_qkv_mlp_proj"]`（若检查点省略了 `attn.` 则去掉）。为保持文本/上下文不被修改，请避免 `add_*` 投影。

Kandinsky 5：使用 `attn1.to_query/key/value`，并加入 `conv_*` 与 `time_embedding.linear_*`。
</details>

<details>
<summary>采样方式（技术）</summary>

带有 `slider_strength` 的 backend 按正负号分组，并以固定循环采样：positive → negative → neutral。每个组内仍使用常规 backend 概率。耗尽的 backend 会被移除，循环继续在剩余部分进行。
</details>
