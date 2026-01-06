# LayerSync（SimpleTuner）

LayerSync 是一种面向 Transformer 的“自教式”约束：让某一层（学生）对齐更强的层（教师）。它轻量、自包含，不需要额外下载教师模型。

## 适用场景

- 你在训练能暴露隐藏状态的 Transformer 家族（如 Flux/Flux Kontext/Flux.2、PixArt Sigma、SD3/SDXL、Sana、Wan、Qwen Image/Edit、Hunyuan Video、LTXVideo、Kandinsky5 Video、Chroma、ACE-Step、HiDream、Cosmos/LongCat/Z-Image/Auraflow）。
- 需要内置正则化，但不想引入外部教师权重。
- 训练中期出现漂移或头部不稳定，希望把中间层拉回更深教师层。
- 有一些 VRAM 余量，可在当前步骤保留学生/教师激活。

## 快速设置（WebUI）

1. 打开 **Training → Loss functions**。
2. 启用 **LayerSync**。
3. 将 **Student Block** 设为中间层，**Teacher Block** 设为更深一层。24 层 DiT 模型（Flux、PixArt、SD3）可从 `8` → `16` 开始；更短的堆栈可让教师比学生深几层即可。
4. **Weight** 保持 `0.2`（LayerSync 启用时默认值）。
5. 正常训练；日志会包含 `layersync_loss` 和 `layersync_similarity`。

## 快速设置（config JSON / CLI）

```json
{
  "layersync_enabled": true,
  "layersync_student_block": 8,
  "layersync_teacher_block": 16,
  "layersync_lambda": 0.2
}
```

## 调参要点

- `layersync_student_block` / `layersync_teacher_block`：兼容 1-based 索引；先尝试 `idx-1`，再尝试 `idx`。
- `layersync_lambda`：缩放余弦损失；启用时必须 > 0（默认 `0.2`）。
- 未指定教师时默认使用学生块，损失变为自相似。
- VRAM：辅助损失计算前会保留两层激活；若内存紧张，禁用 LayerSync（或 CREPA）。
- 与 CREPA/TwinFlow 可共存；共享同一隐藏状态缓冲区。

<details>
<summary>工作原理（实践视角）</summary>

- 计算学生与教师 token 的负余弦相似度；权重越高，学生特征越向教师靠拢。
- 教师 token 始终 detach，避免梯度回流。
- 同时支持图像/视频 Transformer 的 3D `(B, S, D)` 与 4D `(B, T, P, D)` 隐藏状态。
- 上游参数映射：
  - `--encoder-depth` → `--layersync_student_block`
  - `--gt-encoder-depth` → `--layersync_teacher_block`
  - `--reg-weight` → `--layersync_lambda`
- 默认关闭；启用且未指定时，`layersync_lambda=0.2`。

</details>

<details>
<summary>技术细节（SimpleTuner 内部）</summary>

- 实现：`simpletuner/helpers/training/layersync.py`；由 `ModelFoundation._apply_layersync_regularizer` 调用。
- 隐藏状态捕获：当 LayerSync 或 CREPA 请求时触发；Transformer 通过 `_store_hidden_state` 保存为 `layer_{idx}`。
- 层索引解析：先尝试 1-based，再尝试 0-based；缺失则报错。
- 损失路径：归一化学生/教师 token，计算平均余弦相似度，记录 `layersync_loss` 和 `layersync_similarity`，并将缩放后的损失加入主目标。
- 交互：在 CREPA 之后执行，复用同一缓冲区；之后清理缓冲区。

</details>

## 常见问题

- 缺少学生块 → 启动报错；请显式设置 `layersync_student_block`。
- 权重 ≤ 0 → 启动报错；不确定时使用默认 `0.2`。
- 请求超过模型深度的层 → “LayerSync could not find layer” 错误；降低索引。
- 在不暴露 Transformer 隐藏状态的模型上启用（Kolors、Lumina2、Stable Cascade C、Kandinsky5 Image、OmniGen）会失败；请仅用于 Transformer 家族。
- VRAM 峰值：降低块索引或禁用 CREPA/LayerSync 释放缓冲区。

当你需要低成本的内置正则化来温和地约束中间表示而不引入外部教师时，可使用 LayerSync。
