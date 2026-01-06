# LongCat‑Video Edit（图生视频）快速入门

本指南介绍 LongCat‑Video 的图生视频训练与验证流程。无需切换风格；同一 `final` 检查点同时覆盖文生视频与图生视频，区别在于数据集和验证设置。

---

## 1) 与基础 LongCat‑Video 的差异

|                               | Base (text2video) | Edit / I2V |
| ----------------------------- | ----------------- | ---------- |
| Flavour                       | `final`           | `final`（同一权重） |
| Conditioning                  | 无                | **需要条件帧**（首帧 latent 固定） |
| Text encoder                  | Qwen‑2.5‑VL       | Qwen‑2.5‑VL（相同） |
| Pipeline                      | TEXT2IMG          | IMG2VIDEO |
| Validation inputs             | 仅提示词          | 提示词 **和** 条件图 |
| Buckets / stride              | 64px 桶，`(frames-1)%4==0` | 相同 |

**继承的核心默认值**
- shift `12.0` 的流匹配。
- 64px 宽高比桶对齐。
- Qwen‑2.5‑VL 文本编码器；CFG 开启时会自动补空负向。
- 默认帧数 93（满足 `(frames-1)%4==0`）。

---

## 2) 配置调整（CLI/WebUI）

```jsonc
{
  "model_family": "longcat_video",
  "model_flavour": "final",
  "model_type": "lora",
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "lora_rank": 8,
  "learning_rate": 1e-4,
  "validation_resolution": "480x832",
  "validation_num_video_frames": 93,
  "validation_num_inference_steps": 40,
  "validation_guidance": 4.0,
  "validation_using_datasets": true,
  "eval_dataset_id": "longcat-video-val"
}
```

保持 `aspect_bucket_alignment` 为 64。首帧 latent 用于起始图像，保持不变。除非有充分理由，不要修改 93 帧（已满足 VAE 步长规则 `(frames - 1) % 4 == 0`）。

快速设置：
```bash
cp config/config.json.example config/config.json
```
填写 `model_family`、`model_flavour`、`output_dir`、`data_backend_config` 和 `eval_dataset_id`。除非确认需要变更，上面默认值可保持不动。

CUDA 注意力选项：
- 在 CUDA 上，LongCat‑Video 会优先使用随附的 block‑sparse Triton kernel（可用时），否则回退标准调度器。无需手动切换。
- 若想强制 xFormers，在配置/CLI 中设置 `attention_implementation: "xformers"`。

---

## 3) 数据加载器：视频片段 + 起始帧配对

- 创建两套数据集：
  - **视频片段**：目标视频 + 字幕（编辑指令）。标记 `is_i2v: true`，并将 `conditioning_data` 指向起始帧数据集 ID。
  - **起始帧**：每个片段一张图，文件名一致，无需字幕。
- 都保持 64px 网格（如 480x832）。高/宽需可被 16 整除。帧数需满足 `(frames - 1) % 4 == 0`；93 已满足。
- 视频与起始帧的 VAE 缓存路径要分开。

示例 `multidatabackend.json`：
```jsonc
[
  {
    "id": "longcat-video-train",
    "type": "local",
    "dataset_type": "video",
    "is_i2v": true,
    "instance_data_dir": "/data/video-clips",
    "caption_strategy": "textfile",
    "resolution": 480,
    "cache_dir_vae": "/cache/vae/longcat/video",
    "conditioning_data": ["longcat-video-cond"]
  },
  {
    "id": "longcat-video-cond",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/data/video-start-frames",
    "caption_strategy": null,
    "resolution": 480,
    "cache_dir_vae": "/cache/vae/longcat/video-cond"
  }
]
```

> caption_strategy 选项和要求见 [DATALOADER.md](../DATALOADER.md#caption_strategy)。

---

## 4) 验证注意点

- 添加与训练相同结构的小型验证集。设置 `validation_using_datasets: true` 并将 `eval_dataset_id` 指向验证集 ID（如 `longcat-video-val`），验证时会自动取起始帧。
- WebUI 预览：启动 `simpletuner server`，选择 LongCat‑Video edit，上传起始帧 + 提示词。
- Guidance：3.5–5.0 效果较好；CFG 开启时会自动补空负向。
- 低 VRAM 预览/训练时，设置 `musubi_blocks_to_swap`（建议 4–8）并视需要设置 `musubi_block_swap_device`，将最后的 Transformer block 从 CPU 流式加载。吞吐会降低，但峰值 VRAM 会下降。
- 条件帧在采样中保持固定，仅后续帧进行去噪。

---

## 5) 开始训练（CLI）

配置与数据加载器就绪后：
```bash
simpletuner train --config config/config.json
```
确保训练数据中存在起始帧，以便生成条件 latent。

---

## 6) 故障排查

- **缺少条件图**：通过 `conditioning_data` 提供条件数据集并保持文件名一致；验证时把 `eval_dataset_id` 指向验证集 ID。
- **高/宽错误**：保持可被 16 整除并对齐 64px 网格。
- **首帧漂移**：降低 guidance（3.5–4.0）或减少步数。
- **OOM**：降低验证分辨率/帧数，减少 `lora_rank`，启用 group offload，或使用 `int8-quanto`/`fp8-torchao`。
