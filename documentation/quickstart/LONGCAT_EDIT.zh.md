# LongCat‑Image Edit 快速入门

这是 LongCat‑Image 的 edit/img2img 变体。请先阅读 [LONGCAT_IMAGE.md](../quickstart/LONGCAT_IMAGE.md)；本文件只列出 edit 版本的差异。

---

## 1) 与基础 LongCat‑Image 的差异

|                               | Base (text2img) | Edit |
| ----------------------------- | --------------- | ---- |
| Flavour                       | `final` / `dev` | `edit` |
| Conditioning                  | 无              | **需要条件 latent（参考图像）** |
| Text encoder                  | Qwen‑2.5‑VL     | 带 **vision context** 的 Qwen‑2.5‑VL（编码提示词需要参考图） |
| Pipeline                      | TEXT2IMG        | IMG2IMG/EDIT |
| Validation inputs             | 仅提示词        | 提示词 **和** 参考图 |

---

## 2) 配置调整（CLI/WebUI）

```jsonc
{
  "model_type": "lora",
  "model_family": "longcat_image",
  "model_flavour": "edit",
  "base_model_precision": "int8-quanto",      // fp8-torchao also fine; helps fit 16–24 GB
  "train_batch_size": 1,
  "gradient_checkpointing": true,
  "learning_rate": 5e-5,
  "validation_guidance": 4.5,
  "validation_num_inference_steps": 40,
  "validation_resolution": "768x768"
}
```

保持 `aspect_bucket_alignment` 为 64。不要关闭条件 latent；edit 管线依赖它们。

快速创建配置：
```bash
cp config/config.json.example config/config.json
```
然后设置 `model_family`、`model_flavour`、数据集路径和 output_dir。

---

## 3) 数据加载器：编辑图 + 参考图配对

使用两个对齐的数据集：**编辑图**（caption = 编辑指令）和 **参考图**。编辑数据集的 `conditioning_data` 必须指向参考数据集 ID。文件名必须一一对应。

```jsonc
[
  {
    "id": "edit-images",
    "type": "local",
    "instance_data_dir": "/data/edits",
    "caption_strategy": "textfile",
    "resolution": 768,
    "cache_dir_vae": "/cache/vae/longcat/edit",
    "conditioning_data": ["ref-images"]
  },
  {
    "id": "ref-images",
    "type": "local",
    "instance_data_dir": "/data/refs",
    "caption_strategy": null,
    "resolution": 768,
    "cache_dir_vae": "/cache/vae/longcat/ref"
  }
]
```

> caption_strategy 选项和要求见 [DATALOADER.md](../DATALOADER.md#caption_strategy)。

注意事项：
- 宽高比桶保持 64px 网格。
- 参考图字幕可选；若存在会替换编辑字幕（通常不希望）。
- 编辑与参考的 VAE 缓存路径要分开。
- 若出现缓存缺失或形状错误，清空两边的 VAE 缓存并重新生成。

---

## 4) 验证注意点

- 验证需要参考图来生成条件 latent。将 `edit-images` 的验证分割通过 `conditioning_data` 指向 `ref-images`。
- Guidance：4–6 效果较好；负向提示词可保持为空。
- 支持预览回调；latent 会在解码前自动 unpack。
- 若验证报条件 latent 缺失，请确认验证数据加载器同时包含 edit 与参考条目且文件名匹配。

---

## 5) 推理 / 验证命令

快速 CLI 验证：
```bash
simpletuner validate \
  --model_family longcat_image \
  --model_flavour edit \
  --validation_resolution 768x768 \
  --validation_guidance 4.5 \
  --validation_num_inference_steps 40
```

WebUI：选择 **Edit** 管线，提供源图和编辑指令。

---

## 6) 开始训练（CLI）

配置与数据加载器就绪后：
```bash
simpletuner train --config config/config.json
```
确保训练期间存在参考数据集，以便生成/读取条件 latent。

---

## 7) 故障排查

- **缺失条件 latent**：确保通过 `conditioning_data` 连接参考数据集且文件名匹配。
- **MPS dtype 错误**：管线会在 MPS 上自动将 pos‑ids 降为 float32；其余保持 float32/bf16。
- **预览通道不匹配**：解码前会 un‑patchify latent（请保持此 SimpleTuner 版本）。
- **edit OOM**：降低验证分辨率/步数，降低 `lora_rank`，启用 group offload，优先 `int8-quanto`/`fp8-torchao`。
