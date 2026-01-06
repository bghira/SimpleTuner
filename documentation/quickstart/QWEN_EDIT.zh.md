# Qwen Image Edit 快速入门

本指南涵盖 SimpleTuner 支持的 Qwen Image **编辑**版本：

- `edit-v1` – 每个训练样本仅一张参考图像。参考图像通过 Qwen2.5-VL 文本编码器编码并缓存为**条件图像嵌入**。
- `edit-v2`（“edit plus”）– 每个样本最多三张参考图像，按需编码为 VAE latent。

两种版本继承了基础 [Qwen Image 快速入门](./QWEN_IMAGE.md) 中的大部分内容；本页仅强调微调编辑检查点的**差异**。

---

## 1. 硬件清单

基础模型仍为 **20B 参数**：

| 要求 | 推荐 |
|-------------|----------------|
| GPU 显存    | 24G 最低（需 int8/nf4 量化） • 强烈推荐 40G+ |
| 精度        | `mixed_precision=bf16`, `base_model_precision=int8-quanto`（或 `nf4-bnb`） |
| 批大小      | 必须保持 `train_batch_size=1`；使用梯度累积模拟有效 batch |

其他训练前置条件与 [Qwen Image 指南](./QWEN_IMAGE.md) 相同（Python ≥ 3.10、CUDA 12.x 镜像等）。

---

## 2. 配置要点

在 `config/config.json` 中：

<details>
<summary>查看示例配置</summary>

```jsonc
{
  "model_type": "lora",
  "model_family": "qwen_image",
  "model_flavour": "edit-v1",      // 或 "edit-v2"
  "train_batch_size": 1,
  "gradient_accumulation_steps": 4,
  "validation_resolution": "1024x1024",
  "validation_guidance": 4.0,
  "validation_num_inference_steps": 30,
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "base_model_precision": "int8-quanto",
  "quantize_via": "cpu",
  "quantize_activations": false,
  "flow_schedule_shift": 1.73,
  "data_backend_config": "config/qwen_edit/multidatabackend.json"
}
```
</details>

- EMA 默认在 CPU 上运行，除非需要更快的检查点，否则可保持启用。
- 24G 显卡需降低 `validation_resolution`（如 `768x768`）。
- 对 `edit-v2`，可在 `model_kwargs` 下添加 `match_target_res`，让控制图像继承目标分辨率而非默认 1MP 打包：

<details>
<summary>查看示例配置</summary>

```jsonc
"model_kwargs": {
  "match_target_res": true
}
```
</details>

### 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可显著提高训练稳定性和性能的实验功能。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：**通过让模型在训练期间生成自己的输入来减少曝光偏差并提高输出质量。

> ⚠️ 这些功能会增加训练的计算开销。

---

</details>

## 3. 数据加载器结构

两种版本都需要**成对数据集**：编辑图像、可选编辑 caption，以及一张或多张**文件名完全一致**的控制/参考图像。

字段详情见 [`conditioning_type`](../DATALOADER.md#conditioning_type) 和 [`conditioning_data`](../DATALOADER.md#conditioning_data)。若提供多个条件数据集，可用 [OPTIONS](../OPTIONS.md#--conditioning_multidataset_sampling) 中的 `conditioning_multidataset_sampling` 控制采样。

### 3.1 edit‑v1（单控制图像）

主数据集应引用一个条件数据集**以及**条件图像嵌入缓存：

<details>
<summary>查看示例配置</summary>

```jsonc
[
  {
    "id": "qwen-edit-images",
    "type": "local",
    "instance_data_dir": "/datasets/qwen-edit/images",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "conditioning_data": ["qwen-edit-reference"],
    "conditioning_image_embeds": "qwen-edit-ref-embeds",
    "cache_dir_vae": "cache/vae/qwen-edit-images"
  },
  {
    "id": "qwen-edit-reference",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/qwen-edit/reference",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/qwen-edit-reference"
  },
  {
    "id": "qwen-edit-ref-embeds",
    "type": "local",
    "dataset_type": "conditioning_image_embeds",
    "cache_dir": "cache/conditioning_image_embeds/qwen-edit"
  }
]
```
</details>

> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).

- `conditioning_type=reference_strict` 可确保裁剪与编辑图像一致。仅在参考图像允许与编辑图像宽高比不一致时使用 `reference_loose`。
- `conditioning_image_embeds` 项用于存储每张参考图的 Qwen2.5-VL 视觉 token。如省略，SimpleTuner 会在 `cache/conditioning_image_embeds/<dataset_id>` 下创建默认缓存。

### 3.2 edit‑v2（多控制）

对 `edit-v2`，将所有控制数据集列入 `conditioning_data`。每一项提供一张额外控制帧。无需条件图像嵌入缓存，因为 latent 会按需计算。

<details>
<summary>查看示例配置</summary>

```jsonc
[
  {
    "id": "qwen-edit-plus-images",
    "type": "local",
    "instance_data_dir": "/datasets/qwen-edit-plus/images",
    "caption_strategy": "textfile",
    "resolution": 1024,
    "conditioning_data": [
      "qwen-edit-plus-reference-a",
      "qwen-edit-plus-reference-b",
      "qwen-edit-plus-reference-c"
    ],
    "cache_dir_vae": "cache/vae/qwen-edit-plus/images"
  },
  {
    "id": "qwen-edit-plus-reference-a",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/qwen-edit-plus/reference_a",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/qwen-edit-plus/ref_a"
  },
  {
    "id": "qwen-edit-plus-reference-b",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/qwen-edit-plus/reference_b",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/qwen-edit-plus/ref_b"
  },
  {
    "id": "qwen-edit-plus-reference-c",
    "type": "local",
    "dataset_type": "conditioning",
    "instance_data_dir": "/datasets/qwen-edit-plus/reference_c",
    "conditioning_type": "reference_strict",
    "resolution": 1024,
    "cache_dir_vae": "cache/vae/qwen-edit-plus/ref_c"
  }
]
```
</details>

使用与你的参考图像数量相同的控制数据集（1–3）。SimpleTuner 会通过文件名匹配来保持每个样本对齐。

---

## 4. 运行训练器

最快的冒烟测试是运行示例预设：

```bash
simpletuner train example=qwen_image.edit-v1-lora
# 或
simpletuner train example=qwen_image.edit-v2-lora
```

手动启动：

```bash
simpletuner train \
  --config config/config.json \
  --data config/qwen_edit/multidatabackend.json
```

### 提示

- 除非确有理由在无编辑指令下训练，否则保持 `caption_dropout_probability` 为 `0.0`。
- 长时间训练时降低验证频率（`validation_step_interval`），避免昂贵的编辑验证占用过多运行时间。
- Qwen 编辑检查点不带 guidance head；`validation_guidance` 通常在 **3.5–4.5** 之间。

---

## 5. 验证预览

若希望在验证中同时预览参考图像，可将验证编辑/参考配对放入单独的数据集（布局与训练集相同），并设置：

<details>
<summary>查看示例配置</summary>

```jsonc
{
  "eval_dataset_id": "qwen-edit-val"
}
```
</details>

SimpleTuner 将复用该数据集的条件图像进行验证。

---

### 故障排除

- **`ValueError: Control tensor list length does not match batch size`** – 确保每个条件数据集都包含所有编辑图像的文件。空目录或文件名不匹配会触发该错误。
- **验证阶段显存不足** – 降低 `validation_resolution`、`validation_num_inference_steps`，或进一步量化（`base_model_precision=int2-quanto`）后重试。
- **`edit-v1` 下缓存不存在** – 检查主数据集的 `conditioning_image_embeds` 是否指向现有缓存数据集条目。

---

现在你可以将基础 Qwen Image 快速入门适配到编辑训练中。有关完整配置选项（文本编码器缓存、多后端采样等），请复用 [FLUX_KONTEXT.md](./FLUX_KONTEXT.md) 的指导——数据集配对流程相同，只是模型家族改为 `qwen_image`。
