# Kontext [dev] 迷你快速入门

> 📝  Kontext 与 Flux 共享 90% 的训练流程，因此本文件仅列出*不同点*。若某一步骤未在此提及，请遵循原始 [说明](../quickstart/FLUX.md)。


---

## 1. 模型概览

|                                                  | Flux-dev               | Kontext-dev                                 |
| ------------------------------------------------ | -------------------    | ------------------------------------------- |
| 许可                                             | 非商用                 | 非商用                                       |
| 引导                                             | 蒸馏 (CFG ≈ 1)         | 蒸馏 (CFG ≈ 1)                               |
| 可用版本                                         | *dev*, schnell,\[pro]  | *dev*, \[pro, max]                          |
| T5 序列长度                                      | 512 dev, 256 schnell   | 512 dev                                     |
| 典型 1024 px 推理时间<br>(4090 @ CFG 1)           | ≈ 20 s                 | **≈ 80 s**                                  |
| 1024 px LoRA @ int8-quanto 的 VRAM                | 18 G                   | **24 G**                                    |

Kontext 保留了 Flux 的 transformer 主干，但引入了**成对参考条件**。

Kontext 提供两种 `conditioning_type` 模式:

* `conditioning_type=reference_loose` (✅ 稳定) – 参考图像可与编辑图像在宽高比/尺寸上不同。
  - 两个数据集都会进行元数据扫描、宽高比桶和裁剪，且彼此独立，这可能显著增加启动时间。
  - 如果你需要确保编辑图像与参考图像对齐（例如数据加载器按文件名一一对应），这可能是问题。
* `conditioning_type=reference_strict` (✅ 稳定) – 参考图像会按与编辑裁剪完全相同的方式预处理。
  - 如果你需要编辑图像与参考图像的裁剪/宽高比桶完全对齐，就应这样配置数据集。
  - 过去需要 `--vae_cache_ondemand` 并增加一些 VRAM，现在不再需要。
  - 启动时会从源数据集复制裁剪/宽高比桶元数据，无需手动处理。

字段定义见 [`conditioning_type`](../DATALOADER.md#conditioning_type) 与 [`conditioning_data`](../DATALOADER.md#conditioning_data)。如需控制多个条件数据集的采样方式，请按 [OPTIONS](../OPTIONS.md#--conditioning_multidataset_sampling) 中说明使用 `conditioning_multidataset_sampling`。


---

## 2. 硬件要求

* **系统内存**: 量化仍需要 50GB。
* **GPU**: 1024 px 训练 **且使用 int8-quanto** 时，3090 (24G) 才是现实的最低配置。
  * 具备 Flash Attention 3 的 Hopper H100/H200 系统可启用 `--fuse_qkv_projections` 以大幅加速训练。
  * 若以 512 px 训练可勉强使用 12G 显卡，但批次会很慢（序列长度仍然很长）。


---

## 3. 快速配置差异

下面是相较于常规 Flux 训练配置，你在 `config/config.json` 中需要的*最小*变更集。

<details>
<summary>查看示例配置</summary>

```jsonc
{
  "model_family":   "flux",
  "model_flavour": "kontext",                       // <‑‑ 将此项从 "dev" 改为 "kontext"
  "base_model_precision": "int8-quanto",            // 1024 px 可适配 24G
  "gradient_checkpointing": true,
  "fuse_qkv_projections": false,                    // <‑‑ 用于 Hopper H100/H200 系统加速训练。警告：需要手动安装 flash-attn。
  "lora_rank": 16,
  "learning_rate": 1e-5,
  "optimizer": "optimi-lion",                       // <‑‑ Lion 速度更快，adamw_bf16 更慢但可能更稳定。
  "max_train_steps": 10000,
  "validation_guidance": 2.5,                       // <‑‑ kontext 在 2.5 的 guidance 下表现最佳
  "validation_resolution": "1024x1024",
  "conditioning_multidataset_sampling": "random"    // <-- 当定义了两个条件数据集时，设为 "combined" 会同时显示而不是切换
}
```
</details>

### 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可显著提高训练稳定性和性能的实验功能。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：**通过让模型在训练期间生成自己的输入来减少曝光偏差并提高输出质量。

> ⚠️ 这些功能会增加训练的计算开销。

</details>

### 数据加载器片段（多数据后端）

如果你手动整理了成对图像数据集，可以用两个独立目录配置：一个放编辑图像，一个放参考图像。

编辑数据集的 `conditioning_data` 字段应指向参考数据集的 `id`。

<details>
<summary>查看示例配置</summary>

```jsonc
[
  {
    "id": "my-edited-images",
    "type": "local",
    "cache_dir_vae": "/cache/vae/flux/kontext/edited-images",   // <-- VAE 输出缓存位置
    "instance_data_dir": "/datasets/edited-images",             // <-- 使用绝对路径
    "conditioning_data": [
      "my-reference-images"                                     // <‑‑ 这里应为参考集的 "id"
                                                                // 可以再指定第二个参考集交替或合并，例如 ["reference-images", "reference-images2"]
    ],
    "resolution": 1024,
    "caption_strategy": "textfile"                              // <-- 这些 caption 应包含编辑指令
  },
  {
    "id": "my-reference-images",
    "type": "local",
    "cache_dir_vae": "/cache/vae/flux/kontext/ref-images",      // <-- VAE 输出缓存位置。必须与其他数据集路径不同。
    "instance_data_dir": "/datasets/reference-images",          // <-- 使用绝对路径
    "conditioning_type": "reference_strict",                    // <‑‑ 若设为 reference_loose，则图像会独立裁剪
    "resolution": 1024,
    "caption_strategy": null,                                   // <‑‑ 参考图不需要 caption，但若提供会覆盖编辑 caption
                                                                // 注意：使用 conditioning_multidataset_sampling=combined 时不能单独指定参考 caption。
                                                                // 仅使用编辑数据集的 caption。
  }
]
```
</details>

> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).

*每个编辑图像**必须**在两个数据集文件夹中存在同名同扩展的对应文件。SimpleTuner 会自动将参考嵌入拼接到编辑条件中。*

已准备的示例数据集 [Kontext Max derived demo dataset](https://huggingface.co/datasets/terminusresearch/KontextMax-Edit-smol) 包含参考/编辑图像及其 caption 文件，可用于浏览了解配置方式。

### 设置专用验证集划分

以下配置示例使用 200,000 样本作为训练集、少量样本作为验证集。
对 Kontext 请保持 `validation_using_datasets` 未设置或为 `false`。验证会自动使用配对的参考数据集；`eval_dataset_id` 仅用于选择哪个编辑数据集提供验证样本。

在 `config.json` 中添加:

<details>
<summary>查看示例配置</summary>

```json
{
  "eval_dataset_id": "edited-images",
}
```
</details>

在 `multidatabackend.json` 中，`edited-images` 和 `reference-images` 应包含与训练集相同结构的验证数据。

<details>
<summary>查看示例配置</summary>

```json
[
    {
        "id": "edited-images",
        "disabled": false,
        "type": "local",
        "instance_data_dir": "/datasets/edit/edited-images",
        "minimum_image_size": 1024,
        "maximum_image_size": 1536,
        "target_downsample_size": 1024,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "caption_strategy": "textfile",
        "cache_dir_vae": "cache/vae/flux-edit",
        "vae_cache_clear_each_epoch": false,
        "conditioning_data": ["reference-images"]
    },
    {
        "id": "reference-images",
        "disabled": false,
        "type": "local",
        "instance_data_dir": "/datasets/edit/reference-images",
        "minimum_image_size": 1024,
        "maximum_image_size": 1536,
        "target_downsample_size": 1024,
        "resolution": 1024,
        "resolution_type": "pixel_area",
        "caption_strategy": null,
        "cache_dir_vae": "cache/vae/flux-ref",
        "vae_cache_clear_each_epoch": false,
        "conditioning_type": "reference_strict"
    },
    {
        "id": "subjects200k-left",
        "disabled": false,
        "type": "huggingface",
        "dataset_name": "Yuanshi/Subjects200K",
        "caption_strategy": "huggingface",
        "metadata_backend": "huggingface",
        "resolution": 512,
        "resolution_type": "pixel_area",
        "conditioning_data": ["subjects200k-right"],
        "huggingface": {
            "caption_column": "description.description_0",
            "image_column": "image",
            "composite_image_config": {
                "enabled": true,
                "image_count": 2,
                "select_index": 0
            }
        }
    },
    {
        "id": "subjects200k-right",
        "disabled": false,
        "type": "huggingface",
        "dataset_type": "conditioning",
        "conditioning_type": "reference_strict",
        "source_dataset_id": "subjects200k-left",
        "dataset_name": "Yuanshi/Subjects200K",
        "caption_strategy": "huggingface",
        "metadata_backend": "huggingface",
        "resolution": 512,
        "resolution_type": "pixel_area",
        "huggingface": {
            "caption_column": "description.description_1",
            "image_column": "image",
            "composite_image_config": {
                "enabled": true,
                "image_count": 2,
                "select_index": 1
            }
        }
    },

    {
        "id": "text-embed-cache",
        "dataset_type": "text_embeds",
        "default": true,
        "type": "local",
        "cache_dir": "cache/text/flux"
    }
]
```
</details>

### 自动生成参考-编辑成对数据

如果你没有现成的参考/编辑配对数据，SimpleTuner 可从单一数据集自动生成。这在训练以下类型模型时尤其有用:
- 图像增强 / 超分辨率
- JPEG 伪影去除
- 去模糊
- 其他修复类任务

#### 示例: 去模糊训练数据集

<details>
<summary>查看示例配置</summary>

```jsonc
[
  {
    "id": "high-quality-images",
    "type": "local",
    "instance_data_dir": "/path/to/sharp-images",
    "resolution": 1024,
    "caption_strategy": "textfile",
    "conditioning": [
      {
        "type": "superresolution",
        "blur_radius": 3.0,
        "blur_type": "gaussian",
        "add_noise": true,
        "noise_level": 0.02,
        "captions": ["enhance sharpness", "deblur", "increase clarity", "sharpen image"]
      }
    ]
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/kontext"
  }
]
```
</details>

该配置将:
1. 从高质量清晰图像生成模糊版本（即“参考图像”）
2. 使用原始高质量图像作为训练目标
3. 训练 Kontext 去增强/去模糊低质量参考图像

> **注意**: 使用 `conditioning_multidataset_sampling=combined` 时不能在条件数据集上定义 `captions`。会改用编辑数据集的 captions。

#### 示例: JPEG 伪影去除

<details>
<summary>查看示例配置</summary>

```jsonc
[
  {
    "id": "pristine-images",
    "type": "local",
    "instance_data_dir": "/path/to/pristine-images",
    "resolution": 1024,
    "caption_strategy": "textfile",
    "conditioning": [
      {
        "type": "jpeg_artifacts",
        "quality_mode": "range",
        "quality_range": [10, 30],
        "compression_rounds": 2,
        "captions": ["remove compression artifacts", "restore quality", "fix jpeg artifacts"]
      }
    ]
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/kontext"
  }
]
```
</details>

#### 重要说明

1. **生成发生在启动时**: 降质版本会在训练开始时自动生成
2. **缓存**: 生成的图像会保存，后续运行不会重复生成
3. **Caption 策略**: conditioning 配置中的 `captions` 提供了任务特定提示，比通用描述效果更好
4. **性能**: 这些基于 CPU 的生成器（模糊、JPEG）速度快且可多进程并行
5. **磁盘空间**: 请确保有足够空间保存生成图像，体积可能很大！目前无法按需生成

更多条件类型与高级配置请参阅 [ControlNet 文档](../CONTROLNET.md)。

---

## 4. Kontext 专属训练提示

1. **序列更长 → 步长更慢。**  1024 px、rank-1 LoRA、bf16 + int8 条件下，单张 4090 约为 ~0.4 it/s。
2. **探索合适的设置。**  Kontext 的微调研究不多，稳妥起见使用 `1e-5` (Lion) 或 `5e-4` (AdamW)。
3. **关注 VAE 缓存时的 VRAM 峰值。**  OOM 时添加 `--offload_during_startup=true`、降低 `resolution`，或在 `config.json` 中启用 VAE 切片。
4. **可以不用参考图像训练，但目前 SimpleTuner 不支持。**  当前实现较为硬编码，要求提供条件图像，但你可以在成对编辑数据旁再提供普通数据集，以学习主体与相似性。
5. **Guidance 再蒸馏。**  与 Flux-dev 一样，Kontext-dev 为 CFG 蒸馏；若需要多样性，可以用 `validation_guidance_real > 1` 重新训练，并在推理中使用 Adaptive-Guidance 节点。注意这会收敛更慢且需要更大的 LoRA rank 或 Lycoris LoKr 才能成功。
6. **全秩训练可能得不偿失。**  Kontext 设计用于低秩训练，全秩训练不一定优于 Lycoris LoKr，而 LoKr 通常比标准 LoRA 更好且更省事。若执意尝试，需要 DeepSpeed。
7. **可使用两张或更多参考图像训练。**  例如有主体-主体-场景组合，可将所有相关图像作为参考输入，确保文件名在各文件夹中一致即可。

---

## 5. 推理注意事项

- 训练与推理精度需一致；int8 训练最好搭配 int8 推理，以此类推。
- 因为每次要处理两张图像，会很慢。4090 上 1024 px 编辑预计 80 秒左右。

---

## 6. 排障速查表

| 症状                                 | 可能原因                 | 快速解决方案                                         |
| ------------------------------------ | ------------------------ | ---------------------------------------------------- |
| 量化时 OOM                           | **系统** 内存不足        | 使用 `quantize_via=cpu`                              |
| 参考图被忽略 / 未应用编辑            | 数据加载器配对错误       | 确保文件名一致且设置 `conditioning_data` 字段       |
| 方格网格伪影                         | 低质量编辑主导           | 提升数据质量、降低学习率、避免 Lion                 |

---

## 7. 延伸阅读

关于 LoKr、NF4 量化、DeepSpeed 等高级调参，请参考 [Flux 原始快速入门](../quickstart/FLUX.md) —— 除非上文另有说明，其余参数均可通用。
