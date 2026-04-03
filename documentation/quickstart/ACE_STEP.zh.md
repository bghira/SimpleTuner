# ACE-Step 快速入门

本示例将训练 ACE-Step 音频生成模型。SimpleTuner 当前同时支持原始 ACE-Step v1 3.5B 训练路径，以及面向 ACE-Step v1.5 bundle 的前向兼容 LoRA 训练路径。

## 概览

ACE-Step 是一个面向高质量音频合成的 transformer 流匹配音频模型。在 SimpleTuner 中：

- `base` 对应原始 ACE-Step v1 3.5B 训练路径。
- `v15-turbo`、`v15-base`、`v15-sft` 对应从 `ACE-Step/Ace-Step1.5` 加载的 ACE-Step v1.5 bundle 变体。

## 硬件要求

ACE-Step 为 3.5B 参数，相比 Flux 等大型图像模型更轻量。

- **最低**：NVIDIA GPU 12GB+ VRAM（如 3060、4070）。
- **推荐**：NVIDIA GPU 24GB+ VRAM（如 3090、4090、A10G）用于更大 batch。
- **Mac**：Apple Silicon 可通过 MPS 支持（需约 36GB+ 统一内存）。

### 存储要求

> ⚠️ **磁盘占用警告：** 音频模型的 VAE 缓存可能很大。例如，单个 60 秒音频片段可产生约 89MB 的缓存 latent 文件。此缓存策略可显著降低训练所需 VRAM。请确保磁盘空间充足。

> 💡 **提示：**对大型数据集，可使用 `--vae_cache_disable` 禁止将嵌入写入磁盘。这会隐式启用按需缓存，节省磁盘空间但会增加训练时间和内存占用（因为编码发生在训练循环中）。

> 💡 **提示：**使用 `int8-quanto` 量化可在较低 VRAM（如 12GB-16GB）GPU 上训练，且质量损失较小。

## 前提条件

确保已有可用的 Python 3.10+ 环境。

```bash
pip install simpletuner
```

## 配置

建议将配置整理有序。本示例将创建专用文件夹。

```bash
mkdir -p config/acestep-training-demo
```

### 关键设置

SimpleTuner 当前支持以下 ACE-Step flavour：

- `base`：原始 ACE-Step v1 3.5B
- `v15-turbo`、`v15-base`、`v15-sft`：ACE-Step v1.5 bundle 变体

请根据目标变体选择对应配置。

现成可用的示例 preset 位于：

- `simpletuner/examples/ace_step-v1-0.peft-lora`
- `simpletuner/examples/ace_step-v1-5.peft-lora`

你可以直接使用 `simpletuner train example=ace_step-v1-0.peft-lora` 或 `simpletuner train example=ace_step-v1-5.peft-lora` 启动。

#### ACE-Step v1 示例

创建 `config/acestep-training-demo/config.json`，填写以下内容：

<details>
<summary>查看示例配置</summary>

```json
{
  "model_family": "ace_step",
  "model_type": "lora",
  "model_flavour": "base",
  "pretrained_model_name_or_path": "ACE-Step/ACE-Step-v1-3.5B",
  "resolution": 0,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "data_backend_config": "config/acestep-training-demo/multidatabackend.json"
}
```
</details>

#### ACE-Step v1.5 示例

对于 ACE-Step v1.5，请保持 `model_family: "ace_step"`，选择一个 v1.5 flavour，并将 checkpoint 根路径指向共享的 v1.5 bundle：

<details>
<summary>查看示例配置</summary>

```json
{
  "model_family": "ace_step",
  "model_type": "lora",
  "model_flavour": "v15-base",
  "pretrained_model_name_or_path": "ACE-Step/Ace-Step1.5",
  "resolution": 0,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "data_backend_config": "config/acestep-training-demo/multidatabackend.json"
}
```
</details>

### 验证设置

在 `config.json` 中添加以下内容以监控训练进度：

- **`validation_prompt`**：用于生成验证音频的文本描述（如 “A catchy pop song with upbeat drums”）。
- **`validation_lyrics`**：（可选）模型需要演唱的歌词。
- **`validation_audio_duration`**：验证音频时长（秒），默认 30.0。
- **`validation_guidance`**：引导尺度（默认约 3.0 - 5.0）。
- **`validation_step_interval`**：生成样本的频率（如每 100 步）。

> ⚠️ **ACE-Step v1.5 当前限制：** 目前的 SimpleTuner 集成支持 v1.5 训练，但内置 ACE-Step 验证/推理 pipeline 仍然只支持 v1.0。进行 v1.5 训练时，请关闭训练内验证，或使用上游/外部推理工具完成验证。

### 高级实验功能

<details>
<summary>显示高级实验详情</summary>


SimpleTuner 包含可显著提高训练稳定性和性能的实验功能。

*   **[计划采样（Rollout）](../experimental/SCHEDULED_SAMPLING.md)：**通过让模型在训练期间生成自己的输入来减少曝光偏差并提高输出质量。

> ⚠️ 这些功能会增加训练的计算开销。

</details>

## 数据集配置

ACE-Step 需要**音频专用**的数据集配置。

### 选项 1：演示数据集（Hugging Face）

快速上手可使用准备好的 [ACEStep-Songs 预设](../data_presets/preset_audio_dataset_with_lyrics.md)。

创建 `config/acestep-training-demo/multidatabackend.json`：

<details>
<summary>查看示例配置</summary>

```json
[
  {
    "id": "acestep-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "Yi3852/ACEStep-Songs",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "cache_dir_vae": "cache/vae/{model_family}/acestep-demo-data"
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/{model_family}"
  }
]
```
</details>

> See caption_strategy options and requirements in [DATALOADER.md](../DATALOADER.md#caption_strategy).

### 选项 2：本地音频文件

创建 `config/acestep-training-demo/multidatabackend.json`：

<details>
<summary>查看示例配置</summary>

```json
[
  {
    "id": "my-audio-dataset",
    "type": "local",
    "dataset_type": "audio",
    "instance_data_dir": "datasets/my_audio_files",
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "disabled": false
  },
  {
    "id": "text-embeds",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "local",
    "cache_dir": "cache/text/{model_family}"
  }
]
```
</details>

### 数据结构

将音频文件放入 `datasets/my_audio_files`。SimpleTuner 支持多种格式：

- **无损：** `.wav`, `.flac`, `.aiff`, `.alac`
- **有损：** `.mp3`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.opus`

> ℹ️ **注意：** 要支持 MP3、AAC、WMA 等格式，系统需安装 **FFmpeg**。

对于 captions 与歌词，将相应文本文件放在音频文件旁：

- **音频：** `track_01.wav`
- **Caption（提示词）：** `track_01.txt`（文本描述，如 “A slow jazz ballad”）
- **歌词（可选）：** `track_01.lyrics`（歌词文本）

<details>
<summary>示例数据集结构</summary>

```text
datasets/my_audio_files/
├── track_01.wav
├── track_01.txt
└── track_01.lyrics
```
</details>

> 💡 **高级：**如果你的数据集使用不同命名规范（如 `_lyrics.txt`），可在数据集配置中自定义。

<details>
<summary>自定义歌词文件名示例</summary>

```json
"audio": {
  "lyrics_filename_format": "{filename}_lyrics.txt"
}
```
</details>

> ⚠️ **关于歌词：**若某个样本缺少 `.lyrics` 文件，则歌词嵌入会被置零。ACE-Step 预期存在歌词条件；若大量训练在无歌词数据（纯音乐）上，模型可能需要更多步骤才能学会在歌词为零的情况下生成高质量纯音乐。

## 训练

指定环境启动训练：

```bash
simpletuner train env=acestep-training-demo
```

该命令会让 SimpleTuner 在 `config/acestep-training-demo/` 中寻找 `config.json`。

> 💡 **提示（继续训练）：**若要从现有 LoRA（如官方 ACE-Step 检查点或社区适配器）继续微调，使用 `--init_lora` 选项：
> ```bash
> simpletuner train env=acestep-training-demo --init_lora=/path/to/existing_lora.safetensors
> ```

### 训练歌词嵌入器（上游方式）

> ℹ️ **版本说明：** `lyrics_embedder_train` 当前仅适用于 ACE-Step v1 训练路径。SimpleTuner 中的 v1.5 前向兼容 LoRA 路径是 decoder-only。

上游 ACE-Step 训练器会同时微调歌词嵌入器与去噪器。若要在 SimpleTuner 中复现（仅适用于 full 或 standard LoRA）：

- 启用：`lyrics_embedder_train: true`
- 可选覆盖项（否则复用主优化器/调度器）：
  - `lyrics_embedder_lr`
  - `lyrics_embedder_optimizer`
  - `lyrics_embedder_lr_scheduler`

示例片段：

<details>
<summary>查看示例配置</summary>

```json
{
  "lyrics_embedder_train": true,
  "lyrics_embedder_lr": 5e-5,
  "lyrics_embedder_optimizer": "torch-adamw",
  "lyrics_embedder_lr_scheduler": "cosine_with_restarts"
}
```
</details>

嵌入器权重会随 LoRA 检查点一起保存，并在恢复时加载。

## 故障排除

- **验证错误：**确保未使用图像类验证参数（如 `num_validation_images` > 1 或图像指标 CLIP 分数）。
- **内存问题：**如出现 OOM，可降低 `train_batch_size` 或启用 `gradient_checkpointing`。

## 从上游训练器迁移

如果你使用的是原始 ACE-Step 训练脚本，以下是参数映射关系：

| 上游参数 | SimpleTuner `config.json` | 默认值 / 说明 |
| :--- | :--- | :--- |
| `--learning_rate` | `learning_rate` | `1e-4` |
| `--num_workers` | `dataloader_num_workers` | `8` |
| `--max_steps` | `max_train_steps` | `2000000` |
| `--every_n_train_steps` | `checkpointing_steps` | `2000` |
| `--precision` | `mixed_precision` | `"fp16"` 或 `"bf16"`（`"no"` 为 fp32） |
| `--accumulate_grad_batches` | `gradient_accumulation_steps` | `1` |
| `--gradient_clip_val` | `max_grad_norm` | `0.5` |
| `--shift` | `flow_schedule_shift` | `3.0`（ACE-Step 专用） |

### 转换原始数据

如果你有原始音频/文本/歌词文件，并希望使用 Hugging Face 数据集格式（上游 `convert2hf_dataset.py` 工具），可直接在 SimpleTuner 中使用转换后的数据集。

上游转换器会生成带 `tags` 和 `norm_lyrics` 列的数据集。使用方式如下：

<details>
<summary>查看示例配置</summary>

```json
{
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "path/to/converted/dataset",
    "config": {
        "audio_caption_fields": ["tags"],
        "lyrics_column": "norm_lyrics"
    }
}
```
</details>
