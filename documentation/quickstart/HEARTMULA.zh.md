# HeartMuLa 快速上手

在本示例中，我们将训练 HeartMuLa oss 3B 音频生成模型。

## 概览

HeartMuLa 是一个 3B 参数的自回归 Transformer，可根据标签和歌词预测离散的音频 token。生成的 token 由 HeartCodec 解码为波形。

## 硬件要求

HeartMuLa 是 3B 参数模型，相比 Flux 等大型图像生成模型要轻量得多。

- **最低:** 12GB+ 显存的 NVIDIA GPU（例如 3060、4070）。
- **推荐:** 24GB+ 显存的 NVIDIA GPU（例如 3090、4090、A10G）以支持更大批量。
- **Mac:** Apple Silicon 上支持 MPS（需要约 36GB+ 统一内存）。

### 存储要求

> ⚠️ **Token 数据集提示:** HeartMuLa 训练使用预先计算的音频 token。SimpleTuner 不会在训练期间生成 token，因此你的数据集必须提供 `audio_tokens` 或 `audio_tokens_path` 元数据。Token 文件可能较大，请预留足够磁盘空间。

> 💡 **提示:** 使用 `int8-quanto` 量化可以在更低显存的 GPU（例如 12GB-16GB）上训练，同时尽量减少质量损失。

## 前置条件

请确保你有可用的 Python 3.10+ 环境。

```bash
pip install simpletuner
```

## 配置

建议将配置集中管理。我们将为此示例创建一个专用文件夹。

```bash
mkdir -p config/heartmula-training-demo
```

### 关键设置

创建 `config/heartmula-training-demo/config.json` 并写入以下内容:

<details>
<summary>查看示例配置</summary>

```json
{
  "model_family": "heartmula",
  "model_type": "lora",
  "model_flavour": "3b",
  "pretrained_model_name_or_path": "HeartMuLa/HeartMuLa-oss-3B",
  "resolution": 0,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "data_backend_config": "config/heartmula-training-demo/multidatabackend.json"
}
```
</details>

### 验证设置

将以下内容加入 `config.json` 以监控训练进度:

- **`validation_prompt`**: 标签或音频描述（例如“明亮合成器的轻快流行”）。
- **`validation_lyrics`**: （可选）用于演唱的歌词。纯器乐可使用空字符串。
- **`validation_audio_duration`**: 验证音频时长（秒，默认 30.0）。
- **`validation_guidance`**: 引导强度（建议从 1.5 - 3.0 开始）。
- **`validation_step_interval`**: 生成样本的频率（例如每 100 步）。

### 高级实验特性

<details>
<summary>显示高级实验细节</summary>


SimpleTuner 包含一些实验特性，可显著提升训练稳定性和性能。

*   **[Scheduled Sampling (Rollout)](../experimental/SCHEDULED_SAMPLING.md):** 通过让模型在训练中生成自身输入来减少曝光偏差并提升输出质量。

> ⚠️ 这些特性会增加训练开销。

</details>

## 数据集配置

HeartMuLa 需要包含预计算 token 的 **音频专用** 数据集。

每条样本必须提供:

- `tags`（字符串）
- `lyrics`（字符串，可为空）
- `audio_tokens` 或 `audio_tokens_path`

Token 数组必须是 2D，形状为 `[frames, num_codebooks]` 或 `[num_codebooks, frames]`。

> 💡 **注意:** HeartMuLa 不使用单独的文本编码器，因此不需要 text-embeds 后端。

### 选项 1: Hugging Face 数据集（列中包含 token）

创建 `config/heartmula-training-demo/multidatabackend.json`:

<details>
<summary>查看示例配置</summary>

```json
[
  {
    "id": "heartmula-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "your-org/heartmula-audio-tokens",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "config": {
      "audio_caption_fields": ["tags"],
      "lyrics_column": "lyrics"
    }
  }
]
```
</details>

请确保数据集中包含 `audio_tokens` 或 `audio_tokens_path` 列以及文本字段。

### 选项 2: 本地音频文件 + Token 元数据

创建 `config/heartmula-training-demo/multidatabackend.json`:

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
  }
]
```
</details>

请确保元数据后端能为每条样本提供 `audio_tokens` 或 `audio_tokens_path`。

### 数据结构

将音频文件放在 `datasets/my_audio_files` 中。SimpleTuner 支持多种格式:

- **无损:** `.wav`, `.flac`, `.aiff`, `.alac`
- **有损:** `.mp3`, `.ogg`, `.m4a`, `.aac`, `.wma`, `.opus`

> ℹ️ **注意:** 若要支持 MP3、AAC、WMA 等格式，请确保系统已安装 **FFmpeg**。

如果使用 `caption_strategy: textfile`，请将对应的标签和歌词文本文件放在音频旁边:

- **音频:** `track_01.wav`
- **标签（Prompt）:** `track_01.txt`（如“慢速爵士抒情曲”）
- **歌词（可选）:** `track_01.lyrics`

通过元数据提供 token 数组（例如 `audio_tokens_path` 指向 `.npy` 或 `.npz` 文件）。

<details>
<summary>示例数据集结构</summary>

```text
datasets/my_audio_files/
├── track_01.wav
├── track_01.txt
├── track_01.lyrics
└── track_01.tokens.npy
```
</details>

> ⚠️ **歌词说明:** HeartMuLa 需要每条样本都有歌词字符串。纯器乐请使用空字符串，不要省略该字段。

## 训练

指定环境并启动训练:

```bash
simpletuner train env=heartmula-training-demo
```

该命令会在 `config/heartmula-training-demo/` 下查找 `config.json`。

> 💡 **提示（继续训练）:** 如需从已有 LoRA 继续微调，请使用 `--init_lora`:
> ```bash
> simpletuner train env=heartmula-training-demo --init_lora=/path/to/existing_lora.safetensors
> ```

## 故障排除

- **验证错误:** 请勿使用以图像为中心的验证功能，如 `num_validation_images` > 1（在音频中对应批量大小）或图像指标（CLIP 分数）。
- **内存问题:** 若出现 OOM，请减少 `train_batch_size` 或启用 `gradient_checkpointing`。
