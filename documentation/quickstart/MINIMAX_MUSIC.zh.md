# MiniMax Music 3 快速开始

本指南说明如何在 SimpleTuner 中配置 MiniMax Music 3 LoRA 训练。

## 概览

MiniMax Music 3 是基于描述和歌词条件的音乐生成模型。Diffusers 布局使用 Qwen3 自回归语言模型生成文本/音频条件，使用 flow-matching transformer 训练 128 通道 DAV latent，并通过 decoder/vocoder 生成验证音频。

SimpleTuner 支持：

- MiniMax Music 3 transformer 的 LoRA、LyCORIS 和 full-rank 训练
- 通过原始 `dav.pth` autoencoder 让 VAECache 从原始音频编码 latent
- 从音频数据集 metadata 读取 caption、lyrics 和 duration
- 使用 `validation_prompt`、`validation_lyrics`、`validation_audio_duration` 和 prompt library 生成验证音频
- 使用 `lora_format: "comfyui"` 导入/导出 ComfyUI MiniMax Music LoRA
- AnyFlow、TwinFlow、CREPA self-flow 和 LayerSync

## 硬件要求

MiniMax Music 3 包含 2.4B flow transformer 和 8B Qwen3 AR 文本/音频条件模型。

- **最低：** 24GB+ VRAM 的 NVIDIA GPU，用于保守 LoRA 训练。
- **推荐：** 48GB+ VRAM，或为更高 rank、更长音频和频繁验证使用 CPU/RAM offload。
- **Mac：** MPS 可能支持部分组件，但训练和验证的实际目标仍是 CUDA。

建议从 `base_model_precision: "int8-quanto"`、`text_encoder_1_precision: "int8-quanto"` 和 `gradient_checkpointing: true` 开始。如果 text encoder 仍然占用过多显存，先使用 text encoder offload，再提高 LoRA rank。

## 前置条件

安装 SimpleTuner，并安装 FFmpeg 以便加载音频：

```bash
pip install simpletuner
```

手动安装或开发环境设置见[安装文档](../INSTALL.md)。

## 配置

创建专用配置目录：

```bash
mkdir -p config/minimaxmusic-training-demo
```

创建 `config/minimaxmusic-training-demo/config.json`：

<details>
<summary>查看示例配置</summary>

```json
{
  "model_family": "minimaxmusic",
  "model_type": "lora",
  "model_flavour": "music3",
  "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-Music3",
  "pretrained_vae_model_name_or_path": "SimpleTuner/MiniMax-Music-3-Encoder",
  "resolution": 512,
  "mixed_precision": "bf16",
  "base_model_precision": "int8-quanto",
  "text_encoder_1_precision": "int8-quanto",
  "gradient_checkpointing": true,
  "lora_rank": 64,
  "lora_format": "comfyui",
  "optimizer": "adamw_bf16",
  "learning_rate": 0.00005,
  "train_batch_size": 1,
  "vae_batch_size": 1,
  "data_backend_config": "config/minimaxmusic-training-demo/multidatabackend.json",
  "validation_prompt": "bright synth pop with clean vocal melody and crisp percussion",
  "validation_lyrics": "[verse]\nturning sparks into a skyline\n[chorus]\nwe keep singing through the night",
  "validation_audio_duration": 30,
  "validation_guidance": 1.7,
  "validation_num_inference_steps": 30,
  "validation_steps": 50,
  "validation_disable_unconditional": true
}
```
</details>

可用模板：

- `simpletuner/examples/minimaxmusic-music3.peft-lora`
- `simpletuner/examples/minimaxmusic-audio.json`
- `simpletuner/examples/minimaxmusic-prompts.json`

运行示例：

```bash
simpletuner train example=minimaxmusic-music3.peft-lora
```

## VAECache

MiniMax Music 3 的原始音频缓存使用 DAV audio autoencoder。推荐使用 SimpleTuner VAE 仓库 `SimpleTuner/MiniMax-Music-3-Encoder`，其中转换后的组件位于 `audio_vae/` 子目录，可按 Diffusers 风格加载。

上游 `MiniMaxAI/MiniMax-Music3` 仓库也包含原始 `dav.pth`，SimpleTuner 可以直接加载。如果使用本地转换后的 Diffusers 目录，请把 `dav.pth` 放在 checkpoint 根目录，或将 `pretrained_vae_model_name_or_path` 指向包含 `dav.pth` 或 `audio_vae/` 子目录的位置。仅有 `vocoder/` 子目录可以用于验证解码，但不能用于原始音频 VAE 缓存。

## 数据集配置

MiniMax Music 3 需要一个 **audio** 数据集和一个 **text embeds** cache backend。

如果目标人声身份和某个狭窄风格或流派绑定太强，可使用 [Voice Cloning Data Transforms](../experimental/VOICE_CLONING.zh.md) 中的 RVC `data_transforms` workflow，在 VAE 和 text cache 前生成扩展 audio split。

```json
[
  {
    "id": "minimaxmusic-demo-data",
    "type": "huggingface",
    "dataset_type": "audio",
    "dataset_name": "Yi3852/ACEStep-Songs",
    "metadata_backend": "huggingface",
    "caption_strategy": "huggingface",
    "audio": {
      "bucket_strategy": "duration",
      "duration_interval": 3.0,
      "max_duration_seconds": 30
    },
    "cache_dir_vae": "cache/vae/{model_family}/minimaxmusic-demo-data"
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

本地音频可以使用 `.txt` 描述文件和 `.lyrics` 歌词文件：

```text
datasets/minimaxmusic-audio/
├── track_01.wav
├── track_01.txt
└── track_01.lyrics
```

## 验证设置

- **`validation_prompt`**：音乐描述或 tags。
- **`validation_lyrics`**：演唱歌词；纯音乐可使用空字符串。
- **`validation_audio_duration`**：生成片段时长，单位为秒。
- **`validation_guidance`**：CFG scale，建议从 `1.5` 到 `2.0` 开始。
- **`validation_num_inference_steps`**：验证采样步数，建议从 `30` 左右开始。
- **`validation_steps`**：每隔多少 step 生成验证音频。
- **`validation_prompt_library`**：使用 `"audio"` 选择内置音乐 caption + lyrics 库。
- **`user_prompt_library`**：JSON 库路径。条目可使用 `prompt` 或 `caption`，并可选提供多行 `lyrics`。

## 训练

```bash
simpletuner train env=minimaxmusic-training-demo
```

从已有 MiniMax Music 3 LoRA 开始：

```bash
simpletuner train env=minimaxmusic-training-demo --init_lora=/path/to/adapter.safetensors --init_lora_step=0
```

如果 adapter 是原生 ComfyUI 格式，请在配置中保留 `lora_format: "comfyui"`。SimpleTuner 会在训练时转换，并以相同格式导出。

## 高级功能

MiniMax Music 3 使用 SimpleTuner 的 flow-matching 训练路径，因此可使用 AnyFlow、TwinFlow、CREPA self-flow 和 LayerSync。建议先使用标准 LoRA，再逐个启用高级功能。

## 语言模型（AR 阶段）训练

规划 MiniMax Music 3 语义码的 Qwen3 语言模型可以代替音乐 DiT 进行训练——适用于 dreambooth 式触发词，将某种音乐风格绑定到一个关键词。

请参阅 [fiona crapple](https://huggingface.co/terminusresearch/minimax-music3-lm-lora-fiona-crapple)：这是使用此模式完成的 LM LoRA 训练示例，包含训练设置、检查点和音频对比。

```json
{
  "minimax_music_train_component": "language_model",
  "minimax_music_lm_max_frames": 0,
  "minimax_music_lm_window_mode": "prefix"
}
```

要求以及与 DiT 训练的区别：

- 每个数据集样本必须提供 `prompt`（或 `tags`）、`lyrics`，以及指向 `.pt` 文件的 `audio_tokens_path` 元数据，该文件包含形状为 `[frames, codebooks]` 的原始逐码本 RVQ 码（语义码 `< 16384`，残差码 `< audio_vocab_size`，不含词表偏移）。请使用专用 `minimax-music3-latent-replanner` 仓库中的 `precompute_rvq_codes.py --raw-codes` 导出。
- 损失是语义码本上的下一 token 交叉熵，仅作用于音频位置；RVQ depth decoder 保持冻结，并提供残差码输入嵌入。
- 仅支持标准 PEFT LoRA，`lora_format: "comfyui"` 会被拒绝。检查点保存带 `language_model.` 前缀键的 `pytorch_lora_weights.safetensors`。
- 此模式下训练器内验证音频被禁用；请使用标准生成栈从保存的检查点渲染。
- 此模式下不进行 VAE 或文本嵌入缓存——训练直接读取 token，因此 `cache_dir_vae` 和文本嵌入后端不会被使用。
- 将触发关键词（例如 `"fiona crapple"`）放入每个样本的 caption/`prompt` 字段；歌词保持原样。
- 对较短的帧上限运行，设置 `minimax_music_lm_window_mode: "random"` 可采样带位置的 RVQ 窗口，而不是总是训练前奏。随机窗口会把开始/结束/时长加入 prompt，并省略完整歌词，除非样本提供 `lyrics_window`。
- 对歌曲结构训练，请使用 `minimax_music_lm_window_mode: "continuation"`。末尾的 `minimax_music_lm_target_frames` 接受损失，更早的可见帧作为被屏蔽的因果上下文。`full` 裁剪总是从歌曲开头开始；`random` 可在曲目内移动，同时在目标前保留至少一个原生 128 帧上下文分段。时长按原生 128 帧/5.12 秒间隔对齐；最大值 `0` 使用可用曲目长度。

带显存上限的 full-prefix continuation 配置示例：

```json
{
  "minimax_music_lm_window_mode": "continuation",
  "minimax_music_lm_target_frames": 128,
  "minimax_music_lm_continuation_crop_mode": "full",
  "minimax_music_lm_min_duration_seconds": 5.12,
  "minimax_music_lm_max_duration_seconds": 30.72
}
```

把 crop mode 改为 `random`，即可在相同显存上限下训练定位 continuation。定位片段会把时间范围加入 prompt；没有 `lyrics_window` 时会省略完整歌词。当终止和非终止片段都可用时，固定 25% 的样本会到达真实曲目末尾，使 EOS 监督不依赖曲目长度。采样发生在完整缓存 RVQ 序列的 LM collate 阶段，不会修改数据集音频或缓存。
- **先验保持**：添加第二个音频后端并设置 `is_regularisation_data: true`，其中包含无关歌曲（允许空歌词）。在这些批次上，损失以冻结基础模型自身的下一 token 分布为目标，而不是真实码，因此 LoRA 保持外科手术式的精准：无关的 caption 仍然会像基础模型那样预测，大幅减少风格渗漏。

## 故障排查

- **`VAE caching requires the original dav.pth checkpoint`**：使用 `SimpleTuner/MiniMax-Music-3-Encoder` 或 `MiniMaxAI/MiniMax-Music3`，把 `dav.pth` 放在本地 checkpoint 根目录，或将 `pretrained_vae_model_name_or_path` 指向包含它的位置。
- **歌词缺失**：确认 backend metadata 包含 `lyrics`，或使用 `caption_strategy: "textfile"` 时在音频旁边放置 `.lyrics` sidecar。
- **Text embedding 或 validation OOM**：降低 validation duration，使用 int8 text encoder precision，或启用 text encoder offload。

## 相关 MiniMax Music 3 实验

- [开放 RVQ 编码器](https://huggingface.co/SimpleTuner/open-rvq-encoder-minimax-music3)
- [RVQ 参考音频集成](https://github.com/bghira/minimax-music3-rvq-reference-audio)
- [Fiona Crapple LM LoRA](https://huggingface.co/terminusresearch/minimax-music3-lm-lora-fiona-crapple)
- [Latent refiner](https://github.com/bghira/minimax-music3-latent-refiner) 和 [v0.10 权重](https://huggingface.co/terminusresearch/minimax-music3-latent-refiner-v0.10)
- [Latent replanner](https://github.com/bghira/minimax-music3-latent-replanner) 和 [实验记录](https://huggingface.co/terminusresearch/minimax-music3-replanner-experiment)
