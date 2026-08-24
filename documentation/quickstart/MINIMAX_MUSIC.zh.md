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
- 不要让 cropped-window training 把每个裁剪窗口都当作已结束片段来教。如果输出反复在 crop boundary 处 fade out 或收束，请检查 crop label 和 target：内部窗口应该按内部窗口监督，end-of-audio 行为只应在真实歌曲结尾处教授。
- 对歌曲结构训练，请使用 `minimax_music_lm_window_mode: "continuation"`。它会采样目标窗口，保留从曲目开头到该窗口的所有音频 token 作为因果上下文，并屏蔽此前上下文的损失。它比孤立的随机裁剪使用更多显存，但能避免把每个片段都当作歌曲开头来教。
- 在小型 LM 音频数据集上要谨慎使用激进 optimizer。Prodigy 在较高 learning rate 下可能严重越界，Lion 可能在前一千步内过度适配；先用 AdamW 作为 baseline，再测试更快的 optimizer。
- **先验保持**：添加第二个音频后端并设置 `is_regularisation_data: true`，其中包含纯音乐或无关歌曲（允许空歌词）。在这些批次上，损失以冻结基础模型自身的下一 token 分布为目标，而不是真实码，因此 LoRA 保持外科手术式的精准：regularisation caption 仍然会像基础模型那样预测，大幅减少风格渗漏。

### 如何配置风格和歌手数据集

音乐风格适配和歌手身份适配需要不同的数据集设计。不要把歌手名当作详细音乐 caption 的替代品。

#### 音乐风格

音乐风格相对宽容。如果目标是流派、编曲或制作风格，而不是某个具体声线，24 首以上的多样化曲目就可能足以训练出有用的 adapter。

- 在不偏离目标风格的前提下优化多样性。包含推理时用户可能会请求的 tempo、乐器组合、制作选择、情绪和相邻子流派。
- 为每个音频样本提供多个完整的风格 caption。单独的 trigger word 会把数据集压缩成平均关联，不能教会模型复现其范围所需的控制。
- 将 vocal timbre 视为附带信息。使用多个 vocalist 或 instrumental material，避免某个声音意外成为学习到的风格的一部分。
- 用固定 validation prompt 和多个 checkpoint strength 观察 collapse。Style adapter 往往在需要大量 optimizer step 之前就已经有用。

使用 `caption_strategy: "textfile"` 且保持默认 `disable_multiline_split: false` 时，`.txt` sidecar 中每个非空行都是单独的 caption candidate。SimpleTuner 每次采样该音频条目时会选择一个 candidate；它不会把所有行合并为一个 grouped caption。DiT workflow 会分别缓存每个不同 caption，而 LM training 会在线 tokenize 被选中的 caption，不使用 text-embedding cache。例如：

```text
syncopated art rock, dry drums, angular guitar, abrupt dynamic changes
melodic alternative metal, layered harmonies, restless bass, theatrical pacing
tense progressive rock, odd-meter accents, sparse verse, explosive refrain
```

这是 caption augmentation，不是 multiline prompt：对于某个训练样本，模型只会看到其中一行。

#### 歌手身份

歌手身份宽容度低得多。每个歌手构建一个 adapter，并移除所有包含其他 vocalist 的 track 或 section，包括 duet、交替 verse、backing lead 和 guest appearance。`[Verse: ...]` 或 `[Chorus: ...]` 这样的歌词标签不能可靠地解开混合声音。

- 在每个 caption candidate 中放入同一个唯一 singer trigger，后面接完整且多样的风格描述。把 trigger 放在一行、描述放在其他行是错误的，因为每次只会选择一行。
- 狭窄的单一流派歌手数据集通常学到的是该编曲中的歌手，而不是可迁移的歌手身份。Identity delta 会与总是共同出现的 genre、instrumentation、mix 和 song structure 纠缠，因此 trigger 可能只在域内有效。跨流派 vocal control 需要歌手数据集中有实际的 genre 和 arrangement 多样性。
- 保持 lyrics 忠实，但不要依靠歌词 section label 来教授身份。真正有用的信号来自 audio 和 caption 的关联。
- 对非常小的 corpus，instrumental counterpart 可以提供 prior preservation。六首仔细隔离的 vocal track 在与这些 track 构建出的 regularisation 配对时可能可用。

```text
vocalist_xyz, sparse alternative rock, dry drums, tense verse, explosive refrain
vocalist_xyz, melodic art metal, layered guitar, mid-tempo groove, close vocal
vocalist_xyz, acoustic chamber rock, hand percussion, soft opening, dramatic lift
```

一种实用的 regularisation workflow 使用 Demucs 去除人声：

```bash
python -m demucs --two-stems=vocals path/to/track.wav
```

将每个生成的 `no_vocals.wav` 放入单独的 audio backend，配上只描述风格的 `.txt` caption，不包含 singer trigger，并放置内容为 `[Instrumental]` 的 `.lyrics` sidecar。在该 backend 上设置 `is_regularisation_data: true`。Regularisation batch 的目标是冻结的 base planner，帮助 adapter 区分“这种音乐”和“这个歌手”，而不是围绕很小的 vocal corpus 重写整个风格。

对于更大且多样的单歌手 corpus，先不要添加这个 regularisation branch；只有当 validation 显示 style bleed 或 base-model damage 时再加入。若 vocal dataset 已经提供足够覆盖，regularisation 可能会减慢身份学习。一个合理解释是额外的 preservation signal 会进一步稀释已经多样的 identity gradient，但请把它当作调参假设，而不是通用规则。

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
