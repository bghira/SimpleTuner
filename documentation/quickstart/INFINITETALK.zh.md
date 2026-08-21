# InfiniteTalk 快速入门

InfiniteTalk 是基于 Wan 2.1 I2V 14B 的音频驱动视频模型。SimpleTuner 先加载 Wan 基础模型，再叠加官方音频投影器和 40 个块中的音频注意力。

此集成训练官方单说话人模型。多说话人模式需要多路同步音频和说话人掩码；当前数据加载器每个视频只表示一路音频。

## 要求

- 支持 bf16 的 NVIDIA GPU
- 64 GB 内存；RamTorch 或非量化加载建议 96 GB 以上
- `ffmpeg`
- 25 fps 且音频对齐的视频

```bash
python -m venv .venv
source .venv/bin/activate
pip install 'simpletuner[cuda]'
```

示例通过 `trust_remote_code: true` 授权加载固定版本的 `kernels-community/flash-attn3` Hub 内核。改用本地或内置后端时请删除该选项。

## 起始配置

| 显存 | 帧数 | 权重 | 驻留方式 | 示例 |
| --- | ---: | --- | --- | --- |
| 24 GB | 17 | bf16 | RamTorch 流式加载全部块 | `infinitetalk-14b-480p-24gb.peft-lora` |
| 32 GB | 17 | int8 TorchAO | 交换 20 个块 | `infinitetalk-14b-480p-32gb.peft-lora` |
| 48 GB | 33 | bf16 | 交换 24 个块 | `infinitetalk-14b-480p-48gb.peft-lora` |
| 80 GB | 49 | bf16 | 常驻 | `infinitetalk-14b-480p-80gb.peft-lora` |

## 数据

将说明文本放在视频旁，例如 `clip-001.mp4` 和 `clip-001.txt`。附带配置会自动提取 16 kHz 单声道音频：

```json
"audio": {"auto_split": true, "sample_rate": 16000, "channels": 1}
```

- 使用 25 fps。
- 帧数必须为 `4k + 1`，例如 17、33、49。
- 音频必须覆盖所选视频片段的同一时间区间。
- 不要把随机时间裁剪与完整音轨配对。
- 无音频片段会被拒绝。

## 训练

```bash
simpletuner train \
  --config simpletuner/examples/infinitetalk-14b-480p-80gb.peft-lora/config.json
```

```json
{
  "model_family": "infinitetalk",
  "model_flavour": "single-14b-480p",
  "pretrained_model_name_or_path": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
  "framerate": 25
}
```

降低显存的顺序：减少帧数、增加 `musubi_blocks_to_swap`、使用 int8 TorchAO、最后使用 RamTorch。音频注意力依赖精确帧边界，因此不支持 TREAD 和上下文并行。

验证需要图像和音频。内置验证只做文本 CFG，并在两个分支中保留音频；独立文本/音频 CFG 请使用官方实现。

支持 LoRA、LyCORIS、全量训练、适配器量化、梯度检查点、块交换、RamTorch、FFN 分块、CREPA 和 LayerSync。不支持多说话人训练。

来源：[代码](https://github.com/MeiGen-AI/InfiniteTalk)、[报告](https://arxiv.org/abs/2508.14033)、[权重](https://huggingface.co/MeiGen-AI/InfiniteTalk)。
