# SimpleTuner 训练脚本选项

## 概览

本指南以易读方式介绍 SimpleTuner `train.py` 中可用的命令行选项。这些选项提供高度可定制性，使你能按需求训练模型。

### JSON 配置文件格式

默认 JSON 文件名为 `config.json`，键名与下方 `--arguments` 一致。JSON 中不需要前导 `--`，但也可以保留。

需要可直接运行的示例？请查看 [simpletuner/examples/README.md](/simpletuner/examples/README.md) 中的预置。

### 简易配置脚本（***推荐***）

可使用 `simpletuner configure` 命令生成包含理想默认值的 `config.json`。

#### 修改现有配置

`configure` 可接收一个兼容的 `config.json`，以交互方式修改训练设置：

```bash
simpletuner configure config/foo/config.json
```

其中 `foo` 为你的配置环境；若未使用环境，直接用 `config/config.json`。

<img width="1484" height="560" alt="image" src="https://github.com/user-attachments/assets/67dec8d8-3e41-42df-96e6-f95892d2814c" />

> ⚠️ 若所在地区无法方便访问 Hugging Face Hub，请在 `~/.bashrc` 或 `~/.zshrc`（取决于你的 `$SHELL`）中添加 `HF_ENDPOINT=https://hf-mirror.com`。

---

## 🌟 核心模型配置

### `--model_type`

- **内容**：选择 LoRA 或全量微调。
- **选项**：lora, full.
- **默认**：lora
  - 若使用 lora，`--lora_type` 决定使用 PEFT 还是 LyCORIS。部分模型（PixArt）仅支持 LyCORIS 适配器。

### `--model_family`

- **内容**：指定训练的模型架构。
- **选项**：pixart_sigma, flux, sd3, sdxl, kolors, legacy

### `--lora_format`

- **内容**：选择 LoRA 检查点的加载/保存键格式。
- **选项**：`diffusers`（默认），`comfyui`
- **说明**：
  - `diffusers` 为标准 PEFT/Diffusers 格式。
  - `comfyui` 会转换为 ComfyUI 风格键（`diffusion_model.*`，含 `lora_A/lora_B` 与 `.alpha` 张量）。Flux、Flux2、Lumina2、Z-Image 即便保持 `diffusers` 也会自动识别 ComfyUI 输入，但若希望保存时强制 ComfyUI 输出，请设为 `comfyui`。

### `--fuse_qkv_projections`

- **内容**：融合注意力块中的 QKV 投影，提高硬件效率。
- **说明**：仅在手动安装 Flash Attention 3 的 NVIDIA H100 或 H200 上可用。

### `--offload_during_startup`

- **内容**：在 VAE 缓存期间将文本编码器权重卸载到 CPU。
- **原因**：对 HiDream、Wan 2.1 等大模型，加载 VAE 缓存时可能 OOM。该选项不影响训练质量，但在超大文本编码器或慢 CPU 下可能显著延长启动时间，故默认关闭。
- **提示**：对内存极其紧张的系统，可配合下方分组卸载功能使用。

### `--offload_during_save`

- **内容**：在 `save_hooks.py` 准备检查点时，将整条管线临时迁移到 CPU，确保 FP8/量化权重在设备外写入。
- **原因**：保存 fp8-quanto 可能导致显存突增（例如 `state_dict()` 序列化时）。该选项在训练期间保持模型在加速器上，仅在触发保存时短暂卸载以避免 CUDA OOM。
- **提示**：仅在保存因 OOM 失败时启用；保存后模型会自动迁回以继续训练。

### `--delete_model_after_load`

- **内容**：模型加载进内存后，从 Hugging Face 缓存中删除模型文件。
- **原因**：减少按 GB 计费的存储开销。模型加载到 VRAM/RAM 后，磁盘缓存直到下次运行前都不需要。以牺牲后续运行的网络带宽来换取存储节省。
- **说明**：
  - 启用验证时不会删除 VAE，因为生成验证图像需要它。
  - 文本编码器在数据后端工厂完成启动（嵌入缓存后）被删除。
  - Transformer/UNet 在加载后立即删除。
  - 多节点环境下，仅每个节点的 local-rank 0 执行删除。为处理共享网络存储中的竞态，删除失败会被静默忽略。
  - 不影响训练检查点，仅影响预训练基础模型缓存。

### `--enable_group_offload`

- **内容**：启用 diffusers 的分组模块卸载，使模型块在前向之间驻留在 CPU（或磁盘）。
- **原因**：在大 Transformer（Flux、Wan、Auraflow、LTXVideo、Cosmos2Image）上显著降低显存峰值；配合 CUDA streams 时性能影响较小。
- **说明**：
  - 与 `--enable_model_cpu_offload` 互斥，每次运行只能选择一种策略。
  - 需要 diffusers **v0.33.0** 或更新版本。

### `--group_offload_type`

- **选项**：`block_level`（默认）、`leaf_level`
- **内容**：控制层的分组方式。`block_level` 在显存与吞吐之间取得平衡；`leaf_level` 最大化节省，但 CPU 传输更多。

### `--group_offload_blocks_per_group`

- **内容**：使用 `block_level` 时，每组包含的 Transformer 块数量。
- **默认**：`1`
- **原因**：增加该值可减少传输频率（更快），但会在加速器上保留更多参数（占用更多显存）。

### `--group_offload_use_stream`

- **内容**：使用专用 CUDA 流将主机/设备传输与计算重叠。
- **默认**：`False`
- **说明**：
  - 在非 CUDA 后端（Apple MPS、ROCm、CPU）自动回退为 CPU 风格传输。
  - 在 NVIDIA GPU 上有空闲拷贝引擎时推荐开启。

### `--group_offload_to_disk_path`

- **内容**：将分组参数溢出到磁盘而非 RAM 的目录路径。
- **原因**：适用于 CPU RAM 极为紧张的系统（例如带大容量 NVMe 的工作站）。
- **提示**：请使用高速本地 SSD；网络文件系统会显著拖慢训练。

### `--musubi_blocks_to_swap`

- **内容**：为 LongCat-Video、Wan、LTXVideo、Kandinsky5-Video、Qwen-Image、Flux、Flux.2、Cosmos2Image、HunyuanVideo 提供 Musubi 块交换。将最后 N 个 Transformer 块保留在 CPU，并在前向中按块流式加载权重。
- **默认**：`0`（禁用）
- **说明**：Musubi 风格权重卸载，会降低吞吐以换取显存节省，且在启用梯度时会跳过。
- **说明**：Musubi 风格权重卸载，通过降低吞吐换取显存节省；启用梯度时会跳过。

### `--musubi_block_swap_device`

- **内容**：存放交换的 Transformer 块的设备字符串（如 `cpu`、`cuda:0`）。
- **默认**：`cpu`
- **说明**：仅当 `--musubi_blocks_to_swap` > 0 时生效。

### `--ramtorch`

- **内容**：将 `nn.Linear` 层替换为 RamTorch 的 CPU 流式实现。
- **原因**：在 CPU 内存共享 Linear 权重并流式传到加速器，以降低显存压力。
- **说明**：
  - 需要 CUDA 或 ROCm（不支持 Apple/MPS）。
  - 与 `--enable_group_offload` 互斥。
  - 自动启用 `--set_grads_to_none`。

### `--ramtorch_target_modules`

- **内容**：逗号分隔的 glob 模式，用于限制哪些 Linear 模块转换为 RamTorch。
- **默认**：未提供模式时转换所有 Linear 层。
- **说明**：
  - 使用 `fnmatch` glob 语法匹配完整模块名或类名。
  - 模式必须包含末尾的 `.*` 通配符才能匹配块内的层。例如，`transformer_blocks.0.*` 匹配块 0 内的所有层，而 `transformer_blocks.*` 匹配所有 transformer 块。不带 `.*` 的裸名称如 `transformer_blocks.0` 也可以工作（会自动展开），但建议使用明确的通配符形式以提高清晰度。
  - 示例：`"transformer_blocks.*,single_transformer_blocks.0.*,single_transformer_blocks.1.*"`

### `--ramtorch_text_encoder`

- **内容**：对所有文本编码器 Linear 层应用 RamTorch 替换。
- **默认**：`False`

### `--ramtorch_vae`

- **内容**：仅对 VAE 的 mid-block Linear 层进行 RamTorch 转换（实验性）。
- **默认**：`False`
- **说明**：VAE 的上下采样卷积块保持不变。

### `--ramtorch_controlnet`

- **内容**：训练 ControlNet 时对其 Linear 层应用 RamTorch 替换。
- **默认**：`False`

### `--ramtorch_transformer_percent`

- **内容**：使用 RamTorch 卸载的 transformer Linear 层的百分比（0-100）。
- **默认**：`100`（所有符合条件的层）
- **原因**：允许部分卸载以平衡显存节省与性能。较低的值保留更多层在 GPU 上以加快训练，同时仍减少内存使用。
- **说明**：层按模块遍历顺序从头开始选择。可与 `--ramtorch_target_modules` 结合使用。

### `--ramtorch_text_encoder_percent`

- **内容**：使用 RamTorch 卸载的文本编码器 Linear 层的百分比（0-100）。
- **默认**：`100`（所有符合条件的层）
- **原因**：当启用 `--ramtorch_text_encoder` 时，允许部分卸载文本编码器。
- **说明**：仅在启用 `--ramtorch_text_encoder` 时适用。

### `--ramtorch_disable_sync_hooks`

- **内容**：禁用 RamTorch 层后添加的 CUDA 同步钩子。
- **默认**：`False`（同步钩子已启用）
- **原因**：同步钩子修复了 RamTorch 乒乓缓冲系统中可能导致非确定性输出的竞争条件。禁用可能提高性能，但有结果不正确的风险。
- **说明**：仅在同步钩子出现问题或想要测试时才禁用。

### `--ramtorch_disable_extensions`

- **内容**：仅对 Linear 层应用 RamTorch，跳过 Embedding/RMSNorm/LayerNorm/Conv。
- **默认**：`True`（扩展已禁用）
- **原因**：SimpleTuner 将 RamTorch 扩展到 Linear 层之外，包括 Embedding、RMSNorm、LayerNorm 和 Conv 层。使用此选项禁用这些扩展，仅卸载 Linear 层。
- **说明**：可能减少显存节省，但有助于调试扩展层类型的问题。

### `--pretrained_model_name_or_path`

- **内容**：预训练模型路径或 <https://huggingface.co/models> 上的标识符。
- **原因**：指定训练起点的基础模型。可用 `--revision` 与 `--variant` 选择仓库中的特定版本。也支持 SDXL、Flux、SD3.x 的单文件 `.safetensors` 路径。

### `--pretrained_t5_model_name_or_path`

- **内容**：预训练 T5 模型路径或 <https://huggingface.co/models> 上的标识符。
- **原因**：训练 PixArt 时，可指定 T5 权重来源，以避免切换基础模型时重复下载。

### `--pretrained_gemma_model_name_or_path`

- **内容**：预训练 Gemma 模型路径或 <https://huggingface.co/models> 上的标识符。
- **原因**：训练 Gemma 系模型（例如 LTX-2、Sana、Lumina2）时，可单独指定 Gemma 权重来源，而无需更换基础扩散模型路径。

### `--custom_text_encoder_intermediary_layers`

- **内容**：覆盖 FLUX.2 模型中从文本编码器提取的隐藏状态层。
- **格式**：层索引的 JSON 数组，例如 `[10, 20, 30]`
- **默认值**：未设置时使用模型特定默认值：
  - FLUX.2-dev (Mistral-3)：`[10, 20, 30]`
  - FLUX.2-klein (Qwen3)：`[9, 18, 27]`
- **原因**：允许尝试不同的文本编码器隐藏状态组合，用于自定义对齐或研究目的。
- **注意**：此选项为实验性功能，仅适用于 FLUX.2 模型。更改层索引会使已缓存的文本嵌入失效，需要重新生成。层数必须与模型期望的输入一致（3 层）。

### `--gradient_checkpointing`

- **内容**：训练过程中按层计算并累积梯度，以降低显存峰值，但训练会变慢。

### `--gradient_checkpointing_interval`

- **内容**：每 *n* 个块进行一次 checkpoint，*n* 必须大于 0。1 等同于启用 `--gradient_checkpointing`，2 则每隔一个块进行一次。
- **说明**：目前仅 SDXL 与 Flux 支持此选项。SDXL 使用较为权宜的实现。

### `--gradient_checkpointing_backend`

- **选项**：`torch`、`unsloth`
- **内容**：选择梯度 checkpoint 的实现方式。
  - `torch`（默认）：标准 PyTorch checkpoint，在反向传播时重新计算激活值。约 20% 时间开销。
  - `unsloth`：异步将激活值卸载到 CPU 而非重新计算。额外节省约 30% 显存，仅约 2% 开销。需要较快的 PCIe 带宽。
- **说明**：仅在启用 `--gradient_checkpointing` 时生效。`unsloth` 后端需要 CUDA。

### `--refiner_training`

- **内容**：启用自定义混合专家模型系列训练。详情参见 [Mixture-of-Experts](MIXTURE_OF_EXPERTS.md)。

## 精度

### `--quantize_via`

- **选项**：`cpu`、`accelerator`、`pipeline`
  - `accelerator` 可能稍快，但在 Flux 等大模型上 24G 卡可能 OOM。
  - `cpu` 量化约需 30 秒。（**默认**）
  - `pipeline` 将量化交给 Diffusers，通过 `--quantization_config` 或管线可用的预设（如 `nf4-bnb`、`int8-torchao`、`fp8-torchao`、`int8-quanto`、`.gguf` 检查点）。

### `--base_model_precision`

- **内容**：降低模型精度并用更少内存训练。支持三种量化后端：BitsAndBytes（pipeline）、TorchAO（pipeline 或手动）、Optimum Quanto（pipeline 或手动）。

#### Diffusers 管线预设

- `nf4-bnb` 通过 Diffusers 加载 4-bit NF4 BitsAndBytes 配置（仅 CUDA）。需要 `bitsandbytes` 与支持 BnB 的 diffusers。
- `int4-torchao`、`int8-torchao`、`fp8-torchao` 通过 Diffusers 使用 TorchAoConfig（CUDA）。需要 `torchao` 与较新的 diffusers/transformers。
- `int8-quanto`、`int4-quanto`、`int2-quanto`、`fp8-quanto`、`fp8uz-quanto` 通过 Diffusers 使用 QuantoConfig。Diffusers 会将 FP8-NUZ 映射为 float8 权重；如需 NUZ 变体请使用手动 quanto 量化。
- `.gguf` 检查点会被自动检测，并在可用时使用 `GGUFQuantizationConfig` 加载。需安装较新的 diffusers/transformers 以支持 GGUF。

#### Optimum Quanto

由 Hugging Face 提供的 optimum-quanto 在所有支持平台上表现可靠。

- `int8-quanto` 兼容性最好，通常效果也最佳
  - 在 RTX4090 及其它 GPU 上训练最快
  - CUDA 设备对 int8/int4 使用硬件加速 matmul
    - int4 仍然非常慢
  - 支持 `TRAINING_DYNAMO_BACKEND=inductor`（`torch.compile()`）
- `fp8uz-quanto` 是 CUDA 与 ROCm 的实验性 fp8 变体
  - 在 AMD Instinct 等新架构上支持更好
  - 在 4090 上训练可能略快于 `int8-quanto`，但推理更慢（慢 1 秒）
  - 支持 `TRAINING_DYNAMO_BACKEND=inductor`（`torch.compile()`）
- `fp8-quanto` 目前不使用 fp8 matmul，且在 Apple 系统不可用
  - CUDA/ROCm 仍缺少硬件 fp8 matmul，可能明显慢于 int8
    - 使用 MARLIN 内核进行 fp8 GEMM
  - 与 dynamo 不兼容，尝试组合时会自动禁用 dynamo

#### TorchAO

PyTorch 的新库 AO，可将线性层与 2D 卷积（如 unet 结构）替换为量化版本。
<!-- Additionally, it provides an experimental CPU offload optimiser that essentially provides a simpler reimplementation of DeepSpeed. -->

- `int8-torchao` 将内存占用降低到与 Quanto 各精度级别相当
  - 撰写时在 Apple MPS 上略慢（11s/iter），Quanto 为 9s/iter
  - 不使用 `torch.compile` 时，在 CUDA 上与 `int8-quanto` 速度与内存相当，ROCm 速度未知
  - 使用 `torch.compile` 时慢于 `int8-quanto`
- `fp8-torchao` 仅支持 Hopper（H100/H200）或更新（Blackwell B200）加速器

##### 优化器

TorchAO 提供通用 4bit/8bit 优化器：`ao-adamw8bit`、`ao-adamw4bit`

还提供面向 Hopper（H100 或更高）的优化器：`ao-adamfp8`、`ao-adamwfp8`

#### SDNQ（SD.Next Quantization Engine）

[SDNQ](https://github.com/disty0/sdnq) 是针对训练优化的量化库，支持 AMD（ROCm）、Apple（MPS）、NVIDIA（CUDA）。它提供随机舍入与量化优化器状态，实现高效量化训练。

##### 推荐精度等级

##### 推荐精度等级

**全量微调**（更新模型权重）：
- `uint8-sdnq` - 内存节省与训练质量的最佳平衡
- `uint16-sdnq` - 更高精度，追求最高质量（例如 Stable Cascade）
- `int16-sdnq` - 有符号 16-bit 备选
- `fp16-sdnq` - 量化 FP16，结合 SDNQ 优势的最高精度

**LoRA 训练**（冻结基础模型权重）：
- `int8-sdnq` - 有符号 8-bit，通用选择
- `int6-sdnq`, `int5-sdnq` - 更低精度，更小内存
- `uint5-sdnq`, `uint4-sdnq`, `uint3-sdnq`, `uint2-sdnq` - 激进压缩

**注记：** `int7-sdnq` 可用但不推荐（慢，且比 int8 小不了多少）。

**重要：** 低于 5-bit 精度时，SDNQ 会自动启用 8 步 SVD（奇异值分解）以保持质量。SVD 量化更慢且非确定，因此 Disty0 在 Hugging Face 提供预量化的 SVD 模型。SVD 在训练中增加计算开销，所以对权重会更新的全量微调应避免使用。

**关键特性：**
- 跨平台：在 AMD、Apple、NVIDIA 上表现一致
- 训练优化：使用随机舍入减少量化误差累积
- 内存高效：支持量化优化器状态缓冲
- 解耦矩阵乘：权重精度与 matmul 精度独立（支持 INT8/FP8/FP16 matmul）

##### SDNQ 优化器

SDNQ 提供可选量化状态缓冲的优化器以进一步节省内存：

- `sdnq-adamw` - 带量化状态缓冲的 AdamW（uint8，group_size=32）
- `sdnq-adamw+no_quant` - 不量化状态的 AdamW（对比用）
- `sdnq-adafactor` - 带量化状态缓冲的 Adafactor
- `sdnq-came` - 带量化状态缓冲的 CAME
- `sdnq-lion` - 带量化状态缓冲的 Lion
- `sdnq-muon` - 带量化状态缓冲的 Muon
- `sdnq-muon+quantized_matmul` - Muon 在 zeropower 计算中使用 INT8 matmul

所有 SDNQ 优化器默认使用随机舍入，可通过 `--optimizer_config` 设置如 `use_quantized_buffers=false` 以禁用状态量化。

**Muon 特定选项：**
- `use_quantized_matmul` - 在 zeropower_via_newtonschulz5 中启用 INT8/FP8/FP16 matmul
- `quantized_matmul_dtype` - matmul 精度：`int8`（消费级 GPU）、`fp8`（数据中心）、`fp16`
- `zeropower_dtype` - zeropower 计算精度（当 `use_quantized_matmul=True` 时忽略）
- 使用 `muon_` 或 `adamw_` 前缀为 Muon 与 AdamW 回退设置不同值

**预量化模型：** Disty0 在 [huggingface.co/collections/Disty0/sdnq](https://huggingface.co/collections/Disty0/sdnq) 提供预量化 uint4 SVD 模型。正常加载后，在导入 SDNQ 后调用 `convert_sdnq_model_to_training()` 进行转换（必须在加载前导入 SDNQ 以注册 Diffusers）。

**检查点说明：** SDNQ 训练模型会同时保存为训练恢复用的 PyTorch 原生格式（`.pt`）与推理用的 safetensors 格式。由于 SDNQ 的 `SDNQTensor` 使用自定义序列化，恢复训练必须使用原生格式。

**磁盘节省提示：** 可仅保留量化权重，需要推理时使用 SDNQ 的 [dequantize_sdnq_training.py](https://github.com/Disty0/sdnq/blob/main/scripts/dequantize_sdnq_training.py) 进行反量化。

### `--quantization_config`

- **内容**：在 `--quantize_via=pipeline` 时，用于覆盖 Diffusers `quantization_config` 的 JSON 对象或文件路径。
- **方式**：接受内联 JSON（或文件）定义组件级配置，键可包含 `unet`、`transformer`、`text_encoder` 或 `default`。
- **示例**：

```json
{
  "unet": {"load_in_4bit": true, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16"},
  "text_encoder": {"quant_type": {"group_size": 128}}
}
```

该示例在 UNet 上启用 4-bit NF4 BnB，并在文本编码器上启用 TorchAO int4。

#### Torch Dynamo

在 WebUI 中启用 `torch.compile()`，前往 **Hardware → Accelerate (advanced)** 并将 **Torch Dynamo Backend** 设为你偏好的编译器（如 *inductor*）。额外开关可选择优化 **mode**、启用 **dynamic shape** 防护，或开启 **regional compilation** 以加快深层 Transformer 的冷启动。

相同配置可在 `config/config.env` 中表达：

```bash
TRAINING_DYNAMO_BACKEND=inductor
```

可选与 `--dynamo_mode=max-autotune` 或 UI 中暴露的其他 Dynamo 参数搭配以细调。

注意训练前几个步骤会比平时慢，因为后台正在编译。

要将设置持久化到 `config.json`，添加等效键：

```json
{
  "dynamo_backend": "inductor",
  "dynamo_mode": "max-autotune",
  "dynamo_fullgraph": false,
  "dynamo_dynamic": false,
  "dynamo_use_regional_compilation": true
}
```

省略你希望继承 Accelerate 默认值的项（例如省略 `dynamo_mode` 以使用自动选择）。

### `--attention_mechanism`

支持多种注意力机制，兼容性与权衡不同：

- `diffusers` 使用 PyTorch 原生 SDPA 内核，是默认选项。
- `xformers` 在模型暴露 `enable_xformers_memory_efficient_attention` 时启用 Meta 的 [xformers](https://github.com/facebookresearch/xformers) 注意力内核（训练+推理）。
- `flash-attn`、`flash-attn-2`、`flash-attn-3`、`flash-attn-3-varlen` 接入 Diffusers 的 `attention_backend`，将注意力路由到 FlashAttention v1/2/3 内核。需安装对应的 `flash-attn` / `flash-attn-interface`，且 FA3 目前需要 Hopper GPU。
- `flex` 选择 PyTorch 2.5 的 FlexAttention 后端（CUDA 上 FP16/BF16）。需单独编译/安装 Flex 内核，见 [documentation/attention/FLEX.md](attention/FLEX.md)。
- `cudnn`、`native-efficient`、`native-flash`、`native-math`、`native-npu`、`native-xla` 选择 `torch.nn.attention.sdpa_kernel` 暴露的对应 SDPA 后端。适合需要确定性（`native-math`）、CuDNN SDPA 内核或厂商原生加速器（NPU/XLA）的场景。
- `sla` 启用 [Sparse–Linear Attention (SLA)](https://github.com/thu-ml/SLA)，提供可调的稀疏/线性混合内核，训练与验证均可使用。
  - 选择该后端前请安装 SLA 包（例如 `pip install -e ~/src/SLA`）。
  - SimpleTuner 会将 SLA 学到的投影权重保存到每个检查点内的 `sla_attention.pt`，恢复与推理时需保留该文件以保持 SLA 状态。
  - 由于骨干网络围绕 SLA 的混合稀疏/线性行为进行调优，推理时也需要 SLA。详见 `documentation/attention/SLA.md`。
  - 如需实验，可使用 `--sla_config '{"topk":0.15,"blkq":32,"tie_feature_map_qk":false}'`（JSON 或 Python dict）覆盖 SLA 默认运行时设置。
- `sageattention`、`sageattention-int8-fp16-triton`、`sageattention-int8-fp16-cuda`、`sageattention-int8-fp8-cuda` 包装对应的 [SageAttention](https://github.com/thu-ml/SageAttention) 内核。这些面向推理，必须配合 `--sageattention_usage` 防止误用于训练。
  - 简单来说，SageAttention 可降低推理计算量。

> ℹ️ Flash/Flex/PyTorch 后端选择依赖 Diffusers 的 `attention_backend` 分发器，因此目前主要受益于已启用该路径的 Transformer 风格模型（Flux、Wan 2.x、LTXVideo、QwenImage 等）。经典 SD/SDXL UNet 仍直接使用 PyTorch SDPA。

使用 `--sageattention_usage` 启用 SageAttention 训练需谨慎，因为其自定义 CUDA 实现不会为 QKV 线性层追踪或传播梯度。

- 这会导致这些层完全未训练，可能引发模型崩塌，或仅在短期训练中带来轻微改善。

---

## 📰 Publishing

### `--push_to_hub`

- **内容**：提供后，训练完成会将模型上传至 [Hugging Face Hub](https://huggingface.co)。使用 `--push_checkpoints_to_hub` 还会推送每个中间检查点。

### `--push_to_hub_background`

- **内容**：使用后台工作线程上传至 Hugging Face Hub，使检查点推送不会暂停训练循环。
- **原因**：Hub 上传异步进行，同时训练与验证继续运行。最终上传仍会在结束前等待，以确保失败可见。

### `--webhook_config`

- **内容**：Webhook 目标配置（如 Discord、自定义端点），用于实时接收训练事件。
- **原因**：允许通过外部工具和看板监控训练进度，在关键阶段接收通知。
- **说明**：可在训练前设置 `SIMPLETUNER_JOB_ID` 环境变量以填充 webhook payload 的 `job_id` 字段：
  ```bash
  export SIMPLETUNER_JOB_ID="my-training-run-name"
  python train.py
  ```
对接多个训练任务的监控工具可据此识别事件来源。若未设置 SIMPLETUNER_JOB_ID，webhook payload 中 job_id 为 null。

### `--publishing_config`

- **内容**：可选 JSON/dict/文件路径，描述非 Hugging Face 的发布目标（S3 兼容存储、Backblaze B2、Azure Blob Storage、Dropbox）。
- **原因**：与 `--webhook_config` 解析方式一致，可将产物分发到 Hub 之外。发布在验证后由主进程使用当前 `output_dir` 执行。
- **说明**：这些目标可与 `--push_to_hub` 叠加使用。启用前请在 `.venv` 中安装相应 SDK（例如 `boto3`、`azure-storage-blob`、`dropbox`）。完整示例见 `documentation/publishing/README.md`。

### `--hub_model_id`

- **内容**：Hugging Face Hub 模型名称与本地结果目录名。
- **原因**：该值用作 `--output_dir` 下的目录名称。若设置 `--push_to_hub`，这将成为 Hugging Face Hub 上的模型名。

### `--modelspec_comment`

- **内容**：嵌入到 safetensors 文件元数据中的文本，键为 `modelspec.comment`
- **默认值**：None（禁用）
- **说明**：
  - 在外部模型查看器（ComfyUI、模型信息工具）中可见
  - 接受字符串或字符串数组（用换行符连接）
  - 支持 `{env:VAR_NAME}` 占位符用于环境变量替换
  - 每个检查点使用保存时的当前配置值

**示例（字符串）**：
```json
"modelspec_comment": "在我的自定义数据集 v2.1 上训练"
```

**示例（数组，多行）**：
```json
"modelspec_comment": [
  "训练运行：experiment-42",
  "数据集：custom-portraits-v2",
  "备注：{env:TRAINING_NOTES}"
]
```

### `--disable_benchmark`

- **内容**：禁用启动时在 step 0 的验证/基准测试。其输出会拼接到训练模型验证图像的左侧。

## 📂 数据存储与管理

### `--data_backend_config`

- **内容**：SimpleTuner 数据集配置路径。
- **原因**：可将不同存储介质上的多个数据集组合到同一训练会话中。
- **示例**：配置示例见 [multidatabackend.json.example](/multidatabackend.json.example)，更多信息见 [此文档](DATALOADER.md)。

### `--override_dataset_config`

- **内容**：提供后允许 SimpleTuner 忽略缓存配置与当前配置之间的差异。
- **原因**：SimpleTuner 首次在数据集上运行时会创建缓存文档，其中包含数据集配置（包括 `crop` 与 `resolution` 等）。随意或误改这些值可能导致训练随机崩溃，因此不建议使用该参数，最好通过其他方式解决配置差异。

### `--data_backend_sampling`

- **内容**：多数据后端时可选择不同采样策略。
- **选项**：
  - `uniform` - v0.9.8.1 及更早版本的行为，不考虑数据集长度，只按手动概率权重采样。
  - `auto-weighting` - 默认行为，使用数据集长度均匀采样各数据集，以保持整体分布均匀。
    - 当数据集大小不同但希望模型均衡学习时必须使用。
    - 但 Dreambooth 图像与正则化集的正确采样**需要**手动调整 `repeats`。

### `--vae_cache_scan_behaviour`

- **内容**：配置完整性扫描的行为。
- **原因**：数据集可能在多个训练阶段被错误配置，例如误删 `.json` 缓存并改用方形裁剪配置，导致缓存不一致。可在 `multidatabackend.json` 中设置 `scan_for_errors=true` 进行修复。扫描时会根据 `--vae_cache_scan_behaviour` 决定如何修复不一致：`recreate`（默认）删除异常缓存以重建，`sync` 则更新桶元数据以匹配真实样本。推荐值：`recreate`。

### `--dataloader_prefetch`

- **内容**：提前加载批次。
- **原因**：当批次较大时，从磁盘（即使是 NVMe）加载会导致训练“停顿”，影响 GPU 利用率。启用预取会保持批次缓冲，便于即时加载。

> ⚠️ 这主要适用于 H100 或更高、低分辨率且 I/O 成为瓶颈的场景。对大多数用途而言属于不必要的复杂度。

### `--dataloader_prefetch_qlen`

- **内容**：增减内存中缓存的批次数。
- **原因**：启用预取后，默认每个 GPU/进程保留 10 个条目。该值可调以增加或减少预取批次数。

### `--compress_disk_cache`

- **内容**：压缩磁盘上的 VAE 与文本嵌入缓存。
- **原因**：DeepFloyd、SD3、PixArt 使用的 T5 编码器生成的文本嵌入很大，短或重复字幕会产生大量空白。启用 `--compress_disk_cache` 可节省最多 75% 空间，平均约 40%。

> ⚠️ 需要手动删除已有缓存目录，以便训练器以压缩方式重建。

---

## 🌈 图像与文本处理

许多设置通过 [dataloader 配置](DATALOADER.md) 控制，但以下设置为全局生效。

### `--resolution_type`

- **内容**：指定 SimpleTuner 使用 `area` 面积计算还是 `pixel` 边长计算。`pixel_area` 为混合方式，用像素代替百万像素表示 `area`。
- **选项**：
  - `resolution_type=pixel_area`
    - `resolution` 设为 1024 时，会内部映射为准确面积以便高效分桶。
    - 结果尺寸示例：1024x1024、1216x832、832x1216
  - `resolution_type=pixel`
    - 数据集内所有图像会将短边缩放到该分辨率，可能导致较高显存占用。
    - 结果尺寸示例：1024x1024、1766x1024、1024x1766
  - `resolution_type=area`
    - **已弃用**。请使用 `pixel_area`。

### `--resolution`

- **内容**：输入图像分辨率（短边像素长度）
- **默认**：1024
- **说明**：若数据集未设置分辨率，该值为全局默认。

### `--validation_resolution`

- **内容**：输出图像分辨率（像素），或 `widthxheight` 格式（如 `1024x1024`）。可用逗号分隔多个分辨率。
- **原因**：验证阶段生成的所有图像都将使用该分辨率。适用于训练分辨率与验证分辨率不同的场景。

### `--validation_method`

- **内容**：选择验证运行方式。
- **选项**：`simpletuner-local`（默认）使用内置管线；`external-script` 运行用户提供的可执行文件。
- **原因**：可将验证交给外部系统，无需本地管线暂停训练。

### `--validation_external_script`

- **内容**：当 `--validation_method=external-script` 时运行的可执行命令。使用 shell 风格切分，请正确引用命令字符串。
- **占位符**：可嵌入这些 token（`.format` 格式）以传递训练上下文。缺失值默认替换为空字符串（除非另有说明）：
  - `{local_checkpoint_path}` → `output_dir` 下最新检查点目录（至少存在一个检查点）。
  - `{local_checkpoint_path}` → `output_dir` 下最新检查点目录（至少存在一个检查点）。
  - `{global_step}` → 当前全局步数。
  - `{tracker_run_name}` → `--tracker_run_name` 的值。
  - `{tracker_project_name}` → `--tracker_project_name` 的值。
  - `{model_family}` → `--model_family` 的值。
  - `{model_type}` / `{lora_type}` → 模型类型与 LoRA 变体。
  - `{huggingface_path}` → `--hub_model_id` 的值（若已设置）。
  - `{remote_checkpoint_path}` → 最近一次上传的远程 URL（验证钩子为空）。
  - 任意 `validation_*` 配置值（例如 `validation_num_inference_steps`、`validation_guidance`、`validation_noise_scheduler`）。
- **示例**：`--validation_external_script="/opt/tools/validate.sh {local_checkpoint_path} {global_step}"`

### `--validation_external_background`

- **内容**：启用后，`--validation_external_script` 会在后台启动（fire-and-forget）。
- **原因**：无需等待外部脚本即可继续训练；该模式不检查退出码。

### `--post_upload_script`

- **内容**：每个发布目标与 Hugging Face Hub 上传完成后运行的可执行脚本（最终模型与检查点上传）。异步运行，不阻塞训练。
- **占位符**：与 `--validation_external_script` 相同的替换，加上 `{remote_checkpoint_path}`（提供者返回的 URI），可用于将发布 URL 传递给下游系统。
- **说明**：
  - 脚本按提供者/上传逐次运行；错误会记录但不会中止训练。
  - 即使没有远程上传也会调用，可用于本地自动化（例如在另一块 GPU 上推理）。
  - SimpleTuner 不会读取脚本结果；如需记录指标或图像，请直接写入你的追踪器。
- **示例**：
  ```bash
  --post_upload_script='/opt/hooks/notify.sh {remote_checkpoint_path} {tracker_project_name} {tracker_run_name}'
  ```
  `/opt/hooks/notify.sh` 向追踪系统提交的示例：
  ```bash
  #!/usr/bin/env bash
  REMOTE="$1"
  PROJECT="$2"
  RUN="$3"
  curl -X POST "https://tracker.internal/api/runs/${PROJECT}/${RUN}/artifacts" \
       -H "Content-Type: application/json" \
       -d "{\"remote_uri\":\"${REMOTE}\"}"
  ```
- **可用示例**：
  - `simpletuner/examples/external-validation/replicate_post_upload.py` 展示 Replicate 钩子，使用 `{remote_checkpoint_path}`、`{model_family}`、`{model_type}`、`{lora_type}`、`{huggingface_path}` 在上传后触发推理。
  - `simpletuner/examples/external-validation/wavespeed_post_upload.py` 展示 WaveSpeed 钩子，使用相同占位符并包含异步轮询。
  - `simpletuner/examples/external-validation/fal_post_upload.py` 展示 fal.ai Flux LoRA 钩子（需要 `FAL_KEY`）。
  - `simpletuner/examples/external-validation/use_second_gpu.py` 在第二块 GPU 上运行 Flux LoRA 推理，即使没有远程上传也能工作。

### `--post_checkpoint_script`

- **内容**：每个检查点目录写入磁盘后立即运行的可执行文件（在任何上传开始之前）。在主进程中异步运行。
- **占位符**：与 `--validation_external_script` 相同的替换，包括 `{local_checkpoint_path}`、`{global_step}`、`{tracker_run_name}`、`{tracker_project_name}`、`{model_family}`、`{model_type}`、`{lora_type}`、`{huggingface_path}` 以及任何 `validation_*` 值。该钩子中 `{remote_checkpoint_path}` 为空。
- **说明**：
  - 对定期/手动/滚动检查点都会触发，保存完成即执行。
  - 适合触发本地自动化（复制到其他卷、运行评估作业），无需等待上传完成。
- **示例**：
  ```bash
  --post_checkpoint_script='/opt/hooks/run_eval.sh {local_checkpoint_path} {global_step}'
  ```


### `--validation_adapter_path`

- **内容**：在计划验证时临时加载单个 LoRA 适配器。
- **格式**：
  - Hugging Face 仓库：`org/repo` 或 `org/repo:weight_name.safetensors`（默认 `pytorch_lora_weights.safetensors`）。
  - 指向 safetensors 适配器的本地文件或目录路径。
- **说明**：
  - 与 `--validation_adapter_config` 互斥，同时提供会报错。
  - 适配器仅在验证运行期间挂载（基线训练权重不变）。

### `--validation_adapter_name`

- **内容**：对 `--validation_adapter_path` 加载的临时适配器指定可选标识。
- **原因**：控制日志/Web UI 中的标注方式，保证多个适配器顺序测试时名称可预测。

### `--validation_adapter_strength`

- **内容**：启用临时适配器时的强度倍率（默认 `1.0`）。
- **原因**：无需改变训练状态即可在验证中尝试更轻/更重的 LoRA 缩放；接受任意大于零的值。

### `--validation_adapter_mode`

- **选项**：`adapter_only`、`comparison`、`none`
- **内容**：
  - `adapter_only`：仅用临时适配器运行验证。
  - `comparison`：同时生成基础模型与启用适配器的样本，便于并排对比。
  - `none`：跳过适配器挂载（保留 CLI 参数但禁用功能）。

### `--validation_adapter_config`

- **内容**：描述多个验证适配器组合的 JSON 文件或内联 JSON。
- **格式**：可为数组，或包含 `runs` 数组的对象。每个条目可包含：
  - `label`: 在日志/UI 中显示的友好名称。
  - `path`: Hugging Face 仓库 ID 或本地路径（同 `--validation_adapter_path`）。
  - `adapter_name`: 每个适配器的可选标识。
  - `strength`: 可选标量覆盖。
  - `adapters`/`paths`: 在同一次运行中加载多个适配器的对象/字符串数组。
- **说明**：
  - 提供该配置后，单适配器选项（`--validation_adapter_path`、`--validation_adapter_name`、`--validation_adapter_strength`、`--validation_adapter_mode`）在 UI 中会被忽略/禁用。
  - 每个 run 会逐个加载，并在下一个 run 开始前完全卸载。

### `--validation_preview`

- **内容**：使用 Tiny AutoEncoder 在扩散采样中流式输出验证预览
- **默认**：False
- **原因**：实时预览验证图像生成过程。通过轻量 Tiny AutoEncoder 解码并通过 webhook 回调发送，使你无需等待完整生成。
- **说明**：
  - 仅适用于支持 Tiny AutoEncoder 的模型家族（如 Flux、SDXL、SD3）
  - 需要 webhook 配置以接收预览图像
  - 使用 `--validation_preview_steps` 控制预览频率

### `--validation_preview_steps`

- **内容**：验证预览解码与流式输出的间隔
- **默认**：1
- **原因**：控制验证采样中解码中间潜变量的频率。设为更高值（如 3）可减少 Tiny AutoEncoder 解码开销。
- **示例**：当 `--validation_num_inference_steps=20` 且 `--validation_preview_steps=5` 时，生成过程在 5、10、15、20 步各收到 1 张预览图，共 4 张。

### `--evaluation_type`

- **内容**：在验证期间启用 CLIP 评估生成图像。
- **原因**：CLIP 分数计算生成图像特征与验证提示词之间的距离，可用于判断提示词一致性是否提升，但需要大量验证提示词才有意义。
- **选项**："none" 或 "clip"
- **调度**：使用 `--eval_steps_interval` 进行步数调度，或 `--eval_epoch_interval` 进行 epoch 调度（如 `0.5` 表示每个 epoch 多次）。若两者都设定，训练器会记录警告并同时运行。

### `--eval_loss_disable`

- **内容**：在验证期间禁用评估损失计算。
- **原因**：配置评估数据集后会自动计算损失；若启用 CLIP 评估，两者都会运行。此标志可在保留 CLIP 的同时关闭评估损失。

### `--validation_using_datasets`

- **内容**：使用训练数据集中的图像进行验证，而非纯文本到图像生成。
- **原因**：启用图像到图像（img2img）或图像到视频（i2v）验证模式，模型使用训练图像作为条件输入。适用于：
  - 测试需要输入图像的编辑/修复模型
  - 评估模型对图像结构的保留程度
  - 支持双重文本到图像和图像到图像工作流的模型（如 Flux2、LTXVideo2）
  - **I2V 视频模型**（HunyuanVideo、WAN、Kandinsky5Video）：使用图像数据集中的图像作为视频生成验证的首帧条件输入
- **注意**：
  - 需要模型注册 `IMG2IMG` 或 `IMG2VIDEO` 管线
  - 可与 `--eval_dataset_id` 结合从特定数据集获取图像
  - 对于 i2v 模型，这允许使用简单的图像数据集进行验证，而无需设置训练时使用的复杂条件数据集配对
  - 去噪强度由正常的验证时间步设置控制

### `--eval_dataset_id`

- **内容**：用于评估/验证图像来源的特定数据集 ID。
- **原因**：使用 `--validation_using_datasets` 或基于条件的验证时，控制哪个数据集提供输入图像：
  - 无此选项时，从所有训练数据集随机选择图像
  - 有此选项时，仅使用指定数据集进行验证输入
- **注意**：
  - 数据集 ID 必须与数据加载器配置中的已配置数据集匹配
  - 适用于使用专用评估数据集保持评估一致性
  - 对于条件模型，数据集的条件数据（如有）也会被使用

---

## 理解条件化和验证模式

SimpleTuner 为使用条件输入（参考图像、控制信号等）的模型支持三种主要范式：

### 1. 需要条件化的模型

部分模型没有条件输入无法运行：

- **Flux Kontext**：编辑式训练始终需要参考图像
- **ControlNet 训练**：需要控制信号图像

对于这些模型，条件数据集是强制性的。WebUI 将条件选项显示为必需，没有它们训练将失败。

### 2. 支持可选条件化的模型

部分模型可在文本到图像和图像到图像两种模式下运行：

- **Flux2**：支持可选参考图像的双重 T2I/I2I 训练
- **LTXVideo2**：支持可选首帧条件的 T2V 和 I2V（图像到视频）
- **LongCat-Video**：支持可选帧条件
- **HunyuanVideo i2v**：支持首帧条件的 I2V（型号：`i2v-480p`、`i2v-720p` 等）
- **WAN i2v**：支持首帧条件的 I2V
- **Kandinsky5Video i2v**：支持首帧条件的 I2V

对于这些模型，你可以添加条件数据集但不是必需的。WebUI 将条件选项显示为可选。

**I2V 验证快捷方式**：对于 i2v 视频模型，你可以使用 `--validation_using_datasets` 配合图像数据集（通过 `--eval_dataset_id` 指定）直接获取验证条件图像，而无需设置训练时使用的完整条件数据集配对。

### 3. 验证模式

| 模式 | 标志 | 行为 |
|------|------|------|
| **文本到图像/视频** | （默认） | 仅从文本提示生成 |
| **基于数据集（img2img）** | `--validation_using_datasets` | 对数据集中的图像部分去噪 |
| **基于数据集（i2v）** | `--validation_using_datasets` | 对于 i2v 视频模型，使用图像作为首帧条件 |
| **基于条件** | （配置条件时自动） | 验证时使用条件输入 |

**组合模式**：当模型支持条件化且 `--validation_using_datasets` 已启用时：
- 验证系统从数据集获取图像
- 如果这些数据集有条件数据，会自动使用
- 使用 `--eval_dataset_id` 控制哪个数据集提供输入

**I2V 模型与 `--validation_using_datasets`**：对于 i2v 视频模型（HunyuanVideo、WAN、Kandinsky5Video），启用此标志允许你使用简单的图像数据集进行验证。图像作为首帧条件输入用于生成验证视频，无需复杂的条件数据集配对设置。

### 条件数据类型

不同模型期望不同的条件数据：

| 类型 | 模型 | 数据集设置 |
|------|------|-----------|
| `conditioning` | ControlNet, Control | 数据集配置中 `type: conditioning` |
| `image` | Flux Kontext | `type: image`（标准图像数据集） |
| `latents` | Flux, Flux2 | 条件自动 VAE 编码 |

---

### `--caption_strategy`

- **内容**：派生图像字幕的策略。**选项**：`textfile`, `filename`, `parquet`, `instanceprompt`
- **原因**：决定训练图像的字幕生成方式。
  - `textfile` 使用与图像同名的 `.txt` 文件内容
  - `filename` 对文件名做清理后作为字幕
  - `parquet` 需要数据集中存在 parquet 文件，默认使用 `caption` 列，除非指定 `parquet_caption_column`。如未提供 `parquet_fallback_caption_column`，所有字幕都必须存在。
  - `instanceprompt` 使用数据集配置中的 `instance_prompt` 作为所有图像的提示词。

### `--conditioning_multidataset_sampling` {#--conditioning_multidataset_sampling}

- **内容**：多条件数据集的采样方式。**选项**：`combined`, `random`
- **原因**：当训练使用多个条件数据集（如多参考图或控制信号）时，决定如何使用它们：
  - `combined` 将条件输入拼接在一起，训练时同时展示。适用于多图合成任务。
  - `random` 每个样本随机选择一个条件数据集，在训练中切换条件。
- **说明**：使用 `combined` 时不能在条件数据集中定义单独的 `captions`，会使用源数据集的字幕。
- **另见**：[DATALOADER.md](DATALOADER.md#conditioning_data) 获取多条件数据集配置说明。

---

## 🎛 训练参数

### `--num_train_epochs`

- **内容**：训练轮数（全图像被看到的次数）。设为 0 时由 `--max_train_steps` 优先。
- **原因**：决定图像重复次数，影响训练时长。更多 epoch 更易过拟合，但可能需要以学习目标概念。合理范围通常为 5–50。

### `--max_train_steps`

- **内容**：训练步数上限。设为 0 时由 `--num_train_epochs` 优先。
- **原因**：用于缩短训练时长。

### `--ignore_final_epochs`

- **内容**：忽略最后计算的 epoch，改由 `--max_train_steps` 控制。
- **原因**：当 dataloader 长度变化导致训练提前结束时，该选项会忽略最后的 epoch 并继续训练至 `--max_train_steps`。

### `--learning_rate`

- **内容**：潜在 warmup 后的初始学习率。
- **原因**：学习率类似于梯度更新的“步长”。太高会越过解，太低则无法到达理想解。`full` 调优的最小值可能低至 `1e-7`，最大约 `1e-6`；`lora` 调优最小 `1e-5`，最大可到 `1e-3`。高学习率时建议配合 EMA 与 warmup（见 `--use_ema`、`--lr_warmup_steps`、`--lr_scheduler`）。

### `--lr_scheduler`

- **内容**：随时间调整学习率的方式。
- **选项**：constant, constant_with_warmup, cosine, cosine_with_restarts, **polynomial**（推荐）, linear
- **原因**：模型受益于持续的学习率调整以探索损失景观。默认使用 cosine，可在两个极值间平滑过渡。恒定学习率易过高导致发散或过低导致陷入局部最小。polynomial 与 warmup 配合最佳，会逐渐接近 `learning_rate`，随后减速并在末尾接近 `--lr_end`。

### `--optimizer`

- **内容**：训练使用的优化器。
- **选项**：adamw_bf16, ao-adamw8bit, ao-adamw4bit, ao-adamfp8, ao-adamwfp8, adamw_schedulefree, adamw_schedulefree+aggressive, adamw_schedulefree+no_kahan, optimi-stableadamw, optimi-adamw, optimi-lion, optimi-radam, optimi-ranger, optimi-adan, optimi-adam, optimi-sgd, soap, bnb-adagrad, bnb-adagrad8bit, bnb-adam, bnb-adam8bit, bnb-adamw, bnb-adamw8bit, bnb-adamw-paged, bnb-adamw8bit-paged, bnb-lion, bnb-lion8bit, bnb-lion-paged, bnb-lion8bit-paged, bnb-ademamix, bnb-ademamix8bit, bnb-ademamix-paged, bnb-ademamix8bit-paged, prodigy

> 注记：部分优化器在非 NVIDIA 硬件上可能不可用。

### `--optimizer_config`

- **内容**：调整优化器设置。
- **原因**：优化器设置众多，无法为每个提供单独 CLI 参数。可用逗号分隔的键值覆盖默认设置。
- **示例**：为 **prodigy** 设置 `d_coef`：`--optimizer_config=d_coef=0.1`

> 注记：优化器 betas 使用专用参数 `--optimizer_beta1` 与 `--optimizer_beta2` 覆盖。

### `--train_batch_size`

- **内容**：训练数据加载器的批大小。
- **原因**：影响内存消耗、收敛质量与训练速度。批越大结果通常越好，但过大可能导致过拟合或训练不稳定，并延长训练时间。需要实验，通常希望在不降低训练速度的前提下尽量用满显存。

### `--gradient_accumulation_steps`

- **内容**：在执行反向/更新前累积的更新步数，相当于将工作分摊到多个批次以节省内存，但增加训练耗时。
- **原因**：适用于更大的模型或数据集。

> 注记：使用梯度累积时，不要为任何优化器启用 fused backward pass。

### `--allow_dataset_oversubscription` {#--allow_dataset_oversubscription}

- **内容**：当数据集小于有效批大小时自动调整 `repeats`。
- **原因**：避免多 GPU 配置下因数据集不足而训练失败。
- **工作方式**：
  - 计算**有效批大小**：`train_batch_size × num_gpus × gradient_accumulation_steps`
  - 若某些纵横比桶样本数少于有效批大小，则自动增加 `repeats`
  - 仅在数据集配置未显式设置 `repeats` 时生效
  - 记录警告说明调整原因与结果
- **适用场景**：
  - 小数据集（< 100 张）+ 多 GPU
  - 试验不同 batch 大小而无需重配数据集
  - 在收集完整数据集前做原型验证
- **示例**：25 张图像、8 GPU、`train_batch_size=4` 时，有效批大小为 32。该标志会自动设定 `repeats=1`，提供 50 样本（25 × 2）。
- **说明**：不会覆盖 dataloader 配置中手动设置的 `repeats`。与 `--disable_bucket_pruning` 类似，此标志在不产生意外行为的前提下提供便利。

更多关于多 GPU 数据集规模的细节见 [DATALOADER.md](DATALOADER.md#automatic-dataset-oversubscription)。

---

## 🛠 高级优化

### `--use_ema`

- **内容**：在训练期间维护权重的指数移动平均，相当于定期将模型“回合并”到自身。
- **原因**：可提升训练稳定性，但会增加系统资源开销，并略微增加训练时间。

### `--ema_device`

- **选项**：`cpu`, `accelerator`；默认：`cpu`
- **内容**：选择 EMA 权重在更新间存放的位置。
- **原因**：放在加速器上更新最快但占用显存；放在 CPU 上可降低显存压力，但除非设置 `--ema_cpu_only`，否则会产生权重传输。

### `--ema_cpu_only`

- **内容**：当 `--ema_device=cpu` 时，阻止 EMA 权重在更新时搬回加速器。
- **原因**：节省大规模 EMA 的主机-设备传输时间与显存占用。若 `--ema_device=accelerator` 则无效，因为权重已在加速器上。

### `--ema_foreach_disable`

### `--ema_foreach_disable`

- **内容**：禁用 EMA 更新中对 `torch._foreach_*` 内核的使用。
- **原因**：某些后端或硬件组合在 foreach 操作上有问题。禁用后回退到标量实现，更新会略慢。

### `--ema_update_interval`

- **内容**：降低 EMA 影子参数的更新频率。
- **原因**：对许多工作流而言每步更新并不必要。例如 `--ema_update_interval=100` 仅每 100 个优化步更新一次 EMA，可减少 `--ema_device=cpu` 或 `--ema_cpu_only` 时的开销。

### `--ema_decay`

- **内容**：控制 EMA 更新时的平滑系数。
- **原因**：较高值（例如 `0.999`）响应更慢但更稳定；较低值（例如 `0.99`）更快适应新信号。

### `--snr_gamma`

- **内容**：使用 min-SNR 加权损失因子。
- **原因**：Minimum SNR gamma 按时间步位置调整损失权重。噪声较大的时间步贡献降低，噪声较小的时间步贡献提高。原论文推荐值为 **5**，但可用 **1** 到 **20**。通常 **20** 被视为上限，超过 20 数学上变化不大。**1** 为最强权重。

### `--use_soft_min_snr`

- **内容**：使用更渐进的损失权重进行训练。
- **原因**：像素扩散模型若不使用特定损失权重会退化。对 DeepFloyd 而言，soft-min-snr-gamma 基本必需。潜在扩散模型可能也有效，但小规模实验中可能产生模糊结果。

### `--diff2flow_enabled`

- **内容**：为 epsilon 或 v-prediction 模型启用 Diffusion-to-Flow 桥接。
- **原因**：允许以标准扩散目标训练的模型在不改变架构的情况下使用流匹配目标（noise - latents）。
- **说明**：实验性功能。

### `--diff2flow_loss`

- **内容**：使用 Flow Matching 损失替代原生预测损失进行训练。
- **原因**：与 `--diff2flow_enabled` 一起启用时，将损失计算于 flow 目标（noise - latents），而非模型原生目标（epsilon 或 velocity）。
- **说明**：需要 `--diff2flow_enabled`。

### `--scheduled_sampling_max_step_offset`

- **内容**：训练中“rollout”的最大步数。
- **原因**：启用计划采样（Rollout），让模型在训练过程中生成自身输入数步，帮助纠正错误并减少暴露偏差。
- **默认**：0（禁用）。设置为正整数（如 5 或 10）启用。

### `--scheduled_sampling_strategy`

- **内容**：选择 rollout 偏移的策略。
- **选项**：`uniform`, `biased_early`, `biased_late`。
- **默认**：`uniform`。
- **原因**：控制 rollout 长度分布。`uniform` 均匀采样；`biased_early` 偏短；`biased_late` 偏长。

### `--scheduled_sampling_probability`

- **内容**：对某个样本应用非零 rollout 偏移的概率。
- **默认**：0.0。
- **原因**：控制计划采样的应用频率。0.0 即使 `max_step_offset` > 0 也会禁用；1.0 表示每个样本都应用。

### `--scheduled_sampling_prob_start`

- **内容**：概率爬坡起始值。
- **默认**：0.0。

### `--scheduled_sampling_prob_end`

- **内容**：概率爬坡终止值。
- **默认**：0.5。

### `--scheduled_sampling_ramp_steps`

- **内容**：从 `prob_start` 过渡到 `prob_end` 的步数。
- **默认**：0（无爬坡）。

### `--scheduled_sampling_start_step`

- **内容**：开始计划采样爬坡的全局步数。
- **默认**：0.0。

### `--scheduled_sampling_ramp_shape`

- **内容**：概率爬坡形状。
- **选项**：`linear`, `cosine`。
- **默认**：`linear`。

### `--scheduled_sampling_sampler`

- **内容**：rollout 生成步骤所用的求解器。
- **选项**：`unipc`, `euler`, `dpm`, `rk4`。
- **默认**：`unipc`。

### `--scheduled_sampling_order`

- **内容**：rollout 使用的求解器阶数。
- **默认**：2。

### `--scheduled_sampling_reflexflow`

- **内容**：在 flow-matching 模型的计划采样中启用 ReflexFlow 风格增强（抗漂移 + 频率补偿权重）。
- **原因**：通过方向正则与偏差感知重加权降低 flow-matching 模型 rollout 的暴露偏差。
- **默认**：当 `--scheduled_sampling_max_step_offset` > 0 时，对 flow-matching 模型自动启用；可用 `--scheduled_sampling_reflexflow=false` 覆盖。

### `--scheduled_sampling_reflexflow_alpha`

- **内容**：来自暴露偏差的频率补偿权重缩放系数。
- **默认**：1.0。
- **原因**：值越大，rollout 中暴露偏差更大的区域权重越高。

### `--scheduled_sampling_reflexflow_beta1`

- **内容**：ReflexFlow 抗漂移（方向）正则项的权重。
- **默认**：10.0。
- **原因**：控制 flow-matching 模型在计划采样中预测方向与目标干净样本对齐的强度。

### `--scheduled_sampling_reflexflow_beta2`

- **内容**：ReflexFlow 频率补偿（损失重加权）项的权重。
- **默认**：1.0。
- **原因**：缩放重加权后的 flow-matching 损失，对应 ReflexFlow 论文中的 β₂。

---

## 🎯 CREPA（Cross-frame Representation Alignment）

CREPA 是一种用于视频扩散模型微调的正则化技术，通过将隐藏状态与相邻帧的预训练视觉特征对齐来提升时间一致性。基于论文 ["Cross-Frame Representation Alignment for Fine-Tuning Video Diffusion Models"](https://arxiv.org/abs/2506.09229)。

### `--crepa_enabled`

- **内容**：训练时启用 CREPA 正则化。
- **原因**：通过将 DiT 隐藏状态与相邻帧的 DINOv2 特征对齐，提高视频帧之间的语义一致性。
- **默认**：`false`
- **注意**：仅适用于 Transformer 扩散模型（DiT 风格）。UNet 模型（SDXL、SD1.5、Kolors）请使用 U-REPA。

### `--crepa_block_index`

- **内容**：用于对齐的 Transformer 块隐藏状态索引。
- **原因**：论文建议 CogVideoX 使用块 8，Hunyuan Video 使用块 10。较早的块往往效果更好，因为它们充当 DiT 的“编码器”部分。
- **必需**：启用 CREPA 时必需。

### `--crepa_lambda`

- **内容**：CREPA 对齐损失相对于主训练损失的权重。
- **原因**：控制对齐正则对训练的影响强度。论文对 CogVideoX 使用 0.5，对 Hunyuan Video 使用 1.0。
- **默认**：`0.5`

### `--crepa_adjacent_distance`

- **内容**：邻帧对齐距离 `d`。
- **原因**：按论文公式 6，$K = \{f-d, f+d\}$ 定义要对齐的邻帧。`d=1` 时，每帧与其相邻帧对齐。
- **默认**：`1`

### `--crepa_adjacent_tau`

- **内容**：指数距离权重的温度系数。
- **原因**：通过 $e^{-|k-f|/\tau}$ 控制对齐权重随帧距的衰减速度。值越小越强调近邻帧。
- **默认**：`1.0`

### `--crepa_cumulative_neighbors`

- **内容**：使用累计模式而非邻接模式。
- **原因**：
  - **邻接模式（默认）**：仅与距离正好为 `d` 的帧对齐（对应论文的 $K = \{f-d, f+d\}$）
  - **累计模式**：与距离 1 到 `d` 的所有帧对齐，梯度更平滑
- **默认**：`false`

### `--crepa_normalize_neighbour_sum`

- **内容**：按帧权重和对邻帧加权和进行归一化。
- **原因**：使 `crepa_alignment_score` 保持在 [-1, 1] 并让损失尺度更直观。这是对论文公式 (6) 的实验性偏离。
- **默认**：`false`

### `--crepa_normalize_by_frames`

- **内容**：按帧数对对齐损失归一化。
- **原因**：保证不同视频长度下损失尺度一致。禁用时更长视频会获得更强对齐信号。
- **默认**：`true`

### `--crepa_spatial_align`

- **内容**：当 DiT 与编码器 token 数不同，使用空间插值对齐。
- **原因**：DiT 隐藏状态与 DINOv2 特征可能具有不同空间分辨率。启用后使用双线性插值对齐，禁用则回退到全局池化。
- **默认**：`true`

### `--crepa_model`

- **内容**：用于特征提取的预训练编码器。
- **原因**：论文使用 DINOv2-g（ViT-Giant）。更小的 `dinov2_vitb14` 等占用更少内存。
- **默认**：`dinov2_vitg14`
- **选项**：`dinov2_vitg14`, `dinov2_vitb14`, `dinov2_vits14`

### `--crepa_encoder_frames_batch_size`

- **内容**：外部特征编码器并行处理的帧数。0 或负数表示整批全部帧同时处理；若不能整除，剩余会作为更小批次处理。
- **原因**：DINO 类编码器是图像模型，可通过切片批处理减少显存占用（速度降低）。
- **默认**：`-1`

### `--crepa_use_backbone_features`

- **内容**：跳过外部编码器，在扩散模型内部将学生块与教师块对齐。
- **原因**：当主干已经包含更强语义层可监督时，可避免加载 DINOv2。
- **默认**：`false`

### `--crepa_teacher_block_index`

- **内容**：使用主干特征时的教师块索引。
- **原因**：无需外部编码器即可让早期学生块对齐到后续教师块。未设置时回退为学生块。
- **默认**：未提供时使用 `crepa_block_index`。

### `--crepa_encoder_image_size`

- **内容**：编码器输入分辨率。
- **原因**：DINOv2 在训练分辨率上效果最好。巨型模型使用 518x518。
- **默认**：`518`

### `--crepa_scheduler`

- **内容**：训练过程中 CREPA 系数的衰减调度方式。
- **原因**：允许在训练进行时逐渐降低 CREPA 正则化强度，防止对深层编码器特征过拟合。
- **选项**：`constant`、`linear`、`cosine`、`polynomial`
- **默认**：`constant`

### `--crepa_warmup_steps`

- **内容**：将 CREPA 权重从 0 线性升至 `crepa_lambda` 的步数。
- **原因**：渐进预热有助于在 CREPA 正则化生效前稳定早期训练。
- **默认**：`0`

### `--crepa_decay_steps`

- **内容**：衰减总步数（预热后）。设为 0 则在整个训练过程中衰减。
- **原因**：灵活控制衰减的时间跨度。
- **默认**：`0`（使用 `max_train_steps`）

### `--crepa_lambda_end`

- **内容**：衰减完成后的最终 CREPA 权重。
- **原因**：设为 0 可在训练末期有效禁用 CREPA，适用于 text2video 等可能产生伪影的场景。
- **默认**：`0.0`

### `--crepa_power`

- **内容**：多项式衰减的幂因子。1.0 = 线性，2.0 = 二次，以此类推。
- **原因**：控制衰减曲线形状。
- **默认**：`1.0`

### `--crepa_cutoff_step`

- **内容**：硬截止步数，超过此步后禁用 CREPA。
- **原因**：适用于模型时序对齐收敛后禁用 CREPA 的场景。
- **默认**：`0`（无基于步数的截止）

### `--crepa_similarity_threshold`

- **内容**：触发 CREPA 截止的相似度 EMA 阈值。
- **原因**：当对齐分数（`crepa_alignment_score`）的指数移动平均达到此值时，CREPA 被禁用以防止对深层编码器特征过拟合。对于 text2video 训练尤其有用。除非启用 `crepa_normalize_neighbour_sum`，否则对齐分数可能超过 1.0。
- **默认**：None（禁用）

### `--crepa_similarity_ema_decay`

- **内容**：相似度跟踪的指数移动平均衰减因子。
- **原因**：控制相似度指标的平滑程度。
- **默认**：`0.99`

### `--crepa_threshold_mode`

- **内容**：达到相似度阈值后的行为。
- **选项**：`permanent`（一旦达到阈值，CREPA 保持关闭）、`recoverable`（若相似度下降，CREPA 重新启用）
- **默认**：`permanent`

### 配置示例

```toml
# 启用 CREPA 用于视频微调
crepa_enabled = true
crepa_block_index = 8          # 根据模型调整
crepa_lambda = 0.5
crepa_adjacent_distance = 1
crepa_adjacent_tau = 1.0
crepa_cumulative_neighbors = false
crepa_normalize_neighbour_sum = false
crepa_normalize_by_frames = true
crepa_spatial_align = true
crepa_model = "dinov2_vitg14"
crepa_encoder_frames_batch_size = -1
crepa_use_backbone_features = false
# crepa_teacher_block_index = 16
crepa_encoder_image_size = 518

# CREPA 调度（可选）
# crepa_scheduler = "cosine"           # 衰减类型：constant、linear、cosine、polynomial
# crepa_warmup_steps = 100             # CREPA 生效前的预热步数
# crepa_decay_steps = 1000             # 衰减步数（0 = 整个训练过程）
# crepa_lambda_end = 0.0               # 衰减后的最终权重
# crepa_cutoff_step = 5000             # 硬截止步数（0 = 禁用）
# crepa_similarity_threshold = 0.9    # 基于相似度的截止
# crepa_threshold_mode = "permanent"   # permanent 或 recoverable
```

---

## 🎯 U-REPA（UNet 表征对齐）

U-REPA 是用于 UNet 扩散模型（SDXL、SD1.5、Kolors）的正则化方法。它将 UNet 中间块特征与预训练视觉特征对齐，并加入流形损失以保持相对相似性结构。

### `--urepa_enabled`

- **内容**：训练时启用 U-REPA 正则化。
- **原因**：使用冻结的视觉编码器对齐 UNet 中间块特征。
- **默认**：`false`
- **注意**：仅适用于 UNet 模型（SDXL、SD1.5、Kolors）。

### `--urepa_lambda`

- **内容**：U-REPA 对齐损失权重（相对于主训练损失）。
- **原因**：控制对齐正则强度。
- **默认**：`0.5`

### `--urepa_manifold_weight`

- **内容**：流形损失相对于对齐损失的权重。
- **原因**：强调特征相对相似性结构（论文默认 3.0）。
- **默认**：`3.0`

### `--urepa_model`

- **内容**：冻结视觉编码器的 torch hub 标识符。
- **原因**：默认 DINOv2 ViT-G/14；较小模型（如 `dinov2_vits14`）速度更快。
- **默认**：`dinov2_vitg14`

### `--urepa_encoder_image_size`

- **内容**：视觉编码器预处理的输入分辨率。
- **原因**：使用编码器原生分辨率（DINOv2 ViT-G/14 为 518，ViT-S/14 为 224）。
- **默认**：`518`

### `--urepa_use_tae`

- **内容**：使用 Tiny AutoEncoder 代替完整 VAE 解码。
- **原因**：更快、显存占用更低，但解码质量较低。
- **默认**：`false`

### `--urepa_scheduler`

- **内容**：U-REPA 系数随训练衰减的调度方式。
- **原因**：允许训练过程中逐渐降低 U-REPA 强度。
- **选项**：`constant`、`linear`、`cosine`、`polynomial`
- **默认**：`constant`

### `--urepa_warmup_steps`

- **内容**：将 U-REPA 权重从 0 线性升至 `urepa_lambda` 的步数。
- **原因**：渐进预热有助于稳定早期训练。
- **默认**：`0`

### `--urepa_decay_steps`

- **内容**：衰减总步数（预热之后）。设为 0 表示在整个训练过程中衰减。
- **原因**：控制衰减阶段的持续时间。
- **默认**：`0`（使用 `max_train_steps`）

### `--urepa_lambda_end`

- **内容**：衰减完成后的最终 U-REPA 权重。
- **原因**：设为 0 可在训练末期有效禁用 U-REPA。
- **默认**：`0.0`

### `--urepa_power`

- **内容**：多项式衰减的幂指数。1.0 = 线性，2.0 = 二次等。
- **原因**：更高值会在早期衰减更快、后期更慢。
- **默认**：`1.0`

### `--urepa_cutoff_step`

- **内容**：超过该步数后硬截止禁用 U-REPA。
- **原因**：当模型收敛后可关闭 U-REPA。
- **默认**：`0`（无步数截止）

### `--urepa_similarity_threshold`

- **内容**：触发 U-REPA 截止的相似度 EMA 阈值。
- **原因**：当相似度（`urepa_similarity`）的指数移动平均达到该值时禁用 U-REPA，以防止过拟合。
- **默认**：None（禁用）

### `--urepa_similarity_ema_decay`

- **内容**：相似度跟踪的指数移动平均衰减因子。
- **原因**：较高值更平滑（0.99 ≈ 100 步窗口），较低值响应更快。
- **默认**：`0.99`

### `--urepa_threshold_mode`

- **内容**：达到相似度阈值后的行为。
- **选项**：`permanent`（一旦达到阈值，U-REPA 保持关闭）、`recoverable`（若相似度下降，U-REPA 重新启用）
- **默认**：`permanent`

### 配置示例

```toml
# 启用 U-REPA 用于 UNet 微调（SDXL、SD1.5、Kolors）
urepa_enabled = true
urepa_lambda = 0.5
urepa_manifold_weight = 3.0
urepa_model = "dinov2_vitg14"
urepa_encoder_image_size = 518
urepa_use_tae = false

# U-REPA 调度（可选）
# urepa_scheduler = "cosine"           # 衰减类型：constant、linear、cosine、polynomial
# urepa_warmup_steps = 100             # U-REPA 生效前的预热步数
# urepa_decay_steps = 1000             # 衰减步数（0 = 整个训练过程）
# urepa_lambda_end = 0.0               # 衰减后的最终权重
# urepa_cutoff_step = 5000             # 硬截止步数（0 = 禁用）
# urepa_similarity_threshold = 0.9     # 基于相似度的截止
# urepa_threshold_mode = "permanent"   # permanent 或 recoverable
```

---

## 🔄 检查点与恢复

### `--checkpoint_step_interval`（别名：`--checkpointing_steps`）

- **内容**：保存训练状态检查点的间隔（步数）。
- **原因**：有助于恢复训练与推理。每 *n* 次迭代会通过 Diffusers 文件布局保存 `.safetensors` 部分检查点。

---

## 🔁 LayerSync（隐藏状态自对齐）

LayerSync 通过在同一 Transformer 内让“学生”层对齐更强的“教师”层，使用隐藏 token 的余弦相似度进行对齐。

### `--layersync_enabled`

- **内容**：启用同一模型内两个 Transformer 块之间的 LayerSync 隐藏状态对齐。
- **备注**：会分配隐藏状态缓冲区；若缺少必需参数，启动时会报错。
- **默认**：`false`

### `--layersync_student_block`

- **内容**：作为学生锚点的 Transformer 块索引。
- **索引**：接受 LayerSync 论文式 1-based 深度或 0-based 层编号；实现会先尝试 `idx-1`，再尝试 `idx`。
- **必需**：启用 LayerSync 时必需。

### `--layersync_teacher_block`

- **内容**：作为教师目标的 Transformer 块索引（可以比学生更深）。
- **索引**：与学生块一致，优先 1-based，再回退 0-based。
- **默认**：未设置时使用学生块，使损失变为自相似。

### `--layersync_lambda`

- **内容**：LayerSync 学生/教师隐藏状态之间的余弦对齐损失权重（负余弦相似度）。
- **影响**：缩放叠加在主损失上的辅助正则项；值越大越推动学生 token 向教师 token 对齐。
- **上游名称**：原始 LayerSync 代码库中的 `--reg-weight`。
- **必需**：启用 LayerSync 时必须大于 0（否则训练终止）。
- **默认**：启用 LayerSync 时为 `0.2`（与参考仓库一致），否则为 `0.0`。

上游选项映射（LayerSync → SimpleTuner）:
- `--encoder-depth` → `--layersync_student_block`（上游接受 1-based 深度，也可 0-based 层索引）
- `--gt-encoder-depth` → `--layersync_teacher_block`（推荐 1-based；未设置时默认为学生）
- `--reg-weight` → `--layersync_lambda`

> 注记：LayerSync 在计算相似度前总是先对教师隐藏状态执行 detach，与参考实现一致。它依赖能输出 Transformer 隐藏状态的模型（SimpleTuner 中的大多数 Transformer 主干），并在每个步骤增加隐藏状态缓冲区的内存开销；显存紧张时请禁用。

### `--checkpoint_epoch_interval`

- **内容**：每完成 N 个 epoch 进行检查点保存。
- **原因**：补充按步保存，确保在多数据集采样导致步数变化时，仍能在 epoch 边界捕获状态。

### `--resume_from_checkpoint`

- **内容**：指定是否以及从何处恢复训练。接受 `latest`、本地检查点名称/路径或 S3/R2 URI。
- **原因**：允许从已保存的状态继续训练，可手动指定或使用最新可用检查点。
- **远程恢复**：提供完整 URI (`s3://bucket/jobs/.../checkpoint-100`) 或相对 bucket 的 key (`jobs/.../checkpoint-100`)。`latest` 仅适用于本地 `output_dir`。
- **要求**：远程恢复需要在 publishing_config 中提供可读取检查点的 S3 配置（bucket + credentials）。
- **注意**：远程检查点必须包含 `checkpoint_manifest.json`（由较新的 SimpleTuner 运行生成）。检查点由 `unet` 与可选 `unet_ema` 子目录组成。`unet` 可直接放入 Diffusers 布局的 SDXL 模型中，像普通模型一样使用。

> ℹ️ PixArt、SD3、Hunyuan 等 Transformer 模型使用 `transformer` 与 `transformer_ema` 子目录名称。

### `--disk_low_threshold`

- **内容**：检查点保存前所需的最小可用磁盘空间。
- **原因**：通过提前检测磁盘空间不足并采取配置的操作，防止训练因磁盘已满错误而崩溃。
- **格式**：大小字符串，如 `100G`、`50M`、`1T`、`500K`，或纯字节数。
- **默认**：无（功能禁用）

### `--disk_low_action`

- **内容**：磁盘空间低于阈值时采取的操作。
- **选项**：`stop`、`wait`、`script`
- **默认**：`stop`
- **行为**：
  - `stop`：立即停止训练并显示错误消息。
  - `wait`：每 30 秒循环检查直到空间可用。可能无限等待。
  - `script`：运行 `--disk_low_script` 指定的脚本以释放空间。

### `--disk_low_script`

- **内容**：磁盘空间不足时运行的清理脚本路径。
- **原因**：允许在磁盘空间不足时自动清理（如删除旧检查点、清除缓存）。
- **注意**：仅在 `--disk_low_action=script` 时使用。脚本必须可执行。如果脚本失败或未能释放足够空间，训练将停止并报错。
- **默认**：无

---

## 📊 日志与监控

### `--logging_dir`

- **内容**：TensorBoard 日志目录。
- **原因**：便于监控训练进度与性能指标。

### `--report_to`

- **内容**：指定结果与日志的上报平台。
- **原因**：可与 TensorBoard、wandb、comet_ml 集成监控。多个值用逗号分隔；
- **选项**：wandb, tensorboard, comet_ml

## 环境配置变量

上述选项大多适用于 `config.json`，但有些条目必须在 `config.env` 中设置。

- `TRAINING_NUM_PROCESSES` 应设置为系统中的 GPU 数量。多数场景下已足够启用 DistributedDataParallel（DDP）。如果不想使用 `config.env`，可在 `config.json` 中设置 `num_processes`。
- `TRAINING_DYNAMO_BACKEND` 默认 `no`，但可设置为任意支持的 torch.compile 后端（例如 `inductor`, `aot_eager`, `cudagraphs`），并与 `--dynamo_mode`、`--dynamo_fullgraph` 或 `--dynamo_use_regional_compilation` 配合微调。
- `SIMPLETUNER_LOG_LEVEL` 默认 `INFO`，设置为 `DEBUG` 可在 `debug.log` 中记录更多问题报告信息。
- `VENV_PATH` 可设置为 Python 虚拟环境的位置（如果不在常见 `.venv` 目录）。
- `ACCELERATE_EXTRA_ARGS` 可留空，或包含额外参数，例如 `--multi_gpu` 或 FSDP 专用标志。

---

这是一个基础概览，旨在帮助你快速上手。完整选项列表与更详细说明请参阅完整规范：

```
usage: train.py [-h] --model_family
                {kolors,auraflow,omnigen,flux,deepfloyd,cosmos2image,sana,qwen_image,pixart_sigma,sdxl,sd1x,sd2x,wan,hidream,sd3,lumina2,ltxvideo}
                [--model_flavour MODEL_FLAVOUR] [--controlnet [CONTROLNET]]
                [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH]
                --output_dir OUTPUT_DIR [--logging_dir LOGGING_DIR]
                --model_type {full,lora} [--seed SEED]
                [--resolution RESOLUTION]
                [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                [--prediction_type {epsilon,v_prediction,sample,flow_matching}]
                [--pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH]
                [--vae_dtype {default,fp32,fp16,bf16}]
                [--vae_cache_ondemand [VAE_CACHE_ONDEMAND]]
                [--accelerator_cache_clear_interval ACCELERATOR_CACHE_CLEAR_INTERVAL]
                [--aspect_bucket_rounding {1,2,3,4,5,6,7,8,9}]
                [--base_model_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_1_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_2_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_3_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_4_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--gradient_checkpointing_interval GRADIENT_CHECKPOINTING_INTERVAL]
                [--offload_during_startup [OFFLOAD_DURING_STARTUP]]
                [--quantize_via {cpu,accelerator,pipeline}]
                [--quantization_config QUANTIZATION_CONFIG]
                [--fuse_qkv_projections [FUSE_QKV_PROJECTIONS]]
                [--control [CONTROL]]
                [--controlnet_custom_config CONTROLNET_CUSTOM_CONFIG]
                [--controlnet_model_name_or_path CONTROLNET_MODEL_NAME_OR_PATH]
                [--tread_config TREAD_CONFIG]
                [--pretrained_transformer_model_name_or_path PRETRAINED_TRANSFORMER_MODEL_NAME_OR_PATH]
                [--pretrained_transformer_subfolder PRETRAINED_TRANSFORMER_SUBFOLDER]
                [--pretrained_unet_model_name_or_path PRETRAINED_UNET_MODEL_NAME_OR_PATH]
                [--pretrained_unet_subfolder PRETRAINED_UNET_SUBFOLDER]
                [--pretrained_t5_model_name_or_path PRETRAINED_T5_MODEL_NAME_OR_PATH]
                [--pretrained_gemma_model_name_or_path PRETRAINED_GEMMA_MODEL_NAME_OR_PATH]
                [--revision REVISION] [--variant VARIANT]
                [--base_model_default_dtype {bf16,fp32}]
                [--unet_attention_slice [UNET_ATTENTION_SLICE]]
                [--num_train_epochs NUM_TRAIN_EPOCHS]
                [--max_train_steps MAX_TRAIN_STEPS]
                [--train_batch_size TRAIN_BATCH_SIZE]
                [--learning_rate LEARNING_RATE] --optimizer
                {adamw_bf16,ao-adamw8bit,ao-adamw4bit,ao-adamfp8,ao-adamwfp8,adamw_schedulefree,adamw_schedulefree+aggressive,adamw_schedulefree+no_kahan,optimi-stableadamw,optimi-adamw,optimi-lion,optimi-radam,optimi-ranger,optimi-adan,optimi-adam,optimi-sgd,soap,prodigy}
                [--optimizer_config OPTIMIZER_CONFIG]
                [--lr_scheduler {linear,sine,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                [--lr_warmup_steps LR_WARMUP_STEPS]
                [--checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT]
                [--gradient_checkpointing [GRADIENT_CHECKPOINTING]]
                [--train_text_encoder [TRAIN_TEXT_ENCODER]]
                [--text_encoder_lr TEXT_ENCODER_LR]
                [--lr_num_cycles LR_NUM_CYCLES] [--lr_power LR_POWER]
                [--use_soft_min_snr [USE_SOFT_MIN_SNR]] [--use_ema [USE_EMA]]
                [--ema_device {accelerator,cpu}]
                [--ema_cpu_only [EMA_CPU_ONLY]]
                [--ema_update_interval EMA_UPDATE_INTERVAL]
                [--ema_foreach_disable [EMA_FOREACH_DISABLE]]
                [--ema_decay EMA_DECAY] [--lora_rank LORA_RANK]
                [--lora_alpha LORA_ALPHA] [--lora_type {standard,lycoris}]
                [--lora_dropout LORA_DROPOUT]
                [--lora_init_type {default,gaussian,loftq,olora,pissa}]
                [--peft_lora_mode {standard,singlora}]
                [--peft_lora_target_modules PEFT_LORA_TARGET_MODULES]
                [--singlora_ramp_up_steps SINGLORA_RAMP_UP_STEPS]
                [--init_lora INIT_LORA] [--lycoris_config LYCORIS_CONFIG]
                [--init_lokr_norm INIT_LOKR_NORM]
                [--flux_lora_target {mmdit,context,context+ffs,all,all+ffs,ai-toolkit,tiny,nano,controlnet,all+ffs+embedder,all+ffs+embedder+controlnet}]
                [--use_dora [USE_DORA]]
                [--resolution_type {pixel,area,pixel_area}]
                --data_backend_config DATA_BACKEND_CONFIG
                [--caption_strategy {filename,textfile,instance_prompt,parquet}]
                [--conditioning_multidataset_sampling {combined,random}]
                [--instance_prompt INSTANCE_PROMPT]
                [--parquet_caption_column PARQUET_CAPTION_COLUMN]
                [--parquet_filename_column PARQUET_FILENAME_COLUMN]
                [--ignore_missing_files [IGNORE_MISSING_FILES]]
                [--vae_cache_scan_behaviour {recreate,sync}]
                [--vae_enable_slicing [VAE_ENABLE_SLICING]]
                [--vae_enable_tiling [VAE_ENABLE_TILING]]
                [--vae_enable_patch_conv [VAE_ENABLE_PATCH_CONV]]
                [--vae_batch_size VAE_BATCH_SIZE]
                [--caption_dropout_probability CAPTION_DROPOUT_PROBABILITY]
                [--tokenizer_max_length TOKENIZER_MAX_LENGTH]
                [--validation_step_interval VALIDATION_STEP_INTERVAL]
                [--validation_epoch_interval VALIDATION_EPOCH_INTERVAL]
                [--disable_benchmark [DISABLE_BENCHMARK]]
                [--validation_prompt VALIDATION_PROMPT]
                [--num_validation_images NUM_VALIDATION_IMAGES]
                [--num_eval_images NUM_EVAL_IMAGES]
                [--eval_steps_interval EVAL_STEPS_INTERVAL]
                [--eval_epoch_interval EVAL_EPOCH_INTERVAL]
                [--eval_timesteps EVAL_TIMESTEPS]
                [--eval_dataset_pooling [EVAL_DATASET_POOLING]]
                [--evaluation_type {none,clip}]
                [--pretrained_evaluation_model_name_or_path PRETRAINED_EVALUATION_MODEL_NAME_OR_PATH]
                [--validation_guidance VALIDATION_GUIDANCE]
                [--validation_num_inference_steps VALIDATION_NUM_INFERENCE_STEPS]
                [--validation_on_startup [VALIDATION_ON_STARTUP]]
                [--validation_using_datasets [VALIDATION_USING_DATASETS]]
                [--validation_torch_compile [VALIDATION_TORCH_COMPILE]]
                [--validation_guidance_real VALIDATION_GUIDANCE_REAL]
                [--validation_no_cfg_until_timestep VALIDATION_NO_CFG_UNTIL_TIMESTEP]
                [--validation_negative_prompt VALIDATION_NEGATIVE_PROMPT]
                [--validation_randomize [VALIDATION_RANDOMIZE]]
                [--validation_seed VALIDATION_SEED]
                [--validation_disable [VALIDATION_DISABLE]]
                [--validation_prompt_library [VALIDATION_PROMPT_LIBRARY]]
                [--user_prompt_library USER_PROMPT_LIBRARY]
                [--eval_dataset_id EVAL_DATASET_ID]
                [--validation_stitch_input_location {left,right}]
                [--validation_guidance_rescale VALIDATION_GUIDANCE_RESCALE]
                [--validation_disable_unconditional [VALIDATION_DISABLE_UNCONDITIONAL]]
                [--validation_guidance_skip_layers VALIDATION_GUIDANCE_SKIP_LAYERS]
                [--validation_guidance_skip_layers_start VALIDATION_GUIDANCE_SKIP_LAYERS_START]
                [--validation_guidance_skip_layers_stop VALIDATION_GUIDANCE_SKIP_LAYERS_STOP]
                [--validation_guidance_skip_scale VALIDATION_GUIDANCE_SKIP_SCALE]
                [--validation_lycoris_strength VALIDATION_LYCORIS_STRENGTH]
                [--validation_noise_scheduler {ddim,ddpm,euler,euler-a,unipc,dpm++,perflow}]
                [--validation_num_video_frames VALIDATION_NUM_VIDEO_FRAMES]
                [--validation_audio_only [VALIDATION_AUDIO_ONLY]]
                [--validation_resolution VALIDATION_RESOLUTION]
                [--validation_seed_source {cpu,gpu}]
                [--i_know_what_i_am_doing [I_KNOW_WHAT_I_AM_DOING]]
                [--flow_sigmoid_scale FLOW_SIGMOID_SCALE]
                [--flux_fast_schedule [FLUX_FAST_SCHEDULE]]
                [--flow_use_uniform_schedule [FLOW_USE_UNIFORM_SCHEDULE]]
                [--flow_use_beta_schedule [FLOW_USE_BETA_SCHEDULE]]
                [--flow_beta_schedule_alpha FLOW_BETA_SCHEDULE_ALPHA]
                [--flow_beta_schedule_beta FLOW_BETA_SCHEDULE_BETA]
                [--flow_schedule_shift FLOW_SCHEDULE_SHIFT]
                [--flow_schedule_auto_shift [FLOW_SCHEDULE_AUTO_SHIFT]]
                [--flux_guidance_mode {constant,random-range}]
                [--flux_attention_masked_training [FLUX_ATTENTION_MASKED_TRAINING]]
                [--flux_guidance_value FLUX_GUIDANCE_VALUE]
                [--flux_guidance_min FLUX_GUIDANCE_MIN]
                [--flux_guidance_max FLUX_GUIDANCE_MAX]
                [--t5_padding {zero,unmodified}]
                [--sd3_clip_uncond_behaviour {empty_string,zero}]
                [--sd3_t5_uncond_behaviour {empty_string,zero}]
                [--soft_min_snr_sigma_data SOFT_MIN_SNR_SIGMA_DATA]
                [--mixed_precision {no,fp16,bf16,fp8}]
                [--attention_mechanism {diffusers,xformers,flash-attn,flash-attn-2,flash-attn-3,flash-attn-3-varlen,flex,cudnn,native-efficient,native-flash,native-math,native-npu,native-xla,sla,sageattention,sageattention-int8-fp16-triton,sageattention-int8-fp16-cuda,sageattention-int8-fp8-cuda}]
                [--sageattention_usage {training,inference,training+inference}]
                [--disable_tf32 [DISABLE_TF32]]
                [--set_grads_to_none [SET_GRADS_TO_NONE]]
                [--noise_offset NOISE_OFFSET]
                [--noise_offset_probability NOISE_OFFSET_PROBABILITY]
                [--input_perturbation INPUT_PERTURBATION]
                [--input_perturbation_steps INPUT_PERTURBATION_STEPS]
                [--lr_end LR_END] [--lr_scale [LR_SCALE]]
                [--lr_scale_sqrt [LR_SCALE_SQRT]]
                [--ignore_final_epochs [IGNORE_FINAL_EPOCHS]]
                [--freeze_encoder_before FREEZE_ENCODER_BEFORE]
                [--freeze_encoder_after FREEZE_ENCODER_AFTER]
                [--freeze_encoder_strategy {before,between,after}]
                [--layer_freeze_strategy {none,bitfit}]
                [--fully_unload_text_encoder [FULLY_UNLOAD_TEXT_ENCODER]]
                [--save_text_encoder [SAVE_TEXT_ENCODER]]
                [--text_encoder_limit TEXT_ENCODER_LIMIT]
                [--prepend_instance_prompt [PREPEND_INSTANCE_PROMPT]]
                [--only_instance_prompt [ONLY_INSTANCE_PROMPT]]
                [--data_aesthetic_score DATA_AESTHETIC_SCORE]
                [--delete_unwanted_images [DELETE_UNWANTED_IMAGES]]
                [--delete_problematic_images [DELETE_PROBLEMATIC_IMAGES]]
                [--disable_bucket_pruning [DISABLE_BUCKET_PRUNING]]
                [--disable_segmented_timestep_sampling [DISABLE_SEGMENTED_TIMESTEP_SAMPLING]]
                [--preserve_data_backend_cache [PRESERVE_DATA_BACKEND_CACHE]]
                [--override_dataset_config [OVERRIDE_DATASET_CONFIG]]
                [--cache_dir CACHE_DIR] [--cache_dir_text CACHE_DIR_TEXT]
                [--cache_dir_vae CACHE_DIR_VAE]
                [--compress_disk_cache [COMPRESS_DISK_CACHE]]
                [--aspect_bucket_disable_rebuild [ASPECT_BUCKET_DISABLE_REBUILD]]
                [--keep_vae_loaded [KEEP_VAE_LOADED]]
                [--skip_file_discovery SKIP_FILE_DISCOVERY]
                [--data_backend_sampling {uniform,auto-weighting}]
                [--image_processing_batch_size IMAGE_PROCESSING_BATCH_SIZE]
                [--write_batch_size WRITE_BATCH_SIZE]
                [--read_batch_size READ_BATCH_SIZE]
                [--enable_multiprocessing [ENABLE_MULTIPROCESSING]]
                [--max_workers MAX_WORKERS]
                [--aws_max_pool_connections AWS_MAX_POOL_CONNECTIONS]
                [--torch_num_threads TORCH_NUM_THREADS]
                [--dataloader_prefetch [DATALOADER_PREFETCH]]
                [--dataloader_prefetch_qlen DATALOADER_PREFETCH_QLEN]
                [--aspect_bucket_worker_count ASPECT_BUCKET_WORKER_COUNT]
                [--aspect_bucket_alignment {8,16,24,32,64}]
                [--minimum_image_size MINIMUM_IMAGE_SIZE]
                [--maximum_image_size MAXIMUM_IMAGE_SIZE]
                [--target_downsample_size TARGET_DOWNSAMPLE_SIZE]
                [--max_upscale_threshold MAX_UPSCALE_THRESHOLD]
                [--metadata_update_interval METADATA_UPDATE_INTERVAL]
                [--debug_aspect_buckets [DEBUG_ASPECT_BUCKETS]]
                [--debug_dataset_loader [DEBUG_DATASET_LOADER]]
                [--print_filenames [PRINT_FILENAMES]]
                [--print_sampler_statistics [PRINT_SAMPLER_STATISTICS]]
                [--timestep_bias_strategy {earlier,later,range,none}]
                [--timestep_bias_begin TIMESTEP_BIAS_BEGIN]
                [--timestep_bias_end TIMESTEP_BIAS_END]
                [--timestep_bias_multiplier TIMESTEP_BIAS_MULTIPLIER]
                [--timestep_bias_portion TIMESTEP_BIAS_PORTION]
                [--training_scheduler_timestep_spacing {leading,linspace,trailing}]
                [--inference_scheduler_timestep_spacing {leading,linspace,trailing}]
                [--loss_type {l2,huber,smooth_l1}]
                [--huber_schedule {snr,exponential,constant}]
                [--huber_c HUBER_C] [--snr_gamma SNR_GAMMA]
                [--masked_loss_probability MASKED_LOSS_PROBABILITY]
                [--hidream_use_load_balancing_loss [HIDREAM_USE_LOAD_BALANCING_LOSS]]
                [--hidream_load_balancing_loss_weight HIDREAM_LOAD_BALANCING_LOSS_WEIGHT]
                [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2]
                [--optimizer_beta1 OPTIMIZER_BETA1]
                [--optimizer_beta2 OPTIMIZER_BETA2]
                [--optimizer_cpu_offload_method {none}]
                [--gradient_precision {unmodified,fp32}]
                [--adam_weight_decay ADAM_WEIGHT_DECAY]
                [--adam_epsilon ADAM_EPSILON] [--prodigy_steps PRODIGY_STEPS]
                [--max_grad_norm MAX_GRAD_NORM]
                [--grad_clip_method {value,norm}]
                [--optimizer_offload_gradients [OPTIMIZER_OFFLOAD_GRADIENTS]]
                [--fuse_optimizer [FUSE_OPTIMIZER]]
                [--optimizer_release_gradients [OPTIMIZER_RELEASE_GRADIENTS]]
                [--push_to_hub [PUSH_TO_HUB]]
                [--push_to_hub_background [PUSH_TO_HUB_BACKGROUND]]
                [--push_checkpoints_to_hub [PUSH_CHECKPOINTS_TO_HUB]]
                [--publishing_config PUBLISHING_CONFIG]
                [--hub_model_id HUB_MODEL_ID]
                [--model_card_private [MODEL_CARD_PRIVATE]]
                [--model_card_safe_for_work [MODEL_CARD_SAFE_FOR_WORK]]
                [--model_card_note MODEL_CARD_NOTE]
                [--modelspec_comment MODELSPEC_COMMENT]
                [--report_to {tensorboard,wandb,comet_ml,all,none}]
                [--checkpoint_step_interval CHECKPOINT_STEP_INTERVAL]
                [--checkpoint_epoch_interval CHECKPOINT_EPOCH_INTERVAL]
                [--checkpointing_rolling_steps CHECKPOINTING_ROLLING_STEPS]
                [--checkpointing_use_tempdir [CHECKPOINTING_USE_TEMPDIR]]
                [--checkpoints_rolling_total_limit CHECKPOINTS_ROLLING_TOTAL_LIMIT]
                [--tracker_run_name TRACKER_RUN_NAME]
                [--tracker_project_name TRACKER_PROJECT_NAME]
                [--tracker_image_layout {gallery,table}]
                [--enable_watermark [ENABLE_WATERMARK]]
                [--framerate FRAMERATE]
                [--seed_for_each_device [SEED_FOR_EACH_DEVICE]]
                [--snr_weight SNR_WEIGHT]
                [--rescale_betas_zero_snr [RESCALE_BETAS_ZERO_SNR]]
                [--webhook_config WEBHOOK_CONFIG]
                [--webhook_reporting_interval WEBHOOK_REPORTING_INTERVAL]
                [--distillation_method {lcm,dcm,dmd,perflow}]
                [--distillation_config DISTILLATION_CONFIG]
                [--ema_validation {none,ema_only,comparison}]
                [--local_rank LOCAL_RANK] [--ltx_train_mode {t2v,i2v}]
                [--ltx_i2v_prob LTX_I2V_PROB]
                [--ltx_partial_noise_fraction LTX_PARTIAL_NOISE_FRACTION]
                [--ltx_protect_first_frame [LTX_PROTECT_FIRST_FRAME]]
                [--offload_param_path OFFLOAD_PARAM_PATH]
                [--offset_noise [OFFSET_NOISE]]
                [--quantize_activations [QUANTIZE_ACTIVATIONS]]
                [--refiner_training [REFINER_TRAINING]]
                [--refiner_training_invert_schedule [REFINER_TRAINING_INVERT_SCHEDULE]]
                [--refiner_training_strength REFINER_TRAINING_STRENGTH]
                [--sdxl_refiner_uses_full_range [SDXL_REFINER_USES_FULL_RANGE]]
                [--sana_complex_human_instruction SANA_COMPLEX_HUMAN_INSTRUCTION]

The following SimpleTuner command-line options are available:

options:
  -h, --help            show this help message and exit
  --model_family {kolors,auraflow,omnigen,flux,deepfloyd,cosmos2image,sana,qwen_image,pixart_sigma,sdxl,sd1x,sd2x,wan,hidream,sd3,lumina2,ltxvideo}
                        The base model architecture family to train
  --model_flavour MODEL_FLAVOUR
                        Specific variant of the selected model family
  --controlnet [CONTROLNET]
                        Train ControlNet (full or LoRA) branches alongside the
                        primary network.
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                        Optional override of the model checkpoint. Leave blank
                        to use the default path for the selected model
                        flavour.
  --output_dir OUTPUT_DIR
                        Directory where model checkpoints and logs will be
                        saved
  --logging_dir LOGGING_DIR
                        Directory for TensorBoard logs
  --model_type {full,lora}
                        Choose between full model training or LoRA adapter
                        training
  --seed SEED           Seed used for deterministic training behaviour
  --resolution RESOLUTION
                        Resolution for training images
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        Select checkpoint to resume training from
  --prediction_type {epsilon,v_prediction,sample,flow_matching}
                        The parameterization type for the diffusion model
  --pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH
                        Path to pretrained VAE model
  --vae_dtype {default,fp32,fp16,bf16}
                        Precision for VAE encoding/decoding. Lower precision
                        saves memory.
  --vae_cache_ondemand [VAE_CACHE_ONDEMAND]
                        Process VAE latents during training instead of
                        precomputing them
  --vae_cache_disable [VAE_CACHE_DISABLE]
                        Implicitly enables on-demand caching and disables
                        writing embeddings to disk.
  --accelerator_cache_clear_interval ACCELERATOR_CACHE_CLEAR_INTERVAL
                        Clear the cache from VRAM every X steps to prevent
                        memory leaks
  --aspect_bucket_rounding {1,2,3,4,5,6,7,8,9}
                        Number of decimal places to round aspect ratios to for
                        bucket creation
  --base_model_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for loading the base model. Lower precision
                        saves memory.
  --text_encoder_1_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_2_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_3_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_4_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --gradient_checkpointing_interval GRADIENT_CHECKPOINTING_INTERVAL
                        Checkpoint every N transformer blocks
  --offload_during_startup [OFFLOAD_DURING_STARTUP]
                        Offload text encoders to CPU during VAE caching
  --quantize_via {cpu,accelerator,pipeline}
                        Where to perform model quantization
  --quantization_config QUANTIZATION_CONFIG
                        JSON or file path describing Diffusers quantization
                        config for pipeline quantization
  --fuse_qkv_projections [FUSE_QKV_PROJECTIONS]
                        Enables Flash Attention 3 when supported; otherwise
                        falls back to PyTorch SDPA.
  --control [CONTROL]   Enable channel-wise control style training
  --controlnet_custom_config CONTROLNET_CUSTOM_CONFIG
                        Custom configuration for ControlNet models
  --controlnet_model_name_or_path CONTROLNET_MODEL_NAME_OR_PATH
                        Path to ControlNet model weights to preload
  --tread_config TREAD_CONFIG
                        Configuration for TREAD training method
  --pretrained_transformer_model_name_or_path PRETRAINED_TRANSFORMER_MODEL_NAME_OR_PATH
                        Path to pretrained transformer model
  --pretrained_transformer_subfolder PRETRAINED_TRANSFORMER_SUBFOLDER
                        Subfolder containing transformer model weights
  --pretrained_unet_model_name_or_path PRETRAINED_UNET_MODEL_NAME_OR_PATH
                        Path to pretrained UNet model
  --pretrained_unet_subfolder PRETRAINED_UNET_SUBFOLDER
                        Subfolder containing UNet model weights
  --pretrained_t5_model_name_or_path PRETRAINED_T5_MODEL_NAME_OR_PATH
                        Path to pretrained T5 model
  --pretrained_gemma_model_name_or_path PRETRAINED_GEMMA_MODEL_NAME_OR_PATH
                        Path to pretrained Gemma model
  --revision REVISION   Git branch/tag/commit for model version
  --variant VARIANT     Model variant (e.g., fp16, bf16)
  --base_model_default_dtype {bf16,fp32}
                        Default precision for quantized base model weights
  --unet_attention_slice [UNET_ATTENTION_SLICE]
                        Enable attention slicing for SDXL UNet
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Number of times to iterate through the entire dataset
  --max_train_steps MAX_TRAIN_STEPS
                        Maximum number of training steps (0 = use epochs
                        instead)
  --train_batch_size TRAIN_BATCH_SIZE
                        Number of samples processed per forward/backward pass
                        (per device).
  --learning_rate LEARNING_RATE
                        Base learning rate for training
  --optimizer {adamw_bf16,ao-adamw8bit,ao-adamw4bit,ao-adamfp8,ao-adamwfp8,adamw_schedulefree,adamw_schedulefree+aggressive,adamw_schedulefree+no_kahan,optimi-stableadamw,optimi-adamw,optimi-lion,optimi-radam,optimi-ranger,optimi-adan,optimi-adam,optimi-sgd,soap,prodigy}
                        Optimization algorithm for training
  --optimizer_config OPTIMIZER_CONFIG
                        Comma-separated key=value pairs forwarded to the
                        selected optimizer
  --lr_scheduler {linear,sine,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
                        How learning rate changes during training
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of steps to accumulate gradients
  --lr_warmup_steps LR_WARMUP_STEPS
                        Number of steps to gradually increase LR from 0
  --checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT
                        Maximum number of checkpoints to keep on disk
  --gradient_checkpointing [GRADIENT_CHECKPOINTING]
                        Trade compute for memory during training
  --train_text_encoder [TRAIN_TEXT_ENCODER]
                        Also train the text encoder (CLIP) model
  --text_encoder_lr TEXT_ENCODER_LR
                        Separate learning rate for text encoder
  --lr_num_cycles LR_NUM_CYCLES
                        Number of cosine annealing cycles
  --lr_power LR_POWER   Power for polynomial decay scheduler
  --use_soft_min_snr [USE_SOFT_MIN_SNR]
                        Use soft clamping instead of hard clamping for Min-SNR
  --use_ema [USE_EMA]   Maintain an exponential moving average copy of the
                        model during training.
  --ema_device {accelerator,cpu}
                        Where to keep the EMA weights in-between updates.
  --ema_cpu_only [EMA_CPU_ONLY]
                        Keep EMA weights exclusively on CPU even when
                        ema_device would normally move them.
  --ema_update_interval EMA_UPDATE_INTERVAL
                        Update EMA weights every N optimizer steps
  --ema_foreach_disable [EMA_FOREACH_DISABLE]
                        Fallback to standard tensor ops instead of
                        torch.foreach updates.
  --ema_decay EMA_DECAY
                        Smoothing factor for EMA updates (closer to 1.0 =
                        slower drift).
  --lora_rank LORA_RANK
                        Dimension of LoRA update matrices
  --lora_alpha LORA_ALPHA
                        Scaling factor for LoRA updates
  --lora_type {standard,lycoris}
                        LoRA implementation type
  --lora_dropout LORA_DROPOUT
                        LoRA dropout randomly ignores neurons during training.
                        This can help prevent overfitting.
  --lora_init_type {default,gaussian,loftq,olora,pissa}
                        The initialization type for the LoRA model
  --peft_lora_mode {standard,singlora}
                        PEFT LoRA training mode
  --peft_lora_target_modules PEFT_LORA_TARGET_MODULES
                        JSON array (or path to a JSON file) listing PEFT
                        LoRA target module names. Overrides preset targets.
  --singlora_ramp_up_steps SINGLORA_RAMP_UP_STEPS
                        Number of ramp-up steps for SingLoRA
  --slider_lora_target [SLIDER_LORA_TARGET]
                        Route LoRA training to slider-friendly targets
                        (self-attn + conv/time embeddings). Only affects
                        standard PEFT LoRA.
  --init_lora INIT_LORA
                        Specify an existing LoRA or LyCORIS safetensors file
                        to initialize the adapter
  --lycoris_config LYCORIS_CONFIG
                        Path to LyCORIS configuration JSON file
  --init_lokr_norm INIT_LOKR_NORM
                        Perturbed normal initialization for LyCORIS LoKr
                        layers
  --flux_lora_target {mmdit,context,context+ffs,all,all+ffs,ai-toolkit,tiny,nano,controlnet,all+ffs+embedder,all+ffs+embedder+controlnet}
                        Which layers to train in Flux models
  --use_dora [USE_DORA]
                        Enable DoRA (Weight-Decomposed LoRA)
  --resolution_type {pixel,area,pixel_area}
                        How to interpret the resolution value
  --data_backend_config DATA_BACKEND_CONFIG
                        Select a saved dataset configuration (managed in
                        Datasets & Environments tabs)
  --caption_strategy {filename,textfile,instance_prompt,parquet}
                        How to load captions for images
  --conditioning_multidataset_sampling {combined,random}
                        How to sample from multiple conditioning datasets
  --instance_prompt INSTANCE_PROMPT
                        Instance prompt for training
  --parquet_caption_column PARQUET_CAPTION_COLUMN
                        Column name containing captions in parquet files
  --parquet_filename_column PARQUET_FILENAME_COLUMN
                        Column name containing image paths in parquet files
  --ignore_missing_files [IGNORE_MISSING_FILES]
                        Continue training even if some files are missing
  --vae_cache_scan_behaviour {recreate,sync}
                        How to scan VAE cache for missing files
  --vae_enable_slicing [VAE_ENABLE_SLICING]
                        Enable VAE attention slicing for memory efficiency
  --vae_enable_tiling [VAE_ENABLE_TILING]
                        Enable VAE tiling for large images
  --vae_enable_patch_conv [VAE_ENABLE_PATCH_CONV]
                        Enable patch-based 3D conv for HunyuanVideo VAE to
                        reduce peak VRAM (slight slowdown)
  --vae_batch_size VAE_BATCH_SIZE
                        Batch size for VAE encoding during caching
  --caption_dropout_probability CAPTION_DROPOUT_PROBABILITY
                        Caption dropout will randomly drop captions and, for
                        SDXL, size conditioning inputs based on this
                        probability
  --tokenizer_max_length TOKENIZER_MAX_LENGTH
                        Override the tokenizer sequence length (advanced).
  --validation_step_interval VALIDATION_STEP_INTERVAL
                        Run validation every N training steps (deprecated alias: --validation_steps)
  --validation_epoch_interval VALIDATION_EPOCH_INTERVAL
                        Run validation every N training epochs
  --disable_benchmark [DISABLE_BENCHMARK]
                        Skip generating baseline comparison images before
                        training starts
  --validation_prompt VALIDATION_PROMPT
                        Prompt to use for validation images
  --num_validation_images NUM_VALIDATION_IMAGES
                        Number of images to generate per validation
  --num_eval_images NUM_EVAL_IMAGES
                        Number of images to generate for evaluation metrics
  --eval_steps_interval EVAL_STEPS_INTERVAL
                        Run evaluation every N training steps
  --eval_epoch_interval EVAL_EPOCH_INTERVAL
                        Run evaluation every N training epochs (decimals run
                        multiple times per epoch)
  --eval_timesteps EVAL_TIMESTEPS
                        Number of timesteps for evaluation
  --eval_dataset_pooling [EVAL_DATASET_POOLING]
                        Combine evaluation metrics from all datasets into a
                        single chart
  --evaluation_type {none,clip}
                        Type of evaluation metrics to compute
  --pretrained_evaluation_model_name_or_path PRETRAINED_EVALUATION_MODEL_NAME_OR_PATH
                        Path to pretrained model for evaluation metrics
  --validation_guidance VALIDATION_GUIDANCE
                        CFG guidance scale for validation images
  --validation_num_inference_steps VALIDATION_NUM_INFERENCE_STEPS
                        Number of diffusion steps for validation renders
  --validation_on_startup [VALIDATION_ON_STARTUP]
                        Run validation on the base model before training
                        starts
  --validation_using_datasets [VALIDATION_USING_DATASETS]
                        Use random images from training datasets for
                        validation
  --validation_torch_compile [VALIDATION_TORCH_COMPILE]
                        Use torch.compile() on validation pipeline for speed
  --validation_guidance_real VALIDATION_GUIDANCE_REAL
                        CFG value for distilled models (e.g., FLUX schnell)
  --validation_no_cfg_until_timestep VALIDATION_NO_CFG_UNTIL_TIMESTEP
                        Skip CFG for initial timesteps (Flux only)
  --validation_negative_prompt VALIDATION_NEGATIVE_PROMPT
                        Negative prompt for validation images
  --validation_randomize [VALIDATION_RANDOMIZE]
                        Use random seeds for each validation
  --validation_seed VALIDATION_SEED
                        Fixed seed for reproducible validation images
  --validation_disable [VALIDATION_DISABLE]
                        Completely disable validation image generation
  --validation_prompt_library [VALIDATION_PROMPT_LIBRARY]
                        Use SimpleTuner's built-in prompt library
  --user_prompt_library USER_PROMPT_LIBRARY
                        Path to custom JSON prompt library
  --eval_dataset_id EVAL_DATASET_ID
                        Specific dataset to use for evaluation metrics
  --validation_stitch_input_location {left,right}
                        Where to place input image in img2img validations
  --validation_guidance_rescale VALIDATION_GUIDANCE_RESCALE
                        CFG rescale value for validation
  --validation_disable_unconditional [VALIDATION_DISABLE_UNCONDITIONAL]
                        Disable unconditional image generation during
                        validation
  --validation_guidance_skip_layers VALIDATION_GUIDANCE_SKIP_LAYERS
                        JSON list of transformer layers to skip during
                        classifier-free guidance
  --validation_guidance_skip_layers_start VALIDATION_GUIDANCE_SKIP_LAYERS_START
                        Starting layer index to skip guidance
  --validation_guidance_skip_layers_stop VALIDATION_GUIDANCE_SKIP_LAYERS_STOP
                        Ending layer index to skip guidance
  --validation_guidance_skip_scale VALIDATION_GUIDANCE_SKIP_SCALE
                        Scale guidance strength when applying layer skipping
  --validation_lycoris_strength VALIDATION_LYCORIS_STRENGTH
                        Strength multiplier for LyCORIS validation
  --validation_noise_scheduler {ddim,ddpm,euler,euler-a,unipc,dpm++,perflow}
                        Noise scheduler for validation
  --validation_num_video_frames VALIDATION_NUM_VIDEO_FRAMES
                        Number of frames for video validation
  --validation_audio_only [VALIDATION_AUDIO_ONLY]
                        Disable video generation during validation and emit
                        audio only
  --validation_resolution VALIDATION_RESOLUTION
                        Override resolution for validation images (pixels or
                        megapixels)
  --validation_seed_source {cpu,gpu}
                        Source device used to generate validation seeds
  --i_know_what_i_am_doing [I_KNOW_WHAT_I_AM_DOING]
                        Unlock experimental overrides and bypass built-in
                        safety limits.
  --flow_sigmoid_scale FLOW_SIGMOID_SCALE
                        Scale factor for sigmoid timestep sampling for flow-
                        matching models.
  --flux_fast_schedule [FLUX_FAST_SCHEDULE]
                        Use experimental fast schedule for Flux training
  --flow_use_uniform_schedule [FLOW_USE_UNIFORM_SCHEDULE]
                        Use uniform schedule instead of sigmoid for flow-
                        matching
  --flow_use_beta_schedule [FLOW_USE_BETA_SCHEDULE]
                        Use beta schedule instead of sigmoid for flow-matching
  --flow_beta_schedule_alpha FLOW_BETA_SCHEDULE_ALPHA
                        Alpha value for beta schedule (default: 2.0)
  --flow_beta_schedule_beta FLOW_BETA_SCHEDULE_BETA
                        Beta value for beta schedule (default: 2.0)
  --flow_schedule_shift FLOW_SCHEDULE_SHIFT
                        Shift the noise schedule for flow-matching models
  --flow_schedule_auto_shift [FLOW_SCHEDULE_AUTO_SHIFT]
                        Auto-adjust schedule shift based on image resolution
  --flux_guidance_mode {constant,random-range}
                        Guidance mode for Flux training
  --flux_attention_masked_training [FLUX_ATTENTION_MASKED_TRAINING]
                        Enable attention masked training for Flux models
  --flux_guidance_value FLUX_GUIDANCE_VALUE
                        Guidance value for constant mode
  --flux_guidance_min FLUX_GUIDANCE_MIN
                        Minimum guidance value for random-range mode
  --flux_guidance_max FLUX_GUIDANCE_MAX
                        Maximum guidance value for random-range mode
  --t5_padding {zero,unmodified}
                        Padding behavior for T5 text encoder
  --sd3_clip_uncond_behaviour {empty_string,zero}
                        How SD3 handles unconditional prompts
  --sd3_t5_uncond_behaviour {empty_string,zero}
                        How SD3 T5 handles unconditional prompts
  --soft_min_snr_sigma_data SOFT_MIN_SNR_SIGMA_DATA
                        Sigma data for soft min SNR weighting
  --mixed_precision {no,fp16,bf16,fp8}
                        Precision for training computations
  --attention_mechanism {diffusers,xformers,flash-attn,flash-attn-2,flash-attn-3,flash-attn-3-varlen,flex,cudnn,native-efficient,native-flash,native-math,native-npu,native-xla,sla,sageattention,sageattention-int8-fp16-triton,sageattention-int8-fp16-cuda,sageattention-int8-fp8-cuda}
                        Attention computation backend
  --sageattention_usage {training,inference,training+inference}
                        When to use SageAttention
  --disable_tf32 [DISABLE_TF32]
                        Force IEEE FP32 precision (disables TF32) using
                        PyTorch's fp32_precision controls when available
  --set_grads_to_none [SET_GRADS_TO_NONE]
                        Set gradients to None instead of zero
  --noise_offset NOISE_OFFSET
                        Add noise offset to training
  --noise_offset_probability NOISE_OFFSET_PROBABILITY
                        Probability of applying noise offset
  --input_perturbation INPUT_PERTURBATION
                        Add additional noise only to the inputs fed to the
                        model during training
  --input_perturbation_steps INPUT_PERTURBATION_STEPS
                        Only apply input perturbation over the first N steps
                        with linear decay
  --lr_end LR_END       A polynomial learning rate will end up at this value
                        after the specified number of warmup steps
  --lr_scale [LR_SCALE]
                        Scale the learning rate by the number of GPUs,
                        gradient accumulation steps, and batch size
  --lr_scale_sqrt [LR_SCALE_SQRT]
                        If using --lr_scale, use the square root of (number of
                        GPUs * gradient accumulation steps * batch size)
  --ignore_final_epochs [IGNORE_FINAL_EPOCHS]
                        When provided, the max epoch counter will not
                        determine the end of the training run
  --freeze_encoder_before FREEZE_ENCODER_BEFORE
                        When using 'before' strategy, we will freeze layers
                        earlier than this
  --freeze_encoder_after FREEZE_ENCODER_AFTER
                        When using 'after' strategy, we will freeze layers
                        later than this
  --freeze_encoder_strategy {before,between,after}
                        When freezing the text encoder, we can use the
                        'before', 'between', or 'after' strategy
  --layer_freeze_strategy {none,bitfit}
                        When freezing parameters, we can use the 'none' or
                        'bitfit' strategy
  --fully_unload_text_encoder [FULLY_UNLOAD_TEXT_ENCODER]
                        If set, will fully unload the text_encoder from memory
                        when not in use
  --save_text_encoder [SAVE_TEXT_ENCODER]
                        If set, will save the text encoder after training
  --text_encoder_limit TEXT_ENCODER_LIMIT
                        When training the text encoder, we want to limit how
                        long it trains for to avoid catastrophic loss
  --prepend_instance_prompt [PREPEND_INSTANCE_PROMPT]
                        When determining the captions from the filename,
                        prepend the instance prompt as an enforced keyword
  --only_instance_prompt [ONLY_INSTANCE_PROMPT]
                        Use the instance prompt instead of the caption from
                        filename
  --data_aesthetic_score DATA_AESTHETIC_SCORE
                        Since currently we do not calculate aesthetic scores
                        for data, we will statically set it to one value. This
                        is only used by the SDXL Refiner
  --delete_unwanted_images [DELETE_UNWANTED_IMAGES]
                        If set, will delete images that are not of a minimum
                        size to save on disk space for large training runs
  --delete_problematic_images [DELETE_PROBLEMATIC_IMAGES]
                        If set, any images that error out during load will be
                        removed from the underlying storage medium
  --disable_bucket_pruning [DISABLE_BUCKET_PRUNING]
                        When training on very small datasets, you might not
                        care that the batch sizes will outpace your image
                        count. Setting this option will prevent SimpleTuner
                        from deleting your bucket lists that do not meet the
                        minimum image count requirements. Use at your own
                        risk, it may end up throwing off your statistics or
                        epoch tracking
  --disable_segmented_timestep_sampling [DISABLE_SEGMENTED_TIMESTEP_SAMPLING]
                        By default, the timestep schedule is divided into
                        roughly `train_batch_size` number of segments, and
                        then each of those are sampled from separately. This
                        improves the selection distribution, but may not be
                        desired in certain training scenarios, eg. when
                        limiting the timestep selection range
  --preserve_data_backend_cache [PRESERVE_DATA_BACKEND_CACHE]
                        For very large cloud storage buckets that will never
                        change, enabling this option will prevent the trainer
                        from scanning it at startup, by preserving the cache
                        files that we generate. Be careful when using this,
                        as, switching datasets can result in the preserved
                        cache being used, which would be problematic.
                        Currently, cache is not stored in the dataset itself
                        but rather, locally. This may change in a future
                        release
  --override_dataset_config [OVERRIDE_DATASET_CONFIG]
                        When provided, the dataset's config will not be
                        checked against the live backend config
  --cache_dir CACHE_DIR
                        The directory where the downloaded models and datasets
                        will be stored
  --cache_dir_text CACHE_DIR_TEXT
                        This is the path to a local directory that will
                        contain your text embed cache
  --cache_dir_vae CACHE_DIR_VAE
                        This is the path to a local directory that will
                        contain your VAE outputs
  --compress_disk_cache [COMPRESS_DISK_CACHE]
                        If set, will gzip-compress the disk cache for Pytorch
                        files. This will save substantial disk space, but may
                        slow down the training process
  --aspect_bucket_disable_rebuild [ASPECT_BUCKET_DISABLE_REBUILD]
                        When using a randomised aspect bucket list, the VAE
                        and aspect cache are rebuilt on each epoch. With a
                        large and diverse enough dataset, rebuilding the
                        aspect list may take a long time, and this may be
                        undesirable. This option will not override
                        vae_cache_clear_each_epoch. If both options are
                        provided, only the VAE cache will be rebuilt
  --keep_vae_loaded [KEEP_VAE_LOADED]
                        If set, will keep the VAE loaded in memory. This can
                        reduce disk churn, but consumes VRAM during the
                        forward pass
  --skip_file_discovery SKIP_FILE_DISCOVERY
                        Comma-separated values of which stages to skip
                        discovery for. Skipping any stage will speed up
                        resumption, but will increase the risk of errors, as
                        missing images or incorrectly bucketed images may not
                        be caught. Valid options: aspect, vae, text, metadata
  --data_backend_sampling {uniform,auto-weighting}
                        When using multiple data backends, the sampling
                        weighting can be set to 'uniform' or 'auto-weighting'
  --image_processing_batch_size IMAGE_PROCESSING_BATCH_SIZE
                        When resizing and cropping images, we do it in
                        parallel using processes or threads. This defines how
                        many images will be read into the queue before they
                        are processed
  --write_batch_size WRITE_BATCH_SIZE
                        When using certain storage backends, it is better to
                        batch smaller writes rather than continuous
                        dispatching. In SimpleTuner, write batching is
                        currently applied during VAE caching, when many small
                        objects are written. This mostly applies to S3, but
                        some shared server filesystems may benefit as well.
                        Default: 64
  --read_batch_size READ_BATCH_SIZE
                        Used by the VAE cache to prefetch image data. This is
                        the number of images to read ahead
  --enable_multiprocessing [ENABLE_MULTIPROCESSING]
                        If set, will use processes instead of threads during
                        metadata caching operations
  --max_workers MAX_WORKERS
                        How many active threads or processes to run during VAE
                        caching
  --aws_max_pool_connections AWS_MAX_POOL_CONNECTIONS
                        When using AWS backends, the maximum number of
                        connections to keep open to the S3 bucket at a single
                        time
  --torch_num_threads TORCH_NUM_THREADS
                        The number of threads to use for PyTorch operations.
                        This is not the same as the number of workers
  --dataloader_prefetch [DATALOADER_PREFETCH]
                        When provided, the dataloader will read-ahead and
                        attempt to retrieve latents, text embeds, and other
                        metadata ahead of the time when the batch is required,
                        so that it can be immediately available
  --dataloader_prefetch_qlen DATALOADER_PREFETCH_QLEN
                        Set the number of prefetched batches
  --aspect_bucket_worker_count ASPECT_BUCKET_WORKER_COUNT
                        The number of workers to use for aspect bucketing.
                        This is a CPU-bound task, so the number of workers
                        should be set to the number of CPU threads available.
                        If you use an I/O bound backend, an even higher value
                        may make sense. Default: 12
  --aspect_bucket_alignment {8,16,24,32,64}
                        When training diffusion models, the image sizes
                        generally must align to a 64 pixel interval
  --minimum_image_size MINIMUM_IMAGE_SIZE
                        The minimum resolution for both sides of input images
  --maximum_image_size MAXIMUM_IMAGE_SIZE
                        When cropping images that are excessively large, the
                        entire scene context may be lost, eg. the crop might
                        just end up being a portion of the background. To
                        avoid this, a maximum image size may be provided,
                        which will result in very-large images being
                        downsampled before cropping them. This value uses
                        --resolution_type to determine whether it is a pixel
                        edge or megapixel value
  --target_downsample_size TARGET_DOWNSAMPLE_SIZE
                        When using --maximum_image_size, very-large images
                        exceeding that value will be downsampled to this
                        target size before cropping
  --max_upscale_threshold MAX_UPSCALE_THRESHOLD
                        Limit upscaling of small images to prevent quality
                        degradation (opt-in). When set, filters out aspect
                        buckets requiring upscaling beyond this threshold.
                        For example, 0.2 allows up to 20% upscaling. Default
                        (None) allows unlimited upscaling. Must be between 0
                        and 1.
  --metadata_update_interval METADATA_UPDATE_INTERVAL
                        When generating the aspect bucket indicies, we want to
                        save it every X seconds
  --debug_aspect_buckets [DEBUG_ASPECT_BUCKETS]
                        If set, will print excessive debugging for aspect
                        bucket operations
  --debug_dataset_loader [DEBUG_DATASET_LOADER]
                        If set, will print excessive debugging for data loader
                        operations
  --print_filenames [PRINT_FILENAMES]
                        If any image files are stopping the process eg. due to
                        corruption or truncation, this will help identify
                        which is at fault
  --print_sampler_statistics [PRINT_SAMPLER_STATISTICS]
                        If provided, will print statistics about the dataset
                        sampler. This is useful for debugging
  --timestep_bias_strategy {earlier,later,range,none}
                        Strategy for biasing timestep sampling
  --timestep_bias_begin TIMESTEP_BIAS_BEGIN
                        Beginning of timestep bias range
  --timestep_bias_end TIMESTEP_BIAS_END
                        End of timestep bias range
  --timestep_bias_multiplier TIMESTEP_BIAS_MULTIPLIER
                        Multiplier for timestep bias probability
  --timestep_bias_portion TIMESTEP_BIAS_PORTION
                        Portion of training steps to apply timestep bias
  --training_scheduler_timestep_spacing {leading,linspace,trailing}
                        Timestep spacing for training scheduler
  --inference_scheduler_timestep_spacing {leading,linspace,trailing}
                        Timestep spacing for inference scheduler
  --loss_type {l2,huber,smooth_l1}
                        Loss function for training
  --huber_schedule {snr,exponential,constant}
                        Schedule for Huber loss transition threshold
  --huber_c HUBER_C     Transition point between L2 and L1 regions for Huber
                        loss
  --snr_gamma SNR_GAMMA
                        SNR weighting gamma value (0 = disabled)
  --masked_loss_probability MASKED_LOSS_PROBABILITY
                        Probability of applying masked loss weighting per
                        batch
  --hidream_use_load_balancing_loss [HIDREAM_USE_LOAD_BALANCING_LOSS]
                        Apply experimental load balancing loss when training
                        HiDream models.
  --hidream_load_balancing_loss_weight HIDREAM_LOAD_BALANCING_LOSS_WEIGHT
                        Strength multiplier for HiDream load balancing loss.
  --adam_beta1 ADAM_BETA1
                        First moment decay rate for Adam optimizers
  --adam_beta2 ADAM_BETA2
                        Second moment decay rate for Adam optimizers
  --optimizer_beta1 OPTIMIZER_BETA1
                        First moment decay rate for optimizers
  --optimizer_beta2 OPTIMIZER_BETA2
                        Second moment decay rate for optimizers
  --optimizer_cpu_offload_method {none}
                        Method for CPU offloading optimizer states
  --gradient_precision {unmodified,fp32}
                        Precision for gradient computation
  --adam_weight_decay ADAM_WEIGHT_DECAY
                        L2 regularisation strength for Adam-family optimizers.
  --adam_epsilon ADAM_EPSILON
                        Small constant added for numerical stability.
  --prodigy_steps PRODIGY_STEPS
                        Number of steps Prodigy should spend adapting its
                        learning rate.
  --max_grad_norm MAX_GRAD_NORM
                        Gradient clipping threshold to prevent exploding
                        gradients.
  --grad_clip_method {value,norm}
                        Strategy for applying max_grad_norm during clipping.
  --optimizer_offload_gradients [OPTIMIZER_OFFLOAD_GRADIENTS]
                        Move optimizer gradients to CPU to save GPU memory.
  --fuse_optimizer [FUSE_OPTIMIZER]
                        Enable fused kernels when offloading to reduce memory
                        overhead.
  --optimizer_release_gradients [OPTIMIZER_RELEASE_GRADIENTS]
                        Free gradient tensors immediately after optimizer step
                        when using Optimi optimizers.
  --push_to_hub [PUSH_TO_HUB]
                        Automatically upload the trained model to your Hugging
                        Face Hub repository.
  --push_to_hub_background [PUSH_TO_HUB_BACKGROUND]
                        Run Hub uploads in a background worker so training is
                        not blocked while pushing.
  --push_checkpoints_to_hub [PUSH_CHECKPOINTS_TO_HUB]
                        Upload intermediate checkpoints to the same Hugging
                        Face repository during training.
  --publishing_config PUBLISHING_CONFIG
                        Optional JSON/file path describing additional
                        publishing targets (S3/Backblaze B2/Azure Blob/Dropbox).
  --hub_model_id HUB_MODEL_ID
                        If left blank, SimpleTuner derives a name from the
                        project settings when pushing to Hub.
  --model_card_private [MODEL_CARD_PRIVATE]
                        Create the Hugging Face repository as private instead
                        of public.
  --model_card_safe_for_work [MODEL_CARD_SAFE_FOR_WORK]
                        Remove the default NSFW warning from the generated
                        model card on Hugging Face Hub.
  --model_card_note MODEL_CARD_NOTE
                        Optional note that appears at the top of the generated
                        model card.
  --modelspec_comment MODELSPEC_COMMENT
                        Text embedded in safetensors file metadata as
                        modelspec.comment, visible in external model viewers.
  --report_to {tensorboard,wandb,comet_ml,all,none}
                        Where to log training metrics
  --checkpoint_step_interval CHECKPOINT_STEP_INTERVAL
                        Save model checkpoint every N steps (deprecated alias: --checkpointing_steps)
  --checkpoint_epoch_interval CHECKPOINT_EPOCH_INTERVAL
                        Save model checkpoint every N epochs
  --checkpointing_rolling_steps CHECKPOINTING_ROLLING_STEPS
                        Rolling checkpoint window size for continuous
                        checkpointing
  --checkpointing_use_tempdir [CHECKPOINTING_USE_TEMPDIR]
                        Use temporary directory for checkpoint files before
                        final save
  --checkpoints_rolling_total_limit CHECKPOINTS_ROLLING_TOTAL_LIMIT
                        Maximum number of rolling checkpoints to keep
  --tracker_run_name TRACKER_RUN_NAME
                        Name for this training run in tracking platforms
  --tracker_project_name TRACKER_PROJECT_NAME
                        Project name in tracking platforms
  --tracker_image_layout {gallery,table}
                        How validation images are displayed in trackers
  --enable_watermark [ENABLE_WATERMARK]
                        Add invisible watermark to generated images
  --framerate FRAMERATE
                        Framerate for video model training
  --seed_for_each_device [SEED_FOR_EACH_DEVICE]
                        Use a unique deterministic seed per GPU instead of
                        sharing one seed across devices.
  --snr_weight SNR_WEIGHT
                        Weight factor for SNR-based loss scaling
  --rescale_betas_zero_snr [RESCALE_BETAS_ZERO_SNR]
                        Rescale betas for zero terminal SNR
  --webhook_config WEBHOOK_CONFIG
                        Path to webhook configuration file
  --webhook_reporting_interval WEBHOOK_REPORTING_INTERVAL
                        Interval for webhook reports (seconds)
  --distillation_method {lcm,dcm,dmd,perflow}
                        Method for model distillation
  --distillation_config DISTILLATION_CONFIG
                        Path to distillation configuration file
  --ema_validation {none,ema_only,comparison}
                        Control how EMA weights are used during validation
                        runs.
  --local_rank LOCAL_RANK
                        Local rank for distributed training
  --ltx_train_mode {t2v,i2v}
                        Training mode for LTX models
  --ltx_i2v_prob LTX_I2V_PROB
                        Probability of using image-to-video training for LTX
  --ltx_partial_noise_fraction LTX_PARTIAL_NOISE_FRACTION
                        Fraction of noise to add for LTX partial training
  --ltx_protect_first_frame [LTX_PROTECT_FIRST_FRAME]
                        Protect the first frame from noise in LTX training
  --offload_param_path OFFLOAD_PARAM_PATH
                        Path to offloaded parameter files
  --offset_noise [OFFSET_NOISE]
                        Enable offset-noise training
  --quantize_activations [QUANTIZE_ACTIVATIONS]
                        Quantize model activations during training
  --refiner_training [REFINER_TRAINING]
                        Enable refiner model training mode
  --refiner_training_invert_schedule [REFINER_TRAINING_INVERT_SCHEDULE]
                        Invert the noise schedule for refiner training
  --refiner_training_strength REFINER_TRAINING_STRENGTH
                        Strength of refiner training
  --sdxl_refiner_uses_full_range [SDXL_REFINER_USES_FULL_RANGE]
                        Use full timestep range for SDXL refiner
  --sana_complex_human_instruction SANA_COMPLEX_HUMAN_INSTRUCTION
                        Complex human instruction for Sana model training
```
