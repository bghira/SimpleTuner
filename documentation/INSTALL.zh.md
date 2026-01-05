# 设置

对于希望使用 Docker 或其他容器编排平台的用户，请先查看[此文档](DOCKER.md)。

## 安装

对于使用 Windows 10 或更新版本的用户，可以参考基于 Docker 和 WSL 的安装指南[此文档](DOCKER.md)。

### Pip 安装方法

您可以通过 pip 简单地安装 SimpleTuner，这是大多数用户推荐的方式：

```bash
# for CUDA
pip install 'simpletuner[cuda]'
# for ROCm
pip install 'simpletuner[rocm]'
# for Apple Silicon
pip install 'simpletuner[apple]'
# for CPU-only (not recommended)
pip install 'simpletuner[cpu]'
# for JPEG XL support (optional)
pip install 'simpletuner[jxl]'

# development requirements (optional, only for submitting PRs or running tests)
pip install 'simpletuner[dev]'
```

### Git 仓库方法

对于本地开发或测试，您可以克隆 SimpleTuner 仓库并设置 Python 虚拟环境：

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git

cd SimpleTuner

# if python --version shows 3.11 will have to upgrade to 3.12.
python3.12 -m venv .venv

source .venv/bin/activate
```

> ℹ️ 您可以通过在 `config/config.env` 文件中设置 `export VENV_PATH=/path/to/.venv` 来使用自定义的虚拟环境路径。

**注意：** 我们目前安装的是 `release` 分支；`main` 分支可能包含实验性功能，这些功能可能会有更好的效果或更低的内存使用。

使用自动平台检测安装 SimpleTuner：

```bash
# Basic installation (auto-detects CUDA/ROCm/Apple)
pip install -e .

# With JPEG XL support
pip install -e .[jxl]
```

**注意：** setup.py 会自动检测您的平台（CUDA/ROCm/Apple）并安装相应的依赖项。

#### NVIDIA Hopper / Blackwell 后续步骤

可选地，Hopper（或更新）设备可以使用 FlashAttention3 来改善使用 `torch.compile` 时的推理和训练性能。

您需要在 SimpleTuner 目录中激活虚拟环境后运行以下命令序列：

```bash
git clone https://github.com/Dao-AILab/flash-attention
pushd flash-attention
  pushd hopper
    python setup.py install
  popd
popd
```

> ⚠️ 目前 SimpleTuner 对 flash_attn 构建的管理支持不完善。这可能会在更新时出现问题，需要您不时手动重新运行此构建过程。

#### AMD ROCm 后续步骤

要使 AMD MI300X 可用，必须执行以下操作：

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
  python3 -m pip install --upgrade pip
  python3 -m pip install .
popd
```

> ℹ️ **ROCm 加速默认设置**：当 SimpleTuner 检测到启用 HIP 的 PyTorch 构建时，它会自动导出 `PYTORCH_TUNABLEOP_ENABLED=1`（除非您已经设置过），以便 TunableOp 内核可用。在 MI300/gfx94x 设备上，我们还默认设置 `HIPBLASLT_ALLOW_TF32=1`，无需手动调整环境即可启用 hipBLASLt 的 TF32 路径。

### 所有平台

- 2a. **选项一（推荐）**：运行 `simpletuner configure`
- 2b. **选项二**：将 `config/config.json.example` 复制到 `config/config.json`，然后填写详细信息。

> ⚠️ 对于位于无法便捷访问 Hugging Face Hub 的国家/地区的用户，您应该在 `~/.bashrc` 或 `~/.zshrc` 中添加 `HF_ENDPOINT=https://hf-mirror.com`，具体取决于您的系统使用哪个 `$SHELL`。

#### 多 GPU 训练 {#multiple-gpu-training}

SimpleTuner 现在通过 WebUI 包含**自动 GPU 检测和配置**。首次加载时，您将通过引导步骤来检测您的 GPU 并自动配置 Accelerate。

##### WebUI 自动检测（推荐）

当您首次启动 WebUI 或使用 `simpletuner configure` 时，您将遇到"Accelerate GPU 默认设置"引导步骤，它会：

1. **自动检测**系统上所有可用的 GPU
2. **显示 GPU 详细信息**，包括名称、内存和设备 ID
3. **推荐最佳设置**用于多 GPU 训练
4. **提供三种配置模式：**

   - **自动模式**（推荐）：使用所有检测到的 GPU 并采用最佳进程数
   - **手动模式**：选择特定的 GPU 或设置自定义进程数
   - **禁用模式**：仅使用单 GPU 训练

**工作原理：**
- 系统通过 CUDA/ROCm 检测您的 GPU 硬件
- 根据可用设备计算最佳 `--num_processes`
- 选择特定 GPU 时自动设置 `CUDA_VISIBLE_DEVICES`
- 保存您的偏好设置供将来训练使用

##### 手动配置

如果不使用 WebUI，您可以直接在 `config.json` 中控制 GPU 可见性：

```json
{
  "accelerate_visible_devices": [0, 1, 2],
  "num_processes": 3
}
```

这将把训练限制在 GPU 0、1 和 2 上，启动 3 个进程。

3. 如果您使用 `--report_to='wandb'`（默认设置），以下步骤将帮助您报告统计数据：

```bash
wandb login
```

按照打印的说明找到您的 API 密钥并进行配置。

完成后，您的任何训练会话和验证数据都将在 Weights & Biases 上可用。

> ℹ️ 如果您想完全禁用 Weights & Biases 或 Tensorboard 报告，请使用 `--report-to=none`


4. 使用 simpletuner 启动训练；日志将写入 `debug.log`

```bash
simpletuner train
```

> ⚠️ 此时，如果您使用了 `simpletuner configure`，您就完成了！如果没有——这些命令可以工作，但需要进一步配置。有关更多信息，请参阅[教程](TUTORIAL.md)。

### 运行单元测试

运行单元测试以确保安装已成功完成：

```bash
python -m unittest discover tests/
```

## 高级：多配置环境

对于训练多个模型或需要在不同数据集或设置之间快速切换的用户，启动时会检查两个环境变量。

使用方法：

```bash
simpletuner train env=default config_backend=env
```

- `env` 默认为 `default`，指向本指南帮助您配置的典型 `SimpleTuner/config/` 目录
  - 使用 `simpletuner train env=pixart` 将使用 `SimpleTuner/config/pixart` 目录来查找 `config.env`
- `config_backend` 默认为 `env`，使用本指南帮助您配置的典型 `config.env` 文件
  - 支持的选项：`env`、`json`、`toml`，或者如果您依赖手动运行 `train.py` 则使用 `cmd`
  - 使用 `simpletuner train config_backend=json` 将搜索 `SimpleTuner/config/config.json` 而不是 `config.env`
  - 类似地，`config_backend=toml` 将使用 `config.env`

您可以创建包含以下一个或两个值的 `config/config.env`：

```bash
ENV=default
CONFIG_BACKEND=json
```

它们将在后续运行中被记住。请注意，这些可以与[上面](#multiple-gpu-training)描述的多 GPU 选项一起添加。

## 训练数据

[Hugging Face Hub 上](https://huggingface.co/datasets/bghira/pseudo-camera-10k)提供了一个公开可用的数据集，包含约 10k 张带有文件名作为描述的图片，可直接用于 SimpleTuner。

您可以将图片组织在单个文件夹中，或整齐地组织到子目录中。

### 图片选择指南

**质量要求：**
- 无 JPEG 伪影或模糊图片——现代模型会捕捉到这些
- 避免颗粒状的 CMOS 传感器噪点（会出现在所有生成的图片中）
- 无水印、徽章或签名（这些会被学习）
- 电影帧通常由于压缩而不起作用（请使用制作剧照代替）

**技术规格：**
- 图片最好能被 64 整除（允许无需调整大小即可重用）
- 混合使用方形和非方形图片以获得平衡的能力
- 使用多样化、高质量的数据集以获得最佳结果

### 描述（Captioning）

SimpleTuner 提供了用于批量重命名文件的[描述脚本](/scripts/toolkit/README.md)。支持的描述格式：
- 文件名作为描述（默认）
- 使用 `--caption_strategy=textfile` 的文本文件
- JSONL、CSV 或高级元数据文件

**推荐的描述工具：**
- **InternVL2**：质量最佳但速度慢（小数据集）
- **BLIP3**：最佳轻量级选项，具有良好的指令遵循能力
- **Florence2**：最快但有些人不喜欢其输出

### 训练批次大小

您的最大批次大小取决于 VRAM 和分辨率：
```
vram use = batch size * resolution + base_requirements
```

**关键原则：**
- 使用最大的批次大小而不出现 VRAM 问题
- 更高的分辨率 = 更多 VRAM = 更小的批次大小
- 如果在 128x128 分辨率下批次大小为 1 都无法工作，则硬件不足

#### 多 GPU 数据集要求

使用多个 GPU 训练时，您的数据集必须足够大以满足**有效批次大小**：
```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

**示例：** 使用 4 个 GPU 和 `train_batch_size=4`，每个宽高比桶至少需要 16 个样本。

**小数据集的解决方案：**
- 使用 `--allow_dataset_oversubscription` 自动调整重复次数
- 在数据加载器配置中手动设置 `repeats`
- 减少批次大小或 GPU 数量

完整详情请参阅 [DATALOADER.md](DATALOADER.md#multi-gpu-training-and-dataset-sizing)。

## 发布到 Hugging Face Hub

要在完成时自动将模型推送到 Hub，请在 `config/config.json` 中添加：

```json
{
  "push_to_hub": true,
  "hub_model_name": "your-model-name"
}
```

训练前登录：
```bash
huggingface-cli login
```

## 调试

通过在 `config/config.env` 中添加以下内容来启用详细日志记录：

```bash
export SIMPLETUNER_LOG_LEVEL=DEBUG
export SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG
```

将在项目根目录中创建一个包含所有日志条目的 `debug.log` 文件。
