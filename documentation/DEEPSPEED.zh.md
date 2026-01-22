# DeepSpeed 卸载 / 多 GPU 训练

SimpleTuner v0.7 为使用 DeepSpeed ZeRO 1–3 阶段训练 SDXL 提供了初步支持。
在 v3.0 中，这一支持显著改进，加入了 WebUI 配置构建器、更好的优化器支持以及更完善的卸载管理。

> ⚠️ DeepSpeed 在 macOS（MPS）或 ROCm 系统上不可用。

**在 9237MiB 显存上训练 SDXL 1.0**：
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------|
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:08:00.0 Off |                  Off |
|  0%   43C    P2   100W / 450W |   9237MiB / 24564MiB |    100%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     11500      C   ...uner/.venv/bin/python3.13     9232MiB |
+-----------------------------------------------------------------------------+
```

这些显存节省来自 DeepSpeed ZeRO Stage 2 卸载。如果没有卸载，SDXL 的 U-Net 会消耗超过 24G 显存，导致 CUDA Out of Memory。

## 什么是 DeepSpeed？

ZeRO 是 **Zero Redundancy Optimizer** 的缩写。该技术通过将模型训练状态（权重、梯度、优化器状态）在可用设备（GPU 与 CPU）之间分区，从而减少每张 GPU 的内存消耗。

ZeRO 以阶段式优化实现，早期阶段的优化会在后续阶段继续可用。深入了解 ZeRO 可参阅原始 [论文](https://arxiv.org/abs/1910.02054v3)（1910.02054v3）。

## 已知问题

### LoRA 支持

由于 DeepSpeed 改变了模型保存流程，目前不支持使用 DeepSpeed 训练 LoRA。

未来版本可能会改变。

### 在已有检查点上启用/禁用 DeepSpeed

目前 SimpleTuner 中，若检查点最初未使用 DeepSpeed，则无法在恢复训练时**启用** DeepSpeed。

相反，若检查点使用了 DeepSpeed，则恢复训练时也无法**禁用** DeepSpeed。

规避方式是：在切换 DeepSpeed 启用状态前，先将训练流水线导出为完整模型权重。

由于 DeepSpeed 的优化器与常规选项差异较大，这种支持不太可能实现。

## DeepSpeed 阶段

DeepSpeed 提供三种优化级别，每增加一级都伴随更多开销。

尤其在多 GPU 训练中，DeepSpeed 内部的 CPU 传输目前并未高度优化。

因此建议选择能运行的**最低** DeepSpeed 级别。

### Stage 1

优化器状态（例如 Adam 的 32-bit 权重，以及一阶/二阶矩估计）在进程间分区，每个进程只更新其分区。

### Stage 2

用于更新模型权重的 32-bit 梯度也会分区，每个进程只保留与其优化器状态分区对应的梯度。

### Stage 3

16-bit 模型参数在进程间分区。ZeRO-3 会在前向与反向传播中自动收集并分区。

## 启用 DeepSpeed

[官方教程](https://www.deepspeed.ai/tutorials/zero/) 结构清晰，涵盖许多本文未涉及的场景。

### 方法 1：WebUI 配置构建器（推荐）

SimpleTuner 提供了用户友好的 WebUI 来配置 DeepSpeed：

1. 打开 SimpleTuner WebUI
2. 切换到 **Hardware** 标签并打开 **Accelerate & Distributed** 部分
3. 在 `DeepSpeed Config (JSON)` 字段旁点击 **DeepSpeed Builder** 按钮
4. 在交互界面中：
   - 选择 ZeRO 优化阶段（1、2 或 3）
   - 配置卸载选项（CPU、NVMe）
   - 选择优化器与调度器
   - 设置梯度累积与裁剪参数
5. 预览生成的 JSON 配置
6. 保存并应用配置

该构建器保持 JSON 结构一致，并在需要时将不支持的优化器替换为安全默认值，从而避免常见配置错误。

### 方法 2：手动 JSON 配置

若你更倾向于直接编辑，可在 `config.json` 中加入 DeepSpeed 配置：

```json
{
  "deepspeed_config": {
    "zero_optimization": {
      "stage": 2,
      "offload_param": {
        "device": "cpu",
        "pin_memory": true
      },
      "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true
      }
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "optimizer": {
      "type": "AdamW",
      "params": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01
      }
    },
    "scheduler": {
      "type": "WarmupLR",
      "params": {
        "warmup_min_lr": 0,
        "warmup_max_lr": 1e-4,
        "warmup_num_steps": 500
      }
    },
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 2
  }
}
```

**关键配置项：**

- `zero_optimization.stage`: 设为 1、2 或 3 对应不同 ZeRO 优化级别
- `offload_param.device`: 参数卸载位置（"cpu" 或 "nvme"）
- `offload_optimizer.device`: 优化器卸载位置（"cpu" 或 "nvme"）
- `optimizer.type`: 从支持的优化器中选择（AdamW、Adam、Adagrad、Lamb 等）
- `gradient_accumulation_steps`: 梯度累积步数

**NVMe 卸载示例：**
```json
{
  "deepspeed_config": {
    "zero_optimization": {
      "stage": 3,
      "offload_param": {
        "device": "nvme",
        "nvme_path": "/path/to/nvme/storage",
        "buffer_size": 100000000.0,
        "pin_memory": true
      }
    }
  }
}
```

### 方法 3：通过 accelerate config 手动配置

高级用户仍可通过 `accelerate config` 启用 DeepSpeed：

```
----------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
No distributed training
Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)? [yes/NO]:NO
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO
----------------------------------------------------------------------------------------------------------------------------
What should be your DeepSpeed's ZeRO optimization stage?
1
How many gradient accumulation steps you're passing in your script? [1]: 4
Do you want to use gradient clipping? [yes/NO]:
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]:
How many GPU(s) should be used for distributed training? [1]:
----------------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?bf16
accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml
```

这会生成如下 yaml 文件：

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 4
  zero3_init_flag: false
  zero_stage: 1
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## 配置 SimpleTuner

SimpleTuner 在使用 DeepSpeed 时不需要特别配置。

若使用 ZeRO stage 2 或 3 并启用 NVMe 卸载，可提供 `--offload_param_path=/path/to/offload`，将参数/优化器卸载文件写入专用分区。理想情况下是 NVMe 设备，但其他存储也可使用。

### 近期改进（v0.7+）

#### WebUI 配置构建器
SimpleTuner 的 WebUI 现包含完整的 DeepSpeed 配置构建器，可：
- 通过直观界面创建自定义 DeepSpeed JSON 配置
- 自动发现可用参数
- 在应用前可视化配置影响
- 保存并复用配置模板

#### 优化器支持增强
优化器名称归一化与校验得到改进：
- **支持的优化器**: AdamW, Adam, Adagrad, Lamb, OneBitAdam, OneBitLamb, ZeroOneAdam, MuAdam, MuAdamW, MuSGD, Lion, Muon
- **不支持的优化器**（会自动替换为 AdamW）: cpuadam, fusedadam
- 指定不支持的优化器时会自动回退并记录警告

#### 卸载管理改进
- **自动清理**：过期的 DeepSpeed 卸载交换目录会被自动移除，防止恢复状态损坏
- **NVMe 支持增强**：更好地处理 NVMe 卸载路径，并自动分配缓冲区大小
- **平台检测**：在不兼容平台（macOS/ROCm）上自动禁用 DeepSpeed

#### 配置验证
- 应用更改时自动规范化优化器名称与配置结构
- 对不支持的优化器选择与错误 JSON 提供安全保护
- 改进错误处理与日志以便排查问题

### DeepSpeed 优化器 / 学习率调度器

DeepSpeed 使用自己的学习率调度器，并默认使用高度优化的 AdamW（但不是 8bit）。在 DeepSpeed 中这点影响较小，因为更多工作会落在 CPU 上。

如果在 `default_config.yaml` 中配置了 `scheduler` 或 `optimizer`，将使用这些设置。若未定义，则默认使用 `AdamW` 与 `WarmUp` 作为优化器与调度器。

## 简要测试结果

使用 4090 24G GPU：

* 现在可在 1 百万像素（1024^2 像素面积）下使用批大小 8 训练完整 U-Net，显存仅 **13102MiB**
  * 每次迭代约 8 秒，1000 步训练可在 2 小时半内完成。
  * 如 DeepSpeed 教程所示，适当降低批大小可能更有利于将显存用于参数和优化器状态。
    * 但 SDXL 相对较小，若性能影响可接受，也可不遵循部分建议。
* 在 **128x128** 图像、批大小 8 时，训练显存可低至 **9237MiB**。这是一个较小众的用例，适合需要与潜空间 1:1 对应的像素风训练。

在这些条件下，可能会获得不同程度的成功，甚至可能在 1024x1024、批大小 1 时将完整 U-Net 训练塞进约 8GiB 显存（未测试）。

由于 SDXL 在多种图像分辨率与纵横比分布上训练，你还可以把像素面积降到 0.75 兆像素（约 768x768）以进一步优化内存。

# AMD 设备支持

我没有消费级或工作站级 AMD GPU，但有报告称 MI50（即将停止支持）及其他高端 Instinct 卡**可以**与 DeepSpeed 一起工作。AMD 维护了其实现的仓库。

## 故障排查

### 常见问题与解决方案

#### “DeepSpeed 恢复时崩溃”
**问题**：启用 DeepSpeed 卸载的检查点在恢复训练时崩溃。

**解决方案**：SimpleTuner 现在会自动清理过期的 DeepSpeed 卸载交换目录，避免损坏的恢复状态。该问题已在最新更新中解决。

#### “不支持的优化器错误”
**问题**：DeepSpeed 配置包含不支持的优化器名称。

**解决方案**：系统会自动规范化优化器名称，并将不支持的优化器（cpuadam、fusedadam）替换为 AdamW。当发生回退时会记录警告。

#### “此平台不支持 DeepSpeed”
**问题**：DeepSpeed 选项被禁用或不可用。

**解决方案**：DeepSpeed 仅支持 CUDA 系统。macOS（MPS）与 ROCm 会自动禁用，这是为了避免兼容性问题。

#### “NVMe 卸载路径问题”
**问题**：NVMe 卸载路径配置相关错误。

**解决方案**：确保 `--offload_param_path` 指向有效且有足够空间的目录。系统会自动处理缓冲区大小分配与路径验证。

#### “配置验证错误”
**问题**：DeepSpeed 配置参数无效。

**解决方案**：使用 WebUI 配置构建器生成 JSON；它会在应用前规范化优化器选择与结构。

### 调试信息

排查 DeepSpeed 问题时，请检查：
- WebUI Hardware 标签（Hardware → Accelerate）或 `nvidia-smi` 的硬件兼容性
- 训练日志中的 DeepSpeed 配置
- 卸载路径权限与可用空间
- 平台检测日志

# EMA 训练（指数滑动平均）

EMA 能平滑梯度并提升权重的泛化能力，但它非常占内存。

EMA 会在内存中保留模型参数的影子副本，实质上将模型占用翻倍。SimpleTuner 中 EMA 不通过 Accelerator 模块，因此不受 DeepSpeed 影响。这意味着基础 U-Net 的节省不会体现在 EMA 上。

不过默认情况下，EMA 模型保存在 CPU 上。
