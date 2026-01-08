# FSDP2 分片 / 多 GPU 训练

SimpleTuner 现已一流支持 PyTorch Fully Sharded Data Parallel v2（基于 DTensor 的 FSDP）。WebUI 在全模型训练中默认使用 v2，并暴露关键 accelerate 选项，使你无需自定义启动脚本即可扩展到多 GPU。

> ⚠️ FSDP2 依赖在 CUDA 构建中启用分布式 DTensor 栈的最新 PyTorch 2.x。WebUI 仅在 CUDA 主机上显示上下文并行控制；其他后端视为实验性。

## 什么是 FSDP2？

FSDP2 是 PyTorch 分片数据并行引擎的下一代版本。不同于 FSDP v1 的旧式扁平参数逻辑，v2 插件基于 DTensor。它在各 rank 间分片模型参数、梯度与优化器，同时保持每个 rank 的工作集较小。与经典的 ZeRO 风格方案相比，它保留了 Hugging Face accelerate 的启动流程，因此检查点、优化器与推理路径能与 SimpleTuner 其余部分保持兼容。

## 功能概览

- WebUI 开关（Hardware → Accelerate）可生成带合理默认值的 FullyShardedDataParallelPlugin
- CLI 自动规范化（`--fsdp_version`、`--fsdp_state_dict_type`、`--fsdp_auto_wrap_policy`），手动拼写更宽容
- 可选上下文并行分片（`--context_parallel_size`、`--context_parallel_comm_strategy`），在 FSDP2 之上用于长序列模型
- 内置 Transformer 块检测弹窗，解析基础模型并建议自动包裹的类名
- 检测元数据缓存于 `~/.simpletuner/fsdp_block_cache.json`，WebUI 可一键维护
- 检查点格式切换（分片/完整）以及面向低主机内存的高效恢复模式

## 已知限制

- 仅当 `model_type` 为 `full` 时可启用 FSDP2。PEFT/LoRA 仍使用单设备路径。
- DeepSpeed 与 FSDP 互斥。若同时提供 `--fsdp_enable` 与 DeepSpeed 配置，CLI/WebUI 会明确报错。
- 上下文并行仅限 CUDA，且需要 `--context_parallel_size > 1` 与 `--fsdp_version=2`。
- 现在验证可与 `--fsdp_reshard_after_forward=true` 协同工作：FSDP 包裹模型会直接传入 pipelines，自动处理 all-gather/reshard。
- 块检测会在本地实例化基础模型，扫描大检查点时会有短暂停顿并占用较高主机内存。
- FSDP v1 仍保留以兼容旧配置，但在 UI 与 CLI 日志中标记为弃用。

## 启用 FSDP2

### 方法 1：WebUI（推荐）

1. 打开 SimpleTuner WebUI 并载入计划使用的训练配置。
2. 切换到 **Hardware → Accelerate**。
3. 打开 **Enable FSDP v2**。版本默认 `2`，除非需要 v1，否则保持不变。
4. （可选）调整：
   - **Reshard After Forward**：在显存与通信间权衡
   - **Checkpoint Format**：`Sharded` / `Full`
   - **CPU RAM Efficient Loading**：在主机内存受限时恢复
   - **Auto Wrap Policy** 与 **Transformer Classes to Wrap**（见下方检测流程）
   - **Context Parallel Size / Rotation**：需要序列分片时使用
5. 保存配置，启动时会传入正确的 accelerate 插件。

### 方法 2：CLI

使用与 WebUI 相同的参数运行 `simpletuner-train`。下面是两张 GPU 的 SDXL 全模型示例：

```bash
simpletuner-train \
  --model_type=full \
  --model_family=sdxl \
  --output_dir=/data/experiments/sdxl-fsdp2 \
  --fsdp_enable \
  --fsdp_version=2 \
  --fsdp_state_dict_type=SHARDED_STATE_DICT \
  --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
  --num_processes=2
```

如果你已有 accelerate 配置文件仍可继续使用；SimpleTuner 会把 FSDP 插件合并到启动参数中，而不是覆盖你的整个配置。

## 上下文并行

上下文并行是 FSDP2 之上的可选层，仅适用于 CUDA 主机。设置 `--context_parallel_size`（或 WebUI 对应字段）为需要分片序列维度的 GPU 数量。通信方式包括：

- `allgather`（默认）– 优先重叠，是最佳起点
- `alltoall` – 对超大注意力窗口的少量任务可能有益，但编排成本更高

当请求上下文并行时，训练器会强制 `fsdp_enable` 且 `fsdp_version=2`。将大小设回 `1` 会干净禁用该功能，并规范化 rotation 字符串以保持配置一致。

## FSDP 块检测流程

SimpleTuner 内置检测器，可解析所选基础模型并提示最适合 FSDP 自动包裹的模块类：

1. 在训练表单中选择 **Model Family**（必要时选择 **Model Flavour**）。
2. 若从自定义权重目录训练，输入检查点路径。
3. 点击 **Transformer Classes to Wrap** 旁的 **Detect Blocks**。SimpleTuner 会实例化模型、遍历模块，并记录每类的参数总量。
4. 查看分析弹窗：
   - **Select** 勾选需要包裹的类（首列复选框）
   - **Total Params** 显示参数占比大的模块
   - `_no_split_modules`（若存在）以徽标显示，需加入排除列表
5. 点击 **Apply Selection** 写入 `--fsdp_transformer_layer_cls_to_wrap`。
6. 除非点击 **Refresh Detection**，否则会复用缓存结果。

检测结果存储在 `~/.simpletuner/fsdp_block_cache.json`，按模型家族、检查点路径与风味为键。切换不同检查点或更新权重后，可在 **Settings → WebUI Preferences → Cache Maintenance → Clear FSDP Detection Cache** 清理。

## 检查点处理

- **分片 state dict**（`SHARDED_STATE_DICT`）保存本地分片，适合大模型扩展。
- **完整 state dict**（`FULL_STATE_DICT`）将参数聚合到 rank 0 以兼容外部工具，但内存压力更高。
- **CPU RAM Efficient Loading** 在恢复时延迟 all-rank 实体化，缓解主机内存峰值。
- **Reshard After Forward** 在前向间保持参数分片轻量。验证可直接将 FSDP 模型传入 diffusers pipelines 正常工作。

根据恢复频率与下游工具选择组合。对超大模型而言，分片检查点 + RAM 高效加载是最安全的组合。

## 维护工具

WebUI 在 **WebUI Preferences → Cache Maintenance** 中提供维护功能：

- **Clear FSDP Detection Cache** 清除全部缓存的块扫描（`FSDP_SERVICE.clear_cache()` 的封装）。
- **Clear DeepSpeed Offload Cache** 仍为 ZeRO 用户提供，且与 FSDP 相互独立。

两项操作都会显示 toast 通知并更新维护状态区域，无需翻阅日志。

## 故障排查

| 症状 | 可能原因 | 解决办法 |
|---------|--------------|-----|
| `"FSDP and DeepSpeed cannot be enabled simultaneously."` | 同时指定了两种插件（如 DeepSpeed JSON + `--fsdp_enable`）。 | 移除 DeepSpeed 配置或禁用 FSDP。 |
| `"Context parallelism requires FSDP2."` | `context_parallel_size > 1`，但 FSDP 未启用或仍是 v1。 | 启用 FSDP 并保持 `--fsdp_version=2`，或将 size 设为 `1`。 |
| `Unknown model_family` 导致检测失败 | 表单未选择支持的家族/风味。 | 从下拉菜单选择模型；自定义家族需在 `model_families` 注册。 |
| 检测显示旧类 | 复用了缓存结果。 | 点击 **Refresh Detection** 或清理缓存。 |
| 恢复时耗尽主机 RAM | 加载时聚合完整 state dict。 | 切换为 `SHARDED_STATE_DICT` 并/或启用 CPU RAM Efficient Loading。 |

## CLI 参数参考

- `--fsdp_enable` – 启用 FullyShardedDataParallelPlugin
- `--fsdp_version` – 选择 `1` 或 `2`（默认 `2`，v1 已弃用）
- `--fsdp_reshard_after_forward` – 前向后释放参数分片（默认 `true`）
- `--fsdp_state_dict_type` – `SHARDED_STATE_DICT`（默认）或 `FULL_STATE_DICT`
- `--fsdp_cpu_ram_efficient_loading` – 降低恢复时主机内存峰值
- `--fsdp_auto_wrap_policy` – `TRANSFORMER_BASED_WRAP`、`SIZE_BASED_WRAP`、`NO_WRAP`，或点路径可调用
- `--fsdp_transformer_layer_cls_to_wrap` – 由检测器填充的逗号分隔类列表
- `--context_parallel_size` – 在此数量的 rank 间分片注意力（仅 CUDA + FSDP2）
- `--context_parallel_comm_strategy` – `allgather`（默认）或 `alltoall`
- `--num_processes` – 无配置文件时传给 accelerate 的总 rank 数

这些参数与 Hardware → Accelerate 的 WebUI 控件 1:1 对应，因此从界面导出的配置可直接在 CLI 重放，无需额外修改。
