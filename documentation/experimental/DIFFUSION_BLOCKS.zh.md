# DiffusionBlocks

DiffusionBlocks 将兼容的扩散 Transformer 转换为可独立训练的层组。每个组负责一个噪声区间；一次训练前向只执行当前批次对应的组。

这是基于 [DiffusionBlocks](https://arxiv.org/abs/2506.14202) 的实验性架构转换，不是普通的冻结层。推理时必须使用与训练相同的分块路由。

## 配置

```json
{
  "diffusion_blocks_config": {
    "layers_per_block": 4,
    "overlap": 0.05
  },
  "find_unused_parameters": true
}
```

DDP 会自动启用 `find_unused_parameters`；显式设为 `false` 会报错。

| 键 | 默认值 | 含义 |
| --- | --- | --- |
| `layers_per_block` | 必填 | 每个噪声块包含的最大连续 Transformer 层数。 |
| `overlap` | `0.05` | 相邻训练噪声区间的扩展比例，范围 `0.0` 到 `0.5`。 |
| `blocks_to_train` | `"all"` | 当前任务负责的块索引；其他组在创建 adapter 后冻结。 |
| `block_paths` | 自动 | 自动发现不足时使用的显式 `ModuleList` 路径。 |
| `timestep_boundaries` | 自动 | 从 `0.0` 到 `1.0` 的升序边界，数量必须为 `num_blocks + 1`。 |

自动边界按等概率切分已配置的 timestep 分布。块 `0` 负责最高噪声和最前面的层；最后一个块负责最低噪声和最后的层。

## 支持范围

共享实现支持具有同构 Transformer block 列表的 diffusion 和 flow-matching 模型，包括单阶段、joint/single stream、double/single stream、`blocks` 和 `layers`。

以下配置会在启动时拒绝：UNet、ControlNet、Musubi block swap、TwinFlow、多 timestep scheduled sampling、使用固定层捕获的 CREPA，以及 LayerSync。TREAD 路由保留模型的全局层索引，并裁剪到活动组的全局范围。

路由会改变 denoiser 架构。初始 loss 和输出质量不应被认为与普通全深度训练一致。启用此选项不会把已有的普通 LoRA 变成训练完成的 DiffusionBlocks adapter。

仅在确认每条路径都是顺序 denoiser 阶段后才设置 `block_paths`。不要选择文本 adapter、VAE block 或带 skip connection 的 UNet 阶段。
不会发现依赖 skip 的 encoder-decoder Transformer 堆栈，例如 i1 的 `in_blocks`/`out_blocks`，因为输出组无法在没有配对输入组 activation 的情况下运行。

## 内存

只有活动层组创建 Transformer activation。这是单任务训练全部块时的主要显存节省。全部块最终仍会分配各自的 optimizer state。

要使用独立块任务，请为每个任务设置不同的 `blocks_to_train`。未归属组会被冻结，不分配 optimizer state。推理前必须按参数归属合并这些 checkpoint。

模型权重仍常驻，除非启用兼容的 offload。Group offload 兼容；Musubi block swap 不兼容。

## 推理

SimpleTuner validation 会自动使用控制器。标准 Diffusers pipeline 无法从 LoRA 权重推断该转换，必须先应用控制器：

```python
from simpletuner.helpers.training.diffusion_blocks import DiffusionBlocksConfig, DiffusionBlocksController

config = DiffusionBlocksConfig.from_dict({"layers_per_block": 4, "overlap": 0.05})
controller = DiffusionBlocksController(pipe.transformer, config)
```

在 pipeline 生命周期内保留 `controller`，并使用 `simpletuner_config.json` 中的原始配置。

## Anima 示例

示例位于 `simpletuner/examples/anima.peft-lora+diffusion-blocks/config.json`。Anima v1.0 有 28 层；`layers_per_block=4` 会创建 7 个块。

```bash
simpletuner train env=examples/anima.peft-lora+diffusion-blocks max_train_steps=10 validation_steps=10
```

恢复训练时不得改变 block 路径、层数、边界、`blocks_to_train`、拓扑、world size、batch sampling 或 timestep 配置。推理时执行全部层会改变架构并使训练目标失效。
