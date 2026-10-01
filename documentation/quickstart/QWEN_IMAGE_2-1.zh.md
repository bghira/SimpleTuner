# Qwen Image 2.1 快速入门

[Qwen Image 2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) 使用 7B、32 层图像 Transformer、Qwen3-VL 文本编码器，以及空间压缩率为 16 倍的 64 通道 VAE。SimpleTuner 默认选择 `model_family: "qwen_image"` 和 `model_flavour: "v2.1"`。本指南介绍文生图 LoRA 训练。

20B 的 `v1.0` / `v2.0` 请参阅[旧版 Qwen Image 指南](QWEN_IMAGE.zh.md)。旧版 `edit-*` 的配对参考图训练见 [Qwen Edit](QWEN_EDIT.zh.md)。它们的适配器、文本嵌入和潜变量缓存不能与 2.1 混用。

[Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE)

## 安装

使用 Python 3.12–3.14，并按照[安装指南](../INSTALL.zh.md)配置平台。以下预设在 NVIDIA GPU 上测试。SimpleTuner 已包含 `webshart` 依赖；正则化数据集需要网络访问和本地缓存空间。

```bash
pip install 'simpletuner[cuda]'
```

<a id="vram-presets"></a>

## 选择显存预设

标准示例结合[辅助适配器 v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3)、合成数据正则化、REPA 和自动流调度偏移。默认的 `qwen_image.peft-lora` 与 48 GB 预设一样使用 512px + 1024px。24/32 GB 预设仅使用 512px，验证也相同。

| 显存预算 | 示例 | 基础分辨率 | 更新次数 | 梯度检查点间隔 |
| --- | --- | --- | --- | --- |
| 24 GB | `qwen_image-2.1-24g.peft-lora` | 512px | 2000 | 1 |
| 32 GB | `qwen_image-2.1-32g.peft-lora` | 512px | 2000 | 2 |
| 48 GB | `qwen_image-2.1-48g.peft-lora` | 512px + 1024px | 4000 | 2 |
| 80 GB | `qwen_image-2.1-80g.peft-lora` | 512px + 1024px | 4000 | 2 |
| 144 GB | `qwen_image-2.1-144g.peft-lora` | 512px + 1024px | 4000 | 2 |

```bash
simpletuner train example=qwen_image-2.1-48g.peft-lora
```

所有预设使用 BF16、rank/alpha 32、batch 1、学习率 `1e-4` 的 AdamW BF16、25 步预热和 1.0 范数裁剪。REPA 使用 `dinov2_vitg14`、第 8 层、权重 0.5、编码尺寸 518、空间对齐，时间距离为 0。启用自动偏移，静态偏移为 0。辅助适配器在训练中冻结、验证时禁用。正则化目标来自同时禁用两个适配器的原始基础模型预测。

标准概率采样器将一半权重分配给触发词为 `🟫` 的 `RareConcepts/Domokun`，另一半分配给设置了 `is_regularisation_data: true` 的 `webshart/qwen-image-2.1-generated-images`。多尺度预设将每一半均分给 512px 和 1024px。面积分桶保持图像比例，分别使用 0.262144 和 1.048576 百万像素。合成数据的方形、竖图和横图后端均分对应分辨率的正则化权重。这是概率采样，不是严格交替。每 250 步验证并保存一次。

内置数据加载器将每个合成数据宽高比子集限制为 1,024 张，各分辨率和数据源使用独立潜变量缓存。关闭 VAE 分块。这些更新预算只是起点；延长训练前先检查检查点图像。

<details markdown="1">
<summary>显存测量结果</summary>

L40S 显存检查使用每个后端 4 张图像，运行 16 次更新并完成验证与保存。512px、检查点间隔 1 的配置在 24 GiB 分配限制下通过，PyTorch 峰值分配/保留显存为 20.06 / 21.10 GiB。多尺度、间隔 2 的配置在 L40S 上通过，峰值为 32.49 / 41.21 GiB，但在 32 GiB 限制下显存不足。这是小子集显存检查，不是吞吐量或收敛测量；较小显存预算是在 L40S 上模拟的，并非使用独立显卡测试。 最终的 32 GB 预设使用 512px、间隔 2，也通过了检查：峰值分配/保留显存为 22.12 / 23.68 GiB。

</details>

## 使用自己的数据

将所选示例的 `config.json` 和 `dataloader.json` 复制到训练环境，并提供自己的提示词库。更新 `data_backend_config`、`user_prompt_library` 和 `output_dir` 以对应这些文件。配置布局见[训练教程](../TUTORIAL.zh.md)和[数据加载器参考](../DATALOADER.zh.md)。

将 Domokun 后端替换为概念或照片数据集。多分辨率训练需要独立的 512px 和 1024px 后端，使用不同 ID 和 VAE 缓存路径。`resolution_type: "area"` 的单位是百万像素，分别使用 **0.262144** 和 **1.048576**；`crop: false` 保持宽高比。训练配置中保留 `aspect_bucket_alignment: 32`。

组合配方保留 `is_regularisation_data: true` 的合成后端，并将训练后端设为 false。训练数据与正则化各占一半采样权重，再在不同分辨率之间平分。使用概率采样器，不是严格交替。repeats 决定数据可用时长，不决定采样概率。修改数据集、分辨率、批处理或分布式拓扑后，应开始新训练。

## 选择训练辅助适配器

main 和内置示例目前使用 **v3**。训练时将其冻结并使用强度 1，验证时禁用。正则化批次的父模型目标是**裸基础模型预测**，关闭训练 LoRA 和辅助适配器。

[辅助适配器 v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3) 是默认的 rank-64 适配器，在 512/1024/1536/2048 分辨率上训练了 30k 更新。预设显存测试使用 v2；使用 v3 时请检查显存余量。要复现旧训练，可使用以下配置显式选择[辅助适配器 v2](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v2)：

```json
{
  "assistant_lora_path": "SimpleTuner/Qwen-Image-2.1-training-assistant-v2",
  "assistant_lora_strength": 1.0,
  "assistant_lora_inference_strength": 0.0
}
```

不使用辅助适配器时，设置 `disable_assistant_lora: true`。创建新辅助适配器时也应如此。下游推理只加载训练好的 LoRA；辅助适配器和 REPA 投影器仅用于训练。

## 验证学习效果和质量

示例每 250 更新验证和保存一次，使用 40 推理步、true CFG 1 和种子 42。24/32 GB 预设仅验证 512px，较大预设验证 512px 和 1024px。自定义配置应保留以下相关设置：

```json
{
  "validation_guidance": 1.0,
  "validation_guidance_real": 1.0,
  "validation_num_inference_steps": 40,
  "validation_seed": 42,
  "validation_step_interval": 250,
  "checkpoint_step_interval": 250
}
```

概念提示词应包含触发词，同时增加无关主体来检查连贯性和图像质量。比较检查点时保持提示词、种子、解码器、分辨率和引导设置一致。训练与输出分辨率不同：也应以 1024px 测试 512px 训练的适配器。这些配方不要求 2048px 的概念数据。

[实验汇总](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-LoRA-experiments)包括辅助适配器、正则化、REPA、自动偏移和长期训练的比较。有些训练会先退化再恢复，不应仅凭一个较差检查点判断。[照片美学 v3 比较](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-LoRA-photo-aesthetics-v3)中，512px 的 50k 更新模型在 1MP 输出时也比 1024px 的 10k 模型呈现更多场景细节；两者训练更新预算不同。

## VAE 与推理

Qwen Image 2.1 默认使用 [Ollin 修复纹理的 VAE](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1) 进行验证及其他 VAE 解码。该模型仅微调了解码器，编码器未变，因此现有的 2.1 训练潜变量仍然兼容。VAE 的版本独立于基础模型固定。要使用原始 VAE 复现输出，请设置 `pretrained_vae_model_name_or_path: "Qwen/Qwen-Image-2.1"`。显式指定的 VAE 和旧版本模型保持原有行为。

标准 PEFT 适配器可使用 SimpleTuner 内置管线，无需通过 Git 安装 Diffusers。以下示例读取 LoRA 导出并排除仅用于训练的 REPA 投影器张量，使用整帧解码且不加载辅助适配器：

模型 CPU 卸载可避免文本编码器、Transformer 和 VAE 同时占用显存。显存充足时，将 `pipe.enable_model_cpu_offload()` 替换为 `pipe.to("cuda")` 可加快重复推理。

```python
import torch
from safetensors.torch import load_file
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from simpletuner.helpers.models.qwen_image.autoencoder_21 import AutoencoderKLQwenImage21
from simpletuner.helpers.models.qwen_image.pipeline_21 import QwenImage21Pipeline
from simpletuner.helpers.models.qwen_image.transformer_21 import QwenImage21Transformer2DModel

vae = AutoencoderKLQwenImage21.from_pretrained(
    "madebyollin/texture-fix-vae-for-qwen-image-2.1",
    revision="e9f84623d22c47f8bc9fb799bc54201fa53cf80b",
    torch_dtype=torch.bfloat16,
)
model_id = "Qwen/Qwen-Image-2.1"
pipe = QwenImage21Pipeline(
    transformer=QwenImage21Transformer2DModel.from_pretrained(
        model_id, subfolder="transformer", torch_dtype=torch.bfloat16,
    ),
    vae=vae,
    text_encoder=Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, subfolder="text_encoder", dtype=torch.bfloat16,
    ),
    processor=Qwen3VLProcessor.from_pretrained(model_id, subfolder="processor"),
    scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler"),
)
pipe.enable_model_cpu_offload()
pipe.vae.disable_tiling()
pipe.vae.enable_slicing()
weights = load_file("output/my-qwen21/pytorch_lora_weights.safetensors")
weights = {
    key: value for key, value in weights.items()
    if ".lora_A." in key or ".lora_B." in key
}
pipe.load_lora_weights(weights, adapter_name="trained")
pipe.set_adapters("trained", adapter_weights=1.0)
image = pipe(
    prompt="🟫 wearing a red scarf, standing in a snowy forest",
    width=1024, height=1024, num_inference_steps=40, true_cfg_scale=1.0,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]
image.save("output.png")
```

## 显存与图像伪影

先使用 BF16（`base_model_precision: "no_change"`）。量化前先减少批量或选择较小预设；量化节省显存，但不保证更快。示例启用区域编译。长描述、额外分辨率、验证和 REPA 都会改变峰值显存；增加批量前应测量余量。

显存允许时保持 `vae_enable_tiling: false`。纹理修复解码器处理画布状纹理；分块色缝是另一个空间上下文问题。范数裁剪 1.0 对应 `grad_clip_method: "norm"` 和 `max_grad_norm: 1.0`，不同于旧 AnyFlow 试验的逐元素 0.01 裁剪。

<details markdown="1">
<summary>显存测量结果: VAE</summary>

Qwen Image 2.1 解码单张图像时不再保留未使用的时序特征缓存。在 H200 上以 BF16 单独解码一张 2048×2048 图像时，峰值已分配显存从 26.87 GiB 降至 15.28 GiB，输出完全一致。分块解码可进一步节省显存，但限制空间上下文可能产生彩色接缝；移除未使用的缓存并不能修复这些接缝。

</details>

## 历史与实验配方

以下配方用于创建辅助适配器或测试蒸馏，与上方推荐的下游训练配方分开。旧试验设置和结果保留供参考。

<details markdown="1">
<summary>AnyFlow 试验与优化器历史</summary>

<a id="experimental-anyflow-pilot"></a>

### 实验性 AnyFlow 试验

三个 `qwen_image-2.1-anyflow-stage*.peft-lora` 示例测试区间蒸馏能否在引入 `🟫` 的同时保留基础模型的行为。这些配置仍属实验性质；Qwen 模型卡并未证实之前的退化由引导蒸馏导致。

所有阶段使用 1024px、BF16、AdamW、rank 32 和 batch 4。阶段 1 在 H200 上关闭梯度检查点；阶段 2 和 3 对连续两个块进行梯度检查点计算。这是与上述辅助 REPA 预设不同的蒸馏训练。

这些 AnyFlow 示例明确使用 `grad_clip_method: "value"` 和 `max_grad_norm: 0.01`，将每个梯度元素限制在 ±0.01 范围内，而不是限制全局梯度范数。若要测试阈值为 1.0 的范数裁剪，须同时设置 `grad_clip_method: "norm"` 和 `max_grad_norm: 1.0`。

1. **阶段 1：** 以 `1e-5` 进行 10,000 次 forward AnyFlow 更新。CC12M 使用 `webshart/cc12m-structured-captions` 和 `caption_key: "long_caption"`；e621 使用 `webshart/e621-2024-webp-4Mpixel-webshart-indices`。两个先验数据集各限制为 4,096 张合格图片，采样权重各为 0.49；`RareConcepts/Domokun` 使用触发词 `🟫`、权重 0.02 和 `repeats: 0`。
2. **阶段 2：** 以 `2e-6` 进行 2,000 次 on-policy DMD 更新，同时在相同数据混合上训练 forward 目标。这是 AnyFlow DMD，并非偏好对 DPO。两个阶段均包含角色样本；仅靠冻结教师无法教授新角色。
3. **阶段 3：** 以 `5e-7` 更新 100 次。Domokun 权重为 0.5，两个正则化数据集权重各为 0.25、各限制为 64 张图片。所有区间使用 `r=t` 和原始 flow 目标，保留区间嵌入器并进行简短监督微调。正则化批次使用禁用 adapter 的基础模型预测。

先检查第 2,000、5,000 和 10,000 次更新的检查点，再决定是否将阶段 1 延长至 20,000 次。阶段 2 的预算为 2,000 次更新；若相同条件下的验证图片退化，应提前停止。更新次数本身并不能证明模型已完成训练。

配置启用了动态编译以处理可变描述长度。`schedule_shift: 2.000802574061872` 对应已发布调度器在 1024px（4,096 个潜变量 token）下的设置；更改分辨率时需重新计算。

```bash
simpletuner train example=qwen_image-2.1-anyflow-stage1.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage2.peft-lora
simpletuner train example=qwen_image-2.1-anyflow-stage3.peft-lora
```

阶段 2、3 通过 `init_lora` 加载上一阶段的最终 adapter，重新初始化优化器和数据加载器状态。采样权重控制尚未耗尽的数据集之间的交替，并不保证最终样本比例。此受限试验不覆盖完整先验语料库。字符串字段 `long_caption` 可使用现有 Webshart caption selector，无需原生 JSON 对象 caption 支持。

阶段 1、2 使用 `diffusion_target: "base_prediction"` 和 `fuse_guidance_scale: 1.0`：diffusion 分支保留冻结的条件预测场，其余分支从数据学习。这是保留能力的假设，并不保证学会角色。请在 4、16、40 个推理步下比较所附角色和先验提示词，并以基础模型的 40 步结果为参考；自动验证使用 4 步。目标函数和检查点要求见 [AnyFlow](../experimental/ANYFLOW.zh.md)。

已完成的试验：阶段 1 达到 10,000 次更新，阶段 2 达到 2,000 次。重新加载适配器后的 40 步狐狸图像和角色提示词图像仍然连贯，但角色提示词生成的是人而非 Domokun。4 步图像仍然模糊或充满噪声。另一次 L40S 对比对两个检查点均使用 1024px、种子 42、CFG 1/2/4/6 和空负面提示词。前向调用断言确认 CFG 大于 1 时每步有两次前向。已检查的狐狸和海滩样本未得到修复：更高 CFG 增加了饱和度和伪影。这些结果尚不能确定原因；阶段 3 尚未验证。

从同一检查点继续阶段 1 的两个分支各增加了 1,000 次更新，比较逐元素裁剪阈值 0.01 和 1.0。重新加载后的 4 步狐狸图像在两组中仍有明显噪声；40 步海滩图像仍然呈现人物。阈值 0.01 在 65/1,000 次更新中触发裁剪，阈值 1.0 为 0/1,000 次。两组均使用未修正的优化器，且在裁剪首次触发前梯度已存在差异，因此不能把细微变化完全归因于阈值。

<a id="qwen21-optimizer-correction"></a>

此处的阶段 1 和辅助 LoRA 试验均在修复 AdamW BF16 的随机舍入加法辅助函数之前运行：该函数计算的是 `other + alpha * input`，而非 `input + alpha * other`。当 β₁ = 0.9 时，一阶矩因此按 `m = 0.09 * m + g` 更新，而非 `m = 0.9 * m + 0.1 * g`。精确算术回归测试已覆盖 CPU、MPS 和 CUDA。图像观察结果对这些检查点仍然有效，但不能据此独立判断架构或蒸馏目标是否有效；必须使用修正后的优化器重新验证训练。

使用修正后的优化器、相同起始检查点和逐元素裁剪阈值 1.0 再继续更新 1,000 次，仍未修复已检查的 4 步狐狸和海滩图像。40 步狐狸图像保持连贯，而海滩提示词仍生成人物。此试验检验的是现有检查点的恢复能力，而不是使用修正后的优化器从头训练；整体失败原因仍未确定。

</details>

<details markdown="1">
<summary>创建辅助适配器：在线、多分辨率和离线配方</summary>

<a id="assistant-lora"></a>

### 使用字幕训练辅助 LoRA

`qwen_image-2.1-assistant-lora.peft-lora` 是面向 L40S 的实验起点：BF16、批量 1、间隔 2 的梯度检查点、rank 32，以及学习率 `1e-4` 的 AdamW BF16。预算为 1,000 次更新，每 50 次验证和保存。正式训练前，请用多样化字幕替换十二条冒烟测试字幕；此示例尚未验证收敛效果。

`grad_clip_method: "norm"`, `max_grad_norm: 1.0`.

设置 `distillation_method: assistant_lora`。字幕后端预计算文本嵌入；每批禁用适配器，以 40 个原生推理步、CFG 1 生成新的基础模型潜变量。独立生成管线复用 transformer，不加载 VAE、处理器或文本编码器。随后恢复适配器并执行普通去噪训练，不缓存终态潜变量。目前仅支持 Qwen Image 2.1 文生图。

`distillation_config.assistant_lora` 接受 `num_inference_steps`（默认 40）、`resolutions`（非空 `[宽, 高]` 列表，默认 `[[1024, 1024]]`）和 `seed`（默认 42）。尺寸必须为 32 的倍数。分辨率按批轮换，噪声种子按样本递增，包括不足一批的样本；检查点保存计数器。恢复时保持数据集、批量、梯度累积和分布式拓扑不变。不支持按需文本缓存。

流程测试可用 8 次更新、2 个教师步；评估图像前恢复 40 步。在 100、250、500 和 1,000 次更新时评估，并以相同提示词和种子比较适配器与基础模型。辅助适配器的实际价值仍需另一次概念训练验证。

Qwen Image 2.1 LoRA 训练默认加载 [训练辅助适配器 v3](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v3)，训练时冻结，验证时禁用。设置 `disable_assistant_lora: true` 可关闭；设置 `assistant_lora_path` 可替换适配器。旧版本不使用此默认值。训练新的辅助适配器时，请像相关示例一样保留 `disable_assistant_lora: true`。

请将此方法作为唯一蒸馏方法；不支持与其他蒸馏器组合。

字幕数据集要求 `dataloader_prefetch: false`，确保检查点游标对应已消费的字幕。恢复时若字幕标识或内容、批量、重复次数、打乱设置、种子、梯度累积或分布式布局发生变化，将报错。 辅助 LoRA 检查点同样拒绝更改生成种子、分辨率列表或教师推理步数。

<a id="assistant-lora-multires"></a>

#### 四种基础分辨率与宽高比分桶的辅助 LoRA 实验

`qwen_image-2.1-assistant-lora-multires.peft-lora` 循环使用基础分辨率 512、1024、1536 和 2048 对应的共 12 个宽高比桶。每种基础分辨率都有正方形、4:7 竖图和 7:4 横图三个桶；每个批次使用一个桶，在完整循环中各基础分辨率及宽高比的训练次数相同。它保留基础示例的教师 40 步、BF16 批量 1、每 2 个块进行检查点重计算、秩 32、AdamW BF16 和 1,000 次更新，并共用其描述文本后端及验证提示词。请将测试文本替换为基准实验中使用的同一组多样化描述。在新的输出目录中从全新的适配器和优化器开始；`resume_from_checkpoint: ""` 禁用恢复。保留第一次运行的权重以便比较。

此实验检验多种图像尺寸是否能改善辅助适配器，并不证明原生分辨率要求或质量收益。L40S 的 1,000 次更新已完成全部 12 个桶，并在 1024×1024 和 2048×2048 下验证。最终狐狸和肖像图像仍然连贯，但存在分块 VAE 解码已知的颜色接缝。教师直接生成潜变量目标，不经过 VAE 解码。对后续概念训练的收益仍未验证。

Assistant LoRA 与 AnyFlow 一样，仅在运行期间将 Dynamo 缓存下限设为 32 项，以容纳教师、学生和验证的不同变体。保留用户设置的更高上限，退出时恢复原值。增加分辨率或更长描述时，请监测重编译。

后续 Domokun 对照试验以 2048px、batch 1、学习率 `1e-5` 和逐元素裁剪阈值 1.0 分别训练两个全新适配器，各更新 250 次。对照组的训练辅助强度为 0，辅助组为 1；两组推理时均禁用辅助适配器。初始适配器权重及验证图像完全一致。以 1024px、40 个推理步检查最终结果时，两组的角色提示词仍生成人物，狐狸和肖像先验仍然连贯，尚未证明辅助适配器有益。两组训练及辅助适配器的准备过程均使用了[上文](#qwen21-optimizer-correction)所述的未修正优化器。

<a id="assistant-lora-offline"></a>

#### 使用可复用生成图像训练辅助 LoRA

`qwen_image-2.1-assistant-lora-offline.peft-lora` 通过 Webshart 使用 [10,000 张生成图像](https://huggingface.co/datasets/webshart/qwen-image-2.1-generated-images)训练。数据集使用 CC12M 的 `long_caption` 提示词、40 步原生教师推理、CFG 1 和完整图像 VAE 解码。12 个图像后端覆盖 512、1024、1536、2048 基础分辨率的正方形、竖幅和横幅桶，采样权重相等，不设置重复。

此配方从头训练，使用已修正的 `adamw_bf16`、学习率 `1e-4`、`grad_clip_method: "norm"`、`max_grad_norm: 1.0`、BF16、批量 1、秩 32 和每两块一组的梯度检查点。计划训练 1,000 次更新，每 50 次保存，每 100 次以 1024 分辨率验证。发布前另行生成 2048px 预览。关闭 VAE 分块，编码批量为 1。`vae_cache_ondemand: true` 在采样时编码并缓存图像，避免在 1,000 次更新前先编码全部 10,000 张图像。普通图像训练取代在线教师生成；创建辅助适配器时省略 `distillation_method`，并保留 `disable_assistant_lora: true`。

数据集可跨实验复用。PNG 需要再次经过 VAE 编码，因此目标不完全等同于教师的最终潜变量。发布前检查验证图像，并通过独立的概念训练评估辅助效果。从纯字幕配方切换时应重新开始，不要恢复原优化器或数据集状态。

[替换后的 v1 助手](https://huggingface.co/SimpleTuner/Qwen-Image-2.1-training-assistant-v1) 已在 L40S 上完成此方案的 1,000 次更新，并检查了最终的 1024 和 2048 图像。随后进行的 Domokun 对照实验采用 2048 分辨率、LR `1e-4`、全局范数裁剪 1.0，各训练 1,000 次更新；助手在训练时冻结，在验证时禁用。对照组和助手组在两个角色提示词下仍生成人物。对照组将明显的 Domokun 特征带入了无关的狐狸提示词；助手组保留了可辨认的狐狸。两组的人像均保持连贯。这仅是概念外溢减少的有限证据，不代表成功学会角色或普遍提升质量；评估仅使用一个训练种子和四个提示词。

```bash
simpletuner train example=qwen_image-2.1-assistant-lora-offline.peft-lora
```

</details>
