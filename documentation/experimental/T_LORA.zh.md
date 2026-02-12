# T-LoRA（时间步依赖的 LoRA）

## 背景

标准 LoRA 微调在所有扩散时间步上均匀应用固定的低秩自适应。当训练数据有限时（尤其是单图像定制），这会导致过拟合——模型会在高噪声时间步上记忆噪声模式，而这些时间步几乎不包含语义信息。

**T-LoRA**（[Soboleva 等人，2025](https://arxiv.org/abs/2507.05964)）通过根据当前扩散时间步动态调整活跃 LoRA 秩的数量来解决这个问题：

- **高噪声**（去噪早期，$t \to T$）：较少的秩被激活，防止模型记忆无信息量的噪声模式。
- **低噪声**（去噪后期，$t \to 0$）：更多的秩被激活，使模型能够捕捉精细的概念细节。

SimpleTuner 的 T-LoRA 支持基于 [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) 库构建，需要包含 `lycoris.modules.tlora` 模块的 LyCORIS 版本。

> 实验性：T-LoRA 用于视频模型时可能产生较差的结果，因为时间压缩会在时间步边界之间混合帧。

## 快速设置

### 1. 设置训练配置

在 `config.json` 中，使用 LyCORIS 并指定单独的 T-LoRA 配置文件：

```json
{
    "model_type": "lora",
    "lora_type": "lycoris",
    "lycoris_config": "config/lycoris_tlora.json",
    "validation_lycoris_strength": 1.0
}
```

### 2. 创建 LyCORIS T-LoRA 配置

创建 `config/lycoris_tlora.json`：

```json
{
    "algo": "tlora",
    "multiplier": 1.0,
    "linear_dim": 64,
    "linear_alpha": 32,
    "apply_preset": {
        "target_module": ["Attention", "FeedForward"]
    }
}
```

这就是开始训练所需的全部配置。以下章节涵盖可选调优和推理。

## 配置参考

### 必填字段

| 字段 | 类型 | 描述 |
|------|------|------|
| `algo` | string | 必须为 `"tlora"` |
| `multiplier` | float | LoRA 强度乘数。除非你清楚自己在做什么，否则保持 `1.0` |
| `linear_dim` | int | LoRA 秩。在掩码调度中将成为 `max_rank` |
| `linear_alpha` | int | LoRA 缩放因子（与 `tlora_alpha` 不同） |

### 可选字段

| 字段 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `tlora_min_rank` | int | `1` | 在最高噪声级别时的最小活跃秩数 |
| `tlora_alpha` | float | `1.0` | 掩码调度指数。`1.0` 为线性；大于 `1.0` 的值会将更多容量分配给精细细节步骤 |
| `apply_preset` | object | — | 通过 `target_module` 和 `module_algo_map` 进行模块定向 |

### 特定模型的模块目标

对于大多数模型，通用的 `["Attention", "FeedForward"]` 目标即可工作。对于 Flux 2（Klein），请使用自定义类名：

```json
{
    "algo": "tlora",
    "multiplier": 1.0,
    "linear_dim": 64,
    "linear_alpha": 32,
    "apply_preset": {
        "target_module": [
            "Flux2Attention", "Flux2FeedForward", "Flux2ParallelSelfAttention"
        ]
    }
}
```

有关每个模型的完整模块目标列表，请参阅 [LyCORIS 文档](../LYCORIS.md)。

## 调优参数

### `linear_dim`（秩）

更高的秩 = 更多参数和表达能力，但在数据有限时更容易过拟合。原始 T-LoRA 论文在 SDXL 单图像定制中使用秩 64。

### `tlora_min_rank`

控制在噪声最大的时间步上秩激活的下限。增加此值可以让模型学习更粗糙的结构，但会降低防止过拟合的效果。从默认值 `1` 开始，仅在收敛太慢时才提高。

### `tlora_alpha`（调度指数）

控制掩码调度的曲线形状：

- `1.0` — 在 `min_rank` 和 `max_rank` 之间线性插值
- `> 1.0` — 在高噪声时更激进地掩码；大多数秩仅在去噪接近结束时才激活
- `< 1.0` — 更温和的掩码；秩更早激活

<details>
<summary>调度可视化（秩与时间步的关系）</summary>

当 `linear_dim=64`、`tlora_min_rank=1` 时，对于 1000 步调度器：

```
alpha=1.0（线性）：
  t=0   （干净）  → 64 个活跃秩
  t=250 （25%）   → 48 个活跃秩
  t=500 （50%）   → 32 个活跃秩
  t=750 （75%）   → 16 个活跃秩
  t=999 （噪声）  →  1 个活跃秩

alpha=2.0（二次——偏向细节）：
  t=0   （干净）  → 64 个活跃秩
  t=250 （25%）   → 60 个活跃秩
  t=500 （50%）   → 48 个活跃秩
  t=750 （75%）   → 20 个活跃秩
  t=999 （噪声）  →  1 个活跃秩

alpha=0.5（平方根——偏向结构）：
  t=0   （干净）  → 64 个活跃秩
  t=250 （25%）   → 55 个活跃秩
  t=500 （50%）   → 46 个活跃秩
  t=750 （75%）   → 33 个活跃秩
  t=999 （噪声）  →  1 个活跃秩
```

</details>

## 使用 SimpleTuner 管线进行推理

SimpleTuner 的内置管线具有 T-LoRA 支持。在验证期间，训练中的掩码参数会在每个去噪步骤中自动复用——无需额外配置。

对于训练之外的独立推理，你可以直接导入 SimpleTuner 的管线并设置 `_tlora_config` 属性。这确保了每步掩码与模型训练时使用的一致。

### SDXL 示例

```py
import torch
from lycoris import create_lycoris_from_weights

# 使用 SimpleTuner 的内置 SDXL 管线（内建 T-LoRA 支持）
from simpletuner.helpers.models.sdxl.pipeline import StableDiffusionXLPipeline
from diffusers import AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

model_id = "stabilityai/stable-diffusion-xl-base-1.0"
dtype = torch.bfloat16
device = "cuda"

# 加载管线组件
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype)
text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=dtype)
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
from diffusers import UNet2DConditionModel
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=dtype)

# 加载并应用 LyCORIS T-LoRA 权重
lora_path = "path/to/pytorch_lora_weights.safetensors"
wrapper, _ = create_lycoris_from_weights(1.0, lora_path, unet)
wrapper.merge_to()

unet.to(device)

pipe = StableDiffusionXLPipeline(
    scheduler=scheduler,
    vae=vae,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    unet=unet,
)

# 启用 T-LoRA 推理掩码——必须与训练配置匹配
pipe._tlora_config = {
    "max_rank": 64,      # 你的 lycoris 配置中的 linear_dim
    "min_rank": 1,       # tlora_min_rank（默认 1）
    "alpha": 1.0,        # tlora_alpha（默认 1.0）
}

with torch.inference_mode():
    image = pipe(
        prompt="a sks dog riding a surfboard",
        width=1024,
        height=1024,
        num_inference_steps=25,
        guidance_scale=5.0,
    ).images[0]

image.save("tlora_output.png")
```

### Flux 示例

```py
import torch
from lycoris import create_lycoris_from_weights

# 使用 SimpleTuner 的内置 Flux 管线（内建 T-LoRA 支持）
from simpletuner.helpers.models.flux.pipeline import FluxPipeline
from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

bfl_repo = "black-forest-labs/FLUX.1-dev"
dtype = torch.bfloat16
device = "cuda"

scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(bfl_repo, subfolder="scheduler")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=dtype)
tokenizer_2 = T5TokenizerFast.from_pretrained(bfl_repo, subfolder="tokenizer_2")
vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=dtype)
transformer = FluxTransformer2DModel.from_pretrained(bfl_repo, subfolder="transformer", torch_dtype=dtype)

# 加载并应用 LyCORIS T-LoRA 权重
lora_path = "path/to/pytorch_lora_weights.safetensors"
wrapper, _ = create_lycoris_from_weights(1.0, lora_path, transformer)
wrapper.merge_to()

transformer.to(device)

pipe = FluxPipeline(
    scheduler=scheduler,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    text_encoder_2=text_encoder_2,
    tokenizer_2=tokenizer_2,
    vae=vae,
    transformer=transformer,
)

# 启用 T-LoRA 推理掩码
pipe._tlora_config = {
    "max_rank": 64,
    "min_rank": 1,
    "alpha": 1.0,
}

with torch.inference_mode():
    image = pipe(
        prompt="a sks dog riding a surfboard",
        width=1024,
        height=1024,
        num_inference_steps=25,
        guidance_scale=3.5,
    ).images[0]

image.save("tlora_flux_output.png")
```

> **注意：** 你必须使用 SimpleTuner 的内置管线（例如 `simpletuner.helpers.models.flux.pipeline.FluxPipeline`），而不是原版 Diffusers 管线。只有内置管线包含每步 T-LoRA 掩码逻辑。

### 为什么不直接使用 `merge_to()` 并跳过掩码？

`merge_to()` 将 LoRA 权重永久烘焙到基础模型中——这是让 LoRA 参数在前向传播中生效所必需的。然而，T-LoRA 是在**时间步依赖的秩掩码**下训练的：某些秩会根据噪声级别被置零。如果在推理时不重新应用相同的掩码，所有秩会在每个时间步都被激活，导致图像过饱和或出现烧焦的外观。

在管线上设置 `_tlora_config` 会告诉去噪循环在每次模型前向传播之前应用正确的掩码，并在之后清除掩码。

<details>
<summary>掩码的内部工作原理</summary>

在每个去噪步骤中，管线会调用：

```python
from simpletuner.helpers.training.lycoris import apply_tlora_inference_mask, clear_tlora_mask

_tlora_cfg = getattr(self, "_tlora_config", None)
if _tlora_cfg:
    apply_tlora_inference_mask(
        timestep=int(t),
        max_timestep=self.scheduler.config.num_train_timesteps,
        max_rank=_tlora_cfg["max_rank"],
        min_rank=_tlora_cfg["min_rank"],
        alpha=_tlora_cfg["alpha"],
    )
try:
    noise_pred = self.unet(...)  # 或 self.transformer(...)
finally:
    if _tlora_cfg:
        clear_tlora_mask()
```

`apply_tlora_inference_mask` 使用以下公式计算形状为 `(1, max_rank)` 的二值掩码：

$$r = \left\lfloor\left(\frac{T - t}{T}\right)^\alpha \cdot (R_{\max} - R_{\min})\right\rfloor + R_{\min}$$

其中 $T$ 是调度器的最大时间步，$R_{\max}$ 是 `linear_dim`，$R_{\min}$ 是 `tlora_min_rank`。掩码的前 $r$ 个元素设为 `1.0`，其余设为 `0.0`。然后通过 LyCORIS 的 `set_timestep_mask()` 将此掩码全局设置到所有 T-LoRA 模块上。

前向传播完成后，`clear_tlora_mask()` 会移除掩码状态，以避免泄漏到后续操作中。

</details>

<details>
<summary>SimpleTuner 在验证期间如何传递配置</summary>

在训练期间，T-LoRA 配置字典（`max_rank`、`min_rank`、`alpha`）存储在 Accelerator 对象上。当验证运行时，`validation.py` 会将此配置复制到管线上：

```python
# setup_pipeline()
if getattr(self.accelerator, "_tlora_active", False):
    self.model.pipeline._tlora_config = self.accelerator._tlora_config

# clean_pipeline()
if hasattr(self.model.pipeline, "_tlora_config"):
    del self.model.pipeline._tlora_config
```

这是完全自动的——验证图像使用正确的掩码无需用户配置。

</details>

## 上游：T-LoRA 论文

<details>
<summary>论文详情与算法</summary>

**T-LoRA: Single Image Diffusion Model Customization Without Overfitting**
Vera Soboleva, Aibek Alanov, Andrey Kuznetsov, Konstantin Sobolev
[arXiv:2507.05964](https://arxiv.org/abs/2507.05964) — 被 AAAI 2026 录用

该论文引入了两项互补的创新：

### 1. 时间步依赖的秩掩码

核心洞察是：较高的扩散时间步（噪声更大的输入）比较低的时间步更容易过拟合。在高噪声下，潜变量主要包含随机噪声，几乎没有语义信号——在此基础上训练全秩适配器会让模型记忆噪声模式，而非学习目标概念。

T-LoRA 通过动态掩码调度来解决这个问题，根据当前时间步限制活跃的 LoRA 秩。

### 2. 正交权重参数化（可选）

论文还提出通过对原始模型权重进行 SVD 分解来初始化 LoRA 权重，并通过正则化损失强制正交性。这确保了适配器各组件之间的独立性。

SimpleTuner 的 LyCORIS 集成主要关注时间步掩码组件，它是减少过拟合的主要驱动力。正交初始化是独立 T-LoRA 实现的一部分，但目前未被 LyCORIS 的 `tlora` 算法使用。

### 引用

```bibtex
@misc{soboleva2025tlorasingleimagediffusion,
      title={T-LoRA: Single Image Diffusion Model Customization Without Overfitting},
      author={Vera Soboleva and Aibek Alanov and Andrey Kuznetsov and Konstantin Sobolev},
      year={2025},
      eprint={2507.05964},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.05964},
}
```

</details>

## 常见问题

- **推理时忘记设置 `_tlora_config`：** 图像看起来过饱和或烧焦。所有秩在每个时间步都被激活，而不是遵循训练时的掩码调度。
- **使用原版 Diffusers 管线：** 原版管线不包含 T-LoRA 掩码逻辑。你必须使用 SimpleTuner 的内置管线。
- **`linear_dim` 不匹配：** `_tlora_config` 中的 `max_rank` 必须与训练时使用的 `linear_dim` 匹配，否则掩码维度将不正确。
- **视频模型：** 时间压缩会在时间步边界之间混合帧，这会削弱时间步依赖的掩码信号。结果可能较差。
- **SDXL + FeedForward 模块：** 在 SDXL 上使用 LyCORIS 训练 FeedForward 模块可能导致 NaN 损失——这是 LyCORIS 的通用问题，并非 T-LoRA 特有。详见 [LyCORIS 文档](../LYCORIS.md#potential-problems)。
