# Prompt2Effect

Prompt2Effect 是一个实验性的纯 CLI 工作流，用于训练一个从效果提示词生成 PEFT LoRA 权重的超网络。它与 SimpleTuner 常规的图像/视频去噪训练器是分开的。

关键区别是：Prompt2Effect 并不会让超网络训练本身只需要 3.3 秒。它把昂贵的工作转移到一次性的训练阶段，这个阶段需要一批已有的效果 LoRA。等超网络训练好之后，从新提示词生成 LoRA 才是一次 forward pass。

## 训练内容

训练样本是已有的 LoRA checkpoint，而不是媒体文件：

- 效果提示词
- 该效果对应的 PEFT LoRA checkpoint
- 固定的基础模型和固定的目标层 schema

prepare 步骤会把每个 LoRA update 转换成 SVD 正则化的 canonical factors。训练损失是这些 canonical LoRA factors 上的 normalized MSE，不是 latent 上的 diffusion loss。

## 支持的模型族

当前脚本支持：

- `ltxvideo2`
- `wan` I2V 变体
- `hunyuanvideo`

生成的产物是普通的 `pytorch_lora_weights.safetensors` 文件，包含 PEFT 的 `lora_A`、`lora_B` 和 `alpha` key。

## 文件

Prompt2Effect 位于 `scripts/prompt2effect/`：

- `prepare.py`：验证 LoRA manifest，并写出 SVD canonical targets。
- `train.py`：训练 Prompt2Effect 超网络。
- `generate.py`：用训练好的超网络和效果提示词输出 PEFT LoRA。

该功能不会出现在 WebUI 中。

## Manifest

创建一个 JSONL 文件，每行一个效果 LoRA：

```json
{"id":"blue_mood","effect_prompt":"blue mood cinematic atmosphere","lora_path":"/path/to/pytorch_lora_weights.safetensors"}
```

同一次 Prompt2Effect 运行中的所有 LoRA 必须使用相同的目标模块 schema，并且输入/输出维度必须一致。在 prepare 阶段使用 `--rank` 选择 canonical/generated LoRA rank；如果省略，则使用第一个 LoRA 的 rank。

## 准备 Targets

```bash
.venv/bin/python scripts/prompt2effect/prepare.py \
  --manifest /path/to/effects.jsonl \
  --output_dir cache/prompt2effect/wan-i2v-targets \
  --model_family wan \
  --model_flavour i2v-14b-2.1
```

常用选项：

- `--model_family`：`ltxvideo2`、`wan` 或 `hunyuanvideo`。
- `--base_model`：覆盖基础模型 repo 或本地路径。
- `--model_flavour`：未提供 `--base_model` 时使用已知的模型族默认值。
- `--target_modules`：逗号分隔的 PEFT 目标 suffix、`default` 或 `all-linear`。
- `--rank`：生成 LoRA 的 rank。默认使用第一个源 LoRA 的 rank。
- `--component_subfolder`：基础模型组件子目录。默认使用该模型族的 transformer 子目录。

`prepare.py` 会写出：

- `schema.json`
- `targets.safetensors`

如果某个 LoRA 缺少必需模块、包含意外模块，或与基础模型 tensor shape 不匹配，脚本会失败。

## 训练

```bash
.venv/bin/python scripts/prompt2effect/train.py \
  --prepared_dir cache/prompt2effect/wan-i2v-targets \
  --output_dir output/prompt2effect/wan-i2v \
  --text_encoder_model google/t5-v1_1-base \
  --max_train_steps 10000
```

文本编码器是冻结的，只用于编码效果提示词。基础模型权重也被冻结，并作为超网络的结构条件输入。

默认情况下，基础权重保留在 CPU。只有当选中的目标层能够放入 accelerator 时，才使用 `--base_weights_device training`。

## 生成 LoRA

```bash
.venv/bin/python scripts/prompt2effect/generate.py \
  --checkpoint output/prompt2effect/wan-i2v/prompt2effect_hypernetwork.pt \
  --prompt "blue mood cinematic atmosphere" \
  --output output/blue_mood_prompt2effect
```

输出目录会包含 `pytorch_lora_weights.safetensors`。像加载其他 SimpleTuner/Diffusers PEFT LoRA 一样加载它即可。

## 限制

- 仅支持 PEFT linear LoRA。LyCORIS、convolution LoRA、DoRA magnitude vectors 和任意 sidecar tensors 目前不在该工作流中支持。
- 一个超网络绑定到一个模型族、基础模型 shape、目标模块 schema 和 rank。
- 这些脚本没有集成 Accelerate、WebUI 或 SimpleTuner 主 checkpoint manager。
- 训练质量取决于源效果 LoRA 的数量和多样性。少量 LoRA 足以测试流程，但不足以期待泛化能力。
- 发布或在生产工作流中使用前，应像普通 LoRA 一样验证生成的 LoRA。
